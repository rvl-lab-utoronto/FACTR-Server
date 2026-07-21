# Arm-parameterized FACTR FastAPI server.
#
# ONE server instance per leader arm: single-arm control, single endpoint, one
# port — matching the teleop node's per-arm design (factr_teleop arm_index 0/1).
# Launch it twice, once per leader:
#
#   ros2 run factr_fastapi factr_api --ros-args -p arm_index:=0   # left
#   ros2 run factr_fastapi factr_api --ros-args -p arm_index:=1   # right
#
# arm_index selects everything for that instance:
#   0 -> left  : GET :5000/get_joint_positions_left   <- ROS topic /joint_pos_left
#   1 -> right : GET :5001/get_joint_positions_right  <- ROS topic /joint_pos_right
#
# The route is registered dynamically in __init__ (NOT a class-body decorator),
# so each process serves exactly its own side; the port is 5000 + arm_index.

import asyncio
import json
import signal
import threading

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


#: DoF+1 placeholder (7 arm joints + trailing gripper) served before a publisher connects.
_DEFAULT_JOINT_POS = [0.0, 0.0, 0.0, 3.12, 0.0, 0.0, 0.0, 0.0]
_BASE_PORT = 5000   # left = 5000 + 0, right = 5000 + 1


class JointResponse(BaseModel):
    joint_pos: list[float]


class GravCompStatus(BaseModel):
    #: The master output-gain target this trigger published: 1.0 (enable) or 0.0
    #: (disable). The teleop runs the ramp itself, driving applied torque to the target
    #: over ~1s (0 = limp, 1 = full forces). These endpoints are pure triggers -- the
    #: caller does not choose a gain value.
    side: str
    gain_target: float


class FactrStatus(BaseModel):
    #: This leader's live master output gain(s). ``force_gain`` is the actual ramped
    #: multiplier applied to every commanded torque right now (0 = limp, 1 = full),
    #: reported by the teleop; ``force_gain_target`` is what this relay last commanded.
    #: ``grav_comp_enabled`` is True once the gain has effectively reached full (>= 0.99).
    side: str
    force_gain: float
    force_gain_target: float
    grav_comp_enabled: bool


class DiagnosticsStatus(BaseModel):
    available: bool
    side: str
    raw_q_rad: list[float] = []
    configured_home_q_rad: list[float] = []
    joint_offsets_rad: list[float] = []
    model_signs: list[float] = []
    model_q_rad: list[float] = []
    dfc_home_q_rad: list[float] = []
    dfc_to_factr_signs: list[float] = []
    dfc_to_factr_offset_rad: list[float] = []
    dfc_raw_offsets_deg: list[float] = []
    dfc_sign_flip_joints: list[int] = []
    dfc_wrap_deg: bool | None = None
    dfc_drop_trailing: int | None = None
    dfc_gripper_open: float | None = None
    dfc_gripper_closed: float | None = None
    enable_samples: list[dict] = []


def _side_for(arm_index: int) -> str:
    return "left" if arm_index == 0 else "right"


class FactrAPI(Node):
    """One leader arm's ROS<->HTTP bridge.

    Reads ``/joint_pos_{side}`` and serves the latest positions at
    ``GET /get_joint_positions_{side}``; the ``POST /enable_grav_comp_{side}`` and
    ``POST /disable_grav_comp_{side}`` triggers republish a master output-gain target
    (1.0 / 0.0) on ``/factr_gain_{side}`` for the teleop to ramp to; and
    ``GET /status_{side}`` reports the teleop's LIVE master gain (read back from
    ``/factr_gain_state_{side}``) so the dashboard can gate collection behind grav comp.
    Everything (side, port, topics, routes) is derived from the
    ``arm_index`` ROS parameter, so the same executable is launched once per leader arm.
    """

    def __init__(self, app: FastAPI, arm_index: int):
        super().__init__(f"factr_api_{arm_index}")
        arm_index = self.declare_parameter("arm_index", arm_index).get_parameter_value().integer_value

        self.side = _side_for(arm_index)
        self.port = _BASE_PORT + arm_index
        self.topic = f"/joint_pos_{self.side}"
        self.route = f"/get_joint_positions_{self.side}"
        self.gain_topic = f"/factr_gain_{self.side}"
        self.gain_state_topic = f"/factr_gain_state_{self.side}"
        self.enable_route = f"/enable_grav_comp_{self.side}"
        self.disable_route = f"/disable_grav_comp_{self.side}"
        self.status_route = f"/status_{self.side}"
        self.diagnostics_topic = f"/factr_diagnostics_{self.side}"
        self.diagnostics_route = f"/diagnostics_{self.side}"

        self.joint_pos: list[float] = list(_DEFAULT_JOINT_POS)
        #: Last target this relay commanded, and the live gain the teleop reports back.
        self.gain_target: float = 0.0
        self.force_gain: float = 0.0
        self.diagnostics: dict = {}
        self.lock = threading.Lock()

        self.create_subscription(JointState, self.topic, self._update_joint_pos, 10)
        # The teleop reports its LIVE master gain here; served at GET /status_<side>.
        self.create_subscription(Float64, self.gain_state_topic, self._update_gain_state, 10)
        self.create_subscription(String, self.diagnostics_topic, self._update_diagnostics, 10)
        # Setpoint out to the teleop: the master output-gain target it ramps toward.
        self.gain_pub = self.create_publisher(Float64, self.gain_topic, 10)

        # Per-side routes: read positions (GET), gain status (GET), enable/disable triggers
        # (POST). The enable/disable calls take no body -- they kick off the teleop's ramp.
        app.add_api_route(
            self.route, self.get_joint_positions, methods=["GET"], response_model=JointResponse
        )
        app.add_api_route(
            self.status_route, self.get_status, methods=["GET"], response_model=FactrStatus
        )
        app.add_api_route(
            self.diagnostics_route, self.get_diagnostics, methods=["GET"],
            response_model=DiagnosticsStatus,
        )
        app.add_api_route(
            self.enable_route, self.enable_grav_comp, methods=["POST"], response_model=GravCompStatus
        )
        app.add_api_route(
            self.disable_route, self.disable_grav_comp, methods=["POST"], response_model=GravCompStatus
        )
        self.get_logger().info(
            f"FACTR API [{self.side}]: GET :{self.port}{self.route}  <-  {self.topic}"
        )
        self.get_logger().info(
            f"FACTR API [{self.side}]: GET :{self.port}{self.status_route}  <-  {self.gain_state_topic}"
        )
        self.get_logger().info(
            f"FACTR API [{self.side}]: POST :{self.port}{self.enable_route} / "
            f"{self.disable_route}  ->  {self.gain_topic}"
        )

    async def get_joint_positions(self) -> JointResponse:
        """Return this leader's most recent joint positions (7 arm joints + gripper)."""
        with self.lock:
            return JointResponse(joint_pos=list(self.joint_pos))

    async def get_status(self) -> FactrStatus:
        """Return this leader's live master gain(s): actual, commanded target, and enabled."""
        with self.lock:
            force_gain = self.force_gain
            gain_target = self.gain_target
        return FactrStatus(
            side=self.side,
            force_gain=force_gain,
            force_gain_target=gain_target,
            grav_comp_enabled=force_gain >= 0.99,
        )

    async def get_diagnostics(self) -> DiagnosticsStatus:
        """Return FACTR's immutable startup-calibration snapshot."""
        with self.lock:
            payload = dict(self.diagnostics)
        if not payload:
            return DiagnosticsStatus(available=False, side=self.side)
        payload.pop("captured_monotonic_ns", None)
        payload.update(available=True, side=self.side)
        return DiagnosticsStatus(**payload)

    async def enable_grav_comp(self) -> GravCompStatus:
        """Trigger the teleop's ramp-up routine: applied torque fades 0->full over ~1s."""
        return self._publish_gain_target(1.0)

    async def disable_grav_comp(self) -> GravCompStatus:
        """Trigger the teleop's ramp-down routine: applied torque fades full->0 over ~1s."""
        return self._publish_gain_target(0.0)

    def _publish_gain_target(self, target: float) -> GravCompStatus:
        msg = Float64()
        msg.data = float(target)
        self.gain_pub.publish(msg)
        with self.lock:
            self.gain_target = float(target)
        self.get_logger().info(f"FACTR API [{self.side}]: grav comp gain target -> {target:.1f}")
        return GravCompStatus(side=self.side, gain_target=float(target))

    def _update_joint_pos(self, msg: JointState) -> None:
        with self.lock:
            self.joint_pos = list(msg.position)

    def _update_gain_state(self, msg: Float64) -> None:
        with self.lock:
            self.force_gain = float(msg.data)

    def _update_diagnostics(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError) as exc:
            self.get_logger().error(f"invalid diagnostics JSON: {exc}")
            return
        if isinstance(payload, dict):
            with self.lock:
                self.diagnostics = payload


def _ros_spin(node: Node) -> None:
    """Spin one node's executor on a background thread so uvicorn owns the main one.

    ``spin()`` returns once ``main``'s teardown calls ``rclpy.shutdown()``; the
    node/executor are then torn down here, on this same thread — so ``main`` MUST
    join this thread before returning, or the interpreter finalizes mid-teardown
    and the process aborts (``terminate called ...``) instead of exiting cleanly.
    """
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()


def _make_server(app: FastAPI, port: int) -> uvicorn.Server:
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    return uvicorn.Server(config)


async def _serve_all(servers) -> None:
    """Run every uvicorn server concurrently in this thread's single event loop."""
    await asyncio.gather(*(server.serve() for server in servers))


def main(args=None):
    rclpy.init(args=args)

    # One FastAPI app + ROS node per arm, each bound to its own port (left 5000,
    # right 5001), so this single process serves BOTH arms. Each arm keeps its own
    # config/topic/route; only its route is registered on its app.
    app_left, app_right = FastAPI(), FastAPI()
    node_left = FactrAPI(app_left, 0)    # GET :5000/get_joint_positions_left  <- /joint_pos_left
    node_right = FactrAPI(app_right, 1)  # GET :5001/get_joint_positions_right <- /joint_pos_right

    # Keep the thread handles: the finally must join them so ROS teardown finishes
    # before the interpreter exits (see _ros_spin).
    threads = [
        threading.Thread(target=_ros_spin, args=(node,), daemon=True)
        for node in (node_left, node_right)
    ]
    for thread in threads:
        thread.start()

    servers = [
        _make_server(app_left, node_left.port),
        _make_server(app_right, node_right.port),
    ]

    # Graceful stop on SIGINT (the supervisor's normal stop) OR SIGTERM (its
    # escalation). uvicorn installs its OWN SIGINT/SIGTERM handlers when a server
    # starts serving (Server.capture_signals, unconditional since uvicorn 0.29 —
    # the old server.install_signal_handlers hook it predates is dead); each such
    # handler flips only ITS server's should_exit, then re-raises the signal to the
    # handler installed before it. So the servers chain their stops one-to-the-next
    # and finally to the handler we install here, which is what makes shutdown work:
    #   - it flips should_exit on EVERY server, so the other one actually stops too
    #     (otherwise only the last-started server exits and asyncio.gather hangs on
    #     the first — the hang the now-dead per-server monkeypatch tried to avoid);
    #   - being a real handler (not Python's default), it also stops SIGTERM from
    #     killing the process at the end of the chain, so control returns to
    #     asyncio.run and the ROS teardown below runs for BOTH signals.
    def _request_stop(_signum, _frame):
        for server in servers:
            server.should_exit = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    try:
        asyncio.run(_serve_all(servers))
    except KeyboardInterrupt:
        pass
    finally:
        # Stop ROS, then WAIT for each spin thread to finish its executor/node
        # teardown before returning (see _ros_spin). Guard on rclpy.ok() so a stop
        # that already shut the context down doesn't double-shutdown here.
        if rclpy.ok():
            rclpy.shutdown()
        for thread in threads:
            thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
