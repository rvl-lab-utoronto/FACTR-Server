# Arm-parameterized FACTR WebSocket/control server.
#
# One process hosts one server instance per leader arm: side-specific state and
# routes on separate ports, matching the teleop nodes' per-arm design.
#
# arm_index selects everything for that instance:
#   0 -> left  : WS :5000/ws/left   <- ROS /joint_pos_left + /factr_telemetry_left
#   1 -> right : WS :5001/ws/right  <- ROS /joint_pos_right + /factr_telemetry_right
#
# The WebSocket is duplex: inbound force_feedback frames (follower external joint
# torques) and gripper_feedback frames (signed Flexiv finger force [N]) are
# republished to the teleop, with POST fallbacks for stream-less debugging.
#
# Each route is registered dynamically in __init__ (not a class-body decorator);
# the port is 5000 + arm_index.

import asyncio
import contextlib
import json
import math
import signal
import threading
from typing import Literal

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, String

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, ValidationError, field_validator


#: DoF+1 placeholder (7 arm joints + trailing gripper) served before a publisher connects.
_DEFAULT_JOINT_POS = [0.0, 0.0, 0.0, 3.12, 0.0, 0.0, 0.0, 0.0]
_BASE_PORT = 5000   # left = 5000 + 0, right = 5000 + 1
_BROADCAST_HZ = 200.0


class GravCompStatus(BaseModel):
    #: The gravity-compensation activation target this trigger published.
    side: str
    gain_target: float


class FactrStatus(BaseModel):
    #: Component gains remain independently configured, but all physical output is
    #: gated by ``leader_torque_gain`` and the Dynamixel torque-enable state.
    side: str
    leader_torque_state: Literal["disabled", "disabling", "enabling", "enabled"]
    leader_torque_enabled: bool
    leader_torque_gain: float
    leader_torque_gain_target: float
    grav_comp_gain: float
    grav_comp_gain_target: float
    force_feedback_gain: float
    force_feedback_gain_target: float
    grav_comp_enabled: bool
    force_feedback_enabled: bool


class ForceFeedback(BaseModel):
    #: Follower external joint torques for the leader's force feedback. Joint space is
    #: the only supported space today: ``tau`` holds one torque per arm joint [Nm] in
    #: the follower's joint convention. ``space`` is on the wire so a task-space
    #: variant (a 6-D TCP wrench under ``space="tcp"``) can be added later without a
    #: breaking payload change; until then anything but ``"joint"`` is rejected.
    space: Literal["joint"] = "joint"
    tau: list[float]

    @field_validator("tau")
    @classmethod
    def _finite(cls, tau: list[float]) -> list[float]:
        # Python's json module happily parses NaN/Infinity literals; a NaN torque
        # must never reach the teleop's torque command.
        if not tau or not all(math.isfinite(x) for x in tau):
            raise ValueError("tau must be a non-empty list of finite numbers")
        return tau


class ForceFeedbackAck(BaseModel):
    side: str
    space: str
    dof: int


class GripperFeedback(BaseModel):
    #: Flexiv GripperStates.force [N]: positive opening, negative closing.
    force_n: float

    @field_validator("force_n")
    @classmethod
    def _finite(cls, force_n: float) -> float:
        if not math.isfinite(force_n):
            raise ValueError("force_n must be finite")
        return force_n


class GripperFeedbackAck(BaseModel):
    side: str
    force_n: float


class ForceFeedbackToggleStatus(BaseModel):
    side: str
    force_feedback_enabled: bool
    gain_target: float


class NullSpaceControl(BaseModel):
    """One atomic collection-scoped null-space target and activation command."""

    enabled: bool
    target_dfc_rad: list[float]

    @field_validator("target_dfc_rad")
    @classmethod
    def _finite_target(cls, target: list[float]) -> list[float]:
        if not target or not all(math.isfinite(x) for x in target):
            raise ValueError(
                "target_dfc_rad must be a non-empty list of finite numbers"
            )
        return target


class NullSpaceControlStatus(BaseModel):
    side: str
    enabled: bool
    target_dfc_rad: list[float]


class TelemetryStatus(BaseModel):
    """One live FACTR dynamics/control sample."""

    side: str
    stamp_monotonic_ns: int
    raw_q_rad: list[float]
    model_q_rad: list[float]
    model_dq_rad_s: list[float]
    home_error_rad: list[float]
    joint_offsets_rad: list[float]
    model_signs: list[float]
    null_space_target_rad: list[float]
    limit_torque_nm: list[float]
    null_torque_nm: list[float]
    gravity_torque_nm: list[float]
    friction_torque_nm: list[float]
    force_feedback_torque_nm: list[float]
    calculated_torque_nm: list[float]
    commanded_torque_nm: list[float]
    present_current_raw: list[float]
    current_estimated_torque_nm: list[float]
    applied_torque_nm: list[float]
    null_space_gain: float
    grav_comp_gain: float
    grav_comp_gain_target: float
    friction_gain: list[float]
    force_feedback_gain: float
    force_feedback_gain_target: float
    leader_torque_gain: float
    leader_torque_gain_target: float
    leader_torque_enabled: bool
    # Driver diagnostics are already JSON-normalized by the teleop. Keeping the
    # nested structure intact is important: the Rerun sidecar deduplicates its
    # retained event ring by driver session/sequence.
    dynamixel: list[dict] = Field(default_factory=list)


def _side_for(arm_index: int) -> str:
    return "left" if arm_index == 0 else "right"


class FactrAPI(Node):
    """One leader arm's ROS<->WebSocket bridge plus HTTP control surface.

    ``WS /ws/{side}`` continuously emits typed JSON frames for both the latest
    ``/joint_pos_{side}`` reading and ``/factr_telemetry_{side}`` live sample,
    and accepts inbound ``force_feedback`` frames that it republishes on
    ``/factr_force_feedback_{side}`` (``POST /force_feedback_{side}`` is the
    stream-less fallback). The legacy-named bodyless
    ``POST /enable_grav_comp_{side}`` and ``POST /disable_grav_comp_{side}``
    routes energize or fully de-energize the leader, and
    ``GET /status_{side}`` remains an on-demand HTTP status query. Everything
    (side, port, topics, routes) is derived from ``arm_index``.
    """

    def __init__(self, app: FastAPI, arm_index: int):
        super().__init__(f"factr_api_{arm_index}")
        arm_index = (
            self.declare_parameter("arm_index", arm_index)
            .get_parameter_value()
            .integer_value
        )

        self.side = _side_for(arm_index)
        self.port = _BASE_PORT + arm_index
        self.topic = f"/joint_pos_{self.side}"
        self.stream_route = f"/ws/{self.side}"
        self.grav_comp_gain_topic = f"/factr_grav_comp_gain_{self.side}"
        self.grav_comp_gain_state_topic = f"/factr_grav_comp_gain_state_{self.side}"
        self.force_feedback_gain_topic = f"/factr_force_feedback_gain_{self.side}"
        self.force_feedback_gain_state_topic = (
            f"/factr_force_feedback_gain_state_{self.side}"
        )
        self.null_space_control_topic = (
            f"/factr_null_space_control_{self.side}"
        )
        self.leader_torque_gain_state_topic = (
            f"/factr_leader_torque_gain_state_{self.side}"
        )
        self.leader_torque_enabled_state_topic = (
            f"/factr_leader_torque_enabled_state_{self.side}"
        )
        self.enable_route = f"/enable_grav_comp_{self.side}"
        self.disable_route = f"/disable_grav_comp_{self.side}"
        self.status_route = f"/status_{self.side}"
        self.telemetry_topic = f"/factr_telemetry_{self.side}"
        self.force_feedback_topic = f"/factr_force_feedback_{self.side}"
        self.force_feedback_route = f"/force_feedback_{self.side}"
        self.gripper_feedback_topic = f"/factr_gripper_feedback_{self.side}"
        self.gripper_feedback_route = f"/gripper_feedback_{self.side}"
        self.enable_force_feedback_route = f"/enable_force_feedback_{self.side}"
        self.disable_force_feedback_route = f"/disable_force_feedback_{self.side}"
        self.null_space_route = f"/null_space_{self.side}"

        self.joint_pos: list[float] = list(_DEFAULT_JOINT_POS)
        self.joint_pos_source_stamp_ns: int = 0
        #: Last targets this relay commanded and the independent live gains reported
        #: by the teleop.
        self.grav_comp_gain_target: float = 0.0
        self.grav_comp_gain: float = 0.0
        self.force_feedback_gain_target: float = 0.0
        self.force_feedback_gain: float = 0.0
        self.leader_torque_gain_target: float = 0.0
        self.leader_torque_gain: float = 0.0
        self.leader_torque_enabled: bool = False
        self.telemetry: dict = {}
        self.telemetry_version = 0
        self.lock = threading.Lock()

        self.create_subscription(JointState, self.topic, self._update_joint_pos, 10)
        self.create_subscription(
            Float64,
            self.grav_comp_gain_state_topic,
            self._update_grav_comp_gain_state,
            10,
        )
        self.create_subscription(
            Float64,
            self.force_feedback_gain_state_topic,
            self._update_force_feedback_gain_state,
            10,
        )
        self.create_subscription(
            Float64,
            self.leader_torque_gain_state_topic,
            self._update_leader_torque_gain_state,
            10,
        )
        self.create_subscription(
            Bool,
            self.leader_torque_enabled_state_topic,
            self._update_leader_torque_enabled_state,
            10,
        )
        self.create_subscription(String, self.telemetry_topic, self._update_telemetry, 10)
        self.grav_comp_gain_pub = self.create_publisher(
            Float64, self.grav_comp_gain_topic, 10
        )
        self.force_feedback_gain_pub = self.create_publisher(
            Float64, self.force_feedback_gain_topic, 10
        )
        self.null_space_control_pub = self.create_publisher(
            Float64MultiArray, self.null_space_control_topic, 10
        )
        # Follower external joint torques in to the teleop's force-feedback term
        # (JointState.effort, one torque per arm joint).
        self.force_feedback_pub = self.create_publisher(
            JointState, self.force_feedback_topic, 10
        )
        self.gripper_feedback_pub = self.create_publisher(
            Float64, self.gripper_feedback_topic, 10
        )

        # One push stream carries independent raw-reading and telemetry frames.
        # Gain status and the enable/disable commands remain request/response HTTP.
        app.add_api_websocket_route(self.stream_route, self.stream)
        app.add_api_route(
            self.status_route, self.get_status, methods=["GET"], response_model=FactrStatus
        )
        app.add_api_route(
            self.enable_route,
            self.enable_grav_comp,
            methods=["POST"],
            response_model=GravCompStatus,
        )
        app.add_api_route(
            self.disable_route,
            self.disable_grav_comp,
            methods=["POST"],
            response_model=GravCompStatus,
        )
        app.add_api_route(
            self.force_feedback_route,
            self.post_force_feedback,
            methods=["POST"],
            response_model=ForceFeedbackAck,
        )
        app.add_api_route(
            self.gripper_feedback_route,
            self.post_gripper_feedback,
            methods=["POST"],
            response_model=GripperFeedbackAck,
        )
        app.add_api_route(
            self.enable_force_feedback_route,
            self.enable_force_feedback,
            methods=["POST"],
            response_model=ForceFeedbackToggleStatus,
        )
        app.add_api_route(
            self.disable_force_feedback_route,
            self.disable_force_feedback,
            methods=["POST"],
            response_model=ForceFeedbackToggleStatus,
        )
        app.add_api_route(
            self.null_space_route,
            self.set_null_space,
            methods=["POST"],
            response_model=NullSpaceControlStatus,
        )
        self.get_logger().info(
            f"FACTR API [{self.side}]: WS :{self.port}{self.stream_route}  <-  "
            f"{self.topic} + {self.telemetry_topic}"
        )
        self.get_logger().info(
            f"FACTR API [{self.side}]: WS force_feedback frames / POST "
            f":{self.port}{self.force_feedback_route}  ->  {self.force_feedback_topic}"
        )
        self.get_logger().info(
            f"FACTR API [{self.side}]: POST :{self.port}"
            f"{self.enable_force_feedback_route} / {self.disable_force_feedback_route}"
            f"  ->  {self.force_feedback_gain_topic}"
        )
        self.get_logger().info(
            f"FACTR API [{self.side}]: POST :{self.port}{self.null_space_route}"
            f"  ->  {self.null_space_control_topic}"
        )
        self.get_logger().info(
            f"FACTR API [{self.side}]: GET :{self.port}{self.status_route}  <-  "
            f"{self.grav_comp_gain_state_topic} + "
            f"{self.force_feedback_gain_state_topic}"
        )
        self.get_logger().info(
            f"FACTR API [{self.side}]: POST :{self.port}{self.enable_route} / "
            f"{self.disable_route}  ->  {self.grav_comp_gain_topic}"
        )

    async def stream(self, websocket: WebSocket) -> None:
        """Serve one client on a duplex WebSocket: push state out, take feedback in.

        Outbound (:meth:`_stream_sender`, a concurrent task): readings at 200 Hz and
        each new live telemetry sample. Frames are tagged with ``type`` so clients
        can update the raw-reading and telemetry caches independently.

        Inbound (this coroutine): arm ``force_feedback`` and ``gripper_feedback``
        frames, republished to their side-specific teleop topics. Malformed frames
        are logged and dropped, never fatal to the stream.
        """
        await websocket.accept()
        sender = asyncio.create_task(self._stream_sender(websocket))
        try:
            while True:
                self._handle_inbound_frame(await websocket.receive_text())
        except (WebSocketDisconnect, KeyError, OSError, RuntimeError):
            # KeyError: a binary frame, which receive_text has no "text" key for.
            pass
        finally:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender

    async def _stream_sender(self, websocket: WebSocket) -> None:
        last_telemetry_version = -1
        period_s = 1.0 / _BROADCAST_HZ
        try:
            while True:
                with self.lock:
                    joint_pos = list(self.joint_pos)
                    source_stamp_ns = self.joint_pos_source_stamp_ns
                    telemetry = dict(self.telemetry)
                    telemetry_version = self.telemetry_version

                if telemetry and telemetry_version != last_telemetry_version:
                    payload = self._telemetry_payload(telemetry)
                    await websocket.send_json({"type": "telemetry", **payload})
                    last_telemetry_version = telemetry_version

                await websocket.send_json({
                    "type": "reading",
                    "side": self.side,
                    "joint_pos": joint_pos,
                    "source_stamp_ns": source_stamp_ns,
                })
                await asyncio.sleep(period_s)
        except (WebSocketDisconnect, OSError, RuntimeError):
            # The push direction notices disconnects on the next send.
            return

    def _handle_inbound_frame(self, message: str) -> None:
        """Validate one client->server frame and republish it to the teleop."""
        try:
            payload = json.loads(message)
            if not isinstance(payload, dict):
                raise ValueError("frame is not a JSON object")
            frame_side = payload.get("side")
            if frame_side is not None and frame_side != self.side:
                raise ValueError(f"frame is for side {frame_side!r}")
            frame_type = payload.get("type")
            if frame_type == "force_feedback":
                feedback = ForceFeedback(
                    **{k: v for k, v in payload.items() if k in ("space", "tau")}
                )
            elif frame_type == "gripper_feedback":
                feedback = GripperFeedback(
                    **{k: v for k, v in payload.items() if k == "force_n"}
                )
            else:
                raise ValueError(f"unknown inbound frame type {frame_type!r}")
        except (ValueError, ValidationError) as exc:
            self.get_logger().warning(
                f"FACTR API [{self.side}]: dropped inbound WS frame: {exc}",
                throttle_duration_sec=1.0,
            )
            return
        if isinstance(feedback, ForceFeedback):
            self._publish_force_feedback(feedback)
        else:
            self._publish_gripper_feedback(feedback)

    async def post_force_feedback(self, feedback: ForceFeedback) -> ForceFeedbackAck:
        """HTTP fallback for one force-feedback sample (debugging, curl).

        The steady-state path is the WebSocket frame; this route exists so the
        feedback pipeline can be exercised without a stream client.
        """
        self._publish_force_feedback(feedback)
        return ForceFeedbackAck(
            side=self.side, space=feedback.space, dof=len(feedback.tau)
        )

    def _publish_force_feedback(self, feedback: ForceFeedback) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.effort = [float(x) for x in feedback.tau]
        self.force_feedback_pub.publish(msg)

    async def post_gripper_feedback(
        self, feedback: GripperFeedback
    ) -> GripperFeedbackAck:
        """HTTP fallback for one signed follower-gripper force sample."""
        self._publish_gripper_feedback(feedback)
        return GripperFeedbackAck(side=self.side, force_n=feedback.force_n)

    def _publish_gripper_feedback(self, feedback: GripperFeedback) -> None:
        msg = Float64()
        msg.data = float(feedback.force_n)
        self.gripper_feedback_pub.publish(msg)

    async def enable_force_feedback(self) -> ForceFeedbackToggleStatus:
        """Ramp only the follower-force term up to its configured strength."""
        self._publish_force_feedback_gain_target(1.0)
        return ForceFeedbackToggleStatus(
            side=self.side, force_feedback_enabled=True, gain_target=1.0,
        )

    async def disable_force_feedback(self) -> ForceFeedbackToggleStatus:
        """Ramp only the follower-force term down, leaving gravity compensation alone."""
        self._publish_force_feedback_gain_target(0.0)
        return ForceFeedbackToggleStatus(
            side=self.side, force_feedback_enabled=False, gain_target=0.0,
        )

    async def set_null_space(
        self, command: NullSpaceControl
    ) -> NullSpaceControlStatus:
        """Publish target and enabled state together for one control-loop update."""
        msg = Float64MultiArray()
        msg.data = [float(command.enabled), *map(float, command.target_dfc_rad)]
        self.null_space_control_pub.publish(msg)
        self.get_logger().info(
            f"FACTR API [{self.side}]: null-space enabled -> {command.enabled}; "
            "task target updated"
        )
        return NullSpaceControlStatus(
            side=self.side,
            enabled=command.enabled,
            target_dfc_rad=command.target_dfc_rad,
        )

    async def get_status(self) -> FactrStatus:
        """Return component gains plus the authoritative hardware torque state."""
        with self.lock:
            grav_comp_gain = self.grav_comp_gain
            grav_comp_gain_target = self.grav_comp_gain_target
            force_feedback_gain = self.force_feedback_gain
            force_feedback_gain_target = self.force_feedback_gain_target
            leader_torque_gain = self.leader_torque_gain
            leader_torque_gain_target = self.leader_torque_gain_target
            leader_torque_enabled = self.leader_torque_enabled
        if leader_torque_gain_target <= 0.01:
            leader_torque_state = (
                "disabled"
                if not leader_torque_enabled and leader_torque_gain <= 0.01
                else "disabling"
            )
        else:
            leader_torque_state = (
                "enabled"
                if leader_torque_enabled and leader_torque_gain >= 0.99
                else "enabling"
            )
        return FactrStatus(
            side=self.side,
            leader_torque_state=leader_torque_state,
            leader_torque_enabled=leader_torque_enabled,
            leader_torque_gain=leader_torque_gain,
            leader_torque_gain_target=leader_torque_gain_target,
            grav_comp_gain=grav_comp_gain,
            grav_comp_gain_target=grav_comp_gain_target,
            force_feedback_gain=force_feedback_gain,
            force_feedback_gain_target=force_feedback_gain_target,
            grav_comp_enabled=(
                leader_torque_state == "enabled" and grav_comp_gain >= 0.99
            ),
            force_feedback_enabled=force_feedback_gain >= 0.99,
        )

    def _telemetry_payload(self, payload: dict) -> dict:
        """Validate and normalize one live telemetry sample for the wire."""
        payload["side"] = self.side
        return TelemetryStatus(**payload).model_dump()

    async def enable_grav_comp(self) -> GravCompStatus:
        """Energize the leader and ramp every configured torque term on."""
        return self._publish_grav_comp_gain_target(1.0)

    async def disable_grav_comp(self) -> GravCompStatus:
        """Ramp all leader torque to zero, then disable Dynamixel torque mode."""
        return self._publish_grav_comp_gain_target(0.0)

    def _publish_grav_comp_gain_target(self, target: float) -> GravCompStatus:
        msg = Float64()
        msg.data = float(target)
        self.grav_comp_gain_pub.publish(msg)
        with self.lock:
            self.grav_comp_gain_target = float(target)
            self.leader_torque_gain_target = float(target)
        self.get_logger().info(
            f"FACTR API [{self.side}]: leader torque + grav comp target -> {target:.1f}"
        )
        return GravCompStatus(side=self.side, gain_target=float(target))

    def _publish_force_feedback_gain_target(self, target: float) -> None:
        msg = Float64()
        msg.data = float(target)
        self.force_feedback_gain_pub.publish(msg)
        with self.lock:
            self.force_feedback_gain_target = float(target)
        self.get_logger().info(
            f"FACTR API [{self.side}]: force feedback gain target -> {target:.1f}"
        )

    def _update_joint_pos(self, msg: JointState) -> None:
        with self.lock:
            self.joint_pos = list(msg.position)
            self.joint_pos_source_stamp_ns = (
                int(msg.header.stamp.sec) * 1_000_000_000
                + int(msg.header.stamp.nanosec)
            )

    def _update_grav_comp_gain_state(self, msg: Float64) -> None:
        with self.lock:
            self.grav_comp_gain = float(msg.data)

    def _update_force_feedback_gain_state(self, msg: Float64) -> None:
        with self.lock:
            self.force_feedback_gain = float(msg.data)

    def _update_leader_torque_gain_state(self, msg: Float64) -> None:
        with self.lock:
            self.leader_torque_gain = float(msg.data)

    def _update_leader_torque_enabled_state(self, msg: Bool) -> None:
        with self.lock:
            self.leader_torque_enabled = bool(msg.data)

    def _update_telemetry(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                raise ValueError("payload is not a JSON object")
            payload = self._telemetry_payload(payload)
            payload.pop("side", None)  # sender adds its authoritative side
        except (TypeError, ValueError, ValidationError) as exc:
            self.get_logger().error(f"invalid telemetry JSON: {exc}")
            return
        with self.lock:
            self.telemetry = payload
            self.telemetry_version += 1
            # The teleop is authoritative. Rehydrate relay state after an API
            # restart instead of showing a false "disabling" target forever.
            self.grav_comp_gain = float(payload["grav_comp_gain"])
            self.grav_comp_gain_target = float(payload["grav_comp_gain_target"])
            self.force_feedback_gain = float(payload["force_feedback_gain"])
            self.force_feedback_gain_target = float(
                payload["force_feedback_gain_target"]
            )
            self.leader_torque_gain = float(payload["leader_torque_gain"])
            self.leader_torque_gain_target = float(
                payload["leader_torque_gain_target"]
            )
            self.leader_torque_enabled = bool(payload["leader_torque_enabled"])


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
    # topics and WebSocket route; HTTP remains only for control/status calls.
    app_left, app_right = FastAPI(), FastAPI()
    node_left = FactrAPI(app_left, 0)    # WS :5000/ws/left  <- left ROS topics
    node_right = FactrAPI(app_right, 1)  # WS :5001/ws/right <- right ROS topics

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
