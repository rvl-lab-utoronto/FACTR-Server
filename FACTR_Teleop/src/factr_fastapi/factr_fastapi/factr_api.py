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

import threading

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


#: DoF+1 placeholder (7 arm joints + trailing gripper) served before a publisher connects.
_DEFAULT_JOINT_POS = [0.0, 0.0, 0.0, 3.12, 0.0, 0.0, 0.0, 0.0]
_BASE_PORT = 5000
API_PORT = 5001

class JointResponse(BaseModel):
    joint_pos: list[float]


def _side_for(arm_index: int) -> str:
    return "left" if arm_index == 0 else "right"


class FactrAPI(Node):
    """One leader arm's ROS->HTTP bridge.

    Subscribes to ``/joint_pos_{side}`` and serves the latest positions at
    ``GET /get_joint_positions_{side}``. Everything (side, port, topic, route) is
    derived from the ``arm_index`` ROS parameter, so the same executable is
    launched once per leader arm.
    """

    def __init__(self, app: FastAPI, arm_index: int):
        super().__init__(f"factr_api_{arm_index}")
        arm_index = self.declare_parameter("arm_index", arm_index).get_parameter_value().integer_value

        self.side = _side_for(arm_index)
        self.port = _BASE_PORT + arm_index
        self.topic = f"/joint_pos_{self.side}"
        self.route = f"/get_joint_positions_{self.side}"

        self.joint_pos: list[float] = list(_DEFAULT_JOINT_POS)
        self.lock = threading.Lock()

        self.create_subscription(JointState, self.topic, self._update_joint_pos, 10)

        # Register this instance's single per-side route dynamically.
        app.add_api_route(
            self.route, self.get_joint_positions, methods=["GET"], response_model=JointResponse
        )
        self.get_logger().info(
            f"FACTR API [{self.side}]: GET :{self.port}{self.route}  <-  {self.topic}"
        )

    async def get_joint_positions(self) -> JointResponse:
        """Return this leader's most recent joint positions (7 arm joints + gripper)."""
        with self.lock:
            return JointResponse(joint_pos=list(self.joint_pos))

    def _update_joint_pos(self, msg: JointState) -> None:
        with self.lock:
            self.joint_pos = list(msg.position)


def _ros_spin(node: Node) -> None:
    """Spin ROS in a background thread so uvicorn can own the main thread."""
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    executor.shutdown()
    node.destroy_node()


def main(args=None):
    rclpy.init(args=args)

    app = FastAPI()
    # 1 is the right arm, 0 is the left arm
    node_right = FactrAPI(app, 1) 
    node_left = FactrAPI(app, 0) # reads joint positions of the left arm

    left_thread = threading.Thread(target=_ros_spin, args=(node_left,), daemon=True)
    left_thread.start()
    right_thread = threading.Thread(target=_ros_spin, args=(node_right,), daemon=True)
    right_thread.start()

    uvicorn.run(app, port = API_PORT, log_level="warning")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
