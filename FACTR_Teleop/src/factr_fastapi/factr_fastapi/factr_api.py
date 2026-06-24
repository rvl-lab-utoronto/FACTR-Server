# 2026 Jun 23

# -- ROS2 -- 
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading
from sensor_msgs.msg import JointState

# -- FastAPI -- 
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel


app = FastAPI()


class JointResponse(BaseModel):
    joint_pos: list[float]


class FactrAPI(Node):
    def __init__(self):
        super().__init__("factr_api")
        group_b = MutuallyExclusiveCallbackGroup()
        group_c = MutuallyExclusiveCallbackGroup()
        self.joint_subscriber_left = self.create_subscription(JointState, '/joint_pos_left', self.update_joint_pos_left, 10, callback_group = group_b)
        # self.joint_subscriber_right = self.create_subscription(JointState, '/joint_pos_right', self.update_joint_pos_right,10, callback_group = group_c)

        self.joint_pos_left: list[float] = [0.0, 0.0, 0.0, 3.12, 0.0, 0.0, 0.0, 0.0]
        self.joint_pos_right: list[float] = [0.0, 0.0, 0.0, 3.12, 0.0, 0.0, 0.0, 0.0]


    @app.get("/get_joint_positions_left", response_model=JointResponse)
    async def get_joint_positions(request: Request):
        """
        GET endpoint: returns the current joint positions of the 7 joints and the gripper of the FACTR leader.  
        """
        node = request.app.state.node 
        response = JointResponse(joint_pos=node.joint_pos_left)
        node.get_logger().info("new joint position request")
        return response


    def update_joint_pos_left(self, msg):
        self.joint_pos_left = list(msg.position)

    def update_joint_pos_right(self, msg):
        self.joint_pos_right = list(msg.position)


def ros2_multithread(node):
    """ Need this function to run multithread """
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()

    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


def main(args=None):
    rclpy.init()
    factrapi = FactrAPI()
    app.state.node = factrapi  # acting like self

    ros_thread = threading.Thread(target=ros2_multithread, args=(factrapi,), daemon=True)
    ros_thread.start()

    uvicorn.run(app, port=5000, log_level="warning")

    ros_thread.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()