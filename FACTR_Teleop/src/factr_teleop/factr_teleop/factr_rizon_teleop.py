# 2026 Jun 18
# Leo Wang

# -- ROS2 -- 
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading

# -- Message types -- 
from std_msgs.msg import String
from sensor_msgs.msg import JointState # from ROS2

from factr_teleop.factr_teleop import FACTRTeleop

import numpy as np
import time

class FactrRizonTeleop(FACTRTeleop):
    def __init__(self):
        super().__init__()
        self.joint_positions = np.zeros(7)
        self.joint_velocities = np.zeros(7) # later

        self.group_a = MutuallyExclusiveCallbackGroup()
        self.group_b = MutuallyExclusiveCallbackGroup()

        self.joint_pos_publisher = self.publisher_ = self.create_publisher(JointState, '/joint_pos', 10, callback_group=self.group_a)
        self.create_timer(0.002, self.publish_joint_pos, callback_group=self.group_b)
        # publish joint_pos every 2ms


    def get_leader_joint_pos(self):
        """
        Returns ONLY the current joint positions
        """
        self.gripper_pos_prev = self.gripper_pos
        joint_pos, joint_vel = self.driver.get_positions_and_velocities()
        print("joint_pos ABABABAB: ", joint_pos)
        return joint_pos


    # def control_loop_callback(self):
    #     """
    #     Uncomment this function if you just want to see the joint positions!
    #     """
    #     joint_pos = self.get_leader_joint_pos()
    #     self.joint_positions = joint_pos

    
    def publish_joint_pos(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f'joint_{i}' for i in range(7)]

        msg.position = self.joint_positions.tolist()   # np.array -> list[float]
        msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        print(f"Publishing joint positions: {self.joint_positions}")
        self.joint_pos_publisher.publish(msg)

        


    def set_up_communication(self):
        pass
        
    def get_leader_gripper_feedback(self):
        pass
    
    def gripper_feedback(self, leader_gripper_pos, leader_gripper_vel, gripper_feedback):
        pass
    
    def get_leader_arm_external_joint_torque(self):
        pass

    def update_communication(self, leader_arm_pos, leader_gripper_pos):
        pass
        


def main(args=None):
    rclpy.init(args=args)
    FRT= FactrRizonTeleop() # single thread is sufficient since it's only publishing messages. 

    executor = MultiThreadedExecutor() # mutli thread needed
    executor.add_node(FRT)
    executor.spin()

    rclpy.shutdown()

if __name__ == "__main__":
    main()