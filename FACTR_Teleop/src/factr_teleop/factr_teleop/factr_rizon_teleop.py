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

# -- FastAPI 
# from fact import FACTRAPI

from factr_teleop.factr_teleop import FACTRTeleop

import numpy as np
import time

class FactrRizonTeleop(FACTRTeleop):
    def __init__(self, arm_index: int):
        super().__init__(arm_index)
        self.joint_positions = np.zeros(7)
        self.joint_velocities = np.zeros(7) # later

        self.group_a = MutuallyExclusiveCallbackGroup()
        self.group_b = MutuallyExclusiveCallbackGroup()

        if arm_index == 0:
            # left arm
            self.joint_pos_publisher = self.publisher_ = self.create_publisher(JointState, '/joint_pos_left', 10, callback_group=self.group_a)
        else:
            # right arm
            self.joint_pos_publisher = self.publisher_ = self.create_publisher(JointState, '/joint_pos_right', 10, callback_group=self.group_a)
            
        self.create_timer(0.002, self.publish_joint_pos, callback_group=self.group_b)
        # publish joint_pos every 2ms


    def get_leader_joint_pos(self):
        """
        Returns ONLY the current joint positions
        """
        self.gripper_pos_prev = self.gripper_pos
        joint_pos, joint_vel = self.driver.get_positions_and_velocities()
        return joint_pos


    def control_loop_callback(self):    
        """
        Additional control loop feature: update the joint positions of one of the leader arms (left or right), runs at 500Hz
        """
        super().control_loop_callback() 

        # update joint positions
        joint_pos = self.get_leader_joint_pos()
        self.joint_positions = joint_pos


    
    def publish_joint_pos(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f'joint_{i}' for i in range(7)]

        msg.position = self.joint_positions.tolist()   # np.array -> list[float]
        msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        self.get_logger().info(f"publishing joint positions: {self.joint_positions}")
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
        # use publish_joint_pos() instead
        pass
        


def main(args=None):
    rclpy.init(args=args)
    left_factr = FactrRizonTeleop(0)  
    # right_factr = FactrRizonTeleop(1)   

    executor = MultiThreadedExecutor() # mutli thread needed
    executor.add_node(left_factr)
    # executor.add_node(right_factr)
    executor.spin()

    rclpy.shutdown()

if __name__ == "__main__":
    main()