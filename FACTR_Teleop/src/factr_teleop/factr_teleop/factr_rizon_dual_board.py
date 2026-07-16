# 2026 Jul 13
# This file is intended for controlling the right leader arm with two seperate power boards

# -- ROS2 -- 
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading

# -- Message types -- 
from std_msgs.msg import String
from sensor_msgs.msg import JointState # from ROS2

from factr_teleop.factr_teleop_dual_base import FACTRTeleopDualBase

import numpy as np
import time
import math

import time


def rad_to_deg_signed(rad):
        deg = math.degrees(rad) % 360
        if deg > 180:
            deg -= 360
        return deg

msg = "something"


class FactrRizonTeleopDualBoard(FACTRTeleopDualBase):
    def __init__(self, arm_index: int):
        super().__init__(arm_index)
        self.joint_positions = np.zeros(7)
        self.joint_velocities = np.zeros(7) # later

        self.group_a = MutuallyExclusiveCallbackGroup()
        self.group_b = MutuallyExclusiveCallbackGroup()

        self.joint_pos_publisher = self.publisher_ = self.create_publisher(JointState, '/joint_pos_right', 10, callback_group=self.group_a)

        self.create_timer(0.002, self.publish_joint_pos, callback_group=self.group_b)
        # publish joint_pos every 2ms

        self.index = arm_index
        self.lock = threading.Lock()


    def get_leader_joint_pos(self):
        """
        Returns ONLY the current joint positions, merged over both boards in full
        arm order (config-driven via the base class).
        """
        self.gripper_pos_prev = self.gripper_pos
        joint_pos, _ = self._read_merged_pos_vel()
        return joint_pos


    def control_loop_callback(self):    
        """
        Additional control loop feature: update the joint positions of one of the leader arms (left or right), runs at 500Hz
        """

        leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel = self.get_leader_joint_states()

        torque_arm = np.zeros(self.num_arm_joints)
        torque_l, torque_gripper = self.joint_limit_barrier(
            leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel
        )
        torque_arm += torque_l
        torque_arm += self.null_space_regulation(leader_arm_pos, leader_arm_vel)

        if self.enable_gravity_comp:
            torque_arm += self.gravity_compensation(leader_arm_pos, leader_arm_vel)
            torque_arm += self.friction_compensation(leader_arm_vel)
        
        # if self.enable_torque_feedback:
        #     print(self.get_leader_arm_external_joint_torque())
        #     external_joint_torque = self.get_leader_arm_external_joint_torque()
        #     torque_arm += self.torque_feedback(external_joint_torque, leader_arm_vel)
        
        # if self.enable_gripper_feedback:
        #     gripper_feedback = self.get_leader_gripper_feedback()
        #     torque_gripper += self.gripper_feedback(leader_gripper_pos, leader_gripper_vel, gripper_feedback)

        self.set_leader_joint_torque(torque_arm, torque_gripper)
        
        # update joint positions
        joint_pos = self.get_leader_joint_pos()
        self.joint_positions = joint_pos


    def publish_joint_pos(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f'joint_{i}' for i in range(7)]

        with self.lock:
            positions = self.joint_positions.tolist()   # np.array -> list[float]
            msg.position = positions

        msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 

        if len(positions) > 8:
            self.get_logger().info("motors not ready")
        else:   
            # for testing
            print("\033[H\033[2J", end="")

            # TODO change the size to 7 after big servos are added

            for i in range(7):
                print(f"A{i+1} right: {(positions[i])}")
            print(f"G1 right: {(positions[-1])}")

            # for i in range(7):
            #     print(f"A{i+1} right: {rad_to_deg_signed(positions[i])}")
            # print(f"G1 right: {rad_to_deg_signed(positions[-1])}")

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
        


async def web_server(websocket):
    while True:
        await websocket.send(msg)



def main(args=None):
    # before running this file, goto factr_rizon_left.yaml and factr_rizon_right.yaml
    # change dynamixel_port: "xxxx_xxx" to your USB port!

    # print("modify your USB port first!") 
    # return # if you have done so, comment this line!

    # index 0 is left, 1 is right
    rclpy.init(args=args)
    right_factr = FactrRizonTeleopDualBoard(1)

    executor = MultiThreadedExecutor() # mutli thread needed
    executor.add_node(right_factr)
    executor.spin()

    rclpy.shutdown()

if __name__ == "__main__":
    main()

    