# 2026 Jul 13
# This file is intended for controlling the right leader arm with two seperate power boards

# -- ROS2 -- 
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# -- Message types -- 
from std_msgs.msg import String
from sensor_msgs.msg import JointState # from ROS2

# Relative import so the base class loads from THIS src tree (the launch runs
# `-m src.factr_teleop...`); a bare `factr_teleop.` import would resolve to the stale
# colcon install/ copy and miss edits made here.
from .factr_teleop_dual_base import FACTRTeleopDualBase
from .gain_control import compose_arm_torque

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

        self.group_a = MutuallyExclusiveCallbackGroup()
        self.group_b = MutuallyExclusiveCallbackGroup()

        self.joint_pos_publisher = self.publisher_ = self.create_publisher(JointState, '/joint_pos_right', 10, callback_group=self.group_a)

        self.create_timer(0.002, self.publish_joint_pos, callback_group=self.group_b)
        # publish joint_pos every 2ms

        self.index = arm_index


    def get_leader_joint_pos(self):
        """
        Return the cached merged positions from the latest control-loop read.

        This performs no board I/O; both boards are read once in
        ``get_leader_joint_states`` and that exact merged sample is published.
        """
        return self.get_cached_raw_joint_state().position


    def control_loop_callback(self):    
        """
        Additional control loop feature: update the joint positions of one of the leader arms (left or right), runs at 500Hz
        """

        leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel = self.get_leader_joint_states()

        torque_l, torque_gripper = self.joint_limit_barrier(
            leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel
        )
        torque_null = self.null_space_regulation(leader_arm_pos, leader_arm_vel)
        torque_gravity = np.zeros(self.num_arm_joints)
        torque_friction = np.zeros(self.num_arm_joints)

        if self.enable_gravity_comp:
            torque_gravity = self.gravity_compensation(leader_arm_pos, leader_arm_vel)
            torque_friction = self.friction_compensation(leader_arm_vel)
        torque_feedback = np.zeros(self.num_arm_joints)
        if self.enable_torque_feedback:
            external_joint_torque = self.get_leader_arm_external_joint_torque()
            torque_feedback = self.torque_feedback(
                external_joint_torque, leader_arm_vel
            )

        # if self.enable_gripper_feedback:
        #     gripper_feedback = self.get_leader_gripper_feedback()
        #     torque_gripper += self.gripper_feedback(leader_gripper_pos, leader_gripper_vel, gripper_feedback)

        grav_gain, feedback_gain = self._update_component_gains()
        torque_arm = compose_arm_torque(
            torque_l,
            torque_null,
            torque_gravity,
            torque_friction,
            torque_feedback,
            grav_gain,
            feedback_gain,
        )
        self._capture_enable_tick(
            leader_arm_pos, leader_arm_vel, torque_l, torque_null,
            torque_gravity, torque_friction, torque_feedback, torque_arm, grav_gain,
        )
        self.set_leader_joint_torque(torque_arm, torque_gripper)

        self._log_servo_health()


    def publish_joint_pos(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f'joint_{i}' for i in range(7)]

        positions = self.get_leader_joint_pos().tolist()
        msg.position = positions

        msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 

        if len(positions) > 8:
            self.get_logger().info("motors not ready")

        self.joint_pos_publisher.publish(msg)


    def set_up_communication(self):
        pass
        
    def get_leader_gripper_feedback(self):
        pass

    def gripper_feedback(self, leader_gripper_pos, leader_gripper_vel, gripper_feedback):
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

    # index 0 is left, 1 is right
    rclpy.init(args=args)
    right_factr = FactrRizonTeleopDualBoard(1)

    executor = MultiThreadedExecutor() # mutli thread needed
    executor.add_node(right_factr)
    try:
        executor.spin()
    except KeyboardInterrupt:
        right_factr.get_logger().info("KeyboardInterrupt: de-energizing servos and closing boards...")
    finally:
        # Graceful shutdown: zero + disable torque on BOTH boards, then close the ports.
        try:
            right_factr.shut_down()
        except Exception as e:
            right_factr.get_logger().error(f"shut_down failed: {e}")
        for driver in right_factr._drivers():
            try:
                driver.close()
            except Exception:
                pass
        right_factr.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main()
