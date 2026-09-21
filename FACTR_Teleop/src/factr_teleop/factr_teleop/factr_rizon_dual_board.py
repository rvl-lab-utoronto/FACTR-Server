# 2026 Jul 13
# This file is intended for controlling the right leader arm with two seperate power boards

# -- ROS2 -- 
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

# -- Message types -- 
from std_msgs.msg import String
from sensor_msgs.msg import JointState # from ROS2

# Relative import so the base class loads from THIS src tree (the launch runs
# `-m src.factr_teleop...`); a bare `factr_teleop.` import would resolve to the stale
# colcon install/ copy and miss edits made here.
from .factr_teleop_dual_base import FACTRTeleopDualBase
from .gain_control import compose_arm_torque
from .dynamixel.driver import DynamixelReadError

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

        self.joint_pos_publisher = self.publisher_ = self.create_publisher(
            JointState, '/joint_pos_right', 10
        )

        self.index = arm_index
        self._last_published_joint_sequence = 0
        self._initialize_position()


    def get_leader_joint_pos(self):
        """
        Return the cached merged positions from the latest control-loop read.

        This performs no board I/O; both boards are read once in
        ``get_leader_joint_states`` and that exact merged sample is published.
        """
        return self.get_cached_raw_joint_state().position


    def control_loop_callback(self):
        """
        Acquire state and update one leader-arm torque command.
        """

        try:
            leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel = self.get_leader_joint_states()
        except DynamixelReadError as exc:
            # The next dedicated-loop iteration is the retry. Never spend one
            # control iteration making repeated serial attempts; retain the
            # previous torque command until a complete state is acquired.
            self.get_logger().warning(
                f"FACTR TELEOP {self.name}: skipped control tick after {exc}",
                throttle_duration_sec=1.0,
            )
            self._disable_leader_torque_after_read_failure()
            return False

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

        null_gain, grav_gain, feedback_gain, leader_gain = self._update_component_gains()
        if self.enable_gripper_spring:
            torque_gripper += self.gripper_spring_torque(
                leader_gripper_pos, leader_gripper_vel
            )
        torque_arm = compose_arm_torque(
            torque_l,
            torque_null,
            torque_gravity,
            torque_friction,
            torque_feedback,
            null_gain,
            grav_gain,
            feedback_gain,
            leader_gain,
        )
        commanded_torque = self._apply_leader_torque(
            torque_arm, leader_gain * torque_gripper
        )
        self._last_commanded_torque_nm = commanded_torque.copy()
        self._capture_enable_tick(
            leader_arm_pos, leader_arm_vel, torque_l, torque_null,
            torque_gravity, torque_friction, torque_feedback, torque_arm,
            commanded_torque, grav_gain,
        )
        self.publish_joint_pos()
        return True


    def publish_joint_pos(self):
        state = self.get_cached_raw_joint_state()
        if state.sequence <= self._last_published_joint_sequence:
            return

        msg = JointState()
        # Preserve when the merged Dynamixel acquisition completed, not when this
        # 2 ms publisher timer happened to observe it. Convert the monotonic cache
        # age to the ROS wall clock so the API/client can measure end-to-end age.
        now_monotonic_ns = time.monotonic_ns()
        now_ros_ns = self.get_clock().now().nanoseconds
        source_ros_ns = max(
            0,
            now_ros_ns - max(0, now_monotonic_ns - state.stamp_monotonic_ns),
        )
        msg.header.stamp.sec = int(source_ros_ns // 1_000_000_000)
        msg.header.stamp.nanosec = int(source_ros_ns % 1_000_000_000)
        msg.name = [f'joint_{i}' for i in range(7)]

        positions = state.position.tolist()
        msg.position = positions

        msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        msg.effort = self._last_commanded_torque_nm.tolist()

        if len(positions) > 8:
            self.get_logger().info("motors not ready")

        self.joint_pos_publisher.publish(msg)
        self._last_published_joint_sequence = state.sequence


    def set_up_communication(self):
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

    # Serial I/O and torque control have their own thread. Two ROS workers are
    # sufficient for the always-on force-feedback subscription and telemetry/gain
    # callbacks, without the 48-worker GIL contention of the default constructor.
    executor = MultiThreadedExecutor(num_threads=2)
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
