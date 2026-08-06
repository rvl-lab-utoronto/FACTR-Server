# 2026 Jun 18

# -- ROS2 -- 
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# -- Message types -- 
from std_msgs.msg import String
from sensor_msgs.msg import JointState # from ROS2

# -- FastAPI 
# from fact import FACTRAPI

# Relative import so the base class loads from THIS src tree (the launch runs
# `-m src.factr_teleop...`); a bare `factr_teleop.` import would resolve to the stale
# colcon install/ copy and miss edits made here (e.g. the master force-gain ramp).
from .factr_teleop import FACTRTeleop

import time
import math


"""
improvements:
    1, enforce joint limit 
    2, I think the servos have memories about its joint positions
"""

def rad_to_deg_signed(rad):
        deg = math.degrees(rad) % 360
        if deg > 180:
            deg -= 360
        return deg

class FactrRizonTeleop(FACTRTeleop):
    def __init__(self, arm_index: int):
        super().__init__(arm_index)

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

        self.index = arm_index


    def get_leader_joint_pos(self):
        """
        Return the cached raw positions from the latest control-loop read.

        This method deliberately performs no Dynamixel I/O. DFC must receive the
        exact sample used by gravity compensation, not an independent second read.
        """
        return self.get_cached_raw_joint_state().position


    def control_loop_callback(self):    
        """
        Additional control loop feature: update the joint positions of one of the leader arms (left or right), runs at 500Hz
        """
        super().control_loop_callback()


    def publish_joint_pos(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f'joint_{i}' for i in range(7)]

        positions = self.get_leader_joint_pos().tolist()
        msg.position = positions

        msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 

        if len(positions) < 8:
            self.get_logger().info("motors not ready", throttle_duration_sec=0.5)
        else:
            arm_name = "left" if self.index == 0 else "right"

            readout = " | ".join(
                f"A{i+1}:{rad_to_deg_signed(positions[i]):.1f}" for i in range(7)
            )
            readout += f" | G1:{rad_to_deg_signed(positions[7]):.1f}"
            self.get_logger().info(f"{arm_name}: {readout}", throttle_duration_sec=0.5)

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
        


def main(args=None):
    # before running this file, goto factr_rizon_left.yaml and factr_rizon_right.yaml
    # change dynamixel_port: "xxxx_xxx" to your USB port!

    # print("modify your USB port first!") 
    # return # if you have done so, comment this line!

    rclpy.init(args=args)
    left_factr = FactrRizonTeleop(0)  
    # right_factr = FactrRizonTeleop(1)   

    executor = MultiThreadedExecutor() # mutli thread needed
    executor.add_node(left_factr)
    # executor.add_node(right_factr)
    try:
        executor.spin()
    except KeyboardInterrupt:
        left_factr.get_logger().info("KeyboardInterrupt: de-energizing servos and closing board...")
    finally:
        # Graceful shutdown: zero + disable torque, then close the serial port.
        try:
            left_factr.shut_down()
        except Exception as e:
            left_factr.get_logger().error(f"shut_down failed: {e}")
        try:
            left_factr.driver.close()
        except Exception:
            pass
        left_factr.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main()
