# ---------------------------------------------------------------------------
# FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning
# https://arxiv.org/abs/2502.17432
# Copyright (c) 2025 Jason Jingzhou Liu and Yulong Li

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------


import time
import numpy as np

import rclpy
from factr_teleop.factr_teleop import FACTRTeleop

class FACTRTeleopGravComp(FACTRTeleop):
    """
    This class demonstrates the gravity compensation and null-space regulation function of the 
    FACTR teleop leader arm. Communication between the leader arm and the follower Franka arm
    is not implemented in this example.
    """

    def __init__(self):
        super().__init__()

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
    factr_teleop_grav_comp = FACTRTeleopGravComp()

    try:
        while rclpy.ok():
            rclpy.spin(factr_teleop_grav_comp)
    except KeyboardInterrupt:
        factr_teleop_grav_comp.get_logger().info("Keyboard interrupt received. Shutting down...")
        factr_teleop_grav_comp.shut_down()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

