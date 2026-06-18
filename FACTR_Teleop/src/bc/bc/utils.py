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


import cv2
import numpy as np

from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseArray, PoseStamped

from cv_bridge import CvBridge


def ros2_time_to_ns(ros2_time):
    """
    Convert a ROS2 time to nanoseconds.
    """
    return int(ros2_time.sec * 1e9 + ros2_time.nanosec)

def create_joint_state_msg(data):
    """
    Create a JointState message from a numpy array.
    """
    msg = JointState()
    msg.position = list(map(float, data))
    return msg

def process_msg(msg):
    """
    Process a ROS2 message and return a numpy array.
    Low-dimensional data is returned as a 1D numpy array, while image data is returned as a JPEG encoded byte string.
    """
    cv_bridge = CvBridge()
    if isinstance(msg, JointState):
        data = np.array(msg.position)
    elif isinstance(msg, PoseArray):
        poses = msg.poses
        data = []
        for pose in poses:
            pose = np.array([
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
            ])
            data.append(pose)
        data = np.array(data)
    elif isinstance(msg, Image):
        if msg.encoding == "rgb8":
            cv_image = cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        elif msg.encoding == "32FC1":
            cv_image = cv_bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        else:
            raise ValueError(f"Unsupported image encoding {msg.encoding}")
        _, data = cv2.imencode('.jpg', cv_image)
    elif isinstance(msg, PoseStamped):
        data = np.array([
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
        ])
    else:
        raise ValueError(f"Unsupported message type {type(msg)}")
    return data