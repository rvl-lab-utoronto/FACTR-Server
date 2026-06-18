
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

import yaml
import functools
import numpy as np
from pathlib import Path
from cv_bridge import CvBridge
from collections import defaultdict, deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image

from bc.utils import create_joint_state_msg

class Rollout(Node):
    def __init__(self):
        super().__init__('rollout_node')

        self.data_dir = Path(self.declare_parameter('data_dir', "").get_parameter_value().string_value)
        self.init_rollout_config()
        self.state_obs = defaultdict(lambda: deque(maxlen=20))
        self.image_obs = {}
        self.init_messengers()
        self.bridge = CvBridge()
    
    def init_rollout_config(self):
        """
        Initialize the rollout configuration.
        Load normalization stats, observation topics, and action config.
        """
        with open(self.data_dir / "rollout_config.yaml", "r") as f:
            self.rollout_config = yaml.safe_load(f)

        self.action_norm_stats = None
        self.state_norm_stats = None
        if "norm_stats" in self.rollout_config:
            self.action_norm_stats = self.rollout_config["norm_stats"]["action"]
            self.state_norm_stats = self.rollout_config["norm_stats"]["state"]
        
        self.obs_config = self.rollout_config["obs_config"]
        self.state_topics = self.obs_config["state_topics"]
        self.camera_topics = self.obs_config["camera_topics"]
        
        self.action_config = self.rollout_config["action_config"]
    
    def _low_dim_callback(self, msg: JointState, name: str):
        """
        Callback for low-dimensional state messages.
        """
        self.state_obs[name].append(np.array(msg.position))
    
    def _image_callback(self, msg: Image, name: str):
        """
        Callback for image messages.
        """
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.image_obs[name] = image
    
    def init_messengers(self):
        """
        Initialize subscribers for observation topics and publishers for action topics.
        """
        self.get_logger().info(f"Initializing messengers")
        self.obs_subscribers = {}
        for state_topic in self.state_topics:
            self.obs_subscribers[state_topic] = self.create_subscription(
                JointState, state_topic, 
                functools.partial(self._low_dim_callback, name=state_topic),
                1,
            )
        for camera_topic in self.camera_topics:
            self.obs_subscribers[camera_topic] = self.create_subscription(
                Image, camera_topic, 
                functools.partial(self._image_callback, name=camera_topic),
                1,
            )
        
        self.action_publishers = {}
        for action_topic in self.action_config.keys():
            self.action_publishers[action_topic] = self.create_publisher(JointState, action_topic, 10)
    
    def decode_action(self, action):
        """
        Decode a raw action into a dictionary of robot actions for publishing.
        """
        # unnormalize action
        if self.action_norm_stats is not None:
            mean = np.array(self.action_norm_stats["mean"])
            std = np.array(self.action_norm_stats["std"])
            action = action * std + mean
        # decode action
        dim_pointer = 0
        action_dict = {}
        for action_key, action_dim in self.action_config.items():
            action_dict[action_key] = action[dim_pointer:dim_pointer+action_dim]
            dim_pointer += action_dim
        return action_dict
        
    def send_command(self, action_dict: dict):
        """
        Send commands to the robots.
        """
        for action_key, action_value in action_dict.items():
            self.action_publishers[action_key].publish(create_joint_state_msg(action_value))
        
    
def main(args=None):
    rclpy.init(args=args)
    node = Rollout()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()