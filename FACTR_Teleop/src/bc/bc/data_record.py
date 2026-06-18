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

import rclpy
from rclpy.node import Node
from pynput import keyboard

import pickle
from pathlib import Path
from termcolor import colored

from bc import utils
from sensor_msgs.msg import JointState, Image
from python_utils.utils import get_workspace_root


class DataRecord(Node):
    def __init__(self, name="data_record_node"):
        super().__init__(name)

        self.declare_parameter('state_topics', [""])
        self.state_topics = self.get_parameter('state_topics').value
        
        self.declare_parameter('image_topics', [""])
        self.image_topics = self.get_parameter('image_topics').value
        
        self.declare_parameter('dataset_name', "")
        dataset_name = self.get_parameter('dataset_name').value
        self.output_dir = Path(f"{get_workspace_root()}/raw_data/{dataset_name}")
        
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f"Saving to {self.output_dir}")
        
        self.recording = False 

        listener = keyboard.Listener(on_press=self.on_press_key)
        listener.start()
        
        self.topics_to_record = []
        for state_topic in self.state_topics:
            callback = self.create_callback(state_topic)
            self.create_subscription(JointState, state_topic, callback, 10)
            self.topics_to_record.append(state_topic)
        for image_topic in self.image_topics:
            callback = self.create_callback(image_topic)
            self.create_subscription(Image, image_topic, callback, 1)
            self.topics_to_record.append(image_topic)
        
        self.get_logger().info(colored(f"{self.topics_to_record}", 'green'))
    
    def get_timestamp(self):
        current_time = self.get_clock().now().to_msg()
        time_ns = utils.ros2_time_to_ns(current_time)
        return time_ns
    
    def create_callback(self, topic_name):
        def callback(msg):
            if not self.recording:
                return
            time_ns = self.get_timestamp()
            data = utils.process_msg(msg)
            self.data_log["data"][topic_name].append(data)
            self.data_log["timestamps"][topic_name].append(time_ns)
            self.data_log["all_timestamps"].append(time_ns)
        return callback

    def save_data(self, ep_index):
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"ep_{ep_index:05d}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(self.data_log, f, protocol=pickle.HIGHEST_PROTOCOL)
        duration = (self.data_log['all_timestamps'][-1] - self.data_log['all_timestamps'][0]) / 1e9
        total_messages = len(self.data_log['all_timestamps'])
        self.get_logger().info(colored(f"Data saved to {output_path}, traj duration={duration:.2f}s, total messages={total_messages}", 'light_blue'))

    def delete_last_trajectory(self):
        if not self.output_dir.exists():
            self.get_logger().info(colored(f"{self.output_dir} does not exist", 'light_blue'))
            return
        all_episodes = [f for f in self.output_dir.iterdir() if f.name.startswith('ep_') and f.name.endswith('.pkl')]
        sorted_episodes = sorted(all_episodes, key=lambda x: int(x.name.split('_')[1].split('.')[0]))
        if len(sorted_episodes) == 0:
            self.get_logger().info(colored(f"No trajectories to delete", 'light_blue'))
            return
        output_path = sorted_episodes[-1]
        output_path.unlink()
        self.get_logger().info(colored(f"Deleted trajectory {output_path}", 'light_blue'))

    def on_press_key(self, key):
        """Callback function for key press events."""
        try:
            if key == keyboard.Key.delete:
                self.delete_last_trajectory()
                return
            elif key == keyboard.Key.space:
                if not self.recording:
                    self.get_logger().info(f"Starting data recording")
                    # initialize data log
                    self.data_log = {
                        "data": {},
                        "timestamps": {},
                        "all_timestamps": [],
                    }
                    for topic in self.topics_to_record:
                        self.data_log["data"][topic] = []
                        self.data_log["timestamps"][topic] = []
                    self.recording = True
                else:
                    self.get_logger().info(f"Stopping data recording")
                    self.recording = False
                    
                    all_episodes = [f for f in self.output_dir.iterdir() if f.name.startswith('ep_') and f.name.endswith('.pkl')]
                    ep_index = len(all_episodes)
                    self.save_data(ep_index)
            else:
                self.get_logger().info("Press space to start/stop recording; press delete to delete last trajectory")

        except AttributeError:
            pass

def main(args=None):
    rclpy.init(args=args)
    data_record_node = DataRecord()
    rclpy.spin(data_record_node)
    data_record_node.destroy_node()
    rclpy.shutdown()
