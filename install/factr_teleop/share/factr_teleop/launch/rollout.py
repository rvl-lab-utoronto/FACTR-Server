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

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    franka_bridge_node = Node(
        package='bc',
        executable='franka_bridge',
        name='franka_bridge_node',          
        output='screen',                
        parameters=[
            {"torque_feedback": True},
        ]
    )
    
    policy_rollout_node = Node(
        package='bc',
        executable='policy_rollout',
        name='policy_rollout_node',          
        output='screen',                
        parameters=[
            {"save_data": True},
            {"data_dir":"../factr/checkpoints/test/rollout"}

        ]
    )
    
    zed_node = Node(
        package='cameras',
        executable='zed',
        name='front',
        output='screen',
        emulate_tty=True,
        parameters=[
            {"serial": 22176523},
            {"name": "front"},
        ]
    )
    
    return LaunchDescription([
        franka_bridge_node,
        zed_node,
        policy_rollout_node,
    ])
