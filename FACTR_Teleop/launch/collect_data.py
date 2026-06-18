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

    data_record_node = Node(
        package='bc',
        executable='data_record',
        name='data_record_node',          
        output='screen',                
        parameters=[
            {
                "state_topics": [
                    "/factr_teleop/right/cmd_franka_pos",
                    "/franka/right/obs_franka_torque",
                ]
            },
            {
                "image_topics": [
                    "/zed/front/im_left",
                ]
            },
            {"dataset_name": "test"}
        ]
    )
    
    factr_teleop_franka_right = Node(
        package='factr_teleop',
        executable='factr_teleop_franka',
        name='factr_teleop_franka_right',
        output='screen',
        emulate_tty=True,
        parameters=[
            {"config_file": "franka_right.yaml"}
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
        factr_teleop_franka_right,
        data_record_node,
        zed_node
    ])