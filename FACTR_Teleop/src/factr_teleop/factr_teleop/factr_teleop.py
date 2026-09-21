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

import json
import os
import threading
import time
import yaml
import subprocess
import numpy as np
import pinocchio as pin
from abc import ABC, abstractmethod

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from python_utils.utils import get_workspace_root
from .angles import periodic_joint_error, resolve_null_space_target
from .gain_control import compose_arm_torque, null_space_pd_objective, ramp_gain
from .gripper_feedback import torsional_spring_torque_nm
from .leader_config import apply_control_gains, load_leader_model
from .raw_joint_state import RawJointStateCache
from .dynamixel.driver import DynamixelDriver, DynamixelReadError


def find_ttyusb(port_name):
    """
    This function is used to locate the underlying ttyUSB device.
    """
    base_path = "/dev/serial/by-id/"
    full_path = os.path.join(base_path, port_name)
    if not os.path.exists(full_path):
        raise Exception(f"Port '{port_name}' does not exist in {base_path}.")
    try:
        resolved_path = os.readlink(full_path)
        actual_device = os.path.basename(resolved_path)
        if actual_device.startswith("ttyUSB"):
            print("FOUND DEVICE!")
            return actual_device
        else:
            raise Exception(
                f"The port '{port_name}' does not correspond to a ttyUSB device. It links to {resolved_path}."
            )
    except Exception as e:
        raise Exception(f"Unable to resolve the symbolic link for '{port_name}'. {e}")


class FACTRTeleop(Node, ABC):
    """
    Base class for implementing the FACTR low-cost force-feedback teleoperation system 
    for a follower arm.

    This class implements the control loop for the leader teleoperation arm, including
    features such as gravity compensation, null-space regulation, friction compensation,
    and force-feedback.

    Note that this class should be used as a parent class, where the defined abstract 
    methods must be implemented by subclasses for handling communication between the 
    leader and follower arms, as well as force-feedback for the leader gripper.
    """
    def __init__(self, arm_index: int):
        super().__init__(f'factr_teleop_{arm_index}')

        if arm_index not in [0, 1]:
            self.get_logger().error("please specify arm_index as 0 (left) or 1 (right)")

        if arm_index == 0:
            # left arm
            config_file_name = self.declare_parameter('config_file', 'factr_rizon_left.yaml').get_parameter_value().string_value
        else:
            # right arm
            config_file_name = self.declare_parameter('config_file', 'factr_rizon_right.yaml').get_parameter_value().string_value

        config_path = os.path.join(get_workspace_root(), f"src/factr_teleop/factr_teleop/configs/{config_file_name}")
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
        
        self.name = self.config["name"]
        self.side = "left" if arm_index == 0 else "right"
        self.dt = 1 / self.config["controller"]["frequency"]
        self.num_arm_joints = self.config["arm_teleop"]["num_arm_joints"]
        self.initial_match_joint_pos = np.array(
            self.config["arm_teleop"]["initialization"]["model_home_q_rad"],
            dtype=float,
        )
        self._leader_model = load_leader_model(
            self.side,
            self.num_arm_joints,
            self.initial_match_joint_pos,
        )
        if self._leader_model.leader_urdf is not None:
            self.config["arm_teleop"]["leader_urdf"] = (
                self._leader_model.leader_urdf
            )
        apply_control_gains(
            self.config["controller"], self._leader_model.control_gains
        )
        self._last_commanded_torque_nm = np.zeros(self.num_arm_joints)
        
        self._prepare_dynamixel()
        self._raw_joint_state_cache = RawJointStateCache(self.num_motors)
        self._prepare_inverse_dynamics()

        # Reading from /configs/franka_example.yaml
        # NEED TO modify the urdf file for the FACTR-Rizon setup!
        # leader arm parameters
        self.safety_margin = self.config["arm_teleop"]["arm_joint_limits_safety_margin"]
        self.arm_joint_limits_max = np.array(self.config["arm_teleop"]["arm_joint_limits_max"]) - self.safety_margin
        self.arm_joint_limits_min = np.array(self.config["arm_teleop"]["arm_joint_limits_min"]) + self.safety_margin
        self._motor_joint_signs = self._leader_model.motor_signs(self.gripper_sign)
        self.gripper_spring_open_rad = (
            self._leader_model.gripper_open_raw * self.gripper_sign
        )
        self.gripper_spring_closed_rad = (
            self._leader_model.gripper_closed_raw * self.gripper_sign
        )
        self.calibration_joint_pos = self._leader_model.home_factr_q_rad.copy()
        assert self.num_arm_joints == len(self.arm_joint_limits_max) == len(self.arm_joint_limits_min), \
            "num_arm_joints and the length of arm joint limits must be the same"
        assert self.num_arm_joints == len(self.calibration_joint_pos) == len(self.initial_match_joint_pos), \
            "num_arm_joints and the FACTR model-home vectors must have the same length"
        
        # leader gripper parameters
        self.gripper_limit_min = -10
        self.gripper_limit_max = self.config["gripper_teleop"]["actuation_range"]
        self.gripper_pos_prev = 0.0
        self.gripper_pos = 0.0

        # Gravity compensation has its own configured strength and runtime activation
        # ramp. It never scales force feedback or the controller's other torque terms.
        gravity_comp_config = self.config["controller"]["gravity_comp"]
        self.enable_gravity_comp = gravity_comp_config["enable"]
        self.gravity_comp_modifier = gravity_comp_config["gain"]
        self.tau_g = np.zeros(self.num_arm_joints)

        self.grav_comp_gain_ramp_time = self.config["controller"].get(
            "grav_comp_gain_ramp_time", 1.0
        )
        self.grav_comp_gain = 0.0
        self.grav_comp_gain_target = 0.0
        self._grav_comp_gain_last_update = None
        # The dashboard's gravity button is the leader's master energize gate.
        # Every torque component, including the gripper, sits behind this ramp;
        # once it reaches zero the Dynamixel torque-enable register is cleared.
        self.leader_torque_gain = 0.0
        self.leader_torque_gain_target = 0.0
        self._leader_torque_gain_last_update = None
        self.leader_torque_enabled = False
        self._enable_capture_remaining = 0
        self._enable_capture_samples = []
        # friction comp
        self.stiction_comp_enable_speed = self.config["controller"]["static_friction_comp"]["enable_speed"]
        self.stiction_comp_gain = np.asarray(
            self.config["controller"]["static_friction_comp"]["gain"],
            dtype=float,
        )
        self.stiction_dither_flag = np.ones((self.num_arm_joints), dtype=bool)
        # joint limit barrier:
        # Set controller.joint_limit_barrier.enable: false to switch off the leader-side
        # end-stop repulsion entirely (no follower-range protection from FACTR). Defaults
        # to on if the key is absent.
        self.joint_limit_barrier_enable = self.config["controller"]["joint_limit_barrier"].get("enable", True)
        self.joint_limit_kp = self.config["controller"]["joint_limit_barrier"]["kp"]
        self.joint_limit_kd = self.config["controller"]["joint_limit_barrier"]["kd"]
        # null space regulation
        null_space_config = self.config["controller"]["null_space_regulation"]
        self.null_space_regulation_enable = null_space_config.get("enable", True)
        self.null_space_joint_target = resolve_null_space_target(
            null_space_config, self.initial_match_joint_pos, self.num_arm_joints
        )
        self.null_space_kp = null_space_config["kp"]
        self.null_space_kd = null_space_config["kd"]
        self.null_space_torque_gain = np.asarray(
            null_space_config["torque_gain"], dtype=float
        )
        self.null_space_gain_ramp_time = null_space_config.get("gain_ramp_time", 1.0)
        self.null_space_gain = 0.0
        # Null-space regulation is a collection-scoped feature. Leader torque can
        # be energized for viewing/calibration without enabling this term.
        self.null_space_gain_target = 0.0
        self._null_space_gain_last_update = None
        # torque feedback
        self.enable_torque_feedback = self.config["controller"]["torque_feedback"]["enable"]
        self.torque_feedback_gain = self.config["controller"]["torque_feedback"]["gain"]
        self.torque_feedback_motor_scalar = self.config["controller"]["torque_feedback"]["motor_scalar"]
        self.torque_feedback_damping = self.config["controller"]["torque_feedback"]["damping"]
        self.force_feedback_gain_ramp_time = self.config["controller"].get(
            "force_feedback_gain_ramp_time", self.grav_comp_gain_ramp_time
        )
        self.force_feedback_gain = 0.0
        self.force_feedback_gain_target = 0.0
        self._force_feedback_gain_last_update = None
        # Staleness cutoff and per-joint clip for the follower external-torque feed
        # (see the /factr_force_feedback_<side> subscription below). Optional keys so
        # configs that predate force feedback keep launching.
        self.torque_feedback_timeout = self.config["controller"]["torque_feedback"].get("timeout", 0.25)
        self.torque_feedback_max = self.config["controller"]["torque_feedback"].get("max_torque", 10.0)
        # One-sided virtual torsional spring: zero at the calibrated open trigger,
        # increasingly restorative toward closed, with optional damping and a
        # separate final torque limit. It is local to the leader and does not use
        # follower-gripper force feedback.
        gripper_spring_config = self.config["controller"]["gripper_spring"]
        self.enable_gripper_spring = gripper_spring_config["enable"]
        self.gripper_spring_stiffness = gripper_spring_config[
            "stiffness_nm_per_rad"
        ]
        self.gripper_spring_damping = gripper_spring_config[
            "damping_nm_s_per_rad"
        ]
        self.gripper_spring_max_torque = gripper_spring_config["max_torque_nm"]
        
        # needs to be implemented to establish communication between the leader and the follower
        self.set_up_communication()

        # Gravity activation target, published by the relay's grav-comp endpoints.
        # Its OWN callback group: without it this subscription shares the node's default
        # group with the 500 Hz control-loop timer, which starves it under the
        # MultiThreadedExecutor -- the target gets published but the callback never runs,
        # so the ramp never starts. A separate group lets the executor run it concurrently.
        side = self.side
        self._gain_cb_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            Float64,
            f"/factr_grav_comp_gain_{side}",
            self._on_grav_comp_gain_target,
            10,
            callback_group=self._gain_cb_group,
        )
        self.create_subscription(
            Float64,
            f"/factr_force_feedback_gain_{side}",
            self._on_force_feedback_gain_target,
            10,
            callback_group=self._gain_cb_group,
        )
        self.create_subscription(
            Float64MultiArray,
            f"/factr_null_space_control_{side}",
            self._on_null_space_control,
            10,
            callback_group=self._gain_cb_group,
        )
        self._grav_comp_gain_state_pub = self.create_publisher(
            Float64, f"/factr_grav_comp_gain_state_{side}", 10
        )
        self._force_feedback_gain_state_pub = self.create_publisher(
            Float64, f"/factr_force_feedback_gain_state_{side}", 10
        )
        self._leader_torque_gain_state_pub = self.create_publisher(
            Float64, f"/factr_leader_torque_gain_state_{side}", 10
        )
        self._leader_torque_enabled_state_pub = self.create_publisher(
            Bool, f"/factr_leader_torque_enabled_state_{side}", 10
        )
        self.create_timer(
            0.1, self._publish_gain_states, callback_group=self._gain_cb_group
        )
        self._telemetry_snapshot = {}
        self._telemetry_pub = self.create_publisher(String, f"/factr_telemetry_{side}", 10)
        # Publish the latest diagnostic snapshot at approximately the configured
        # control frequency. This timer is intentionally independent of the
        # control-loop timer: consumers need comparable temporal resolution, not
        # a one-to-one handshake between the two ROS streams.
        self.create_timer(
            self.dt,
            self._publish_telemetry,
            callback_group=self._gain_cb_group,
        )

        # Follower external joint torques for the force-feedback term, pushed by the
        # DFC client through the FACTR API relay (WS force_feedback frames or
        # POST /force_feedback_<side> -> /factr_force_feedback_<side>). Joint-space:
        # JointState.effort holds one torque per arm joint in the follower's joint
        # convention — the same convention every torque this loop composes uses, so it
        # feeds torque_feedback() directly. (A future task-space variant would carry a
        # 6-D TCP wrench instead and map it through the leader Jacobian.) The callback
        # keeps a latest-value cache; get_leader_arm_external_joint_torque() returns
        # zeros once the feed is older than torque_feedback.timeout, so a dropped
        # client or dead relay never leaves a standing force on the leader. Its own
        # callback group for the same starvation reason as the gain subscription.
        self._external_torque = np.zeros(self.num_arm_joints)
        self._external_torque_at = 0.0   # monotonic receive time; 0.0 = never
        self._external_torque_lock = threading.Lock()
        self._force_feedback_cb_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            JointState, f"/factr_force_feedback_{side}", self._on_force_feedback, 10,
            callback_group=self._force_feedback_cb_group,
        )
        # DFC owns calibration; FACTR uses the derived dynamics-model contract.
        self._get_dynamixel_offsets()

        # start the control loop
        # self.dt = 500Hz
        self.timer = self.create_timer(self.dt, self.control_loop_callback)


    def _prepare_dynamixel(self):
        """
        Instantiates driver for interfacing with Dynamixel servos.
        """
        self.servo_types = self.config["dynamixel"]["servo_types"]
        self.num_motors = len(self.servo_types)
        if self.num_motors != self.num_arm_joints + 1:
            raise RuntimeError(
                "left FACTR hardware profile must describe seven arm motors "
                "plus one gripper"
            )
        legacy_signs = self.config["dynamixel"].get("joint_signs", [1.0])
        self.gripper_sign = float(
            self.config["gripper_teleop"].get("hardware_sign", legacy_signs[-1])
        )
        assert self.gripper_sign in (-1.0, 1.0), \
            "gripper_teleop.hardware_sign must be -1 or +1"
        self.dynamixel_port = "/dev/serial/by-id/" + self.config["dynamixel"]["dynamixel_port"]

        # checks of the latency timer on ttyUSB of the corresponding port is 1
        # if it is not 1, the control loop cannot run at above 200 Hz, which will 
        # cause extremely undesirable behaviour for the leader arm. If the latency
        # timer is not 1, one can set it to 1 as follows:
        # echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB{NUM}/latency_timer
        ttyUSBx = find_ttyusb(self.dynamixel_port)
        command = f"cat /sys/bus/usb-serial/devices/{ttyUSBx}/latency_timer"        
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        ttyUSB_latency_timer = int(result.stdout)
        if ttyUSB_latency_timer != 1:
            raise Exception(
                f"Please ensure the latency timer of {ttyUSBx} is 1. Run: \n \
                echo 1 | sudo tee /sys/bus/usb-serial/devices/{ttyUSBx}/latency_timer"
            )

        # DFC owns the physical leader identity and injects its ID range at launch.
        id_start = self._leader_model.dynamixel_id_start
        self._hardware_motor_indices = self._leader_model.hardware_motor_indices()
        joint_ids = id_start + self._hardware_motor_indices
        hardware_servo_types = [
            self.servo_types[index] for index in self._hardware_motor_indices
        ]
        if len(self._hardware_motor_indices) != self.num_motors:
            self.get_logger().warning(
                f"FACTR TELEOP {self.name}: partial leader mode; physical "
                f"joint indices={self._hardware_motor_indices.tolist()} "
                f"servo IDs={joint_ids.tolist()}; absent joints remain at home"
            )
        try:
            self.driver = DynamixelDriver(
                joint_ids,
                hardware_servo_types,
                self.dynamixel_port,
            )
        except FileNotFoundError:
            self.get_logger().info(f"Port {self.dynamixel_port} not found. Please check the connection.")
            return
        self.driver.set_torque_mode(False)
        # Set current-control mode, explicitly latch a zero goal current, then
        # leave the leader de-energized until the dashboard enables it.
        self.driver.set_operating_mode(0)
        self.driver.set_torque_mode(True)
        self.driver.set_torque(np.zeros(len(joint_ids)))
        self.driver.set_torque_mode(False)

    def _prepare_inverse_dynamics(self):
        """
        Creates a model of the leader arm given the its URDF for kinematic and dynamic
        computations used in gravity compensation and null-space regulation calculations.
        """
        self.leader_urdf = os.path.join(
            'src/factr_teleop/factr_teleop/urdf/', 
            self.config["arm_teleop"]["leader_urdf"]
        )
        workspace_root = get_workspace_root()
        urdf_model_path = os.path.join(workspace_root, self.leader_urdf)
        urdf_model_dir = os.path.join(workspace_root, os.path.dirname(urdf_model_path))
        self.pin_model, _, _ = pin.buildModelsFromUrdf(filename=urdf_model_path, package_dirs=urdf_model_dir)
        self.pin_data = self.pin_model.createData()

    def _read_full_joint_state(self, source):
        """Read physical motors and synthesize the normal full leader frame."""
        position, velocity = self.driver.get_positions_and_velocities(source=source)
        return self._leader_model.expand_hardware_state(position, velocity)

    def _get_dynamixel_offsets(self, verbose=True):
        """
        Install the FACTR model offsets derived from DFC's launch contract.

        The current pose is sampled only for startup audit telemetry; it never
        changes calibration.
        """
        # Every acquisition still gets exactly one serial attempt. Startup has
        # no timer yet, so a missed attempt is ignored here and a separate one
        # begins after ``dt`` until enough complete samples arrive.
        startup_deadline = time.monotonic() + 5.0
        successful_samples = 0
        curr_joints = curr_vel = None
        while successful_samples < 11:
            try:
                curr_joints, curr_vel = self._read_full_joint_state("startup")
            except DynamixelReadError as exc:
                if time.monotonic() >= startup_deadline:
                    raise DynamixelReadError(
                        "Dynamixel startup did not produce 11 complete samples "
                        "within 5 seconds"
                    ) from exc
                time.sleep(self.dt)
                continue
            successful_samples += 1
        self._raw_joint_state_cache.update(curr_joints, curr_vel)

        self.joint_offsets = np.concatenate([
            self._leader_model.joint_offsets_rad, [0.0]
        ])
        model_q, _ = self._leader_model.model_state(
            curr_joints[:self.num_arm_joints],
            np.zeros(self.num_arm_joints),
        )
        self._last_raw_arm_q = np.asarray(
            curr_joints[:self.num_arm_joints], dtype=float
        )
        if verbose:
            self.get_logger().info(
                f"FACTR TELEOP {self.name}: DFC-derived model calibration "
                f"joint_offsets={self.joint_offsets[:-1].tolist()} "
                f"model_q={model_q.tolist()}"
            )
        return

    def _match_start_pos(self):
        """
        Waits until the leader arm is manually moved to roughly the same configuration as the
        follower arm before the follower arm starts mirroring the leader arm. 
        """
        curr_pos, _, _, _ = self.get_leader_joint_states()
        while (np.linalg.norm(curr_pos - self.initial_match_joint_pos[0:self.num_arm_joints]) > 5):
            current_joint_error = np.linalg.norm(
                curr_pos - self.initial_match_joint_pos[0:self.num_arm_joints]
            )
            curr_pos_str = ", ".join(f"{x:.3f}" for x in curr_pos)
            self.get_logger().info(
                f"FACTR TELEOP {self.name}: Please match starting joint pos. "
                f"Current error: {current_joint_error:.3f} | current joint pos: [{curr_pos_str}]"
            )
            curr_pos, _, _, _ = self.get_leader_joint_states()
            time.sleep(0.5)
        self.get_logger().info(f"FACTR TELEOP {self.name}: Initial joint position matched.")

    def shut_down(self):
        """
        Disables all torque on the leader arm and gripper during node shutdown.
        """
        if bool(getattr(self.driver, "torque_enabled", True)):
            self.set_leader_joint_torque(np.zeros(self.num_arm_joints), 0.0)
        self.driver.set_torque_mode(False)
        self.leader_torque_enabled = False

    def _set_leader_torque_enabled(self, enabled):
        """Change hardware torque state; the control-loop callback is the sole caller."""
        enabled = bool(enabled)
        if enabled:
            self.driver.set_torque_mode(True)
            self.leader_torque_enabled = True
            return

        errors = []
        if bool(getattr(self.driver, "torque_enabled", True)):
            try:
                physical_count = len(getattr(
                    self, "_hardware_motor_indices", range(self.num_motors)
                ))
                self.driver.set_torque(np.zeros(physical_count))
            except Exception as exc:  # keep attempting the hardware disable
                errors.append(f"zero torque failed: {exc}")
        try:
            self.driver.set_torque_mode(False)
        except Exception as exc:
            errors.append(f"torque disable failed: {exc}")
        self.leader_torque_enabled = bool(
            getattr(self.driver, "torque_enabled", False)
        )
        if errors:
            raise RuntimeError("; ".join(errors))

    def _apply_leader_torque(self, arm_torque, gripper_torque):
        """Write one gated command and finish a zero-gain hardware transition."""
        if self.leader_torque_gain_target > 0.0 and not self.leader_torque_enabled:
            # The previous disable (and startup) left goal current latched at zero.
            self._set_leader_torque_enabled(True)
        if not self.leader_torque_enabled:
            return np.zeros(self.num_arm_joints)

        if self.leader_torque_gain_target <= 0.0 and self.leader_torque_gain <= 0.0:
            # The helper attempts hardware torque-off even if its zero-current
            # write fails, so do not let a regular command block that transition.
            self._set_leader_torque_enabled(False)
            return np.zeros(self.num_arm_joints)
        return self.set_leader_joint_torque(arm_torque, gripper_torque)

    def _disable_leader_torque_after_read_failure(self):
        """Honor an off request even when state acquisition is failing."""
        if (
            self.leader_torque_gain_target <= 0.0
            and self.leader_torque_enabled
        ):
            self.leader_torque_gain = 0.0
            self._set_leader_torque_enabled(False)

    def get_leader_joint_states(self):
        """
        Returns the current joint positions and velocities of the leader arm and gripper,
        aligned with the joint conventions (range and direction) of the follower arm.
        """
        self.gripper_pos_prev = self.gripper_pos
        joint_pos, joint_vel = self._read_full_joint_state("control")
        self._raw_joint_state_cache.update(joint_pos, joint_vel)
        raw_q = np.asarray(joint_pos[0:self.num_arm_joints], dtype=float)
        joint_pos_arm, joint_vel_arm = self._leader_model.model_state(
            raw_q, joint_vel[0:self.num_arm_joints]
        )
        self._last_raw_arm_q = raw_q
        self._last_present_current_raw = self._leader_model.expand_hardware_values(
            self.driver.present_current_raw()
        )[:self.num_arm_joints]
        estimated_motor_torque = self._leader_model.expand_hardware_values(
            self.driver.current_estimated_torque()
        )
        self._last_current_estimated_torque = (
            estimated_motor_torque * self._motor_joint_signs
        )[:self.num_arm_joints]
        self.gripper_pos = (joint_pos[-1] - self.joint_offsets[-1]) * self.gripper_sign
        
        gripper_vel = (self.gripper_pos - self.gripper_pos_prev) / self.dt
        return joint_pos_arm, joint_vel_arm, self.gripper_pos, gripper_vel

    def get_cached_raw_joint_state(self):
        """Return the exact full raw sample used by the latest control iteration."""
        return self._raw_joint_state_cache.snapshot()

        
    
    def set_leader_joint_pos(self, goal_joint_pos, goal_gripper_pos):
        """
        Moves the leader arm and gripper to a specified joint configuration using a PD control loop.
        This method is useful for aligning the leader arm with a desired configuration, such as 
        matching the follower arm's configuration. It interpolates the motion toward the target 
        position and applies torque commands based on a PD controller.

        **Note:** This function is not used by default in the main teleoperation loop. To ensure 
        controller stability, please ensure the latency of Dynamixel servos is minimized such
        that the control loop frequency is at least 200 Hz. Otherwise, the PD controller tuning 
        is unstable for low control frequencies.
        """
        interpolation_step_size = np.ones(7)*self.config["controller"]["interpolation_step_size"]
        kp = self.config["controller"]["joint_position_control"]["kp"]
        kd = self.config["controller"]["joint_position_control"]["kd"]

        curr_pos, curr_vel, curr_gripper_pos, curr_gripper_vel = self.get_leader_joint_states()
        while (np.linalg.norm(curr_pos - goal_joint_pos) > 0.1):
            next_joint_pos_target = np.where(
                np.abs(curr_pos - goal_joint_pos) > interpolation_step_size, 
                curr_pos + interpolation_step_size*np.sign(goal_joint_pos-curr_pos),
                goal_joint_pos,
            )
            torque = -kp*(curr_pos-next_joint_pos_target)-kd*(curr_vel)
            gripper_torque = -kp*(curr_gripper_pos-goal_gripper_pos)-kd*(curr_gripper_vel)
            self.set_leader_joint_torque(torque, gripper_torque)
            curr_pos, curr_vel, curr_gripper_pos, curr_gripper_vel = self.get_leader_joint_states()
    
    def set_leader_joint_torque(self, arm_torque, gripper_torque):
        """
        Applies torque to the leader arm and gripper.
        """
        arm_torque = self._leader_model.scale_joint_torque(arm_torque)
        arm_gripper_torque = np.append(arm_torque, gripper_torque)
        full_motor_torque = arm_gripper_torque * self._motor_joint_signs
        commanded_motor_torque = self.driver.set_torque(
            self._leader_model.select_hardware_values(full_motor_torque)
        )
        commanded_motor_torque = self._leader_model.expand_hardware_values(
            commanded_motor_torque
        )
        return (commanded_motor_torque * self._motor_joint_signs)[
            :self.num_arm_joints
        ]


    def joint_limit_barrier(self, arm_joint_pos, arm_joint_vel, gripper_joint_pos, gripper_joint_vel):
        """
        Computes joint limit repulsive torque to prevent the leader arm and gripper from 
        exceeding the physical joint limits of the follower arm.

        This method implements a simplified control law compared to the one described in 
        Section IX.B of the paper, while achieving the same protective effect. It applies 
        repulsive torques proportional to the distance from the joint limits and the joint
        velocity when limits are approached or exceeded.
        """
        # Disabled from config -> no arm or gripper end-stop torque at all.
        if not self.joint_limit_barrier_enable:
            return np.zeros(self.num_arm_joints), 0.0
        exceed_max_mask = arm_joint_pos > self.arm_joint_limits_max
        tau_l = (-self.joint_limit_kp * (arm_joint_pos - self.arm_joint_limits_max) \
            - self.joint_limit_kd * arm_joint_vel) * exceed_max_mask
        exceed_min_mask = arm_joint_pos < self.arm_joint_limits_min
        tau_l += (-self.joint_limit_kp * (arm_joint_pos - self.arm_joint_limits_min) \
            - self.joint_limit_kd * arm_joint_vel) * exceed_min_mask
        
        # gripper joint-limit barrier DISABLED: no repulsive end-stop torque on the
        # gripper trigger (was: repel above gripper_limit_max / below gripper_limit_min).
        # Required by the gripper-zeroing removal in _get_dynamixel_offsets -- with the
        # offset pinned to 0, gripper_pos is the raw servo angle and would otherwise trip
        # this barrier at boot.
        tau_l_gripper = 0.0
        return tau_l, tau_l_gripper

    def _on_force_feedback(self, msg):
        """Cache one follower external-torque sample (``JointState.effort``, arm joints).

        Wrong-length or non-finite samples are dropped with a throttled warning;
        accepted torques are clipped to ±torque_feedback.max_torque per joint before
        the control loop can see them.
        """
        tau = np.asarray(msg.effort, dtype=float)
        if tau.shape != (self.num_arm_joints,) or not np.all(np.isfinite(tau)):
            self.get_logger().warning(
                f"FACTR TELEOP {self.name}: dropped force-feedback sample "
                f"(shape {tau.shape}, want ({self.num_arm_joints},), finite only)",
                throttle_duration_sec=1.0,
            )
            return
        np.clip(tau, -self.torque_feedback_max, self.torque_feedback_max, out=tau)
        with self._external_torque_lock:
            self._external_torque = tau
            self._external_torque_at = time.monotonic()

    def _on_grav_comp_gain_target(self, msg):
        """Set gravity and the master leader-torque target together."""
        new_target = float(np.clip(msg.data, 0.0, 1.0))
        if new_target > 0.0 and self.grav_comp_gain_target <= 0.0:
            self._enable_capture_remaining = 25
            self._enable_capture_samples = []
        self.grav_comp_gain_target = new_target
        if new_target > 0.0 and self.leader_torque_gain_target <= 0.0:
            # Enabling must begin from a true zero command after a fresh read.
            self.leader_torque_gain = 0.0
            self._leader_torque_gain_last_update = None
        self.leader_torque_gain_target = new_target
        self.get_logger().info(
            f"FACTR TELEOP {self.name}: grav comp gain target -> "
            f"{self.grav_comp_gain_target:.3f}"
        )

    def _on_force_feedback_gain_target(self, msg):
        """Set only the follower-force activation target."""
        self.force_feedback_gain_target = float(np.clip(msg.data, 0.0, 1.0))
        self.get_logger().info(
            f"FACTR TELEOP {self.name}: force feedback gain target -> "
            f"{self.force_feedback_gain_target:.3f}"
        )

    def _on_null_space_control(self, msg):
        """Atomically install a DFC-coordinate target and activation state."""
        command = np.asarray(msg.data, dtype=float)
        if (
            command.shape != (self.num_arm_joints + 1,)
            or not np.all(np.isfinite(command))
            or command[0] not in (0.0, 1.0)
        ):
            self.get_logger().warning(
                f"FACTR TELEOP {self.name}: dropped invalid null-space control "
                f"message with shape {command.shape}",
                throttle_duration_sec=1.0,
            )
            return
        enabled = bool(command[0]) and bool(self.null_space_regulation_enable)
        self.null_space_joint_target = self._leader_model.dfc_position_to_model(
            command[1:]
        )
        self.null_space_gain_target = float(enabled)
        self.get_logger().info(
            f"FACTR TELEOP {self.name}: null-space target updated; "
            f"gain target -> {self.null_space_gain_target:.1f}"
        )

    def _publish_gain_states(self):
        """Publish component gains plus the actual master hardware state."""
        grav_msg = Float64()
        grav_msg.data = float(self.grav_comp_gain)
        self._grav_comp_gain_state_pub.publish(grav_msg)
        feedback_msg = Float64()
        feedback_msg.data = float(self.force_feedback_gain)
        self._force_feedback_gain_state_pub.publish(feedback_msg)
        leader_gain_msg = Float64()
        leader_gain_msg.data = float(self.leader_torque_gain)
        self._leader_torque_gain_state_pub.publish(leader_gain_msg)
        leader_enabled_msg = Bool()
        leader_enabled_msg.data = bool(self.leader_torque_enabled)
        self._leader_torque_enabled_state_pub.publish(leader_enabled_msg)

    def _publish_telemetry(self):
        """Publish the latest control state and zero-I/O diagnostics at native rate."""
        if self._telemetry_snapshot:
            msg = String()
            payload = dict(self._telemetry_snapshot)
            # diagnostics_snapshot() only copies counters/data already obtained by
            # normal reads. It never touches the serial bus.
            payload["dynamixel"] = [self.driver.diagnostics_snapshot()]
            msg.data = json.dumps(payload, separators=(",", ":"))
            self._telemetry_pub.publish(msg)

    def _capture_enable_tick(self, q, dq, tau_limit, tau_null, tau_gravity,
                             tau_friction, tau_feedback, tau_calculated,
                             tau_commanded, grav_gain):
        raw_q = self._last_raw_arm_q
        sample = {
            "stamp_monotonic_ns": time.monotonic_ns(),
            "raw_q_rad": raw_q.tolist(),
            "model_q_rad": q.tolist(),
            "model_dq_rad_s": dq.tolist(),
            "home_error_rad": (q - self.initial_match_joint_pos[:self.num_arm_joints]).tolist(),
            "limit_torque_nm": tau_limit.tolist(),
            "null_torque_nm": tau_null.tolist(),
            "null_space_gain": float(self.null_space_gain),
            "null_space_target_rad": self.null_space_joint_target.tolist(),
            "gravity_torque_nm": tau_gravity.tolist(),
            "friction_torque_nm": tau_friction.tolist(),
            "force_feedback_torque_nm": tau_feedback.tolist(),
            "grav_comp_gain": float(grav_gain),
            "grav_comp_gain_target": float(self.grav_comp_gain_target),
            "friction_gain": self.stiction_comp_gain.tolist(),
            "force_feedback_gain": float(self.force_feedback_gain),
            "force_feedback_gain_target": float(self.force_feedback_gain_target),
            "leader_torque_gain": float(self.leader_torque_gain),
            "leader_torque_gain_target": float(self.leader_torque_gain_target),
            "leader_torque_enabled": bool(self.leader_torque_enabled),
            "joint_offsets_rad": self.joint_offsets[:self.num_arm_joints].tolist(),
            "model_signs": self._leader_model.model_signs.tolist(),
            "calculated_torque_nm": tau_calculated.tolist(),
            "commanded_torque_nm": tau_commanded.tolist(),
            "present_current_raw": self._last_present_current_raw.tolist(),
            "current_estimated_torque_nm": (
                self._last_current_estimated_torque.tolist()
            ),
            # Backward-compatible alias for the historical pre-limit value.
            "applied_torque_nm": tau_calculated.tolist(),
        }
        self._telemetry_snapshot = sample
        if self._enable_capture_remaining <= 0:
            return
        samples = [*self._enable_capture_samples, sample]
        self._enable_capture_samples = samples
        self._enable_capture_remaining -= 1
        if len(samples) == 1:
            self.get_logger().info(
                "FACTR ENABLE CAPTURE " + json.dumps(sample, separators=(",", ":"))
            )

    def _update_component_gains(self):
        """Advance component ramps and the final leader-torque gate."""
        now = time.monotonic()
        self.null_space_gain, self._null_space_gain_last_update = ramp_gain(
            self.null_space_gain,
            self.null_space_gain_target,
            self._null_space_gain_last_update,
            self.null_space_gain_ramp_time,
            now,
        )
        self.grav_comp_gain, self._grav_comp_gain_last_update = ramp_gain(
            self.grav_comp_gain,
            self.grav_comp_gain_target,
            self._grav_comp_gain_last_update,
            self.grav_comp_gain_ramp_time,
            now,
        )
        self.force_feedback_gain, self._force_feedback_gain_last_update = ramp_gain(
            self.force_feedback_gain,
            self.force_feedback_gain_target,
            self._force_feedback_gain_last_update,
            self.force_feedback_gain_ramp_time,
            now,
        )
        self.leader_torque_gain, self._leader_torque_gain_last_update = ramp_gain(
            self.leader_torque_gain,
            self.leader_torque_gain_target,
            self._leader_torque_gain_last_update,
            self.grav_comp_gain_ramp_time,
            now,
        )
        return (
            self.null_space_gain,
            self.grav_comp_gain,
            self.force_feedback_gain,
            self.leader_torque_gain,
        )

    def gravity_compensation(self, arm_joint_pos, arm_joint_vel):
        """
        Computes joint torque for gravity compensation using inverse dynamics.
        This method uses the Recursive Newton-Euler Algorithm (RNEA), provided by the
        Pinocchio library, to calculate the torques required to counteract gravity
        at the current joint states. The result is scaled by the configured
        ``gravity_comp_modifier`` weight.

        The configured gravity strength is applied here; the independent runtime
        activation ramp is applied only to gravity and friction in the control loop.

        This implementation corresponds to the gravity compensation strategy
        described in Section III.C of the paper.
        """
        self.tau_g = pin.rnea(
            self.pin_model, self.pin_data,
            arm_joint_pos, arm_joint_vel, np.zeros_like(arm_joint_vel)
        )
        self.tau_g *= self.gravity_comp_modifier
        return self.tau_g

    def friction_compensation(self, arm_joint_vel):
        """
        Compute joint torques to compensate for static friction during teleoperation.

        This method implements static friction compensation as described in Equation 7,
        Section IX.A of the paper. It omits kinetic friction compensation, which was 
        necessary in earlier hardware versions to achieve smooth teleoperation, but has 
        since become unnecessary due to hardware improvements, such as weight reduction. 
        """
        tau_ss = np.zeros(self.num_arm_joints)
        for i in range(self.num_arm_joints):
            if abs(arm_joint_vel[i]) < self.stiction_comp_enable_speed:
                if self.stiction_dither_flag[i]:
                    tau_ss[i] += self.stiction_comp_gain[i] * abs(self.tau_g[i])
                else:
                    tau_ss[i] -= self.stiction_comp_gain[i] * abs(self.tau_g[i])
                self.stiction_dither_flag[i] = ~self.stiction_dither_flag[i]
        return tau_ss
    
    def null_space_regulation(self, arm_joint_pos, arm_joint_vel):
        """
        Computes joint torques to perform null-space regulation for redundancy resolution 
        of the leader arm.

        This method enables the specification of a desired null-space joint configuration 
        via `self.null_space_joint_target`. It implements the control strategy described 
        in Equation 3 of Section III.B in the paper, projecting a PD control law into 
        the null space of the task Jacobian to achieve secondary objectives without 
        affecting the primary task.
        """
        if not self.null_space_regulation_enable:
            return np.zeros(self.num_arm_joints)
        J = pin.computeJointJacobian(
            self.pin_model, self.pin_data, arm_joint_pos, self.num_arm_joints
        )
        J_dagger = np.linalg.pinv(J)
        null_space_projector = np.eye(self.num_arm_joints) - J_dagger @ J
        # Present Position can jump to an equivalent +/-2*pi encoder branch after
        # a servo reboot.  The Jacobian and gravity model are periodic, but this
        # linear PD error is not; without wrapping it generated ~0.6 Nm of false
        # null-space torque on the right leader while the physical pose was valid.
        q_error = periodic_joint_error(
            arm_joint_pos,
            self.null_space_joint_target[0:self.num_arm_joints],
        )
        pd_objective = null_space_pd_objective(
            q_error,
            arm_joint_vel,
            self.null_space_kp,
            self.null_space_kd,
            self.null_space_torque_gain,
        )
        tau_n = null_space_projector @ pd_objective
        return tau_n
    
    def torque_feedback(self, external_torque, arm_joint_vel):
        """
        Computes joint torque for the leader arm to achieve force-feedback based on
        the external joint torque from the follower arm.

        This method implements Equation 1 in Section III.A of the paper.
        """
        tau_ff = -1.0*self.torque_feedback_gain/self.torque_feedback_motor_scalar * external_torque
        tau_ff -= self.torque_feedback_damping*arm_joint_vel
        return tau_ff

    def control_loop_callback(self):
        """
        Runs the main control loop of the leader arm. 

        Note that while the control loop can run at up to 500 Hz, lower frequencies 
        such as 200 Hz can still yield comparable performance, although they may 
        require additional tuning of control parameters. For Dynamixel servos to 
        support a 500 Hz control frequency, ensure that the Baud Rate is set to 4 Mbps 
        and the Return Delay Time is set to 0 using the Dynamixel Wizard software.
        """
        try:
            leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel = self.get_leader_joint_states()
        except DynamixelReadError as exc:
            # A timer tick is the retry. Never spend a real-time callback making
            # repeated serial attempts; retain the previous torque command and
            # let the next scheduled tick acquire a new state.
            self.get_logger().warning(
                f"FACTR TELEOP {self.name}: skipped control tick after {exc}",
                throttle_duration_sec=1.0,
            )
            self._disable_leader_torque_after_read_failure()
            return

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
        # self.update_communication(leader_arm_pos, leader_gripper_pos)


    @abstractmethod
    def set_up_communication(self):
        """
        This method should be implemented to set up communication between the leader arm
        and the follower arm for bilateral teleoperation. This method is called once
        in the __init__ method.
        
        For example, a subscriber can  be set up to receive external joint torque from 
        the leader arm and a publisher can be set up to send joint position target commands 
        to the follower arm. Publishers and subscribers can also be set up to record
        the follower arm's joint states

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        pass


    def get_leader_arm_external_joint_torque(self):
        """
        Returns the current external joint torque of the follower arm, used to compute
        force-feedback in the leader arm. Called at every iteration of the control loop
        if self.enable_torque_feedback is set to True.

        The base implementation reads the latest sample cached from the
        /factr_force_feedback_<side> subscription (follower torques pushed by the DFC
        client through the FACTR API relay) and returns zeros once that feed is older
        than torque_feedback.timeout, so feedback dies with the connection instead of
        freezing at the last value. Subclasses with their own follower link may
        still override this.

        Returns:
            np.ndarray: A NumPy array of shape (num_arm_joints,) containing the external
            joint torques.
        """
        with self._external_torque_lock:
            tau = self._external_torque
            received_at = self._external_torque_at
        if time.monotonic() - received_at > self.torque_feedback_timeout:
            return np.zeros(self.num_arm_joints)
        return tau


    def gripper_spring_torque(self, leader_gripper_pos, leader_gripper_vel):
        """Return the calibrated one-sided trigger spring torque [Nm]."""
        return torsional_spring_torque_nm(
            leader_gripper_pos,
            leader_gripper_vel,
            self.gripper_spring_open_rad,
            self.gripper_spring_closed_rad,
            self.gripper_spring_stiffness,
            self.gripper_spring_damping,
            self.gripper_spring_max_torque,
        )


    @abstractmethod
    def update_communication(self, leader_arm_pos, leader_gripper_pos):
        """
        This method is intended to be called at every iteration of the control loop to transmit 
        relevant data, such as joint position targets, from the leader to the follower arm.

        Args:
            leader_arm_pos (np.ndarray): A NumPy array containing the joint positions of the leader arm.
            leader_gripper_pos (np.ndarray): A NumPy array containing the position of the leader gripper.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        pass
