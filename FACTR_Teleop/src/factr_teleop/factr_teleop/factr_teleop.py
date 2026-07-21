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
import time
import yaml
import subprocess
import numpy as np
import pinocchio as pin
from abc import ABC, abstractmethod

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import Float64, String
from python_utils.utils import get_workspace_root
from factr_teleop.dynamixel.driver import DynamixelDriver


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
        self.dt = 1 / self.config["controller"]["frequency"]
        
        self._prepare_dynamixel()
        # Fixed motor/model directions belong to the FACTR mechanism and URDF.  They are
        # not DFC follower-convention sign flips and must never be overwritten by a
        # calibration push.
        self._model_joint_signs = self.joint_signs.copy()
        self._prepare_inverse_dynamics()

        # Reading from /configs/franka_example.yaml
        # NEED TO modify the urdf file for the FACTR-Rizon setup!
        # leader arm parameters
        self.num_arm_joints = self.config["arm_teleop"]["num_arm_joints"]
        self.safety_margin = self.config["arm_teleop"]["arm_joint_limits_safety_margin"]
        self.arm_joint_limits_max = np.array(self.config["arm_teleop"]["arm_joint_limits_max"]) - self.safety_margin
        self.arm_joint_limits_min = np.array(self.config["arm_teleop"]["arm_joint_limits_min"]) + self.safety_margin
        self.calibration_joint_pos = np.array(self.config["arm_teleop"]["initialization"]["calibration_joint_pos"])
        self.initial_match_joint_pos = np.array(self.config["arm_teleop"]["initialization"]["initial_match_joint_pos"])
        assert self.num_arm_joints == len(self.arm_joint_limits_max) == len(self.arm_joint_limits_min), \
            "num_arm_joints and the length of arm joint limits must be the same"
        assert self.num_arm_joints == len(self.calibration_joint_pos) == len(self.initial_match_joint_pos), \
            "num_arm_joints and the length of calibration_joint_pos and initial_match_joint_pos must be the same"
        
        # leader gripper parameters
        self.gripper_limit_min = -10
        self.gripper_limit_max = self.config["gripper_teleop"]["actuation_range"]
        self.gripper_pos_prev = 0.0
        self.gripper_pos = 0.0

        # gravity comp (per-component weight only; the master gain below ramps it in)
        gravity_comp_config = self.config["controller"]["gravity_comp"]
        self.enable_gravity_comp = gravity_comp_config["enable"]
        self.gravity_comp_modifier = gravity_comp_config["gain"]
        self.tau_g = np.zeros(self.num_arm_joints)

        # master output gain -- a single [0, 1] multiplier applied to EVERY force this
        # controller commands (gravity comp, friction comp, null-space regulation, the
        # joint-limit barrier, force feedback). It starts at 0 so the arm boots energized
        # but limp (zero commanded torque) and ramps to a target the FACTR API relay's
        # POST /{enable,disable}_grav_comp_<side> triggers publish on /factr_gain_<side>.
        # The ramp advances at 1/force_gain_ramp_time per second; see _update_force_gain().
        self.force_gain_ramp_time = self.config["controller"].get("force_gain_ramp_time", 1.0)
        self.force_gain = 0.0
        self.force_gain_target = 0.0
        self._force_gain_last_update = None   # seeded on the first _update_force_gain() tick
        self._enable_capture_remaining = 0
        self._enable_capture_samples = []
        # friction comp
        self.stiction_comp_enable_speed = self.config["controller"]["static_friction_comp"]["enable_speed"]
        self.stiction_comp_gain = self.config["controller"]["static_friction_comp"]["gain"]
        self.stiction_dither_flag = np.ones((self.num_arm_joints), dtype=bool)
        # joint limit barrier:
        # Set controller.joint_limit_barrier.enable: false to switch off the leader-side
        # end-stop repulsion entirely (no follower-range protection from FACTR). Defaults
        # to on if the key is absent.
        self.joint_limit_barrier_enable = self.config["controller"]["joint_limit_barrier"].get("enable", True)
        self.joint_limit_kp = self.config["controller"]["joint_limit_barrier"]["kp"]
        self.joint_limit_kd = self.config["controller"]["joint_limit_barrier"]["kd"]
        # null space regulation
        self.null_space_joint_target = np.array(self.config["controller"]["null_space_regulation"]["null_space_joint_target"])
        self.null_space_kp = self.config["controller"]["null_space_regulation"]["kp"]
        self.null_space_kd = self.config["controller"]["null_space_regulation"]["kd"]
        # torque feedback
        self.enable_torque_feedback = self.config["controller"]["torque_feedback"]["enable"]
        self.torque_feedback_gain = self.config["controller"]["torque_feedback"]["gain"]
        self.torque_feedback_motor_scalar = self.config["controller"]["torque_feedback"]["motor_scalar"]
        self.torque_feedback_damping = self.config["controller"]["torque_feedback"]["damping"]
        # gripper feedback
        self.enable_gripper_feedback = self.config["controller"]["gripper_feedback"]["enable"]
        
        # needs to be implemented to establish communication between the leader and the follower
        self.set_up_communication()

        # master output-gain target, published over ROS by the FACTR API relay's
        # enable/disable_grav_comp_<side> triggers; the loop ramps self.force_gain toward it.
        # Its OWN callback group: without it this subscription shares the node's default
        # group with the 500 Hz control-loop timer, which starves it under the
        # MultiThreadedExecutor -- the target gets published but the callback never runs,
        # so the ramp never starts. A separate group lets the executor run it concurrently.
        side = "left" if arm_index == 0 else "right"
        self.side = side
        self._gain_cb_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            Float64, f"/factr_gain_{side}", self._on_force_gain_target, 10,
            callback_group=self._gain_cb_group,
        )
        # Publish the LIVE master gain so the FACTR API can serve it (GET /status_<side>)
        # and the dashboard can gate collection behind grav comp. Low-rate timer in the
        # gain group so it never contends with the 500 Hz control loop.
        self._gain_state_pub = self.create_publisher(Float64, f"/factr_gain_state_{side}", 10)
        self.create_timer(0.1, self._publish_gain_state, callback_group=self._gain_cb_group)
        self._calibration_snapshot = {}
        self._diagnostics_pub = self.create_publisher(String, f"/factr_diagnostics_{side}", 10)
        self.create_timer(0.5, self._publish_diagnostics, callback_group=self._gain_cb_group)

        # FACTR owns gravity-model calibration, using its configured physical home pose.
        self._get_dynamixel_offsets()

        # --- servo health diagnostics (throttled; for torque-dropout debugging) ---
        # Reads temperature / present current / hardware-error per servo at low rate and
        # logs it, with a loud warning if any servo latches a hardware error (e.g. the
        # base joint OVERLOAD/OVERHEAT that silently kills its gravity comp). This is
        # SEPARATE from the subclass's joint-position readout log. See
        # DynamixelDriver.read_health(). Set period to 0 to disable.
        self.health_log_period = self.config["controller"].get("health_log_period", 1.0)
        self._last_health_log = time.time()

        # start the control loop
        # self.dt = 500Hz
        self.timer = self.create_timer(self.dt, self.control_loop_callback)


    def _prepare_dynamixel(self):
        """
        Instantiates driver for interfacing with Dynamixel servos.
        """
        self.servo_types = self.config["dynamixel"]["servo_types"]
        self.num_motors = len(self.servo_types)
        self.joint_signs = np.array(self.config["dynamixel"]["joint_signs"], dtype=float)
        assert self.num_motors == len(self.joint_signs), \
            "The number of motors and the number of joint signs must be the same"
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

        # Physical Dynamixel IDs can be either 1-8 or 9-16. Select via the
        # optional "id_start" config field (defaults to 1 for the 1-8 range).
        id_start = self.config["dynamixel"].get("id_start", 1)
        if id_start not in (1, 9):
            raise Exception(f"dynamixel id_start must be 1 or 9, got {id_start}")
        joint_ids = np.arange(self.num_motors) + id_start
        try:
            self.driver = DynamixelDriver(
                joint_ids, self.servo_types, self.dynamixel_port
            )
        except FileNotFoundError:
            self.get_logger().info(f"Port {self.dynamixel_port} not found. Please check the connection.")
            return
        self.driver.set_torque_mode(False)
        # set operating mode to current mode
        self.driver.set_operating_mode(0)
        # enable torque
        self.driver.set_torque_mode(True)

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

    def _get_dynamixel_offsets(self, verbose=True):
        """
        Establishes the offset between each Dynamixel servo's raw reading and the follower
        joint convention, so the leader joint angles fed to gravity comp match the follower.

        Each arm-joint offset is a fixed multiple of pi/2 set by the servo-horn mounting, so
        it is identical on every launch. It is therefore captured ONCE and pinned in the
        config under ``arm_teleop.initialization.joint_offsets`` (radians, one per arm joint).
        When that key is present it is used directly -- gravity comp is correct IMMEDIATELY
        at launch with NO need to pose the leader at calibration_joint_pos first.

        The key is mandatory. The current launch pose is never used to derive calibration;
        a missing saved measurement is a fatal configuration error.

        NOTE: these offsets affect ONLY the internal control-loop / grav-comp joint angles;
        the leader stream published to the follower is the raw servo reading (the DFC side
        applies its own offset convention to that), so pinning them here is safe.
        """
        # Warm up the serial reads before measuring the configured calibration pose.
        for _ in range(10):
            self.driver.get_positions_and_velocities()
        curr_joints, _ = self.driver.get_positions_and_velocities()

        # With saved arm calibration, do not infer any arm offset from the launch pose.
        # The gripper remains a raw signed servo angle in this path.
        saved = self.config["arm_teleop"]["initialization"].get("joint_offsets", None)
        if saved is not None:
            arm_offsets = np.asarray(saved, dtype=float)
            assert arm_offsets.shape[0] == self.num_arm_joints, \
                "arm_teleop.initialization.joint_offsets length must equal num_arm_joints"
            self.joint_offsets = np.concatenate([arm_offsets, [0.0]])
            self._set_calibration_snapshot(curr_joints)
            self._validate_dfc_factr_transform()
            if verbose:
                offsets_str = ", ".join(f"{x:.3f}" for x in self.joint_offsets)
                self.get_logger().info(
                    f"FACTR TELEOP {self.name}: using saved joint_offsets, no pose needed: "
                    f"[{offsets_str}]"
                )
            return

        raise RuntimeError(
            f"FACTR TELEOP {self.name}: missing saved joint_offsets; "
            "launch-pose calibration is forbidden"
        )

        def _get_error(calibration_joint_pos, offset, index, joint_state):
            joint_sign_i = self.joint_signs[index]
            joint_i = joint_sign_i * (joint_state[index] - offset)
            start_i = calibration_joint_pos[index]
            return np.abs(joint_i - start_i)

        # -- fallback: no saved offsets -> compute from the current (posed) arm --
        arm_offsets = []
        for i in range(self.num_arm_joints):
            best_offset = 0
            best_error = 1e9
            # intervals of pi/2
            for offset in np.linspace(-20 * np.pi, 20 * np.pi, 20 * 4 + 1):
                error = _get_error(self.calibration_joint_pos, offset, i, curr_joints)
                if error < best_error:
                    best_error = error
                    best_offset = offset
            arm_offsets.append(best_offset)

        # Friday behavior: zero the gripper at the same calibration pose/read.
        self.joint_offsets = np.asarray(arm_offsets + [float(curr_joints[-1])])
        self._set_calibration_snapshot(curr_joints)
        self._validate_dfc_factr_transform()
        if verbose:
            offsets_str = ", ".join(f"{x:.3f}" for x in self.joint_offsets)
            offsets_pi = ", ".join(
                f"{int(np.round(x/(np.pi/2)))}*np.pi/2" for x in self.joint_offsets
            )
            self.get_logger().info(f"FACTR TELEOP {self.name}: best offsets: [{offsets_str}]")
            self.get_logger().info(f"FACTR TELEOP {self.name}: best offsets (pi): [{offsets_pi}]")
            # Paste-ready line so the operator can pin these and skip posing next launch.
            save_str = ", ".join(f"{x:.4f}" for x in arm_offsets)
            self.get_logger().warn(
                f"FACTR TELEOP {self.name}: no saved joint_offsets in config -- computed from "
                f"the current pose. To skip posing on future launches, add under "
                f"arm_teleop.initialization:  joint_offsets: [{save_str}]"
            )

    def _set_calibration_snapshot(self, curr_joints):
        init = self.config["arm_teleop"]["initialization"]
        model_q = (
            curr_joints[:self.num_arm_joints]
            - self.joint_offsets[:self.num_arm_joints]
        ) * self.joint_signs[:self.num_arm_joints]
        self._calibration_snapshot = {
            "side": self.side,
            "captured_monotonic_ns": time.monotonic_ns(),
            "raw_q_rad": curr_joints[:self.num_arm_joints].tolist(),
            "configured_home_q_rad": self.calibration_joint_pos.tolist(),
            "joint_offsets_rad": self.joint_offsets[:self.num_arm_joints].tolist(),
            "model_signs": self.joint_signs[:self.num_arm_joints].tolist(),
            "model_q_rad": model_q.tolist(),
            "dfc_home_q_rad": list(init.get("dfc_home_joint_pos", [])),
            "dfc_to_factr_signs": list(init.get("dfc_to_factr_signs", [])),
            "dfc_to_factr_offset_rad": list(init.get("dfc_to_factr_offset_rad", [])),
            "dfc_raw_offsets_deg": list(init.get("dfc_raw_offsets_deg", [])),
            "dfc_sign_flip_joints": list(init.get("dfc_sign_flip_joints", [])),
            "dfc_wrap_deg": init.get("dfc_wrap_deg"),
            "dfc_drop_trailing": init.get("dfc_drop_trailing"),
            "dfc_gripper_open": init.get("dfc_gripper_open"),
            "dfc_gripper_closed": init.get("dfc_gripper_closed"),
        }

    def _validate_dfc_factr_transform(self):
        init = self.config["arm_teleop"]["initialization"]
        dfc_home = np.asarray(init["dfc_home_joint_pos"], dtype=float)
        signs = np.asarray(init["dfc_to_factr_signs"], dtype=float)
        bias = np.asarray(init["dfc_to_factr_offset_rad"], dtype=float)
        if any(x.shape != (self.num_arm_joints,) for x in (dfc_home, signs, bias)):
            raise RuntimeError("DFC/FACTR transform vectors must have seven joints")
        if not np.all(np.isin(signs, (-1.0, 1.0))):
            raise RuntimeError("DFC/FACTR transform signs must be -1 or +1")
        mapped = signs * dfc_home + bias
        if not np.allclose(mapped, self.calibration_joint_pos, atol=1e-6):
            raise RuntimeError(f"DFC home does not map to FACTR home: {mapped.tolist()}")
        raw_offsets_deg = np.asarray(init["dfc_raw_offsets_deg"], dtype=float)
        flips = [int(i) for i in init["dfc_sign_flip_joints"]]
        drop = int(init["dfc_drop_trailing"])
        gripper_open = float(init["dfc_gripper_open"])
        gripper_closed = float(init["dfc_gripper_closed"])
        if raw_offsets_deg.shape != (self.num_arm_joints,) or drop != 1:
            raise RuntimeError("raw FACTR/DFC convention must contain seven offsets and drop one field")
        if len(set(flips)) != len(flips) or any(i < 0 or i >= self.num_arm_joints for i in flips):
            raise RuntimeError(f"invalid DFC sign-flip indices: {flips}")
        if not np.all(np.isfinite(raw_offsets_deg)) or not np.isfinite([gripper_open, gripper_closed]).all() or gripper_open == gripper_closed:
            raise RuntimeError("raw FACTR/DFC convention contains invalid values")
        raw_home = self.joint_offsets[:self.num_arm_joints] + self.calibration_joint_pos / self.joint_signs[:self.num_arm_joints]
        raw_home_deg = np.degrees(raw_home) + raw_offsets_deg
        raw_home_deg[flips] *= -1.0
        if bool(init["dfc_wrap_deg"]):
            raw_home_deg = (raw_home_deg + 180.0) % 360.0 - 180.0
        raw_mapped_home = np.radians(raw_home_deg)
        if not np.allclose(raw_mapped_home, dfc_home, atol=1e-6):
            raise RuntimeError(f"raw FACTR home does not map to DFC home: {raw_mapped_home.tolist()}")
        self.get_logger().info(
            f"FACTR {self.side} convention: dfc_home={dfc_home.tolist()} -> "
            f"factr_home={mapped.tolist()} raw_home={raw_home.tolist()} "
            f"raw_to_dfc_home={raw_mapped_home.tolist()} signs={signs.tolist()} bias={bias.tolist()}"
        )
    
    def _on_calibration(self, msg):
        """
        Apply a gravity-comp calibration pushed from DFC (the single source of truth).

        ``msg.data`` is DFC's ``arms.<side>.convention.offsets_deg`` -- the per-arm-joint
        offset (degrees) mapping this leader's raw servo reading into the follower joint
        convention. The offsets identify the same physical joint zeros, but DFC's sign
        flips belong only to the follower-command convention. Gravity compensation keeps
        FACTR's native mechanism/model signs and applies the fixed URDF model bias:

            joint_offset[i] = -radians(offsets_deg[i])
                              - model_sign[i] * model_bias[i]

        Rebinds self.joint_offsets to a fresh array (never mutates in place) so the 500 Hz
        control loop reading it concurrently always sees a complete old-or-new array. The
        gripper offset stays pinned to 0.0.
        """
        data = np.asarray(msg.data, dtype=float)
        if data.shape[0] != 2 * self.num_arm_joints:
            self.get_logger().warn(
                f"FACTR TELEOP {self.name}: ignoring calibration push of length "
                f"{data.shape[0]} (expected {2 * self.num_arm_joints})"
            )
            return
        offsets_deg = data[:self.num_arm_joints]
        signs = data[self.num_arm_joints:]
        if not np.all(np.isin(signs, (-1.0, 1.0))):
            self.get_logger().warn(f"FACTR TELEOP {self.name}: invalid DFC joint signs")
            return
        model_bias = np.asarray(
            self.config["arm_teleop"]["initialization"].get(
                "model_joint_bias", [0.0] * self.num_arm_joints
            ),
            dtype=float,
        )
        if model_bias.shape != (self.num_arm_joints,) or not np.all(np.isfinite(model_bias)):
            self.get_logger().error(
                f"FACTR TELEOP {self.name}: invalid model_joint_bias; refusing calibration"
            )
            return
        # DFC's offsets provide the physical zero: raw + radians(offset) == 0 at
        # DFC home. FACTR's inverse-dynamics coordinates use the mechanism's native
        # model signs plus a fixed URDF bias. DFC sign flips affect follower commands
        # only; copying them here corrupts the dynamics model (the left arm exposes
        # this because DFC flips joint 6 while the FACTR mechanism does not).
        model_signs = self._model_joint_signs[:self.num_arm_joints]
        arm_offsets = -np.radians(offsets_deg) - model_signs * model_bias
        self.joint_signs = self._model_joint_signs.copy()
        self.joint_offsets = np.concatenate([arm_offsets, [0.0]])
        self._dfc_calibration_received = True
        # Full conversion audit.  `model_at_dfc_zero` evaluates the downstream
        # controller coordinates at raw=-radians(DFC offsets), i.e. the physical pose
        # that DFC calls all-zero.  `offsets_pi_over_2` is directly comparable with the
        # Friday pose-calibration log's "best offsets (pi)" output.
        dfc_zero_raw = -np.radians(offsets_deg)
        model_at_dfc_zero = model_signs * (dfc_zero_raw - arm_offsets)
        offsets_pi_over_2 = arm_offsets / (np.pi / 2.0)
        self.get_logger().info(
            f"FACTR CALIBRATION AUDIT {self.name}: "
            f"dfc_offsets_deg={offsets_deg.tolist()} "
            f"dfc_follower_signs={signs.tolist()} "
            f"model_signs={model_signs.tolist()} "
            f"model_bias_rad={model_bias.tolist()} "
            f"joint_offsets_rad={arm_offsets.tolist()} "
            f"offsets_pi_over_2={offsets_pi_over_2.tolist()} "
            f"model_at_dfc_zero_rad={model_at_dfc_zero.tolist()}"
        )
        offsets_str = ", ".join(f"{x:.3f}" for x in self.joint_offsets)
        self.get_logger().info(
            f"FACTR TELEOP {self.name}: applied DFC calibration passthrough -> "
            f"joint_offsets [{offsets_str}]"
        )

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
        self.set_leader_joint_torque(np.zeros(self.num_arm_joints), 0.0)
        self.driver.set_torque_mode(False)

    def get_leader_joint_states(self):
        """
        Returns the current joint positions and velocities of the leader arm and gripper,
        aligned with the joint conventions (range and direction) of the follower arm.
        """
        self.gripper_pos_prev = self.gripper_pos
        joint_pos, joint_vel = self.driver.get_positions_and_velocities()
        joint_pos_arm = (
            joint_pos[0:self.num_arm_joints] - self.joint_offsets[0:self.num_arm_joints]
        ) * self.joint_signs[0:self.num_arm_joints]
        self.gripper_pos = (joint_pos[-1] - self.joint_offsets[-1]) * self.joint_signs[-1]
        joint_vel_arm = joint_vel[0:self.num_arm_joints] * self.joint_signs[0:self.num_arm_joints]
        
        gripper_vel = (self.gripper_pos - self.gripper_pos_prev) / self.dt
        return joint_pos_arm, joint_vel_arm, self.gripper_pos, gripper_vel

        
    
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
        arm_gripper_torque = np.append(arm_torque, gripper_torque)
        self.driver.set_torque(arm_gripper_torque*self.joint_signs)


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

    def _on_force_gain_target(self, msg):
        """Set the master output-gain target (clamped to [0, 1]); the loop ramps to it."""
        new_target = float(np.clip(msg.data, 0.0, 1.0))
        if new_target > 0.0 and self.force_gain_target <= 0.0:
            self._enable_capture_remaining = 25
            self._enable_capture_samples = []
        self.force_gain_target = new_target
        self.get_logger().info(
            f"FACTR TELEOP {self.name}: force gain target -> {self.force_gain_target:.3f}"
        )

    def _publish_gain_state(self):
        """Publish the live master output gain so the FACTR API can serve it as status."""
        msg = Float64()
        msg.data = float(self.force_gain)
        self._gain_state_pub.publish(msg)

    def _publish_diagnostics(self):
        """Republish the immutable startup-calibration snapshot at low rate."""
        if self._calibration_snapshot:
            msg = String()
            payload = dict(self._calibration_snapshot)
            payload["enable_samples"] = self._enable_capture_samples
            msg.data = json.dumps(payload, separators=(",", ":"))
            self._diagnostics_pub.publish(msg)

    def _capture_enable_tick(self, q, dq, tau_limit, tau_null, tau_gravity,
                             tau_friction, tau_total, gain):
        if self._enable_capture_remaining <= 0:
            return
        signs = self.joint_signs[:self.num_arm_joints]
        offsets = self.joint_offsets[:self.num_arm_joints]
        raw_q = offsets + q / signs
        sample = {
            "stamp_monotonic_ns": time.monotonic_ns(),
            "raw_q_rad": raw_q.tolist(),
            "model_q_rad": q.tolist(),
            "model_dq_rad_s": dq.tolist(),
            "home_error_rad": (q - self.initial_match_joint_pos[:self.num_arm_joints]).tolist(),
            "limit_torque_nm": tau_limit.tolist(),
            "null_torque_nm": tau_null.tolist(),
            "gravity_torque_nm": tau_gravity.tolist(),
            "friction_torque_nm": tau_friction.tolist(),
            "total_torque_pre_gain_nm": tau_total.tolist(),
            "force_gain": float(gain),
            "applied_torque_nm": (tau_total * gain).tolist(),
        }
        samples = [*self._enable_capture_samples, sample]
        self._enable_capture_samples = samples
        self._enable_capture_remaining -= 1
        if len(samples) == 1:
            self.get_logger().info(
                "FACTR ENABLE CAPTURE " + json.dumps(sample, separators=(",", ":"))
            )

    def _update_force_gain(self):
        """Advance the master output gain toward its target and return it.

        Moves at a constant 1/force_gain_ramp_time per second (so a full 0->1 enable takes
        force_gain_ramp_time seconds). Uses the MONOTONIC clock so a wall-clock step (e.g.
        NTP) can never make the elapsed time negative and invert/amplify the step. The first
        call only seeds the baseline, so the long blocking calibration before the control
        loop starts can't turn into a one-tick jump. force_gain is clamped to [0, 1] as a
        hard invariant -- it multiplies every force the loop commands. Ramps up and down.
        """
        now = time.monotonic()
        if self._force_gain_last_update is None:
            self._force_gain_last_update = now
            return self.force_gain
        elapsed = now - self._force_gain_last_update
        self._force_gain_last_update = now
        if self.force_gain_ramp_time <= 0.0:
            self.force_gain = self.force_gain_target
        else:
            step = elapsed / self.force_gain_ramp_time
            if self.force_gain < self.force_gain_target:
                self.force_gain = min(self.force_gain_target, self.force_gain + step)
            elif self.force_gain > self.force_gain_target:
                self.force_gain = max(self.force_gain_target, self.force_gain - step)
        self.force_gain = min(1.0, max(0.0, self.force_gain))
        return self.force_gain

    def gravity_compensation(self, arm_joint_pos, arm_joint_vel):
        """
        Computes joint torque for gravity compensation using inverse dynamics.
        This method uses the Recursive Newton-Euler Algorithm (RNEA), provided by the
        Pinocchio library, to calculate the torques required to counteract gravity
        at the current joint states. The result is scaled by the configured
        ``gravity_comp_modifier`` weight.

        The overall fade-in is handled by the master output gain applied in the control
        loop (see _update_force_gain()), which scales this and every other force term
        together; ``self.tau_g`` is stored unscaled by that master gain so
        friction_compensation() tracks the raw gravity-comp magnitude.

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
                    tau_ss[i] += self.stiction_comp_gain * abs(self.tau_g[i])
                else:
                    tau_ss[i] -= self.stiction_comp_gain * abs(self.tau_g[i])
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
        J = pin.computeJointJacobian(
            self.pin_model, self.pin_data, arm_joint_pos, self.num_arm_joints
        )
        J_dagger = np.linalg.pinv(J)
        null_space_projector = np.eye(self.num_arm_joints) - J_dagger @ J
        q_error = arm_joint_pos - self.null_space_joint_target[0:self.num_arm_joints]
        tau_n = null_space_projector @ (-self.null_space_kp*q_error-self.null_space_kd*arm_joint_vel)
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
        torque_arm = torque_l + torque_null + torque_gravity + torque_friction
        
        # if self.enable_torque_feedback:
        #     print(self.get_leader_arm_external_joint_torque())
        #     external_joint_torque = self.get_leader_arm_external_joint_torque()
        #     torque_arm += self.torque_feedback(external_joint_torque, leader_arm_vel)
        
        # if self.enable_gripper_feedback:
        #     gripper_feedback = self.get_leader_gripper_feedback()
        #     torque_gripper += self.gripper_feedback(leader_gripper_pos, leader_gripper_vel, gripper_feedback)

        # Master output gain (ramped 0->1 on enable) scales EVERY force term at once.
        gain = self._update_force_gain()
        self._capture_enable_tick(
            leader_arm_pos, leader_arm_vel, torque_l, torque_null,
            torque_gravity, torque_friction, torque_arm, gain,
        )
        self.set_leader_joint_torque(torque_arm * gain, torque_gripper * gain)
        # self.update_communication(leader_arm_pos, leader_gripper_pos)

        self._log_servo_health()

    def _log_servo_health(self):
        """Throttled (~``health_log_period`` s) servo health read + log.

        Diagnoses torque dropouts such as the base joint silently losing gravity comp:
        watch ``T`` (temperature) climb over the minute and ``I`` (present current) sit
        pegged near its limit, and get an explicit ERROR the moment a servo latches a
        Hardware Error (OVERLOAD/OVERHEAT/...) or disables its own torque. On such a
        latch the servo stays limp until a reboot/power-cycle.

        This is independent of, and does NOT replace, the subclass joint-position log.

        NOTE: read_health() issues per-servo register reads, so this briefly stalls the
        control loop (the servos hold their last goal current across the gap, so gravity
        comp does not drop out). Rate-limited for that reason; set health_log_period=0
        (or config controller.health_log_period: 0) to disable.
        """
        if not self.health_log_period:
            return
        now = time.time()
        if now - self._last_health_log < self.health_log_period:
            return
        self._last_health_log = now

        try:
            health = self.driver.read_health()
        except Exception as e:
            self.get_logger().warning(f"[health] read failed: {e}")
            return

        lines = ["[health] err(70) pwm(124) I(126) vel(128) pos(132) postraj(140) Vin(144) T(146):"]
        for h in health:
            vin = f"{h['input_voltage'] / 10:.1f}" if h["input_voltage"] is not None else "?"
            lines.append(
                f"  id{h['id']} err={h['hw_error_flags'] or 'OK'} te={h['torque_enable']}"
                f" pwm={h['present_pwm']} I={h['present_current']}/{h['current_limit']}"
                f" vel={h['present_velocity']} pos={h['present_position']}"
                f" postraj={h['position_trajectory']} Vin={vin}V T={h['temperature']}C"
            )
        self.get_logger().info("\n".join(lines))

        for h in health:
            if h["hw_error"]:
                self.get_logger().error(
                    f"[health] servo id{h['id']} HARDWARE ERROR {h['hw_error_flags']} "
                    f"(0x{h['hw_error']:02x}) T={h['temperature']}C I={h['present_current']} "
                    f"-- torque has latched OFF; reboot/power-cycle the servo to clear"
                )
            elif h["torque_enable"] == 0 and self.driver.torque_enabled:
                self.get_logger().error(
                    f"[health] servo id{h['id']} disabled its own torque while the node still "
                    f"commands torque ON (likely protective shutdown) T={h['temperature']}C "
                    f"I={h['present_current']}"
                )


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


    @abstractmethod
    def get_leader_arm_external_joint_torque(self):
        """
        This method should retrieve the current external joint torque from the follower arm.
        This is used to compute force-feedback in the leader arm. This method is called at
        every iteration of the control loop if self.enable_torque_feedback is set to True.

        Returns:
            np.ndarray: A NumPy array of shape (num_arm_joints,) containing the external 
            joint torques. 

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        pass


    @abstractmethod
    def get_leader_gripper_feedback(self):
        """
        This method should retrieve any data from the follower gripper that might be required
        to achieve force-feedback in the leader gripper. For example, this method can be used
        to get the current position of the follower gripper for position-position force-feedback
        or the current force of the follower gripper for position-force force-feedback in the
        leader gripper. This method is called at every iteration of the control loop if 
        self.enable_gripper_feedback is set to True.

        Returns:
            Any: Feedback data required by the leader gripper. This can be a NumPy array, a 
            scalar, or any other data type depending on the implementation.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        pass


    @abstractmethod
    def gripper_feedback(self, leader_gripper_pos, leader_gripper_vel, gripper_feedback):
        """
        Processes feedback data from the follower gripper. This method is intended to compute 
        force-feedback for the leader gripper. This method is called at every iteration of the 
        control loop if self.enable_gripper_feedback is set to True.

        Args:
            leader_gripper_pos (float): Leader gripper position. Can be used to provide force-
            feedback for the gripper.
            leader_gripper_vel (float): Leader gripper velocity. Can be used to provide force-
            feedback for the gripper.
            gripper_feedback (Any): Feedback data from the gripper. The format can vary depending 
            on the implementation, such as a NumPy array, scalar, or custom object.
        
        Returns:
            float: The computed joint torque value to apply force-feedback to the leader gripper.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        pass


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
