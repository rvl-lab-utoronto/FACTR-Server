"""
Read before making any changes to this file!

The class FACTRTeleopDualBase is intended for controlling the Dynamixel servos with two seperate boards. It is to be used together with the factr_rizon_dual_board.py file. 
"""

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
from .angles import periodic_joint_error
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


class FACTRTeleopDualBase(Node, ABC):
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


        #!BOOKMARK, probably delete
        if arm_index == 0:
            # left arm
            config_file_name = self.declare_parameter('config_file', 'factr_rizon_left.yaml').get_parameter_value().string_value
        else:
            # right arm
            print("TRUE")
            
            config_file_name = self.declare_parameter('config_file', 'factr_rizon_right.yaml').get_parameter_value().string_value

        config_path = os.path.join(get_workspace_root(), f"src/factr_teleop/factr_teleop/configs/{config_file_name}")
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
        
        self.name = self.config["name"]
        print(self.name)
        self.dt = 1 / self.config["controller"]["frequency"]
        
        self._prepare_dynamixel()
        self._model_joint_signs = self.joint_signs.copy()
        self._prepare_inverse_dynamics()

        # Reading from /configs/franka_example.yaml
        # NEED TO modify the urdf file for the FACTR-Rizon setup!c FACTRT
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
        # friction compdriver_small
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
        # base joint OVERLOAD/OVERHEAT that silently kills its gravity comp). See
        # DynamixelDriver.read_health(). Set period to 0 to disable.c FACTRT
        self.health_log_period = self.config["controller"].get("health_log_period", 1.0)
        self._last_health_log = time.time()

        # start the control loop
        # self.dt = 500Hz
        # self.timer = self.create_timer(self.dt, self.control_loop_callback)

        
        self._initialize_position()

    def _initialize_position(self):
        """ Initialize positions"""
        # # refer to Dynamixel Wizard
        # self.driver_big.set_torque_mode(False) # turn it off first
        # self.driver_big.set_operating_mode(3) # mode 3
        # self.driver_big.set_torque_mode(True)
        # self.driver_big.set_position([3, 1.5]) # move the two big servos to the initial positions

        # self.driver_small.set_torque_mode(False)
        # self.driver_small.set_operating_mode(3)
        # self.driver_small.set_torque_mode(True)
        # time.sleep(1)
        # self.driver_small.set_position([0, 6.4, 3, 4.7, 2, 0])
        
        # time.sleep(4)
        # start control loop only after the positions are initialized
        self.timer = self.create_timer(self.dt, self.control_loop_callback)
        return 


    def _prepare_dynamixel(self):
        """
        Instantiates the Dynamixel drivers for the small and big power boards.

        Each board is described by its own config section (``dynamixel_small`` and,
        optionally, ``dynamixel_big``) listing that board's Dynamixel ``ids``,
        ``servo_types`` and ``joint_signs`` in id order. The two boards are merged
        into a single full-arm ordering (ascending Dynamixel id) so the rest of the
        controller can treat the leader arm as one contiguous set of joints.
        """
        small_cfg = self.config["dynamixel_small"]
        big_cfg = self.config.get("dynamixel_big") or {}

        # --- per-board servo tables, each in that board's own id order ---
        self.small_servo_ids = [int(i) for i in small_cfg["ids"]]
        small_types = list(small_cfg["servo_types"])
        small_signs = list(small_cfg["joint_signs"])
        self.dynamixel_port = "/dev/serial/by-id/" + small_cfg["dynamixel_port"]

        self.big_servo_ids = [int(i) for i in big_cfg.get("ids", [])]
        big_types = list(big_cfg.get("servo_types", []))
        big_signs = list(big_cfg.get("joint_signs", []))
        self.dynamixel_port_big = (
            "/dev/serial/by-id/" + big_cfg["dynamixel_port"] if self.big_servo_ids else None
        )

        for label, ids, types, signs in (
            ("dynamixel_small", self.small_servo_ids, small_types, small_signs),
            ("dynamixel_big", self.big_servo_ids, big_types, big_signs),
        ):
            assert len(ids) == len(types) == len(signs), \
                f"{label}: ids, servo_types and joint_signs must have equal length"

        # --- merge both boards into one full-arm ordering (ascending Dynamixel id) ---
        all_ids = sorted(self.small_servo_ids + self.big_servo_ids)
        assert len(all_ids) == len(set(all_ids)), "duplicate Dynamixel id across boards"
        id_to_index = {sid: idx for idx, sid in enumerate(all_ids)}

        self.num_motors = len(all_ids)
        self.servo_types = [None] * self.num_motors
        self.joint_signs = np.zeros(self.num_motors)
        for sid, stype, ssign in zip(
            self.small_servo_ids + self.big_servo_ids,
            small_types + big_types,
            small_signs + big_signs,
        ):
            self.servo_types[id_to_index[sid]] = stype
            self.joint_signs[id_to_index[sid]] = ssign

        # merged-array positions of each board's servos, kept in that board's id
        # order so they line up with what the driver reads/writes for those ids.
        self.small_servo_indices = np.array(
            [id_to_index[sid] for sid in self.small_servo_ids], dtype=int
        )
        self.big_servo_indices = np.array(
            [id_to_index[sid] for sid in self.big_servo_ids], dtype=int
        )
        self._big_index_set = set(int(i) for i in self.big_servo_indices)

        # The latency timer on each ttyUSB must be 1, otherwise the control loop
        # cannot run above 200 Hz, causing very undesirable leader-arm behaviour.
        self._assert_latency_timer(self.dynamixel_port)
        if self.big_servo_ids and self.dynamixel_port_big != self.dynamixel_port:
            self._assert_latency_timer(self.dynamixel_port_big)

        try:
            self.driver_small = DynamixelDriver(
                self.small_servo_ids, small_types, self.dynamixel_port
            )
            if self.big_servo_ids:
                self.driver_big = DynamixelDriver(
                    self.big_servo_ids, big_types, self.dynamixel_port_big
                )
        except FileNotFoundError:
            self.get_logger().info(f"Port {self.dynamixel_port} not found. Please check the connection.")
            return

        # set every board to current-control mode (0) with torque enabled
        for driver in self._drivers():
            driver.set_torque_mode(False)
            driver.set_operating_mode(0)
            driver.set_torque_mode(True)

    def _drivers(self):
        """Iterate over the active Dynamixel drivers (big board only if configured)."""
        yield self.driver_small
        if self.big_servo_ids:
            yield self.driver_big

    def _assert_latency_timer(self, port):
        """Raise unless the ttyUSB latency timer for ``port`` is 1.

        If it is not 1, set it with:
            echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB{NUM}/latency_timer
        """
        ttyusb = find_ttyusb(port)
        result = subprocess.run(
            f"cat /sys/bus/usb-serial/devices/{ttyusb}/latency_timer",
            shell=True, capture_output=True, text=True, check=True,
        )
        if int(result.stdout) != 1:
            raise Exception(
                f"Please ensure the latency timer of {ttyusb} is 1. Run:\n"
                f"echo 1 | sudo tee /sys/bus/usb-serial/devices/{ttyusb}/latency_timer"
            )

    def _merge_small_big(self, small_vals, big_vals):
        """Interleave small/big board readings into full arm order (length num_motors)."""
        full = np.zeros(self.num_motors)
        full[self.small_servo_indices] = small_vals
        if self.big_servo_ids:
            full[self.big_servo_indices] = big_vals
        return full

    def _read_merged_pos_vel(self):
        """Read both boards; return (positions, velocities) in full arm order."""
        pos_small, vel_small = self.driver_small.get_positions_and_velocities()
        if self.big_servo_ids:
            pos_big, vel_big = self.driver_big.get_positions_and_velocities()
        else:
            pos_big = vel_big = np.empty(0)
        return self._merge_small_big(pos_small, pos_big), self._merge_small_big(vel_small, vel_big)


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
        # Build the kinematic/dynamic model only (no geometry). Gravity compensation and
        # null-space regulation use inertial data only, not the visual/collision meshes,
        # so this avoids any dependency on mesh files being present on disk.
        self.pin_model = pin.buildModelFromUrdf(urdf_model_path)
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
        # Warm up both boards before measuring the configured calibration pose.
        for _ in range(10):
            self._read_merged_pos_vel()
        curr_joints, _ = self._read_merged_pos_vel()

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
        convention. DFC's sign flips belong only to follower commands; gravity
        compensation retains FACTR's native mechanism/model signs and fixed URDF bias:

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
        model_signs = self._model_joint_signs[:self.num_arm_joints]
        arm_offsets = -np.radians(offsets_deg) - model_signs * model_bias
        self.joint_signs = self._model_joint_signs.copy()
        self.joint_offsets = np.concatenate([arm_offsets, [0.0]])
        self._dfc_calibration_received = True
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
        Waits until the leader arm is manually moved to roughly the same configuration a
        joint_pos, _ = self.driver_small.get_positions_and_velocities()s the 
        follower arm before the follower arm starts mirroring the leader arm. 
        """
        curr_pos, _, _, _ = self.get_leader_joint_states()
        while (np.linalg.norm(curr_pos - self.initial_match_joint_pos[0:self.num_arm_joints]) > 10):
            current_joint_error = np.linalg.norm(
                curr_pos - self.initial_match_joint_pos[0:self.num_arm_joints]
            )
            print("current joint pos: ", [f"{x:.3f}" for x in curr_pos])
            self.get_logger().info(
                f"FACTR TELEOP {self.name}: Please match starting joint pos. Current error: {current_joint_error}"
            )
            curr_pos, _, _, _ = self.get_leader_joint_states()
            time.sleep(0.5)
        self.get_logger().info(f"FACTR TELEOP {self.name}: Initial joint position matched.")

    def shut_down(self):
        """
        Disables all torque on the leader arm and gripper during node shutdown.
        """
        self.set_leader_joint_torque(np.zeros(self.num_arm_joints), 0.0)
        for driver in self._drivers():
            driver.set_torque_mode(False)

    def get_leader_joint_states(self):
        """
        Returns the current joint positions and velocities of the leader arm and gripper,
        aligned with the joint conventions (range and direction) of the follower arm.
        """
        self.gripper_pos_prev = self.gripper_pos

        joint_pos, joint_vel = self._read_merged_pos_vel()

        joint_pos_arm = (
            joint_pos[0:self.num_arm_joints] - self.joint_offsets[0:self.num_arm_joints]
        ) * self.joint_signs[0:self.num_arm_joints]
        self.gripper_pos = (joint_pos[-1] - self.joint_offsets[-1]) * self.joint_signs[-1]
        joint_vel_arm = joint_vel[0:self.num_arm_joints] * self.joint_signs[0:self.num_arm_joints]
        
        gripper_vel = (self.gripper_pos - self.gripper_pos_prev) / self.dt
        print(joint_pos_arm, joint_vel_arm)
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
this is not what I want to 
        The full command (arm joints followed by the gripper, in merged arm order) is
        split across the small and big boards and converted back to each servo's
        physical direction using the configured joint_signs.
        """
        arm_gripper_torque = np.append(arm_torque, gripper_torque)

        small_torque = arm_gripper_torque[self.small_servo_indices]
        small_signs = self.joint_signs[self.small_servo_indices]
        self.driver_small.set_torque(small_torque * small_signs)

        if self.big_servo_ids:
            big_torque = arm_gripper_torque[self.big_servo_indices]
            big_signs = self.joint_signs[self.big_servo_indices]
            self.driver_big.set_torque(big_torque * big_signs)

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
        sample = {
            "stamp_monotonic_ns": time.monotonic_ns(),
            "raw_q_rad": (offsets + q / signs).tolist(),
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
        via `self.null_space_joint_target`. It implcontinuehe control strategy described 
        in Equation 3 of Section III.B in the paper, projecting a PD control law into 
        the null space of the task Jacobian to achieve secondary objectives without 
        affecting the primary task.
        """
        J = pin.computeJointJacobian(
            self.pin_model, self.pin_data, arm_joint_pos, self.num_arm_joints
        )
        J_dagger = np.linalg.pinv(J)
        null_space_projector = np.eye(self.num_arm_joints) - J_dagger @ J
        # Ignore DYNAMIXEL multi-turn branch selection in this linear controller.
        # Raw readings remain untouched for streaming and diagnostics.
        q_error = periodic_joint_error(
            arm_joint_pos,
            self.null_space_joint_target[0:self.num_arm_joints],
        )
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
        # visit factr_rizon_dual_board.py
        a = 1 + 1

    def _log_servo_health(self):
        """Throttled (~``health_log_period`` s) servo health read + log across BOTH boards.

        Reads temperature / present current / hardware-error for every servo on the small
        AND big boards (via ``self._drivers()``), so id2/id4 on the big board show up next
        to the small-board servos. Watch ``T`` climb and ``I`` sit near its limit, and get
        a loud ERROR the moment a servo latches a Hardware Error (OVERLOAD/OVERHEAT/...) or
        disables its own torque -- e.g. the elbow (id4) silently dropping gravity comp. On
        such a latch the servo stays limp until a reboot/power-cycle.

        Purely additive telemetry: it does NOT touch the control/torque path. read_health()
        issues per-servo register reads, so it briefly stalls the loop (the servos hold
        their last goal current across the gap). Set health_log_period=0 to disable.
        """
        if not self.health_log_period:
            return
        now = time.time()
        if now - self._last_health_log < self.health_log_period:
            return
        self._last_health_log = now

        health = []
        for driver in self._drivers():
            try:
                health.extend(driver.read_health())
            except Exception as e:
                self.get_logger().warning(f"[health] read failed: {e}")
                return
        health.sort(key=lambda h: h["id"])   # ascending id so id2/id4 line up with the rest

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
            elif h["torque_enable"] == 0:
                self.get_logger().error(
                    f"[health] servo id{h['id']} has torque DISABLED (likely a protective "
                    f"shutdown) T={h['temperature']}C I={h['present_current']}"
                )


    # def _log_servo_health(self):
    #     """Throttled (~``health_log_period`` s) servo health read + log.

    #     Diagnoses torque dropouts such as the base joint silently losing gravity comp:
    #     watch ``T`` (temperature) climb over the minute and ``I`` (present current) sit
    #     pegged near its limit, and get an explicit ERROR the moment a servo latches a
    #     Hardware Error (OVERLOAD/OVERHEAT/...) or disables its own torque. On such a
    #     latch FACTR_Teleop/src/factr_teleop/factr_teleop/factr_rizarm_joint_limits_max: [4.5, 4.131, 6, 6, 8, 7.77, 3.32]  
    #     if not self.health_log_period:
    #         return
    #     now = tFACTR_Teleop/src/factr_teleop/factr_teleop/factr_rizon_dual_board.py- self._last_health_log < self.health_log_period:
    #         return
    #     self._last_health_log = now

    #     try:
    #         health = self.driver.read_health()
    #     except Exception as e:
    #         self.get_logger().warning(f"[health] read failed: {e}")
    #         return

    #     summary = "  ".join(
    #         f"id{h['id']}:T={h['temperature']}C I={h['present_current']}/eturn

    #     summary = "  ".join(
    #         f"id{h['id']}:T={h['temperature']}C I={h['present_current']}/{h['current_limit']}"
    #         for h in health{h['current_limit']}"
    #         for h in health
    #     )
    #     self.get_logger().info(f"[health] {summary}")

    #     for h in health:
    #         if h["hw_error"]:
    #             self.get_logger().error(
    #                 f"[health] servo id{h['id']} HARDWARE ERROR {h['hw_error_flags']} "
    #                 f"(0x{h['hw_error']:02x}) T={h['temperature']}C I={h['present_current']} "
    #                 f"-- torque has latched OFF; reboot/power-cycle the servo to clear"
    #             )
    #         elif h["torque_enable"] == 0 and self.driver.torque_enabled:
    #             self.get_logger().error(
    #                 f"[health] servo id{h['id']} disabled its own torque while the node still "
    #                 f"commands torque ON (likely protective shutdown) T={h['temperature']}C "
    #                 f"I={h['present_current']}"
    #             )


    @abstractmethod
    def set_up_communication(self):
        """
        This method should be implemented to set up communication between the leader arm
        and the follower arm for bilateral teleoperation. This method iarm_joint_limits_max: [4.5, 4.131, 6, 6, 8, 7.77, 3.32]  
  arm_joint_limits_min: [-0.5, -4.5, -3, -6, -4, -4, -3.32] # THIS DOES NOT INCLUDE THE TRIGGER!  
  arm_joint_limits_safety_margin: 0.1
  initialization:
    # The expected leader arm jo0int position before launching the script
    calibration_joint_pos: [0., -0, 0.0, 0, 0, 0, 0] 
    # The arm needs to be held in a certain position before launching!
    # More information in the README.md file!
    # The leader arm must be brought to this joint position before the follower starts mirroring the leader
    initial_match_joint_pos: [0.0, -0, 0.0, 0, 0, 0, 0]s called once
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
