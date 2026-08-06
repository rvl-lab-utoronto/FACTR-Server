"""
Read before making any changes to this file!

The class FACTRTeleopDualBase is intended for controlling the Dynamixel servos with two seperate boards. It is to be used together with the factr_rizon_dual_board.py file. 
"""

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
from std_msgs.msg import Float64, String
from python_utils.utils import get_workspace_root
from .angles import periodic_joint_error
from .gain_control import compose_arm_torque, ramp_gain
from .leader_config import load_leader_model
from .raw_joint_state import RawJointStateCache
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
            config_file_name = self.declare_parameter('config_file', 'factr_rizon_right.yaml').get_parameter_value().string_value

        config_path = os.path.join(get_workspace_root(), f"src/factr_teleop/factr_teleop/configs/{config_file_name}")
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
        
        self.name = self.config["name"]
        self.side = "left" if arm_index == 0 else "right"
        self.dt = 1 / self.config["controller"]["frequency"]
        
        self._prepare_dynamixel()
        self._raw_joint_state_cache = RawJointStateCache(self.num_motors)
        self._model_joint_signs = self.joint_signs.copy()
        self._prepare_inverse_dynamics()

        # Reading from /configs/franka_example.yaml
        # NEED TO modify the urdf file for the FACTR-Rizon setup!c FACTRT
        # leader arm parameters
        self.num_arm_joints = self.config["arm_teleop"]["num_arm_joints"]
        self.safety_margin = self.config["arm_teleop"]["arm_joint_limits_safety_margin"]
        self.arm_joint_limits_max = np.array(self.config["arm_teleop"]["arm_joint_limits_max"]) - self.safety_margin
        self.arm_joint_limits_min = np.array(self.config["arm_teleop"]["arm_joint_limits_min"]) + self.safety_margin
        self.initial_match_joint_pos = np.array(
            self.config["arm_teleop"]["initialization"]["model_home_q_rad"],
            dtype=float,
        )
        self._leader_model = load_leader_model(
            self.side,
            self.num_arm_joints,
            self._model_joint_signs,
            self.initial_match_joint_pos,
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
        null_space_config = self.config["controller"]["null_space_regulation"]
        self.null_space_regulation_enable = null_space_config.get("enable", True)
        self.null_space_joint_target = np.array(null_space_config["null_space_joint_target"])
        self.null_space_kp = null_space_config["kp"]
        self.null_space_kd = null_space_config["kd"]
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
        # gripper feedback
        self.enable_gripper_feedback = self.config["controller"]["gripper_feedback"]["enable"]
        
        self.set_up_communication()

        # Independent gravity and feedback activation targets published by the relay.
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
        self._grav_comp_gain_state_pub = self.create_publisher(
            Float64, f"/factr_grav_comp_gain_state_{side}", 10
        )
        self._force_feedback_gain_state_pub = self.create_publisher(
            Float64, f"/factr_force_feedback_gain_state_{side}", 10
        )
        self.create_timer(
            0.1, self._publish_gain_states, callback_group=self._gain_cb_group
        )
        self._telemetry_snapshot = {}
        self._telemetry_pub = self.create_publisher(String, f"/factr_telemetry_{side}", 10)
        self.create_timer(0.02, self._publish_telemetry, callback_group=self._gain_cb_group)

        # Follower external joint torques for the force-feedback term, pushed by the
        # DFC client through the FACTR API relay (WS force_feedback frames or
        # POST /force_feedback_<side> -> /factr_force_feedback_<side>). Joint-space:
        # JointState.effort holds one torque per arm joint in the follower's joint
        # convention — the same convention every torque this loop composes; a future
        # task-space variant would carry a 6-D TCP wrench and map it through the
        # leader Jacobian instead. Latest-value cache with a staleness cutoff so a
        # dropped client or dead relay never leaves a standing force on the leader.
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

    def _read_merged_pos_vel(self, source="unspecified"):
        """Read both boards; return (positions, velocities) in full arm order."""
        pos_small, vel_small = self.driver_small.get_positions_and_velocities(
            source=source
        )
        if self.big_servo_ids:
            pos_big, vel_big = self.driver_big.get_positions_and_velocities(
                source=source
            )
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
        Install the FACTR model offsets derived from DFC's launch contract.

        The current pose is sampled only for startup audit telemetry; it never
        changes calibration.
        """
        # Warm up both boards before auditing the DFC-provided model mapping.
        for _ in range(10):
            self._read_merged_pos_vel()
        curr_joints, curr_vel = self._read_merged_pos_vel()
        self._raw_joint_state_cache.update(curr_joints, curr_vel)

        # With saved arm calibration, do not infer any arm offset from the launch pose.
        # The gripper remains a raw signed servo angle in this path.
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
        Waits until the leader arm is manually moved to roughly the same configuration a
        joint_pos, _ = self.driver_small.get_positions_and_velocities()s the 
        follower arm before the follower arm starts mirroring the leader arm. 
        """
        curr_pos, _, _, _ = self.get_leader_joint_states()
        while (np.linalg.norm(curr_pos - self.initial_match_joint_pos[0:self.num_arm_joints]) > 10):
            current_joint_error = np.linalg.norm(
                curr_pos - self.initial_match_joint_pos[0:self.num_arm_joints]
            )
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

        joint_pos, joint_vel = self._read_merged_pos_vel(source="control")
        self._raw_joint_state_cache.update(joint_pos, joint_vel)

        raw_q = np.asarray(joint_pos[0:self.num_arm_joints], dtype=float)
        joint_pos_arm, joint_vel_arm = self._leader_model.model_state(
            raw_q, joint_vel[0:self.num_arm_joints]
        )
        self._last_raw_arm_q = raw_q
        self.gripper_pos = (joint_pos[-1] - self.joint_offsets[-1]) * self.joint_signs[-1]
        
        gripper_vel = (self.gripper_pos - self.gripper_pos_prev) / self.dt
        return joint_pos_arm, joint_vel_arm, self.gripper_pos, gripper_vel

    def get_cached_raw_joint_state(self):
        """Return the exact merged raw sample used by the latest control iteration."""
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
        """Set only the gravity-compensation activation target."""
        new_target = float(np.clip(msg.data, 0.0, 1.0))
        if new_target > 0.0 and self.grav_comp_gain_target <= 0.0:
            self._enable_capture_remaining = 25
            self._enable_capture_samples = []
        self.grav_comp_gain_target = new_target
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

    def _publish_gain_states(self):
        """Publish both independent live gains for the status endpoint."""
        grav_msg = Float64()
        grav_msg.data = float(self.grav_comp_gain)
        self._grav_comp_gain_state_pub.publish(grav_msg)
        feedback_msg = Float64()
        feedback_msg.data = float(self.force_feedback_gain)
        self._force_feedback_gain_state_pub.publish(feedback_msg)

    def _publish_telemetry(self):
        """Publish live control state plus zero-I/O Dynamixel diagnostics at 50 Hz."""
        if self._telemetry_snapshot:
            msg = String()
            payload = dict(self._telemetry_snapshot)
            payload["dynamixel"] = [
                driver.diagnostics_snapshot() for driver in self._drivers()
            ]
            msg.data = json.dumps(payload, separators=(",", ":"))
            self._telemetry_pub.publish(msg)

    def _capture_enable_tick(self, q, dq, tau_limit, tau_null, tau_gravity,
                             tau_friction, tau_feedback, tau_command, grav_gain):
        raw_q = self._last_raw_arm_q
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
            "force_feedback_torque_nm": tau_feedback.tolist(),
            "grav_comp_gain": float(grav_gain),
            "grav_comp_gain_target": float(self.grav_comp_gain_target),
            "friction_gain": float(self.stiction_comp_gain),
            "force_feedback_gain": float(self.force_feedback_gain),
            "force_feedback_gain_target": float(self.force_feedback_gain_target),
            "joint_offsets_rad": self.joint_offsets[:self.num_arm_joints].tolist(),
            "model_signs": self._leader_model.model_signs.tolist(),
            "applied_torque_nm": tau_command.tolist(),
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
        """Advance gravity and feedback activation ramps independently."""
        now = time.monotonic()
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
        return self.grav_comp_gain, self.force_feedback_gain

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
        if not self.null_space_regulation_enable:
            return np.zeros(self.num_arm_joints)
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
        and the follower arm for bilateral teleoperation. This method is called
        once in ``__init__``.
        
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
