"""
Read before making any changes to this file!

The class FACTRTeleopDualBase is intended for controlling the Dynamixel servos with two seperate boards. It is to be used together with the factr_rizon_dual_board.py file. 
"""

import os
import time
import yaml
import subprocess
import numpy as np
import pinocchio as pin
from abc import ABC, abstractmethod

from rclpy.node import Node
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

        # gravity comp
        self.enable_gravity_comp = self.config["controller"]["gravity_comp"]["enable"]
        self.gravity_comp_modifier = self.config["controller"]["gravity_comp"]["gain"]
        self.tau_g = np.zeros(self.num_arm_joints)
        # friction compdriver_small
        self.stiction_comp_enable_speed = self.config["controller"]["static_friction_comp"]["enable_speed"]
        self.stiction_comp_gain = self.config["controller"]["static_friction_comp"]["gain"]
        self.stiction_dither_flag = np.ones((self.num_arm_joints), dtype=bool)
        # joint limit barrier:
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
        
        # needs to be implemented tarm_joint_limits_max: [4.5, 4.131, 6, 6, 8, 7.77, 3.32]  
        self.set_up_communication()

        # calibrate the leader arm joints before starting
        self._get_dynamixel_offsets()
        # ensure the leader and the follower arms have the same joint positions before starting
        self._match_start_pos()

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
        Instantiates driver for interfacing with Dynamixel servos.
        """
        self.servo_types = self.config["dynamixel"]["servo_types"]
        print(self.servo_types)
        self.num_motors = len(self.servo_types)
        self.joint_signs = np.array(self.config["dynamixel"]["joint_signs"], dtype=float)
        assert self.num_motors == len(self.joint_signs), \
            "The number of motors and the number of joint signs must be the same"
        self.dynamixel_port = "/dev/serial/by-id/" + self.config["dynamixel"]["dynamixel_port"]
        self.dynamixel_port_big = "/dev/serial/by-id/" + "usb-FTDI_USB__-__Serial_Converter_FTA2U2FX-if00-port0"

        # checks of the latency timer on ttyUSB of the corresponding port is 1
        # if it is not 1, the control loop cannot run at above 200 Hz, which will 
        # cause extremely undesirable behaviour for the leader arm. If the latency 
        # timer is not 1, one can set it to 1 as follows:
        # echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB{NUM}/latency_timer
        ttyUSBx = find_ttyusb(self.dynamixel_port)
        ttyUSBx_2 = find_ttyusb(self.dynamixel_port_big)
        command = f"cat /sys/bus/usb-serial/devices/{ttyUSBx}/latency_timer"        
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        ttyUSB_latency_timer = int(result.stdout)
        if ttyUSB_latency_timer != 1:
            raise Exception(
                f"Please ensure the latency timer of {ttyUSBx} is 1. Run: \n \
                echo 1 | sudo tee /sys/bus/usb-serial/devices/{ttyUSBx}/latency_timer"

                f"Please ensure the latency timer of {ttyUSBx_2} is 1. Run: \n \
                echo 1 | sudo tee /sys/bus/usb-serial/devices/{ttyUSBx_2}/latency_timer"
            )

        # Physical Dynamixel IDs can be either 1-8 or 9-16. Select via the
        # optional "id_start" config field (defaults to 1 for the 1-8 range).
        id_start = self.config["dynamixel"].get("id_start", 1)
        if id_start not in (1, 9):
            raise Exception(f"dynamixel id_start must be 1 or 9, got {id_start}")
        joint_ids = np.arange(self.num_motors) + id_start

        try:

            self.driver_small = DynamixelDriver(
                [1, 3, 5, 6, 7, 8], ['XC330_T288_T', 'XC330_T288_T', 'XC330_T288_T', 'XC330_T288_T', 'XC330_T288_T', 'XC330_T288_T'], self.dynamixel_port
            )
            self.driver_big = DynamixelDriver(
                [2,4], ['XM430_W210_T', 'XM430_W210_T'], self.dynamixel_port_big
            )

        except FileNotFoundError:
            self.get_logger().info(f"Port {self.dynamixel_port} not found. Please check the connection.")
            return
        self.driver_small.set_torque_mode(False)
        # set operating mode to current mode
        self.driver_small.set_operating_mode(0)
        # enable torque
        self.driver_small.set_torque_mode(True)
        self.driver_big.set_torque_mode(False)
        self.driver_big.set_operating_mode(0)
        self.driver_big.set_torque_mode(True)


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
        Calibrates the Dynamixel servos with respect to the Franka arm to ensure the joint
        position readings of the leader arm correspond to those of the follower arm.

        Before launching this program, the leader arm should be manually placed in a 
        configuration roughly corresponding to the follower's calibration position 
        described in self.calibration_joint_pos (within ±90 degrees per joint).
        """
        # warm up
        for _ in range(10):
            self.driver_small.get_positions_and_velocities()

        for _ in range(10):
            self.driver_big.get_positions_and_velocities()
        
        def _get_error(calibration_joint_pos, offset, index, joint_state):
            joint_sign_i = self.joint_signs[index]
            joint_i = joint_sign_i * (joint_state[index] - offset)
            start_i = calibration_joint_pos[index]
            return np.abs(joint_i - start_i)

        # get arm offsets
        self.joint_offsets = []

        curr_joints_pos, _ = self.driver_small.get_positions_and_velocities()
        curr_joints_pos_big, _ = self.driver_big.get_positions_and_velocities()

        curr_joints = np.insert(curr_joints_pos, 1, curr_joints_pos_big[0])
        curr_joints = np.insert(curr_joints, 3, curr_joints_pos_big[1])


        for i in range(self.num_arm_joints):
            print("Inside method _get_dynamixel_offsets", self.joint_offsets)
            # new changes
            if (i == 1 or i == 3):
                # big servos - not implemented yet
                self.joint_offsets.append(0) # space holder
                continue

            best_offset = 0
            best_error = 1e9
            # intervals of pi/2
            for offset in np.linspace(-20 * np.pi, 20 * np.pi, 20 * 4 + 1):  
                error = _get_error(self.calibration_joint_pos, offset, i, curr_joints)
                if error < best_error:
                    best_error = error
                    best_offset = offset
            self.joint_offsets.append(best_offset)

        print(self.joint_offsets)

        # get gripper offset:
        curr_gripper_joint = curr_joints[-1]
        self.joint_offsets.append(curr_gripper_joint)

        self.joint_offsets = np.asarray(self.joint_offsets)
        if verbose:
            print(self.joint_offsets)
            print("best offsets               : ", [f"{x:.3f}" for x in self.joint_offsets])
            print(
                "best offsets function of pi: ["
                + ", ".join([f"{int(np.round(x/(np.pi/2)))}*np.pi/2" for x in self.joint_offsets])
                + " ]",
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
        self.driver.set_torque_mode(Falsejoint_sign)

    def get_leader_joint_states(self):
        """
        Returns the current joint positions and velocities of the leader arm and gripper,
        aligned with the joint conventions (range and direction) of the follower arm.
        """
        self.gripper_pos_prev = self.gripper_pos
        joint_pos, joint_vel = self.driver_small.get_positions_and_velocities()
        joint_pos_big, joint_vel_big = self.driver_big.get_positions_and_velocities()

        joint_pos = np.insert(joint_pos, 1, joint_pos_big[0])
        joint_pos = np.insert(joint_pos, 3, joint_pos_big[1])
        joint_vel = np.insert(joint_vel, 1, joint_vel_big[0])
        joint_vel = np.insert(joint_vel, 3, joint_vel_big[1])

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
        Applies torque to the leader arm and gripper.    #             )
    #         elif h["torque_enable"] == 0 and self.driver.torque_enabled:
    #             self.get_logger().error(
    #                 f"[health] servo id{h['id']} disabled its own torque while the node sti
        """
        arm_gripper_torque = np.append(arm_torque, gripper_torque)
        small_signs = [1, 1, 1, 1, 1, 1]
        big_signs = [1, 1]
        
        # TODO!
        small_torque = np.delete(arm_gripper_torque, [1, 3]) # remove the first and third servos for now
        self.driver_small.set_torque(small_torque * small_signs) # equilvalent to joint_signs

        # TODO - for the 2 big servos
        big_torque = np.delete(arm_gripper_torque,[0, 2, 4, 5, 6, 7])
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
        exceed_max_mask = arm_joint_pos > self.arm_joint_limits_max
        tau_l = (-self.joint_limit_kp * (arm_joint_pos - self.arm_joint_limits_max) \
            - self.joint_limit_kd * arm_joint_vel) * exceed_max_mask
        exceed_min_mask = arm_joint_pos < self.arm_joint_limits_min
        tau_l += (-self.joint_limit_kp * (arm_joint_pos - self.arm_joint_limits_min) \
            - self.joint_limit_kd * arm_joint_vel) * exceed_min_mask
        
        if gripper_joint_pos > self.gripper_limit_max:
            tau_l_gripper = -self.joint_limit_kp * (gripper_joint_pos - self.gripper_limit_max) \
                - self.joint_limit_kd * gripper_joint_vel
        elif gripper_joint_pos < self.gripper_limit_min:
            tau_l_gripper = -self.joint_limit_kp * (gripper_joint_pos - self.gripper_limit_min) \
                - self.joint_limit_kd * gripper_joint_vel
        else:
            tau_l_gripper = 0.0
        return tau_l, tau_l_gripper

    def gravity_compensation(self, arm_joint_pos, arm_joint_vel):
        """
        Computes joint torque for gravity compensation using inverse dynamics.
        This method uses the Recursive Newton-Euler Algorithm (RNEA), provided by the 
        Pinocchio library, to calculate the torques required to counteract gravity 
        at the current joint states. The result is scaled by a modifier to tune the 
        compensation strength.

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
        # visit factr_rizon_dual_board.py
        a = 1 + 1


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
