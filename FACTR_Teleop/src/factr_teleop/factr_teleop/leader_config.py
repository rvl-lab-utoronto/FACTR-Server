"""DFC-owned leader calibration consumed by FACTR at process launch.

The persisted source of truth lives in DFC. FACTR receives one JSON contract in
``DFC_LEADER_CONFIG`` and derives the coordinates required by its dynamics model.
"""

from dataclasses import dataclass
import json
import os

import numpy as np


@dataclass(frozen=True)
class LeaderControlGains:
    """Every physical controller gain supplied by DFC at process launch."""

    grav_comp_gain_ramp_time: float
    force_feedback_gain_ramp_time: float
    null_space_gain_ramp_time: float
    gravity_comp_gain: float
    force_feedback_gain: float
    force_feedback_damping: float
    force_feedback_motor_scalar: float
    force_feedback_max_torque: float
    static_friction_gain: np.ndarray
    joint_limit_kp: float
    joint_limit_kd: float
    null_space_kp: float
    null_space_kd: float
    null_space_torque_gain: np.ndarray
    joint_position_kp: float
    joint_position_kd: float
    gripper_spring_stiffness_nm_per_rad: float
    gripper_spring_damping_nm_s_per_rad: float
    gripper_spring_max_torque_nm: float


@dataclass(frozen=True)
class LeaderModelConfig:
    side: str
    dynamixel_id_start: int
    leader_urdf: str | None
    connected_joint_indices: np.ndarray | None
    gripper_open_raw: float
    gripper_closed_raw: float
    joint_torque_gain: np.ndarray
    raw_offsets_rad: np.ndarray
    raw_signs: np.ndarray
    dfc_to_factr_signs: np.ndarray
    home_factr_q_rad: np.ndarray
    model_signs: np.ndarray
    joint_offsets_rad: np.ndarray
    control_gains: LeaderControlGains

    def model_state(self, raw_q, raw_dq):
        """Convert raw Dynamixel state using DFC's authoritative arm signs."""
        raw_q = np.asarray(raw_q, dtype=float)
        raw_dq = np.asarray(raw_dq, dtype=float)
        q_factr = self.model_signs * (raw_q - self.joint_offsets_rad)
        dq_factr = self.model_signs * raw_dq
        return q_factr, dq_factr

    def dfc_position_to_model(self, q_dfc):
        """Map one canonical DFC/Rizon joint vector into FACTR model axes."""
        q_dfc = np.asarray(q_dfc, dtype=float)
        if q_dfc.shape != self.dfc_to_factr_signs.shape:
            raise ValueError(
                "DFC position must contain exactly one value per configured joint"
            )
        return self.dfc_to_factr_signs * q_dfc

    def motor_signs(self, gripper_sign):
        """Return model-torque-to-motor signs for arm joints plus gripper."""
        gripper_sign = float(gripper_sign)
        if not np.isfinite(gripper_sign) or gripper_sign not in (-1.0, 1.0):
            raise RuntimeError("FACTR gripper hardware sign must be -1 or +1")
        return np.append(self.model_signs, gripper_sign)

    def scale_joint_torque(self, arm_torque):
        """Apply per-joint gains to one final FACTR arm-torque command."""
        arm_torque = np.asarray(arm_torque, dtype=float)
        if arm_torque.shape != self.joint_torque_gain.shape:
            raise ValueError(
                "arm torque must contain exactly one value per configured joint"
            )
        return self.joint_torque_gain * arm_torque

    def hardware_motor_indices(self):
        """Full-frame motor slots actually present on the Dynamixel bus."""
        if self.connected_joint_indices is None:
            return np.arange(self.joint_torque_gain.size + 1, dtype=int)
        return self.connected_joint_indices.copy()

    def synthetic_raw_position(self):
        """Full raw frame with absent joints at model home and gripper open."""
        arm_home_raw = (
            self.joint_offsets_rad + self.home_factr_q_rad / self.model_signs
        )
        return np.append(arm_home_raw, self.gripper_open_raw)

    def expand_hardware_state(self, position, velocity):
        """Expand one physical read into the normal 7-joint-plus-gripper frame."""
        indices = self.hardware_motor_indices()
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        expected = (indices.size,)
        if position.shape != expected or velocity.shape != expected:
            raise ValueError(
                f"physical state must have shape {expected}, got "
                f"position={position.shape}, velocity={velocity.shape}"
            )
        full_position = self.synthetic_raw_position()
        full_velocity = np.zeros_like(full_position)
        full_position[indices] = position
        full_velocity[indices] = velocity
        return full_position, full_velocity

    def expand_hardware_values(self, values, fill=0.0):
        """Place physical current/torque values into a full motor vector."""
        indices = self.hardware_motor_indices()
        values = np.asarray(values, dtype=float)
        if values.shape != (indices.size,):
            raise ValueError(
                f"physical values must have shape {(indices.size,)}, got {values.shape}"
            )
        full = np.full(self.joint_torque_gain.size + 1, float(fill))
        full[indices] = values
        return full

    def select_hardware_values(self, values):
        """Select commands for only the motors physically present on the bus."""
        values = np.asarray(values, dtype=float)
        expected = (self.joint_torque_gain.size + 1,)
        if values.shape != expected:
            raise ValueError(f"full motor values must have shape {expected}")
        return values[self.hardware_motor_indices()]


def apply_control_gains(controller, gains):
    """Overlay DFC's required gain contract onto FACTR's hardware config."""
    controller["grav_comp_gain_ramp_time"] = gains.grav_comp_gain_ramp_time
    controller["force_feedback_gain_ramp_time"] = gains.force_feedback_gain_ramp_time
    controller["gravity_comp"]["gain"] = gains.gravity_comp_gain
    controller["torque_feedback"].update(
        gain=gains.force_feedback_gain,
        damping=gains.force_feedback_damping,
        motor_scalar=gains.force_feedback_motor_scalar,
        max_torque=gains.force_feedback_max_torque,
    )
    controller["static_friction_comp"]["gain"] = gains.static_friction_gain.tolist()
    controller["joint_limit_barrier"].update(
        kp=gains.joint_limit_kp, kd=gains.joint_limit_kd
    )
    controller["null_space_regulation"].update(
        gain_ramp_time=gains.null_space_gain_ramp_time,
        kp=gains.null_space_kp,
        kd=gains.null_space_kd,
        torque_gain=gains.null_space_torque_gain.tolist(),
    )
    controller["joint_position_control"].update(
        kp=gains.joint_position_kp, kd=gains.joint_position_kd
    )
    controller["gripper_spring"].update(
        stiffness_nm_per_rad=gains.gripper_spring_stiffness_nm_per_rad,
        damping_nm_s_per_rad=gains.gripper_spring_damping_nm_s_per_rad,
        max_torque_nm=gains.gripper_spring_max_torque_nm,
    )
    return controller


def load_leader_model(expected_side, dof, model_home_q_rad, environ=None):
    """Parse DFC's calibrated raw-joint contract for the FACTR model.

    ``raw_to_dfc`` is the sole affine encoder calibration. FACTR's mechanism
    configuration supplies only its model home, while ``dfc_to_factr.signs``
    describes model-axis directions.
    """
    environ = os.environ if environ is None else environ
    encoded = environ.get("DFC_LEADER_CONFIG")
    if not encoded:
        raise RuntimeError(
            "DFC_LEADER_CONFIG is required; leader calibration is owned by DFC"
        )
    try:
        data = json.loads(encoded)
        side = str(data["side"])
        dynamixel_id_start = int(data["dynamixel_id_start"])
        leader_urdf_value = data.get("leader_urdf")
        leader_urdf = (
            None if leader_urdf_value is None else str(leader_urdf_value)
        )
        if leader_urdf is not None and (
            os.path.basename(leader_urdf) != leader_urdf
            or not leader_urdf.endswith(".urdf")
        ):
            raise ValueError("leader_urdf must be a local .urdf basename")
        connected_value = data.get("connected_joint_indices")
        if connected_value is None:
            connected_joint_indices = None
        else:
            if (
                not isinstance(connected_value, list)
                or not connected_value
                or any(type(index) is not int for index in connected_value)
            ):
                raise ValueError(
                    "connected_joint_indices must be a non-empty list of integers"
                )
            connected_joint_indices = np.asarray(connected_value, dtype=int)
        joint_torque_gain = np.asarray(data["joint_torque_gain"], dtype=float)
        raw = data["raw_to_dfc"]
        transform = data["dfc_to_factr"]
        if "offset_rad" in transform:
            raise ValueError(
                "dfc_to_factr.offset_rad is derived by FACTR and must not be persisted"
            )
        offsets = np.radians(np.asarray(raw["offsets_deg"], dtype=float))
        flips = [int(i) for i in raw["sign_flip_joints"]]
        raw_signs = np.ones(dof)
        raw_signs[flips] = -1.0
        transform_signs = np.asarray(transform["signs"], dtype=float)
        drop = int(raw["drop_trailing"])
        gripper_open = float(raw["gripper_open"])
        gripper_closed = float(raw["gripper_closed"])
        gains = data["control_gains"]
        gravity = gains["gravity_comp"]
        feedback = gains["torque_feedback"]
        friction = gains["static_friction_comp"]
        limits = gains["joint_limit_barrier"]
        null = gains["null_space_regulation"]
        position = gains["joint_position_control"]
        gripper_spring = gains["gripper_spring"]
        control_gains = LeaderControlGains(
            grav_comp_gain_ramp_time=float(gains["grav_comp_gain_ramp_time"]),
            force_feedback_gain_ramp_time=float(
                gains["force_feedback_gain_ramp_time"]
            ),
            null_space_gain_ramp_time=float(gains["null_space_gain_ramp_time"]),
            gravity_comp_gain=float(gravity["gain"]),
            force_feedback_gain=float(feedback["gain"]),
            force_feedback_damping=float(feedback["damping"]),
            force_feedback_motor_scalar=float(feedback["motor_scalar"]),
            force_feedback_max_torque=float(feedback["max_torque"]),
            static_friction_gain=_per_joint_gain(friction["gain"], dof),
            joint_limit_kp=float(limits["kp"]),
            joint_limit_kd=float(limits["kd"]),
            null_space_kp=float(null["kp"]),
            null_space_kd=float(null["kd"]),
            null_space_torque_gain=_per_joint_gain(
                null.get("torque_gain", 1.0),
                dof,
                "null_space_regulation.torque_gain",
            ),
            joint_position_kp=float(position["kp"]),
            joint_position_kd=float(position["kd"]),
            gripper_spring_stiffness_nm_per_rad=float(
                gripper_spring["stiffness_nm_per_rad"]
            ),
            gripper_spring_damping_nm_s_per_rad=float(
                gripper_spring["damping_nm_s_per_rad"]
            ),
            gripper_spring_max_torque_nm=float(gripper_spring["max_torque_nm"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid DFC_LEADER_CONFIG: {exc}") from exc

    home_factr = np.asarray(model_home_q_rad, dtype=float)
    vectors = (offsets, raw_signs, transform_signs, home_factr)
    if side != expected_side:
        raise RuntimeError(
            f"DFC leader contract is for {side!r}, expected {expected_side!r}"
        )
    if dynamixel_id_start not in (1, 9):
        raise RuntimeError(
            f"DFC leader dynamixel_id_start must be 1 or 9, got {dynamixel_id_start}"
        )
    if joint_torque_gain.shape != (dof,):
        raise RuntimeError(
            f"DFC leader joint_torque_gain must contain {dof} arm joints"
        )
    if not np.isfinite(joint_torque_gain).all() or np.any(
        (joint_torque_gain < 0.0) | (joint_torque_gain > 2.0)
    ):
        raise RuntimeError("DFC leader joint_torque_gain values must be in [0, 2]")
    if connected_joint_indices is not None and (
        np.any(connected_joint_indices < 0)
        or np.any(connected_joint_indices >= dof)
        or np.any(np.diff(connected_joint_indices) <= 0)
    ):
        raise RuntimeError(
            "DFC leader connected_joint_indices must be unique, increasing, "
            f"and within [0, {dof - 1}]"
        )
    if drop != 1 or any(v.shape != (dof,) for v in vectors):
        raise RuntimeError(f"DFC leader contract must contain {dof} arm joints")
    if len(set(flips)) != len(flips) or any(i < 0 or i >= dof for i in flips):
        raise RuntimeError(f"invalid DFC sign-flip indices: {flips}")
    if not all(np.all(np.isfinite(v)) for v in vectors):
        raise RuntimeError("DFC leader contract contains non-finite values")
    if not np.all(np.isin(transform_signs, (-1.0, 1.0))):
        raise RuntimeError("DFC-to-FACTR signs must be -1 or +1")
    if not np.isfinite([gripper_open, gripper_closed]).all() or gripper_open == gripper_closed:
        raise RuntimeError("invalid DFC gripper calibration")
    gain_values = np.concatenate([
        np.asarray(value, dtype=float).reshape(-1)
        for value in control_gains.__dict__.values()
    ])
    if not np.isfinite(gain_values).all() or np.any(gain_values < 0.0):
        raise RuntimeError("DFC control gains must be finite and non-negative")
    if control_gains.force_feedback_motor_scalar <= 0.0:
        raise RuntimeError("DFC force-feedback motor scalar must be positive")
    if control_gains.force_feedback_max_torque <= 0.0:
        raise RuntimeError("DFC force-feedback max torque must be positive")
    if control_gains.gripper_spring_max_torque_nm <= 0.0:
        raise RuntimeError("DFC gripper spring max_torque_nm must be positive")

    # DFC owns both sign stages. Their product is the raw Dynamixel -> FACTR-model
    # slope used by state conversion and by the inverse torque conversion. Since
    # raw_to_dfc adds ``offsets`` before its sign, the equivalent FACTR driver
    # convention q_model = model_signs * (q_raw - joint_offsets) uses -offsets.
    model_signs = raw_signs * transform_signs
    joint_offsets = -offsets
    return LeaderModelConfig(
        side=side,
        dynamixel_id_start=dynamixel_id_start,
        leader_urdf=leader_urdf,
        connected_joint_indices=connected_joint_indices,
        gripper_open_raw=gripper_open,
        gripper_closed_raw=gripper_closed,
        joint_torque_gain=joint_torque_gain,
        raw_offsets_rad=offsets,
        raw_signs=raw_signs,
        dfc_to_factr_signs=transform_signs,
        home_factr_q_rad=home_factr,
        model_signs=model_signs,
        joint_offsets_rad=joint_offsets,
        control_gains=control_gains,
    )


def _per_joint_gain(value, dof, label="static_friction_comp.gain"):
    """Normalize a legacy scalar or a per-joint list to one strict vector."""
    gain = np.asarray(value, dtype=float)
    if gain.shape == ():
        return np.full(dof, float(gain))
    if gain.shape != (dof,):
        raise ValueError(f"{label} must be scalar or contain {dof} arm joints")
    return gain
