"""Independent activation-gain helpers for FACTR torque composition."""

import numpy as np


def null_space_pd_objective(q_error, joint_velocity, kp, kd, torque_gain):
    """Weight a posture PD objective before it is projected into null space."""
    q_error = np.asarray(q_error, dtype=float)
    joint_velocity = np.asarray(joint_velocity, dtype=float)
    torque_gain = np.asarray(torque_gain, dtype=float)
    if joint_velocity.shape != q_error.shape or torque_gain.shape != q_error.shape:
        raise ValueError(
            "q_error, joint_velocity, and torque_gain must have matching shapes"
        )
    return torque_gain * (-float(kp) * q_error - float(kd) * joint_velocity)


def ramp_gain(value, target, last_update, ramp_time, now):
    """Advance one clamped activation gain without coupling it to another."""
    if last_update is None:
        return value, now
    elapsed = now - last_update
    if ramp_time <= 0.0:
        value = target
    else:
        step = elapsed / ramp_time
        if value < target:
            value = min(target, value + step)
        elif value > target:
            value = max(target, value - step)
    return min(1.0, max(0.0, value)), now


def compose_arm_torque(
    limit_torque,
    null_torque,
    gravity_torque,
    friction_torque,
    feedback_torque,
    null_space_gain,
    grav_comp_gain,
    force_feedback_gain,
    leader_torque_gain,
):
    """Compose every controller term behind one final leader-torque gate."""
    component_torque = (
        np.asarray(limit_torque)
        + null_space_gain * np.asarray(null_torque)
        + grav_comp_gain
        * (np.asarray(gravity_torque) + np.asarray(friction_torque))
        + force_feedback_gain * np.asarray(feedback_torque)
    )
    return leader_torque_gain * component_torque
