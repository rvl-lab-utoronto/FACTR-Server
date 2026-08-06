"""Independent activation-gain helpers for FACTR torque composition."""

import numpy as np


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
    grav_comp_gain,
    force_feedback_gain,
):
    """Compose torque with gravity and feedback scaled independently."""
    return (
        np.asarray(limit_torque)
        + np.asarray(null_torque)
        + grav_comp_gain
        * (np.asarray(gravity_torque) + np.asarray(friction_torque))
        + force_feedback_gain * np.asarray(feedback_torque)
    )
