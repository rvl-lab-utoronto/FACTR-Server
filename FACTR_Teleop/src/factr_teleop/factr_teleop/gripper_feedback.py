"""Virtual torsional spring for a FACTR leader's gripper trigger."""

import math


def torsional_spring_torque_nm(
    position_rad: float,
    velocity_rad_s: float,
    open_position_rad: float,
    closed_position_rad: float,
    stiffness_nm_per_rad: float,
    damping_nm_s_per_rad: float,
    max_torque_nm: float,
) -> float:
    """Return a one-sided spring torque that restores the trigger toward open.

    The calibrated endpoints determine the closing direction, so the same law
    works for leaders whose raw trigger angles run in opposite directions. The
    spring is inactive on the open side, its displacement is capped at the
    calibrated closed endpoint, and the final torque has an independent safety
    limit. Damping is applied only while the spring is engaged.
    """
    values = (
        position_rad,
        velocity_rad_s,
        open_position_rad,
        closed_position_rad,
        stiffness_nm_per_rad,
        damping_nm_s_per_rad,
        max_torque_nm,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("gripper spring parameters must be finite")

    position_rad = float(position_rad)
    velocity_rad_s = float(velocity_rad_s)
    open_position_rad = float(open_position_rad)
    closed_position_rad = float(closed_position_rad)
    stiffness_nm_per_rad = float(stiffness_nm_per_rad)
    damping_nm_s_per_rad = float(damping_nm_s_per_rad)
    max_torque_nm = float(max_torque_nm)

    closing_span_rad = closed_position_rad - open_position_rad
    if closing_span_rad == 0.0:
        raise ValueError("gripper spring endpoints must differ")
    if stiffness_nm_per_rad < 0.0 or damping_nm_s_per_rad < 0.0:
        raise ValueError("gripper spring stiffness and damping must be non-negative")
    if max_torque_nm <= 0.0:
        raise ValueError("gripper spring max torque must be positive")

    closing_direction = 1.0 if closing_span_rad > 0.0 else -1.0
    closure_rad = closing_direction * (position_rad - open_position_rad)
    closure_rad = min(max(0.0, closure_rad), abs(closing_span_rad))
    if closure_rad == 0.0:
        return 0.0

    spring_torque_nm = -closing_direction * stiffness_nm_per_rad * closure_rad
    damping_torque_nm = -damping_nm_s_per_rad * velocity_rad_s
    torque_nm = spring_torque_nm + damping_torque_nm
    return min(max(torque_nm, -max_torque_nm), max_torque_nm)
