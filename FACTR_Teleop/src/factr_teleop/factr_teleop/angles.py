"""Strict angular-coordinate helpers used by FACTR control laws."""

from __future__ import annotations

import numpy as np


def periodic_joint_error(position, target) -> np.ndarray:
    """Return the shortest signed rotary-joint error in ``[-pi, pi]``.

    DYNAMIXEL Present Position may select a different multi-turn branch after a
    reboot.  The physical configuration is unchanged by adding ``2*pi*k``, so a
    linear controller must never react to that branch choice.  Reject malformed
    inputs rather than producing an unsafe torque from them.
    """
    q = np.asarray(position, dtype=float)
    q_target = np.asarray(target, dtype=float)
    if q.shape != q_target.shape:
        raise ValueError(
            f"joint position/target shape mismatch: {q.shape} != {q_target.shape}"
        )
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(q_target)):
        raise ValueError("joint position/target contains a non-finite value")
    delta = q - q_target
    return np.arctan2(np.sin(delta), np.cos(delta))
