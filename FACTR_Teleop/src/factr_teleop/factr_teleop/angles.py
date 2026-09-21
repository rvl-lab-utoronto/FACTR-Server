"""Strict angular-coordinate helpers used by FACTR control laws."""

from __future__ import annotations

import numpy as np


def resolve_null_space_target(config, model_home, dof) -> np.ndarray:
    """Resolve a strict FACTR-model rest posture for null-space control.

    Omitting ``null_space_joint_target`` selects the dynamics model's calibrated
    home.  An explicit target is allowed for task-specific redundancy resolution,
    but it must contain exactly one finite value per arm joint; a trailing gripper
    value is not part of the arm model and must not be silently ignored.
    """
    home = np.asarray(model_home, dtype=float)
    if home.shape != (dof,) or not np.all(np.isfinite(home)):
        raise ValueError(
            f"model home must contain exactly {dof} finite arm joints, got {home.shape}"
        )

    configured = config.get("null_space_joint_target")
    if configured is None:
        return home.copy()

    target = np.asarray(configured, dtype=float)
    if target.shape != (dof,):
        raise ValueError(
            f"null-space target must contain exactly {dof} arm joints, got {target.shape}"
        )
    if not np.all(np.isfinite(target)):
        raise ValueError("null-space target contains a non-finite value")
    return target


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
