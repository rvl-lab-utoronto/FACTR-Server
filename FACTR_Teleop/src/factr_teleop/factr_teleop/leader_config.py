"""DFC-owned leader calibration consumed by FACTR at process launch.

The persisted source of truth lives in DFC. FACTR receives one JSON contract in
``DFC_LEADER_CONFIG`` and derives the coordinates required by its dynamics model.
"""

from dataclasses import dataclass
import json
import os

import numpy as np


@dataclass(frozen=True)
class LeaderModelConfig:
    side: str
    raw_offsets_rad: np.ndarray
    raw_signs: np.ndarray
    wrap: bool
    dfc_to_factr_signs: np.ndarray
    dfc_to_factr_offset_rad: np.ndarray
    home_factr_q_rad: np.ndarray
    model_signs: np.ndarray
    joint_offsets_rad: np.ndarray

    def model_state(self, raw_q, raw_dq):
        """Convert raw Dynamixel state using FACTR's native mechanism signs."""
        raw_q = np.asarray(raw_q, dtype=float)
        raw_dq = np.asarray(raw_dq, dtype=float)
        q_factr = self.model_signs * (raw_q - self.joint_offsets_rad)
        dq_factr = self.model_signs * raw_dq
        return q_factr, dq_factr


def load_leader_model(
    expected_side, dof, physical_model_signs, model_home_q_rad, environ=None
):
    """Parse DFC's measured contract and derive FACTR model calibration.

    ``model_home_q_rad`` is owned by FACTR's mechanism configuration. The affine
    DFC→FACTR offset is derived here and is never accepted as persisted input.
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
        home_dfc = np.asarray(data["home_q_rad"], dtype=float)
        drop = int(raw["drop_trailing"])
        gripper_open = float(raw["gripper_open"])
        gripper_closed = float(raw["gripper_closed"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid DFC_LEADER_CONFIG: {exc}") from exc

    home_factr = np.asarray(model_home_q_rad, dtype=float)
    vectors = (offsets, raw_signs, transform_signs, home_dfc, home_factr)
    if side != expected_side:
        raise RuntimeError(
            f"DFC leader contract is for {side!r}, expected {expected_side!r}"
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

    physical_model_signs = np.asarray(physical_model_signs, dtype=float)[:dof]
    if physical_model_signs.shape != (dof,) or not np.all(
        np.isin(physical_model_signs, (-1.0, 1.0))
    ):
        raise RuntimeError(
            "FACTR mechanism signs must contain one -1 or +1 per arm joint"
        )

    transform_offset = home_factr - transform_signs * home_dfc
    home_raw = raw_signs * home_dfc - offsets
    # FACTR owns the mechanism signs; DFC supplies the captured physical home.
    # q_model = model_signs * (q_raw - joint_offsets).
    model_signs = physical_model_signs
    joint_offsets = home_raw - home_factr / model_signs
    return LeaderModelConfig(
        side=side,
        raw_offsets_rad=offsets,
        raw_signs=raw_signs,
        wrap=bool(raw["wrap_deg"]),
        dfc_to_factr_signs=transform_signs,
        dfc_to_factr_offset_rad=transform_offset,
        home_factr_q_rad=home_factr,
        model_signs=model_signs,
        joint_offsets_rad=joint_offsets,
    )
