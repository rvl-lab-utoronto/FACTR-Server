import json

import numpy as np
import pytest

from factr_teleop.leader_config import load_leader_model


def _contract():
    return {
        "side": "left",
        "raw_to_dfc": {
            "offsets_deg": [10.0, -20.0],
            "sign_flip_joints": [1],
            "wrap_deg": True,
            "drop_trailing": 1,
            "gripper_open": 1.0,
            "gripper_closed": 2.0,
        },
        "home_q_rad": [0.1, 0.2],
        "dfc_to_factr": {
            "signs": [1.0, -1.0],
            "offset_rad": [0.3, -0.4],
        },
    }


def test_derives_model_from_dfc_contract():
    env = {"DFC_LEADER_CONFIG": json.dumps(_contract())}
    model = load_leader_model("left", 2, [1.0, 1.0], env)
    q, dq = model.model_state([0.5, 0.7], [0.1, -0.2])
    q_dfc = np.array([0.5 + np.radians(10), -(0.7 + np.radians(-20))])
    np.testing.assert_allclose(q, np.array([1.0, -1.0]) * q_dfc + [0.3, -0.4])
    np.testing.assert_allclose(dq, [0.1, -0.2])


def test_factr_native_signs_override_dfc_model_slopes_at_same_home():
    contract = _contract()
    env = {"DFC_LEADER_CONFIG": json.dumps(contract)}
    physical_signs = np.array([1.0, -1.0])
    model = load_leader_model("left", 2, physical_signs, env)

    offsets = np.radians(contract["raw_to_dfc"]["offsets_deg"])
    raw_signs = np.array([1.0, -1.0])
    home_dfc = np.asarray(contract["home_q_rad"])
    home_factr = (
        np.asarray(contract["dfc_to_factr"]["signs"]) * home_dfc
        + np.asarray(contract["dfc_to_factr"]["offset_rad"])
    )
    raw_home = raw_signs * home_dfc - offsets

    q_home, _ = model.model_state(raw_home, [0.0, 0.0])
    np.testing.assert_allclose(q_home, home_factr)
    np.testing.assert_allclose(model.model_signs, physical_signs)
    q, dq = model.model_state(raw_home + [0.2, 0.3], [0.4, 0.5])
    np.testing.assert_allclose(q, home_factr + physical_signs * [0.2, 0.3])
    np.testing.assert_allclose(dq, physical_signs * [0.4, 0.5])


def test_contract_is_required():
    with pytest.raises(RuntimeError, match="required"):
        load_leader_model("left", 2, [1.0, 1.0], {})
