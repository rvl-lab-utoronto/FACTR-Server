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
            "drop_trailing": 1,
            "gripper_open": 1.0,
            "gripper_closed": 2.0,
        },
        "home_q_rad": [0.1, 0.2],
        "dfc_to_factr": {
            "signs": [1.0, -1.0],
        },
    }


MODEL_HOME = np.array([0.4, -0.6])


def test_derives_model_from_dfc_contract():
    env = {"DFC_LEADER_CONFIG": json.dumps(_contract())}
    model = load_leader_model("left", 2, MODEL_HOME, env)
    q, dq = model.model_state([0.5, 0.7], [0.1, -0.2])
    q_dfc = np.array([0.5 + np.radians(10), -(0.7 + np.radians(-20))])
    derived_offset = MODEL_HOME - np.array([1.0, -1.0]) * [0.1, 0.2]
    np.testing.assert_allclose(
        q, np.array([1.0, -1.0]) * q_dfc + derived_offset
    )
    np.testing.assert_allclose(dq, [0.1, -0.2])


def test_dfc_signs_own_model_and_motor_slopes_at_same_home():
    contract = _contract()
    env = {"DFC_LEADER_CONFIG": json.dumps(contract)}
    model = load_leader_model("left", 2, MODEL_HOME, env)

    offsets = np.radians(contract["raw_to_dfc"]["offsets_deg"])
    raw_signs = np.array([1.0, -1.0])
    home_dfc = np.asarray(contract["home_q_rad"])
    home_factr = MODEL_HOME
    raw_home = raw_signs * home_dfc - offsets

    q_home, _ = model.model_state(raw_home, [0.0, 0.0])
    np.testing.assert_allclose(q_home, home_factr)
    expected_signs = raw_signs * np.array(contract["dfc_to_factr"]["signs"])
    np.testing.assert_allclose(model.model_signs, expected_signs)
    np.testing.assert_allclose(model.motor_signs(-1), [*expected_signs, -1])
    q, dq = model.model_state(raw_home + [0.2, 0.3], [0.4, 0.5])
    np.testing.assert_allclose(q, home_factr + expected_signs * [0.2, 0.3])
    np.testing.assert_allclose(dq, expected_signs * [0.4, 0.5])


def test_left_zero_indexed_j6_direction_comes_from_dfc():
    contract = {
        "side": "left",
        "raw_to_dfc": {
            "offsets_deg": [0.0] * 7,
            "sign_flip_joints": [5],
            "drop_trailing": 1,
            "gripper_open": 1.0,
            "gripper_closed": 2.0,
        },
        "home_q_rad": [0.0] * 7,
        "dfc_to_factr": {"signs": [1.0] * 7},
    }
    env = {"DFC_LEADER_CONFIG": json.dumps(contract)}
    model = load_leader_model("left", 7, [0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0], env)

    assert model.model_signs[5] == -1.0
    assert model.model_signs[6] == 1.0
    assert model.motor_signs(1.0)[6] == 1.0


def test_contract_is_required():
    with pytest.raises(RuntimeError, match="required"):
        load_leader_model("left", 2, MODEL_HOME, {})


def test_persisted_offset_is_rejected():
    contract = _contract()
    contract["dfc_to_factr"]["offset_rad"] = [9.0, 9.0]
    env = {"DFC_LEADER_CONFIG": json.dumps(contract)}
    with pytest.raises(RuntimeError, match="must not be persisted"):
        load_leader_model("left", 2, MODEL_HOME, env)
