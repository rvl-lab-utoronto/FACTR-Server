import json

import numpy as np
import pytest

from factr_teleop.leader_config import apply_control_gains, load_leader_model


def _contract():
    return {
        "side": "left",
        "dynamixel_id_start": 9,
        "leader_urdf": "experimental-left.urdf",
        "connected_joint_indices": [0, 1],
        "joint_torque_gain": [0.25, 0.5],
        "raw_to_dfc": {
            "offsets_deg": [10.0, -20.0],
            "sign_flip_joints": [1],
            "drop_trailing": 1,
            "gripper_open": 1.0,
            "gripper_closed": 2.0,
        },
        "dfc_to_factr": {
            "signs": [1.0, -1.0],
        },
        "control_gains": {
            "grav_comp_gain_ramp_time": 1.0,
            "force_feedback_gain_ramp_time": 2.0,
            "null_space_gain_ramp_time": 3.0,
            "gravity_comp": {"gain": 0.8},
            "torque_feedback": {
                "gain": 3.0,
                "damping": 0.1,
                "motor_scalar": 94.5,
                "max_torque": 20.0,
            },
            "static_friction_comp": {"gain": [0.2, 0.4]},
            "joint_limit_barrier": {"kp": 0.3, "kd": 0.4},
            "null_space_regulation": {
                "kp": 0.5,
                "kd": 0.6,
                "torque_gain": [1.5, 1.0],
            },
            "joint_position_control": {"kp": 0.7, "kd": 0.8},
            "gripper_spring": {
                "stiffness_nm_per_rad": 0.08,
                "damping_nm_s_per_rad": 0.002,
                "max_torque_nm": 0.05,
            },
        },
    }


MODEL_HOME = np.array([0.4, -0.6])


def test_derives_model_from_dfc_contract():
    env = {"DFC_LEADER_CONFIG": json.dumps(_contract())}
    model = load_leader_model("left", 2, MODEL_HOME, env)
    assert model.dynamixel_id_start == 9
    assert model.leader_urdf == "experimental-left.urdf"
    np.testing.assert_array_equal(model.connected_joint_indices, [0, 1])
    np.testing.assert_array_equal(model.joint_torque_gain, [0.25, 0.5])
    np.testing.assert_array_equal(
        model.control_gains.static_friction_gain, [0.2, 0.4]
    )
    np.testing.assert_array_equal(
        model.control_gains.null_space_torque_gain, [1.5, 1.0]
    )
    assert model.control_gains.force_feedback_max_torque == 20.0
    np.testing.assert_array_equal(model.scale_joint_torque([2.0, 3.0]), [0.5, 1.5])
    q, dq = model.model_state([0.5, 0.7], [0.1, -0.2])
    q_dfc = np.array([0.5 + np.radians(10), -(0.7 + np.radians(-20))])
    np.testing.assert_allclose(q, np.array([1.0, -1.0]) * q_dfc)
    np.testing.assert_allclose(dq, [0.1, -0.2])
    np.testing.assert_allclose(
        model.dfc_position_to_model([0.25, -0.5]), [0.25, 0.5]
    )


def test_partial_hardware_expands_missing_joints_and_drops_absent_commands():
    contract = _contract()
    contract["connected_joint_indices"] = [0]
    model = load_leader_model(
        "left", 2, MODEL_HOME,
        {"DFC_LEADER_CONFIG": json.dumps(contract)},
    )

    position, velocity = model.expand_hardware_state([1.2], [0.3])
    synthetic = model.synthetic_raw_position()
    np.testing.assert_allclose(position, [1.2, synthetic[1], 1.0])
    np.testing.assert_allclose(velocity, [0.3, 0.0, 0.0])
    np.testing.assert_allclose(model.select_hardware_values([10, 20, 30]), [10])
    np.testing.assert_allclose(
        model.expand_hardware_values([5.0]), [5.0, 0.0, 0.0]
    )


def test_full_hardware_remains_default_when_partial_indices_are_absent():
    contract = _contract()
    contract.pop("connected_joint_indices")
    model = load_leader_model(
        "left", 2, MODEL_HOME,
        {"DFC_LEADER_CONFIG": json.dumps(contract)},
    )
    assert model.connected_joint_indices is None
    np.testing.assert_array_equal(model.hardware_motor_indices(), [0, 1, 2])


def test_dfc_calibration_and_model_signs_compose_directly():
    contract = _contract()
    env = {"DFC_LEADER_CONFIG": json.dumps(contract)}
    model = load_leader_model("left", 2, MODEL_HOME, env)

    offsets = np.radians(contract["raw_to_dfc"]["offsets_deg"])
    raw_signs = np.array([1.0, -1.0])
    expected_signs = raw_signs * np.array(contract["dfc_to_factr"]["signs"])
    np.testing.assert_allclose(model.model_signs, expected_signs)
    np.testing.assert_allclose(model.motor_signs(-1), [*expected_signs, -1])
    raw_q = np.array([0.2, 0.3])
    q_dfc = raw_signs * (raw_q + offsets)
    q, dq = model.model_state(raw_q, [0.4, 0.5])
    np.testing.assert_allclose(
        q, np.asarray(contract["dfc_to_factr"]["signs"]) * q_dfc
    )
    np.testing.assert_allclose(dq, expected_signs * [0.4, 0.5])


def test_left_zero_indexed_j6_direction_comes_from_dfc():
    gains = _contract()["control_gains"]
    gains["static_friction_comp"]["gain"] = 0.2
    gains["null_space_regulation"]["torque_gain"] = [1.5] + [1.0] * 6
    contract = {
        "side": "left",
        "dynamixel_id_start": 9,
        "joint_torque_gain": [1.0] * 7,
        "raw_to_dfc": {
            "offsets_deg": [0.0] * 7,
            "sign_flip_joints": [5],
            "drop_trailing": 1,
            "gripper_open": 1.0,
            "gripper_closed": 2.0,
        },
        "dfc_to_factr": {"signs": [1.0] * 7},
        "control_gains": gains,
    }
    env = {"DFC_LEADER_CONFIG": json.dumps(contract)}
    model = load_leader_model("left", 7, [0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0], env)

    assert model.model_signs[5] == -1.0
    assert model.model_signs[6] == 1.0
    assert model.motor_signs(1.0)[6] == 1.0


def test_contract_is_required():
    with pytest.raises(RuntimeError, match="required"):
        load_leader_model("left", 2, MODEL_HOME, {})


@pytest.mark.parametrize(
    "value, message",
    [
        ([True], "must contain 2 arm joints"),
        ([0.0, -0.1], "values must be in \\[0, 2\\]"),
        ([0.0, 2.1], "values must be in \\[0, 2\\]"),
    ],
)
def test_joint_torque_gain_contract_is_strict(value, message):
    contract = _contract()
    contract["joint_torque_gain"] = value

    with pytest.raises(RuntimeError, match=message):
        load_leader_model(
            "left", 2, MODEL_HOME,
            {"DFC_LEADER_CONFIG": json.dumps(contract)},
        )


def test_joint_torque_gain_accepts_zero_through_two():
    contract = _contract()
    contract["joint_torque_gain"] = [0.0, 2.0]

    model = load_leader_model(
        "left", 2, MODEL_HOME,
        {"DFC_LEADER_CONFIG": json.dumps(contract)},
    )

    np.testing.assert_array_equal(model.scale_joint_torque([3.0, 3.0]), [0.0, 6.0])


def test_dfc_control_gains_override_factr_yaml():
    model = load_leader_model(
        "left", 2, MODEL_HOME,
        {"DFC_LEADER_CONFIG": json.dumps(_contract())},
    )
    controller = {
        "gravity_comp": {"enable": True, "gain": 99.0},
        "torque_feedback": {"enable": True, "gain": 99.0, "max_torque": 99.0},
        "static_friction_comp": {"enable_speed": 0.9, "gain": 99.0},
        "joint_limit_barrier": {"enable": False, "kp": 99.0, "kd": 99.0},
        "null_space_regulation": {
            "enable": False,
            "kp": 99.0,
            "kd": 99.0,
            "torque_gain": [99.0, 99.0],
        },
        "joint_position_control": {"kp": 99.0, "kd": 99.0},
        "gripper_spring": {
            "enable": False,
            "stiffness_nm_per_rad": 99.0,
            "damping_nm_s_per_rad": 99.0,
            "max_torque_nm": 99.0,
        },
    }

    apply_control_gains(controller, model.control_gains)

    assert controller["gravity_comp"]["gain"] == 0.8
    assert controller["torque_feedback"]["gain"] == 3.0
    assert controller["torque_feedback"]["max_torque"] == 20.0
    assert controller["force_feedback_gain_ramp_time"] == 2.0
    assert controller["static_friction_comp"]["gain"] == [0.2, 0.4]
    assert controller["null_space_regulation"] == {
        "enable": False,
        "gain_ramp_time": 3.0,
        "kp": 0.5,
        "kd": 0.6,
        "torque_gain": [1.5, 1.0],
    }
    assert controller["gripper_spring"] == {
        "enable": False,
        "stiffness_nm_per_rad": 0.08,
        "damping_nm_s_per_rad": 0.002,
        "max_torque_nm": 0.05,
    }


def test_legacy_scalar_static_friction_gain_is_broadcast_per_joint():
    contract = _contract()
    contract["control_gains"]["static_friction_comp"]["gain"] = 0.3

    model = load_leader_model(
        "left", 2, MODEL_HOME,
        {"DFC_LEADER_CONFIG": json.dumps(contract)},
    )

    np.testing.assert_allclose(model.control_gains.static_friction_gain, [0.3, 0.3])


def test_missing_null_space_torque_gain_defaults_to_neutral():
    contract = _contract()
    contract["control_gains"]["null_space_regulation"].pop("torque_gain")

    model = load_leader_model(
        "left", 2, MODEL_HOME,
        {"DFC_LEADER_CONFIG": json.dumps(contract)},
    )

    np.testing.assert_allclose(model.control_gains.null_space_torque_gain, [1.0, 1.0])


@pytest.mark.parametrize("gain", ([0.1], [0.1, -0.2]))
def test_static_friction_gain_contract_is_strict(gain):
    contract = _contract()
    contract["control_gains"]["static_friction_comp"]["gain"] = gain

    with pytest.raises(RuntimeError, match="static_friction|non-negative"):
        load_leader_model(
            "left", 2, MODEL_HOME,
            {"DFC_LEADER_CONFIG": json.dumps(contract)},
        )


def test_null_space_torque_gain_contract_is_strict():
    contract = _contract()
    contract["control_gains"]["null_space_regulation"]["torque_gain"] = [1.0]

    with pytest.raises(RuntimeError, match="null_space_regulation.torque_gain"):
        load_leader_model(
            "left", 2, MODEL_HOME,
            {"DFC_LEADER_CONFIG": json.dumps(contract)},
        )


def test_persisted_offset_is_rejected():
    contract = _contract()
    contract["dfc_to_factr"]["offset_rad"] = [9.0, 9.0]
    env = {"DFC_LEADER_CONFIG": json.dumps(contract)}
    with pytest.raises(RuntimeError, match="must not be persisted"):
        load_leader_model("left", 2, MODEL_HOME, env)
