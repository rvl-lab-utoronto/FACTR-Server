import numpy as np
import pytest

from factr_teleop.angles import periodic_joint_error, resolve_null_space_target


def test_periodic_joint_error_ignores_encoder_turn_branches():
    physical = np.array([0.01, -0.03, -0.09, 1.53, 0.20, -0.05, 0.10])
    branches = physical + 2.0 * np.pi * np.array([-1, 0, 1, 0, 1, 0, 0])

    np.testing.assert_allclose(
        periodic_joint_error(branches, np.zeros(7)),
        physical,
        atol=1e-12,
    )


def test_periodic_joint_error_rejects_nonfinite_input():
    with pytest.raises(ValueError, match="non-finite"):
        periodic_joint_error([0.0, np.nan], [0.0, 0.0])


def test_null_space_target_defaults_to_independent_model_home_copy():
    home = np.array([0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0])
    target = resolve_null_space_target({}, home, 7)
    np.testing.assert_array_equal(target, home)
    target[3] = 0.0
    assert home[3] == 1.57


def test_null_space_target_rejects_trailing_gripper_and_nonfinite_values():
    home = np.zeros(7)
    with pytest.raises(ValueError, match="exactly 7 arm joints"):
        resolve_null_space_target({"null_space_joint_target": np.zeros(8)}, home, 7)
    with pytest.raises(ValueError, match="non-finite"):
        resolve_null_space_target(
            {"null_space_joint_target": [0, 0, 0, 0, 0, 0, np.nan]}, home, 7
        )
