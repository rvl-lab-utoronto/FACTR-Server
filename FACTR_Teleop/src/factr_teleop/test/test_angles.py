import numpy as np
import pytest

from factr_teleop.angles import periodic_joint_error


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
