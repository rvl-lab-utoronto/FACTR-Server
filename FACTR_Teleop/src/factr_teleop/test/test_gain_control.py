import numpy as np

from factr_teleop.gain_control import compose_arm_torque
from factr_teleop.gain_control import ramp_gain


def test_gravity_and_feedback_gains_are_independent():
    limit = np.array([1.0, 1.0])
    null = np.array([0.5, 0.5])
    gravity = np.array([10.0, 20.0])
    friction = np.array([2.0, 4.0])
    feedback = np.array([100.0, 200.0])

    no_gravity = compose_arm_torque(
        limit, null, gravity, friction, feedback, 0.0, 1.0
    )
    no_feedback = compose_arm_torque(
        limit, null, gravity, friction, feedback, 1.0, 0.0
    )

    np.testing.assert_allclose(no_gravity, limit + null + feedback)
    np.testing.assert_allclose(no_feedback, limit + null + gravity + friction)


def test_gain_ramp_clamps_and_keeps_its_own_clock():
    gain, stamp = ramp_gain(0.0, 1.0, 10.0, 2.0, 10.5)
    assert gain == 0.25
    assert stamp == 10.5
    gain, stamp = ramp_gain(gain, 0.0, stamp, 2.0, 11.5)
    assert gain == 0.0
    assert stamp == 11.5
