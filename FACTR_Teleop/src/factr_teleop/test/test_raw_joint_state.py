"""Tests for sharing one control-loop Dynamixel read with ROS publishers."""

import numpy as np
import pytest

from factr_teleop.raw_joint_state import RawJointStateCache


def test_cache_returns_defensive_copy_of_one_control_sample():
    cache = RawJointStateCache(3)
    position = np.array([1.0, 2.0, 3.0])
    velocity = np.array([0.1, 0.2, 0.3])

    written = cache.update(position, velocity)
    position[0] = 99.0
    written.position[1] = 88.0
    published = cache.snapshot()

    assert published.sequence == 1
    assert published.stamp_monotonic_ns > 0
    np.testing.assert_array_equal(published.position, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(published.velocity, [0.1, 0.2, 0.3])


def test_cache_sequence_advances_once_per_update():
    cache = RawJointStateCache(2)

    first = cache.update([0.0, 1.0], [2.0, 3.0])
    second = cache.update([4.0, 5.0], [6.0, 7.0])

    assert first.sequence == 1
    assert second.sequence == 2
    np.testing.assert_array_equal(cache.snapshot().position, [4.0, 5.0])


def test_cache_rejects_wrong_motor_count():
    cache = RawJointStateCache(2)

    with pytest.raises(ValueError, match="position shape"):
        cache.update([1.0], [2.0, 3.0])
