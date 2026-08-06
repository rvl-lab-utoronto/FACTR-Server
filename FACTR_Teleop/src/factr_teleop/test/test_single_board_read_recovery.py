"""Regression tests for bounded Dynamixel reads on the single-board leader."""

from types import SimpleNamespace

import numpy as np
import pytest

from factr_teleop.dynamixel.driver import DynamixelReadError
from factr_teleop.factr_rizon_teleop import FactrRizonTeleop
from factr_teleop.factr_teleop import FACTRTeleop
from factr_teleop.raw_joint_state import RawJointState, RawJointStateCache


class _Logger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def warning(self, message, **kwargs):
        self.warnings.append((message, kwargs))

    def info(self, message, **kwargs):
        self.infos.append((message, kwargs))


def test_failed_control_read_skips_tick_without_commanding_torque():
    logger = _Logger()

    class Teleop:
        name = "left"

        @staticmethod
        def get_leader_joint_states():
            raise DynamixelReadError("test timeout")

        @staticmethod
        def get_logger():
            return logger

        @staticmethod
        def set_leader_joint_torque(*_args):
            raise AssertionError("a failed read must not produce a torque command")

    FACTRTeleop.control_loop_callback(Teleop())

    assert len(logger.warnings) == 1
    assert "skipped control tick" in logger.warnings[0][0]
    assert logger.warnings[0][1]["throttle_duration_sec"] == 1.0


def test_startup_collects_eleven_successes_across_isolated_misses():
    position = np.arange(8, dtype=float)
    velocity = np.zeros(8)

    class Driver:
        calls = 0

        def get_positions_and_velocities(self, *, source):
            assert source == "startup"
            self.calls += 1
            if self.calls in (1, 4):
                raise DynamixelReadError("test timeout")
            return position, velocity

    class LeaderModel:
        joint_offsets_rad = np.zeros(7)

        @staticmethod
        def model_state(q, dq):
            return q, dq

    teleop = SimpleNamespace(
        driver=Driver(),
        dt=0.0,
        _raw_joint_state_cache=RawJointStateCache(8),
        _leader_model=LeaderModel(),
        num_arm_joints=7,
    )

    FACTRTeleop._get_dynamixel_offsets(teleop, verbose=False)

    assert teleop.driver.calls == 13
    assert teleop._raw_joint_state_cache.snapshot().sequence == 1
    np.testing.assert_array_equal(
        teleop._raw_joint_state_cache.snapshot().position, position
    )


def test_startup_failure_is_bounded_by_deadline(monkeypatch):
    class Driver:
        @staticmethod
        def get_positions_and_velocities(*, source):
            assert source == "startup"
            raise DynamixelReadError("test timeout")

    monotonic_values = iter((10.0, 15.0))
    monkeypatch.setattr(
        "factr_teleop.factr_teleop.time.monotonic",
        lambda: next(monotonic_values),
    )
    teleop = SimpleNamespace(driver=Driver(), dt=0.0)

    with pytest.raises(
        DynamixelReadError,
        match="startup did not produce 11 complete samples within 5 seconds",
    ):
        FACTRTeleop._get_dynamixel_offsets(teleop, verbose=False)


def test_publisher_emits_each_successful_acquisition_once(monkeypatch):
    state = RawJointState(
        position=np.arange(8, dtype=float),
        velocity=np.zeros(8),
        sequence=3,
        stamp_monotonic_ns=1_500_000_000,
    )
    published = []
    logger = _Logger()
    monkeypatch.setattr(
        "factr_teleop.factr_rizon_teleop.time.monotonic_ns",
        lambda: 2_000_000_000,
    )

    class Teleop:
        index = 0
        _last_published_joint_sequence = 2
        joint_pos_publisher = SimpleNamespace(publish=published.append)

        @staticmethod
        def get_cached_raw_joint_state():
            return state

        @staticmethod
        def get_clock():
            return SimpleNamespace(
                now=lambda: SimpleNamespace(nanoseconds=10_000_000_000)
            )

        @staticmethod
        def get_logger():
            return logger

    teleop = Teleop()
    FactrRizonTeleop.publish_joint_pos(teleop)
    FactrRizonTeleop.publish_joint_pos(teleop)

    assert len(published) == 1
    assert published[0].header.stamp.sec == 9
    assert published[0].header.stamp.nanosec == 500_000_000
    assert teleop._last_published_joint_sequence == state.sequence
