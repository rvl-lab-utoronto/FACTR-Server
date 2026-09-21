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


def test_null_space_control_uses_commanded_dfc_target_not_current_leader_pose():
    logger = _Logger()
    teleop = SimpleNamespace(
        name="left",
        num_arm_joints=3,
        null_space_regulation_enable=True,
        null_space_joint_target=np.zeros(3),
        null_space_gain_target=0.0,
        _leader_model=SimpleNamespace(
            dfc_position_to_model=lambda q: np.asarray([1.0, -1.0, 1.0]) * q
        ),
        get_logger=lambda: logger,
    )
    task_target = [0.1, -0.2, 0.3]

    FACTRTeleop._on_null_space_control(
        teleop, SimpleNamespace(data=[1.0, *task_target])
    )

    np.testing.assert_allclose(teleop.null_space_joint_target, [0.1, 0.2, 0.3])
    assert teleop.null_space_gain_target == 1.0

    FACTRTeleop._on_null_space_control(
        teleop, SimpleNamespace(data=[0.0, *task_target])
    )
    assert teleop.null_space_gain_target == 0.0


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

        @staticmethod
        def _disable_leader_torque_after_read_failure():
            pass

    FACTRTeleop.control_loop_callback(Teleop())

    assert len(logger.warnings) == 1
    assert "skipped control tick" in logger.warnings[0][0]
    assert logger.warnings[0][1]["throttle_duration_sec"] == 1.0


def test_single_board_master_off_writes_zero_then_disables_hardware():
    order = []

    class Driver:
        torque_enabled = True

        def set_torque(self, torque):
            np.testing.assert_array_equal(torque, np.zeros(3))
            order.append("zero")

        def set_torque_mode(self, enabled):
            assert enabled is False
            self.torque_enabled = False
            order.append("disable")

    teleop = SimpleNamespace(
        driver=Driver(),
        num_motors=8,
        _hardware_motor_indices=np.array([0, 2, 7]),
        num_arm_joints=2,
        leader_torque_gain=0.0,
        leader_torque_gain_target=0.0,
        leader_torque_enabled=True,
    )
    teleop._set_leader_torque_enabled = lambda enabled: (
        FACTRTeleop._set_leader_torque_enabled(teleop, enabled)
    )
    teleop.set_leader_joint_torque = lambda arm, gripper: (
        order.append("command") or np.asarray(arm, dtype=float)
    )

    commanded = FACTRTeleop._apply_leader_torque(
        teleop, np.zeros(2), 0.0
    )

    np.testing.assert_array_equal(commanded, np.zeros(2))
    assert order == ["zero", "disable"]
    assert teleop.leader_torque_enabled is False


def test_single_board_master_off_attempts_disable_after_zero_write_failure():
    order = []

    class Driver:
        torque_enabled = True

        def set_torque(self, _torque):
            order.append("zero")
            raise RuntimeError("write failed")

        def set_torque_mode(self, enabled):
            assert enabled is False
            order.append("disable")
            self.torque_enabled = False

    teleop = SimpleNamespace(
        driver=Driver(),
        num_motors=8,
        _hardware_motor_indices=np.array([0, 2, 7]),
        leader_torque_enabled=True,
    )

    with pytest.raises(RuntimeError, match="zero torque failed"):
        FACTRTeleop._set_leader_torque_enabled(teleop, False)

    assert order == ["zero", "disable"]
    assert teleop.leader_torque_enabled is False


def test_single_board_scales_each_joint_before_motor_write():
    class Driver:
        command = None

        def set_torque(self, torque):
            self.command = np.asarray(torque, dtype=float)
            return self.command

    driver = Driver()
    teleop = SimpleNamespace(
        driver=driver,
        num_arm_joints=2,
        _motor_joint_signs=np.array([-1.0, 1.0, 1.0]),
        _leader_model=SimpleNamespace(
            scale_joint_torque=lambda torque: np.asarray(torque, dtype=float)
            * [0.0, 0.5],
            select_hardware_values=lambda values: np.asarray(values, dtype=float),
            expand_hardware_values=lambda values: np.asarray(values, dtype=float),
        ),
    )

    commanded = FACTRTeleop.set_leader_joint_torque(teleop, [2.0, 4.0], 3.0)

    np.testing.assert_array_equal(driver.command, [0.0, 2.0, 3.0])
    np.testing.assert_array_equal(commanded, [0.0, 2.0])


def test_static_friction_compensation_uses_per_joint_gains():
    teleop = SimpleNamespace(
        num_arm_joints=3,
        stiction_comp_enable_speed=1.0,
        stiction_comp_gain=np.array([0.0, 0.5, 1.5]),
        stiction_dither_flag=np.ones(3, dtype=bool),
        tau_g=np.array([2.0, -4.0, 6.0]),
    )

    torque = FACTRTeleop.friction_compensation(teleop, [0.0, 0.0, 0.0])

    np.testing.assert_allclose(torque, [0.0, 2.0, 9.0])
    np.testing.assert_array_equal(teleop.stiction_dither_flag, [False] * 3)


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

        @staticmethod
        def expand_hardware_state(position, velocity):
            return position, velocity

    teleop = SimpleNamespace(
        driver=Driver(),
        dt=0.0,
        _raw_joint_state_cache=RawJointStateCache(8),
        _leader_model=LeaderModel(),
        num_arm_joints=7,
    )
    teleop._read_full_joint_state = lambda source: FACTRTeleop._read_full_joint_state(
        teleop, source
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
    teleop._read_full_joint_state = lambda source: FACTRTeleop._read_full_joint_state(
        teleop, source
    )

    with pytest.raises(
        DynamixelReadError,
        match="startup did not produce 11 complete samples within 5 seconds",
    ):
        FACTRTeleop._get_dynamixel_offsets(teleop, verbose=False)


def test_partial_single_board_routes_only_connected_joint_commands():
    class Driver:
        command = None

        def set_torque(self, torque):
            self.command = np.asarray(torque, dtype=float)
            return self.command

    class LeaderModel:
        @staticmethod
        def scale_joint_torque(torque):
            return np.asarray(torque, dtype=float)

        @staticmethod
        def select_hardware_values(values):
            return np.asarray(values, dtype=float)[:3]

        @staticmethod
        def expand_hardware_values(values):
            return np.pad(np.asarray(values, dtype=float), (0, 5))

    teleop = SimpleNamespace(
        driver=Driver(),
        num_arm_joints=7,
        _motor_joint_signs=np.ones(8),
        _leader_model=LeaderModel(),
    )

    commanded = FACTRTeleop.set_leader_joint_torque(
        teleop, [1, 2, 3, 4, 5, 6, 7], 8
    )

    np.testing.assert_array_equal(teleop.driver.command, [1, 2, 3])
    np.testing.assert_array_equal(commanded, [1, 2, 3, 0, 0, 0, 0])


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
        _last_commanded_torque_nm = np.arange(7, dtype=float) / 10.0
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
    np.testing.assert_allclose(published[0].effort, teleop._last_commanded_torque_nm)
    assert teleop._last_published_joint_sequence == state.sequence
