"""Regression tests for the dual-board leader's dedicated control thread."""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from factr_teleop.dynamixel.driver import DynamixelReadError
from factr_teleop.factr_rizon_dual_board import FactrRizonTeleopDualBoard
from factr_teleop.factr_teleop_dual_base import FACTRTeleopDualBase


class _Logger:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, message, **kwargs):
        self.errors.append((message, kwargs))

    def warning(self, message, **kwargs):
        self.warnings.append((message, kwargs))


class _Context:
    def __init__(self):
        self.shutdown = threading.Event()

    def try_shutdown(self):
        self.shutdown.set()


class _ThreadHarness:
    start_control_loop = FACTRTeleopDualBase.start_control_loop
    _control_loop_worker = FACTRTeleopDualBase._control_loop_worker
    stop_control_loop = FACTRTeleopDualBase.stop_control_loop

    def __init__(self, callback):
        self.name = self.side = "right"
        self.dt = 0.001
        self._control_stop = threading.Event()
        self._control_thread = None
        self._control_thread_error = None
        self.context = _Context()
        self.logger = _Logger()
        self.callback = callback

    def control_loop_callback(self):
        self.callback()

    def get_logger(self):
        return self.logger


class _ReadDriver:
    def __init__(self, positions, velocities, currents, torques, barrier=None,
                 error=None, finished=None):
        self.positions = np.asarray(positions, dtype=float)
        self.velocities = np.asarray(velocities, dtype=float)
        self.currents = np.asarray(currents, dtype=float)
        self.torques = np.asarray(torques, dtype=float)
        self.barrier = barrier
        self.error = error
        self.finished = finished

    def get_positions_and_velocities(self, source):
        assert source == "test"
        if self.barrier is not None:
            self.barrier.wait(timeout=0.5)
        try:
            if self.error is not None:
                raise self.error
            return self.positions, self.velocities
        finally:
            if self.finished is not None:
                self.finished.set()

    def present_current_raw(self):
        return self.currents

    def current_estimated_torque(self):
        return self.torques


class _ParallelReadHarness:
    _merge_small_big = FACTRTeleopDualBase._merge_small_big
    _read_merged_pos_vel = FACTRTeleopDualBase._read_merged_pos_vel

    def __init__(self, small, big):
        self.driver_small = small
        self.driver_big = big
        self.small_servo_indices = np.array([0, 2])
        self.big_servo_indices = np.array([1])
        self.big_servo_ids = [2]
        self.num_motors = 3
        self.num_arm_joints = 2
        self._motor_joint_signs = np.ones(3)
        self._board_io_executor = ThreadPoolExecutor(max_workers=2)

    def close(self):
        self._board_io_executor.shutdown(wait=True, cancel_futures=True)


class _WriteDriver:
    def __init__(self, barrier, error=None, finished=None):
        self.barrier = barrier
        self.error = error
        self.finished = finished
        self.command = None

    def set_torque(self, torque):
        self.command = np.asarray(torque, dtype=float)
        self.barrier.wait(timeout=0.5)
        try:
            if self.error is not None:
                raise self.error
            return self.command
        finally:
            if self.finished is not None:
                self.finished.set()


class _ParallelWriteHarness:
    _merge_small_big = FACTRTeleopDualBase._merge_small_big
    set_leader_joint_torque = FACTRTeleopDualBase.set_leader_joint_torque

    def __init__(self, small, big, gains=(1.0, 1.0)):
        self.driver_small = small
        self.driver_big = big
        self.small_servo_indices = np.array([0, 2])
        self.big_servo_indices = np.array([1])
        self.big_servo_ids = [2]
        self.num_motors = 3
        self.num_arm_joints = 2
        self._motor_joint_signs = np.ones(3)
        self._leader_model = type(
            "LeaderModel",
            (),
            {
                "scale_joint_torque": staticmethod(
                    lambda torque: gains * np.asarray(torque, dtype=float)
                )
            },
        )()
        self._board_io_executor = ThreadPoolExecutor(max_workers=2)

    def close(self):
        self._board_io_executor.shutdown(wait=True, cancel_futures=True)


def test_dual_board_reads_overlap_and_merge_complete_samples():
    barrier = threading.Barrier(2)
    teleop = _ParallelReadHarness(
        _ReadDriver([1.0, 3.0], [10.0, 30.0], [100, 300], [0.1, 0.3], barrier),
        _ReadDriver([2.0], [20.0], [200], [0.2], barrier),
    )
    try:
        positions, velocities = teleop._read_merged_pos_vel(source="test")
    finally:
        teleop.close()

    np.testing.assert_array_equal(positions, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(velocities, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(teleop._last_present_current_raw, [100, 200])
    np.testing.assert_allclose(
        teleop._last_current_estimated_torque, [0.1, 0.2]
    )


def test_parallel_read_joins_healthy_board_before_raising_failure():
    barrier = threading.Barrier(2)
    big_finished = threading.Event()
    teleop = _ParallelReadHarness(
        _ReadDriver(
            [], [], [], [], barrier, DynamixelReadError("small timeout")
        ),
        _ReadDriver(
            [2.0], [20.0], [200], [0.2], barrier, finished=big_finished
        ),
    )
    try:
        with pytest.raises(DynamixelReadError, match="small timeout"):
            teleop._read_merged_pos_vel(source="test")
        assert big_finished.is_set()
    finally:
        teleop.close()


def test_dual_board_torque_writes_overlap_and_merge_commands():
    barrier = threading.Barrier(2)
    small = _WriteDriver(barrier)
    big = _WriteDriver(barrier)
    teleop = _ParallelWriteHarness(small, big)
    try:
        commanded = teleop.set_leader_joint_torque([1.0, 2.0], 3.0)
    finally:
        teleop.close()

    np.testing.assert_array_equal(small.command, [1.0, 3.0])
    np.testing.assert_array_equal(big.command, [2.0])
    np.testing.assert_array_equal(commanded, [1.0, 2.0])


def test_dual_board_scales_each_joint_before_splitting_writes():
    barrier = threading.Barrier(2)
    small = _WriteDriver(barrier)
    big = _WriteDriver(barrier)
    teleop = _ParallelWriteHarness(small, big, gains=np.array([0.0, 0.5]))
    try:
        commanded = teleop.set_leader_joint_torque([1.0, 2.0], 3.0)
    finally:
        teleop.close()

    np.testing.assert_array_equal(small.command, [0.0, 3.0])
    np.testing.assert_array_equal(big.command, [1.0])
    np.testing.assert_array_equal(commanded, [0.0, 1.0])


def test_parallel_torque_write_joins_healthy_board_before_raising_failure():
    barrier = threading.Barrier(2)
    big_finished = threading.Event()
    teleop = _ParallelWriteHarness(
        _WriteDriver(barrier, RuntimeError("small write failed")),
        _WriteDriver(barrier, finished=big_finished),
    )
    try:
        with pytest.raises(RuntimeError, match="small write failed"):
            teleop.set_leader_joint_torque([1.0, 2.0], 3.0)
        assert big_finished.is_set()
    finally:
        teleop.close()


def test_dedicated_control_thread_runs_and_joins():
    completed = threading.Event()
    iterations = 0

    def callback():
        nonlocal iterations
        iterations += 1
        if iterations >= 3:
            completed.set()

    teleop = _ThreadHarness(callback)
    teleop.start_control_loop()
    thread = teleop._control_thread

    assert completed.wait(0.5)
    teleop.stop_control_loop()

    assert iterations >= 3
    assert not thread.is_alive()
    assert teleop._control_thread is None
    assert teleop._control_thread_error is None


def test_control_thread_exception_requests_node_shutdown():
    def callback():
        raise RuntimeError("control failed")

    teleop = _ThreadHarness(callback)
    teleop.start_control_loop()
    thread = teleop._control_thread
    thread.join(timeout=0.5)

    assert not thread.is_alive()
    assert isinstance(teleop._control_thread_error, RuntimeError)
    assert teleop.context.shutdown.is_set()
    assert "control failed" in teleop.logger.errors[0][0]
    teleop.stop_control_loop()


def test_shutdown_joins_control_thread_before_serial_writes():
    order = []

    class ReadExecutor:
        @staticmethod
        def shutdown(wait, cancel_futures):
            assert wait is True
            assert cancel_futures is True
            assert order == ["stop"]
            order.append("read-stop")

    class Driver:
        @staticmethod
        def set_torque(torque):
            assert order == ["stop", "read-stop"]
            np.testing.assert_array_equal(torque, np.zeros(2))
            order.append("zero")

        @staticmethod
        def set_torque_mode(enabled):
            assert enabled is False
            order.append("disable")

    class Teleop:
        _board_io_executor = ReadExecutor()
        _shutdown_board_io_executor = (
            FACTRTeleopDualBase._shutdown_board_io_executor
        )

        @staticmethod
        def stop_control_loop():
            order.append("stop")

        @staticmethod
        def _driver_groups():
            yield "small", Driver(), [1, 2]

    FACTRTeleopDualBase.shut_down(Teleop())

    assert order == ["stop", "read-stop", "zero", "disable"]


def test_shutdown_attempts_both_boards_after_one_board_fails():
    order = []

    class Driver:
        def __init__(self, name, fail):
            self.name = name
            self.fail = fail

        def set_torque(self, torque):
            order.append(f"{self.name}-zero")
            if self.fail:
                raise RuntimeError("disconnected")

        def set_torque_mode(self, enabled):
            assert enabled is False
            order.append(f"{self.name}-disable")
            if self.fail:
                raise RuntimeError("disconnected")

    small = Driver("small", fail=True)
    big = Driver("big", fail=False)

    class Teleop:
        _board_io_executor = None
        _shutdown_board_io_executor = (
            FACTRTeleopDualBase._shutdown_board_io_executor
        )

        @staticmethod
        def stop_control_loop():
            order.append("stop")

        @staticmethod
        def _driver_groups():
            yield "small", small, [1, 3]
            yield "big", big, [2]

    with pytest.raises(RuntimeError, match="small board zero torque failed"):
        FACTRTeleopDualBase.shut_down(Teleop())

    assert order == [
        "stop",
        "small-zero",
        "big-zero",
        "small-disable",
        "big-disable",
    ]


def test_dual_board_master_off_zeros_and_disables_every_board():
    order = []

    class Driver:
        def __init__(self, name):
            self.name = name
            self.torque_enabled = True

        def set_torque(self, torque):
            np.testing.assert_array_equal(torque, np.zeros(2))
            order.append(f"{self.name}-zero")

        def set_torque_mode(self, enabled):
            assert enabled is False
            self.torque_enabled = False
            order.append(f"{self.name}-disable")

    small = Driver("small")
    big = Driver("big")
    teleop = type("Teleop", (), {})()
    teleop.leader_torque_enabled = True
    teleop._driver_groups = lambda: iter((
        ("small", small, [1, 3]),
        ("big", big, [2, 4]),
    ))

    FACTRTeleopDualBase._set_leader_torque_enabled(teleop, False)

    assert order == [
        "small-zero", "big-zero", "small-disable", "big-disable",
    ]
    assert teleop.leader_torque_enabled is False


def test_failed_dual_board_read_does_not_publish_or_write_torque():
    logger = _Logger()

    class Teleop:
        name = "right"

        @staticmethod
        def get_leader_joint_states():
            raise DynamixelReadError("test timeout")

        @staticmethod
        def get_logger():
            return logger

        @staticmethod
        def set_leader_joint_torque(*_args):
            raise AssertionError("a failed read must not write torque")

        @staticmethod
        def _disable_leader_torque_after_read_failure():
            pass

        @staticmethod
        def publish_joint_pos():
            raise AssertionError("a failed read must not publish a new sample")

    assert FactrRizonTeleopDualBoard.control_loop_callback(Teleop()) is False
    assert "skipped control tick" in logger.warnings[0][0]


def test_successful_dual_board_iteration_writes_then_publishes():
    order = []
    zeros = np.zeros(7)

    class Teleop:
        num_arm_joints = 7
        enable_gravity_comp = False
        enable_torque_feedback = False
        enable_gripper_spring = False
        _last_commanded_torque_nm = zeros.copy()

        @staticmethod
        def get_leader_joint_states():
            return zeros, zeros, 0.0, 0.0

        @staticmethod
        def joint_limit_barrier(*_args):
            return zeros, 0.0

        @staticmethod
        def null_space_regulation(*_args):
            return zeros

        @staticmethod
        def _update_component_gains():
            return 0.0, 0.0, 0.0, 0.0

        @staticmethod
        def _capture_enable_tick(*_args):
            pass

        @staticmethod
        def set_leader_joint_torque(arm_torque, gripper_torque):
            np.testing.assert_array_equal(arm_torque, zeros)
            assert gripper_torque == 0.0
            order.append("torque")
            return zeros.copy()

        @staticmethod
        def _apply_leader_torque(arm_torque, gripper_torque):
            return Teleop.set_leader_joint_torque(arm_torque, gripper_torque)

        @staticmethod
        def publish_joint_pos():
            order.append("publish")

    assert FactrRizonTeleopDualBoard.control_loop_callback(Teleop()) is True
    assert order == ["torque", "publish"]
