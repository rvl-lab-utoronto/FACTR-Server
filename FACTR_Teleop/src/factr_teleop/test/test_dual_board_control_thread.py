"""Regression tests for the dual-board leader's dedicated control thread."""

import threading

import numpy as np

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

    class Driver:
        @staticmethod
        def set_torque_mode(enabled):
            assert enabled is False
            order.append("disable")

    class Teleop:
        num_arm_joints = 7

        @staticmethod
        def stop_control_loop():
            order.append("stop")

        @staticmethod
        def set_leader_joint_torque(arm_torque, gripper_torque):
            assert order == ["stop"]
            np.testing.assert_array_equal(arm_torque, np.zeros(7))
            assert gripper_torque == 0.0
            order.append("zero")

        @staticmethod
        def _drivers():
            yield Driver()

    FACTRTeleopDualBase.shut_down(Teleop())

    assert order == ["stop", "zero", "disable"]


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
            return 0.0, 0.0

        @staticmethod
        def _capture_enable_tick(*_args):
            pass

        @staticmethod
        def set_leader_joint_torque(arm_torque, gripper_torque):
            np.testing.assert_array_equal(arm_torque, zeros)
            assert gripper_torque == 0.0
            order.append("torque")

        @staticmethod
        def publish_joint_pos():
            order.append("publish")

    assert FactrRizonTeleopDualBoard.control_loop_callback(Teleop()) is True
    assert order == ["torque", "publish"]
