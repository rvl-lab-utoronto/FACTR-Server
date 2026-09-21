"""Tests for event-driven Dynamixel diagnostics (no periodic health polling)."""

import time
from collections import deque
from threading import Lock

import numpy as np
import pytest

from factr_teleop.dynamixel.driver import (
    DIAGNOSTIC_EVENT_HISTORY,
    DiagnosticGroupSyncRead,
    DynamixelDriver,
    DynamixelReadError,
    LowLatencyPortHandler,
)


def _driver_without_hardware():
    driver = DynamixelDriver.__new__(DynamixelDriver)
    driver._ids = [1, 2]
    driver._port = "/dev/test-dynamixel"
    driver._diagnostics_lock = Lock()
    driver._diagnostic_session = time.monotonic_ns()
    driver._read_sequence = 0
    driver._event_sequence = 0
    driver._comm_retry_count = 0
    driver._comm_failure_count = 0
    driver._status_alert_count = 0
    driver._position_jump_count = 0
    driver._last_position_ticks = None
    driver._last_status_errors = {}
    driver._latest_reads = {}
    driver._diagnostic_events = deque(maxlen=DIAGNOSTIC_EVENT_HISTORY)
    return driver


def test_group_sync_read_retains_status_packet_error_byte():
    class PacketHandler:
        @staticmethod
        def getProtocolVersion():
            return 2.0

        @staticmethod
        def readRx(_port, dxl_id, length):
            assert dxl_id == 1
            return [0] * length, 0, 0x80

    read = DiagnosticGroupSyncRead(object(), PacketHandler(), 128, 8)
    assert read.addParam(1) is True

    assert read.rxPacket() == 0
    assert read.error_dict == {1: 0x80}


def test_fast_sync_read_retains_all_payloads_and_status_error_bytes():
    class PacketHandler:
        @staticmethod
        def getProtocolVersion():
            return 2.0

        @staticmethod
        def syncReadTx(_port, start, length, ids, count, fast):
            assert (start, length, ids, count, fast) == (128, 2, [1, 2], 2, True)
            return 0

        @staticmethod
        def fastSyncReadRx(_port, dxl_id, length):
            assert dxl_id == 0xFE
            assert length == 12
            # Each aggregate block is ERR, ID, payload, CRC16.
            return [0x80, 1, 10, 11, 0, 0, 0x00, 2, 20, 21, 0, 0], 0, 0

    read = DiagnosticGroupSyncRead(object(), PacketHandler(), 128, 2)
    assert read.addParam(1) is True
    assert read.addParam(2) is True

    assert read.fastSyncRead() == 0
    assert read.error_dict == {1: 0x80, 2: 0x00}
    assert read.data_dict == {1: bytearray([10, 11]), 2: bytearray([20, 21])}


def test_fast_sync_read_rejects_duplicate_or_missing_device_ids():
    class PacketHandler:
        @staticmethod
        def getProtocolVersion():
            return 2.0

        @staticmethod
        def syncReadTx(*_args):
            return 0

        @staticmethod
        def fastSyncReadRx(*_args):
            return [0, 1, 10, 11, 0, 0, 0, 1, 20, 21, 0, 0], 0, 0

    read = DiagnosticGroupSyncRead(object(), PacketHandler(), 128, 2)
    read.addParam(1)
    read.addParam(2)

    assert read.fastSyncRead() == -3002
    assert read.last_result is False


def test_driver_always_uses_fast_sync_read():
    class Read:
        normal_calls = 0
        fast_calls = 0

        def txRxPacket(self):
            self.normal_calls += 1
            return 101

        def fastSyncRead(self):
            self.fast_calls += 1
            return 202

    driver = _driver_without_hardware()
    driver._groupSyncRead = Read()

    assert driver._read_state_packet() == 202
    assert driver._groupSyncRead.normal_calls == 0
    assert driver._groupSyncRead.fast_calls == 1


def test_write_alert_is_recorded_but_not_treated_as_instruction_failure():
    driver = _driver_without_hardware()

    class PacketHandler:
        @staticmethod
        def getTxRxResult(_result):
            return "unused"

        @staticmethod
        def getRxPacketError(_error):
            return "unused"

    driver._packetHandler = PacketHandler()
    driver._check_write_result("set torque mode", 9, 0, 0x80)

    event = driver.diagnostics_snapshot()["events"][0]
    assert event["kind"] == "status_alert"
    assert event["servo_ids"] == [9]
    assert event["status_error_bytes"] == {"9": 0x80}


def test_write_instruction_error_still_fails_with_details():
    driver = _driver_without_hardware()

    class PacketHandler:
        @staticmethod
        def getTxRxResult(_result):
            return "Success"

        @staticmethod
        def getRxPacketError(_error):
            return "Data range error"

    driver._packetHandler = PacketHandler()
    with pytest.raises(RuntimeError, match=r"ID 9.*status=0x84"):
        driver._check_write_result("set torque mode", 9, 0, 0x84)


def test_startup_write_retries_transient_communication_failure():
    driver = _driver_without_hardware()
    driver._portHandler = object()

    class PacketHandler:
        results = iter([(-3001, 0), (-3001, 0), (0, 0)])
        calls = 0

        @classmethod
        def write1ByteTxRx(cls, *_args):
            cls.calls += 1
            return next(cls.results)

    driver._packetHandler = PacketHandler()
    driver._write_1_byte_with_retry("set torque mode", 1, 64, 1)

    snapshot = driver.diagnostics_snapshot()
    assert PacketHandler.calls == 3
    assert snapshot["comm_retry_count"] == 2
    assert snapshot["comm_failure_count"] == 0
    assert snapshot["events"][0]["kind"] == "communication_retry"


def test_low_latency_packet_deadline_matches_one_ms_ftdi_setting():
    port = LowLatencyPortHandler.__new__(LowLatencyPortHandler)
    port.tx_time_per_byte = 0.0025  # 4 Mbps, 10 serial bits per byte, in ms
    port.setPacketTimeout(126)      # six 10-byte status packets

    assert port.packet_timeout == pytest.approx(10.315)


def test_current_telemetry_uses_same_conversion_as_torque_commands():
    driver = _driver_without_hardware()
    driver._present_currents = np.array([116, -372])
    driver._last_goal_current_raw = np.array([58, -186])
    driver.torque_to_current_map = np.array([1158.73, 1000 / 2.69])

    np.testing.assert_allclose(
        driver.current_estimated_torque(),
        driver._present_currents / driver.torque_to_current_map,
    )
    np.testing.assert_allclose(
        driver.commanded_torque(),
        driver._last_goal_current_raw / driver.torque_to_current_map,
    )


def test_default_read_attempts_once_then_returns_control_to_caller():
    class FailedRead:
        error_dict = {}
        calls = 0

        def fastSyncRead(self):
            self.calls += 1
            return -3001

    driver = _driver_without_hardware()
    driver._groupSyncRead = FailedRead()

    with pytest.raises(DynamixelReadError):
        driver.get_positions_and_velocities(source="control")

    assert driver._groupSyncRead.calls == 1
    assert driver.diagnostics_snapshot()["comm_failure_count"] == 1


def test_records_raw_reads_retries_jump_and_alert_without_extra_bus_reads():
    driver = _driver_without_hardware()
    driver._record_successful_read(
        "control", np.array([0, 0]), np.array([1, 2]), [], {1: 0, 2: 0},
        currents=np.array([3, -4]),
    )
    driver._record_successful_read(
        "publication",
        np.array([0, 2048]),
        np.array([3, 4]),
        [-3001],
        {1: 0, 2: 0x80},
    )

    snapshot = driver.diagnostics_snapshot()

    assert snapshot["latest_reads"]["control"]["raw_position_ticks"] == [0, 0]
    assert snapshot["latest_reads"]["control"]["raw_present_current"] == [3, -4]
    assert snapshot["latest_reads"]["publication"]["raw_position_ticks"] == [0, 2048]
    assert snapshot["comm_retry_count"] == 1
    assert snapshot["position_jump_count"] == 1
    assert snapshot["status_alert_count"] == 1
    assert [event["kind"] for event in snapshot["events"]] == [
        "communication_retry", "position_jump", "status_alert"
    ]
    jump = snapshot["events"][1]
    assert jump["servo_ids"] == [2]
    assert jump["delta_rad"] == [0.0, np.pi]
    assert "health" not in jump


def test_records_terminal_communication_failure():
    driver = _driver_without_hardware()

    driver._record_failed_read("control", [-3001, -3001, -3001])

    snapshot = driver.diagnostics_snapshot()
    assert snapshot["comm_retry_count"] == 3
    assert snapshot["comm_failure_count"] == 1
    assert snapshot["events"][0]["kind"] == "communication_failure"


def test_latched_status_alert_is_recorded_on_edges_not_every_read():
    driver = _driver_without_hardware()
    positions = np.array([10, 20])
    velocities = np.array([0, 0])

    driver._record_successful_read(
        "control", positions, velocities, [], {1: 0x80, 2: 0}
    )
    driver._record_successful_read(
        "publication", positions, velocities, [], {1: 0x80, 2: 0}
    )
    driver._record_successful_read(
        "control", positions, velocities, [], {1: 0, 2: 0}
    )

    snapshot = driver.diagnostics_snapshot()
    assert snapshot["status_alert_count"] == 1
    assert [event["kind"] for event in snapshot["events"]] == [
        "status_alert", "status_alert_cleared"
    ]
