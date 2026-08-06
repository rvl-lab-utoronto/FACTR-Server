"""Tests for event-driven Dynamixel diagnostics (no periodic health polling)."""

import time
from collections import deque
from threading import Lock

import numpy as np

from factr_teleop.dynamixel.driver import (
    DIAGNOSTIC_EVENT_HISTORY,
    DiagnosticGroupSyncRead,
    DynamixelDriver,
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
    driver._event_health_snapshot = lambda: [{
        "id": 2,
        "hw_error": 8,
        "realtime_tick": 12,
        "input_voltage": 118,
        "torque_enable": 0,
        "temperature": 31,
        "present_position": 2048,
    }]
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


def test_records_raw_reads_retries_jump_alert_and_fault_snapshot():
    driver = _driver_without_hardware()
    driver._record_successful_read(
        "control", np.array([0, 0]), np.array([1, 2]), [], {1: 0, 2: 0}
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
    assert jump["health"][0]["hw_error"] == 8


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
