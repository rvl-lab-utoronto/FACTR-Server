# ---------------------------------------------------------------------------
# FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning
# https://arxiv.org/abs/2502.17432
# Copyright (c) 2025 Jason Jingzhou Liu and Yulong Li

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------
# Based on: 
# https://github.com/wuphilipp/gello_software/blob/main/gello/dynamixel/driver.py
# ---------------------------------------------------------------------------


import copy
import time
from collections import deque
from threading import Lock
from typing import Protocol, Sequence

import numpy as np
from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.group_sync_write import GroupSyncWrite
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import (
    BROADCAST_ID,
    COMM_NOT_AVAILABLE,
    COMM_RX_CORRUPT,
    COMM_RX_FAIL,
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
)

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_CURRENT = 102
LEN_GOAL_CURRENT = 2
ADDR_PRESENT_CURRENT = 126
LEN_PRESENT_CURRENT = 2
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4
ADDR_PRESENT_VELOCITY = 128
LEN_PRESENT_VELOCITY = 4
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
ADDR_OPERATING_MODE = 11
CURRENT_CONTROL_MODE = 0
POSITION_CONTROL_MODE = 3

# A change this large between adjacent bus reads cannot be physical on the leader
# mechanism (the two reads are normally ~1 ms apart). It is recorded, not filtered:
# control behavior remains unchanged while Rerun gets the exact before/after ticks.
POSITION_JUMP_THRESHOLD_RAD = 0.5
DIAGNOSTIC_EVENT_HISTORY = 32
USB_LATENCY_TIMER_MS = 1.0
PACKET_RESPONSE_MARGIN_MS = 8.0
DXL_ALERT_MASK = 0x80
DXL_INSTRUCTION_ERROR_MASK = 0x7F
STARTUP_WRITE_ATTEMPTS = 3
STARTUP_WRITE_RETRY_DELAY_S = 0.002


class DynamixelReadError(RuntimeError):
    """One bounded Dynamixel state acquisition did not complete."""


class LowLatencyPortHandler(PortHandler):
    """Port handler whose software deadline matches the asserted FTDI latency.

    The ROS Humble SDK hard-codes a 16 ms latency allowance even when Linux's
    ``latency_timer`` is 1 ms. Its deadline therefore adds 34 ms to every
    failed Sync Read. FACTR asserts the kernel value before opening a board, so
    use that same value plus a bounded response/scheduling margin without
    modifying ``/opt/ros``. The two-board leader needs more than the SDK's
    nominal 2 ms margin in practice; 8 ms avoids false timeouts while keeping
    a missing response far below the upstream deadline.
    """

    def setPacketTimeout(self, packet_length):
        self.packet_start_time = self.getCurrentTime()
        self.packet_timeout = (
            self.tx_time_per_byte * packet_length
            + USB_LATENCY_TIMER_MS * 2.0
            + PACKET_RESPONSE_MARGIN_MS
        )


class DiagnosticGroupSyncRead(GroupSyncRead):
    """GroupSyncRead that retains each servo status packet's error byte.

    The upstream SDK discards this byte in ``GroupSyncRead.rxPacket``. Protocol
    2.0 uses bit 7 as the Hardware Error alert, so retaining it lets the normal
    position read trigger diagnostics without another periodic bus transaction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_dict = {}

    def rxPacket(self):
        self.last_result = False
        self.error_dict = {}
        if self.ph.getProtocolVersion() == 1.0:
            return COMM_NOT_AVAILABLE
        if not self.data_dict:
            return COMM_NOT_AVAILABLE

        result = COMM_RX_FAIL
        for dxl_id in self.data_dict:
            data, result, error = self.ph.readRx(
                self.port, dxl_id, self.data_length
            )
            if result != COMM_SUCCESS:
                return result
            self.data_dict[dxl_id] = data
            self.error_dict[dxl_id] = int(error)

        self.last_result = True
        return result

    def fastSyncRead(self):
        """Parse one aggregate Fast Sync Read while retaining status errors."""
        self.last_result = False
        self.error_dict = {}
        if self.ph.getProtocolVersion() == 1.0 or not self.data_dict:
            return COMM_NOT_AVAILABLE

        fast_rx = getattr(self.ph, "fastSyncReadRx", None)
        if not callable(fast_rx):
            return COMM_NOT_AVAILABLE

        if self.is_param_changed or not self.param:
            self.makeParam()

        try:
            result = self.ph.syncReadTx(
                self.port,
                self.start_address,
                self.data_length,
                self.param,
                len(self.data_dict),
                True,
            )
        except TypeError:
            # A partially upgraded SDK still has the old syncReadTx signature.
            return COMM_NOT_AVAILABLE
        if result != COMM_SUCCESS:
            return result

        block_length = self.data_length + 4  # error + ID + data + CRC16
        expected_length = block_length * len(self.data_dict)
        raw_data, result, _ = fast_rx(
            self.port, BROADCAST_ID, expected_length
        )
        if result != COMM_SUCCESS:
            return result

        raw_data = bytearray(raw_data)
        if len(raw_data) != expected_length:
            return COMM_RX_CORRUPT

        expected_ids = set(self.data_dict)
        received_ids = set()
        for start in range(0, expected_length, block_length):
            status_error = int(raw_data[start])
            dxl_id = int(raw_data[start + 1])
            if dxl_id not in expected_ids or dxl_id in received_ids:
                return COMM_RX_CORRUPT
            received_ids.add(dxl_id)
            self.error_dict[dxl_id] = status_error
            self.data_dict[dxl_id] = bytearray(
                raw_data[start + 2:start + 2 + self.data_length]
            )

        if received_ids != expected_ids:
            return COMM_RX_CORRUPT
        self.last_result = True
        return COMM_SUCCESS


TORQUE_TO_CURRENT_MAPPING = {
    "XC330_T288_T": 1158.73,
    "XM430_W210_T": 1000/2.69,
    "XC330_M288_T": 1158.73,
    "XL330_M288_T": 1158.73,
}


class DynamixelDriverProtocol(Protocol):
    def set_current(self, currents: Sequence[float]):
        """Set the current for the Dynamixel servos.

        Args:
            currents (Sequence[float]): A list of currents in mA.
        """
        ...

    def torque_enabled(self) -> bool:
        """Check if torque is enabled for the Dynamixel servos.

        Returns:
            bool: True if torque is enabled, False if it is disabled.
        """
        ...

    def set_torque_mode(self, enable: bool):
        """Set the torque mode for the Dynamixel servos.

        Args:
            enable (bool): True to enable torque, False to disable.
        """
        ...

    def get_positions(self) -> np.ndarray:
        """Get the current joint angles in radians.

        Returns:
            np.ndarray: An array of joint angles.
        """
        ...
        dxl_comm_result = self._groupSyncRead.txRxPacket()
    def close(self):
        """Close the driver."""


class DynamixelDriver(DynamixelDriverProtocol):
    def __init__(
        self,
        ids: Sequence[int],
        servo_types: Sequence[str],
        port: str = "/dev/ttyUSB0",
        baudrate: int = 4000000,
    ):
        self._ids = list(ids)
        self._port = port
        self._positions = None
        self._present_currents = None
        self._last_goal_current_raw = np.zeros(len(self._ids), dtype=int)
        self._lock = Lock()
        self._diagnostics_lock = Lock()
        self._diagnostic_session = time.monotonic_ns()
        self._read_sequence = 0
        self._event_sequence = 0
        self._comm_retry_count = 0
        self._comm_failure_count = 0
        self._status_alert_count = 0
        self._position_jump_count = 0
        self._last_position_ticks = None
        self._last_status_errors = {}
        self._latest_reads = {}
        self._diagnostic_events = deque(maxlen=DIAGNOSTIC_EVENT_HISTORY)

        self._portHandler = LowLatencyPortHandler(port)
        self._packetHandler = PacketHandler(2.0)
        self._groupSyncRead = DiagnosticGroupSyncRead(
            self._portHandler,
            self._packetHandler,
            ADDR_PRESENT_CURRENT,
            LEN_PRESENT_CURRENT + LEN_PRESENT_VELOCITY + LEN_PRESENT_POSITION,
        )
        self._groupSyncWrite = GroupSyncWrite(
            self._portHandler, self._packetHandler, ADDR_GOAL_CURRENT, LEN_GOAL_CURRENT,
        )
        if not callable(getattr(self._packetHandler, "fastSyncReadRx", None)):
            raise RuntimeError(
                "DYNAMIXEL SDK 4.0.5 or newer is required; the active runtime "
                "SDK does not provide Fast Sync Read"
            )
        if not self._portHandler.openPort():
            raise RuntimeError("Failed to open the port")
        if not self._portHandler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to change the baudrate, {baudrate}")
        # Keep the SDK's nonblocking packet reader. The bounded packet deadline
        # prevents a missed transaction from spinning indefinitely.
        self._portHandler.ser.timeout = 0

        for dxl_id in self._ids:
            if not self._groupSyncRead.addParam(dxl_id):
                raise RuntimeError(f"Failed to add parameter for Dynamixel with ID {dxl_id}")

        self.torque_to_current_map = np.array(
            [TORQUE_TO_CURRENT_MAPPING[servo] for servo in servo_types]
        )

        self._torque_enabled = False
        try:
            self.set_torque_mode(self._torque_enabled)
        except Exception:
            pass

    @property
    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_torque_mode(self, enable: bool):
        torque_value = TORQUE_ENABLE if enable else TORQUE_DISABLE
        with self._lock:
            for dxl_id in self._ids:
                self._write_1_byte_with_retry(
                    "set torque mode", dxl_id, ADDR_TORQUE_ENABLE, torque_value
                )
        self._torque_enabled = enable

    def close(self):
        self._portHandler.closePort()

    def set_operating_mode(self, mode: int):
        for dxl_id in self._ids:
            self._write_1_byte_with_retry(
                "set operating mode", dxl_id, ADDR_OPERATING_MODE, mode
            )

    def _write_1_byte_with_retry(self, operation, dxl_id, address, value):
        """Perform an idempotent startup write with bounded comm-only retries."""
        failed_results = []
        for attempt in range(STARTUP_WRITE_ATTEMPTS):
            comm_result, status_error = self._packetHandler.write1ByteTxRx(
                self._portHandler, dxl_id, address, value
            )
            if comm_result == COMM_SUCCESS:
                # Instruction errors are deterministic and must never be retried.
                self._check_write_result(
                    operation, dxl_id, comm_result, status_error
                )
                if failed_results:
                    with self._diagnostics_lock:
                        self._comm_retry_count += len(failed_results)
                    self._append_event(
                        "communication_retry",
                        time.monotonic_ns(),
                        {
                            "source": operation.replace(" ", "_"),
                            "servo_ids": [int(dxl_id)],
                            "result_codes": failed_results,
                        },
                    )
                return
            failed_results.append(int(comm_result))
            if attempt + 1 < STARTUP_WRITE_ATTEMPTS:
                time.sleep(STARTUP_WRITE_RETRY_DELAY_S)

        with self._diagnostics_lock:
            self._comm_retry_count += len(failed_results)
            self._comm_failure_count += 1
        self._check_write_result(
            operation, dxl_id, comm_result, status_error
        )

    def _check_write_result(self, operation, dxl_id, comm_result, status_error):
        """Reject failed writes without mistaking Protocol 2 alerts for failures.

        Bit 7 of a Protocol 2 status byte means the servo has a hardware alert;
        it does not mean the preceding instruction failed. Normal sync reads retain
        that alert for diagnostics. Only communication failures and the lower seven
        instruction-error bits make a write fail here.
        """
        status_error = int(status_error)
        instruction_error = status_error & DXL_INSTRUCTION_ERROR_MASK
        if comm_result != COMM_SUCCESS or instruction_error:
            comm_detail = self._packetHandler.getTxRxResult(comm_result)
            status_detail = self._packetHandler.getRxPacketError(status_error)
            raise RuntimeError(
                f"Failed to {operation} for Dynamixel with ID {dxl_id}: "
                f"comm={comm_result} ({comm_detail}), status=0x{status_error:02x} "
                f"({status_detail})"
            )
        if status_error & DXL_ALERT_MASK:
            self._append_event(
                "status_alert",
                time.monotonic_ns(),
                {
                    "source": operation.replace(" ", "_"),
                    "servo_ids": [int(dxl_id)],
                    "status_error_bytes": {str(dxl_id): status_error},
                },
            )
    
    def verify_operating_mode(self, expected_mode: int):
        for dxl_id in self._ids:
            mode, dxl_comm_result, dxl_error = self._packetHandler.read1ByteTxRx(
                self._portHandler, dxl_id, ADDR_OPERATING_MODE
            )
            if dxl_comm_result != COMM_SUCCESS or dxl_error != 0 or mode != expected_mode:
                raise RuntimeError(f"Operating mode mismatch for Dynamixel ID {dxl_id}")

    def get_positions_and_velocities(self, tries=0, source="unspecified"):
        _positions = np.zeros(len(self._ids), dtype=int)
        _velocities = np.zeros(len(self._ids), dtype=int)
        _currents = np.zeros(len(self._ids), dtype=int)

        # Retry the normal transaction exactly as before, but retain the number and
        # result codes instead of silently recursing. No additional bus I/O is added.
        failed_results = []
        for _ in range(max(0, int(tries)) + 1):
            dxl_comm_result = self._read_state_packet()
            if dxl_comm_result == COMM_SUCCESS:
                break
            failed_results.append(int(dxl_comm_result))
        else:
            self._record_failed_read(source, failed_results)
            raise DynamixelReadError(
                f"Dynamixel read failed on {self._port} "
                f"(source={source}, result={dxl_comm_result})"
            )
        
        for i, dxl_id in enumerate(self._ids):
            # Present Current is contiguous with velocity and position, so this
            # comes from the same Sync Read transaction (no extra bus request).
            if self._groupSyncRead.isAvailable(
                dxl_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT
            ):
                current = self._groupSyncRead.getData(
                    dxl_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT
                )
                if current > 0x7FFF:
                    current -= 0x10000
                _currents[i] = current
            else:
                raise RuntimeError(
                    f"Failed to get present current for Dynamixel with ID {dxl_id}"
                )

            # read velocity data
            if self._groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY):
                velocity = self._groupSyncRead.getData(dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
                # apply sign correction
                if velocity > 0x7FFFFFFF:
                    velocity -= 0x100000000
                _velocities[i] = velocity
            else:
                raise RuntimeError(f"Failed to get velocity for Dynamixel with ID {dxl_id}")
            
            # read position data
            if self._groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                position = self._groupSyncRead.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                # apply sign correction
                if position > 0x7FFFFFFF:
                    position -= 0x100000000
                _positions[i] = position
            else:
                raise RuntimeError(f"Failed to get position for Dynamixel with ID {dxl_id}")
            
        self._positions = _positions
        self._velocities = _velocities
        self._present_currents = _currents
        self._record_successful_read(
            source,
            _positions,
            _velocities,
            failed_results,
            self._groupSyncRead.error_dict,
            currents=_currents,
        )
        
        # return positions and velocities in meaningful units
        positions_in_radians = _positions / 2048.0 * np.pi
        velocities_in_units = _velocities * 0.229 * 2 * np.pi / 60
        
        return positions_in_radians, velocities_in_units

    def _read_state_packet(self):
        """Read all servo state with one aggregate Protocol 2.0 response."""
        return self._groupSyncRead.fastSyncRead()

    def _append_event(self, kind, stamp_monotonic_ns, details):
        """Append one JSON-safe diagnostic event to the retained in-process ring."""
        with self._diagnostics_lock:
            self._event_sequence += 1
            event = {
                "sequence": self._event_sequence,
                "stamp_monotonic_ns": int(stamp_monotonic_ns),
                "kind": kind,
                "port": self._port,
                "session": self._diagnostic_session,
                **details,
            }
            self._diagnostic_events.append(event)

    def _record_failed_read(self, source, failed_results):
        stamp = time.monotonic_ns()
        with self._diagnostics_lock:
            self._comm_retry_count += len(failed_results)
            self._comm_failure_count += 1
        self._append_event(
            "communication_failure",
            stamp,
            {"source": source, "result_codes": list(failed_results)},
        )

    def _record_successful_read(
        self, source, positions, velocities, failed_results, status_errors,
        currents=None,
    ):
        stamp = time.monotonic_ns()
        positions = np.asarray(positions, dtype=int)
        velocities = np.asarray(velocities, dtype=int)
        alerts = {
            int(dxl_id): int(error)
            for dxl_id, error in status_errors.items()
            if error
        }

        with self._diagnostics_lock:
            self._read_sequence += 1
            read_sequence = self._read_sequence
            self._comm_retry_count += len(failed_results)
            previous = (
                None if self._last_position_ticks is None
                else self._last_position_ticks.copy()
            )
            previous_alerts = self._last_status_errors
            new_alerts = {
                dxl_id: error for dxl_id, error in alerts.items()
                if previous_alerts.get(dxl_id) != error
            }
            cleared_alerts = sorted(set(previous_alerts) - set(alerts))
            self._last_status_errors = alerts
            self._last_position_ticks = positions.copy()
            self._latest_reads[str(source)] = {
                "sequence": read_sequence,
                "stamp_monotonic_ns": stamp,
                "raw_position_ticks": positions.tolist(),
                "raw_velocity_ticks": velocities.tolist(),
                "retries": len(failed_results),
                "status_error_bytes": {
                    str(dxl_id): int(error)
                    for dxl_id, error in status_errors.items()
                },
            }
            if currents is not None:
                self._latest_reads[str(source)]["raw_present_current"] = (
                    np.asarray(currents, dtype=int).tolist()
                )

        if failed_results:
            self._append_event(
                "communication_retry",
                stamp,
                {
                    "source": source,
                    "read_sequence": read_sequence,
                    "result_codes": list(failed_results),
                    "raw_position_ticks": positions.tolist(),
                },
            )

        jump_indices = []
        delta_rad = None
        if previous is not None and previous.shape == positions.shape:
            delta_rad = (positions - previous) / 2048.0 * np.pi
            jump_indices = np.flatnonzero(
                np.abs(delta_rad) >= POSITION_JUMP_THRESHOLD_RAD
            ).tolist()

        if jump_indices:
            with self._diagnostics_lock:
                self._position_jump_count += 1
            self._append_event(
                "position_jump",
                stamp,
                {
                    "source": source,
                    "read_sequence": read_sequence,
                    "servo_ids": [int(self._ids[i]) for i in jump_indices],
                    "previous_raw_position_ticks": previous.tolist(),
                    "raw_position_ticks": positions.tolist(),
                    "delta_rad": delta_rad.tolist(),
                    "threshold_rad": POSITION_JUMP_THRESHOLD_RAD,
                },
            )

        if new_alerts:
            with self._diagnostics_lock:
                self._status_alert_count += 1
            self._append_event(
                "status_alert",
                stamp,
                {
                    "source": source,
                    "read_sequence": read_sequence,
                    "status_error_bytes": {
                        str(dxl_id): error for dxl_id, error in new_alerts.items()
                    },
                },
            )

        if cleared_alerts:
            self._append_event(
                "status_alert_cleared",
                stamp,
                {
                    "source": source,
                    "read_sequence": read_sequence,
                    "servo_ids": cleared_alerts,
                },
            )

    def diagnostics_snapshot(self):
        """Return counters, latest raw reads, and retained events with no bus I/O."""
        with self._diagnostics_lock:
            return {
                "port": self._port,
                "session": self._diagnostic_session,
                "servo_ids": [int(dxl_id) for dxl_id in self._ids],
                "comm_retry_count": self._comm_retry_count,
                "comm_failure_count": self._comm_failure_count,
                "status_alert_count": self._status_alert_count,
                "position_jump_count": self._position_jump_count,
                "fast_sync_read": True,
                "latest_reads": copy.deepcopy(self._latest_reads),
                "events": copy.deepcopy(list(self._diagnostic_events)),
            }

    def present_current_raw(self):
        """Return the signed Present Current values from the latest state read."""
        if self._present_currents is None:
            raise RuntimeError("Present Current is unavailable before the first state read")
        return self._present_currents.copy()

    def current_estimated_torque(self):
        """Convert measured motor current to torque with FACTR's actuator map.

        This is an estimate based on motor current, not a joint torque-sensor
        measurement. Values are in the driver's motor-coordinate convention.
        """
        return self.present_current_raw() / self.torque_to_current_map

    def commanded_torque(self):
        """Exact post-clipping/quantization goal current, expressed as torque."""
        return self._last_goal_current_raw.copy() / self.torque_to_current_map
    

    def set_current(self, currents: Sequence[float]):
        if len(currents) != len(self._ids):
            raise ValueError("The length of currents must match the number of servos")
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set currents")

        goal_current_raw = np.trunc(
            np.clip(np.asarray(currents, dtype=float), -900, 900)
        ).astype(int)
        for dxl_id, current_value in zip(self._ids, goal_current_raw):

            param_goal_current = [
                DXL_LOBYTE(current_value),
                DXL_HIBYTE(current_value)
            ]

            if not self._groupSyncWrite.addParam(dxl_id, param_goal_current):
                raise RuntimeError(f"Failed to set current for Dynamixel with ID {dxl_id}")
        dxl_comm_result = self._groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError("Failed to syncwrite goal current")
        self._groupSyncWrite.clearParam()
        self._last_goal_current_raw = goal_current_raw
        return goal_current_raw.copy()

    def set_torque(self, torques: Sequence[float]):
        currents = self.torque_to_current_map*torques
        self.set_current(currents)
        return self.commanded_torque()

    def set_position(self, positions: Sequence[float]):
        """Command goal positions in radians.

        Requires the servos to be in POSITION_CONTROL_MODE with torque enabled.
        The operating mode lives in EEPROM, so switch modes with torque disabled::

            driver.set_torque_mode(False)
            driver.set_operating_mode(POSITION_CONTROL_MODE)
            driver.verify_operating_mode(POSITION_CONTROL_MODE)
            driver.set_torque_mode(True)
            driver.set_position([...])
        """
        if len(positions) != len(self._ids):
            raise ValueError("The length of positions must match the number of servos")
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set positions")

        # Separate GroupSyncWrite: self._groupSyncWrite is bound to the 2-byte
        # goal-current register, whereas goal position is a 4-byte register.
        group = GroupSyncWrite(
            self._portHandler, self._packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION
        )
        for dxl_id, position in zip(self._ids, positions):
            # inverse of the conversion in get_positions_and_velocities()
            raw = int(position / np.pi * 2048.0)
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(raw)),
                DXL_HIBYTE(DXL_LOWORD(raw)),
                DXL_LOBYTE(DXL_HIWORD(raw)),
                DXL_HIBYTE(DXL_HIWORD(raw)),
            ]
            if not group.addParam(dxl_id, param_goal_position):
                raise RuntimeError(f"Failed to set position for Dynamixel with ID {dxl_id}")
        dxl_comm_result = group.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError("Failed to syncwrite goal position")
        group.clearParam()

def main():
    # script for testing purposes
    ids = [1, 2, 3, 4, 5, 6, 7]
    port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT8ISV6J-if00-port0"

    try:
        driver = DynamixelDriver(ids, port=port, baudrate=4000000)
    except FileNotFoundError:
        return
    
    driver.set_operating_mode(0)
    driver.set_torque_mode(True)

    try:
        while True:
            positions = driver.get_positions()

            current_values = [0, 0, 0, 0, 0, 0, 0.0]
            driver.set_current(current_values)
    except KeyboardInterrupt:
        driver.set_torque_mode(False)
        driver.close()

if __name__ == "__main__":
    main()
