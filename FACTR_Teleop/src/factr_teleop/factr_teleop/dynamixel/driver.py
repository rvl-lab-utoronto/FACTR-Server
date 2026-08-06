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
    COMM_NOT_AVAILABLE,
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
EVENT_HEALTH_SNAPSHOT_COOLDOWN_S = 1.0

# Health/diagnostics registers (X-series control table; same for XC330 and XM430).
ADDR_CURRENT_LIMIT = 38            # 2 bytes, EEPROM (constant)
ADDR_HARDWARE_ERROR_STATUS = 70    # 1 byte; latched fault bits, cleared only by reboot/power-cycle
ADDR_REALTIME_TICK = 120            # 2 bytes, 1 ms tick, wraps every 32.768 s
ADDR_PRESENT_PWM = 124             # 2 bytes, signed
ADDR_PRESENT_CURRENT = 126         # 2 bytes, signed
# ADDR_PRESENT_VELOCITY = 128 (4B signed), ADDR_PRESENT_POSITION = 132 (4B signed) defined above
ADDR_POSITION_TRAJECTORY = 140     # 4 bytes, signed
ADDR_PRESENT_INPUT_VOLTAGE = 144   # 2 bytes, unit 0.1 V
ADDR_PRESENT_TEMPERATURE = 146     # 1 byte, deg C

# Contiguous RAM blocks for batched GroupSyncRead health telemetry: two bus transactions
# for ALL servos instead of ~N individual reads per servo -> cheap enough for ~10 Hz.
# Block B spans Realtime Tick(120)..Temperature(146), so one read covers the reboot
# clue plus pwm / current / velocity / position / trajectory / voltage / temperature.
LEN_HEALTH_BLOCK_A = ADDR_HARDWARE_ERROR_STATUS - ADDR_TORQUE_ENABLE + 1   # 64..70: torque_enable + hw_error
LEN_HEALTH_BLOCK_B = ADDR_PRESENT_TEMPERATURE - ADDR_REALTIME_TICK + 1     # 120..146


def _signed(value, byte_len):
    """Interpret an unsigned register value as two's-complement signed (2- or 4-byte)."""
    if value is None:
        return None
    bits = byte_len * 8
    return value - (1 << bits) if value >= (1 << (bits - 1)) else value

#: Hardware Error Status (addr 70) bit -> label, per the X-series control table.
HW_ERROR_BITS = (
    (0x01, "INPUT_VOLTAGE"),
    (0x04, "OVERHEATING"),
    (0x08, "MOTOR_ENCODER"),
    (0x10, "ELECTRICAL_SHOCK"),
    (0x20, "OVERLOAD"),
)


def hw_error_flags(bits):
    """Decode a Dynamixel Hardware Error Status byte (addr 70) into a list of labels."""
    if not bits:
        return []
    return [name for bit, name in HW_ERROR_BITS if bits & bit]


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
    def __init__(self, ids: Sequence[int], servo_types: Sequence[str], port: str = "/dev/ttyUSB0", baudrate: int = 4000000):
        self._ids = list(ids)
        self._port = port
        self._positions = None
        self._lock = Lock()
        self._diagnostics_lock = Lock()
        self._current_limit_cache = None   # lazily filled by read_health() (EEPROM, constant)
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
        self._last_event_health_at = 0.0
        self._last_event_health = []

        self._portHandler = PortHandler(port)
        self._packetHandler = PacketHandler(2.0)
        self._groupSyncRead = DiagnosticGroupSyncRead(
            self._portHandler, self._packetHandler, ADDR_PRESENT_VELOCITY, LEN_PRESENT_POSITION + LEN_PRESENT_VELOCITY,
        )
        self._groupSyncWrite = GroupSyncWrite(
            self._portHandler, self._packetHandler, ADDR_GOAL_CURRENT, LEN_GOAL_CURRENT,
        )
        if not self._portHandler.openPort():
            raise RuntimeError("Failed to open the port")
        if not self._portHandler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to change the baudrate, {baudrate}")

        for dxl_id in self._ids:
            if not self._groupSyncRead.addParam(dxl_id):
                raise RuntimeError(f"Failed to add parameter for Dynamixel with ID {dxl_id}")

        # Health telemetry: two contiguous GroupSyncReads (torque/error block + current/
        # temp block) so read_health() costs ~2 bus transactions instead of ~4 per servo.
        self._healthReadA = GroupSyncRead(
            self._portHandler, self._packetHandler, ADDR_TORQUE_ENABLE, LEN_HEALTH_BLOCK_A
        )
        self._healthReadB = GroupSyncRead(
            self._portHandler, self._packetHandler, ADDR_REALTIME_TICK, LEN_HEALTH_BLOCK_B
        )
        for dxl_id in self._ids:
            self._healthReadA.addParam(dxl_id)
            self._healthReadB.addParam(dxl_id)

        self.torque_to_current_map = np.array(
            [TORQUE_TO_CURRENT_MAPPING[servo] for servo in servo_types]
        )

        self._torque_enabled = False
        try:
            self.set_torque_mode(self._torque_enabled)
        except Exception as e:
            print(f"port: {port}, {e}")

    @property
    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_torque_mode(self, enable: bool):
        torque_value = TORQUE_ENABLE if enable else TORQUE_DISABLE
        with self._lock:
            for dxl_id in self._ids:
                dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                    self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, torque_value
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    raise RuntimeError(f"Failed to set torque mode for Dynamixel with ID {dxl_id}")
        self._torque_enabled = enable

    def close(self):
        self._portHandler.closePort()

    def set_operating_mode(self, mode: int):
        for dxl_id in self._ids:
            dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                self._portHandler, dxl_id, ADDR_OPERATING_MODE, mode
            )
            if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                raise RuntimeError(f"Failed to set operating mode for Dynamixel with ID {dxl_id}")
    
    def verify_operating_mode(self, expected_mode: int):
        for dxl_id in self._ids:
            mode, dxl_comm_result, dxl_error = self._packetHandler.read1ByteTxRx(
                self._portHandler, dxl_id, ADDR_OPERATING_MODE
            )
            if dxl_comm_result != COMM_SUCCESS or dxl_error != 0 or mode != expected_mode:
                raise RuntimeError(f"Operating mode mismatch for Dynamixel ID {dxl_id}")

    def get_positions_and_velocities(self, tries=10, source="unspecified"):
        _positions = np.zeros(len(self._ids), dtype=int)
        _velocities = np.zeros(len(self._ids), dtype=int)

        # Retry the normal transaction exactly as before, but retain the number and
        # result codes instead of silently recursing. No additional bus I/O is added.
        failed_results = []
        for _ in range(max(0, int(tries)) + 1):
            dxl_comm_result = self._groupSyncRead.txRxPacket()
            if dxl_comm_result == COMM_SUCCESS:
                break
            failed_results.append(int(dxl_comm_result))
        else:
            self._record_failed_read(source, failed_results)
            raise RuntimeError(f"Warning, communication failed: {dxl_comm_result}")
        
        for i, dxl_id in enumerate(self._ids):
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
        self._record_successful_read(
            source,
            _positions,
            _velocities,
            failed_results,
            self._groupSyncRead.error_dict,
        )
        
        # return positions and velocities in meaningful units
        positions_in_radians = _positions / 2048.0 * np.pi
        velocities_in_units = _velocities * 0.229 * 2 * np.pi / 60
        
        return positions_in_radians, velocities_in_units

    def _append_event(self, kind, stamp_monotonic_ns, details, health=None):
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
            if health is not None:
                event["health"] = health
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
        self, source, positions, velocities, failed_results, status_errors
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

        health = None
        if jump_indices or new_alerts:
            health = self._event_health_snapshot()

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
                health,
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
                health,
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

    def _event_health_snapshot(self):
        """Read fault registers once per anomaly cluster, never periodically."""
        now = time.monotonic()
        with self._diagnostics_lock:
            if now - self._last_event_health_at < EVENT_HEALTH_SNAPSHOT_COOLDOWN_S:
                return copy.deepcopy(self._last_event_health)
            self._last_event_health_at = now
        try:
            health = self.read_health(include_current_limit=False)
        except Exception as exc:
            health = [{"snapshot_error": str(exc)}]
        with self._diagnostics_lock:
            self._last_event_health = copy.deepcopy(health)
        return health

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
                "latest_reads": copy.deepcopy(self._latest_reads),
                "events": copy.deepcopy(list(self._diagnostic_events)),
            }
    

    def set_current(self, currents: Sequence[float]):
        if len(currents) != len(self._ids):
            raise ValueError("The length of currents must match the number of servos")
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set currents")

        currents = np.clip(currents, -900, 900)
        for dxl_id, current in zip(self._ids, currents):
            current_value = int(current)

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

    def set_torque(self, torques: Sequence[float]):
        currents = self.torque_to_current_map*torques
        self.set_current(currents)

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

    def read_health(self, include_current_limit=True):
        """Best-effort per-servo health telemetry for diagnosing torque dropouts.

        Returns one dict per servo id with keys: ``id``, ``torque_enable`` (0/1),
        ``hw_error`` (raw Hardware Error Status byte, reg 70), ``hw_error_flags`` (list
        of labels), ``realtime_tick`` (reg 120, 1 ms), ``present_pwm`` (reg 124,
        signed), ``present_current`` (reg 126,
        signed), ``present_velocity`` (reg 128, signed), ``present_position`` (reg 132,
        signed), ``position_trajectory`` (reg 140, signed), ``input_voltage`` (reg 144,
        raw, 0.1 V/LSB), ``temperature`` (reg 146, deg C), ``current_limit`` (reg 38).
        Any field that fails to read is ``None``.

        Uses TWO GroupSyncRead transactions (block A 64..70 for torque/error, block B
        120..146 for tick/pwm/current/velocity/position/trajectory/voltage/temp)
        covering all servos at once -- ~2 bus round-trips regardless of field count -- so
        it is cheap enough to call at ~10 Hz without badly stalling the control loop. The
        Current Limit is EEPROM/constant, so it is read once (via individual TxRx) and cached.
        """
        if include_current_limit and self._current_limit_cache is None:
            self._current_limit_cache = []
            for dxl_id in self._ids:
                lim, res, _ = self._packetHandler.read2ByteTxRx(
                    self._portHandler, dxl_id, ADDR_CURRENT_LIMIT
                )
                self._current_limit_cache.append(int(lim) if res == COMM_SUCCESS else None)

        # Two batched reads for ALL servos (block A: torque/error, block B: 120..146).
        ok_a = self._healthReadA.txRxPacket() == COMM_SUCCESS
        ok_b = self._healthReadB.txRxPacket() == COMM_SUCCESS

        health = []
        for i, dxl_id in enumerate(self._ids):
            te = he = None
            tick = None
            pwm = present_current = velocity = position = pos_traj = voltage = tmp = None
            if ok_a:
                if self._healthReadA.isAvailable(dxl_id, ADDR_TORQUE_ENABLE, 1):
                    te = self._healthReadA.getData(dxl_id, ADDR_TORQUE_ENABLE, 1)
                if self._healthReadA.isAvailable(dxl_id, ADDR_HARDWARE_ERROR_STATUS, 1):
                    he = self._healthReadA.getData(dxl_id, ADDR_HARDWARE_ERROR_STATUS, 1)
            if ok_b:
                b = self._healthReadB
                if b.isAvailable(dxl_id, ADDR_REALTIME_TICK, 2):
                    tick = b.getData(dxl_id, ADDR_REALTIME_TICK, 2)
                if b.isAvailable(dxl_id, ADDR_PRESENT_PWM, 2):
                    pwm = _signed(b.getData(dxl_id, ADDR_PRESENT_PWM, 2), 2)
                if b.isAvailable(dxl_id, ADDR_PRESENT_CURRENT, 2):
                    present_current = _signed(b.getData(dxl_id, ADDR_PRESENT_CURRENT, 2), 2)
                if b.isAvailable(dxl_id, ADDR_PRESENT_VELOCITY, 4):
                    velocity = _signed(b.getData(dxl_id, ADDR_PRESENT_VELOCITY, 4), 4)
                if b.isAvailable(dxl_id, ADDR_PRESENT_POSITION, 4):
                    position = _signed(b.getData(dxl_id, ADDR_PRESENT_POSITION, 4), 4)
                if b.isAvailable(dxl_id, ADDR_POSITION_TRAJECTORY, 4):
                    pos_traj = _signed(b.getData(dxl_id, ADDR_POSITION_TRAJECTORY, 4), 4)
                if b.isAvailable(dxl_id, ADDR_PRESENT_INPUT_VOLTAGE, 2):
                    voltage = b.getData(dxl_id, ADDR_PRESENT_INPUT_VOLTAGE, 2)   # unsigned, 0.1 V/LSB
                if b.isAvailable(dxl_id, ADDR_PRESENT_TEMPERATURE, 1):
                    tmp = b.getData(dxl_id, ADDR_PRESENT_TEMPERATURE, 1)
            health.append({
                "id": int(dxl_id),
                "torque_enable": int(te) if te is not None else None,
                "hw_error": int(he) if he is not None else None,
                "hw_error_flags": hw_error_flags(he) if he is not None else None,
                "realtime_tick": int(tick) if tick is not None else None,
                "present_pwm": pwm,
                "present_current": present_current,
                "present_velocity": velocity,
                "present_position": position,
                "position_trajectory": pos_traj,
                "input_voltage": int(voltage) if voltage is not None else None,
                "temperature": int(tmp) if tmp is not None else None,
                "current_limit": (
                    self._current_limit_cache[i]
                    if self._current_limit_cache is not None else None
                ),
            })
        return health


def main():
    # script for testing purposes
    ids = [1, 2, 3, 4, 5, 6, 7]
    port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT8ISV6J-if00-port0"

    try:
        driver = DynamixelDriver(ids, port=port, baudrate=4000000)
    except FileNotFoundError:
        print(f"Port {port} not found. Please check the connection.")
        return
    
    driver.set_operating_mode(0)
    driver.set_torque_mode(True)

    try:
        while True:
            positions = driver.get_positions()
            print(f"Current joint positions for IDs {ids}: {positions}")

            current_values = [0, 0, 0, 0, 0, 0, 0.0]
            driver.set_current(current_values)
    except KeyboardInterrupt:
        driver.set_torque_mode(False)
        driver.close()

if __name__ == "__main__":
    main()
