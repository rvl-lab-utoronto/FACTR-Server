"""Thread-safe cache for the one Dynamixel state read used by each control tick."""

from dataclasses import dataclass
from threading import Lock
import time

import numpy as np


@dataclass(frozen=True)
class RawJointState:
    """One immutable copy of a raw Dynamixel position/velocity sample."""

    position: np.ndarray
    velocity: np.ndarray
    sequence: int
    stamp_monotonic_ns: int


class RawJointStateCache:
    """Share a control-loop read with publishers without touching the bus again."""

    def __init__(self, size: int):
        self._lock = Lock()
        self._position = np.zeros(size, dtype=float)
        self._velocity = np.zeros(size, dtype=float)
        self._sequence = 0
        self._stamp_monotonic_ns = 0

    def update(self, position, velocity) -> RawJointState:
        """Copy a newly completed control read into the cache."""
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        if position.shape != self._position.shape:
            raise ValueError(
                f"position shape {position.shape} does not match cache "
                f"shape {self._position.shape}"
            )
        if velocity.shape != self._velocity.shape:
            raise ValueError(
                f"velocity shape {velocity.shape} does not match cache "
                f"shape {self._velocity.shape}"
            )
        with self._lock:
            self._position = position.copy()
            self._velocity = velocity.copy()
            self._sequence += 1
            self._stamp_monotonic_ns = time.monotonic_ns()
            return self._snapshot_locked()

    def snapshot(self) -> RawJointState:
        """Return a defensive copy of the newest control sample."""
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> RawJointState:
        return RawJointState(
            position=self._position.copy(),
            velocity=self._velocity.copy(),
            sequence=self._sequence,
            stamp_monotonic_ns=self._stamp_monotonic_ns,
        )
