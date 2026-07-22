"""Tests for the typed FACTR WebSocket stream."""

import asyncio
import threading
from types import MethodType, SimpleNamespace

from fastapi import WebSocketDisconnect

from factr_fastapi.factr_api import FactrAPI


class FakeWebSocket:
    """Capture two outbound frames, then emulate a disconnected client."""

    def __init__(self):
        self.accepted = False
        self.frames = []

    async def accept(self):
        self.accepted = True

    async def send_json(self, payload):
        self.frames.append(payload)
        if len(self.frames) == 2:
            raise WebSocketDisconnect()


def test_stream_sends_diagnostics_then_reading():
    relay = SimpleNamespace(
        side="left",
        lock=threading.Lock(),
        joint_pos=[0.0, 1.0, 2.0],
        diagnostics={},
        diagnostics_version=0,
    )
    relay._diagnostics_payload = MethodType(
        FactrAPI._diagnostics_payload, relay
    )
    websocket = FakeWebSocket()

    asyncio.run(FactrAPI.stream(relay, websocket))

    assert websocket.accepted is True
    diagnostics, reading = websocket.frames
    assert diagnostics["type"] == "diagnostics"
    assert diagnostics["available"] is False
    assert diagnostics["side"] == "left"
    assert diagnostics["raw_q_rad"] == []
    assert reading == {
        "type": "reading",
        "side": "left",
        "joint_pos": [0.0, 1.0, 2.0],
    }
