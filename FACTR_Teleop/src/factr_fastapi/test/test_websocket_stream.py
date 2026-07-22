"""Tests for the typed FACTR WebSocket stream and its inbound force-feedback path."""

import asyncio
import json
import threading
from types import MethodType, SimpleNamespace

from fastapi import WebSocketDisconnect
from rclpy.clock import Clock

from factr_fastapi.factr_api import FactrAPI


class FakeWebSocket:
    """Feed queued inbound frames, capture outbound ones, then hang up.

    ``send_json`` raises :class:`WebSocketDisconnect` once ``close_after`` outbound
    frames are captured (killing the sender task deterministically); ``receive_text``
    serves the queued inbound frames and then waits for that same point before
    hanging up the receive direction.
    """

    def __init__(self, inbound=(), close_after=2):
        self.accepted = False
        self.frames = []
        self.inbound = list(inbound)
        self.close_after = close_after

    async def accept(self):
        self.accepted = True

    async def send_json(self, payload):
        self.frames.append(payload)
        if len(self.frames) >= self.close_after:
            raise WebSocketDisconnect()

    async def receive_text(self):
        if self.inbound:
            return self.inbound.pop(0)
        while len(self.frames) < self.close_after:
            await asyncio.sleep(0.001)
        raise WebSocketDisconnect()


class FakePublisher:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


class FakeLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, msg, **kwargs):
        self.warnings.append(msg)


def _relay(side="left"):
    relay = SimpleNamespace(
        side=side,
        lock=threading.Lock(),
        joint_pos=[0.0, 1.0, 2.0],
        diagnostics={},
        diagnostics_version=0,
        force_feedback_pub=FakePublisher(),
    )
    for name in (
        "stream",
        "_stream_sender",
        "_handle_inbound_frame",
        "_diagnostics_payload",
        "_publish_force_feedback",
        "post_force_feedback",
    ):
        setattr(relay, name, MethodType(getattr(FactrAPI, name), relay))
    logger = FakeLogger()
    relay.get_logger = lambda: logger
    relay.get_clock = lambda: Clock()
    return relay


def test_stream_sends_diagnostics_then_reading():
    relay = _relay()
    websocket = FakeWebSocket()

    asyncio.run(relay.stream(websocket))

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


def test_stream_republishes_inbound_force_feedback():
    relay = _relay()
    tau = [0.5, -1.25, 2.0, 0.0, 0.0, 0.0, 3.5]
    websocket = FakeWebSocket(inbound=[
        json.dumps({"type": "force_feedback", "side": "left", "space": "joint", "tau": tau}),
    ])

    asyncio.run(relay.stream(websocket))

    assert [list(m.effort) for m in relay.force_feedback_pub.msgs] == [tau]
    assert relay.get_logger().warnings == []


def test_inbound_frame_rejects_malformed_frames():
    relay = _relay()
    bad_frames = [
        "not json",
        json.dumps([1, 2, 3]),                                        # not an object
        json.dumps({"type": "reading", "side": "left"}),              # wrong type
        json.dumps({"type": "force_feedback", "side": "right", "tau": [1.0]}),  # wrong side
        json.dumps({"type": "force_feedback", "tau": []}),            # empty tau
        json.dumps({"type": "force_feedback", "tau": [1.0, float("nan")]}),  # non-finite
        json.dumps({"type": "force_feedback", "space": "tcp", "tau": [0.0] * 6}),  # future space
    ]
    for frame in bad_frames:
        relay._handle_inbound_frame(frame)

    assert relay.force_feedback_pub.msgs == []
    assert len(relay.get_logger().warnings) == len(bad_frames)


def test_inbound_frame_accepts_defaulted_space_and_side():
    relay = _relay()
    relay._handle_inbound_frame(json.dumps({"type": "force_feedback", "tau": [1.0, 2.0]}))
    assert [list(m.effort) for m in relay.force_feedback_pub.msgs] == [[1.0, 2.0]]


def test_post_force_feedback_publishes_and_acks():
    relay = _relay(side="right")
    from factr_fastapi.factr_api import ForceFeedback

    ack = asyncio.run(relay.post_force_feedback(ForceFeedback(tau=[0.0, 1.0, -2.0])))

    assert ack.side == "right"
    assert ack.space == "joint"
    assert ack.dof == 3
    assert [list(m.effort) for m in relay.force_feedback_pub.msgs] == [[0.0, 1.0, -2.0]]
