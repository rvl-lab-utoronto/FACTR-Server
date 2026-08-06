"""Tests for the typed FACTR WebSocket stream and its inbound force-feedback path."""

import asyncio
import json
import sys
import threading
from types import MethodType, ModuleType, SimpleNamespace

from fastapi import WebSocketDisconnect
from rclpy.clock import Clock

from factr_fastapi.factr_api import FactrAPI
from factr_fastapi.factr_rerun import FactrRerunPublisher


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
        self.infos = []

    def warning(self, msg, **kwargs):
        self.warnings.append(msg)

    def info(self, msg, **kwargs):
        self.infos.append(msg)


def test_rerun_diagnostics_publisher_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("FACTR_RERUN_URL", raising=False)
    logger = FakeLogger()

    publisher = FactrRerunPublisher(logger)

    assert publisher.enabled is False
    assert any("disabled" in message for message in logger.infos)


def test_rerun_blueprint_splits_each_leader_into_five_tabs():
    from factr_fastapi import factr_rerun

    class Node:
        def __init__(self, *children, **kwargs):
            self.children = children
            self.kwargs = kwargs

    fake_rrb = SimpleNamespace(
        Blueprint=Node, Horizontal=Node, Vertical=Node, Tabs=Node,
        TimeSeriesView=Node, TextLogView=Node, TextDocumentView=Node,
    )
    blueprint = factr_rerun._blueprint(fake_rrb)

    horizontal = blueprint.children[0]
    assert len(horizontal.children) == 2
    for side, tabs in zip(("left", "right"), horizontal.children):
        assert tabs.kwargs["name"] == f"{side.capitalize()} leader"
        assert tabs.kwargs["active_tab"] == 0
        current, gains, status, monitoring, torques = tabs.children
        assert [tab.kwargs["name"] for tab in tabs.children] == [
            "Current readings", "Gains", "Status", "Monitoring", "Torques"
        ]
        assert [view.kwargs["origin"] for view in current.children] == [
            f"/factr/{side}/telemetry/{field}"
            for field in factr_rerun._READING_FIELDS
        ]
        assert [view.kwargs["origin"] for view in gains.children] == [
            f"/factr/{side}/telemetry/{field}"
            for field in factr_rerun._GAIN_FIELDS
        ]
        assert [view.kwargs["origin"] for view in status.children[:-2]] == [
            f"/factr/{side}/telemetry/{path}"
            for path, _title in factr_rerun._STATUS_DYNAMIXEL_VIEWS
        ]
        assert status.children[-2].kwargs["origin"] == f"/factr/events/{side}"
        assert status.children[-1].kwargs["origin"] == f"/factr/event_ledger/{side}"
        assert [view.kwargs["origin"] for view in monitoring.children] == [
            f"/factr/{side}/telemetry/{path}"
            for path, _title in factr_rerun._MONITORING_DYNAMIXEL_VIEWS
        ]
        assert [view.kwargs["origin"] for view in torques.children] == [
            f"/factr/{side}/telemetry/{field}"
            for field in factr_rerun._TORQUE_FIELDS
        ]


def test_rerun_logs_retained_dynamixel_event_only_once():
    class FakeRecording:
        def __init__(self):
            self.logs = []
            self.times = []

        def log(self, path, value, **kwargs):
            self.logs.append((path, value, kwargs))

        def set_time(self, timeline, **kwargs):
            self.times.append((timeline, kwargs))

    class Value:
        def __init__(self, value):
            self.value = value

    publisher = FactrRerunPublisher.__new__(FactrRerunPublisher)
    publisher._rec = FakeRecording()
    publisher._rr = SimpleNamespace(
        Scalars=Value,
        TextLog=Value,
        TextDocument=lambda value, **_kwargs: Value(value),
        MediaType=SimpleNamespace(MARKDOWN="markdown"),
    )
    publisher._event_logged = {}
    publisher._event_history = {"left": [], "right": []}
    driver = {
        "session": 123,
        "port": "/dev/test",
        "servo_ids": [1],
        "comm_retry_count": 1,
        "comm_failure_count": 0,
        "status_alert_count": 1,
        "position_jump_count": 1,
        "latest_reads": {
            "control": {
                "raw_position_ticks": [10],
                "raw_velocity_ticks": [2],
            }
        },
        "events": [{
            "sequence": 1,
            "session": 123,
            "stamp_monotonic_ns": 1_000_000_000,
            "kind": "status_alert",
            "health": [{
                "id": 1,
                "hw_error": 8,
                "input_voltage": 119,
                "present_current": -12,
                "realtime_tick": 42,
                "torque_enable": 0,
                "temperature": 30,
                "present_position": 10,
            }],
        }],
    }

    publisher._log_dynamixel("left", [driver])
    publisher._log_dynamixel("left", [driver])

    event_logs = [log for log in publisher._rec.logs if log[0] == "factr/events/left"]
    assert len(event_logs) == 1
    voltage_logs = [
        log for log in publisher._rec.logs
        if log[0].endswith("dynamixel/event/input_voltage_v/id_1")
    ]
    assert len(voltage_logs) == 1
    current_logs = [
        log for log in publisher._rec.logs
        if log[0].endswith("dynamixel/event/present_current_raw/id_1")
    ]
    assert len(current_logs) == 1
    assert current_logs[0][1].value == [-12.0]
    assert voltage_logs[0][1].value == [11.9]
    ledger_logs = [
        log for log in publisher._rec.logs
        if log[0] == "factr/event_ledger/left"
    ]
    assert len(ledger_logs) == 1
    assert ledger_logs[0][2] == {"static": True}


def test_rerun_diagnostics_publisher_serves_isolated_ports(monkeypatch):
    from factr_fastapi import factr_rerun

    calls = []

    class FakeRecording:
        def __init__(self, app_id, recording_id):
            calls.append(("recording", app_id, recording_id))

        def serve_grpc(self, **kwargs):
            calls.append(("grpc", kwargs))
            return "rerun+http://127.0.0.1:9877/proxy"

    fake_rr = ModuleType("rerun")
    fake_rr.__path__ = []
    fake_rr.RecordingStream = FakeRecording
    fake_rr.serve_web_viewer = lambda **kwargs: calls.append(("web", kwargs))
    fake_rr.rerun_shutdown = lambda: calls.append(("shutdown",))
    fake_rrb = ModuleType("rerun.blueprint")
    monkeypatch.setitem(sys.modules, "rerun", fake_rr)
    monkeypatch.setitem(sys.modules, "rerun.blueprint", fake_rrb)
    monkeypatch.setattr(factr_rerun, "_blueprint", lambda _rrb: "blueprint")
    monkeypatch.setattr(factr_rerun, "_port_listening", lambda _port: False)
    monkeypatch.setattr(factr_rerun, "_await_port", lambda _port: True)
    monkeypatch.setenv("FACTR_RERUN_URL", "standalone")
    monkeypatch.setenv("FACTR_RERUN_GRPC_PORT", "9877")
    monkeypatch.setenv("FACTR_RERUN_WEB_PORT", "9091")
    logger = FakeLogger()

    publisher = FactrRerunPublisher(logger)

    assert publisher.enabled is True
    grpc = next(call[1] for call in calls if call[0] == "grpc")
    web = next(call[1] for call in calls if call[0] == "web")
    assert grpc["grpc_port"] == 9877
    assert grpc["server_memory_limit"] == "128MiB"
    assert grpc["cors_allow_origin"] == ["*"]
    assert web == {
        "web_port": 9091,
        "open_browser": False,
        "connect_to": "rerun+http://127.0.0.1:9877/proxy",
    }
    assert not any(call[0] == "shutdown" for call in calls)


def _relay(side="left"):
    relay = SimpleNamespace(
        side=side,
        lock=threading.Lock(),
        joint_pos=[0.0, 1.0, 2.0],
        joint_pos_source_stamp_ns=123,
        telemetry={},
        telemetry_version=0,
        grav_comp_gain=0.0,
        grav_comp_gain_target=0.0,
        force_feedback_gain=0.0,
        force_feedback_gain_target=0.0,
        force_feedback_pub=FakePublisher(),
        grav_comp_gain_pub=FakePublisher(),
        force_feedback_gain_pub=FakePublisher(),
    )
    for name in (
        "stream",
        "_stream_sender",
        "_handle_inbound_frame",
        "_telemetry_payload",
        "_publish_force_feedback",
        "post_force_feedback",
        "enable_force_feedback",
        "disable_force_feedback",
        "enable_grav_comp",
        "disable_grav_comp",
        "get_status",
        "_publish_grav_comp_gain_target",
        "_publish_force_feedback_gain_target",
        "_update_grav_comp_gain_state",
        "_update_force_feedback_gain_state",
    ):
        setattr(relay, name, MethodType(getattr(FactrAPI, name), relay))
    logger = FakeLogger()
    relay.get_logger = lambda: logger
    relay.get_clock = lambda: Clock()
    return relay


def _telemetry():
    vectors = {
        name: [0.0, 0.1, 0.2]
        for name in (
            "raw_q_rad", "model_q_rad", "model_dq_rad_s", "home_error_rad",
            "joint_offsets_rad", "model_signs", "limit_torque_nm",
            "null_torque_nm", "gravity_torque_nm", "friction_torque_nm",
            "force_feedback_torque_nm", "applied_torque_nm",
        )
    }
    return {
        "stamp_monotonic_ns": 123,
        **vectors,
        "grav_comp_gain": 0.2,
        "grav_comp_gain_target": 1.0,
        "friction_gain": 0.2,
        "force_feedback_gain": 0.3,
        "force_feedback_gain_target": 0.0,
    }


def test_stream_sends_telemetry_then_reading():
    relay = _relay()
    relay.telemetry = _telemetry()
    relay.telemetry["dynamixel"] = [{
        "session": 1,
        "port": "/dev/test",
        "events": [{"sequence": 1, "kind": "position_jump"}],
    }]
    relay.telemetry_version = 1
    websocket = FakeWebSocket()

    asyncio.run(relay.stream(websocket))

    assert websocket.accepted is True
    telemetry, reading = websocket.frames
    assert telemetry["type"] == "telemetry"
    assert telemetry["side"] == "left"
    assert telemetry["raw_q_rad"] == [0.0, 0.1, 0.2]
    assert telemetry["dynamixel"][0]["events"][0]["kind"] == "position_jump"
    assert reading == {
        "type": "reading",
        "side": "left",
        "joint_pos": [0.0, 1.0, 2.0],
        "source_stamp_ns": 123,
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


def test_force_feedback_toggle_controls_only_its_independent_gain():
    relay = _relay()
    frame = json.dumps({"type": "force_feedback", "tau": [1.0, 2.0]})

    relay._handle_inbound_frame(frame)
    assert [list(m.effort) for m in relay.force_feedback_pub.msgs] == [[1.0, 2.0]]

    enabled = asyncio.run(relay.enable_force_feedback())
    assert enabled.force_feedback_enabled is True
    assert enabled.gain_target == 1.0
    assert relay.force_feedback_gain_target == 1.0
    assert relay.force_feedback_gain_pub.msgs[-1].data == 1.0
    assert relay.grav_comp_gain_target == 0.0

    disabled = asyncio.run(relay.disable_force_feedback())
    assert disabled.force_feedback_enabled is False
    assert disabled.gain_target == 0.0
    assert relay.force_feedback_gain_target == 0.0
    assert relay.force_feedback_gain_pub.msgs[-1].data == 0.0
    assert relay.grav_comp_gain_target == 0.0


def test_grav_comp_toggle_does_not_change_feedback_gain():
    relay = _relay()
    relay.force_feedback_gain_target = 1.0

    enabled = asyncio.run(relay.enable_grav_comp())
    assert enabled.gain_target == 1.0
    assert relay.grav_comp_gain_target == 1.0
    assert relay.grav_comp_gain_pub.msgs[-1].data == 1.0
    assert relay.force_feedback_gain_target == 1.0

    disabled = asyncio.run(relay.disable_grav_comp())
    assert disabled.gain_target == 0.0
    assert relay.grav_comp_gain_target == 0.0
    assert relay.force_feedback_gain_target == 1.0


def test_telemetry_feeds_websocket_cache():
    relay = _relay()
    relay._update_telemetry = MethodType(FactrAPI._update_telemetry, relay)
    payload = _telemetry()

    relay._update_telemetry(SimpleNamespace(data=json.dumps(payload)))

    assert relay.telemetry == {**payload, "dynamixel": []}
    assert relay.telemetry_version == 1


def test_independent_gains_feed_status_cache():
    relay = _relay()

    relay._update_grav_comp_gain_state(SimpleNamespace(data=0.75))
    relay._update_force_feedback_gain_state(SimpleNamespace(data=1.0))
    status = asyncio.run(relay.get_status())

    assert status.grav_comp_gain == 0.75
    assert status.force_feedback_gain == 1.0
    assert status.grav_comp_enabled is False
    assert status.force_feedback_enabled is True
