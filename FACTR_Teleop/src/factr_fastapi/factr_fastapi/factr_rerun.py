# Publish live FACTR telemetry to Rerun.
#
# Teleop nodes publish live JSON samples on /factr_telemetry_<side>. This module
# streams them and the independent gravity/feedback gains straight into the Rerun
# viewer the DFC dashboard serves when explicitly enabled, as its OWN application
# (app id "factr-diagnostics") so the FACTR recording sits beside — never
# inside — DFC's "dual-flexiv-experiments" recording.
#
# Rerun publication is disabled by default. Set FACTR_RERUN_URL=standalone to
# serve an isolated FACTR-only gRPC source and web viewer, or set it to an explicit
# proxy URL to connect elsewhere. Disabled values retain WebSocket telemetry only.
#
# rerun-sdk is an OPTIONAL dependency of the relay: when it is not importable,
# or the sink cannot be created, the publisher disables itself with one warning
# and the relay keeps serving joint positions — telemetry visualization must
# never take down the control-critical API. Install rerun-sdk==0.33.1 into the
# API-only site-packages directory so its NumPy 2 dependency cannot affect teleop.
#
# (match the rerun version of the DFC environment serving the viewer).

import json
import os
import socket
import threading
import time

APP_ID = "factr-diagnostics"
#: One fixed recording: a respawned relay resumes the same recording instead of
#: stacking a new one in the viewer per restart.
RECORDING_ID = "factr-diagnostics"
DEFAULT_URL = "disabled"
STANDALONE_MODES = frozenset({"serve", "standalone", "isolated"})
DEFAULT_GRPC_PORT = 9877
DEFAULT_WEB_PORT = 9091
DEFAULT_MEMORY_LIMIT = "128MiB"
#: All samples share one timeline fed by CLOCK_MONOTONIC seconds-since-boot —
#: the teleop stamps its enable captures with the same clock, so gain ramps and
#: torque captures land at their true relative times.
TIMELINE = "monotonic"

#: Per-joint 7-vector fields of one live sample, each plotted separately under
#: factr/<side>/telemetry/.
_CAPTURE_FIELDS = (
    "raw_q_rad",
    "model_q_rad",
    "model_dq_rad_s",
    "home_error_rad",
    "joint_offsets_rad",
    "model_signs",
    "limit_torque_nm",
    "null_torque_nm",
    "gravity_torque_nm",
    "friction_torque_nm",
    "force_feedback_torque_nm",
    "applied_torque_nm",
)

_GAIN_FIELDS = (
    "grav_comp_gain",
    "grav_comp_gain_target",
    "friction_gain",
    "force_feedback_gain",
    "force_feedback_gain_target",
)

_TELEMETRY_TITLES = {
    "raw_q_rad": "Raw joint position (rad)",
    "model_q_rad": "Model joint position (rad)",
    "model_dq_rad_s": "Model joint velocity (rad/s)",
    "home_error_rad": "Home error (rad)",
    "joint_offsets_rad": "Joint offsets (rad)",
    "model_signs": "Model signs",
    "limit_torque_nm": "Limit torque (Nm)",
    "null_torque_nm": "Null-space torque (Nm)",
    "gravity_torque_nm": "Gravity torque (Nm)",
    "friction_torque_nm": "Friction torque (Nm)",
    "force_feedback_torque_nm": "Force-feedback torque (Nm)",
    "applied_torque_nm": "Applied torque (Nm)",
    "grav_comp_gain": "Gravity-compensation gain",
    "grav_comp_gain_target": "Gravity-compensation gain target",
    "friction_gain": "Static-friction gain",
    "force_feedback_gain": "Force-feedback gain",
    "force_feedback_gain_target": "Force-feedback gain target",
}

_READING_FIELDS = (
    "model_q_rad",
    "model_dq_rad_s",
    "home_error_rad",
    "raw_q_rad",
    "joint_offsets_rad",
    "model_signs",
)

_TORQUE_FIELDS = (
    "applied_torque_nm",
    "force_feedback_torque_nm",
    "gravity_torque_nm",
    "friction_torque_nm",
    "null_torque_nm",
    "limit_torque_nm",
)

_TELEMETRY_FIELDS = _READING_FIELDS + _GAIN_FIELDS + _TORQUE_FIELDS

# Passive Dynamixel diagnostics derived from the normal control read.
_STATUS_DYNAMIXEL_VIEWS = (
    ("dynamixel/counters/comm_retries", "Communication retries (cumulative)"),
    ("dynamixel/counters/comm_failures", "Communication failures (cumulative)"),
    ("dynamixel/counters/status_alerts", "Servo status alerts (cumulative)"),
    ("dynamixel/counters/position_jumps", "Implausible position jumps (cumulative)"),
)

_MONITORING_DYNAMIXEL_VIEWS = (
    ("dynamixel/control/raw_position_ticks", "Control-read raw position (ticks)"),
    ("dynamixel/control/raw_velocity_ticks", "Control-read raw velocity (ticks)"),
)

_DYNAMIXEL_VIEWS = _STATUS_DYNAMIXEL_VIEWS + _MONITORING_DYNAMIXEL_VIEWS

_SIDES = ("left", "right")
_EVENT_LEDGER_SIZE = 32

_LOCK = threading.Lock()
_PUBLISHER = None


def shared_publisher(logger):
    """The process-wide publisher (both relay nodes feed one recording)."""
    global _PUBLISHER
    with _LOCK:
        if _PUBLISHER is None:
            _PUBLISHER = FactrRerunPublisher(logger)
        return _PUBLISHER


def _signal_views(rrb, side: str, fields):
    """One named time-series plot per field for one leader."""
    return [
        rrb.TimeSeriesView(
            origin=f"/factr/{side}/telemetry/{field}",
            name=_TELEMETRY_TITLES[field],
        )
        for field in fields
    ]


def _telemetry_column(rrb, side: str):
    """One leader column split into focused diagnostic tabs."""
    current_readings = rrb.Vertical(
        *_signal_views(rrb, side, _READING_FIELDS),
        name="Current readings",
    )
    gains = rrb.Vertical(
        *_signal_views(rrb, side, _GAIN_FIELDS),
        name="Gains",
    )
    status = rrb.Vertical(
        *[
            rrb.TimeSeriesView(
                origin=f"/factr/{side}/telemetry/{path}", name=title
            )
            for path, title in _STATUS_DYNAMIXEL_VIEWS
        ],
        rrb.TextLogView(
            origin=f"/factr/events/{side}", name="Dynamixel alerts"
        ),
        rrb.TextDocumentView(
            origin=f"/factr/event_ledger/{side}", name="Recent events"
        ),
        name="Status",
    )
    monitoring = rrb.Vertical(
        *[
            rrb.TimeSeriesView(
                origin=f"/factr/{side}/telemetry/{path}", name=title
            )
            for path, title in _MONITORING_DYNAMIXEL_VIEWS
        ],
        name="Monitoring",
    )
    torques = rrb.Vertical(
        *_signal_views(rrb, side, _TORQUE_FIELDS),
        name="Torques",
    )
    return rrb.Tabs(
        current_readings, gains, status, monitoring, torques,
        active_tab=0,
        name=f"{side.capitalize()} leader",
    )

def _blueprint(rrb):
    """Two leader columns, each split into five diagnostic tabs."""
    return rrb.Blueprint(
        rrb.Horizontal(
            *[_telemetry_column(rrb, side) for side in _SIDES],
            column_shares=[1, 1],
            name="FACTR leader diagnostics",
        ),
        collapse_panels=True,
    )


def _port_listening(port: int) -> bool:
    """Whether a local TCP listener already owns ``port``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _await_port(port: int, timeout_s: float = 5.0) -> bool:
    """Wait for an asynchronously started Rerun server to accept connections."""
    deadline = time.monotonic() + timeout_s
    while not _port_listening(port):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


class FactrRerunPublisher:
    """Best-effort Rerun sink for the telemetry both relay nodes receive.

    Every public method is exception-proof: a missing SDK, a bad payload, or a
    dead sink degrades to a (throttled) warning, never an error the ROS
    callback would propagate.
    """

    def __init__(self, logger):
        self._logger = logger
        self._rr = None
        self._rec = None
        #: captured_monotonic_ns of the calibration snapshot last logged per
        #: side — the snapshot is immutable per teleop boot, so it is logged
        #: once per boot, not at its 2 Hz republish rate.
        self._calib_logged = {}
        #: newest enable-capture stamp already logged per side.
        self._capture_logged = {}
        #: Newest event sequence per (side, driver session, port). Teleop retains a
        #: short event ring in every telemetry frame, so this prevents duplicates
        #: while still surviving a WebSocket reconnect.
        self._event_logged = {}
        self._event_history = {side: [] for side in _SIDES}
        self._last_warn = 0.0
        url = os.environ.get("FACTR_RERUN_URL", DEFAULT_URL)
        mode = url.strip().lower()
        if mode in {"", "disabled", "off", "none"}:
            logger.info(
                "FACTR telemetry Rerun publisher disabled; telemetry remains "
                "available over WebSocket"
            )
            return
        try:
            import rerun as rr
            import rerun.blueprint as rrb
        except ImportError as exc:
            logger.warning(
                f"rerun-sdk not importable ({exc}) — FACTR telemetry will NOT "
                "be published to the viewer (the relay itself is unaffected)"
            )
            return
        rec = None
        try:
            rec = rr.RecordingStream(APP_ID, recording_id=RECORDING_ID)
            blueprint = _blueprint(rrb)
            if mode in STANDALONE_MODES:
                grpc_port = int(
                    os.environ.get("FACTR_RERUN_GRPC_PORT", DEFAULT_GRPC_PORT)
                )
                web_port = int(
                    os.environ.get("FACTR_RERUN_WEB_PORT", DEFAULT_WEB_PORT)
                )
                memory_limit = os.environ.get(
                    "FACTR_RERUN_MEMORY_LIMIT", DEFAULT_MEMORY_LIMIT
                )
                for port, label in ((grpc_port, "gRPC"), (web_port, "web")):
                    if _port_listening(port):
                        raise RuntimeError(
                            f"FACTR Rerun {label} port {port} is already in use"
                        )
                grpc_uri = rec.serve_grpc(
                    grpc_port=grpc_port,
                    default_blueprint=blueprint,
                    server_memory_limit=memory_limit,
                    cors_allow_origin=["*"],
                )
                if not _await_port(grpc_port):
                    raise RuntimeError(f"FACTR Rerun gRPC port {grpc_port} did not open")
                rr.serve_web_viewer(
                    web_port=web_port,
                    open_browser=False,
                    connect_to=grpc_uri,
                )
                if not _await_port(web_port):
                    raise RuntimeError(f"FACTR Rerun web port {web_port} did not open")
                destination = f"http://127.0.0.1:{web_port} (gRPC {grpc_port})"
            else:
                rec.connect_grpc(url, default_blueprint=blueprint)
                destination = url
        except Exception as exc:  # noqa: BLE001 - any sink failure -> disabled
            if mode in STANDALONE_MODES:
                try:
                    rr.rerun_shutdown()
                except Exception:  # noqa: BLE001 - best-effort partial-server cleanup
                    pass
            logger.warning(
                f"could not open the Rerun sink at {url} ({exc}) — FACTR "
                "telemetry will NOT be published to the viewer"
            )
            return
        self._rr = rr
        self._rec = rec
        logger.info(
            f"FACTR telemetry -> Rerun at {destination} "
            f"(app id {APP_ID!r})"
        )

    @property
    def enabled(self) -> bool:
        return self._rec is not None

    # -- publishing -----------------------------------------------------------

    def publish_gain(self, side: str, kind: str, gain: float) -> None:
        """Log one live component gain (10 Hz from the teleop)."""
        if self._rec is None:
            return
        try:
            self._rec.set_time(TIMELINE, duration=time.monotonic())
            self._rec.log(
                f"factr/{side}/gain/{kind}", self._rr.Scalars([float(gain)])
            )
        except Exception:  # noqa: BLE001 - never propagate into the ROS callback
            self._warn_throttled("gain")

    def publish_telemetry(self, side: str, payload: dict) -> None:
        """Log one live ``/factr_telemetry_<side>`` sample."""
        if self._rec is None:
            return
        try:
            stamp = payload.get("stamp_monotonic_ns")
            if not isinstance(stamp, int):
                return
            self._rec.set_time(TIMELINE, duration=stamp / 1e9)
            for field in _CAPTURE_FIELDS:
                vec = payload.get(field)
                if isinstance(vec, list) and vec:
                    self._rec.log(
                        f"factr/{side}/telemetry/{field}",
                        self._rr.Scalars([float(x) for x in vec]),
                    )
            for field in _GAIN_FIELDS:
                value = payload.get(field)
                if value is not None:
                    self._rec.log(
                        f"factr/{side}/telemetry/{field}",
                        self._rr.Scalars([float(value)]),
                    )
            self._log_dynamixel(side, payload.get("dynamixel") or [])
        except Exception:  # noqa: BLE001 - never propagate into the ROS callback
            self._warn_throttled("telemetry")

    # -- internals ------------------------------------------------------------

    def _log_dynamixel(self, side: str, drivers: list) -> None:
        """Log normal-read diagnostics and each retained event exactly once."""
        drivers = [driver for driver in drivers if isinstance(driver, dict)]
        if not drivers:
            return

        def ordered_read_values(source, field):
            pairs = []
            for driver in drivers:
                ids = driver.get("servo_ids") or []
                read = (driver.get("latest_reads") or {}).get(source) or {}
                values = read.get(field) or []
                pairs.extend(
                    (int(dxl_id), float(value))
                    for dxl_id, value in zip(ids, values)
                )
            return [value for _, value in sorted(pairs)]

        for source in ("control",):
            for field in ("raw_position_ticks", "raw_velocity_ticks"):
                values = ordered_read_values(source, field)
                if values:
                    self._rec.log(
                        f"factr/{side}/telemetry/dynamixel/{source}/{field}",
                        self._rr.Scalars(values),
                    )

        counters = {
            "comm_retries": "comm_retry_count",
            "comm_failures": "comm_failure_count",
            "status_alerts": "status_alert_count",
            "position_jumps": "position_jump_count",
        }
        for path, field in counters.items():
            value = sum(int(driver.get(field, 0)) for driver in drivers)
            self._rec.log(
                f"factr/{side}/telemetry/dynamixel/counters/{path}",
                self._rr.Scalars([float(value)]),
            )

        ledger_changed = False
        for driver in drivers:
            session = driver.get("session")
            port = str(driver.get("port", "unknown"))
            watermark_key = (side, session, port)
            newest = self._event_logged.get(watermark_key, 0)
            for event in driver.get("events") or []:
                sequence = event.get("sequence")
                stamp = event.get("stamp_monotonic_ns")
                if (
                    not isinstance(sequence, int)
                    or sequence <= newest
                    or not isinstance(stamp, int)
                ):
                    continue
                newest = sequence
                self._rec.set_time(TIMELINE, duration=stamp / 1e9)
                self._rec.log(
                    f"factr/events/{side}",
                    self._rr.TextLog(
                        f"{side} {event.get('kind', 'dynamixel_event')}: "
                        + json.dumps(event, sort_keys=True, separators=(",", ":"))
                    ),
                )
                history = self._event_history.setdefault(side, [])
                history.append(event)
                del history[:-_EVENT_LEDGER_SIZE]
                ledger_changed = True
            self._event_logged[watermark_key] = newest
        if ledger_changed:
            self._log_event_ledger(side)

    def _log_event_ledger(self, side: str) -> None:
        """Keep recent exact event JSON static so time-series GC cannot evict it."""
        history = self._event_history.get(side) or []
        markdown = (
            f"## Recent Dynamixel events · {side}\n\n"
            f"Last {len(history)} events (oldest first).\n\n"
            "```json\n"
            + json.dumps(history, indent=2, sort_keys=True)
            + "\n```"
        )
        self._rec.log(
            f"factr/event_ledger/{side}",
            self._rr.TextDocument(markdown, media_type=self._rr.MediaType.MARKDOWN),
            static=True,
        )

    def _log_calibration(self, side: str, payload: dict) -> None:
        snapshot = {k: v for k, v in payload.items() if k != "enable_samples"}
        stamp = snapshot.get("captured_monotonic_ns")
        if not snapshot or stamp == self._calib_logged.get(side):
            return
        self._calib_logged[side] = stamp
        md = (
            f"## FACTR {side} calibration\n\n"
            "```json\n" + json.dumps(snapshot, indent=2, sort_keys=True) + "\n```"
        )
        # Static: survives the proxy's memory-limit eviction of old timed data.
        self._rec.log(
            f"factr/{side}/calibration",
            self._rr.TextDocument(md, media_type=self._rr.MediaType.MARKDOWN),
            static=True,
        )

    def _log_enable_samples(self, side: str, samples: list) -> None:
        """Log each capture tick exactly once, at its true monotonic stamp.

        The teleop clears its capture buffer when a fresh enable starts, so a
        stamp below the side's watermark simply never reappears; every kept
        sample carries a strictly newer stamp.
        """
        newest = self._capture_logged.get(side, -1)
        for sample in samples:
            stamp = sample.get("stamp_monotonic_ns")
            if not isinstance(stamp, int) or stamp <= newest:
                continue
            newest = stamp
            self._rec.set_time(TIMELINE, duration=stamp / 1e9)
            for field in _CAPTURE_FIELDS:
                vec = sample.get(field)
                if isinstance(vec, list) and vec:
                    self._rec.log(
                        f"factr/{side}/enable/{field}",
                        self._rr.Scalars([float(x) for x in vec]),
                    )
            gain = sample.get("grav_comp_gain")
            if gain is not None:
                self._rec.log(
                    f"factr/{side}/enable/grav_comp_gain",
                    self._rr.Scalars([float(gain)]),
                )
        self._capture_logged[side] = newest

    def _warn_throttled(self, what: str) -> None:
        now = time.monotonic()
        if now - self._last_warn < 30.0:
            return
        self._last_warn = now
        self._logger.warning(
            f"publishing FACTR {what} to Rerun failed (throttled warning; "
            "the relay itself is unaffected)"
        )
