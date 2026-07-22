# Publish FACTR diagnostics to Rerun.
#
# FACTR owns its diagnostics: the teleop nodes publish a JSON snapshot on
# /factr_diagnostics_<side> (the immutable startup-calibration capture plus the
# post-enable control-tick samples) and the live master gain on
# /factr_gain_state_<side>. This module streams both straight into the Rerun
# viewer the DFC dashboard already serves, as its OWN application
# (app id "factr-diagnostics") so the FACTR recording sits beside — never
# inside — DFC's "dual-flexiv-experiments" recording. DFC does not poll any
# diagnostics endpoint anymore; the viewer is the diagnostics surface.
#
# The sink defaults to the dashboard's gRPC proxy on this machine
# (rerun+http://127.0.0.1:9876/proxy); FACTR_RERUN_URL overrides it (the DFC
# session daemon exports it when launching these processes, following its
# DFC_DASHBOARD_GRPC_PORT). The proxy buffers what it receives, so a viewer
# page opened later still sees the retained history. If the dashboard's Rerun
# servers restart (its "Reset services"), this client's stream is severed —
# restart the FACTR servers (or just the relay) to reconnect.
#
# rerun-sdk is an OPTIONAL dependency of the relay: when it is not importable,
# or the sink cannot be created, the publisher disables itself with one warning
# and the relay keeps serving joint positions — diagnostics visualization must
# never take down the control-critical API. Install with:
#
#   /usr/bin/python3 -m pip install --user rerun-sdk==0.34.1
#
# (match the rerun version of the DFC environment serving the viewer).

import json
import os
import threading
import time

APP_ID = "factr-diagnostics"
#: One fixed recording: a respawned relay resumes the same recording instead of
#: stacking a new one in the viewer per restart.
RECORDING_ID = "factr-diagnostics"
DEFAULT_URL = "rerun+http://127.0.0.1:9876/proxy"
#: All samples share one timeline fed by CLOCK_MONOTONIC seconds-since-boot —
#: the teleop stamps its enable captures with the same clock, so gain ramps and
#: torque captures land at their true relative times.
TIMELINE = "monotonic"

#: Per-joint 7-vector fields of one enable-capture sample, each plotted as its
#: own series under factr/<side>/enable/.
_CAPTURE_FIELDS = (
    "raw_q_rad",
    "model_q_rad",
    "model_dq_rad_s",
    "home_error_rad",
    "limit_torque_nm",
    "null_torque_nm",
    "gravity_torque_nm",
    "friction_torque_nm",
    "total_torque_pre_gain_nm",
    "applied_torque_nm",
)

_SIDES = ("left", "right")

_LOCK = threading.Lock()
_PUBLISHER = None


def shared_publisher(logger):
    """The process-wide publisher (both relay nodes feed one recording)."""
    global _PUBLISHER
    with _LOCK:
        if _PUBLISHER is None:
            _PUBLISHER = FactrRerunPublisher(logger)
        return _PUBLISHER


def _blueprint(rrb):
    """Calibration snapshots up top, live gain and enable captures below."""
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*[
                rrb.TextDocumentView(
                    origin=f"/factr/{side}/calibration", name=f"{side} calibration"
                )
                for side in _SIDES
            ]),
            rrb.Horizontal(*[
                rrb.TimeSeriesView(origin=f"/factr/{side}/gain", name=f"{side} force gain")
                for side in _SIDES
            ]),
            rrb.Horizontal(*[
                rrb.TimeSeriesView(
                    origin=f"/factr/{side}/enable", name=f"{side} enable capture"
                )
                for side in _SIDES
            ]),
            row_shares=[2, 1, 3],
        ),
        collapse_panels=True,
    )


class FactrRerunPublisher:
    """Best-effort Rerun sink for the diagnostics both relay nodes receive.

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
        self._last_warn = 0.0
        url = os.environ.get("FACTR_RERUN_URL", DEFAULT_URL)
        try:
            import rerun as rr
            import rerun.blueprint as rrb
        except ImportError as exc:
            logger.warning(
                f"rerun-sdk not importable ({exc}) — FACTR diagnostics will NOT "
                "be published to the viewer (the relay itself is unaffected)"
            )
            return
        try:
            rec = rr.RecordingStream(APP_ID, recording_id=RECORDING_ID)
            rec.connect_grpc(url, default_blueprint=_blueprint(rrb))
        except Exception as exc:  # noqa: BLE001 - any sink failure -> disabled
            logger.warning(
                f"could not open the Rerun sink at {url} ({exc}) — FACTR "
                "diagnostics will NOT be published to the viewer"
            )
            return
        self._rr = rr
        self._rec = rec
        logger.info(f"FACTR diagnostics -> Rerun at {url} (app id {APP_ID!r})")

    @property
    def enabled(self) -> bool:
        return self._rec is not None

    # -- publishing -----------------------------------------------------------

    def publish_gain(self, side: str, gain: float) -> None:
        """Log the live master output gain (10 Hz from the teleop)."""
        if self._rec is None:
            return
        try:
            self._rec.set_time(TIMELINE, duration=time.monotonic())
            self._rec.log(f"factr/{side}/gain", self._rr.Scalars([float(gain)]))
        except Exception:  # noqa: BLE001 - never propagate into the ROS callback
            self._warn_throttled("gain")

    def publish_diagnostics(self, side: str, payload: dict) -> None:
        """Log one /factr_diagnostics_<side> message: snapshot + new captures."""
        if self._rec is None:
            return
        try:
            self._log_calibration(side, payload)
            self._log_enable_samples(side, payload.get("enable_samples") or [])
        except Exception:  # noqa: BLE001 - never propagate into the ROS callback
            self._warn_throttled("diagnostics")

    # -- internals ------------------------------------------------------------

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
            gain = sample.get("force_gain")
            if gain is not None:
                self._rec.log(
                    f"factr/{side}/enable/force_gain",
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
