"""FACTR-only Rerun diagnostics service.

This process consumes the two existing FACTR API WebSockets and publishes only
their telemetry frames. It owns its Rerun gRPC and web servers; no recording is
connected to, registered with, or discoverable through DFC's Rerun proxy.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

import yaml

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from .factr_rerun import FactrRerunPublisher


LOG = logging.getLogger("factr-rerun")
SIDE_URLS = {
    "left": "ws://127.0.0.1:5000/ws/left",
    "right": "ws://127.0.0.1:5001/ws/right",
}
CONFIG_DIR = (
    Path(__file__).resolve().parents[2] / "factr_teleop" / "factr_teleop" / "configs"
)
SIDE_CONFIGS = {
    "left": CONFIG_DIR / "factr_rizon_left.yaml",
    "right": CONFIG_DIR / "factr_rizon_right.yaml",
}


def configured_friction_gain(side: str) -> float | None:
    """Read the same static-friction gain file used by the leader at startup."""
    try:
        with SIDE_CONFIGS[side].open() as stream:
            config = yaml.safe_load(stream)
        gain = float(config["controller"]["static_friction_comp"]["gain"])
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        LOG.warning("could not read %s friction gain (%s)", side, exc)
        return None
    LOG.info("FACTR %s configured static-friction gain: %.6g", side, gain)
    return gain


RECONNECT_DELAY_S = 1.0


async def stream_side(
    side: str,
    url: str,
    publisher: FactrRerunPublisher,
    friction_gain: float | None,
) -> None:
    """Reconnect forever and forward only telemetry frames from one arm."""
    while True:
        try:
            async with connect(
                url,
                open_timeout=5.0,
                ping_interval=20.0,
                ping_timeout=20.0,
                max_size=4 * 1024 * 1024,
            ) as websocket:
                LOG.info("connected to FACTR %s telemetry at %s", side, url)
                async for raw in websocket:
                    try:
                        payload = json.loads(raw)
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if isinstance(payload, dict) and payload.get("type") == "telemetry":
                        if friction_gain is not None:
                            payload.setdefault("friction_gain", friction_gain)
                        publisher.publish_telemetry(side, payload)
        except asyncio.CancelledError:
            raise
        except (ConnectionClosed, OSError, TimeoutError) as exc:
            LOG.warning(
                "FACTR %s telemetry unavailable at %s (%s); retrying in %.0fs",
                side,
                url,
                exc,
                RECONNECT_DELAY_S,
            )
        await asyncio.sleep(RECONNECT_DELAY_S)


async def main() -> None:
    publisher = FactrRerunPublisher(LOG)
    if not publisher.enabled:
        raise RuntimeError("FACTR diagnostics Rerun server could not be started")
    friction_gains = {side: configured_friction_gain(side) for side in SIDE_URLS}
    await asyncio.gather(
        *(
            stream_side(side, url, publisher, friction_gains[side])
            for side, url in SIDE_URLS.items()
        )
    )


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    os.environ.setdefault("FACTR_RERUN_URL", "standalone")
    asyncio.run(main())


if __name__ == "__main__":
    run()
