"""Backward-compatible entry point for the integrated FACTR relay.

Readings and telemetry used to be split between ``factr_api`` HTTP routes and
this standalone, readings-only WebSocket server. They now share the typed
per-arm WebSockets hosted by :mod:`.factr_api`. Keep this module as an alias so
existing launch commands start the new relay rather than the obsolete protocol.
"""

from .factr_api import main


if __name__ == "__main__":
    main()
