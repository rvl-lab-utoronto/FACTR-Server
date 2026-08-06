# FACTR-Server

## Joint conventions

There are three deliberately separate representations of a leader pose:

- **Raw FACTR hardware** is the untouched Dynamixel reading. WebSocket `reading` frames
  expose it, and DFC publishes it as `factr/raw/<side>` and stores it as
  `observation.factr_raw.<side>`.
- **DFC/Rizon** is the canonical convention used by teleoperation, the viewer, and
  training. DFC converts each raw FACTR sample once when it receives the sample, publishes
  it as `factr/<side>`, and stores it as `observation.factr.<side>`.
- **FACTR model** is used only inside FACTR for gravity compensation and other leader-side
  dynamics. It follows the FACTR mechanism and URDF, not the Rizon joint convention.

DFC's `conf/factr/*.yaml` is the single calibration source. Each `leaders.<side>`
object stores the raw Dynamixel→DFC convention, canonical DFC home, and affine
DFC→FACTR transform. The DFC supervisor injects that object into the managed teleop
at launch; FACTR derives its dynamics-model signs and offsets and rejects a mismatch
with the physical mechanism. FACTR's hardware YAML contains no calibration copy.
DFC straight-up home is `[0,0,0,0,0,0,0]`; the corresponding FACTR model reference is
`[0,0,0,pi/2,0,0,0]`. The explicit joint-4 offset is intentional.
The leader may start in any pose. No launch-pose calibration occurs.

## Live telemetry

The teleop nodes publish live model/control samples on `/factr_telemetry_<side>` at
50 Hz. The API relay sends them as typed `telemetry` frames on the same WebSocket
as raw `reading` frames. Telemetry includes raw/model position, model velocity,
home error, derived model offsets/signs, every composed torque term, applied torque,
and both independent gain values and targets. Optional Rerun publication under the application id
`factr-diagnostics` is disabled by default; set `FACTR_RERUN_URL` to an explicit
gRPC proxy URL to opt in. `rerun-sdk` is an optional dependency of the relay
(`/usr/bin/python3 -m pip install --user rerun-sdk==0.34.1`, matching the DFC
environment's version): without it the relay logs one warning and serves everything
else as usual.
