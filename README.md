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
  dynamics. Its arm-axis directions are derived from DFC's two sign stages.

DFC's `conf/factr/*.yaml` is the measured-calibration source. Each `leaders.<side>`
object stores the raw Dynamixel→DFC convention and the mechanism's DFC→FACTR
axis signs. The DFC supervisor injects that object into the managed teleop at launch.
DFC owns every arm-joint direction: FACTR composes `raw_to_dfc` and `dfc_to_factr`
signs for both model-state conversion and motor-torque conversion. FACTR's hardware
YAML owns only the gripper hardware sign and the authoritative `model_home_q_rad`.
The per-leader `joint_torque_gain` vector is also owned by DFC. FACTR multiplies it
with the final arm-torque command, after all controller terms are composed and before
motor-sign conversion. Each gain is continuous from `0.0` (zero torque) to `1.0`.
FACTR derives its native Dynamixel model offsets directly from `raw_to_dfc`; no
second home-pose-derived offset is persisted or accepted in the DFC contract.
The leader may start in any pose. No launch-pose calibration occurs.

Null-space regulation starts disabled even when leader torque is energized. During
collection, DFC atomically posts `enabled` plus the task's canonical
`collection.start_qpos` to `/null_space_<side>`. FACTR converts that configured task
vector to model axes and ramps the null-space gain on; collection ramps it off again
during teardown. The leader's measured pose at activation is not used as the target.

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
