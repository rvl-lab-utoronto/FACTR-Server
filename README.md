# FACTR-Server

## Joint conventions

There are three deliberately separate representations of a leader pose:

- **Raw FACTR hardware** is the untouched Dynamixel reading. DFC publishes it as
  `factr/raw/<side>` and stores it as `observation.factr_raw.<side>`.
- **DFC/Rizon** is the canonical convention used by teleoperation, the viewer, and
  training. DFC converts each raw FACTR sample once when it receives the sample, publishes
  it as `factr/<side>`, and stores it as `observation.factr.<side>`.
- **FACTR model** is used only inside FACTR for gravity compensation and other leader-side
  dynamics. It follows the FACTR mechanism and URDF, not the Rizon joint convention.

The FACTR arm YAML is the single calibration source. It records the complete raw
Dynamixel→DFC convention (`dfc_raw_offsets_deg`, sign flips, wrapping, trailing-field
handling, and gripper endpoints), the distinct DFC and FACTR reference coordinates,
the affine conversion
`q_factr = dfc_to_factr_signs * q_dfc + dfc_to_factr_offset_rad`, and the measured raw
Dynamixel→FACTR `joint_offsets`. These values are validated at startup. DFC reads only
the raw→DFC part of this leader-owned contract, served by the API relay at
`GET /calibration_<side>`, and does not keep or push another copy.
DFC straight-up home is `[0,0,0,0,0,0,0]`; the corresponding FACTR model reference is
`[0,0,0,pi/2,0,0,0]`. The explicit joint-4 offset is intentional.
The saved offsets are reused at every launch; the leader may start in any pose. Its first
reading is captured for diagnostics only and never recalibrates or gates gravity compensation.

## Diagnostics

The teleop nodes publish their full diagnostics on `/factr_diagnostics_<side>` (the
immutable startup-calibration snapshot plus the 25-tick post-enable capture) and the
live master gain on `/factr_gain_state_<side>`. The API relay streams both to Rerun
under the dedicated application id `factr-diagnostics` — a separate recording beside
DFC's `dual-flexiv-experiments` in the same embedded viewer (sink URL from
`FACTR_RERUN_URL`, defaulting to the DFC dashboard's gRPC proxy at
`rerun+http://127.0.0.1:9876/proxy`). DFC never polls diagnostics over HTTP; there is
no diagnostics endpoint. `rerun-sdk` is an optional dependency of the relay
(`/usr/bin/python3 -m pip install --user rerun-sdk==0.34.1`, matching the DFC
environment's version): without it the relay logs one warning and serves everything
else as usual.
