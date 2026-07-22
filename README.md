# FACTR-Server

## Joint conventions

There are three deliberately separate representations of a leader pose:

- **Raw FACTR hardware** is the untouched Dynamixel reading. WebSocket diagnostics frames
  expose it, and DFC publishes it as `factr/raw/<side>` and stores it as
  `observation.factr_raw.<side>`.
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
Dynamixel→FACTR `joint_offsets`. These values are exposed by WebSocket diagnostics frames
and validated at startup. DFC reads this leader-owned contract and does not keep or push
another copy.
DFC straight-up home is `[0,0,0,0,0,0,0]`; the corresponding FACTR model reference is
`[0,0,0,pi/2,0,0,0]`. The explicit joint-4 offset is intentional.
The saved offsets are reused at every launch; the leader may start in any pose. Its first
reading is captured for diagnostics only and never recalibrates or gates gravity compensation.
