---
date: 2026-11-23
stream: frame
branch: lift_override
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_field_v2n/latest.ckpt
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_field_v3r/latest_final.ckpt
  - /home/boeing/umi_ft_ws/worktrees/UMI-FT-lift_override/closedloop_sim/out_lift_ep9/summary.json
---
# Deploy lift-off override, episode 9

**What changed:** New worktree `lift_override` off `7b4fd89`. `lift_retract.py` (flag off unless the runner enables it), `label_frame.py`, `run_lift_ep9.py`. No training files, no robot.

**Why:** Optional retract to +10 mm after contact is lost and the command is not pressing. Also check whether the executor composes ΔR with `mold_R_field_0`.

**Result / status:** Raw query vs `mold_R_field_0` is 174°. The executor's own R_d is 1.8° from that label composition on A, so it does not have the scorer bug. Forcing the label frame adds 9 tilt snaps (max 33.6°). v3r `R_tcp @ ΔR` is 175° off the executor and the jump guard aborts at 1.3 s. With the override on, v3r native (zero RGB) retracts once at 23.2 s to 10 mm and holds; mid-stroke contact losses 4 → 0; spurious retracts 0; tilt snaps 0. A never contacts, so the override does not arm. Sim RGB is zeros.

**Repro:** `python closedloop_sim/run_lift_ep9.py` in `UMI-FT-lift_override` (pyrite). Viewer `python closedloop_sim/serve_8812.py` (geodude venv) on port 8812.
