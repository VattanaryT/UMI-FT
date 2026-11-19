---
date: 2026-11-19
stream: frame
branch: sim_harness
commit: uncommitted
status: done
artifacts:
  - /home/boeing/data/control_log/sim_label_check_ep40_1795084761
  - /home/boeing/data/control_log/sim_replay_ep40_1795084764
  - /home/boeing/data/control_log/sim_rollout_oct-v35-h4s-r4vt-track-clean_e100_bs64_v_1795084849
  - /home/boeing/data/control_log/sim_rollout_oct-v35-h4s-r4vt-track-clean_e100_bs64_v_1795084859
  - /home/boeing/data/control_log/sim_rollout_oct-v35-h4s-r4vt-track-clean-step-mask_e_1795084869
---

# Kinematic sim harness, stage 1

**What changed:** `UMIFT_Data/surface_frames_scripts/sim_harness.py` with `label_check`, `replay`, and `rollout`. The fake arm's measured TCP is the last command (`--lag 1` optional). It imports the live smooth_frame runner (`decode_chunk`, foot walk, guards, D405 corners, flange plane). Runner sha `0b31ea8`. No robot.

**Why:** Check the runner and a checkpoint without moving the UR.

**Result / status:** Episode 40, `UMIFT_FOOT_CLIMB=1`. Builder vs stored foot is ~0 (median 5e-6 mm). Runner foot vs label, safety off, is not: median 17.8 mm, max 136 mm; frame median 10°. Demo tilt from normal (tool +z vs −n), crown/fillet/wall median 6.9/24.4/35.5°. Open-loop label replay (demo TCP as the measurement) stops at 12.8 s on `tcp_penetration`; min camera clearance 18.4 mm. GT replay through the fake arm, `--bench`, runs to 54.4 s with no guard; min camera clearance 6.9 mm; foot error vs label median 124 mm. Three 60 s rollouts, no bench floor, all stop at 2.4 s (horizon 6) on `tcp_penetration` and `normal_flip`. Epoch 55 commanded δn goes from +29 mm to −56 mm. Min camera clearance 33.6 / 24.0 / 27.0 mm (epoch 55, track_clean latest, step-mask latest). Stage 2 (IK arm, viewer) not built.

**Repro:** `source worktrees/activate.sh UMI-FT-sim_harness`, same env as `scripts/run_rollout.sh`, `UMIFT_FOOT_CLIMB=1`. `python UMIFT_Data/surface_frames_scripts/sim_harness.py label_check --episode 40 --bench` then `replay --episode 40 --bench`, then `rollout --episode 40 --seconds 60 --ckpt <path>` for epoch 55, track_clean latest (43351238), and step-mask latest (43363017).
