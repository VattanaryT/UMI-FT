---
date: 2026-11-20
stream: frame
branch: runner_causal_foot
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_causal_foot/oct-v35-h4s-r4vt-causal-foot_e200_bs64_valtopk_43390127/checkpoints/epoch=0025-val_loss=0.058.ckpt
  - /mnt/data/umidata/ft/data/control_log/e25_flag_matrix.json
  - /mnt/data/umidata/ft/data/control_log/e25_flag_matrix_lead.json
---
# Lead-in, geodesic causal foot, image-aligned obs

**What changed:** `UMIFT_LEAD_IN_S`, `UMIFT_FOOT_WALK=geodesic` on the causal-foot decode, and `UMIFT_ALIGN_OBS_TO_IMAGE` are off unless set. The lead-in walks +t2 at 0.12 m/s and keeps the start clearance.

**Why:** The first closed-loop window was a static pose, and the causal foot ignored the geodesic walk.

**Result / status:** e25, episode 40, 8 s, seeds 0/1/2 (identical). Flags off and align: 0.063 m/s, ΔR about n leaves the plateau by 3.3°, foot 99 mm, `max_distance_from_mesh` at 0.4 s. Geodesic: 0.051 m/s, 4.9°, foot 43 mm, `tcp_penetration` at 0.8 s. Lead-in: 0.052 m/s, 7.1°, foot 36 mm, `tcp_penetration` at 0.8 s. All three: 0.052 m/s, 6.4°, foot 36 mm, `tcp_penetration` at 0.8 s.

**Repro:** `UMIFT_SEED=0 python UMIFT_Data/surface_frames_scripts/sim_harness.py rollout --episode 40 --seconds 8 --ckpt <e25>`. Set one of `UMIFT_LEAD_IN_S=3`, `UMIFT_FOOT_WALK=geodesic`, `UMIFT_ALIGN_OBS_TO_IMAGE=1`.
