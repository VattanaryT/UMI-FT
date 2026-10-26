---
date: 2026-10-23
stream: bc
branch: logging (analysis ran on worktree standard_bc / simple_umi_test cd93324 code)
commit: uncommitted (analysis scripts in /tmp/obscmp, not in git)
status: done
artifacts: [/home/boeing/umi_ft_ws/data/data/control_log/episode_1792561421, /home/boeing/umi_ft_ws/data/data/control_log/episode_1792561791, /home/boeing/umi_ft_ws/data/training_outputs/stringer_v3.5/time/epoch=0150-train_loss=0.015.ckpt, /home/boeing/umi_ft_ws/data/datasets/stringer_v3.5/acp_replay_buffer.zarr]
---
# Policy inputs on real rollouts vs BC val data (`time` ckpt)

**What changed:** nothing in code. Compared `horizon_*/obs_policy` from two robot rollouts (2026-10-20 22:43, 22:49; `time/epoch=0150`)
with val-split samples (eps 9, 12, 28, 37, 41) built through the dataset sampler.

**Why:** policy does the right thing on val inputs but gets stuck on the robot; check whether the live inputs match training.

**Result / status:**
- RGB framing differs: training frames are letterboxed (black bars ~12% top and bottom, full 4:3 FoV; converter `get_image_transform_with_border`);
  live frames are resize+center-crop (`get_image_transform`, `manip_server_layup_env.py`), so they fill the frame, narrower FoV, tool ~1.3x larger. Also brighter (+0.04..0.07) and softer (Laplacian var 0.0034 vs 0.0078).
- Obs windows are not stretched by `slow_down_factor`: live pose displacement over the window ~10 mm vs ~31-35 mm in val; wrench window std (Fy/Fz) ~7-8 vs 13-14; frame-to-frame diff 0.08 vs 0.14.
- `elapsed_time_0` reaches 138/171 s live vs max 65 s in val.
- Depth is all zeros in both (no mismatch). `wrench_right` zeros in both.

**Repro:** scripts in `/tmp/obscmp/` (`dump_all.py`, `cmp.py`); needs `PYTHONPATH=` cleared and the standard_bc worktree on the path.
