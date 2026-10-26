---
date: 2026-10-24
stream: bc
branch: simple_umi_test (worktree UMI-FT-standard_bc)
commit: e34b10e
status: in-progress
artifacts: [docker umift-train:local / umift-train:e34b10e, /home/boeing/umi_ft_ws/data/images/umift-train_v7.sif]
---
# Slow-down-scaled obs windows + letterboxed live rgb; images rebuilt

**What changed:** (1) runner `control_para["scale_obs_windows_by_slow_down"]` (default True) multiplies `down_sample_steps` of every
obs history window (rgb/depth/pose/wrench, horizon>1) by `slow_down_factor` (helper `scale_obs_windows_by_slow_down` in `policy_clock.py`);
runner raises if the stretched window exceeds `robot_buffer_size`/`wrench_buffer_size` in `ur5e_wrist_ft.yaml` (2400 samples = 4.8 s; the `hist` ckpt will not fit).
(2) `ManipServerLayupEnv` uses `get_image_transform_with_border` (letterbox, as training) instead of resize+center-crop.
Image rebuilt from the worktree (includes the elapsed-time code from cd93324). Training image only copies PyriteUtility/PyriteML/PyriteConfig; the runner fixes run on the host.

**Why:** rollout-vs-val obs comparison (2026-10-23-rollout-vs-val-policy-obs.md): live frames not letterboxed; windows spanned training wall-time while the robot runs 2x slower. Upstream never scaled windows either.

**Result / status:** unit tests pass (20). Not yet run on robot; re-measure rollout obs with `/tmp/obscmp` scripts. Brightness/sharpness not addressed.

**Repro:** `docker build -f docker/Dockerfile --build-arg GIT_SHA=e34b10e -t umift-train:local .` then `apptainer build umift-train_v7.sif docker-daemon://umift-train:local` (in `data/images`). Not copied to Hyak.
