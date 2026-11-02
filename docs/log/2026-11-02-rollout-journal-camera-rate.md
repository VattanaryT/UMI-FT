---
date: 2026-11-02
stream: frame
branch: surface_frame
commit: uncommitted
status: in-progress
artifacts:
  - /mnt/data/umidata/ft/training_outputs/oct_v35_fixb/r1_world_world/epoch=0195-train_loss=0.013.ckpt
  - /mnt/data/umidata/ft/training_outputs/oct_v35_fixb/r4_world_fixb_tilt/epoch=0195-train_loss=0.013.ckpt
---
# Rollout journal, camera-rate warnings, depth off for October v3.5

**What changed:** Depth is zeroed and the D405 depth stream is not opened. Wrist and center cameras warn live if a frame gap exceeds 1.5 periods at 15 Hz, and print the episode rate at the end. Each episode appends one JSON line to `$PYRITE_CONTROL_LOG_FOLDERS/rollout_log.jsonl` with the checkpoint, depth flag, clamps, and the horizon banner (frames and seconds). Startup prints that banner. The loaded checkpoint is recorded in `2026-11-02-r1-next-rollout.md`.

**Why:** October v3.5 r1/r4 are ready to compare against the September rollouts. The October 29 center videos were tagged 15 fps but only captured a few frames per second, and the runner config was not written down next to the episode order.

**Result / status:** The journal recorded the r4 protective-stop rollout and the r1 rollout. On disk, each October run has only `epoch=0195-train_loss=0.013.ckpt`. No separate best/final copy, and no `sparse_normalizer.pkl` or `logs.json.txt` beside them.

**Repro:** `source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-surface_frames` then `python PyriteEnvSuites/env_runners/umift_env_runner.py`.
