---
date: 2026-11-19
stream: frame
branch: runner_causal_foot
commit: uncommitted
status: done
artifacts: []
---
# Replay normalizer on CPU

**What changed:** GT replay copies checkpoint normalizer tensors with `.detach().cpu().numpy()` before numpy sees them. Option `d` during an r4vt replay uses `UMIFT_R4VT_ZARR` (`r4vt_causal_foot` from `run_rollout.sh`).

**Why:** Startup crashed: `_normalizer_width` called `np.asarray` on a CUDA scale. Option `d` was still the track_clean zarr stored in the checkpoint.

**Result / status:** Unit tests cover a CUDA normalizer, or a mock that raises the same TypeError. Robot not run.

**Repro:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest UMIFT_Data/tests/test_gt_replay_chunks.py PyriteEnvSuites/tests/test_dataset_start_pose.py`
