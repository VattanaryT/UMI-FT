---
date: 2026-10-29
stream: infra
branch: surface_frame
commit: f621a05
status: done
artifacts: []
---
# Cherry-pick letterbox live RGB + slow-down-scaled obs windows onto surface_frame

**What changed:** Cherry-picked `c3a88d1` from `simple_umi_test`. Live D405 uses `get_image_transform_with_border` (training letterbox). Runner `scale_obs_windows_by_slow_down` (default True) stretches obs history `down_sample_steps` by `slow_down_factor`. Restored `policy_clock.py` (was absent on this branch). Did not touch `main`.

**Why:** Train↔deploy mismatch diagnosed in `2026-10-23-rollout-vs-val-policy-obs.md`; same fix already on the BC front. Needed before trusting Fix B robot eval after October training.

**Result / status:** Cherry-pick conflict only on the runner import (surface_frame had no `policy_clock` import). Not yet re-measured on robot.

**Repro:** `source worktrees/activate.sh UMI-FT-surface_frames` then run the env runner; confirm `[slow_down] ... down_sample_steps` prints and live RGB has letterbox bars.
