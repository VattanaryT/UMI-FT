---
date: 2026-11-22
stream: frame
branch: tilt1
commit: uncommitted
status: superseded-by 2026-11-22-tilt1-side-hold.md
artifacts:
  - /home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/UMIFT_Data/surface_frames_scripts/assets/oct_v35_h4s_r4vt_action_normalizer_minmax_tilt1.pkl
---
# Snap 1-tilt heading to the stringer side

**What changed:** Heading is 0° when n_out points toward table +Z and 180° when it points away. a is recomputed in that frame. Normalizer refit, sha256 `d0013b952ec4a3191a06f7cd59cc176adc618b55474040146984235527261853`. Not submitted.

**Why:** A fixed 0° heading missed the demo by about 80° on the return side. The demo's ψ hemisphere follows n_out · +Z (the walls), not table +X (along the stringer).

**Result / status:** Pose miss, all episodes except 35 and 44. Median / p90 degrees: wall_near 3.61 / 8.06, wall_far 3.35 / 7.67, radius 3.99 / 8.84, flange 4.81 / 10.15, crown 11.16 / 148.87. a degrees mean / std: wall_near 40.80 / 10.83, wall_far 34.98 / 25.37, radius 22.11 / 16.27, flange -0.93 / 8.14, crown 15.91 / 35.28. +a moves tool z downward on 98% of near-wall samples and 97% of far-wall samples. Round trip under 1e-6°.

**Repro:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest UMIFT_Data/tests/test_tilt1.py -q`
