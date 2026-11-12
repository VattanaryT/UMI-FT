---
date: 2026-11-10
stream: frame
branch: surface_frame
commit: uncommitted
status: done
artifacts:
  - UMIFT_Data/surface_frames_scripts/assets/oct_v35_h4s_r4vt_action_normalizer_minmax.pkl
  - UMIFT_Data/surface_frames_scripts/assets/oct_v35_h4s_r4vt_action_normalizer_p01p99.pkl
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/r4vt_delta_n_outliers.json
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/long_dt_gt_100ms.npz
---
# r4vt normalizer candidates and flagged long waits

**What changed:** Two new action normalizers next to the untouched October r2/r4 pickle, fit on r4vt October rows excluding episodes 35 and 44. One uses the min/max of delta_n. The other uses the 1st and 99th percentile. A mask marks robot samples whose next-RGB wait exceeds 100 ms. The October viewer gained an r4vt tab, a visible exclusion flag for episodes 35 and 44, and shaded bad spans from `label_stats.json`. Labels were not edited.

**Why:** The min/max of r4vt delta_n is set by a few hard presses near −300 mm and by the episode 38 bad span near +334 mm. A percentile fit keeps those tails from setting the scale. Long waits and the bad spans are flagged for a later filter, not dropped.

**Result / status:** Fit on 586,808 rows. Min/max delta_n scale 3.1453, offset −0.05187. Percentile scale 5.7418, offset 0.3604. 10,839 samples fall outside ±1 after the percentile map. 1,020 samples have Δt > 100 ms (episodes 13, 25, 29, 42 hold most of them). Geometry tests: 5 passed. Not trained.

**Repro:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest UMIFT_Data/tests/test_fixb_label_geometry.py -q`
