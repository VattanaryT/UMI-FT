---
date: 2026-11-11
stream: frame
branch: surface_frame
commit: uncommitted
status: superseded-by 2026-11-11-r4vt-r1rel-hyak-sbatch.md
artifacts:
  - /mnt/data/umidata/ft/training_outputs/oct_v35_r4vt_exbad
  - /mnt/data/umidata/ft/training_outputs/oct_v35_r4vt_exbad_stats.json
  - UMIFT_Data/surface_frames_scripts/assets/oct_v35_h4s_r4vt_action_normalizer_minmax_exbad.pkl
---
# Train r4vt without episodes 35/44, bad spans, or long waits

**What changed:** New task `umift_oct_fixb_r4vt_h4s` and `UmiFTDatasetR4VT`. Same 4.0 s r4 horizons and no depth. Normalizer is the min/max exbad pickle. Queries are dropped when an action target lies in episodes 35 or 44, in `bad_span_0`, or in `long_dt_gt_100ms.npz`. Checkpoints go to a new directory. The old pose-jump flag file is not used.

**Why:** r4vt should learn the virtual-target press, including presses below −0.25 m, and should not train on the two bad pose spans or on steps whose next RGB wait exceeds 100 ms.

**Result / status:** After the usual action-gap filter, train chunks are 16,875 (from 19,125) and val chunks are 1,817 (from 1,945). Train drops: 710 episode, 140 bad span, 1,336 long-dt, 64 both. Val drops: 128 long-dt. The 4090 was busy, so the job waits for free GPU memory and then trains 200 epochs, batch 16, checkpoint top-8. Not finished.

**Repro:** `task=umift_oct_fixb_r4vt_h4s` with `hydra.searchpath` pointed at `UMIFT_Data/surface_frames_scripts/config`. Output dir `/mnt/data/umidata/ft/training_outputs/oct_v35_r4vt_exbad`.
