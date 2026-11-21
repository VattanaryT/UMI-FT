---
date: 2026-11-16
stream: frame
branch: frame_continuity
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_frame_continuity/acp_replay_buffer.zarr
  - UMIFT_Data/surface_frames_scripts/assets/oct_v35_h4s_r4vt_action_normalizer_minmax_exbad.pkl
---
# Pre-retrain checks for the frame-continuity labels

**What changed:** The open-loop viewer takes an optional `--frames_npz`. The live mold tracker keeps the decision face separate from the 25° slew and advances only new robot timestamps after `mold_runtime.reset()`. The October normalizer was recomputed on the new zarr and left in place.

**Why:** Confirm the 0% ΔR mask, the action scale, and that deployment frames use the same tracker as the labels before the retrain.

**Result / status:** The 1565 held-normal events match the tool on 1501/1565 (output step median 0.12°, unexplained median 0.24°). 9 of them sit on the 25° cap with the tool under 1°. All 22 accepted crown-to-wall switches are exactly 25° with the tool under 0.5°, and 3631/607706 steps are slew-saturated the same way. Kept-row normalizer (583035 rows, episodes 35 and 44 and bad spans dropped) is unchanged: delta_n −0.30144128..0.16870293 m, scale 4.254014. Sbatch still points at the old pickle. Not submitted.

**Repro:** Viewer on port 8766: `python UMIFT_Data/surface_frames_scripts/view_openloop.py --export /mnt/data/umidata/ft/open_loop_exports/r4vt_e66520fae9ae --mesh /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz --frames_npz /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_frame_continuity/ep0_ep9_frames.npz`
