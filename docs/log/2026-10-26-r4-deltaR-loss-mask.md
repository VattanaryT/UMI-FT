---
date: 2026-10-26
stream: frame
branch: surface_frame
commit: 54ecf31
status: done
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb/r4_world_fixb_tilt/acp_replay_buffer.zarr
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb/r4_world_fixb_tilt/checks/r4_delta_R_mask_stats.json
---
# r4: ΔR loss mask instead of jump chunk drop

**What changed:** Removed the field-jump / off-edge *chunk drop* for r4. `UmiFTDatasetR4` now samples exactly the same `(epi, rgb_id)` indices as r2. Per-step `action_loss_weight` zeros dims 21:27 after a >30° mesh `R_f` jump inside the chunk; other dims keep full loss. Policy MSE uses the weight when present. Decode holds previous `R_d` on mid-chunk mesh jumps (then rate clamp); runner logs `mesh_jump_hold`.

**Why:** Dropping any chunk with a jump in the full RGB-aligned action window removed ~36% of train chunks (long window + overlapping queries). Earlier “4.1% of kept chunks have a jump” used a short ~24-robot-step window, not the train horizon (~H=24 RGB → hundreds of robot samples). Masking ΔR keeps r2 contact/v∥ labels while skipping bad heading targets.

**Result / status:** r2/r4 train indices equal (63557); val equal (6875). ΔR steps masked: train 25.3%, val 24.9%. `sanity_mold_aware_ep5 --run r4_world_fixb_tilt` → SANITY PASS. Not submitted to Hyak.

**Repro:**
```bash
source worktrees/activate.sh UMI-FT-surface_frames
# train (Hyak later): RUN=r4_world_fixb_tilt sbatch .../fixb_train_scanner_klone.sbatch
python UMIFT_Data/surface_frames_scripts/sanity_mold_aware_ep5.py --run r4_world_fixb_tilt
```
