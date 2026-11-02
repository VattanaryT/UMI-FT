---
date: 2026-11-02
stream: frame
branch: surface_frame
commit: 685f10e
status: in-progress
artifacts:
  - UMIFT_Data/surface_frames_scripts/config/task/umift_oct_fixb_r4_h4s.yaml
  - UMIFT_Data/surface_frames_scripts/config/task/umift_oct_fixb_r1_h4s.yaml
  - UMIFT_Data/surface_frames_scripts/config/task/umift_oct_fixb_r2_h4s.yaml
  - UMIFT_Data/surface_frames_scripts/assets/oct_v35_h4s_r2r4_action_normalizer_3b0863c14453.pkl
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb/shared/oct_v35_h4s_chunk_normalizer.json
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb/shared/oct_v35_h4s_chunk_index.npz
---
# October v3.5 Fix B, 4.0 s lookback, action 64

**What changed:** New task yamls inherit `umift_oct_fixb_r4` / `r1` / `r2` and override horizons only. `umift_mold_aware.yaml` is unchanged. Short-horizon checkpoints cannot be reused. The runner raises `sparse_execution_horizon` from 16 to 32 when the loaded action horizon is at least 64. Horizon-24 checkpoints stay at 16.

**Why:** One shared 4.0 s demo lookback, and a 64-step action chunk. Training numbers are demo time. Inference `slow_down_factor` 2 and `scale_obs_windows_by_slow_down` stay on, so the same frame count is about 8 s of wall clock.

**Measured median dt** on `stringer_v3.5_fixb/r2_world_fixb` (all episodes): RGB 66.68 ms (15.00 Hz), wrench 34.12 ms (29.31 Hz), robot 5.00 ms (200.08 Hz). Wrench is not 28 Hz, so horizon 113 would be 3.82 s. Locked integers: RGB/depth 16×4, wrench 118×1, pose/mold 51×16, action 64×1. Gripper stays 1. Depth stays in shape_meta.

**Normalizer:** `assets/oct_v35_h4s_r2r4_action_normalizer_3b0863c14453.pkl`, sha256 `3b0863c144530d4b68f3ec1e3e933e291fcbf9601e273c921f00698d4c9dbd5b`. Fit on the new r2 train chunks (19125). Not the old `1ddfa2b4331f` file. r4 ΔR mask uses chunk length 64: 71.9% of train action steps masked (880377/1224000), 73.0% of val. Same 2409 field jumps >30°.

**Dataset:** `/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb/{r1_world_world,r2_world_fixb,r4_world_fixb_tilt}/acp_replay_buffer.zarr`. Hyak copy is the same tree under `/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v3.5_fixb/`.

**Result / status:** not yet run. Batch 32, one optimizer step, 200 epochs. Raise the batch after the first epoch fits. Checkpoints go to `oct_v35_h4s_fixb_${RUN}`, not the short-horizon dirs.

**Repro:**
```bash
cd UMIFT_Data/surface_frames_scripts
python fixb_oct_v35_chunks_and_normalizer.py --h4s
# on Hyak, after this commit is checked out at /gscratch/weirdlab/vtevy/UMI-FT-fixb
RUN=r4_world_fixb_tilt sbatch fixb_train_oct_v35_h4s_klone.sbatch
RUN=r1_world_world     sbatch fixb_train_oct_v35_h4s_klone.sbatch
RUN=r2_world_fixb      sbatch --account=cse --mem=256G fixb_train_oct_v35_h4s_klone.sbatch
```
