---
date: 2026-11-19
stream: frame
branch: runner_causal_foot
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_causal_foot/acp_replay_buffer.zarr
---
# Replay the causal-foot labels

**What changed:** `--replay-episode` reads `r4vt_causal_foot` as 10-d actions and round-trips the causal-foot normalizer (`14308b72…`). A 27-d checkpoint normalizer is not applied on top. `UMIFT_RESET_TCP_TO_MEASURED=1` drops a carried TCP more than 15 mm from the measurement and logs the gap. Off by default. The quick plot draws the foot in the fixture frame.

**Why:** The hardware replay was playing track-clean δn and ΔR inside the causal-foot frame.

**Result / status:** `sim_replay_ep40_1795118916`, bench, contact, integrate-v-par, hardware 10-d loader. 54.4 s, 137 horizons, `episode_end`, guards []. Foot error median 0 mm, max 29.9 mm. Tilt crown/fillet/wall 6.75/25.18/35.51° vs causal label 6.92/24.59/35.89. Action δn vs the zarr median 0 mm. Robot not run. Camera margin, 8° slew, and the 5 mm floor are unchanged.

**Repro:** `UMIFT_DECODE_VPAR=1 python UMIFT_Data/surface_frames_scripts/sim_harness.py replay --episode 40 --bench --contact --integrate-v-par`
