---
date: 2026-11-19
stream: infra
branch: sim_harness
commit: b21dfa3
status: done
artifacts:
  - /home/boeing/data/control_log/sim_replay_ep40_1795089017
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_causal_foot/acp_replay_buffer.zarr
  - UMIFT_Data/surface_frames_scripts/assets/oct_v35_h4s_r4vt_action_normalizer_minmax_causal_foot.pkl
---
# Causal foot passes the contact replay gate

**What changed:** Replay decode can carry the demo TCP and command `foot + v·dt + δn·n` (`UMIFT_DECODE_VPAR` / `--integrate-v-par`). `--contact` projects the commanded TCP onto the mesh. `fixb_causal_foot_zarr.py` hardlinks track_clean into `r4vt_causal_foot` and writes a 10-d min-max normalizer. track_clean and its pickle are unchanged.

**Why:** Label `v` is the TCP tangent in the frame at the current sample. The label foot hops when the TCP moves along the normal, so integrating `v` alone leaves that foot. The carried TCP keeps the offset that makes the closest-point hop.

**Result / status:** `sim_replay_ep40_1795089017`, bench, contact, integrate-v-par. 54.4 s, 137 horizons, episode_end, guards []. Foot error n=822 median 0 mm, max 29.9 mm (2 steps > 10 mm). Tilt crown/fillet/wall 6.75/25.18/35.51° (n=245/410/167) vs causal label 6.92/24.59/35.89. Camera min 17.6 mm. Tangential step median commanded 4.64 mm vs demo 5.01 mm. Zarr kept_rows 34112. Normalizer sha256 `14308b72d55600b4359dfa67d10aedab378e42ad26f2bbcac988eed4d97a1ac2`. Not run on the robot.

**Repro:** `python UMIFT_Data/surface_frames_scripts/sim_harness.py replay --episode 40 --bench --contact --integrate-v-par`
