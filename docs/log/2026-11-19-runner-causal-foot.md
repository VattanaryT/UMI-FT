---
date: 2026-11-19
stream: frame
branch: runner_causal_foot
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/data/control_log/sim_replay_ep40_1795109493.json
  - UMIFT_Data/surface_frames_scripts/assets/oct_v35_h4s_r4vt_action_normalizer_minmax_causal_foot.pkl
---
# Runner uses the causal foot

**What changed:** `runner_causal_foot` (off `smooth_frame` `0b31ea8`) decodes r4vt with `CausalFoot` from sim_harness `c4c13de`. The internal TCP advances by the full step and is not reset from the measurement. `x_d = foot + (v1·t1 + v2·t2)·dt + δn·n`, dt = 1/15, `R_d = R_field @ ΔR`. Steps longer than 25 mm are clipped. A 10-d action `(v1, v2, δn, stiffness, ΔR)` expands to 27 before decode. A 10-d checkpoint is compared to normalizer sha `14308b72d55600b4359dfa67d10aedab378e42ad26f2bbcac988eed4d97a1ac2`.

**Why:** The sim gate carried the demo TCP so the foot follows crown hops. The hardware runner had the old foot walk and rebuilt `x_d` on the foot normal.

**Result / status:** `sim_replay_ep40_1795109493`, bench, contact, integrate-v-par. 54.4 s, 137 horizons, `episode_end`, guards []. Foot error n=822 median 0 mm, max 29.9 mm. Tilt crown/fillet/wall 6.75/25.18/35.51° vs causal label 6.92/24.59/35.89. Commanded tangential max is 25.0 mm (the clip). The sim fed the demo TCP step. A robot replay advances by the command. 65 tests passed. Robot not run.

**Repro:** `UMIFT_DECODE_VPAR=1 python UMIFT_Data/surface_frames_scripts/sim_harness.py replay --episode 40 --bench --contact --integrate-v-par`
