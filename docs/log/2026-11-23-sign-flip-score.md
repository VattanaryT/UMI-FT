---
date: 2026-11-23
stream: frame
branch: tilt1
commit: uncommitted
status: superseded-by 2026-11-23-sign-flip-score-fix.md
artifacts:
  - /mnt/data/umidata/ft/training_outputs/oct-v35-h4s-r4vt-causal-foot-carry-k3000_e200_bs64_valtopk_43472139/checkpoints/e199_final.ckpt
  - /mnt/data/umidata/ft/training_outputs/oct-v35-h4s-r4vt-tilt1_e200_bs64_valtopk_43467749/checkpoints/latest_pulled.ckpt
  - UMIFT_Data/surface_frames_scripts/eval/score_sign_flip.json
---
# Sign-flip score, e25 vs latest

**What changed:** Pulled job 43472139 `latest.ckpt` as `e199_final.ckpt` (epoch 199) and job 43467749 `latest.ckpt` as `latest_pulled.ckpt` (epoch 180). New scorer `eval/score_sign_flip.py`. Did not overwrite the older local `latest.ckpt` files.

**Why:** Test-time t1/t2 flip on the thin wall, contact_wall_v2 kept sets, against the training validation split with no flip.

**Result / status:** Thin C n=9, A n=94, sign flip on obs, demo frame, and the wall field. Training is 200 evenly spaced validation windows with no flip; 180 (r1rel 181) have a 64-step horizon. Mean position mm and rotation deg at steps 1/6/16/64:

| run | ckpt | C pos | C rot | A pos | A rot | train pos | train rot |
|---|---|---|---|---|---|---|---|
| k3000-carry | e25 | 19.1/52.6/67.7/240.1 | 32.1/39.3/47.6/85.2 | 11.6/41.4/90.7/212.3 | 33.7/46.8/63.2/89.5 | 14.4/33.9/50.1/83.4 | 12.8/32.5/41.9/49.3 |
| k3000-carry | e199 | 19.1/52.7/72.4/234.4 | 37.4/40.5/51.9/92.0 | 11.6/40.4/87.6/193.3 | 37.9/48.6/60.2/77.8 | 14.4/29.5/42.5/65.8 | 6.6/24.9/33.7/43.0 |
| tilt1 | e25 | 19.1/56.1/70.5/237.5 | 63.0/65.7/67.2/101.4 | 11.6/41.0/81.2/205.3 | 87.3/90.4/92.0/99.9 | 15.9/31.6/44.6/80.2 | 85.7/86.5/87.9/90.3 |
| tilt1 | e180 | 19.1/51.3/64.9/244.6 | 64.0/73.4/73.9/95.8 | 11.6/39.2/81.4/179.9 | 85.8/91.9/90.1/96.8 | 15.9/28.6/40.8/70.2 | 85.4/86.9/87.7/87.9 |
| r1rel | e25 | 13.0/38.7/72.8/184.9 | 5.0/12.9/24.7/59.4 | 11.4/33.5/69.6/146.8 | 3.6/6.9/17.1/55.2 | 7.9/30.5/45.3/76.8 | 2.3/6.6/11.8/22.0 |
| r1rel | e195 | 6.8/34.0/62.3/189.7 | 2.1/11.5/17.0/48.8 | 6.7/33.6/73.7/119.7 | 1.3/7.2/11.8/46.2 | 3.9/22.4/31.6/59.1 | 0.8/4.5/7.3/14.7 |

Surface-model step-1 position is the same on every checkpoint (C 19.1 mm, A 11.6 mm). Tilt1 rotation stays near 86° on the training windows at both epochs. r1rel e195 is the reference.

**Repro:** `python UMIFT_Data/surface_frames_scripts/eval/score_sign_flip.py` from UMI-FT-tilt1, pyrite env.
