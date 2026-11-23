---
date: 2026-11-23
stream: frame
branch: tilt1
commit: uncommitted
status: done
artifacts:
  - UMIFT_Data/surface_frames_scripts/eval/score_sign_flip.json
  - UMIFT_Data/tests/test_score_sign_flip.py
---
# Sign-flip score, label round-trip

**What changed:** Scorer position is the one-step tangent miss. Rotation uses the demonstrated frame. Tests in `test_score_sign_flip.py`. r1rel decode is unchanged.

**Why:** Stored δn is the virtual-target clearance, so a ground-truth r4vt action does not land on the TCP. `v * dt` does, tangentially, and ΔR / tilt a/b invert on the demonstrated frame.

**Result / status:** 4 tests passed. Steps 1/6/16/64. `*` = moved vs the previous table. r1rel did not. k3000 and tilt1 training step-1 rotation did not; later rotation did (frame is no longer walked). k3000 and tilt1 position moved at every step.

| run | ckpt | C pos mm | C rot deg | A pos mm | A rot deg | train pos | train rot |
|---|---|---|---|---|---|---|---|
| k3000-carry | e25 | *8.2/7.6/9.5/6.4 | 32.1/*47.9/*55.9/*86.5 | *7.0/*8.2/*8.2/*9.2 | 33.7/*42.7/*48.5/*75.1 | *6.9/*7.0/*7.4/*9.1 | 12.8/*19.5/*21.7/*30.4 |
| k3000-carry | e199 | *13.7/*4.9/*8.7/*6.5 | 37.4/*54.2/*62.0/*75.7 | *8.0/*5.3/*6.8/*7.5 | 37.9/*47.7/*53.0/*66.0 | *4.9/*5.0/*6.1/*7.6 | 6.6/*12.0/*14.6/*21.3 |
| tilt1 | e25 | *7.8/*8.1/*7.5/*5.8 | *87.7/*86.0/*91.9/*66.4 | *8.0/*7.6/*8.5/*9.2 | *94.2/*95.1/*91.4/*80.8 | *5.5/*6.2/*7.0/*8.4 | 12.0/*14.9/*16.6/*22.5 |
| tilt1 | e180 | *7.4/*5.2/*6.3/*4.9 | *87.6/*88.9/*90.7/*63.4 | *7.8/*6.3/*6.3/*6.5 | *92.3/*94.8/*91.0/*80.0 | *4.8/*5.3/*6.0/*8.4 | 9.5/*11.2/*12.8/*20.9 |
| r1rel | e25 | 13.0/38.7/72.8/184.9 | 5.0/12.9/24.7/59.4 | 11.4/33.5/69.6/146.8 | 3.6/6.9/17.1/55.2 | 7.9/30.5/45.3/76.8 | 2.3/6.6/11.8/22.0 |
| r1rel | e195 | 6.8/34.0/62.3/189.7 | 2.1/11.5/17.0/48.8 | 6.7/33.6/73.7/119.7 | 1.3/7.2/11.8/46.2 | 3.9/22.4/31.6/59.1 | 0.8/4.5/7.3/14.7 |

**Repro:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest UMIFT_Data/tests/test_score_sign_flip.py` then `python UMIFT_Data/surface_frames_scripts/eval/score_sign_flip.py`, pyrite env, from UMI-FT-tilt1.
