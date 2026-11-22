---
date: 2026-11-22
stream: frame
branch: surface_frame
commit: uncommitted
status: done
artifacts: []
---
# Label variance over a step and over a second

**What changed:** `eval/frame_variance.py` and `eval/frame_variance_timescale.py` measure axis shares of the training labels. Outputs stay gitignored.

**Why:** One-step Δδn carried 0.81 of the field-step variance, which looked like pressing.

**Result / status:** That share is the fast part. Δδn std is 5.6 mm per 20 ms step and 2.2 mm after a 0.2 s average. 73% of its power is above 5 Hz, peak 1.5 Hz. Episodes 0, 9, and 28 show the slow δn swing moving with F·n. Over 0.4 s and 1 s, v2 is the largest position axis (share about 0.45 and 0.49). `robot0_mold_geom` has no (u, v). The wrench is TCP.

**Repro:** `python UMIFT_Data/surface_frames_scripts/eval/frame_variance_timescale.py`
