---
date: 2026-11-23
stream: frame
branch: lift_deploy
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_field_v3r/latest_final.ckpt
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_field_v2n/latest.ckpt
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_field_v2n_norgb/latest.ckpt
  - UMIFT_Data/surface_frames_scripts/eval/lift_deploy/v3r_release_best.png
  - closedloop_sim/out_lift_deploy/release_summary.json
  - closedloop_sim/out_lift_deploy/dn_probe.json
---
# Retract releases on tangential travel or a 2 s timeout

**What changed:** RETRACT hands δn back after T_hold once the policy tangential command has moved `travel_m` or reversed, else at 2 s. Ramp in and out stays 0.2 s. Contact latch re-arms after the hand-back.

**Why:** The old exit waited for policy δn > +5 mm. v3r never commands that, so the +10 mm retract held for the rest of the episode.

**Result / status:** v3r latest_final, ep 9. All four (0.5 s / 1.0 s) × (2 cm / 4 cm) settings: 1 retract at 23.25 s, 0 spurious, peak 10.2 mm, then back in contact for ~33 s, tilt snaps 0, final δn −4.5 mm. Shortest air time is 0.5 s / 2 cm (owned 0.7 s, release reason travel). 1.0 s / 2 cm and 1.0 s / 4 cm are the same event: travel already exceeded 4 cm by 1.0 s. A’s +20–60 mm log matches tool clearance; the network’s own first-step δn on that hover is +10 to +22 mm. On demo contact poses with zero RGB, A’s first step is a press. B on this executor contacts (21.6 s) and ends pressing.

**Repro:** `python closedloop_sim/lift_strategies.py` then `python closedloop_sim/run_lift_release.py`. Viewer: `serve_8812.py` on port 8812.
