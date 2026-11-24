---
date: 2026-11-23
stream: frame
branch: lift_deploy
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_field_v2n/latest.ckpt
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_field_v3r/latest_final.ckpt
  - UMIFT_Data/surface_frames_scripts/eval/lift_deploy/lift_fraction.png
  - UMIFT_Data/surface_frames_scripts/eval/lift_deploy/v3r_clearance.png
  - closedloop_sim/out_lift_deploy/summary.json
  - closedloop_sim/out_lift_deploy/open_loop.json
---
# Deploy-time lift strategies, closed-loop ep 9

**What changed:** New worktree `UMI-FT-lift_deploy` off `7b4fd89`. Flags default off. `closedloop_sim/lift_strategies.py`, `closedloop_sim/run_lift_deploy.py`. No executor change. Query vs `mold_R_field_0` is 174.21°. Committed decode already matches the label composition on A (~1.8°). Label-frame replace and v3r `R_tcp @ ΔR` were not applied (tilt snaps 0→9, and a jump abort at 1.3 s).

**Why:** Compare retract rules before a hardware run. Success bar was ≥90% of demo lift events at ≥10 mm within 0.5 s of contact ending, and zero spurious mid-stroke retracts on ep 9.

**Result / status:** No strategy meets the bar. v3r baseline stays in contact 55.5 s (4 mid-stroke losses, 0/26 at 10 mm, path 138 mm, 0 tilt snaps). Force override (5 N / 3 N / 50 ms / δn > −1 mm / +10 mm / 0.2 s) fires once at 23.2 s, aligned 1/26, spurious 0, mid-stroke losses 0, contact 22.1 s, path 105 mm. Holding that +10 mm is why force+air scores 17/26 "reached"; aligned stays 1. δn-only ensemble raises losses to 34. Stroke reversal retracts at 2.1 s and leaves contact. A looser gate (off 4.5 N, δn > −4 mm, 15 mm, release on press) gets aligned 2/26 and one spurious retract at 1.4 s. A with zero RGB never contacts. Open-loop on recorded force, episodes 9/12/28/37/41: force hits 19/168 (A) and 11/168 (v3r) inside 0.5 s, with more retracts than hits. Liftos checkpoint is not on disk.

**Repro:** `python closedloop_sim/lift_strategies.py` then `python closedloop_sim/run_lift_deploy.py` from this worktree, `pyrite` env, `activate.sh UMI-FT-lift_deploy`.
