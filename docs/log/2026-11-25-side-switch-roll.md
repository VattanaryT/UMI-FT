---
date: 2026-11-25
stream: frame
branch: surface_frame
commit: uncommitted
status: done
artifacts:
  - UMIFT_Data/surface_frames_scripts/eval/climb_labels/ep9_side_switch.png
---
# Side-switch roll is not a masked v2

**What changed:** Open-loop check only. No new training config. The ΔR mask stays 1 through idx 5400–6800 (field-vs-tool step max 29.9°).

**Why:** Before submitting the horizon-24 job, check whether predicted v2 dies after the roll.

**Result / status:** Chained 6-step open loop on demo obs matches the climb: v2 median −0.106 vs label −0.132 on idx 6008–6316. Sustained roll lead is 2 steps (idx 6049 vs 6076). Hardware v2 after the twist is about +0.01 m/s on pe_guard and i5600. h24 and the 4× weight are not the fix for that.

**Repro:** `/tmp/side_switch_roll.py` in the pyrite env. Plot `eval/climb_labels/ep9_side_switch.png`.
