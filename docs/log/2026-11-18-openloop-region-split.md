---
date: 2026-11-18
stream: frame
branch: smooth_frame
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/open_loop_exports/r4vt_smooth_e0030_43335222_release/region_split.json
  - /mnt/data/umidata/ft/open_loop_exports/r4vt_causal_e0030_43308311_release/region_split.json
---

# Region split of the released epoch-30 open loop

**What changed:** δn and ΔR on the released exports are split by the label foot's `mold_normal_0`. Crown is under 25° from up, fillet is 25–55°, wall is 55° and above. The whole chunk uses the query anchor. No inference was re-run. The open-loop viewer can filter and color those queries.

**Why:** The tilt summary asked whether the scraper follows the surface. The open-loop errors need the same bins.

**Result / status:** Train 6-step, smooth then causal. Crown δn 40.6 / 49.1 mm, ΔR 22.1 / 28.7°. Fillet 36.4 / 43.4 mm, 18.3 / 27.9°. Wall 34.8 / 45.2 mm, 20.8 / 31.0°. The two zarrs do not share a normal, so the query counts differ (smooth wall 2,260, causal wall 4,708). Not committed.

**Repro:** region comes from `openloop_lib.region_code` at `query_anchor_index`. Tables are `region_split.json` beside each released export.
