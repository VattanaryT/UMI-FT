---
date: 2026-11-18
stream: frame
branch: smooth_frame
commit: uncommitted
status: done
artifacts:
  - /home/boeing/data/control_log/gt_replay_ep40_1795060735
  - /home/boeing/data/control_log/gt_replay_ep40_1795061663
---
# Continuous step_foot, press-axis recovery, table guard

**What changed:** The chunk-start snap and its fallback are gone. Every control step runs `step_foot` on the measured TCP, including across chunks. After more than 5 rejected steps the foot re-anchors to the press-axis ray hit. A commanded or measured TCP more than 5 mm past the mold's lowest vertex, or more than 100 mm from the mesh, holds the previous target and raises a hard stop.

**Why:** The gated snap left the foot on the crown, and the tool met the table in `gt_replay_ep40_1795061663`.

**Result / status:** Offline on those logs' measured TCPs. `1795060735`: foot crown 61/85 mm, fillet 12/62, wall 34/49; tilt command−label +39° / +13° / −6°; 22 rejects, 1 recovery; min height 44 mm above the mold bottom. `1795061663`: foot 94/166, 37/196, 38/184; tilt −2° / +4° / −2°; 75 rejects, 4 recoveries; commanded target 0.7 mm past the mold bottom, inside the 5 mm margin, so the guard did not fire. Not run on the robot. 24 tests passed.

**Repro:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest UMIFT_Data/tests/test_r4vt_tcp_reference.py UMIFT_Data/tests/test_smooth_frame.py`
