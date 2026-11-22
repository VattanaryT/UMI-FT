---
date: 2026-11-18
stream: frame
branch: smooth_frame
commit: uncommitted
status: superseded-by 2026-11-18-camera-housing-guard.md
artifacts:
  - /home/boeing/data/control_log/gt_replay_ep40_1795061663
  - /home/boeing/data/control_log/gt_replay_ep40_1795060735
---
# Camera-table guard and replay orientation hold

**What changed:** A replay holds the previous tool orientation when the command is more than 25° from the label at that demo index. Commanded and measured D405 origins must stay 15 mm above the mold's lowest vertex; closer than that holds the pose and raises a hard stop.

**Why:** In `gt_replay_ep40_1795061663` the wrist camera met the table. At demo 7.00 s the command was 29° flatter than the label, entirely from the frame at a foot 38 mm off, and the measured TCP was 9 mm off the mold with no force.

**Result / status:** Max commanded-vs-label rotation is 46.5° in `1795061663` and 44.0° in `1795060735`. Not run on the robot. 26 tests passed.

**Repro:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest UMIFT_Data/tests/test_r4vt_tcp_reference.py UMIFT_Data/tests/test_smooth_frame.py`
