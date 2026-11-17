---
date: 2026-11-17
stream: frame
branch: frame_continuity
commit: aa4f48a
status: done
artifacts:
  - /home/boeing/data/control_log/gt_replay_ep0_1794952662
---
# Bench foot cap and fixture-up lift

**What changed:** The runtime keeps the previous foot and frame. The first replan queries the measured TCP. Later replans call `track_step` with that TCP and the previous foot. A hop over 10 mm, per replan or per step, holds the previous foot. `--bench-lift 0.08` shifts the foot, the virtual target, and the replay start pose 80 mm along fixture −Y.

**Why:** `gt_replay_ep0_1794952662` tracked the demo until horizon 17 step 5, when the −5 mm floor put the virtual target through the flange. The TCP followed it down (+Y) and the stateless closest point walked onto a face 63° off the demo. Peak force 300.8 N. The apron was not involved. Fixture +Y is down, so the lift is −Y. The surface normal is −Y on the flange and −Z on the wall, and a normal lift would slide the wall section sideways.

**Result / status:** 17 tests passed, including the 10 mm ratchet test and the live-horizon foot median. Not run on the robot.

**Repro:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest UMIFT_Data/tests/test_r4vt_tcp_reference.py UMIFT_Data/tests/test_r4_runtime_fix.py`. Bench, mold removed, no contact: `scripts/run_rollout.sh r4vt --replay-episode 0 --bench --bench-lift 0.08`.
