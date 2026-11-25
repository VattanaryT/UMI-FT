---
date: 2026-11-25
stream: frame
branch: surface_frame
commit: c5871f1
status: done
artifacts:
  - /mnt/data/umidata/ft/vtevy/rollouts/2026-11-24_liftos_hw_pe_guard
  - /mnt/data/umidata/ft/data/control_log/stringer_v4__hist__no_dagger__reviewed__epoch=0090-train_loss=0.017_2026-11-24_17-44-29
---
# Default-off hist stiffness rebuild

**What changed:** `UMIFT_HIST_STIFFNESS=1` or `scripts/run_rollout.sh r4vt --hist-stiffness`. Off by default. On, the runner ensembles the press scalar and rebuilds the 6x6 in the commanded tool frame: tangential 3000 N/m, rotation [100, 20, 20] Nm/rad. Pose-ensemble decay, the 25 N spring cap, and the 90 N wrench hold are unchanged.

**Why:** The stringer_v4 hist runner was averaging 6x6 stiffness matrices, which tilted the soft axis. The uncommitted standard_bc runner rebuilds that matrix after the blend and lowers the tangential gain from 5000 to 3000.

**Result / status:** pe_guard h0–h6, no robot. Logged matrices smear (max stiff-gap 891 N/m, h4 step 0 is 3467/4593/4940). The rebuild is isotropic 3000 because the logged press scalar is already 3000. `pytest PyriteEnvSuites/tests/test_hist_stiffness.py` passed (3). Not run on the robot.

**Repro:** `python offline_hw/check_hist_stiffness_pe_guard.py`
