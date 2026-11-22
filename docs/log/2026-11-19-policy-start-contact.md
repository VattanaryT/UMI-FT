---
date: 2026-11-19
stream: infra
branch: runner_causal_foot
commit: uncommitted
status: in-progress
artifacts:
  - /home/boeing/umi_ft_ws/UMI-FT/runs/episode_1795126839
---
# Policy start: surface stop and descent wait

**What changed:** A policy start whose episode TCP is inside the mold stops the descent where signed distance crosses 0, and the arrival check then uses tangential position (2 mm) and orientation (2 deg). The outward gap is allowed. An episode TCP that is already outside is still the descent goal. After the timed move the runner waits up to 20 s, and re-commands once if the arm stalls short of an outside TCP. The 8 s lookback is unchanged. GT replay is unchanged.

**Why:** The episode-40 start check aborted 29 mm short (right-base y 0.758 vs 0.786) with 0 deg error and wrist 3 at −172 deg (`episode_1795126839`). That TCP is +31.0 mm outside the smooth field, not inside. At the abort the arm was still moving (~10 mm/s), force was 0.3 N, and reached sd was +54.7 mm. The timed descent returns before admittance arrives. The hover only looks finished because of the Enter wait.

**Result / status:** Not yet run on the robot. Unit tests cover the outside-vs-inside alpha and the normal-gap rule.

**Repro:** `UMIFT_POLICY_START_EPISODE=40` with `scripts/run_rollout.sh r4vt --bench` and the causal-foot epoch-40 checkpoint. Do not set `UMIFT_REPLAY_EPISODE`.
