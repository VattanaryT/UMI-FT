---
date: YYYY-MM-DD
stream: bc            # bc | frame | infra | data | hw
branch: <branch>
commit: <sha or "uncommitted">
status: done          # done | in-progress | abandoned | superseded-by <file>
artifacts: []         # ckpt / dataset / scan paths created or used (outside git too)
---
# Short title

**What changed:** 1-3 lines (code, config, data, hardware, calibration).

**Why:** the reason / hypothesis.

**Result / status:** numbers or "not yet run"; next step if any.

**Repro:** exact command(s) or task yaml + overrides.
