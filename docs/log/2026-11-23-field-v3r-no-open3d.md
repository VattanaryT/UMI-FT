---
date: 2026-11-23
stream: frame
branch: field_v3r
commit: uncommitted
status: done
artifacts: []
---
# field_v3r training import does not load open3d

**What changed:** `field_v3r_dataset` imports rot6d pack/unpack from `rot6d.py`. Same math as `fixb_labels.so3_to_rot6d_rows` and `mold_aware_runtime.rot6d_rows_to_R`.

**Why:** Hyak job 43485111 died because `fixb_labels` imports `fixb_field`, which imports open3d. The train image has no open3d.

**Result / status:** Local import-and-batch test. Not resubmitted.

**Repro:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest UMIFT_Data/tests/test_field_v3r_no_open3d.py`
