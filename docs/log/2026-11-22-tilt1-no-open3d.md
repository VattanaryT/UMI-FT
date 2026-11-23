---
date: 2026-11-22
stream: frame
branch: tilt1
commit: uncommitted
status: done
---
# Tilt1 training import does not load open3d

**What changed:** `tilt1.py` converts pose7 with scipy. It no longer imports `fixb_labels`, which imports `fixb_field` and open3d. Field vertices still come from `np.load` of `fixb_field.npz`, same as the label step.

**Why:** Hyak job 43467490 died on `No module named 'open3d'` while importing `UmiFTDatasetTilt1`. The container pyrite env has no open3d. Causal-foot and heading-tilt training never import it.

**Result / status:** With `sys.modules['open3d']=None`, `import fixb_tilt1_dataset` succeeds and `ds[0]` action is `(64, 5)`. `fixb_field` is not imported. `test_tilt1.py` 6 passed. Not resubmitted.

**Repro:** from `UMIFT_Data/surface_frames_scripts`, `python -c "import sys; sys.modules['open3d']=None; import fixb_tilt1_dataset"`
