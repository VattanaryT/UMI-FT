"""Layup workcell frames: loader for layup_workcell_extrinsics.yaml.

All transforms are 4x4 SE3, column-vector convention, meters:  v_a = T_a_from_b @ v_b

Frames
------
fixture       umift_std_v2 fixture frame, origin at tag0. This is the frame of
              ts_pose_fb / ts_pose_command in tag-tracked datasets (e.g.
              stringer_scanner_v2). It moves with the part, so poses expressed in
              it do not change if the stringer + tags are relocated.
scanner_base  base of the Keyence scanner arm; frame of the scan heatmap X/Z axes.
right_base    UR controller base of the policy arm; frame of live ts_pose_fb on
              the robot.

Which yaml is used
------------------
``UMIFT_WORKCELL_EXTRINSICS`` (env var) overrides the default yaml next to this
file. Re-measure + point it at a new yaml after moving the part (the policy
itself does NOT need retraining).
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

ENV_VAR = "UMIFT_WORKCELL_EXTRINSICS"
DEFAULT_EXTRINSICS_PATH = Path(__file__).with_name("layup_workcell_extrinsics.yaml")
DEFAULT_EXTENT_KEY = "scan_heatmap_extent_stringer_scanner_v2"


class WorkcellExtrinsics:
    """Parsed layup_workcell_extrinsics.yaml."""

    def __init__(self, raw: dict, path: str):
        self.path = path
        self.raw = raw
        self.T_right_base_from_fixture = self._mat(raw, "T_right_base_from_fixture")
        self.T_scanner_base_from_fixture = self._mat(raw, "T_scanner_base_from_fixture")
        self.T_right_base_from_scanner_base = self._mat(
            raw, "T_right_base_from_scanner_base"
        )
        self.T_fixture_from_right_base = np.linalg.inv(self.T_right_base_from_fixture)

    @staticmethod
    def _mat(raw: dict, key: str) -> np.ndarray:
        m = np.asarray(raw[key], dtype=np.float64)
        assert m.shape == (4, 4), f"{key} must be 4x4, got {m.shape}"
        return m

    def scan_extent(self, key: str = DEFAULT_EXTENT_KEY) -> dict:
        """Metric extent of the scan heatmap in scanner_base X/Z (row0=z_min, col0=x_min)."""
        ext = self.raw[key]
        return {k: float(ext[k]) for k in ("x_min", "x_max", "z_min", "z_max")}


@lru_cache(maxsize=8)
def _load(path: str) -> WorkcellExtrinsics:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return WorkcellExtrinsics(raw, path)


def load_workcell_extrinsics(path: Optional[str] = None) -> WorkcellExtrinsics:
    """Load extrinsics from ``path`` > $UMIFT_WORKCELL_EXTRINSICS > packaged default."""
    if path is None:
        path = os.environ.get(ENV_VAR) or str(DEFAULT_EXTRINSICS_PATH)
    return _load(str(Path(path).expanduser().resolve()))


def pose_frames_cfg(shape_meta) -> dict:
    """Return {'obs': 'relative'|'absolute', 'action': 'relative'|'absolute'}."""
    cfg = shape_meta.get("pose_frames", None) if shape_meta is not None else None
    obs = "relative"
    action = "relative"
    if cfg is not None:
        obs = str(cfg.get("obs", obs))
        action = str(cfg.get("action", action))
    for name, v in (("obs", obs), ("action", action)):
        if v not in ("relative", "absolute"):
            raise ValueError(f"shape_meta.pose_frames.{name} must be relative|absolute, got {v}")
    return {"obs": obs, "action": action}


def scan_crop_cfg(shape_meta) -> Optional[dict]:
    """Return the egocentric-crop cfg of the scan heatmap key, or None if disabled."""
    if shape_meta is None:
        return None
    for key, attr in shape_meta["obs"].items():
        if "scan_heatmap" not in key:
            continue
        crop = attr.get("crop", None)
        if crop is not None and bool(crop.get("enabled", False)):
            return crop
    return None


def shape_meta_needs_fixture_frame(shape_meta) -> bool:
    """True if the policy consumes poses in the fixture frame (abs obs/action or crop).

    The robot runner uses this to decide whether to supply
    ``SE3_fixture_from_world`` (= T_fixture_from_right_base) to the obs pipeline.
    """
    frames = pose_frames_cfg(shape_meta)
    if frames["obs"] == "absolute" or frames["action"] == "absolute":
        return True
    if any("abs_eef" in k for k in shape_meta["obs"].keys()):
        return True
    return scan_crop_cfg(shape_meta) is not None
