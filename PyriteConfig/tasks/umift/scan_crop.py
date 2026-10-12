"""End-effector-centered ("egocentric") crop of the scan heatmap.

The scan heatmap is an 80x160 top-down raster of the part in the scanner-arm base
frame (row 0 = z_min, col 0 = x_min; vertical axis Y is dropped). Because the
policy's proprioception is relative, it cannot tell *where* on that raster it is.
This module cuts a fixed-metric window out of the heatmap that is

  * centered on the tool (the center of the patch is the tool's current x,z), and
  * rotated so the patch +col axis ("right") is the tool's heading axis
    (default: tool x axis, projected on the scan plane; tool z points down at
    the part for this task, so z is not a usable heading).

The same function is used for training (sampler, pose from the dataset) and for
the robot (runner, pose from RTDE mapped into the fixture frame), so the two
cannot drift apart.

Patch pixel (i, j) -> offset from the tool in the scan plane:
    u = ((j + 0.5) / P - 0.5) * size_m   along heading e_u = (cos t, sin t)  [x, z]
    v = ((i + 0.5) / P - 0.5) * size_m   along e_v = (-sin t, cos t)         [x, z]
with t = atan2(heading_z, heading_x). With ``flip_v`` the sign of v is negated, so the
patch is a proper (non-mirrored) top-down view (right = e_u, up = e_v). Because tool z
points down at the part, heading_axis "-x" + flip_v reproduces the wrist camera image
orientation (right = -tool x, down = -tool y); the legacy default (x, no flip) is a
mirrored view and is kept only so older checkpoints are unchanged. Pixels that fall outside the scanned area
(or on invalid/NaN pixels) are NaN, which obs_rgb_depth_preprocess later maps to 0
("no data"), identical to how the full heatmap treats invalid pixels.
"""

from typing import Dict, Optional

import cv2
import numpy as np

from PyriteConfig.tasks.umift.workcell_frames import (
    DEFAULT_EXTENT_KEY,
    WorkcellExtrinsics,
    load_workcell_extrinsics,
)

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def tool_pose_in_scan_plane(
    SE3_fixture_tool: np.ndarray,
    T_scanner_base_from_fixture: np.ndarray,
    heading_axis: str = "x",
):
    """Return (px, pz, theta): tool position (x,z) in scanner_base and heading angle.

    theta = atan2(heading_z, heading_x) of the chosen tool axis projected on X/Z.
    ``heading_axis`` may be negated with a leading "-" (e.g. "-x" = opposite of the tool
    x axis). If the axis is (almost) normal to the scan plane, theta falls back to 0.
    """
    T = T_scanner_base_from_fixture @ SE3_fixture_tool
    px, pz = float(T[0, 3]), float(T[2, 3])
    name = str(heading_axis).strip().lower()
    sign = -1.0 if name.startswith("-") else 1.0
    axis = sign * T[:3, _AXIS_INDEX[name.lstrip("+-")]]
    hx, hz = float(axis[0]), float(axis[2])
    if np.hypot(hx, hz) < 1e-3:
        return px, pz, 0.0
    return px, pz, float(np.arctan2(hz, hx))


def build_crop_maps(
    px: float,
    pz: float,
    theta: float,
    extent: Dict[str, float],
    src_hw,
    size_m: float,
    pixels: int,
    flip_v: bool = False,
):
    """cv2.remap sampling maps (map_x=col, map_y=row), float32 (P, P).

    flip_v=False (legacy): patch row i goes along +e_v (like the raw array rows go along
    +z), i.e. the patch is a mirrored view relative to a z-up top-down picture.
    flip_v=True: patch row i goes along -e_v, so patch up = +e_v and the patch is a
    proper (non-mirrored) top-down view with right = e_u.
    """
    H, W = src_hw
    P = int(pixels)
    c = (np.arange(P, dtype=np.float64) + 0.5) / P - 0.5
    u = c[None, :] * size_m  # (1, P) along heading
    v = (-c if flip_v else c)[:, None] * size_m  # (P, 1) across heading
    ct, st = np.cos(theta), np.sin(theta)
    x = px + u * ct - v * st
    z = pz + u * st + v * ct
    col = (x - extent["x_min"]) / (extent["x_max"] - extent["x_min"]) * W - 0.5
    row = (z - extent["z_min"]) / (extent["z_max"] - extent["z_min"]) * H - 0.5
    return col.astype(np.float32), row.astype(np.float32)


def crop_scan_heatmap_egocentric(
    heatmap: np.ndarray,
    SE3_fixture_tool: np.ndarray,
    crop_cfg,
    extrinsics: Optional[WorkcellExtrinsics] = None,
) -> np.ndarray:
    """Crop heatmaps around the tool.

    Args:
        heatmap: (T, H, W, C) float, mm, NaN = invalid. A (H, W, C) array is also accepted.
        SE3_fixture_tool: (4, 4) tool pose in the fixture frame at the query time.
            Used for all T entries (scan horizon is 1 by design).
        crop_cfg: mapping with size_m, pixels, heading_axis, [flip_v], [extent_key],
            [extrinsics_path]. Missing flip_v = False (legacy orientation, so
            checkpoints trained before the flag existed behave identically).
        extrinsics: optional pre-loaded WorkcellExtrinsics (defaults to
            load_workcell_extrinsics(crop_cfg.extrinsics_path)).

    Returns:
        (T, P, P, C) float32 (or (P, P, C) for 3-D input). NaN where outside/invalid.
    """
    squeeze = heatmap.ndim == 3
    if squeeze:
        heatmap = heatmap[None]
    T_, H, W, C = heatmap.shape

    if extrinsics is None:
        extrinsics = load_workcell_extrinsics(crop_cfg.get("extrinsics_path", None))
    extent = extrinsics.scan_extent(crop_cfg.get("extent_key", DEFAULT_EXTENT_KEY))
    size_m = float(crop_cfg.get("size_m", 0.16))
    pixels = int(crop_cfg.get("pixels", 64))
    heading_axis = str(crop_cfg.get("heading_axis", "x"))

    px, pz, theta = tool_pose_in_scan_plane(
        SE3_fixture_tool, extrinsics.T_scanner_base_from_fixture, heading_axis
    )
    map_x, map_y = build_crop_maps(
        px, pz, theta, extent, (H, W), size_m, pixels,
        flip_v=bool(crop_cfg.get("flip_v", False)),
    )

    out = np.full((T_, pixels, pixels, C), np.nan, dtype=np.float32)
    for t in range(T_):
        for c in range(C):
            src = heatmap[t, :, :, c].astype(np.float32)
            valid = np.isfinite(src)
            if not valid.any():
                continue
            filled = np.where(valid, src, 0.0).astype(np.float32)
            val_r = cv2.remap(
                filled, map_x, map_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
            )
            w_r = cv2.remap(
                valid.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
            )
            # normalized bilinear: average over valid neighbours only; a pixel is
            # valid only if at least half of its bilinear support was valid.
            ok = w_r >= 0.5
            plane = np.full((pixels, pixels), np.nan, dtype=np.float32)
            plane[ok] = val_r[ok] / w_r[ok]
            out[t, :, :, c] = plane
    return out[0] if squeeze else out
