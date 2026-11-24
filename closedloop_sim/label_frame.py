"""Live mold frame matching the field_v2 label stored as mold_R_field_0.

CausalFoot on frames_pca_r020, then reorient onto the mold PCA axes, then the
30° carry. FixBMoldField.query is a different frame (the scorer's 170° bug).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "UMIFT_Data" / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))

from causal_foot import CausalFoot  # noqa: E402
from field_v2 import carry_frames, reorient  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import pose7_to_Rp  # noqa: E402
from heading_tilt import geodesic_deg  # noqa: E402

FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
FRAMES = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_field_v2_k3000/frames_pca_r020.npz"
)
ZARR = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_field_v2_k3000/acp_replay_buffer.zarr"
)
CARRY_DEG = 30.0


def _axes() -> np.ndarray:
    import zarr

    raw = zarr.open_group(str(ZARR), mode="r").attrs["field_v2_axes"]["axes"]
    return np.asarray(raw, dtype=np.float64).reshape(3, 3)


class LabelFrame:
    """Online twin of field_v2.build's frame. prev is the raw foot frame."""

    def __init__(self):
        self.foot = CausalFoot(FIELD, FRAMES)
        self.axes = _axes()
        self.prev = None
        self.prev_out = None
        self.prev_tool = None

    def reset(self) -> None:
        self.prev = None
        self.prev_out = None
        self.prev_tool = None

    def step(self, tcp: np.ndarray, tool_R: np.ndarray) -> np.ndarray:
        tcp = np.asarray(tcp, dtype=np.float64).reshape(3)
        tool_R = np.asarray(tool_R, dtype=np.float64).reshape(3, 3)
        x, raw = self.foot.step(tcp, self.prev)
        self.prev = (x, raw)
        oriented = reorient(raw.reshape(1, 3, 3), self.axes)[0]
        if self.prev_out is not None:
            ang = float(geodesic_deg(self.prev_out.reshape(1, 3, 3), oriented.reshape(1, 3, 3))[0])
            moved = float(geodesic_deg(self.prev_tool.reshape(1, 3, 3), tool_R.reshape(1, 3, 3))[0])
            if ang > CARRY_DEG and moved < CARRY_DEG:
                oriented = self.prev_out
        self.prev_out = oriented
        self.prev_tool = tool_R
        return oriented


def batch_frames(position: np.ndarray, tool_R: np.ndarray) -> np.ndarray:
    """Same frames field_v2.build writes into mold_R_field_0."""
    foot, frame = CausalFoot(FIELD, FRAMES).track(position)
    del foot
    return carry_frames(reorient(frame, _axes()), tool_R)


def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return geodesic_deg(a, b)


def match_episode(episode: int = 9, stride: int = 20) -> dict:
    import zarr

    group = zarr.open_group(str(ZARR), mode="r")["data"][f"episode_{episode}"]
    tool_R, position = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
    stored = np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
    n = min(len(position), len(stored))
    position, tool_R, stored = position[:n], tool_R[:n], stored[:n]
    label = batch_frames(position, tool_R)
    query = np.asarray(FixBMoldField(FIELD).query(position)["R"], dtype=np.float64)
    idx = np.arange(0, n, stride)
    label_err = _angle(label[idx], stored[idx])
    query_err = _angle(query[idx], stored[idx])
    online = LabelFrame()
    online_R = np.stack([online.step(position[i], tool_R[i]) for i in idx])
    online_err = _angle(online_R, stored[idx])
    return {
        "n": int(idx.size),
        "label_vs_mold_R_field_0_median_deg": round(float(np.median(label_err)), 3),
        "label_vs_mold_R_field_0_p90_deg": round(float(np.percentile(label_err, 90)), 3),
        "online_vs_mold_R_field_0_median_deg": round(float(np.median(online_err)), 3),
        "query_vs_mold_R_field_0_median_deg": round(float(np.median(query_err)), 3),
        "query_vs_mold_R_field_0_p90_deg": round(float(np.percentile(query_err, 90)), 3),
    }


if __name__ == "__main__":
    print(match_episode(9))
