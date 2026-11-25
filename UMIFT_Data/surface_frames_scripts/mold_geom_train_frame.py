"""Pack robot0_mold_geom the way the liftos zarr does.

The runner turns this on. Set UMIFT_MOLD_GEOM_TRAIN_FRAME=0 or pass
--no-mold-geom-train-frame to keep the live smooth-field frame.
Training is CausalFoot on frames_pca_r020.npz, then reorient onto field_v2_axes
and carry_frames. The live query uses shared/fixb_smooth_frames.npz.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from causal_foot import CausalFoot
from field_v2 import carry_frames, reorient
from fixb_labels import so3_to_rot6d_rows
from heading_tilt import geodesic_deg

FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/"
    "stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
# Zarr attr smooth_frames. The runner loads shared/fixb_smooth_frames.npz instead.
TRAIN_FRAMES = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/"
    "stringer_v3.5_fixb_odom_corr/r4vt_field_v2_k3000/frames_pca_r020.npz"
)


def enabled() -> bool:
    """On unless the env is 0, false, no, or off. Unset means on."""
    raw = os.environ.get("UMIFT_MOLD_GEOM_TRAIN_FRAME", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def orientation_info() -> dict:
    """Path and sha256 of the training orientation file."""
    digest = hashlib.sha256(TRAIN_FRAMES.read_bytes()).hexdigest()
    return {"file": str(TRAIN_FRAMES), "sha": digest, "enabled": enabled()}


def pack_episode(
    position: np.ndarray,
    tool_R: np.ndarray,
    axes: np.ndarray,
    field_npz: Path = FIELD,
    frames_npz: Path | None = None,
) -> dict:
    """Same packing as field_v2.build. position (N,3), tool_R (N,3,3)."""
    frames_npz = frames_npz or TRAIN_FRAMES
    foot, causal = CausalFoot(field_npz, frames_npz).track(position)
    frame = carry_frames(reorient(causal, axes), tool_R)
    outward = -frame[:, :, 2]
    sd = np.sum((np.asarray(position) - foot) * outward, axis=1)
    rot6d = so3_to_rot6d_rows(frame).astype(np.float32)
    geom = np.concatenate(
        [sd.astype(np.float32).reshape(-1, 1), outward.astype(np.float32), rot6d],
        axis=1,
    )
    return {"geom": geom, "foot": foot, "frame": frame, "sd": sd, "normal": outward}


class TrainFrameGeom:
    """Incremental version of pack_episode. prev is the pre-reorient causal frame."""

    def __init__(self, axes: np.ndarray, field_npz: Path = FIELD, frames_npz: Path | None = None):
        frames_npz = frames_npz or TRAIN_FRAMES
        self.axes = np.asarray(axes, dtype=np.float64).reshape(3, 3)
        self.tracker = CausalFoot(field_npz, frames_npz)
        self.reset()

    def reset(self) -> None:
        self._n = 0
        self._prev = None
        self._pub = None
        self._tool = None
        self.frames = np.zeros((0, 3, 3), dtype=np.float64)
        self.geom = np.zeros((0, 10), dtype=np.float32)

    def extend(self, position: np.ndarray, tool_R: np.ndarray) -> np.ndarray:
        position = np.asarray(position, dtype=np.float64).reshape(-1, 3)
        tool_R = np.asarray(tool_R, dtype=np.float64).reshape(-1, 3, 3)
        if len(position) != len(tool_R):
            raise RuntimeError("train-frame geom position and tool length differ")
        if self._n > len(position):
            self.reset()
        if len(position) > len(self.frames):
            grown = np.zeros((len(position), 3, 3), dtype=np.float64)
            grown[: len(self.frames)] = self.frames
            self.frames = grown
            grown_g = np.zeros((len(position), 10), dtype=np.float32)
            grown_g[: len(self.geom)] = self.geom
            self.geom = grown_g
        for i in range(self._n, len(position)):
            foot, causal = self.tracker.step(position[i], self._prev)
            self._prev = (foot, causal)
            frame = reorient(causal.reshape(1, 3, 3), self.axes)[0]
            if self._pub is not None:
                ang = float(geodesic_deg(self._pub.reshape(1, 3, 3), frame.reshape(1, 3, 3))[0])
                moved = float(geodesic_deg(self._tool.reshape(1, 3, 3), tool_R[i].reshape(1, 3, 3))[0])
                if ang > 30.0 and moved < 30.0:
                    frame = self._pub
            self._pub = frame
            self._tool = tool_R[i]
            outward = -frame[:, 2]
            sd = float(np.dot(position[i] - foot, outward))
            rot6d = so3_to_rot6d_rows(frame.reshape(1, 3, 3))[0]
            self.frames[i] = frame
            self.geom[i, 0] = sd
            self.geom[i, 1:4] = outward
            self.geom[i, 4:10] = rot6d
        self._n = len(position)
        return self.geom[: self._n]


def apply_train_frame(runtime, position, tool_R, obs_raw: dict, robot_id: int = 0) -> bool:
    """Overwrite the published mold obs. No-op unless the flag is set."""
    if not enabled():
        return False
    paths = getattr(getattr(runtime, "field", None), "_causal_npz", None)
    if paths is None:
        return False
    packer = getattr(runtime, "_train_geom", None)
    if packer is None:
        import zarr

        path = os.environ.get("UMIFT_MOLD_LOCAL_ZARR", "").strip()
        if not path:
            path = (
                "/mnt/data/umidata/ft/mold_aware_datasets/"
                "stringer_v3.5_fixb_odom_corr/r4vt_field_v2n_v3r_liftos/"
                "acp_replay_buffer.zarr"
            )
        axes = np.asarray(zarr.open_group(path, mode="r").attrs["field_v2_axes"]["axes"])
        packer = TrainFrameGeom(axes, Path(paths[0]), TRAIN_FRAMES)
        runtime._train_geom = packer
    geom = packer.extend(position, tool_R)
    obs_raw[f"mold_signed_distance_{robot_id}"] = geom[:, :1]
    obs_raw[f"mold_normal_{robot_id}"] = geom[:, 1:4]
    obs_raw[f"mold_R_field_rot6d_{robot_id}"] = geom[:, 4:10]
    return True
