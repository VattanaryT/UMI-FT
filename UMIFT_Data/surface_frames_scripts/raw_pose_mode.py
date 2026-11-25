"""Play the recorded TCP through the runner, without a policy or a mold foot.

``UMIFT_RAW_POSE_REPLAY`` is the episode id. ``UMIFT_RAW_POSE_START_INDEX``
is the first robot row (default 0). ``UMIFT_RAW_POSE_PRESS=1`` moves the
virtual target along the outward normal by ``-min(|F|, 25 N) / 3000`` when
the demo force is above 5 N. Off, the nominal pose and the virtual target
are the same point, so the spring force is zero.

The 25 mm step shorten and the 8° orientation slew are applied up front, the
same rules as a label replay. The runner then schedules the slices.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from fixb_labels import Rp_to_pose7, pose7_to_Rp
from fixb_smooth_field import SmoothFrameField
from mold_aware_runtime import RD_SLEW_AFTER_RAMP_DEG, VT_RAMP_S, _slew_orientations
from PyriteEnvSuites.env_runners.dataset_start_pose import pose7_change_frame
from PyriteUtility.spatial_math import spatial_utilities as su

ZARR = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_causal_foot/acp_replay_buffer.zarr"
)
FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
DT = 1.0 / 15.0
MAX_STEP_M = 0.025
FORCE_ON_N = 5.0
PRESS_CAP_N = 25.0
PRESS_K = 3000.0


def _limit(position: np.ndarray) -> np.ndarray:
    out = np.empty_like(position)
    prev = position[0].copy()
    out[0] = prev
    elapsed = 0.0
    for i in range(len(position)):
        elapsed += DT
        allow = MAX_STEP_M * (1.0 if i == 0 else min(1.0, elapsed / VT_RAMP_S))
        if i == 0:
            continue
        delta = position[i] - prev
        dist = float(np.linalg.norm(delta))
        if dist > allow and dist > 1e-12:
            prev = prev + delta * (allow / dist)
        else:
            prev = position[i].copy()
        out[i] = prev
    return out


class RawPoseReplay:
    def __init__(self, episode: int, start_index: int, exec_h: int, T_fixture_from_base: np.ndarray, press: bool):
        import zarr

        group = zarr.open_group(str(ZARR), mode="r")["data"][f"episode_{int(episode)}"]
        pose_fix = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
        pose_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        start = int(start_index)
        if start < 0 or start >= len(pose_t):
            raise ValueError(f"start index {start} is outside episode {episode} ({len(pose_t)} rows)")
        times = np.arange(pose_t[start], pose_t[-1] + 1e-9, DT)
        idx = np.clip(np.searchsorted(pose_t, times), 0, len(pose_t) - 1)
        T = np.asarray(T_fixture_from_base, dtype=np.float64).reshape(4, 4)
        base = pose7_change_frame(pose_fix[idx], np.linalg.inv(T))
        rotation, position = pose7_to_Rp(base)
        caps = np.empty(len(base))
        elapsed = 0.0
        for i in range(len(base)):
            caps[i] = 3.0 if elapsed < VT_RAMP_S else RD_SLEW_AFTER_RAMP_DEG
            elapsed += DT
        rotation, _hit = _slew_orientations(rotation, None, caps)
        position = _limit(position)
        self.nominal = su.pose7_to_SE3(Rp_to_pose7(rotation, position))
        self.virtual = self.nominal.copy()
        self.stiffness = None
        if press:
            wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
            wrench_t = np.asarray(group["wrench_time_stamps_0"], dtype=np.float64).reshape(-1)
            widx = np.clip(np.searchsorted(wrench_t, pose_t[idx]), 0, len(wrench_t) - 1)
            force = np.linalg.norm(wrench[widx, :3], axis=1)
            field = SmoothFrameField(FIELD, FIELD.with_name("fixb_smooth_frames.npz"))
            queried = field.query(pose_fix[idx, :3])
            outward_f = -np.asarray(queried["R"], dtype=np.float64)[:, :, 2]
            outward_b = outward_f @ T[:3, :3]
            depth = np.where(force > FORCE_ON_N, -np.minimum(force, PRESS_CAP_N) / PRESS_K, 0.0)
            vt = position + depth[:, None] * outward_b
            self.virtual = su.pose7_to_SE3(Rp_to_pose7(rotation, vt))
            self.stiffness = np.full(len(position), PRESS_K, dtype=np.float64)
        self.exec_h = int(exec_h)
        self.cursor = 0
        self.n = int(len(position))
        self.episode = int(episode)
        self.start_index = start

    def remaining(self) -> bool:
        return self.cursor < self.n

    def chunk(self):
        end = min(self.cursor + self.exec_h, self.n)
        nom = self.nominal[self.cursor:end]
        vt = self.virtual[self.cursor:end]
        if len(nom) < self.exec_h:
            extra = self.exec_h - len(nom)
            nom = np.concatenate([nom, np.repeat(nom[-1:], extra, axis=0)], axis=0)
            vt = np.concatenate([vt, np.repeat(vt[-1:], extra, axis=0)], axis=0)
        stiff = None if self.stiffness is None else np.full(self.exec_h, PRESS_K)
        return nom, vt, stiff

    def advance(self) -> None:
        self.cursor = min(self.cursor + self.exec_h, self.n)


def from_env(exec_h: int, T_fixture_from_base: np.ndarray) -> RawPoseReplay | None:
    raw = os.environ.get("UMIFT_RAW_POSE_REPLAY", "").strip()
    if raw == "":
        return None
    press = os.environ.get("UMIFT_RAW_POSE_PRESS", "").strip().lower() in ("1", "true", "yes")
    start = int(os.environ.get("UMIFT_RAW_POSE_START_INDEX", "0"))
    return RawPoseReplay(int(raw), start, exec_h, T_fixture_from_base, press)
