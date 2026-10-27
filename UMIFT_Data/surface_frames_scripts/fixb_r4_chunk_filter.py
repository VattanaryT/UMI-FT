"""r4-only helpers: per-sample off-edge / field-jump masks + optional analysis filter.

UmiFTDatasetR4 keeps the same chunks as r2 and uses build_episode_mask_table only
for the ΔR loss mask (not filter_sampler_indices). Does not affect r1/r2/r3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from diffusion_policy.common.sampler_umift import sparse_action_indices_from_rgb_timestamps

OFF_EDGE_M = 0.005
FIELD_JUMP_DEG = 30.0


def rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    rel = np.einsum("...ji,...jk->...ik", Ra, Rb)
    rv = Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec()
    return np.degrees(np.linalg.norm(rv, axis=-1)).reshape(rel.shape[:-2])


def episode_masks_from_zarr(
    ep,
    off_edge_m: float = OFF_EDGE_M,
    jump_deg: float = FIELD_JUMP_DEG,
) -> Dict[str, np.ndarray]:
    """Per-robot-sample masks for one episode group."""
    R_f = np.asarray(ep["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
    p = np.asarray(ep["ts_pose_fb_0"], dtype=np.float64)[:, :3]
    x = np.asarray(ep["mold_x_s_0"], dtype=np.float64).reshape(-1, 3)
    n = R_f[:, :, 2]
    sd = np.sum((p - x) * n, axis=-1)
    ip = np.linalg.norm((p - x) - sd[:, None] * n, axis=-1)
    off_edge = ip > off_edge_m
    jump = np.zeros(len(R_f), dtype=bool)
    if len(R_f) > 1:
        jump[1:] = rot_angle_deg(R_f[1:], R_f[:-1]) > jump_deg
    return {"off_edge": off_edge, "field_jump": jump, "x_s": x, "R_f": R_f}


def build_episode_mask_table(
    zarr_path: str | Path,
    off_edge_m: float = OFF_EDGE_M,
    jump_deg: float = FIELD_JUMP_DEG,
) -> Dict[str, Dict[str, np.ndarray]]:
    root = zarr.open(str(zarr_path), "r")["data"]
    out = {}
    for key in sorted(root.keys(), key=lambda s: int(s.split("_")[-1])):
        if not key.startswith("episode_"):
            continue
        out[key] = episode_masks_from_zarr(root[key], off_edge_m=off_edge_m, jump_deg=jump_deg)
    return out


def filter_sampler_indices(
    indices: Sequence[Tuple[int, int]],
    *,
    replay_buffer,
    raw_masks: Dict[str, Dict[str, np.ndarray]],
    shape_meta: dict,
    action_padding: bool,
) -> Tuple[List[Tuple[int, int]], dict]:
    """Drop (epi_id, rgb_id) chunks that are off-edge at k0 or contain a field jump."""
    action_horizon = int(shape_meta["sample"]["action"]["sparse"]["horizon"])
    rgb_frame_stride = int(shape_meta["sample"]["action"]["sparse"]["down_sample_steps"])

    kept: List[Tuple[int, int]] = []
    n_off = 0
    n_jump = 0
    n_both = 0
    before = len(indices)

    for epi_id, rgb_id in indices:
        # SequenceSamplerUmiFT stores epi_id = int from "episode_N" (not ordinal).
        key = f"episode_{int(epi_id)}"
        if key not in raw_masks or key not in replay_buffer["data"]:
            continue
        masks = raw_masks[key]
        ep = replay_buffer["data"][key]
        action_times = np.squeeze(np.asarray(ep["action_time_stamps"], dtype=np.float64))
        rgb_times = np.squeeze(np.asarray(ep["obs"]["rgb_time_stamps_0"], dtype=np.float64))
        try:
            action_idx, _ = sparse_action_indices_from_rgb_timestamps(
                action_times=action_times,
                rgb_times=rgb_times,
                rgb_id=int(rgb_id),
                action_horizon=action_horizon,
                rgb_frame_stride=rgb_frame_stride,
            )
        except (ValueError, IndexError):
            continue
        if len(action_idx) == 0:
            continue
        if (not action_padding) and len(action_idx) < action_horizon:
            continue
        a0 = int(action_idx[0])
        a1 = int(action_idx[-1])
        off = bool(masks["off_edge"][a0])
        # any field jump on robot samples covered by the chunk window
        jmp = bool(np.any(masks["field_jump"][a0 : a1 + 1]))
        if off and jmp:
            n_both += 1
            continue
        if off:
            n_off += 1
            continue
        if jmp:
            n_jump += 1
            continue
        kept.append((int(epi_id), int(rgb_id)))

    after = len(kept)
    stats = {
        "chunks_before": before,
        "chunks_after": after,
        "chunks_removed": before - after,
        "frac_removed": float((before - after) / max(before, 1)),
        "removed_off_edge_only": n_off,
        "removed_field_jump_only": n_jump,
        "removed_both": n_both,
        "off_edge_m": OFF_EDGE_M,
        "field_jump_deg": FIELD_JUMP_DEG,
    }
    return kept, stats
