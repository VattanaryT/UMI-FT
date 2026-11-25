#!/usr/bin/env python3
"""Training mold_x_s versus the runtime foot on episode 9 wall frames.

Training obs for liftos and v3r is the stored mold_x_s (field_v2 CausalFoot.track,
copied through field_v3r and liftos). Runtime command feet are
UMIFT_FOOT_WALK=legacy and geodesic, driven by the stored v_par and the demo
TCP step. Does not write a dataset or a checkpoint.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "UMIFT_Data" / "surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS), str(ROOT)]

import zarr
from causal_foot import CausalFoot
from fixb_labels import pose7_to_Rp
from fixb_smooth_field import SmoothFrameField
from mold_aware_runtime import integrate_tcp_foot
from sliding_foot import NAMES, face_tags

DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
FIELD = DATA / "shared/fixb_field.npz"
FRAMES = DATA / "shared/fixb_smooth_frames.npz"
EP = 9


def _open(name: str):
    return zarr.open_group(str(DATA / name / "acp_replay_buffer.zarr"), mode="r")["data"][f"episode_{EP}"]


def _tagger(field: SmoothFrameField, tags: np.ndarray):
    def tag_at(points: np.ndarray) -> np.ndarray:
        q = field.scene.compute_closest_points(
            __import__("open3d").core.Tensor(np.asarray(points, dtype=np.float32).reshape(-1, 3))
        )
        tri = q["primitive_ids"].numpy().astype(np.int64)
        tri = np.clip(tri, 0, len(tags) - 1)
        return tags[tri]

    return tag_at


def _ang(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def _summary(name: str, foot: np.ndarray, frame: np.ndarray, stored: np.ndarray, stored_n: np.ndarray, wall: np.ndarray, tag_at) -> dict:
    got = tag_at(foot)
    demo = tag_at(stored)
    dist = np.linalg.norm(foot - stored, axis=1) * 1e3
    normal = _ang(-frame[:, :, 2], stored_n)
    w = wall
    same = got[w] == demo[w]
    return {
        "name": name,
        "n": int(len(foot)),
        "wall_n": int(w.sum()),
        "wall_same_face": round(float(same.mean()) if w.any() else 1.0, 4),
        "wall_on_wall": round(float(np.mean(got[w] == 0)) if w.any() else 1.0, 4),
        "wall_dist_mm_median": round(float(np.median(dist[w])) if w.any() else 0.0, 2),
        "wall_dist_mm_p95": round(float(np.percentile(dist[w], 95)) if w.any() else 0.0, 2),
        "wall_normal_deg_median": round(float(np.median(normal[w])) if w.any() else 0.0, 2),
        "wall_normal_deg_p95": round(float(np.percentile(normal[w], 95)) if w.any() else 0.0, 2),
        "wall_got_counts": {NAMES[i]: int(np.sum(got[w] == i)) for i in range(len(NAMES))},
    }


def _runtime(mode: str, field, p0, v, dn, step, dt) -> dict:
    os.environ["UMIFT_FOOT_WALK"] = mode
    os.environ.pop("UMIFT_FOOT_FACE_HYST", None)
    return integrate_tcp_foot(field, p0, v, dn, dt, tcp_step=step)


def main() -> None:
    liftos = _open("r4vt_field_v2n_v3r_liftos")
    v3r = _open("r4vt_field_v3r")
    causal = _open("r4vt_causal_foot")
    stored_l = np.asarray(liftos["mold_x_s_0"], dtype=np.float64)
    stored_v = np.asarray(v3r["mold_x_s_0"], dtype=np.float64)
    stored_c = np.asarray(causal["mold_x_s_0"], dtype=np.float64)
    frame_l = np.asarray(liftos["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
    pose = np.asarray(liftos["ts_pose_fb_0"], dtype=np.float64)
    _R, position = pose7_to_Rp(pose)
    robot_t = np.asarray(liftos["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rgb_t = np.asarray(liftos["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    n = min(len(stored_l), len(stored_v), len(stored_c), len(position), len(frame_l))
    stored_l, stored_v, stored_c = stored_l[:n], stored_v[:n], stored_c[:n]
    frame_l, position, robot_t = frame_l[:n], position[:n], robot_t[:n]
    field = SmoothFrameField(FIELD, FRAMES)
    tags = face_tags(FIELD)
    tag_at = _tagger(field, tags)
    demo_tag = tag_at(stored_l)
    wall = demo_tag == 0
    stored_n = -frame_l[:, :, 2]
    tracker = CausalFoot(FIELD, FRAMES)
    retrace, retrace_R = tracker.track(position)
    v_all = np.asarray(liftos["fixb_v_par_0"], dtype=np.float64)[:n]
    dn_all = np.asarray(liftos["fixb_delta_n_0"], dtype=np.float64).reshape(-1)[:n]
    # Runtime steps once per camera frame, the same rate as a replay chunk.
    rgb_ok = rgb_t[(rgb_t >= robot_t[0]) & (rgb_t <= robot_t[-1])]
    idx = np.unique(np.clip(np.searchsorted(robot_t, rgb_ok), 0, n - 1))
    v = v_all[idx]
    dn = dn_all[idx]
    pos_c = position[idx]
    step = np.zeros_like(pos_c)
    step[:-1] = np.diff(pos_c, axis=0)
    dt = np.full(len(idx), 1.0 / 15.0)
    stored_c_rate = stored_l[idx]
    frame_c = frame_l[idx]
    wall_c = tag_at(stored_c_rate) == 0
    out = {
        "liftos_equals_v3r_mold_x_s": bool(np.array_equal(stored_l, stored_v)),
        "liftos_vs_causal_mm_median": round(float(np.median(np.linalg.norm(stored_l - stored_c, axis=1) * 1e3)), 3),
        "retrace_vs_stored_mm_median": round(float(np.median(np.linalg.norm(retrace - stored_l, axis=1) * 1e3)), 3),
        "retrace_vs_stored_mm_p95": round(float(np.percentile(np.linalg.norm(retrace - stored_l, axis=1) * 1e3, 95)), 3),
        "wall_frames": int(wall.sum()),
    }
    out["retrace"] = _summary("causal_track", retrace, retrace_R, stored_l, stored_n, wall, tag_at)
    out["control_steps"] = int(len(idx))
    out["control_wall_frames"] = int(wall_c.sum())
    for mode in ("legacy", "geodesic"):
        got = _runtime(mode, field, pos_c[0], v, dn, step, dt)
        out[mode] = _summary(
            mode, got["x_s"], got["R"], stored_c_rate, -frame_c[:, :, 2], wall_c, tag_at
        )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
