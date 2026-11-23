#!/usr/bin/env python3
"""Smooth-radius sweep. Writes new sidecars only. Does not touch the 40 mm field."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial import cKDTree

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS), str(ROOT)]

from causal_foot import CausalFoot
from fixb_field import FixBMoldField, orthonormal_frames
from fixb_labels import delta_R_from_field_tool, pose7_to_Rp
from fixb_smooth_field import (
    SmoothFrameField,
    _t1_align_angle,
    mesh_edges,
    project_tangent,
    smooth_normals,
    vertex_areas,
)
from heading_tilt import geodesic_deg
from PyriteUtility.data_pipeline.wrench_baseline_utils import time_based_moving_average

OUT = Path(__file__).resolve().parent
DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
FIELD = DATA / "shared/fixb_field.npz"
CURRENT = DATA / "shared/fixb_smooth_frames.npz"
ZARR = DATA / "r4vt_causal_foot/acp_replay_buffer.zarr"
SKIP = {35, 44}
RADII = (0.010, 0.020, 0.040)
UP = np.array([0.0, -1.0, 0.0])
K = 3000.0


def _check() -> None:
    a = np.eye(3)
    b = a @ np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    ang = float(geodesic_deg(a[None], b[None])[0])
    if abs(ang - 90.0) > 1e-6:
        raise RuntimeError(f"frame angle {ang}")


def write_frames(radius: float, path: Path, samples, tool_x) -> None:
    if path.exists():
        return
    field = FixBMoldField(FIELD)
    areas = vertex_areas(field.vertices, field.faces)
    outward = smooth_normals(field.vertices, field.normals, areas, radius)
    t1 = project_tangent(field.frames[:, :, 0], outward)
    missing = np.linalg.norm(t1, axis=1) < 1e-8
    if np.any(missing):
        t1[missing] = project_tangent(field.frames[missing, :, 1], outward[missing])
    theta, _strength = _t1_align_angle(field, t1, outward, samples, tool_x)
    from fixb_smooth_field import rotate_t1

    t1 = rotate_t1(t1, outward, theta)
    frames = orthonormal_frames(t1, -outward).astype(np.float32)
    np.savez_compressed(path, frames=frames, radius_m=np.float64(radius), t1_align_rad=np.float64(theta))
    print(f"[sweep] wrote {path.name} align {np.degrees(theta):.2f} deg", flush=True)


def region_of(normal, foot, apron) -> np.ndarray:
    n = normal / np.clip(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12, None)
    tilt = np.degrees(np.arccos(np.clip(n @ UP, -1.0, 1.0)))
    name = np.full(len(n), "wall", dtype=object)
    name[tilt < 25.0] = "crown"
    name[(tilt >= 25.0) & (tilt < 55.0)] = "fillet"
    name[apron.query(foot, workers=-1)[0] < 0.015] = "flange"
    return name


def percentiles(values):
    if len(values) == 0:
        return None
    x = np.concatenate(values)
    return {
        "n": int(len(x)),
        "median": round(float(np.median(x)), 2),
        "p90": round(float(np.percentile(x, 90)), 2),
    }


def consistency(dn_mm, dang, foot):
    key = np.floor(foot / 0.015).astype(np.int64)
    code = key[:, 0] + 1000003 * (key[:, 1] + 1000003 * key[:, 2])
    uniq, inv = np.unique(code, return_inverse=True)
    dn_std, ang_std = [], []
    for i in range(len(uniq)):
        sel = inv == i
        if int(sel.sum()) < 8:
            continue
        dn_std.append(float(np.std(dn_mm[sel])))
        ang_std.append(float(np.std(dang[sel])))
    if not dn_std:
        return {"bins": 0}
    return {
        "bins": len(dn_std),
        "dn_std_mm_median": round(float(np.median(dn_std)), 2),
        "dR_std_deg_median": round(float(np.median(ang_std)), 2),
    }


def demo_axes(root):
    pos, axis = [], []
    for ep in range(50):
        if ep in SKIP:
            continue
        group = root[f"episode_{ep}"]
        rot, xyz = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
        pos.append(xyz[::20])
        axis.append(rot[::20, :, 0])
    return np.vstack(pos), np.vstack(axis)


def smooth_wrench(group, robot_t):
    wrench_t = np.asarray(group["wrench_time_stamps_left_0"]).reshape(-1)
    wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
    smooth = time_based_moving_average(wrench, wrench_t, 1.0)
    j = np.clip(np.searchsorted(wrench_t, robot_t), 1, len(wrench_t) - 1)
    left = np.abs(robot_t - wrench_t[j - 1]) <= np.abs(robot_t - wrench_t[j])
    return smooth[np.where(left, j - 1, j)]


def measure(radius: float, frames_path: Path, root, field, apron, tracker=None) -> dict:
    jumps20 = jumps30 = steps = 0
    gap = {name: [] for name in ("crown", "fillet", "wall", "flange")}
    dn_all, ang_all, foot_all = [], [], []
    for ep in range(50):
        if ep in SKIP:
            continue
        group = root[f"episode_{ep}"]
        robot_t = np.asarray(group["robot_time_stamps_0"]).reshape(-1)
        rot, tcp = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
        if tracker is None:
            foot = np.asarray(group["mold_x_s_0"], dtype=np.float64).reshape(-1, 3)
            frame = np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        else:
            foot, frame = tracker.track(tcp)
        if len(frame) > 1:
            ang = geodesic_deg(frame[:-1], frame[1:])
            dt = np.diff(robot_t)
            ok = dt < 0.05
            jumps20 += int(np.sum((ang > 20.0) & ok))
            jumps30 += int(np.sum((ang > 30.0) & ok))
            steps += int(ok.sum())
        wrench = smooth_wrench(group, robot_t)
        raw_f = np.linalg.norm(np.asarray(group["wrench_left_0"], dtype=np.float64)[:, :3], axis=1)
        wt = np.asarray(group["wrench_time_stamps_left_0"]).reshape(-1)
        j = np.clip(np.searchsorted(wt, robot_t), 1, len(wt) - 1)
        take = np.where(np.abs(robot_t - wt[j - 1]) <= np.abs(robot_t - wt[j]), j - 1, j)
        hit = raw_f[take] >= 8.0
        if not np.any(hit):
            continue
        outward = -frame[:, :, 2]
        _h, vid, w = field.closest(tcp[hit])
        raw = np.einsum("qk,qkd->qd", w, field.normals[vid])
        raw /= np.clip(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12, None)
        dot = np.clip(np.sum(outward[hit] * raw, axis=1), -1.0, 1.0)
        deg = np.degrees(np.arccos(dot))
        names = region_of(raw, foot[hit], apron)
        for name in gap:
            sel = names == name
            if np.any(sel):
                gap[name].append(deg[sel])
        fn = np.einsum("ni,ni->n", np.einsum("nij,nj->ni", rot[hit], wrench[hit, :3]), outward[hit])
        delta = delta_R_from_field_tool(frame[hit], rot[hit])
        dang = np.degrees(np.linalg.norm(delta.as_rotvec() if hasattr(delta, "as_rotvec") else _rotvec(delta), axis=1))
        dn_all.append((-fn / K) * 1e3)
        ang_all.append(dang)
        foot_all.append(foot[hit])
        print(f"[sweep] r{radius*1e3:.0f} ep {ep:02d} jumps30 {jumps30}", flush=True)
    return {
        "radius_mm": radius * 1e3,
        "frames": str(frames_path),
        "steps": steps,
        "jumps_gt20": jumps20,
        "jumps_gt30": jumps30,
        "gap_deg": {name: percentiles(vals) for name, vals in gap.items()},
        "consistency": consistency(np.concatenate(dn_all), np.concatenate(ang_all), np.vstack(foot_all)),
    }


def _rotvec(delta):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(delta, dtype=np.float64).reshape(-1, 3, 3)).as_rotvec()


def main() -> None:
    _check()
    root = zarr.open(ZARR, "r")["data"]
    field = FixBMoldField(FIELD)
    apron = cKDTree(field.vertices[np.asarray(np.load(FIELD)["is_apron"]).astype(bool)])
    samples, tool_x = demo_axes(root)
    paths = {}
    for radius in (0.010, 0.020):
        path = OUT / f"frames_r{int(radius*1e3):03d}.npz"
        write_frames(radius, path, samples, tool_x)
        paths[radius] = path
    paths[0.040] = CURRENT
    rows = []
    for radius in RADII:
        tracker = None if radius == 0.040 else CausalFoot(FIELD, paths[radius])
        rows.append(measure(radius, paths[radius], root, field, apron, tracker))
        (OUT / "sweep.json").write_text(json.dumps(rows, indent=2))
    (OUT / "sweep.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
