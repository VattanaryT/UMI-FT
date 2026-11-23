#!/usr/bin/env python3
"""One rigid fit of thin-stringer contact onto the mesh. Does not rescore."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS), str(ROOT)]

from diffusion_policy.common.sampler_umift import sparse_action_indices_from_rgb_timestamps
from fixb_field import FixBMoldField

SRC = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/"
    "remap_to_training/acp_replay_buffer.zarr"
)
FIELD = Path("/mnt/data/umidata/ft/held_out_mold/thin_stringer/field/fixb_field.npz")
OUT = Path(__file__).resolve().parent / "registration.json"
FORCE_N = 8.0
LOOKBACK_S = 4.0
HORIZON = 64


def kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t with dst ~= R @ src + t. Column vectors."""
    a = np.asarray(src, dtype=np.float64)
    b = np.asarray(dst, dtype=np.float64)
    ac, bc = a.mean(0), b.mean(0)
    H = (a - ac).T @ (b - bc)
    _u, _s, vt = np.linalg.svd(H)
    R = vt.T @ _u.T
    if np.linalg.det(R) < 0:
        vt = vt.copy()
        vt[-1] *= -1
        R = vt.T @ _u.T
    return R, bc - R @ ac


def _check() -> None:
    rng = np.random.default_rng(0)
    src = rng.normal(size=(30, 3))
    R = Rotation.from_euler("xyz", [8.0, -3.0, 2.0], degrees=True).as_matrix()
    t = np.array([0.04, -0.02, 0.01])
    got_R, got_t = kabsch(src, (R @ src.T).T + t)
    assert np.allclose(got_R, R, atol=1e-8)
    assert np.allclose(got_t, t, atol=1e-8)


def nearest(times: np.ndarray, query: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(times, query)
    idx = np.clip(idx, 1, len(times) - 1)
    left = times[idx - 1]
    right = times[idx]
    take_left = np.abs(query - left) <= np.abs(query - right)
    return np.where(take_left, idx - 1, idx)


def dist_mm(field, xyz: np.ndarray) -> np.ndarray:
    geom = field.query(np.asarray(xyz, dtype=np.float64))
    foot = np.asarray(geom["x_s"], dtype=np.float64).reshape(-1, 3)
    return np.linalg.norm(np.asarray(xyz, dtype=np.float64).reshape(-1, 3) - foot, axis=1) * 1e3


def stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "median_mm": float(np.median(x)),
        "p90_mm": float(np.percentile(x, 90)),
    }


def windows(robot_t, rgb_t, t0, cutoff):
    chosen = []
    last = -10**9
    for rgb in range(len(rgb_t)):
        if float(rgb_t[rgb] - t0) < LOOKBACK_S or float(rgb_t[rgb]) >= cutoff:
            continue
        if rgb - last < 16:
            continue
        index, _times = sparse_action_indices_from_rgb_timestamps(
            robot_t, rgb_t, rgb, HORIZON, 1
        )
        index = np.asarray(index, dtype=np.int64)
        if len(index) < HORIZON or float(robot_t[index[-1]]) >= cutoff:
            continue
        chosen.append((float(rgb_t[rgb] - t0), float(robot_t[index[0]]), float(robot_t[index[-1]])))
        last = rgb
    return chosen


def main() -> None:
    import zarr

    _check()
    field = FixBMoldField(FIELD)
    ep = zarr.open(SRC, "r")["data"]["episode_0"]
    robot_t = np.asarray(ep["robot_time_stamps_0"]).reshape(-1)
    rgb_t = np.asarray(ep["rgb_time_stamps_0"]).reshape(-1)
    tcp = np.asarray(ep["ts_pose_fb_0"], dtype=np.float64)[:, :3]
    wrench_t = np.asarray(ep["wrench_time_stamps_left_0"]).reshape(-1)
    force = np.linalg.norm(np.asarray(ep["wrench_left_0"], dtype=np.float64)[:, :3], axis=1)
    t0 = float(rgb_t[0])
    cutoff = t0 + 0.5 * float(rgb_t[-1] - t0)
    rel = robot_t - t0
    fi = nearest(wrench_t, robot_t)
    aligned = force[fi]
    contact = aligned >= FORCE_N
    gap = dist_mm(field, tcp)
    rows = []
    for i, (tq, t_a, t_b) in enumerate(windows(robot_t, rgb_t, t0, cutoff), start=1):
        sel = contact & (rel >= (t_a - t0)) & (rel <= (t_b - t0))
        rows.append({"window": i, "t_query_s": tq, "t_end_s": t_b - t0, **stats(gap[sel])})
    fit_on = contact.copy()
    P = tcp[fit_on]
    Q = np.asarray(field.query(P)["x_s"], dtype=np.float64).reshape(-1, 3)
    R, t = kabsch(P, Q)
    moved = (R @ P.T).T + t
    after = dist_mm(field, moved)
    before = gap[fit_on]
    report = {
        "force_n": FORCE_N,
        "n_contact": int(fit_on.sum()),
        "n_robot": int(len(tcp)),
        "episode_median_mm": float(np.median(gap)),
        "contact_before": stats(before),
        "windows": rows,
        "translation_mm": [float(v * 1e3) for v in t],
        "translation_norm_mm": float(np.linalg.norm(t) * 1e3),
        "rotation_deg": float(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec()))),
        "contact_after": stats(after),
        "fit": "one SE(3), contact frames only, TCP -> closest mesh point, then re-queried",
    }
    # A fit that only sees the scored half. Same contact rule.
    half = contact & (rel < (cutoff - t0))
    if half.sum() >= 3:
        Ph, Qh = tcp[half], np.asarray(field.query(tcp[half])["x_s"], dtype=np.float64).reshape(-1, 3)
        Rh, th = kabsch(Ph, Qh)
        moved_all = (Rh @ tcp.T).T + th
        after_h = dist_mm(field, moved_all)
        report["first_half_contact_before"] = stats(gap[half])
        report["first_half_translation_mm"] = [float(v * 1e3) for v in th]
        report["first_half_translation_norm_mm"] = float(np.linalg.norm(th) * 1e3)
        report["first_half_rotation_deg"] = float(np.degrees(np.linalg.norm(Rotation.from_matrix(Rh).as_rotvec())))
        report["first_half_contact_after"] = stats(after_h[half])
        for row, (tq, t_a, t_b) in zip(rows, windows(robot_t, rgb_t, t0, cutoff)):
            sel = contact & (rel >= (t_a - t0)) & (rel <= (t_b - t0))
            row["after_first_half_fit"] = stats(after_h[sel])
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
