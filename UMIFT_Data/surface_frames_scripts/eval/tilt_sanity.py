#!/usr/bin/env python3
"""Zero-tilt baseline and label round trip for heading-tilt e100.

Also measures why the causal pack_tilt split is much larger than the
geodesic. Does not start the robot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / "eval"), str(ROOT / "PyriteEnvSuites/scripts")]

from fixb_field import FixBMoldField
from heading_tilt import (
    e_long_sequence,
    geodesic_deg,
    long_axis,
    pack_tilt,
    rotation_from_tilt,
)
from heading_tilt_43448962_open_loop import FIELD, decode_pose, query_indices, wrap_deg
from heading_tilt_horizons import STEPS, _at, mean_rows
from heading_tilt_vs_causal_foot import (
    CAUSAL_DIR,
    CAUSAL_NORM,
    CAUSAL_ZARR,
    HEAD_LATEST,
    HEAD_NORM,
    HEAD_ZARR,
    build_dataset,
    rotation_of_rot6d,
)
from mold_aware_runtime import integrate_tcp_foot, rot6d_rows_to_R
from run_policy_open_loop_val_export import load_policy_and_cfg, predict_sparse_action

OUT = SCRIPTS / "eval/tilt_sanity.json"
EPISODES = (9, 12, 28, 37, 41)


def _pct(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _pose_error(field, axis, action, gt, p0, r0):
    xp, rp = decode_pose(field, axis, action, p0, r0)
    xg, rg = decode_pose(field, axis, gt, p0, r0)
    pos = np.linalg.norm(xp - xg, axis=1) * 1e3
    rot = geodesic_deg(rp, rg)
    return _at(pos, rot, np.zeros_like(pos), np.zeros_like(pos), np.zeros_like(pos), np.zeros_like(pos))


def round_trip(field, axis, action, p0, r0):
    dt = np.full(action.shape[0], 1.0 / 15.0)
    foot = integrate_tcp_foot(field, p0, action[:, 0:2], action[:, 2], dt, R_tool=r0)
    e_long, _angle = e_long_sequence(foot["n_out"], axis)
    original = rotation_from_tilt(foot["n_out"], e_long, action[:, 4:8])
    packed = pack_tilt(original, foot["n_out"], e_long)
    rebuilt = rotation_from_tilt(foot["n_out"], e_long, packed)
    psi = np.degrees(np.arctan2(action[:, 4], action[:, 5]))
    tilt = np.degrees(np.linalg.norm(action[:, 6:8], axis=1))
    return geodesic_deg(original, rebuilt), psi, tilt


def stretch_deg(rotation, normal, e_long):
    """Parameter degrees moved by a 1° rotation about tool x."""
    base = pack_tilt(rotation, normal, e_long)[0]
    c, s = np.cos(np.radians(1.0)), np.sin(np.radians(1.0))
    local = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    moved = rotation[0] @ local
    packed = pack_tilt(moved[None], normal, e_long)[0]
    dpsi = abs(float(wrap_deg(np.degrees(np.arctan2(packed[0], packed[1]) - np.arctan2(base[0], base[1])))))
    da = abs(np.degrees(packed[2] - base[2]))
    db = abs(np.degrees(packed[3] - base[3]))
    return float(np.hypot(dpsi, np.hypot(da, db)))


def bin_report(err, psi, tilt):
    wrap = np.abs(psi) >= 170.0
    large = tilt >= 30.0
    groups = {
        "all": np.ones(len(err), dtype=bool),
        "psi_within_10deg_of_wrap": wrap,
        "tilt_at_least_30deg": large,
        "neither": ~wrap & ~large,
    }
    out = {}
    for name, mask in groups.items():
        picked = err[mask]
        out[name] = {
            "n": int(mask.sum()),
            "max_deg": None if picked.size == 0 else float(picked.max()),
            "p99_deg": None if picked.size == 0 else _pct(picked, 99),
        }
    return out


def heading(device, field, axis):
    policy, cfg = load_policy_and_cfg(HEAD_LATEST, device)
    dataset = build_dataset(cfg, HEAD_ZARR, HEAD_NORM, "e93b8756", 8).get_validation_dataset()
    queries = query_indices(dataset)
    zero_rows = {ep: [] for ep in EPISODES}
    label_rows = {ep: [] for ep in EPISODES}
    err, psi, tilt = [], [], []
    for episode, ids in queries.items():
        for idx in ids:
            obs, action = dataset.sampler.sample_sequence(idx)
            pred = np.asarray(predict_sparse_action(policy, obs, device), dtype=np.float64).reshape(-1, 8)
            gt = np.asarray(action["sparse"], dtype=np.float64).reshape(-1, 8)
            n = min(len(pred), len(gt))
            pred, gt = pred[:n], gt[:n]
            sparse = obs["sparse"]
            p0 = np.asarray(sparse["robot0_eef_pos"][-1], dtype=np.float64).reshape(3)
            r0 = rot6d_rows_to_R(np.asarray(sparse["robot0_eef_rot_axis_angle"][-1], dtype=np.float64))
            pred_zero = pred.copy()
            pred_zero[:, 6:8] = 0.0
            label_zero = gt.copy()
            label_zero[:, 6:8] = 0.0
            zero_rows[episode].append(_pose_error(field, axis, pred_zero, gt, p0, r0))
            label_rows[episode].append(_pose_error(field, axis, label_zero, gt, p0, r0))
            trip, trip_psi, trip_tilt = round_trip(field, axis, gt, p0, r0)
            err.append(trip)
            psi.append(trip_psi)
            tilt.append(trip_tilt)
        print("heading", episode, "n", len(ids), flush=True)
    err = np.concatenate(err)
    psi = np.concatenate(psi)
    tilt = np.concatenate(tilt)
    del policy
    torch.cuda.empty_cache()
    return {
        "ckpt": HEAD_LATEST,
        "zero_tilt_pred_a_b_0": _mean_eps(zero_rows),
        "zero_tilt_label_a_b_0": _mean_eps(label_rows),
        "round_trip": bin_report(err, psi, tilt),
        "label_tilt_deg_p99": _pct(tilt, 99),
        "label_abs_psi_deg_p99": _pct(np.abs(psi), 99),
    }


def _mean_eps(per_ep):
    return {
        "mean": {
            name: {
                "pos_mm": float(np.mean([mean_rows(per_ep[ep])[name]["pos_mm"] for ep in EPISODES])),
                "rot_deg": float(np.mean([mean_rows(per_ep[ep])[name]["rot_deg"] for ep in EPISODES])),
            }
            for _index, name in STEPS
        }
    }


def _pack_err(left, right):
    psi = np.abs(wrap_deg(np.degrees(np.arctan2(left[:, 0], left[:, 1]) - np.arctan2(right[:, 0], right[:, 1]))))
    a = np.degrees(np.abs(left[:, 2] - right[:, 2]))
    b = np.degrees(np.abs(left[:, 3] - right[:, 3]))
    return psi, a, b


def causal(device, field, axis):
    path = f"{CAUSAL_DIR}/latest.ckpt"
    policy, cfg = load_policy_and_cfg(path, device)
    dataset = build_dataset(cfg, CAUSAL_ZARR, CAUSAL_NORM, "14308b72", 10).get_validation_dataset()
    queries = query_indices(dataset)
    bags = {key: [] for key in ("geo", "dR", "psi", "a", "b", "swing", "stretch", "own_psi")}
    for episode, ids in queries.items():
        for idx in ids:
            obs, action = dataset.sampler.sample_sequence(idx)
            pred = np.asarray(predict_sparse_action(policy, obs, device), dtype=np.float64).reshape(-1, 10)
            gt = np.asarray(action["sparse"], dtype=np.float64).reshape(-1, 10)
            n = min(len(pred), len(gt), 64)
            pred, gt = pred[:n], gt[:n]
            sparse = obs["sparse"]
            p0 = np.asarray(sparse["robot0_eef_pos"][-1], dtype=np.float64).reshape(3)
            r0 = rot6d_rows_to_R(np.asarray(sparse["robot0_eef_rot_axis_angle"][-1], dtype=np.float64))
            _xp, rp, n_p = rotation_of_rot6d(field, pred, p0, r0)
            _xg, rg, n_g = rotation_of_rot6d(field, gt, p0, r0)
            e_g, _ = e_long_sequence(n_g, axis)
            e_p, _ = e_long_sequence(n_p, axis)
            same_p = pack_tilt(rp, n_g, e_g)
            same_g = pack_tilt(rg, n_g, e_g)
            own_p = pack_tilt(rp, n_p, e_p)
            own_g = pack_tilt(rg, n_g, e_g)
            psi, a, b = _pack_err(same_p, same_g)
            own_psi, _own_a, _own_b = _pack_err(own_p, own_g)
            step = 0
            bags["geo"].append(float(geodesic_deg(rp, rg)[step]))
            bags["dR"].append(float(geodesic_deg(rot6d_rows_to_R(pred[:, 4:10]), rot6d_rows_to_R(gt[:, 4:10]))[step]))
            bags["psi"].append(float(psi[step]))
            bags["a"].append(float(a[step]))
            bags["b"].append(float(b[step]))
            bags["swing"].append(float(np.degrees(np.hypot(same_g[step, 2], same_g[step, 3]))))
            bags["stretch"].append(stretch_deg(rg[step:step + 1], n_g[step:step + 1], e_g[step:step + 1]))
            bags["own_psi"].append(float(own_psi[step]))
        print("causal", episode, "n", len(ids), flush=True)
    summary = {key: {"mean": float(np.mean(values)), "p99": _pct(values, 99)} for key, values in bags.items()}
    swing = np.asarray(bags["swing"])
    psi = np.asarray(bags["psi"])
    summary["psi_when_swing_under_10"] = float(np.mean(psi[swing < 10.0])) if np.any(swing < 10) else None
    summary["psi_when_swing_over_30"] = float(np.mean(psi[swing >= 30.0])) if np.any(swing >= 30) else None
    summary["n_swing_under_10"] = int(np.sum(swing < 10.0))
    summary["n_swing_over_30"] = int(np.sum(swing >= 30.0))
    summary["n"] = len(bags["geo"])
    return summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    field = FixBMoldField(Path(FIELD))
    axis = long_axis(field.vertices)
    report = {"heading_e100": heading(device, field, axis), "causal_e199_step1": causal(device, field, axis)}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
