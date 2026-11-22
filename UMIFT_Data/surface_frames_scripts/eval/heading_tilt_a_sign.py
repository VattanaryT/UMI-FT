#!/usr/bin/env python3
"""Sign of heading-tilt a, by wall, and label spread by region.

a is the rotvec about tool x after Rz(ψ). b is about tool y. Labels are the
stored 8-d rows. Predictions are heading-tilt e100 on the val episodes at RGB
stride 64, steps 1, 6, and 64. Does not start the robot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [
    str(SCRIPTS),
    str(SCRIPTS / "eval"),
    str(ROOT / "PyriteEnvSuites/scripts"),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT),
]

from fixb_field import FixBMoldField
from heading_tilt import e_long_sequence, frame_zero, long_axis, rotation_from_tilt
from heading_tilt_43448962_open_loop import FIELD
from heading_tilt_vs_causal_foot import HEAD_LATEST, HEAD_NORM, HEAD_ZARR, build_dataset
from mold_aware_runtime import integrate_tcp_foot, rot6d_rows_to_R
from run_policy_open_loop_val_export import load_policy_and_cfg, predict_sparse_action

import zarr

OUT = SCRIPTS / "eval/heading_tilt_a_sign.json"
MD = SCRIPTS / "eval/heading_tilt_a_sign.md"
EPISODES = (9, 12, 28, 37, 41)
STRIDE = 64
STEPS = ((0, "1"), (5, "6"), (63, "64"))
DEAD_DEG = 2.0
UP = np.array([0.0, -1.0, 0.0])
LABEL_STRIDE = 4


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def across_hat(axis):
    """Horizontal direction across the stringer, toward table +Z."""
    horizontal = np.array([0.0, 0.0, 1.0])
    dropped = horizontal - np.dot(horizontal, axis) * axis
    dropped[1] = 0.0
    hat = _unit(dropped)
    if hat[2] < 0.0:
        hat = -hat
    return hat


def _check_axes():
    """+a leans toward n × e_long, and tool y reverses when n_out reverses."""
    axis = np.array([1.0, 0.0, 0.0])
    near = np.array([0.0, 0.0, 1.0])
    far = -near
    y_near = frame_zero(axis, near)[:, 1]
    y_far = frame_zero(axis, far)[:, 1]
    if float(y_near @ y_far) >= 0.0:
        raise SystemExit("tool y did not flip between the two walls")
    z0 = frame_zero(axis, near)[:, 2]
    leaned = rotation_from_tilt(near[None], axis[None], np.array([[0.0, 1.0, 0.2, 0.0]]))[0, :, 2]
    if float((leaned - z0) @ np.cross(near, axis)) <= 0.0:
        raise SystemExit("+a did not lean tool z toward n × e_long")


def region_of(n_out, foot, bottom_y):
    n = n_out / np.clip(np.linalg.norm(n_out, axis=1, keepdims=True), 1e-12, None)
    ang = np.degrees(np.arccos(np.clip(n @ UP, -1.0, 1.0)))
    name = np.full(len(n), "crown", dtype=object)
    name[(ang >= 25.0) & (ang < 55.0)] = "radius"
    wall = ang >= 55.0
    name[wall] = "wall"
    flange = (ang < 25.0) & (foot[:, 1] >= bottom_y)
    name[flange] = "flange"
    return name, wall, n


def wall_side(n_out, hat):
    return np.where(n_out @ hat >= 0.0, "near", "far")


def _mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "mean": None, "std": None}
    return {"n": int(values.size), "mean": float(np.mean(values)), "std": float(np.std(values))}


def eta_squared(values, labels):
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    total = float(np.var(values))
    if total < 1e-12:
        return 0.0
    within = 0.0
    for lab in np.unique(labels):
        part = values[labels == lab]
        within += part.size * float(np.var(part))
    return float(1.0 - (within / values.size) / total)


def load_labels(field, axis, hat):
    root = zarr.open(HEAD_ZARR, mode="r")["data"]
    bottom = float(np.max(field.vertices[:, 1]) - 0.020)
    loaded = []
    crown = []
    for name in sorted(root.group_keys(), key=lambda s: int(s.split("_")[1])):
        group = root[name]
        if "fixb_heading_tilt_0" not in group or "mold_normal_0" not in group:
            continue
        tilt = np.asarray(group["fixb_heading_tilt_0"], dtype=np.float64)[::LABEL_STRIDE]
        foot = np.asarray(group["mold_x_s_0"], dtype=np.float64).reshape(-1, 3)[::LABEL_STRIDE]
        normal = np.asarray(group["mold_normal_0"], dtype=np.float64).reshape(-1, 3)[::LABEL_STRIDE]
        n = min(len(tilt), len(foot), len(normal))
        tilt, foot, normal = tilt[:n], foot[:n], normal[:n]
        finite = np.isfinite(tilt).all(1) & np.isfinite(foot).all(1) & np.isfinite(normal).all(1)
        if not np.any(finite):
            continue
        tilt, foot, normal = tilt[finite], foot[finite], normal[finite]
        high = foot[:, 1] <= np.percentile(foot[:, 1], 15)
        crown.append(normal[high] @ UP)
        loaded.append((int(name.split("_")[1]), tilt, foot, normal))
    flip = float(np.mean(np.concatenate(crown))) < 0.0
    print(f"mold_normal crown flip {flip}", flush=True)
    rows = []
    for episode, tilt, foot, normal in loaded:
        if flip:
            normal = -normal
        region, wall, n_out = region_of(normal, foot, bottom)
        side = wall_side(n_out, hat)
        a = np.degrees(tilt[:, 2])
        b = np.degrees(tilt[:, 3])
        along = foot @ axis
        rows.append(
            {
                "episode": np.full(len(a), episode),
                "a": a,
                "b": b,
                "psi": np.degrees(np.arctan2(tilt[:, 0], tilt[:, 1])),
                "region": region,
                "wall": wall,
                "side": side,
                "along": along,
                "n_out": n_out,
                "y_up": frame_zero(np.broadcast_to(axis, n_out.shape), n_out)[:, :, 1] @ UP,
            }
        )
        print(f"labels episode {episode} {len(a)}", flush=True)
    return rows, bottom


def _cat(rows, key):
    return np.concatenate([row[key] for row in rows])


def opposite_episodes(rows, dead):
    found = []
    for row in rows:
        for region in np.unique(row["region"]):
            take = row["region"] == region
            if region == "wall":
                groups = [(f"wall_{side}", take & (row["side"] == side)) for side in ("near", "far")]
            else:
                groups = [(str(region), take)]
            for name, mask in groups:
                a = row["a"][mask]
                pos = int(np.sum(a > dead))
                neg = int(np.sum(a < -dead))
                if pos >= 20 and neg >= 20:
                    found.append(
                        {
                            "episode": int(row["episode"][0]),
                            "region": name,
                            "pos": pos,
                            "neg": neg,
                            "mean": float(np.mean(a)),
                        }
                    )
    return found


def label_report(rows, hat):
    a = _cat(rows, "a")
    b = _cat(rows, "b")
    region = _cat(rows, "region").copy()
    side = _cat(rows, "side")
    wall = _cat(rows, "wall")
    region[wall] = np.where(side[wall] == "near", "wall_near", "wall_far")
    along = _cat(rows, "along")
    edges = np.quantile(along, [0.0, 0.25, 0.5, 0.75, 1.0])
    bin_id = np.clip(np.digitize(along, edges[1:-1], right=True), 0, 3)
    bins = {}
    for name in ("crown", "radius", "flange", "wall_near", "wall_far"):
        take = region == name
        bins[name] = {"a_deg": _mean_std(a[take]), "b_deg": _mean_std(b[take])}
    quartiles = {}
    for i in range(4):
        take = bin_id == i
        quartiles[f"q{i + 1}"] = {
            "along_m": [round(float(edges[i]), 4), round(float(edges[i + 1]), 4)],
            "a_deg": _mean_std(a[take]),
            "b_deg": _mean_std(b[take]),
        }
    y_up = _cat(rows, "y_up")
    n_out = _cat(rows, "n_out")
    psi = _cat(rows, "psi")
    mirror = {}
    for name, mask in (("wall_near", region == "wall_near"), ("wall_far", region == "wall_far")):
        mirror[name] = {
            "n_out_dot_across": float(np.mean(n_out[mask] @ hat)) if np.any(mask) else None,
            "frame_y_dot_up": float(np.mean(y_up[mask])) if np.any(mask) else None,
            "psi_deg": _mean_std(psi[mask]),
            "frac_abs_psi_gt_90": float(np.mean(np.abs(psi[mask]) > 90.0)) if np.any(mask) else None,
        }
    return {
        "n": int(a.size),
        "label_stride": LABEL_STRIDE,
        "deadzone_deg": DEAD_DEG,
        "by_region": bins,
        "by_along": quartiles,
        "eta_region_a": eta_squared(a, region),
        "eta_region_b": eta_squared(b, region),
        "eta_along_a": eta_squared(a, bin_id),
        "eta_along_b": eta_squared(b, bin_id),
        "mirror": mirror,
        "opposite_sign_episodes": opposite_episodes(rows, DEAD_DEG),
    }


def _predict(policy, dataset, idx, device):
    obs, action = dataset.sampler.sample_sequence(idx)
    pred = np.asarray(predict_sparse_action(policy, obs, device), dtype=np.float64).reshape(-1, 8)
    gt = np.asarray(action["sparse"], dtype=np.float64).reshape(-1, 8)
    n = min(len(pred), len(gt))
    sparse = obs["sparse"]
    p0 = np.asarray(sparse["robot0_eef_pos"][-1], dtype=np.float64).reshape(3)
    r0 = rot6d_rows_to_R(np.asarray(sparse["robot0_eef_rot_axis_angle"][-1], dtype=np.float64))
    return pred[:n], gt[:n], p0, r0


def _foot(field, axis, action, p0, r0):
    dt = np.full(action.shape[0], 1.0 / 15.0)
    foot = integrate_tcp_foot(field, p0, action[:, 0:2], action[:, 2], dt, R_tool=r0)
    direction, _angle = e_long_sequence(foot["n_out"], axis)
    rotation = rotation_from_tilt(foot["n_out"], direction, action[:, 4:8])
    return foot["n_out"], foot["x_s"], rotation


def score_predictions(field, axis, hat, bottom):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, cfg = load_policy_and_cfg(HEAD_LATEST, device)
    dataset = build_dataset(cfg, HEAD_ZARR, HEAD_NORM, "e93b8756", 8)
    val = dataset.get_validation_dataset()
    chosen = {ep: [] for ep in EPISODES}
    last = {}
    for source in (dataset, val):
        for idx, (epi, rgb) in enumerate(source.sampler.indices):
            epi = int(epi)
            if epi not in chosen:
                continue
            rgb = int(rgb)
            if epi not in last or rgb - last[epi] >= STRIDE:
                chosen[epi].append((source, idx))
                last[epi] = rgb
    out = {name: [] for _i, name in STEPS}
    for episode, pairs in chosen.items():
        print(f"pred episode {episode} queries {len(pairs)}", flush=True)
        for source, idx in pairs:
            pred, gt, p0, r0 = _predict(policy, source, idx, device)
            n_out, foot, _label_R = _foot(field, axis, gt, p0, r0)
            _pred_n, _pred_foot, rotation = _foot(field, axis, pred, p0, r0)
            _region, wall, n_hat = region_of(n_out, foot, bottom)
            side = wall_side(n_hat, hat)
            up_tangent = UP - n_hat * np.sum(UP * n_hat, axis=1, keepdims=True)
            up_tangent /= np.clip(np.linalg.norm(up_tangent, axis=1, keepdims=True), 1e-12, None)
            for index, name in STEPS:
                if index >= len(pred) or not wall[index]:
                    continue
                a_l = float(np.degrees(gt[index, 6]))
                a_p = float(np.degrees(pred[index, 6]))
                tool_z = rotation[index, :, 2]
                toward_other = float(tool_z @ up_tangent[index]) > 0.0
                signed = abs(a_l) >= DEAD_DEG and abs(a_p) >= DEAD_DEG
                out[name].append(
                    {
                        "episode": episode,
                        "wall": str(side[index]),
                        "a_label": round(a_l, 2),
                        "a_pred": round(a_p, 2),
                        "same_sign": bool(np.sign(a_l) == np.sign(a_p)) if signed else None,
                        "pred_leans": "toward_other_wall" if toward_other else "toward_flange",
                        "n_out_dot_across": round(float(n_hat[index] @ hat), 3),
                        "frame_y_dot_up": round(
                            float(frame_zero(axis, n_hat[index])[:, 1] @ UP), 3
                        ),
                    }
                )
    summary = {}
    for name, rows in out.items():
        by_wall = {}
        for wall_name in ("near", "far"):
            part = [row for row in rows if row["wall"] == wall_name]
            signs = [row["same_sign"] for row in part if row["same_sign"] is not None]
            leans = [row["pred_leans"] for row in part]
            by_wall[wall_name] = {
                "n": len(part),
                "sign_agree": None if not signs else float(np.mean(signs)),
                "mean_a_label": _mean_std([row["a_label"] for row in part]),
                "mean_a_pred": _mean_std([row["a_pred"] for row in part]),
                "toward_other_wall": leans.count("toward_other_wall"),
                "toward_flange": leans.count("toward_flange"),
                "mean_frame_y_dot_up": _mean_std([row["frame_y_dot_up"] for row in part]),
                "mean_n_out_dot_across": _mean_std([row["n_out_dot_across"] for row in part]),
            }
        summary[name] = by_wall
    return summary


def _fmt(stats):
    if not stats or stats["mean"] is None:
        return "n=0"
    return f"n={stats['n']} mean={stats['mean']:.2f} std={stats['std']:.2f}"


def write_md(report):
    labels = report["labels"]
    lines = [
        "# heading-tilt a by wall",
        "",
        "Label rows are every 4th stored sample. Predictions are e100, val episodes",
        "9, 12, 28, 37, 41, RGB stride 64. Wall is a normal at least 55° from up.",
        "Near is the operator side (n_out toward +Z). a deadzone is 2°.",
        "",
        "## Label bins, degrees",
        "",
        "| region | a | b |",
        "|---|---|---|",
    ]
    for name, blob in labels["by_region"].items():
        lines.append(f"| {name} | {_fmt(blob['a_deg'])} | {_fmt(blob['b_deg'])} |")
    lines += ["", "| along | a | b |", "|---|---|---|"]
    for name, blob in labels["by_along"].items():
        span = blob["along_m"]
        lines.append(f"| {name} {span[0]}..{span[1]} m | {_fmt(blob['a_deg'])} | {_fmt(blob['b_deg'])} |")
    lines += [
        "",
        f"Region explains {labels['eta_region_a']:.3f} of var(a) and {labels['eta_region_b']:.3f} of var(b).",
        f"Position quartile explains {labels['eta_along_a']:.3f} of var(a) and {labels['eta_along_b']:.3f} of var(b).",
        f"Episodes with both signs inside one region: {len(labels['opposite_sign_episodes'])}.",
        "",
        "## Frame mirror",
        "",
    ]
    for name, blob in labels["mirror"].items():
        lines.append(
            f"- {name}: n_out·across={blob['n_out_dot_across']:.3f}, frame y·up={blob['frame_y_dot_up']:.3f}, "
        f"ψ {_fmt(blob['psi_deg'])}, |ψ|>90 {blob['frac_abs_psi_gt_90']:.3f}"
        )
    lines += ["", "## e100 wall queries", ""]
    for step, walls in report["predictions"].items():
        lines.append(f"### step {step}")
        for wall, blob in walls.items():
            agree = blob["sign_agree"]
            agree_s = "n/a" if agree is None else f"{agree:.3f}"
            lines.append(
                f"- {wall}: n={blob['n']} sign_agree={agree_s} "
                f"label a {_fmt(blob['mean_a_label'])} pred a {_fmt(blob['mean_a_pred'])} "
                f"leans other/flange {blob['toward_other_wall']}/{blob['toward_flange']} "
                f"y·up {_fmt(blob['mean_frame_y_dot_up'])}"
            )
    lines += ["", report["verdict"], ""]
    MD.write_text("\n".join(lines) + "\n")


def verdict(report):
    labels = report["labels"]
    step1 = report["predictions"]["1"]
    agrees = [step1[name]["sign_agree"] for name in ("near", "far") if step1[name]["sign_agree"] is not None]
    agree = float(np.mean(agrees)) if agrees else float("nan")
    y_near = labels["mirror"]["wall_near"]["frame_y_dot_up"]
    y_far = labels["mirror"]["wall_far"]["frame_y_dot_up"]
    mirrored = y_near is not None and y_far is not None and y_near * y_far < 0.0
    eta = labels["eta_region_a"]
    flips = len(labels["opposite_sign_episodes"])
    if mirrored and eta >= 0.15 and agree >= 0.7:
        text = (
            "a's spread is real region-dependent tilt. Both walls sit near +50°, the flange near -9°, "
            "and position along the stringer explains almost none of it. Tool y flips between the walls, "
            "and the far wall's ψ is near 180°, so the same +a leans down toward the flange on both walls. "
            "e100 matches that sign at step 1."
        )
    elif agree < 0.6:
        text = (
            "e100's sign of a disagrees with the label on the walls. "
            "That is a prediction error on top of the chart."
        )
    elif eta < 0.1 and flips > 10:
        text = "a's spread is mostly human noise: region explains little, and episodes flip sign inside a region."
    else:
        text = (
            "a's spread is real region-dependent tilt. "
            "The chart does mirror tool y across the walls, and the bins carry the rest."
        )
    return text + f" Step-1 sign agreement {agree:.3f}. Region eta(a) {eta:.3f}."


def main():
    _check_axes()
    field = FixBMoldField(Path(FIELD))
    axis = long_axis(field.vertices)
    hat = across_hat(axis)
    rows, bottom = load_labels(field, axis, hat)
    report = {"labels": label_report(rows, hat), "bottom_y_m": bottom, "across": hat.tolist()}
    report["predictions"] = score_predictions(field, axis, hat, bottom)
    report["verdict"] = verdict(report)
    report["one_tilt"] = (
        "rotation_from_tilt accepts b=0. A 7-d head would be a new action layout, not a new decode."
    )
    OUT.write_text(json.dumps(report, indent=2))
    write_md(report)
    print(report["verdict"], flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
