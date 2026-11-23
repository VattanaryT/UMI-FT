#!/usr/bin/env python3
"""Does where the tool is on the stringer help predict tilt a and δn?

Held-out episodes are the seed-42 validation split. Inputs never include the
targets, signed distance, the tool rotation, or raw xyz. Images are not used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import zarr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / "eval"), str(ROOT / "PyriteML"), str(ROOT)]

import frame_variance as fv  # noqa: E402
import openloop_causal_foot as cf  # noqa: E402
from fixb_labels import pose7_to_Rp, so3_to_rot6d_rows  # noqa: E402
from heading_tilt import long_axis  # noqa: E402

ZARR = zarr.open(fv.ZARR, "r")["data"]
FIELD = np.load(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
OUT = SCRIPTS / "eval/position_inputs.json"
MD = SCRIPTS / "eval/position_inputs.md"


def _r2(pred, target):
    resid = float(np.sum((target - pred) ** 2))
    total = float(np.sum((target - target.mean()) ** 2))
    return round(1.0 - resid / total, 3) if total > 0 else 0.0


def _wrench_world(group, robot_t, rotation):
    wrench_t = np.asarray(group["wrench_time_stamps_0"], np.float64).reshape(-1)
    wrench = np.asarray(group["wrench_left_0"], np.float64)
    order = np.argsort(wrench_t)
    wrench_t, wrench = wrench_t[order], wrench[order]
    keep = np.ones(len(wrench_t), dtype=bool)
    keep[1:] = np.diff(wrench_t) > 1e-9
    src_t, src = wrench_t[keep], wrench[keep]
    sampled = np.column_stack([np.interp(robot_t, src_t, src[:, i]) for i in range(6)])
    force = np.einsum("nij,nj->ni", rotation, sampled[:, :3])
    torque = np.einsum("nij,nj->ni", rotation, sampled[:, 3:])
    return sampled, force, torque


def load():
    axis = long_axis(np.asarray(FIELD["vertices"], np.float64))
    across = np.cross(axis, np.array([0.0, 1.0, 0.0]))
    across = across / max(float(np.linalg.norm(across)), 1e-12)
    if across[2] < 0.0:
        across = -across
    held = set(cf.held_out_episodes())
    parts = []
    for episode in range(50):
        if episode in fv.SKIP or f"episode_{episode}" not in ZARR:
            continue
        group = ZARR[f"episode_{episode}"]
        sl = slice(None, None, fv.STRIDE)
        robot_t = np.asarray(group["robot_time_stamps_0"][sl], np.float64).reshape(-1)
        pose = np.asarray(group["ts_pose_fb_0"][sl], np.float64)
        tilt = np.asarray(group["fixb_heading_tilt_0"][sl], np.float64)
        dn = np.asarray(group["fixb_delta_n_0"][sl], np.float64).reshape(-1)
        normal = np.asarray(group["mold_normal_0"][sl], np.float64)
        frame = np.asarray(group["mold_R_field_0"][sl], np.float64).reshape(-1, 3, 3)
        foot = np.asarray(group["mold_x_s_0"][sl], np.float64)
        n = min(len(robot_t), len(pose), len(tilt), len(dn), len(normal), len(frame), len(foot))
        rotation, _p = pose7_to_Rp(pose[:n])
        tcp, force, torque = _wrench_world(group, robot_t[:n], rotation)
        field_force = np.einsum("nji,nj->ni", frame[:n], force)
        field_torque = np.einsum("nji,nj->ni", frame[:n], torque)
        parts.append(
            {
                "episode": np.full(n, episode),
                "held": np.full(n, episode in held),
                "a": np.degrees(tilt[:n, 2]),
                "dn": dn[:n],
                "normal": normal[:n],
                "rot6d": so3_to_rot6d_rows(frame[:n]),
                "tcp": tcp,
                "field_wrench": np.column_stack([field_force, field_torque]),
                "foot": foot[:n],
                "u": foot[:n] @ axis,
                "v": foot[:n] @ across,
            }
        )
        print(f"episode {episode}", flush=True)
    packed = {key: np.concatenate([row[key] for row in parts]) for key in parts[0]}
    region = fv.regions_of(
        packed["normal"],
        packed["foot"],
        float(np.percentile(packed["foot"][:, 1], 90)),
        fv.across_hat(packed["foot"]),
    )
    flange = packed["foot"][region == "flange", 1]
    packed["flange_distance"] = np.abs(packed["foot"][:, 1] - float(np.median(flange)))
    packed["region"] = region
    packed["axis"] = axis
    packed["across"] = across
    return packed


def _fit(train_x, train_y, test_x, test_y):
    model = HistGradientBoostingRegressor(
        max_iter=80, max_depth=4, min_samples_leaf=40, learning_rate=0.1, random_state=0
    )
    model.fit(train_x, train_y)
    return {"held_out_r2": _r2(model.predict(test_x), test_y), "train_r2": _r2(model.predict(train_x), train_y)}


def main():
    packed = load()
    names = list(fv.REGIONS)
    one_hot = np.column_stack([packed["region"] == name for name in names]).astype(np.float64)
    base = np.column_stack([packed["tcp"], packed["normal"], packed["rot6d"]])
    uv = np.column_stack([packed["u"], packed["v"]])
    sets = {
        "obs": base,
        "obs_uv": np.column_stack([base, uv]),
        "obs_uv_field_wrench": np.column_stack([base, uv, packed["field_wrench"]]),
        "obs_field_wrench_flange_region": np.column_stack(
            [base, packed["field_wrench"], packed["flange_distance"], one_hot]
        ),
    }
    train = ~packed["held"]
    test = packed["held"]
    report = {
        "uv": (
            "u is the foot dotted with the mold's long axis (PCA, flipped so +X is positive). "
            "v is the foot dotted with the horizontal perpendicular that points toward +Z."
        ),
        "flange_distance": "absolute Y distance from the foot to the median Y of flange samples. +Y is down.",
        "base_obs": "TCP wrench (6), outward normal (3), R_field rot6d (6). No image, xyz, tool rotation, or signed distance.",
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "held_episodes": sorted(set(packed["episode"][test].tolist())),
        "targets": {},
    }
    for target in ("a", "dn"):
        report["targets"][target] = {}
        y_train = packed[target][train]
        y_test = packed[target][test]
        for name, values in sets.items():
            score = _fit(values[train], y_train, values[test], y_test)
            report["targets"][target][name] = score
            print(target, name, score, flush=True)
    OUT.write_text(json.dumps(report, indent=2))
    lines = [
        "# Position inputs for a and δn",
        "",
        report["uv"],
        report["flange_distance"],
        report["base_obs"],
        f"Train rows {report['n_train']}, held-out rows {report['n_test']}, episodes {report['held_episodes']}.",
        "R² is on the held-out episodes.",
        "",
        "| features | a R² | δn R² |",
        "|---|---:|---:|",
    ]
    for name in sets:
        lines.append(
            f"| {name} | {report['targets']['a'][name]['held_out_r2']} | "
            f"{report['targets']['dn'][name]['held_out_r2']} |"
        )
    lines.append("")
    MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
