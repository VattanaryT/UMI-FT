#!/usr/bin/env python3
"""Open-loop and closed-loop comparison of r1rel_norgb and field_v2n_norgb.

Read-only. Does not write checkpoints, zarrs, or normalizers.
r4vt rotation is mold_R_field_0 @ ΔR. r1rel is its relative pose, times the
measured TCP, so both errors are in the fixture frame.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "UMIFT_Data/surface_frames_scripts"),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT / "PyriteEnvSuites"),
]

from fixb_labels import pose7_to_Rp  # noqa: E402
from mold_aware_runtime import rot6d_rows_to_R  # noqa: E402

TRAIN = Path("/mnt/data/umidata/ft/training_outputs")
LOCAL = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
CKPT = {
    "r1rel": TRAIN / "oct_v35_h4s_r1rel_norgb/latest.ckpt",
    "B": TRAIN / "oct_v35_h4s_r4vt_field_v2n_norgb/latest.ckpt",
}
EPISODES = (9, 12, 28, 37, 41)
OUT = ROOT / "closedloop_sim" / "out_r1rel_b"
B_NPZ = ROOT / "closedloop_sim" / "out_lift_deploy" / "policy_B_none_fixb_ep9.npz"
FIELD_V2N_SCRIPTS = Path(
    "/home/boeing/umi_ft_ws/worktrees/UMI-FT-field_v2n/UMIFT_Data/surface_frames_scripts"
)


def _check() -> None:
    got = _phases([6.0, 3.0, 1.0], [0.0, 0.0, 0.02])
    if got != ["contact", "lift-off", "approach"]:
        raise RuntimeError(got)


def _phases(force, clearance) -> list[str]:
    phase = "approach"
    out = []
    for f, c in zip(force, clearance):
        if f > 5.0:
            phase = "contact"
        elif f < 2.0 and c > 0.010:
            phase = "approach"
        elif phase == "contact":
            phase = "lift-off"
        out.append(phase)
    return out


def _angle(a, b) -> float:
    c = float(np.clip((np.trace(np.asarray(a).T @ np.asarray(b)) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _stat(values) -> dict:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "median": round(float(np.median(values)), 2),
        "p90": round(float(np.percentile(values, 90)), 2),
    }


def _point_dataset(cfg) -> None:
    from omegaconf import OmegaConf

    OmegaConf.set_struct(cfg.task.dataset, False)
    optional = {"filter_stats_path", "pose_jump_flag_path", "r2_action_normalizer_path"}
    for key, value in list(cfg.task.dataset.items()):
        if not isinstance(value, str) or not value.startswith("/"):
            continue
        path = Path(value)
        if path.exists():
            continue
        if key == "dataset_path":
            cfg.task.dataset[key] = str(LOCAL / path.parent.name / "acp_replay_buffer.zarr")
        elif key == "action_normalizer_path":
            found = list(Path("/home/boeing/umi_ft_ws/worktrees").glob(f"**/assets/{path.name}"))
            if not found:
                raise SystemExit(f"normalizer missing {path.name}")
            cfg.task.dataset[key] = str(found[0])
        elif key == "long_dt_mask_path":
            cfg.task.dataset[key] = str(LOCAL / "shared/long_dt_gt_100ms.npz")
        elif key in optional or key.endswith("_path"):
            cfg.task.dataset[key] = None
    cfg.task.dataset.exclusion_stats_path = str(OUT / "exclusion.json")


def _world_r1(action, base_p, base_R):
    import PyriteUtility.spatial_math.spatial_utilities as su
    from PyriteConfig.tasks.umift.umift_type_conversions import action21_postprocess

    base = np.eye(4)
    base[:3, :3] = base_R
    base[:3, 3] = base_p
    command, _vt, _stiff, _eoat = action21_postprocess(np.asarray(action, dtype=np.float64), [base], [0])
    pose = su.SE3_to_pose7(command[0])
    rotation, position = pose7_to_Rp(pose)
    return position[0], rotation[0]


def _world_r4(action, foot, frame, dt):
    step = np.asarray(action, dtype=np.float64).reshape(-1)
    dR = rot6d_rows_to_R(step[4:10].reshape(1, 6))[0]
    outward = -np.asarray(frame, dtype=np.float64).reshape(3, 3)[:, 2]
    tang = frame[:, :2] @ step[0:2] * float(dt)
    position = np.asarray(foot, dtype=np.float64).reshape(3) + tang + float(step[2]) * outward
    return position, frame @ dR, float(step[2])


def _open_one(name, ckpt, device) -> dict:
    import hydra
    import torch
    import zarr
    from PyriteEnvSuites.scripts.run_policy_open_loop_val_export import (
        load_policy_and_cfg,
        predict_sparse_action,
    )

    policy, cfg = load_policy_and_cfg(str(ckpt), device)
    _point_dataset(cfg)
    target = str(cfg.task.dataset._target_)
    if "field_v2" in target:
        sys.path.insert(0, str(FIELD_V2N_SCRIPTS))
    if "rgb_0" not in cfg.task.dataset.shape_meta.obs:
        from omegaconf import open_dict

        with open_dict(cfg.task.dataset.shape_meta):
            cfg.task.dataset.shape_meta.obs.rgb_0 = {"type": "rgb", "shape": [3, 224, 224]}
    rgb_keys = list(getattr(policy.obs_encoder, "rgb_keys", []))
    print(
        f"[open] {name} target {target} epochs {int(cfg.training.num_epochs)} rgb {rgb_keys}",
        flush=True,
    )
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    val = dataset.get_validation_dataset()
    zarr_path = str(cfg.task.dataset.dataset_path)
    root = zarr.open_group(zarr_path, mode="r")["data"]
    ds = int(cfg.task.shape_meta["sample"]["action"]["sparse"]["down_sample_steps"])
    dt = ds / 15.0
    present = sorted({int(ep) for ep, _rgb in val.sampler.indices})
    print(f"[open] {name} val episodes {present}", flush=True)
    keep = []
    last = {}
    rgb_t = {}
    for idx, (ep, rgb) in enumerate(val.sampler.indices):
        ep = int(ep)
        if ep not in EPISODES:
            continue
        if ep not in rgb_t:
            rgb_t[ep] = np.asarray(root[f"episode_{ep}"]["rgb_time_stamps_0"]).reshape(-1)
        t = float(rgb_t[ep][int(rgb)] - rgb_t[ep][0])
        if t - last.get(ep, -1e9) < 0.5:
            continue
        last[ep] = t
        keep.append((idx, ep, t))
    modes = ("zeros", "images") if rgb_keys else ("zeros",)
    buckets = {
        mode: {phase: {"position_mm": [], "rotation_deg": [], "delta_n_mm": []} for phase in ("approach", "contact", "lift-off")}
        for mode in modes
    }
    r4 = "field_v2" in target or int(cfg.task.shape_meta["action"]["shape"][0]) == 10
    phased = {}
    for n, (idx, ep, t) in enumerate(keep):
        group = root[f"episode_{ep}"]
        if ep not in phased:
            robot_t = np.asarray(group["robot_time_stamps_0"]).reshape(-1)
            robot_t = robot_t - robot_t[0]
            pose = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
            _rot, pos = pose7_to_Rp(pose)
            frame_all = np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
            foot_all = np.asarray(group["mold_x_s_0"], dtype=np.float64)
            wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
            key = "wrench_time_stamps_left_0" if "wrench_time_stamps_left_0" in group else "wrench_time_stamps_0"
            wt = np.asarray(group[key]).reshape(-1)
            wt = wt - wt[0]
            j = np.clip(np.searchsorted(wt, robot_t), 0, len(wt) - 1)
            force = np.linalg.norm(wrench[j, :3], axis=1)
            clearance = np.einsum("ij,ij->i", pos - foot_all, -frame_all[:, :, 2])
            phased[ep] = (robot_t, _phases(force, clearance), frame_all, foot_all, pose)
        robot_t, labels, frame_all, foot_all, pose = phased[ep]
        sample = val[idx]
        sparse = {
            key: value.numpy()
            for key, value in sample["obs"]["sparse"].items()
            if not str(key).startswith("rgb") or key in rgb_keys
        }
        preds = {}
        for mode in modes:
            obs = dict(sparse)
            if mode == "zeros":
                for key in rgb_keys:
                    obs[key] = np.zeros_like(sparse[key])
            preds[mode] = np.asarray(predict_sparse_action(policy, {"sparse": obs}, device), dtype=np.float64)
        gt = sample["action"]["sparse"].numpy()
        k = int(np.clip(np.searchsorted(robot_t, t), 1, len(robot_t) - 1))
        k = k - 1 if abs(robot_t[k - 1] - t) <= abs(robot_t[k] - t) else k
        rotation, position = pose7_to_Rp(pose[k:k + 1])
        frame = frame_all[k]
        foot = foot_all[k]
        phase = labels[k]
        if r4:
            _p_gt, R_gt, dn_gt = _world_r4(gt[0], foot, frame, dt)
        else:
            _p_gt, R_gt = _world_r1(gt, position[0], rotation[0])
            dn_gt = None
        outward = -frame[:, 2]
        for mode, pred in preds.items():
            if r4:
                p_hat, R_hat, dn_hat = _world_r4(pred[0], foot, frame, dt)
                dn = abs(dn_hat - dn_gt) * 1e3
            else:
                p_hat, R_hat = _world_r1(pred, position[0], rotation[0])
                dn = abs(float(np.dot(p_hat - _p_gt, outward))) * 1e3
            buckets[mode][phase]["position_mm"].append(float(np.linalg.norm(p_hat - _p_gt) * 1e3))
            buckets[mode][phase]["rotation_deg"].append(_angle(R_hat, R_gt))
            buckets[mode][phase]["delta_n_mm"].append(dn)
        if n and n % 40 == 0:
            print(f"[open] {name} {n}/{len(keep)}", flush=True)
    del policy, dataset, val
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "epochs": int(cfg.training.num_epochs),
        "n": len(keep),
        "r4": r4,
        "rgb_keys": rgb_keys,
        "dt_s": dt,
        "modes": {
            mode: {phase: {key: _stat(values) for key, values in cols.items()} for phase, cols in phases.items()}
            for mode, phases in buckets.items()
        },
    }


def _strokes(position, times, field, axes) -> list[str]:
    step = max(1, int(round(0.1 / np.median(np.diff(times)))))
    position, times = position[::step], times[::step]
    query = field.query(position)
    foot = np.asarray(query["x_s"])
    normal = np.asarray(query["R"])[:, :, 2]
    s = foot @ axes[:, 0]
    s = np.convolve(s, np.ones(5) / 5, mode="same")
    slope = np.gradient(s, times)
    sign = np.sign(slope)
    sign[np.abs(slope) < 0.01] = 0
    last = 0.0
    for i, value in enumerate(sign):
        if value == 0:
            sign[i] = last
        else:
            last = value
    cuts = [0] + [int(i) + 1 for i in np.flatnonzero(sign[1:] * sign[:-1] < 0)] + [len(times) - 1]
    rows = []
    for a, b in zip(cuts, cuts[1:]):
        if times[b] - times[a] < 1.5 or abs(s[b] - s[a]) < 0.12:
            continue
        side = "+Z" if np.median(normal[a:b, 2]) > 0 else "-Z"
        start = "base" if s[a] < -0.22 else "far"
        end = "base" if s[b] < -0.22 else "far"
        rows.append(f"{side} {start}->{end}")
    return rows


def _closed(name, path, demo, field, axes) -> dict:
    blob = np.load(path)
    t = np.asarray(blob["t_s"])
    contact = np.asarray(blob["penetration_m"]) > 0
    dt = float(np.median(np.diff(t)))
    demo_t = np.asarray(demo["t_s"])
    demo_p = np.asarray(demo["position"])
    idx = np.clip(np.searchsorted(demo_t, t), 0, len(demo_t) - 1)
    path_mm = np.linalg.norm(blob["position"] - demo_p[idx], axis=1) * 1e3
    edges = np.flatnonzero(contact[:-1] & ~contact[1:])
    mid = 0
    for i in edges:
        later = np.flatnonzero(contact[i + 1:])
        if later.size and float(t[i + 1 + later[0]] - t[i]) <= 0.5:
            mid += 1
    rotation, _p = pose7_to_Rp(np.asarray(blob["pose7"]))
    rel = np.einsum("nji,njk->nik", rotation[:-1], rotation[1:])
    c = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    deg = np.degrees(np.arccos(c))
    strokes = _strokes(blob["position"], t, field, axes)
    return {
        "contact_s": round(float(np.sum(contact) * dt), 1),
        "path_mm_contact": None if not np.any(contact) else round(float(np.median(path_mm[contact])), 1),
        "mid_stroke_losses": int(mid),
        "tilt_gt_10": int(np.sum(deg > 10.0)),
        "tilt_max": round(float(deg.max()), 1),
        "strokes": strokes,
        "final_dn_mm": round(float(blob["delta_n_m"][-1]) * 1e3, 1),
        "final_clearance_mm": round(float(field.query(blob["position"][-1:])["sd"][0]) * 1e3, 1),
    }


def _closed_loop() -> dict:
    import closedloop_sim.policy_rollout as pr
    from fixb_field import FixBMoldField

    _raw = pr._obs_raw

    def _obs(shape, *args):
        if "rgb_0" not in shape["obs"] and "rgb_0" in shape["raw"]:
            from omegaconf import open_dict

            with open_dict(shape):
                del shape.raw["rgb_0"]
                if "rgb_0" in shape.sample.obs.sparse:
                    del shape.sample.obs.sparse["rgb_0"]
        return _raw(shape, *args)

    pr._obs_raw = _obs
    pr.OUT = OUT
    pr.CHECKPOINTS["r1rel_norgb"] = str(CKPT["r1rel"])
    demo = pr._demo_episode(9)
    print("rollout r1rel_norgb", flush=True)
    pr.rollout("r1rel_norgb", demo, episode=9)
    field = FixBMoldField(pr.FIELD)
    import zarr

    axes = np.asarray(
        zarr.open_group(
            "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_field_v2_k3000/acp_replay_buffer.zarr",
            mode="r",
        ).attrs["field_v2_axes"]["axes"],
        dtype=np.float64,
    ).reshape(3, 3)
    return {
        "r1rel": _closed("r1rel", OUT / "policy_r1rel_norgb_fixb_ep9.npz", demo, field, axes),
        "B": _closed("B", B_NPZ, demo, field, axes),
        "demo_strokes": _strokes(demo["position"], demo["t_s"], field, axes),
    }


def main() -> None:
    import torch

    _check()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opened = {}
    for name, ckpt in CKPT.items():
        opened[name] = _open_one(name, ckpt, device)
        (OUT / "open_loop.json").write_text(json.dumps(opened, indent=2) + "\n")
    closed = _closed_loop()
    payload = {"open_loop": opened, "closed_loop": closed, "note": "r1rel 40 epochs, B 200 epochs"}
    (OUT / "compare.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
