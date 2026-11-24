#!/usr/bin/env python3
"""v3r vs r1rel on frames both datasets keep. Recorded RGB.

Horizons are action rows 1, 5, and 16. v3r position integrates tangential
velocity from the query foot and adds δn at the scored row. v3r rotation is
R_tcp @ ΔR. r1rel is the relative pose composed with the measured TCP.
Hold predicts the measured pose and the current normal offset.
"""

from __future__ import annotations

import json
import sys

import numpy as np

import compare_r1rel_b as base
import compare_v3r_r1rel_rgb as rgb

STEPS = (1, 5, 16)
OUT = base.ROOT / "closedloop_sim" / "out_v3r_r1rel_intersect"
PHASES = ("approach", "contact", "lift-off")


def _check() -> None:
    rgb._check()
    frame = np.eye(3)
    foot = np.zeros(3)
    action = np.zeros((2, 10))
    action[:, 4:10] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    action[0, 0] = 1.0
    action[0, 2] = 0.01
    dts = np.array([0.1, 0.1])
    p, R, dn = _v3r_at(action, np.stack([foot, foot]), np.stack([frame, frame]), dts, 1)
    if dn != 0.01 or not np.allclose(R, np.eye(3)) or not np.allclose(p, [0.1, 0.0, -0.01]):
        raise RuntimeError((p, R, dn))


def _v3r_at(action, feet, frames, dts, step):
    """Position integrates v from the query foot. Rotation is the stored tool rot6d.

    The v3r sample keeps absolute tool rotation: action poses are absolute, so
    the dataset's R(t).T @ R(t+k) is applied with R(t) = I.
    """
    row = step - 1
    p = np.asarray(feet[0], dtype=np.float64).reshape(3).copy()
    for k in range(step):
        p = p + frames[k, :, :2] @ action[k, 0:2] * float(dts[k])
    outward = -frames[row, :, 2]
    p = p + float(action[row, 2]) * outward
    R = base.rot6d_rows_to_R(action[row, 4:10].reshape(1, 6))[0]
    return p, R, float(action[row, 2])


def _r1_at(action, base_p, base_R, step):
    import PyriteUtility.spatial_math.spatial_utilities as su
    from PyriteConfig.tasks.umift.umift_type_conversions import action21_postprocess

    base_se3 = np.eye(4)
    base_se3[:3, :3] = base_R
    base_se3[:3, 3] = base_p
    command, _vt, _stiff, _eoat = action21_postprocess(np.asarray(action, dtype=np.float64), [base_se3], [0])
    pose = su.SE3_to_pose7(command[0])
    rotation, position = base.pose7_to_Rp(pose)
    return position[step - 1], rotation[step - 1]


def _empty():
    return {step: {"model": {}, "hold": {}} for step in STEPS}


def _cols():
    return {phase: {"position_mm": [], "rotation_deg": [], "delta_n_mm": []} for phase in PHASES}


def _finish(buckets):
    out = {}
    for step, kinds in buckets.items():
        out[str(step)] = {}
        for kind, phases in kinds.items():
            packed = {}
            for phase, cols in phases.items():
                packed[phase] = {key: base._stat(values) for key, values in cols.items()}
                rot = np.asarray(cols["rotation_deg"], dtype=np.float64)
                packed[phase]["rotation_gt10"] = int(np.sum(rot > 10.0)) if rot.size else 0
            out[str(step)][kind] = packed
    return out


def _dataset(name, cfg):
    import hydra
    import zarr

    base._point_dataset(cfg)
    cfg.task.dataset.exclusion_stats_path = str(OUT / f"{name}_exclusion.json")
    target = str(cfg.task.dataset._target_)
    if "field_v3" in target:
        sys.path.insert(0, str(rgb.V3R_SCRIPTS))
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    val = dataset.get_validation_dataset()
    root = zarr.open_group(str(cfg.task.dataset.dataset_path), mode="r")["data"]
    stride = int(cfg.task.shape_meta["sample"]["action"]["sparse"]["down_sample_steps"])
    horizon = int(cfg.task.shape_meta["sample"]["action"]["sparse"]["horizon"])
    keys = {}
    for idx, (ep, rgb_id) in enumerate(val.sampler.indices):
        ep = int(ep)
        if ep in base.EPISODES:
            keys[(ep, int(rgb_id))] = idx
    return {
        "val": val,
        "root": root,
        "keys": keys,
        "stride": stride,
        "horizon": horizon,
        "v3r": "field_v3" in target,
        "path": str(cfg.task.dataset.dataset_path),
    }


def _prepare(group):
    robot_t = np.asarray(group["robot_time_stamps_0"]).reshape(-1)
    robot_t = robot_t - robot_t[0]
    pose = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    rotation, position = base.pose7_to_Rp(pose)
    frames = np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
    feet = np.asarray(group["mold_x_s_0"], dtype=np.float64)
    wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
    key = "wrench_time_stamps_left_0" if "wrench_time_stamps_left_0" in group else "wrench_time_stamps_0"
    wt = np.asarray(group[key]).reshape(-1)
    wt = wt - wt[0]
    j = np.clip(np.searchsorted(wt, robot_t), 0, len(wt) - 1)
    force = np.linalg.norm(wrench[j, :3], axis=1)
    clearance = np.einsum("ij,ij->i", position - feet, -frames[:, :, 2])
    rgb_t = np.asarray(group["rgb_time_stamps_0"]).reshape(-1)
    return robot_t, base._phases(force, clearance), frames, feet, pose, rotation, position, rgb_t


def _phase_table(path, keep, stride, horizon) -> dict:
    """One phase per query, from the v3r zarr. Both policies use it."""
    import zarr
    from diffusion_policy.common.sampler_umift import sparse_action_indices_from_rgb_timestamps

    root = zarr.open_group(path, mode="r")["data"]
    prepared = {}
    out = {}
    for ep, rgb_id in keep:
        if ep not in prepared:
            prepared[ep] = _prepare(root[f"episode_{ep}"])
        _robot_t, labels, _frames, _feet, _pose, _rotation, _position, rgb_t = prepared[ep]
        index, _times = sparse_action_indices_from_rgb_timestamps(
            np.asarray(root[f"episode_{ep}"]["robot_time_stamps_0"]).reshape(-1),
            rgb_t,
            rgb_id,
            horizon,
            stride,
        )
        if len(index) <= max(STEPS):
            continue
        for step in STEPS:
            out[(ep, rgb_id, step)] = labels[int(index[step - 1])]
    return out


def _score(name, ckpt, device, keep, phases) -> dict:
    import torch
    from diffusion_policy.common.sampler_umift import sparse_action_indices_from_rgb_timestamps
    from PyriteEnvSuites.scripts.run_policy_open_loop_val_export import predict_sparse_action

    policy, cfg, epoch = rgb._load(ckpt, device)
    blob = _dataset(name, cfg)
    rgb_keys = list(getattr(policy.obs_encoder, "rgb_keys", []))
    if "rgb_0" not in rgb_keys:
        raise SystemExit(f"{name} has no rgb encoder")
    buckets = {step: {"model": _cols(), "hold": _cols()} for step in STEPS}
    phased = {}
    n = 0
    for ep, rgb_id in keep:
        idx = blob["keys"].get((ep, rgb_id))
        if idx is None:
            continue
        group = blob["root"][f"episode_{ep}"]
        if ep not in phased:
            phased[ep] = _prepare(group)
        robot_t, labels, frames, feet, pose, rotation, position, rgb_t = phased[ep]
        index, times = sparse_action_indices_from_rgb_timestamps(
            np.asarray(group["robot_time_stamps_0"]).reshape(-1),
            rgb_t,
            rgb_id,
            blob["horizon"],
            blob["stride"],
        )
        index = np.asarray(index, dtype=np.int64)
        if len(index) <= max(STEPS):
            continue
        dts = np.diff(np.asarray(times, dtype=np.float64))
        sample = blob["val"][idx]
        sparse = {
            key: value.numpy()
            for key, value in sample["obs"]["sparse"].items()
            if not str(key).startswith("rgb") or key in rgb_keys
        }
        pred = np.asarray(predict_sparse_action(policy, {"sparse": sparse}, device), dtype=np.float64)
        gt = sample["action"]["sparse"].numpy()
        q = int(index[0])
        p_now = position[q]
        R_now = rotation[q]
        dn_now = float(np.dot(p_now - feet[q], -frames[q, :, 2]))
        for step in STEPS:
            row = int(index[step - 1])
            phase = phases[(ep, rgb_id, step)]
            if blob["v3r"]:
                p_gt, R_gt, dn_gt = _v3r_at(gt, feet[index], frames[index], dts, step)
                p_hat, R_hat, dn_hat = _v3r_at(pred, feet[index], frames[index], dts, step)
                model_dn = abs(dn_hat - dn_gt) * 1e3
                hold_dn = abs(dn_now - dn_gt) * 1e3
            else:
                p_gt, R_gt = _r1_at(gt, p_now, R_now, step)
                p_hat, R_hat = _r1_at(pred, p_now, R_now, step)
                outward = -frames[row, :, 2]
                model_dn = abs(float(np.dot(p_hat - p_gt, outward))) * 1e3
                hold_dn = abs(float(np.dot(p_now - p_gt, outward))) * 1e3
            buckets[step]["model"][phase]["position_mm"].append(float(np.linalg.norm(p_hat - p_gt) * 1e3))
            buckets[step]["model"][phase]["rotation_deg"].append(base._angle(R_hat, R_gt))
            buckets[step]["model"][phase]["delta_n_mm"].append(model_dn)
            buckets[step]["hold"][phase]["position_mm"].append(float(np.linalg.norm(p_now - p_gt) * 1e3))
            buckets[step]["hold"][phase]["rotation_deg"].append(base._angle(R_now, R_gt))
            buckets[step]["hold"][phase]["delta_n_mm"].append(hold_dn)
        n += 1
        if n % 40 == 0:
            print(f"[open] {name} {n}/{len(keep)}", flush=True)
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"ckpt": str(ckpt), "epoch": epoch, "n": n, "horizons": _finish(buckets)}


def _cfg(ckpt):
    import dill
    import torch

    payload = torch.load(open(ckpt, "rb"), map_location="cpu", pickle_module=dill)
    epoch = int(dill.loads(payload["pickles"]["epoch"]))
    cfg = payload["cfg"]
    del payload
    return cfg, epoch


def _keys(name, ckpt) -> dict:
    cfg, epoch = _cfg(ckpt)
    blob = _dataset(name, cfg)
    print(f"[keys] {name} epoch {epoch} kept {len(blob['keys'])} path {blob['path']}", flush=True)
    return {
        "epoch": epoch,
        "keys": set(blob["keys"]),
        "path": blob["path"],
        "stride": blob["stride"],
        "horizon": blob["horizon"],
    }


def main() -> None:
    import torch

    base._check()
    _check()
    OUT.mkdir(parents=True, exist_ok=True)
    listed = {name: _keys(name, ckpt) for name, ckpt in rgb.CKPT.items()}
    keep = sorted(listed["v3r"]["keys"] & listed["r1rel"]["keys"])
    print(
        f"[keys] v3r {len(listed['v3r']['keys'])} r1rel {len(listed['r1rel']['keys'])} "
        f"intersection {len(keep)}",
        flush=True,
    )
    import zarr

    for ep in base.EPISODES:
        a = np.asarray(zarr.open_group(listed["v3r"]["path"], mode="r")["data"][f"episode_{ep}"]["rgb_time_stamps_0"])
        b = np.asarray(zarr.open_group(listed["r1rel"]["path"], mode="r")["data"][f"episode_{ep}"]["rgb_time_stamps_0"])
        if a.shape != b.shape or float(np.max(np.abs(a - b))) > 1e-3:
            raise SystemExit(f"episode {ep} rgb clocks differ")
    phases = _phase_table(
        listed["v3r"]["path"], keep, listed["v3r"]["stride"], listed["v3r"]["horizon"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opened = {
        "n_v3r": len(listed["v3r"]["keys"]),
        "n_r1rel": len(listed["r1rel"]["keys"]),
        "n_intersection": len(keep),
        "n_scored": len({(ep, rgb_id) for ep, rgb_id, _step in phases}),
    }
    for name, ckpt in rgb.CKPT.items():
        opened[name] = _score(name, ckpt, device, keep, phases)
        (OUT / "open_loop.json").write_text(json.dumps(opened, indent=2) + "\n")
    print(json.dumps(opened, indent=2), flush=True)


if __name__ == "__main__":
    main()
