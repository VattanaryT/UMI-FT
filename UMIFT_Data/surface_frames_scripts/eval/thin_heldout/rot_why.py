#!/usr/bin/env python3
"""Why causal/k3000 step-1 rotation is ~160° on thin kept wall queries.

Does not write a score. Prints three blocks: training-val through the thin
scorer, the raw predicted rotation versus the frame it is composed with, and
the observations those models actually receive.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "viewer"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import contact_wall as cw  # noqa: E402
import foot_overlay  # noqa: E402
import heldout  # noqa: E402
from diffusion_policy.common.drop_depth_inputs import apply_drop_depth_inputs  # noqa: E402
from diffusion_policy.common.drop_scan_inputs import apply_drop_scan_inputs  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from heading_tilt import geodesic_deg, long_axis, pack_tilt  # noqa: E402
from mold_aware_runtime import rot6d_rows_to_R  # noqa: E402
from run_policy_open_loop_val_export import load_policy_and_cfg, predict_sparse_action  # noqa: E402

import compare_r1rel as fair  # noqa: E402

TRAIN_ZARR = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_causal_foot/acp_replay_buffer.zarr"
)
TRAIN_FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
LONG_DT = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/long_dt_gt_100ms.npz"
)
CAUSAL = cw.MODELS[0][2]
K3000 = cw.TRAIN / (
    "oct-v35-h4s-r4vt-causal-foot-carry-k3000_e200_bs64_valtopk_43472139/checkpoints/latest.ckpt"
)
LOW = (
    "robot0_eef_pos",
    "robot0_eef_rot_axis_angle",
    "robot0_eef_wrench_left",
    "robot0_eef_wrench_right",
    "robot0_mold_geom",
)
MOLD_NAMES = ("sd", "nx", "ny", "nz", "r0", "r1", "r2", "r3", "r4", "r5")


def _check() -> None:
    eye = rot6d_rows_to_R(np.array([[1, 0, 0, 0, 1, 0]], dtype=np.float64))
    if float(geodesic_deg(eye, np.eye(3)[None])[0]) > 1e-6:
        raise RuntimeError("rot6d identity")


def _dataset(cfg, zarr_path, norm, long_dt):
    ds_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    apply_drop_scan_inputs(ds_cfg)
    apply_drop_depth_inputs(ds_cfg)
    OmegaConf.update(ds_cfg, "task.dataset_path", str(zarr_path), merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.dataset_path", str(zarr_path), merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.action_normalizer_path", str(norm), merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.long_dt_mask_path", str(long_dt), merge=False)
    for key in ("exclusion_stats_path", "filter_stats_path", "pose_jump_flag_path"):
        if key in ds_cfg.task.dataset:
            OmegaConf.update(ds_cfg, f"task.dataset.{key}", None, merge=False)
    dataset = fair.hydra_instantiate(ds_cfg.task.dataset)
    if len(dataset.sampler.indices) == 0:
        dataset = dataset.get_validation_dataset()
    return dataset


def _keep(demo, rows):
    i0 = np.array([int(index[0]) for _idx, _rgb, index in rows], dtype=np.int64)
    _foot, face, _z = foot_overlay._project(*foot_overlay._load(), demo["tool_p"][i0])
    kept = []
    for row, kind in zip(rows, face):
        _idx, _rgb, index = row
        j = heldout._nearest(demo["wrench_t"], demo["robot_t"][index])
        mag = np.linalg.norm(demo["smooth"][j], axis=1)
        ok = (
            str(kind) not in ("cap", "back")
            and float(mag[0]) >= 8.0
            and float(np.mean(mag >= 5.0)) >= 0.80
        )
        if ok:
            kept.append(row)
    return kept


def _demo_ep(path, episode):
    ep = zarr.open(path, "r")["data"][f"episode_{episode}"]
    demo = {
        "robot_t": np.asarray(ep["robot_time_stamps_0"]).reshape(-1),
        "rgb_t": np.asarray(ep["rgb_time_stamps_0"]).reshape(-1),
        "tool_p": None,
        "tool_R": None,
        "foot": np.asarray(ep["mold_x_s_0"], dtype=np.float64).reshape(-1, 3),
        "frame": np.asarray(ep["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3),
        "normal": np.asarray(ep["mold_normal_0"], dtype=np.float64).reshape(-1, 3),
        "dR": np.asarray(ep["fixb_delta_R_rot6d_0"], dtype=np.float64).reshape(-1, 6),
        "cutoff": None,
    }
    from fixb_labels import pose7_to_Rp
    from PyriteUtility.data_pipeline.wrench_baseline_utils import time_based_moving_average

    demo["tool_R"], demo["tool_p"] = pose7_to_Rp(np.asarray(ep["ts_pose_fb_0"], dtype=np.float64))
    demo["wrench_t"] = np.asarray(ep["wrench_time_stamps_0"]).reshape(-1)
    wrench = np.asarray(ep["wrench_left_0"], dtype=np.float64)
    demo["smooth"] = time_based_moving_average(wrench, demo["wrench_t"], 1.0)[:, :3]
    demo["force"] = np.linalg.norm(wrench[:, :3], axis=1)
    return demo


def _angle_axis(R):
    rotvec = Rotation.from_matrix(np.asarray(R, dtype=np.float64).reshape(3, 3)).as_rotvec()
    angle = float(np.degrees(np.linalg.norm(rotvec)))
    axis = rotvec / max(float(np.linalg.norm(rotvec)), 1e-12)
    return angle, axis


def _band(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return None
    return {
        "n": int(len(values)),
        "median": round(float(np.median(values)), 2),
        "p90": round(float(np.percentile(values, 90)), 2),
        "max": round(float(np.max(values)), 2),
    }


def _vec(x):
    x = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _range(policy, key):
    stats = policy.sparse_normalizer[key].get_input_stats()
    return _vec(stats["min"]), _vec(stats["max"])


def _now(obs):
    return {key: np.asarray(obs["sparse"][key][-1], dtype=np.float64) for key in LOW if key in obs["sparse"]}


def _outside(row, lo, hi):
    return (row < lo - 1e-5) | (row > hi + 1e-5)


def main() -> None:
    _check()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jobs = {}
    for name, path in (("causal e199", CAUSAL), ("k3000 e100", K3000)):
        policy, cfg = load_policy_and_cfg(str(path), device)
        norm, _dim, decode, _sha = heldout._norm_of(cfg, "auto")
        if decode != "causal":
            raise SystemExit(f"{name} decodes as {decode}")
        jobs[name] = {"policy": policy, "cfg": cfg, "norm": norm, "epoch": fair.ckpt_epoch(path)}
    causal = jobs["causal e199"]
    print("[rot] loading training zarr", flush=True)
    train_ds = _dataset(causal["cfg"], TRAIN_ZARR, causal["norm"], LONG_DT).get_validation_dataset()
    val_idx = list(range(len(train_ds.sampler.indices)))
    step = max(1, len(val_idx) // 50)
    picked = val_idx[::step][:50]
    by_ep = {}
    for idx in picked:
        epi, rgb = train_ds.sampler.indices[idx]
        by_ep.setdefault(int(epi), []).append((int(idx), int(rgb)))
    field = FixBMoldField(str(TRAIN_FIELD))
    axis = long_axis(field.vertices)
    train_rot = {name: [] for name in jobs}
    train_label = []
    train_frame_tool = []
    train_obs = []
    for epi, pairs in by_ep.items():
        demo = _demo_ep(TRAIN_ZARR, epi)
        rows = []
        for idx, rgb in pairs:
            index, _times = heldout.sparse_action_indices_from_rgb_timestamps(
                demo["robot_t"], demo["rgb_t"], rgb, heldout.H, 1
            )
            index = np.asarray(index, dtype=np.int64)
            if len(index) < heldout.H or index[-1] >= len(demo["robot_t"]):
                continue
            rows.append((idx, rgb, index))
        if not rows:
            continue
        captured = {}

        def _grab(pred, obs, rgb, index, _box=captured):
            _box[int(rgb)] = (_now(obs), pred[0, 4:10].copy())

        for name, job in jobs.items():
            scored = heldout._score_one(
                "causal", job["policy"], train_ds, field, demo, rows, device, axis, on_query=_grab
            )
            train_rot[name].extend(scored["_rot"][:, 0].tolist())
        for _idx, rgb, index in rows:
            i0 = int(index[0])
            label_R = demo["frame"][i0] @ rot6d_rows_to_R(demo["dR"][i0 : i0 + 1])[0]
            train_label.append(float(geodesic_deg(label_R[None], demo["tool_R"][i0][None])[0]))
            train_frame_tool.append(float(geodesic_deg(demo["frame"][i0][None], demo["tool_R"][i0][None])[0]))
            if rgb in captured:
                train_obs.append(captured[rgb][0])
    print("[rot] training done", flush=True)

    thin_field = FixBMoldField(str(heldout.WALL))
    thin_axis = long_axis(thin_field.vertices)
    recordings, _axis = cw._recordings(thin_field)
    thin_out = {}
    thin_obs = []
    for letter, rec in recordings.items():
        dataset, rows = cw._rows_for(causal["cfg"], causal["norm"], rec, {})
        kept = _keep(rec["demo"], rows)
        per_model = {}
        demo = rec["demo"]
        ep = zarr.open(rec["path"], "r")["data"]["episode_0"]
        label_dR = np.asarray(ep["fixb_delta_R_rot6d_0"], dtype=np.float64).reshape(-1, 6)
        for name, job in jobs.items():
            grabbed = {}

            def _grab(pred, obs, rgb, index, _box=grabbed):
                _box[int(rgb)] = (_now(obs), np.asarray(pred[0, 4:10], dtype=np.float64).copy())

            scored = heldout._score_one(
                "causal", job["policy"], dataset, thin_field, demo, kept, device, thin_axis, on_query=_grab
            )
            queries = []
            for row_i, (_idx, rgb, index) in enumerate(kept):
                i0 = int(index[0])
                obs_now, raw = grabbed[int(rgb)]
                if name == "causal e199":
                    thin_obs.append((letter, obs_now))
                pred_R = rot6d_rows_to_R(raw.reshape(1, 6))[0]
                frame = demo["frame"][i0]
                angle, world_axis = _angle_axis(pred_R)
                label_R = rot6d_rows_to_R(label_dR[i0 : i0 + 1])[0]
                label_angle, _label_axis = _angle_axis(label_R)
                local = frame.T @ world_axis
                psi = pack_tilt(demo["tool_R"][i0][None], -frame[:, 2][None], heldout._e_long(-frame[:, 2][None], thin_axis))
                queries.append({
                    "t": round(float(demo["robot_t"][i0] - demo["rgb_t"][0]), 3),
                    "pred_deg": round(angle, 2),
                    "pred_axis_in_frame": [round(float(v), 3) for v in local],
                    "label_deg": round(label_angle, 2),
                    "frame_vs_tool_deg": round(float(geodesic_deg(frame[None], demo["tool_R"][i0][None])[0]), 2),
                    "composed_deg": round(float(scored["_rot"][row_i, 0]), 2),
                    "psi_deg": round(float(np.degrees(np.arctan2(psi[0, 0], psi[0, 1]))), 2),
                })
            per_model[name] = {
                "step1": _band([q["composed_deg"] for q in queries]),
                "pred_deg": _band([q["pred_deg"] for q in queries]),
                "label_deg": _band([q["label_deg"] for q in queries]),
                "frame_vs_tool_deg": _band([q["frame_vs_tool_deg"] for q in queries]),
                "pred_under_15": int(sum(q["pred_deg"] < 15 for q in queries)),
                "pred_over_150": int(sum(q["pred_deg"] > 150 for q in queries)),
                "axis_normal": round(float(np.median([abs(q["pred_axis_in_frame"][2]) for q in queries])), 3),
                "queries": queries,
            }
        thin_out[letter] = {"n": len(kept), "models": per_model}

    lohi = {key: _range(causal["policy"], key) for key in LOW}
    flags = []
    for letter, obs_now in thin_obs:
        for key, row in obs_now.items():
            lo, hi = lohi[key]
            bad = _outside(row, lo, hi)
            if np.any(bad):
                names = MOLD_NAMES if key == "robot0_mold_geom" else [str(i) for i in range(len(row))]
                flags.append({
                    "rec": letter,
                    "key": key,
                    "channels": [names[i] for i in np.flatnonzero(bad)],
                    "value": [round(float(v), 4) for v in row[bad]],
                    "min": [round(float(v), 4) for v in lo[bad]],
                    "max": [round(float(v), 4) for v in hi[bad]],
                })

    def _col(rows, key, sl):
        return np.stack([row[key][sl] for row in rows])

    geom = {
        "train_pos_m": [round(float(v), 4) for v in np.median(_col(train_obs, "robot0_eef_pos", slice(None)), axis=0)],
        "thin_pos_m": [round(float(v), 4) for v in np.median(_col([o for _l, o in thin_obs], "robot0_eef_pos", slice(None)), axis=0)],
        "train_sd_mm": round(float(np.median(_col(train_obs, "robot0_mold_geom", 0)) * 1e3), 2),
        "thin_sd_mm": round(float(np.median(_col([o for _l, o in thin_obs], "robot0_mold_geom", 0)) * 1e3), 2),
        "train_normal": [round(float(v), 3) for v in np.median(_col(train_obs, "robot0_mold_geom", slice(1, 4)), axis=0)],
        "thin_normal": [round(float(v), 3) for v in np.median(_col([o for _l, o in thin_obs], "robot0_mold_geom", slice(1, 4)), axis=0)],
        "train_wrench_N": [round(float(v), 2) for v in np.median(_col(train_obs, "robot0_eef_wrench_left", slice(0, 3)), axis=0)],
        "thin_wrench_N": [round(float(v), 2) for v in np.median(_col([o for _l, o in thin_obs], "robot0_eef_wrench_left", slice(0, 3)), axis=0)],
    }
    report = {
        "epochs": {name: job["epoch"] for name, job in jobs.items()},
        "train_val": {
            "n": len(train_label),
            "picked_from": len(val_idx),
            "label_composed_deg": _band(train_label),
            "frame_vs_tool_deg": _band(train_frame_tool),
            "models": {name: _band(vals) for name, vals in train_rot.items()},
        },
        "thin": thin_out,
        "obs_out_of_range": flags,
        "obs_median": geom,
        "pose_frames": OmegaConf.to_container(causal["cfg"].task.shape_meta.pose_frames, resolve=True),
        "obs_keys": list(causal["cfg"].task.shape_meta.obs.keys()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
