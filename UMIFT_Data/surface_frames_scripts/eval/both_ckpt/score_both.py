#!/usr/bin/env python3
"""Fair rollout at best-val and final checkpoints. No retraining.

Same validation episodes and the same open-loop decode as compare_r1rel.
Heading and robust walk the foot the same way; their rotation is the tilt pack.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [
    str(SCRIPTS / "eval"),
    str(SCRIPTS),
    str(ROOT / "PyriteEnvSuites/scripts"),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT),
]

import torch
import zarr
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

import compare_r1rel as fair
from compare_thin import heading_rotation
from diffusion_policy.common.drop_depth_inputs import apply_drop_depth_inputs
from diffusion_policy.common.drop_scan_inputs import apply_drop_scan_inputs
from run_policy_open_loop_val_export import load_policy_and_cfg, predict_sparse_action

OUT = Path(__file__).resolve().parent
DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
TRAIN = Path("/mnt/data/umidata/ft/training_outputs")
VAL = {9, 12, 28, 37, 41}
STEPS = (1, 6, 16, 64)
UP = np.array([0.0, -1.0, 0.0])
REGIONS = ("crown", "fillet", "wall", "flange")

CAUSAL_DIR = (
    TRAIN / "oct_v35_h4s_r4vt_causal_foot"
    / "oct-v35-h4s-r4vt-causal-foot_e200_bs64_valtopk_43390127/checkpoints"
)
HEAD_DIR = (
    TRAIN / "oct-v35-h4s-r4vt-heading-tilt_sc_e200_bs64_valtopk_43448962/checkpoints"
)


def _check() -> None:
    class _Flat:
        normals = np.array([[0.0, 0.0, 1.0]])
        frames = np.eye(3)[None]

        def closest(self, pts):
            p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
            ones = np.ones((len(p), 3))
            return p, np.zeros((len(p), 3), dtype=np.int64), ones

    action = np.zeros((3, 10))
    action[:, 2] = 0.03
    pos = fair.rollout_tool(
        _Flat(), action, np.zeros(3), np.eye(3), np.array([0.0, 0.0, 0.02]), np.full(3, 0.1)
    )
    err = pos[0] - np.array([0.0, 0.0, 0.02])
    if np.linalg.norm(err - np.array([0.0, 0.0, 0.01])) > 1e-6:
        raise RuntimeError("step-1 error is not the δn miss along the frame normal")


def build_full(cfg, zarr_path, norm, sha_prefix, action_dim):
    ds_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    apply_drop_scan_inputs(ds_cfg)
    apply_drop_depth_inputs(ds_cfg)
    OmegaConf.update(ds_cfg, "task.dataset_path", zarr_path, merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.dataset_path", zarr_path, merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.action_normalizer_path", str(norm), merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.long_dt_mask_path", fair.LONG_DT, merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.exclusion_stats_path", None, merge=False)
    sha = str(ds_cfg.task.dataset.action_normalizer_sha256)
    dim = int(ds_cfg.task.shape_meta.action.shape[0])
    if not sha.startswith(sha_prefix) or dim != action_dim:
        raise SystemExit(f"sha {sha} dim {dim}, expected {sha_prefix} dim {action_dim}")
    dataset = fair.hydra_instantiate(ds_cfg.task.dataset)
    return dataset.get_validation_dataset()


def region_of(normal, foot, apron):
    n = normal / np.clip(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12, None)
    tilt = np.degrees(np.arccos(np.clip(n @ UP, -1.0, 1.0)))
    name = np.full(len(n), "wall", dtype=object)
    name[tilt < 25.0] = "crown"
    name[(tilt >= 25.0) & (tilt < 55.0)] = "fillet"
    name[apron.query(foot, workers=-1)[0] < 0.015] = "flange"
    return name


def raw_normal(field, xyz):
    _hit, vid, w = field.closest(np.asarray(xyz, dtype=np.float64).reshape(1, 3))
    raw = np.asarray(w[0] @ field.normals[vid[0]], dtype=np.float64)
    foot = np.asarray(_hit, dtype=np.float64).reshape(3)
    return raw, foot


def episode_cache(episode):
    group = zarr.open(fair.CAUSAL_ZARR, "r")["data"][f"episode_{episode}"]
    pos, rot, robot_t, rgb_t = fair.episode_arrays(fair.R1REL_ZARR, episode)
    return {
        "pos": pos,
        "rot": rot,
        "robot_t": robot_t,
        "rgb_t": rgb_t,
        "foot": np.asarray(group["mold_x_s_0"], dtype=np.float64).reshape(-1, 3),
        "frame": np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3),
        "normal": np.asarray(group["mold_normal_0"], dtype=np.float64).reshape(-1, 3),
    }


def prepare(causal_ds, r1rel_ds):
    queries = fair.common_queries(causal_ds, r1rel_ds, VAL)
    cache = {}
    rows = []
    for episode, rgb, c_idx, r_idx in queries:
        if episode not in cache:
            cache[episode] = episode_cache(episode)
        ep = cache[episode]
        horizon = fair.HORIZON
        index = fair.target_index(ep["robot_t"], ep["rgb_t"], rgb, 1, horizon)
        if len(index) < horizon:
            print(f"[both] skip ep {episode} rgb {rgb}: targets {len(index)}", flush=True)
            continue
        anchor = int(np.clip(np.searchsorted(ep["robot_t"], ep["rgb_t"][rgb]), 0, len(ep["pos"]) - 1))
        times = ep["robot_t"][index]
        dts = np.diff(times)
        dts = np.append(dts, dts[-1] if len(dts) else fair.DT)
        rows.append({
            "episode": episode,
            "rgb": rgb,
            "c_idx": c_idx,
            "r_idx": r_idx,
            "index": index,
            "anchor_p": ep["pos"][anchor],
            "anchor_R": ep["rot"][anchor],
            "demo": ep["pos"][index],
            "demo_R": ep["rot"][index],
            "dts": dts,
            "foot0": ep["foot"][index[0]],
            "frame0": ep["frame"][index[0]],
            "frames": ep["frame"][index],
            "normals": ep["normal"][index],
        })
    if len(rows) < 10:
        raise SystemExit(f"only {len(rows)} queries")
    return rows


def decode(kind, field, axis, action, row):
    action = np.asarray(action, dtype=np.float64)
    if kind == "r1rel":
        return fair.r1rel_pose(action, row["anchor_p"], row["anchor_R"])
    if kind == "heading":
        pos = fair.rollout_tool(
            field, action, row["foot0"], row["frame0"], row["anchor_p"], row["dts"]
        )
        rot = heading_rotation(action, row["normals"], axis)
        return pos, rot
    feet = np.repeat(row["foot0"].reshape(1, 3), len(action), axis=0)
    return fair.causal_pose(
        field, action, row["anchor_p"], feet, row["frames"], row["dts"]
    )


def score_policy(name, policy, dataset, kind, field, axis, rows, device, qmap):
    out = []
    missed = 0
    for i, row in enumerate(rows, start=1):
        key = (row["episode"], row["rgb"])
        if kind == "r1rel":
            idx = row["r_idx"]
        elif kind == "causal":
            idx = row["c_idx"]
        else:
            idx = qmap.get(key)
            if idx is None:
                missed += 1
                continue
        obs, sample = dataset.sampler.sample_sequence(idx)
        pred = np.asarray(predict_sparse_action(policy, obs, device), dtype=np.float64)
        pred = pred.reshape(-1, pred.shape[-1])[: fair.HORIZON]
        gt = np.asarray(sample["sparse"], dtype=np.float64)[: fair.HORIZON]
        if len(pred) < fair.HORIZON or len(gt) < fair.HORIZON:
            missed += 1
            continue
        xyz, rot = decode(kind, field, axis, pred, row)
        met = fair.metrics(xyz, rot, row["demo"], row["demo_R"], field, axis)
        item = {
            "episode": row["episode"],
            "pos_mm": met["pos_mm"],
            "rot_deg": met["rot_deg"],
            "err1": (xyz[0] - row["demo"][0]) * 1e3,
            "err_anchor": (xyz[0] - row["anchor_p"]) * 1e3,
            "lift": row["frame0"][:, 2].copy(),
            "tool_z": row["demo_R"][0, :, 2].copy(),
        }
        if kind == "heading" and pred.shape[1] >= 8:
            item["ab"] = np.degrees(pred[0, 6:8])
            item["ab_l"] = np.degrees(gt[0, 6:8])
        out.append(item)
        if i % 10 == 0:
            print(f"[both] {name} {i}/{len(rows)}", flush=True)
    print(f"[both] {name} scored {len(out)} missed {missed}", flush=True)
    return out


def label_rows(dataset, field, axis, rows):
    out = []
    for row in rows:
        _obs, sample = dataset.sampler.sample_sequence(row["c_idx"])
        gt = np.asarray(sample["sparse"], dtype=np.float64)[: fair.HORIZON]
        xyz, rot = decode("causal", field, axis, gt, row)
        met = fair.metrics(xyz, rot, row["demo"], row["demo_R"], field, axis)
        out.append({
            "episode": row["episode"],
            "pos_mm": met["pos_mm"],
            "rot_deg": met["rot_deg"],
            "err1": (xyz[0] - row["demo"][0]) * 1e3,
            "err_anchor": (xyz[0] - row["anchor_p"]) * 1e3,
            "lift": row["frame0"][:, 2].copy(),
            "tool_z": row["demo_R"][0, :, 2].copy(),
            "dn": float(gt[0, 2]),
            "implied": float(np.dot(row["anchor_p"] - row["foot0"], row["frame0"][:, 2])),
        })
    return out


def mean_of_episodes(items, key, step):
    by_ep = {}
    for item in items:
        by_ep.setdefault(item["episode"], []).append(float(item[key][step - 1]))
    if not by_ep:
        return None
    return float(np.mean([np.mean(v) for v in by_ep.values()]))


def step1_split(items, regions):
    bag = {name: [] for name in REGIONS}
    for item, region in zip(items, regions):
        bag[region].append(item)
    report = {}
    for name in REGIONS:
        group = bag[name]
        if not group:
            report[name] = {"n": 0}
            continue
        err = np.stack([g["err1"] for g in group])
        lift = np.stack([g["lift"] for g in group])
        tool = np.stack([g["tool_z"] for g in group])
        along = np.sum(err * lift, axis=1)
        tang = err - along[:, None] * lift
        anchor = np.stack([g["err_anchor"] for g in group])
        report[name] = {
            "n": int(len(group)),
            "pos_mm": round(float(np.mean(np.linalg.norm(err, axis=1))), 2),
            "mean_xyz_mm": [round(float(v), 2) for v in err.mean(0)],
            "std_xyz_mm": [round(float(v), 2) for v in err.std(0)],
            "along_frame_z_mm": round(float(along.mean()), 2),
            "along_frame_z_std_mm": round(float(along.std()), 2),
            "tangential_mm": round(float(np.mean(np.linalg.norm(tang, axis=1))), 2),
            "along_tool_z_mm": round(float(np.mean(np.sum(err * tool, axis=1))), 2),
            "from_query_tcp_mm": round(float(np.mean(np.linalg.norm(anchor, axis=1))), 2),
            "from_query_along_z_mm": round(float(np.mean(np.sum(anchor * lift, axis=1))), 2),
        }
    return report


def main() -> None:
    _check()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[both] device", device, flush=True)
    field = fair.FixBMoldField(fair.FIELD_PATH)
    axis = fair.long_axis(field.vertices)
    blob = np.load(fair.FIELD_PATH)
    apron = cKDTree(field.vertices[np.asarray(blob["is_apron"]).astype(bool)])
    families = [
        {
            "kind": "causal",
            "zarr": fair.CAUSAL_ZARR,
            "norm": fair.CAUSAL_NORM,
            "sha": "14308b72",
            "dim": 10,
            "ckpts": [
                ("causal e25", CAUSAL_DIR / "epoch=0025-val_loss=0.058.ckpt"),
                ("causal e199", CAUSAL_DIR / "latest.ckpt"),
            ],
        },
        {
            "kind": "r1rel",
            "zarr": fair.R1REL_ZARR,
            "norm": fair.R1REL_NORM,
            "sha": "2c20fba56851",
            "dim": 21,
            "ckpts": [
                ("r1rel e25", fair.R1REL_CKPT),
                ("r1rel last", TRAIN / "oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt"),
            ],
        },
        {
            "kind": "heading",
            "zarr": str(DATA / "r4vt_heading_tilt/acp_replay_buffer.zarr"),
            "norm": SCRIPTS / "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_heading_tilt_swing_sc.pkl",
            "sha": "e93b8756",
            "dim": 8,
            "ckpts": [
                ("heading e25", HEAD_DIR / "epoch=0025-val_loss=0.059.ckpt"),
                ("heading latest", HEAD_DIR / "latest.ckpt"),
            ],
        },
        {
            "kind": "heading",
            "zarr": str(DATA / "r4vt_heading_tilt/acp_replay_buffer.zarr"),
            "norm": SCRIPTS / "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_heading_tilt_swing_sc_robust_ab.pkl",
            "sha": "3f9b16b1",
            "dim": 8,
            "ckpts": [
                (
                    "robust e25",
                    TRAIN / "oct_v35_h4s_r4vt_heading_tilt_sc_robustab_mem128/epoch=0025-val_loss=0.073.ckpt",
                ),
            ],
        },
    ]
    on_disk = sorted(p.name for p in CAUSAL_DIR.glob("*.ckpt"))
    datasets = {}
    policies_cfg = {}
    for spec in families:
        key = (spec["zarr"], spec["sha"])
        if key in datasets:
            continue
        # Config comes from the first checkpoint. Same run, same normalizer.
        policy, cfg = load_policy_and_cfg(str(spec["ckpts"][0][1]), device)
        datasets[key] = build_full(cfg, spec["zarr"], spec["norm"], spec["sha"], spec["dim"])
        policies_cfg[key] = cfg
        del policy
        torch.cuda.empty_cache()
        print(f"[both] dataset {spec['sha']}", flush=True)
    causal_ds = datasets[(fair.CAUSAL_ZARR, "14308b72")]
    r1rel_ds = datasets[(fair.R1REL_ZARR, "2c20fba56851")]
    rows = prepare(causal_ds, r1rel_ds)
    print(f"[both] queries {len(rows)} episodes {sorted({r['episode'] for r in rows})}", flush=True)
    regions = []
    for row in rows:
        raw, foot = raw_normal(field, row["demo"][0])
        regions.append(str(region_of(raw[None], foot[None], apron)[0]))
    scored = {"label": label_rows(causal_ds, field, axis, rows)}
    notes = {}
    for spec in families:
        key = (spec["zarr"], spec["sha"])
        dataset = datasets[key]
        qmap = fair.query_map(dataset)
        for label, path in spec["ckpts"]:
            epoch = fair.ckpt_epoch(path)
            notes[label] = {"path": str(path), "epoch": epoch, "bytes": path.stat().st_size}
            if path.stat().st_size < 2_000_000_000:
                notes[label]["skipped"] = "file is truncated"
                print(f"[both] skip truncated {path.name}", flush=True)
                continue
            policy, _cfg = load_policy_and_cfg(str(path), device)
            scored[label] = score_policy(
                label, policy, dataset, spec["kind"], field, axis, rows, device, qmap
            )
            del policy
            torch.cuda.empty_cache()
    table = {}
    for label, items in scored.items():
        table[label] = {
            "n": len(items),
            "pos_mm": {str(s): mean_of_episodes(items, "pos_mm", s) for s in STEPS},
            "rot_deg": {str(s): mean_of_episodes(items, "rot_deg", s) for s in STEPS},
        }
    step1 = {}
    for label in ("causal e199", "causal e25", "label", "r1rel e25"):
        items = scored.get(label)
        if not items or len(items) != len(regions):
            step1[label] = {"n": 0 if not items else len(items), "note": "query count does not match"}
            continue
        step1[label] = step1_split(items, regions)
    if scored.get("label") and len(scored["label"]) == len(regions):
        wall = [g for g, region in zip(scored["label"], regions) if region == "wall"]
        if wall:
            step1["label_wall_dn_mm"] = {
                "n": len(wall),
                "pred_dn_mm": round(float(np.mean([g["dn"] for g in wall]) * 1e3), 2),
                "tcp_along_frame_z_mm": round(float(np.mean([g["implied"] for g in wall]) * 1e3), 2),
            }
    report = {
        "episodes": sorted(VAL),
        "n_queries": len(rows),
        "definition": (
            "Mean of per-episode means. Position is the open-loop tool point. "
            "Causal and heading integrate v and δn from the query TCP. "
            "r1rel is action21_postprocess. Step-1 vector is pred minus demo, mm."
        ),
        "causal_checkpoints_on_disk": on_disk,
        "robust_latest": "no later file than epoch=0025-val_loss=0.073.ckpt",
        "notes": notes,
        "table": table,
        "step1": step1,
        "region_counts": {name: int(regions.count(name)) for name in REGIONS},
    }
    (OUT / "both.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"table": table, "step1": step1, "region_counts": report["region_counts"]}, indent=2))


if __name__ == "__main__":
    main()
