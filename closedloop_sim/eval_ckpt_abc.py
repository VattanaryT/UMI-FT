#!/usr/bin/env python3
"""Open-loop score for field_v2n, norgb, v3r, and the previous r4vt / r1rel checkpoints.

New files only. Reads zarrs and checkpoints. Does not write them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-field_v3r")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
V2N = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-field_v2n/UMIFT_Data/surface_frames_scripts")
for entry in (V2N, ROOT, ROOT / "PyriteUtility", ROOT / "PyriteConfig", ROOT / "PyriteML", SCRIPTS):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)

from closedloop_sim.segments import label_segments  # noqa: E402
from diffusion_policy.common.sampler_umift import get_val_mask  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import pose7_to_Rp  # noqa: E402
from mold_aware_runtime import rot6d_rows_to_R  # noqa: E402
from r4vt_horizon import decode_horizon  # noqa: E402

OUT = Path(__file__).resolve().parent / "out_ckpt_abc"
LOCAL_DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
V2N_ZARR = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_field_v2n/acp_replay_buffer.zarr"
)
TRAIN = Path("/mnt/data/umidata/ft/training_outputs")
MODELS = {
    "field_v2_e140": TRAIN / "oct_v35_h4s_r4vt_field_v2_k3000/checkpoints/latest_1005.ckpt",
    "r1rel_e195": TRAIN / "oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt",
    "A_v2n": TRAIN / "oct_v35_h4s_r4vt_field_v2n/latest.ckpt",
    "B_v2n_norgb": TRAIN / "oct_v35_h4s_r4vt_field_v2n_norgb/latest.ckpt",
    "C_v3r": TRAIN / "oct_v35_h4s_r4vt_field_v3r/latest_e160ish.ckpt",
}
K_NPM = 3000.0
STRIDE_S = 0.5
LIFT_S = 0.5
JUMP_DEG = 10.0
PHASES = ("approach", "contact", "lift-off")


def _stat(values) -> dict:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "median": None, "p90": None}
    return {
        "n": int(values.size),
        "median": round(float(np.median(values)), 3),
        "p90": round(float(np.percentile(values, 90)), 3),
    }


def _angle(a, b) -> float:
    c = float(np.clip((np.trace(a.T @ b) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _nearest(times, values, t):
    i = int(np.clip(np.searchsorted(times, t), 0, len(times) - 1))
    j = max(i - 1, 0)
    if abs(times[j] - t) < abs(times[i] - t):
        i = j
    return values[i], i


def _episode_cache(zarr_path: Path, episodes: list[int], field) -> dict:
    import zarr

    root = zarr.open_group(str(zarr_path), mode="r")
    gt = zarr.open_group(str(V2N_ZARR), mode="r")
    out = {}
    for ep in episodes:
        group = root["data"][f"episode_{ep}"]
        robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
        R, p = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
        wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
        wt = np.asarray(group["wrench_time_stamps_0"], dtype=np.float64).reshape(-1)
        dn = np.asarray(gt["data"][f"episode_{ep}"]["fixb_delta_n_0"], dtype=np.float64).reshape(-1)
        if len(dn) != len(robot_t):
            raise RuntimeError(f"episode {ep} v2n δn length {len(dn)} != pose {len(robot_t)}")
        out[ep] = {
            "robot_t": robot_t,
            "rgb_t": rgb_t,
            "R": R,
            "p": p,
            "wrench": wrench,
            "wrench_t": wt,
            "dn": dn,
            "field": field,
        }
    return out


def _query_state(cache, t):
    p, i = _nearest(cache["robot_t"], cache["p"], t)
    R, _ = _nearest(cache["robot_t"], cache["R"], t)
    w, _ = _nearest(cache["wrench_t"], cache["wrench"], t)
    query = cache["field"].query(np.asarray(p, dtype=np.float64).reshape(1, 3))
    frame = np.asarray(query["R"][0], dtype=np.float64)
    normal = frame[:, 2]
    normal = normal / np.linalg.norm(normal)
    outward = -normal
    foot = np.asarray(query["x_s"], dtype=np.float64).reshape(-1, 3)[0]
    clearance = float(np.dot(p - foot, outward))
    f_world = R @ w[:3]
    fn = float(np.dot(f_world, normal))
    return {
        "p": np.asarray(p, dtype=np.float64),
        "R": np.asarray(R, dtype=np.float64),
        "frame": frame,
        "normal": normal,
        "outward": outward,
        "clearance": clearance,
        "fn": fn,
        "force": float(np.linalg.norm(w[:3])),
        "dn": float(cache["dn"][i]),
    }


def _decode(action, state, dts, v3r: bool):
    if action.shape[1] == 10:
        frames = np.repeat(state["frame"].reshape(1, 3, 3), len(action), axis=0)
        xyz, tool_R = decode_horizon(
            action[:, :10], state["p"], frames, dts, state["R"] if v3r else None
        )
        return xyz, tool_R, action[:, 2], -K_NPM * action[:, 2]
    if action.shape[1] != 21:
        raise RuntimeError(f"action {action.shape}")
    from PyriteUtility.spatial_math.spatial_utilities import pose9_to_SE3

    T = np.eye(4)
    T[:3, :3] = state["R"]
    T[:3, 3] = state["p"]
    ref = T @ pose9_to_SE3(action[:, 0:9])
    vt = T @ pose9_to_SE3(action[:, 9:18])
    xyz = ref[:, :3, 3]
    tool_R = ref[:, :3, :3]
    offset = vt[:, :3, 3] - xyz
    dn = np.sum(offset * state["outward"], axis=1)
    force = action[:, 18] * np.sum(offset * state["normal"], axis=1)
    return xyz, tool_R, dn, force


def _jumps(tool_R) -> int:
    if len(tool_R) < 2:
        return 0
    rel = np.einsum("nji,njk->nik", tool_R[:-1], tool_R[1:])
    c = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    return int(np.sum(np.degrees(np.arccos(c)) > JUMP_DEG))


def _pick(queries):
    """0.5 s stride, plus every query in the 0.5 s after contact ends."""
    chosen = []
    last = -1e9
    for row in queries:
        lift = row["lift"]
        if lift or row["t"] - last >= STRIDE_S:
            chosen.append(row)
            if not lift:
                last = row["t"]
    return chosen


def _score_one(name, ckpt, field) -> dict:
    import hydra
    import torch
    from omegaconf import OmegaConf
    from PyriteEnvSuites.scripts.run_policy_open_loop_val_export import (
        load_policy_and_cfg,
        predict_sparse_action,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, cfg = load_policy_and_cfg(str(ckpt), device)
    task = str(cfg.task.name)
    v3r = "V3R" in task.upper()
    shape = cfg.task.shape_meta
    for node in (shape, getattr(cfg.task.dataset, "shape_meta", None)):
        if node is None or "rgb_0" not in node.sample.obs.sparse or "rgb_0" in node.obs:
            continue
        from omegaconf import open_dict

        with open_dict(node):
            node.obs.rgb_0 = {"type": "rgb", "shape": [3, 224, 224]}
    stride = int(shape.sample.action.sparse.down_sample_steps)
    horizon = int(shape.sample.action.sparse.horizon)
    OmegaConf.set_struct(cfg.task.dataset, False)
    for key in list(cfg.task.dataset.keys()):
        value = cfg.task.dataset[key]
        if not isinstance(value, str) or not value.startswith("/"):
            continue
        path = Path(value)
        if path.exists():
            continue
        local = LOCAL_DATA / Path(*path.parts[2:]) if len(path.parts) > 2 and path.parts[1] == "data" else None
        if local is not None and local.exists():
            cfg.task.dataset[key] = str(local)
            continue
        if key == "action_normalizer_path":
            found = next((root / "assets" / path.name for root in (SCRIPTS, V2N) if (root / "assets" / path.name).is_file()), None)
            if found is None:
                raise SystemExit(f"{name} normalizer {path} not local")
            cfg.task.dataset[key] = str(found)
        elif key == "dataset_path":
            zarr = LOCAL_DATA / path.parent.name / "acp_replay_buffer.zarr"
            if not zarr.is_dir():
                raise SystemExit(f"{name} dataset is not local: {value}")
            cfg.task.dataset[key] = str(zarr)
    if "exclusion_stats_path" in cfg.task.dataset:
        cfg.task.dataset.exclusion_stats_path = str(OUT / f"{name}_exclusion.json")
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    val = dataset.get_validation_dataset()
    episodes = sorted({int(ep) for ep, _rgb in val.sampler.indices})
    zarr_path = Path(str(cfg.task.dataset.dataset_path))
    cache = _episode_cache(zarr_path, episodes, field)
    by_ep = {ep: [] for ep in episodes}
    for idx, (ep, rgb) in enumerate(val.sampler.indices):
        ep = int(ep)
        rgb = int(rgb)
        t = float(cache[ep]["rgb_t"][rgb])
        by_ep[ep].append({"idx": idx, "ep": ep, "rgb": rgb, "t": t})
    buckets = {phase: {"pos": [], "rot": [], "dn": [], "force": []} for phase in PHASES}
    lift_dn = []
    lift_pos = 0
    jumps = []
    n_pred = 0
    for ep, rows in by_ep.items():
        rows.sort(key=lambda row: row["t"])
        states = [_query_state(cache[ep], row["t"]) for row in rows]
        labels = label_segments(
            [s["force"] for s in states],
            [max(s["clearance"], 0.0) for s in states],
        )
        ends = []
        for i, label in enumerate(labels):
            nxt = labels[i + 1] if i + 1 < len(labels) else None
            if label == "contact" and nxt != "contact":
                ends.append(rows[i]["t"])
        for row, state, label in zip(rows, states, labels):
            row["phase"] = str(label)
            row["state"] = state
            row["lift"] = any(0.0 < row["t"] - end <= LIFT_S for end in ends)
        for row in _pick(rows):
            if row["phase"] not in buckets:
                continue
            sample = val[row["idx"]]
            keep_rgb = set(policy.obs_encoder.rgb_keys)
            obs = {
                key: value.numpy()
                for key, value in sample["obs"]["sparse"].items()
                if not key.startswith("rgb_") or key in keep_rgb
            }
            pred = np.asarray(predict_sparse_action(policy, {"sparse": obs}, device), dtype=np.float64)
            state = row["state"]
            dt = float(np.median(np.diff(cache[ep]["rgb_t"]))) * stride
            dts = np.full(min(horizon, len(pred)), dt, dtype=np.float64)
            xyz, tool_R, dn, force = _decode(pred[: len(dts)], state, dts, v3r)
            t_next = row["t"] + dt
            p_gt, _ = _nearest(cache[ep]["robot_t"], cache[ep]["p"], t_next)
            step = 1 if len(xyz) > 1 else 0
            pos = float(np.linalg.norm(xyz[step] - p_gt) * 1e3)
            gt_action = sample["action"]["sparse"].numpy()
            if pred.shape[1] == 10:
                rot = _angle(
                    rot6d_rows_to_R(pred[0:1, 4:10])[0],
                    rot6d_rows_to_R(gt_action[0:1, 4:10])[0],
                )
            else:
                rot = _angle(
                    rot6d_rows_to_R(pred[step:step + 1, 3:9])[0],
                    rot6d_rows_to_R(gt_action[step:step + 1, 3:9])[0],
                )
            phase = row["phase"]
            buckets[phase]["pos"].append(pos)
            buckets[phase]["rot"].append(rot)
            buckets[phase]["dn"].append((float(dn[0]) - state["dn"]) * 1e3)
            buckets[phase]["force"].append(abs(float(force[0]) - state["fn"]))
            jumps.append(_jumps(tool_R))
            if row["lift"]:
                lift_dn.append(float(dn[0]) * 1e3)
                lift_pos += int(dn[0] > 0.0)
            n_pred += 1
            if n_pred % 25 == 0:
                print(f"[{name}] {n_pred}", flush=True)
    del policy, dataset, val
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    lift = np.asarray(lift_dn, dtype=np.float64)
    return {
        "name": name,
        "checkpoint": str(ckpt),
        "task": task,
        "episodes": episodes,
        "n": n_pred,
        "v3r_decode": v3r,
        "phases": {
            phase: {key: _stat(values) for key, values in cols.items()}
            for phase, cols in buckets.items()
        },
        "lift_0.5s": {
            "n": int(lift.size),
            "pred_dn_mm_median": None if lift.size == 0 else round(float(np.median(lift)), 3),
            "pred_dn_mm_p90": None if lift.size == 0 else round(float(np.percentile(lift, 90)), 3),
            "frac_clearance_positive": None if lift.size == 0 else round(float(lift_pos / lift.size), 3),
        },
        "tilt_steps_gt_10deg": _stat(jumps),
    }


def _cell(stat) -> str:
    if not stat or stat["n"] == 0:
        return "—"
    return f"{stat['median']} / {stat['p90']}"


def _markdown(rows: list[dict]) -> str:
    metrics = ("pos", "rot", "dn", "force")
    heads = ["checkpoint"]
    for phase in PHASES:
        for metric in metrics:
            heads.append(f"{metric} {phase}")
    heads += ["lift δn mm", "lift frac >0", "tilt >10°"]
    lines = [
        "Open-loop held-out val. Cells are median / p90.",
        "pos mm = decoded TCP one action step vs recorded TCP. rot deg = predicted ΔR vs the label ΔR.",
        "δn mm = predicted δn minus field_v2n δn (air = clearance, contact = -(F·n)/3000).",
        "force N = |predicted normal force − F·n|. r1rel δn is the virtual-target offset along the outward normal.",
        "Tilt is the count of horizon steps whose rotation jumps more than 10°.",
        "",
        "| " + " | ".join(heads) + " |",
        "|" + "|".join("---" for _ in heads) + "|",
    ]
    for row in rows:
        cells = [row["name"]]
        for phase in PHASES:
            for metric in metrics:
                cells.append(_cell(row["phases"][phase][metric]))
        lift = row["lift_0.5s"]
        cells.append("—" if lift["pred_dn_mm_median"] is None else f"{lift['pred_dn_mm_median']} / {lift['pred_dn_mm_p90']}")
        cells.append("—" if lift["frac_clearance_positive"] is None else str(lift["frac_clearance_positive"]))
        cells.append(_cell(row["tilt_steps_gt_10deg"]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mask = get_val_mask(50, 0.1, seed=42)
    print("val episodes", np.flatnonzero(mask).tolist(), flush=True)
    field = FixBMoldField(str(FIELD))
    rows = []
    for name, ckpt in MODELS.items():
        done = OUT / f"{name}.json"
        if done.is_file():
            rows.append(json.loads(done.read_text()))
            print(f"keep {name}", flush=True)
            continue
        print(f"score {name}", flush=True)
        row = _score_one(name, ckpt, field)
        rows.append(row)
        done.write_text(json.dumps(row, indent=2) + "\n")
        print(name, row["lift_0.5s"], row["tilt_steps_gt_10deg"], flush=True)
    text = _markdown(rows)
    (OUT / "table.md").write_text(text)
    (OUT / "table.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
