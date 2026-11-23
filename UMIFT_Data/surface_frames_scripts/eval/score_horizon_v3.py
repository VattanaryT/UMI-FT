#!/usr/bin/env python3
"""Horizon rescore. Writes score_horizon_v3.json and .md. Does not touch v2 or onestep.

r4vt position is the integrated tangent path against the demo's tangent path.
tcp is that same absolute decode against the demo TCP (normal and start included).
δn is not a coordinate. Rotation is R_field @ ΔR at every step. r1rel tcp matches
its position, which is already the TCP error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent
SCRIPTS = EVAL.parent
sys.path[:0] = [str(SCRIPTS), str(EVAL), str(EVAL / "thin_heldout"), str(EVAL / "viewer")]

import importlib.util  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# heldout inserts the surface_frames tree on sys.path. Load these by file first.
sc = _load("score_sign_flip", EVAL / "score_sign_flip.py")
fv = _load("score_field_v2_add", EVAL / "score_field_v2_add.py")
import heldout  # noqa: E402
from fixb_labels import delta_R_from_field_tool, so3_to_rot6d_rows  # noqa: E402
from fixb_tcp_track import project_tcp_velocity  # noqa: E402
from heading_tilt import geodesic_deg  # noqa: E402
from replay_gt import _action_rows  # noqa: E402
from run_policy_open_loop_val_export import load_policy_and_cfg, predict_sparse_action  # noqa: E402
import PyriteUtility.spatial_math.spatial_utilities as su  # noqa: E402
from PyriteConfig.tasks.umift.umift_type_conversions import action21_to_action_sample  # noqa: E402

STEPS = (1, 6, 16, 64)
OUT_JSON = EVAL / "score_horizon_v3.json"
OUT_MD = EVAL / "score_horizon_v3.md"
FIELD_V2 = Path("/mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_field_v2_k3000/checkpoints")
K3000 = sc.TRAIN / (
    "oct-v35-h4s-r4vt-causal-foot-carry-k3000_e200_bs64_valtopk_43472139/checkpoints/e199_final.ckpt"
)
R1REL = sc.TRAIN / "oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt"


def _fmt(block, key) -> str:
    return "/".join(f"{block[key][str(step)]:.1f}" for step in STEPS)


def _tangent_demo(tool_p, frames, index):
    p = np.asarray(tool_p[int(index[0])], dtype=np.float64).copy()
    out = np.empty((len(index), 3))
    for k, i in enumerate(index):
        out[k] = p
        if k + 1 == len(index):
            break
        delta = tool_p[int(index[k + 1])] - tool_p[int(i)]
        normal = frames[k, :, 2]
        p = p + delta - normal * np.dot(delta, normal)
    return out


def _step_median(block):
    return {str(step): round(float(np.median(block[:, step - 1])), 2) for step in STEPS}


def _stats(pos, rot, tcp) -> dict:
    pos = np.stack(pos)
    rot = np.stack(rot)
    tcp = np.stack(tcp)
    return {
        "n": int(len(pos)),
        "pos_median_mm": _step_median(pos),
        "tcp_median_mm": _step_median(tcp),
        "rot_median_deg": _step_median(rot),
        "pos_mm": heldout._mean(pos),
        "tcp_mm": heldout._mean(tcp),
        "rot_deg": heldout._mean(rot),
    }


def _gt_velocity(demo):
    cached = demo.get("_gt_v")
    if cached is None:
        cached, _dt = project_tcp_velocity(
            demo["frame"], demo["tool_p"], demo["robot_t"], demo["rgb_t"]
        )
        demo["_gt_v"] = cached
    return cached


def _gt_action(demo, index):
    index = np.asarray(index, dtype=np.int64)
    frames = np.asarray(demo["frame"], dtype=np.float64)[index]
    action = np.zeros((len(index), 10), dtype=np.float64)
    action[:, 0:2] = _gt_velocity(demo)[index]
    action[:, 4:10] = so3_to_rot6d_rows(delta_R_from_field_tool(frames, demo["tool_R"][index]))
    return action


def _r4vt(action, demo, index):
    """Tangent miss, full TCP miss, rotation. xyz starts at the measured TCP."""
    xyz, got_R = sc.horizon_pose("causal", action, demo, index)
    index = np.asarray(index, dtype=np.int64)
    frames = np.asarray(demo["frame"], dtype=np.float64)[index]
    tool = np.asarray(demo["tool_p"], dtype=np.float64)[index]
    ref = _tangent_demo(demo["tool_p"], frames, index)
    pos = np.linalg.norm(xyz - ref, axis=1) * 1e3
    tcp = np.linalg.norm(xyz - tool, axis=1) * 1e3
    rot = geodesic_deg(got_R, demo["tool_R"][index])
    return pos, rot, tcp


def _rows_errors(kind, rows, demo, predict):
    pos, rot, tcp = [], [], []
    gt_pos, gt_rot, gt_tcp = [], [], []
    for idx, _rgb, index in rows:
        pred = predict(idx)
        if kind == "r1rel":
            p_mm, r_deg, _xyz = sc.pose_error("r1rel", pred, demo, index)
            g_mm, g_deg, _g = sc.pose_error("r1rel", _r1rel_label(demo, index), demo, index)
            t_mm, gt_t = p_mm, g_mm
        else:
            p_mm, r_deg, t_mm = _r4vt(pred, demo, index)
            g_mm, g_deg, gt_t = _r4vt(_gt_action(demo, index), demo, index)
        pos.append(p_mm)
        rot.append(r_deg)
        tcp.append(t_mm)
        gt_pos.append(g_mm)
        gt_rot.append(g_deg)
        gt_tcp.append(gt_t)
    return _stats(pos, rot, tcp), _stats(gt_pos, gt_rot, gt_tcp)


def _r1rel_label(demo, index):
    cached = demo.get("_r1rel_abs")
    if cached is None:
        raise RuntimeError("r1rel demo has no absolute action")
    group_action, pose = cached
    absolute = _action_rows(group_action, index, with_delta_r=False)
    base = su.pose7_to_SE3(pose[int(index[0])].reshape(1, 7))[0]
    relative = action21_to_action_sample(absolute, np.zeros((0,)), [0], [base])["sparse"]
    return np.asarray(relative, dtype=np.float64)


def _predict_fn(policy, dataset, device):
    def predict(idx):
        obs, _sample = dataset.sampler.sample_sequence(idx)
        pred = np.asarray(predict_sparse_action(policy, obs, device), dtype=np.float64)
        return pred.reshape(-1, pred.shape[-1])[: heldout.H]

    return predict


def _print(name, letter, block, gt):
    print(
        f"[h3] {name} {letter} n {block['n']} "
        f"pos {_fmt(block, 'pos_median_mm')} tcp {_fmt(block, 'tcp_median_mm')} "
        f"rot {_fmt(block, 'rot_median_deg')} "
        f"gt pos {_fmt(gt, 'pos_median_mm')} tcp {_fmt(gt, 'tcp_median_mm')} "
        f"rot {_fmt(gt, 'rot_median_deg')}",
        flush=True,
    )


def _pool(parts):
    pos = [row for part in parts for row in part[0]]
    rot = [row for part in parts for row in part[1]]
    tcp = [row for part in parts for row in part[2]]
    return _stats(pos, rot, tcp)


def _train_block(kind, policy, dataset, demos, grouped, device):
    predict = _predict_fn(policy, dataset, device)
    pred_parts, gt_parts = [], []
    done = 0
    for epi, rows in grouped.items():
        demo = demos[epi]
        pos, rot, tcp = [], [], []
        gpos, grot, gtcp = [], [], []
        for idx, _rgb, index in rows:
            pred = predict(idx)
            if kind == "r1rel":
                p_mm, r_deg, _xyz = sc.pose_error("r1rel", pred, demo, index)
                g_mm, g_deg, _g = sc.pose_error("r1rel", _r1rel_label(demo, index), demo, index)
                t_mm, gt_t = p_mm, g_mm
            else:
                p_mm, r_deg, t_mm = _r4vt(pred, demo, index)
                g_mm, g_deg, gt_t = _r4vt(_gt_action(demo, index), demo, index)
            pos.append(p_mm)
            rot.append(r_deg)
            tcp.append(t_mm)
            gpos.append(g_mm)
            grot.append(g_deg)
            gtcp.append(gt_t)
            done += 1
            if done % 40 == 0:
                print(f"[h3] {kind} train {done}", flush=True)
        pred_parts.append((pos, rot, tcp))
        gt_parts.append((gpos, grot, gtcp))
    return _pool(pred_parts), _pool(gt_parts)


def _md(report) -> str:
    lines = [
        "# Horizon v3",
        "",
        "Median mm and degrees at steps 1/6/16/64.",
        "r4vt pos is integrated v against the demo tangent path. δn is not added.",
        "tcp is that absolute path, started at the measured TCP, against the demo TCP.",
        "r1rel pos is already the TCP error, so its tcp column matches pos.",
        "",
        "| run | set | n | pos median | tcp median | rot median | gt pos | gt tcp | gt rot |",
        "|---|---|---:|---|---|---|---|---|---|",
    ]
    for row in report:
        for key in ("C", "A", "train"):
            block, gt = row[key], row[f"{key}_gt"]
            lines.append(
                "| {run} | {key} | {n} | {pos} | {tcp} | {rot} | {gpos} | {gtcp} | {grot} |".format(
                    run=row["name"],
                    key=key,
                    n=block["n"],
                    pos=_fmt(block, "pos_median_mm"),
                    tcp=_fmt(block, "tcp_median_mm"),
                    rot=_fmt(block, "rot_median_deg"),
                    gpos=_fmt(gt, "pos_median_mm"),
                    gtcp=_fmt(gt, "tcp_median_mm"),
                    grot=_fmt(gt, "rot_median_deg"),
                )
            )
    lines.append("")
    return "\n".join(lines)


def _attach_r1rel(demos, zarr_path):
    import zarr

    root = zarr.open(zarr_path, "r")["data"]
    for epi, demo in demos.items():
        group = root[f"episode_{epi}"]
        demo["_r1rel_abs"] = (group, np.asarray(group["ts_pose_fb_0"], dtype=np.float64))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = []

    policy, cfg = load_policy_and_cfg(str(FIELD_V2 / "epoch=0025-val_loss=0.048.ckpt"), "cpu")
    fv._point(cfg)
    recordings, _mesh = fv._thin_plain(cfg, fv.field_v2.NORM)
    del policy

    for which, path in (
        ("field_v2 e25", FIELD_V2 / "epoch=0025-val_loss=0.048.ckpt"),
        ("field_v2 e140", FIELD_V2 / "latest_1005.ckpt"),
    ):
        policy, cfg = load_policy_and_cfg(str(path), device)
        fv._point(cfg)
        epoch = sc._epoch(path)
        row = {"name": which, "epoch": epoch, "flip": False, "ckpt": str(path)}
        gt_row = {"name": f"gt {which}", "epoch": None, "flip": False}
        for letter, rec in recordings.items():
            block, gt = _rows_errors("causal", rec["rows"], rec["demo"], _predict_fn(policy, rec["ds"], device))
            row[letter] = block
            row[f"{letter}_gt"] = gt
            _print(which, letter, block, gt)
        val, zarr_path = sc._train_val(cfg, fv.field_v2.NORM)
        demos, grouped, n_win = sc._train_rows(val, zarr_path)
        block, gt = _train_block("causal", policy, val, demos, grouped, device)
        block["windows"] = n_win
        row["train"] = block
        row["train_gt"] = gt
        _print(which, "train", block, gt)
        report.append(row)
        del policy, val, demos, grouped
        torch.cuda.empty_cache()

    from fixb_field import FixBMoldField

    thin, _axis = sc._thin(FixBMoldField(str(heldout.WALL)))
    policy, cfg = load_policy_and_cfg(str(K3000), device)
    norm = sc._norm(cfg, "causal")
    epoch = sc._epoch(K3000)
    row = {"name": "k3000 e199 flip", "epoch": epoch, "flip": True, "ckpt": str(K3000)}
    for letter, rec in thin.items():
        dataset = heldout._dataset(cfg, rec["path"], norm)
        sc.sign_flip._flip_obs(dataset)
        rows = sc._filter(heldout._queries(dataset, rec["demo"]), rec["kept_i0"])
        block, gt = _rows_errors("causal", rows, rec["demo"], _predict_fn(policy, dataset, device))
        row[letter] = block
        row[f"{letter}_gt"] = gt
        _print(row["name"], letter, block, gt)
    val, zarr_path = sc._train_val(cfg, norm)
    demos, grouped, n_win = sc._train_rows(val, zarr_path)
    block, gt = _train_block("causal", policy, val, demos, grouped, device)
    block["windows"] = n_win
    row["train"] = block
    row["train_gt"] = gt
    _print(row["name"], "train", block, gt)
    report.append(row)
    del policy, val, demos, grouped
    torch.cuda.empty_cache()

    policy, cfg = load_policy_and_cfg(str(R1REL), device)
    norm = sc._norm(cfg, "r1rel")
    epoch = sc._epoch(R1REL)
    row = {"name": "r1rel e195", "epoch": epoch, "flip": False, "ckpt": str(R1REL)}
    # Thin queries are the causal kept set, decoded as r1rel on the same replay.
    for letter, rec in thin.items():
        dataset = heldout._dataset(cfg, rec["path"], norm)
        rows = sc._filter(heldout._queries(dataset, rec["demo"]), rec["kept_i0"])
        import zarr

        group = zarr.open(str(rec["path"]), "r")["data"]["episode_0"]
        rec["demo"]["_r1rel_abs"] = (group, np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
        block, gt = _rows_errors("r1rel", rows, rec["demo"], _predict_fn(policy, dataset, device))
        row[letter] = block
        row[f"{letter}_gt"] = gt
        _print(row["name"], letter, block, gt)
    val, zarr_path = sc._train_val(cfg, norm)
    demos, grouped, n_win = sc._train_rows(val, zarr_path)
    _attach_r1rel(demos, zarr_path)
    block, gt = _train_block("r1rel", policy, val, demos, grouped, device)
    block["windows"] = n_win
    row["train"] = block
    row["train_gt"] = gt
    _print(row["name"], "train", block, gt)
    report.append(row)

    OUT_JSON.write_text(json.dumps(report, indent=2))
    OUT_MD.write_text(_md(report))
    print(f"[h3] wrote {OUT_JSON}", flush=True)
    print(f"[h3] wrote {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
