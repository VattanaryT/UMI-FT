#!/usr/bin/env python3
"""Best-val vs latest. Thin: sign_flip t1/t2 and contact_wall_v2 kept sets.
Training: 200 validation windows, no flip.

r4vt position is the one-step tangent miss (v * control dt). δn is not a TCP
coordinate. Rotation is the demonstrated frame at that step, not a walked one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HELD = Path(__file__).resolve().parent / "thin_heldout"
TILT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HELD), str(HELD.parent / "viewer"), str(TILT)]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import contact_wall as cw  # noqa: E402
import heldout  # noqa: E402

sys.path.insert(0, str(TILT))
import sign_flip  # noqa: E402
from diffusion_policy.common.drop_depth_inputs import apply_drop_depth_inputs  # noqa: E402
from diffusion_policy.common.drop_scan_inputs import apply_drop_scan_inputs  # noqa: E402
from diffusion_policy.common.sampler_umift import sparse_action_indices_from_rgb_timestamps  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import interp_rows, next_control_times, pose7_to_Rp  # noqa: E402
from heading_tilt import geodesic_deg, long_axis, rotation_from_tilt  # noqa: E402
from mold_aware_runtime import rot6d_rows_to_R  # noqa: E402
from PyriteUtility.data_pipeline.wrench_baseline_utils import time_based_moving_average  # noqa: E402
from run_policy_open_loop_val_export import load_policy_and_cfg, predict_sparse_action  # noqa: E402
from tilt1 import heading_half, rotation_from_tilt1  # noqa: E402

TRAIN = Path("/mnt/data/umidata/ft/training_outputs")
DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
TILT_NORM = TILT / "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_tilt1.pkl"
K3000_NORM = TILT / "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_causal_foot_carry_k3000.pkl"
# Horizon table is score_sign_flip_v2.json and is not rewritten here.
OUT = Path(__file__).resolve().parent / "score_sign_flip_onestep.json"
STEPS = (1, 6, 16, 64)


def _flags(face: str, mag: np.ndarray):
    mag = np.asarray(mag, dtype=np.float64)
    off = not (float(mag[0]) >= 8.0 and float(np.mean(mag >= 5.0)) >= 0.80)
    return (face not in ("cap", "back") and not off), face == "cap", face == "back", off


cw._flags = _flags


def _host(path: str) -> str:
    text = str(path)
    if text.startswith("/data/"):
        cand = DATA / text[len("/data/"):]
        if cand.exists():
            return str(cand)
    return text


def _norm(cfg, kind):
    if kind == "tilt1":
        return TILT_NORM
    if kind == "causal":
        if not K3000_NORM.is_file():
            raise SystemExit(f"missing {K3000_NORM}")
        return K3000_NORM
    found = heldout._norm_of(cfg, kind)
    if found is None:
        raise SystemExit(f"{kind} normalizer rejected")
    return found[0]


def _epoch(path: Path):
    if "epoch=" in path.name:
        return int(path.name.split("epoch=")[1].split("-")[0])
    return heldout.fair.ckpt_epoch(path)


def _fmt(block, key) -> str:
    return "/".join(f"{block[key][str(step)]:.1f}" for step in STEPS)


def _public(block):
    return {
        "n": int(block["n"]),
        "pos_mm": heldout._mean(block["_pos"]),
        "rot_deg": heldout._mean(block["_rot"]),
    }


def _check():
    dead = np.zeros((3, 3))
    dead[:, 2] = [0.1, 0.1, -(0.4 + 0.5)]
    if heading_half(dead)[0] == 0.0 or heading_half(dead, previous=0.0)[0] != 0.0:
        raise RuntimeError("window heading must keep the side carried into the chunk")


def _episode_heading(demo, axis):
    cached = demo.get("_heading")
    if cached is None or cached[0] is not axis:
        n_out = -np.asarray(demo["frame"], dtype=np.float64)[:, :, 2]
        direction = heldout._e_long(n_out, axis)
        cached = (axis, direction, heading_half(n_out))
        demo["_heading"] = cached
    return cached[1], cached[2]


def _control_step(demo, index):
    """dt and the TCP at the next control time. v * dt is that tangent step."""
    robot_t = np.asarray(demo["robot_t"], dtype=np.float64)
    t_next = next_control_times(robot_t, np.asarray(demo["rgb_t"], dtype=np.float64))[index]
    dt = t_next - robot_t[index]
    nxt = interp_rows(t_next, robot_t, np.asarray(demo["tool_p"], dtype=np.float64))
    return dt, nxt


def pose_error(kind, action, demo, index, axis=None):
    """pos mm (H,), rot deg (H,), decoded step-1-onward xyz (H, 3).

    r4vt position is the one-step tangent miss. δn is the virtual-target
    clearance, so it is not a TCP coordinate. Rotation uses the demonstrated
    frame at that index: frame @ ΔR, or tilt a/b, or 1-tilt a.
    """
    action = np.asarray(action, dtype=np.float64)
    index = np.asarray(index, dtype=np.int64)
    tcp = np.asarray(demo["tool_p"], dtype=np.float64)[index]
    tool_R = np.asarray(demo["tool_R"], dtype=np.float64)[index]
    if kind == "r1rel":
        xyz, got_R = heldout.fair.r1rel_pose(action, tcp[0], tool_R[0])
        pos = np.linalg.norm(xyz - tcp, axis=1) * 1e3
        return pos, geodesic_deg(got_R, tool_R), xyz
    frame = np.asarray(demo["frame"], dtype=np.float64)[index]
    dt, nxt = _control_step(demo, index)
    step = np.einsum("nij,nj->ni", frame[:, :, :2], action[:, 0:2] * dt[:, None])
    delta = nxt - tcp
    normal = frame[:, :, 2]
    along = np.einsum("ni,ni->n", delta, normal)
    ref = tcp + (delta - along[:, None] * normal)
    xyz = tcp + step
    pos = np.linalg.norm(xyz - ref, axis=1) * 1e3
    if kind == "causal":
        got_R = np.einsum("nij,njk->nik", frame, rot6d_rows_to_R(action[:, 4:10]))
    else:
        if axis is None:
            raise RuntimeError(f"{kind} needs the mold long axis")
        direction, sides = _episode_heading(demo, axis)
        n_out = -frame[:, :, 2]
        e_long = direction[index]
        if kind == "tilt":
            got_R = rotation_from_tilt(n_out, e_long, action[:, 4:8])
        elif kind == "tilt1":
            got_R = rotation_from_tilt1(
                n_out, e_long, action[:, 4], previous=float(sides[int(index[0])])
            )
        else:
            raise RuntimeError(f"unknown action {kind}")
    return pos, geodesic_deg(got_R, tool_R), xyz


def horizon_pose(kind, action, demo, index, axis=None):
    """Integrated v and R_field @ ΔR at every step. δn is not a position."""
    del axis
    if kind != "causal":
        raise RuntimeError(f"horizon walk is r4vt, not {kind}")
    sys.path.insert(0, str(TILT))
    from r4vt_horizon import control_dts, decode_horizon

    index = np.asarray(index, dtype=np.int64)
    frames = np.asarray(demo["frame"], dtype=np.float64)[index]
    dts = control_dts(demo["robot_t"], demo["rgb_t"], index)
    return decode_horizon(action, demo["tool_p"][int(index[0])], frames, dts)


def _score(kind, policy, dataset, field, demo, rows, device, axis):
    if kind == "r1rel":
        return heldout._score_one(kind, policy, dataset, field, demo, rows, device, axis)
    width = 5 if kind == "tilt1" else 10
    pos, rot = [], []
    for n_done, (idx, _rgb, index) in enumerate(rows):
        obs, _sample = dataset.sampler.sample_sequence(idx)
        pred = np.asarray(predict_sparse_action(policy, obs, device), dtype=np.float64)
        pred = pred.reshape(-1, pred.shape[-1])[: heldout.H]
        if pred.shape[1] != width:
            raise RuntimeError(f"{kind} action {pred.shape}")
        p_mm, r_deg, _xyz = pose_error(kind, pred, demo, index, axis)
        pos.append(p_mm)
        rot.append(r_deg)
        if n_done and n_done % 40 == 0:
            print(f"[score] {kind} {n_done}/{len(rows)}", flush=True)
    return {"n": len(rows), "_pos": np.stack(pos), "_rot": np.stack(rot)}


def _filter(rows, kept_i0):
    chosen = [row for row in rows if int(row[2][0]) in kept_i0]
    if len(chosen) != len(kept_i0):
        raise RuntimeError(f"kept {len(chosen)} vs {len(kept_i0)}")
    return chosen


def _thin(field):
    recordings, axis = cw._recordings(field)
    _policy, cfg = load_policy_and_cfg(str(cw.MODELS[0][2]), "cpu")
    norm = heldout._norm_of(cfg, "causal")[0]
    cache = {}
    for letter, rec in recordings.items():
        dataset, rows = cw._rows_for(cfg, norm, rec, cache)
        kept, _detail, counts = cw._select(rec["demo"], rows)
        expect = 9 if letter == "C" else 94
        if counts["kept"] != expect:
            raise SystemExit(f"{letter} kept {counts['kept']}")
        rec["kept_i0"] = {int(index[0]) for _i, _r, index in kept}
        rec["rows"] = _filter(rows, rec["kept_i0"])
        sign_flip._flip_obs(dataset)
        sign_flip._flip_demo(rec["demo"])
        rec["causal_ds"] = dataset
        print(f"[score] thin {letter} {len(rec['rows'])}", flush=True)
    del _policy
    return recordings, axis


def _train_val(cfg, norm):
    ds_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    apply_drop_scan_inputs(ds_cfg)
    apply_drop_depth_inputs(ds_cfg)
    zarr_path = _host(str(ds_cfg.task.dataset.dataset_path))
    OmegaConf.update(ds_cfg, "task.dataset_path", zarr_path, merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.dataset_path", zarr_path, merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.action_normalizer_path", str(norm), merge=False)
    for key in ("long_dt_mask_path", "pose_jump_flag_path"):
        if key in ds_cfg.task.dataset and ds_cfg.task.dataset[key]:
            OmegaConf.update(ds_cfg, f"task.dataset.{key}", _host(str(ds_cfg.task.dataset[key])), merge=False)
    for key in ("exclusion_stats_path", "filter_stats_path"):
        if key in ds_cfg.task.dataset:
            OmegaConf.update(ds_cfg, f"task.dataset.{key}", None, merge=False)
    dataset = heldout.fair.hydra_instantiate(ds_cfg.task.dataset)
    return dataset.get_validation_dataset(), zarr_path


def _demo_group(group):
    robot_t = np.asarray(group["robot_time_stamps_0"]).reshape(-1)
    rgb_t = np.asarray(group["rgb_time_stamps_0"]).reshape(-1)
    tool_R, tool_p = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
    wrench_t = np.asarray(group["wrench_time_stamps_0"]).reshape(-1)
    wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
    smooth = time_based_moving_average(wrench, wrench_t, 1.0)[:, :3]
    return {
        "robot_t": robot_t,
        "rgb_t": rgb_t,
        "tool_p": tool_p,
        "tool_R": tool_R,
        "foot": np.asarray(group["mold_x_s_0"], dtype=np.float64).reshape(-1, 3),
        "frame": np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3),
        "normal": np.asarray(group["mold_normal_0"], dtype=np.float64).reshape(-1, 3),
        "wrench": wrench,
        "wrench_t": wrench_t,
        "smooth": smooth,
        "force": np.linalg.norm(wrench[:, :3], axis=1),
        "cutoff": None,
    }


def _train_rows(val, zarr_path):
    n = len(val.sampler.indices)
    if n < 1:
        raise SystemExit("validation split is empty")
    chosen = np.unique(np.linspace(0, n - 1, min(200, n)).astype(int))
    root = zarr.open(zarr_path, "r")["data"]
    demos, grouped = {}, {}
    for slot in chosen:
        epi, rgb = (int(v) for v in val.sampler.indices[int(slot)])
        if epi not in demos:
            demos[epi] = _demo_group(root[f"episode_{epi}"])
        demo = demos[epi]
        index, _times = sparse_action_indices_from_rgb_timestamps(
            demo["robot_t"], demo["rgb_t"], rgb, heldout.H, 1
        )
        index = np.asarray(index, dtype=np.int64)
        if len(index) < heldout.H or int(index[-1]) >= len(demo["robot_t"]):
            continue
        grouped.setdefault(epi, []).append((int(slot), rgb, index))
    return demos, grouped, int(sum(len(rows) for rows in grouped.values()))


def main() -> None:
    _check()
    sign_flip._check()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thin_field = FixBMoldField(str(heldout.WALL))
    recordings, thin_axis = _thin(thin_field)
    thin_field.frames = np.einsum("vij,jk->vik", thin_field.frames, sign_flip.S)
    train_field = FixBMoldField(str(sign_flip.TRAIN_FIELD))
    train_axis = long_axis(train_field.vertices)
    models = (
        ("k3000-carry", "e25", "causal", TRAIN / "oct-v35-h4s-r4vt-causal-foot-carry-k3000_e200_bs64_valtopk_43472139/checkpoints/epoch=0025-val_loss=0.048.ckpt"),
        ("k3000-carry", "latest", "causal", TRAIN / "oct-v35-h4s-r4vt-causal-foot-carry-k3000_e200_bs64_valtopk_43472139/checkpoints/e199_final.ckpt"),
        ("tilt1", "e25", "tilt1", TRAIN / "oct-v35-h4s-r4vt-tilt1_e200_bs64_valtopk_43467749/checkpoints/epoch=0025-val_loss=0.064.ckpt"),
        ("tilt1", "latest", "tilt1", TRAIN / "oct-v35-h4s-r4vt-tilt1_e200_bs64_valtopk_43467749/checkpoints/latest_pulled.ckpt"),
        ("r1rel", "e25", "r1rel", TRAIN / "oct_v35_h4s_odom_r1rel_valtopk/checkpoints/epoch=0025-val_loss=0.032.ckpt"),
        ("r1rel", "e195", "r1rel", TRAIN / "oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt"),
    )
    report = []
    for name, which, kind, path in models:
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        policy, cfg = load_policy_and_cfg(str(path), device)
        epoch = _epoch(path)
        norm = _norm(cfg, kind)
        row = {"run": name, "which": which, "epoch": epoch, "ckpt": str(path)}
        for letter, rec in recordings.items():
            if kind == "tilt1":
                dataset, rows = rec["causal_ds"], rec["rows"]
            else:
                dataset = heldout._dataset(cfg, rec["path"], norm)
                sign_flip._flip_obs(dataset)
                rows = _filter(heldout._queries(dataset, rec["demo"]), rec["kept_i0"])
            block = _score(kind, policy, dataset, thin_field, rec["demo"], rows, device, thin_axis)
            row[letter] = _public(block)
            print(
                f"[score] {name} {which} e{epoch} {letter} "
                f"pos {_fmt(row[letter], 'pos_mm')} rot {_fmt(row[letter], 'rot_deg')}",
                flush=True,
            )
        val, zarr_path = _train_val(cfg, norm)
        demos, grouped, n_win = _train_rows(val, zarr_path)
        pooled = heldout._pool([
            _score(kind, policy, val, train_field, demos[epi], rows, device, train_axis)
            for epi, rows in grouped.items()
        ])
        row["train"] = {"n": pooled["n"], "windows": n_win, "pos_mm": pooled["pos_mm"], "rot_deg": pooled["rot_deg"]}
        print(
            f"[score] {name} {which} train n {pooled['n']} "
            f"pos {_fmt(row['train'], 'pos_mm')} rot {_fmt(row['train'], 'rot_deg')}",
            flush=True,
        )
        report.append(row)
        del policy, val, demos, grouped
        torch.cuda.empty_cache()
    OUT.write_text(json.dumps(report, indent=2))
    print(f"[score] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
