#!/usr/bin/env python3
"""field_v2 on the sign-flip queries, no t1/t2 flip. Also tilt1 a-sign check.

Does not write score_sign_flip_v2.json or score_sign_flip_onestep.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import score_sign_flip as sc  # noqa: E402
import heldout  # noqa: E402
import contact_wall as cw  # noqa: E402

FV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FV))
import field_v2  # noqa: E402
from field_v2_dataset import mold_local  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import so3_to_rot6d_rows  # noqa: E402
from heading_tilt import geodesic_deg  # noqa: E402
from mold_aware_runtime import rot6d_rows_to_R  # noqa: E402
from run_policy_open_loop_val_export import load_policy_and_cfg, predict_sparse_action  # noqa: E402
from tilt1 import rotation_from_tilt1  # noqa: E402

TRAIN = Path("/mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_field_v2_k3000/checkpoints")
CKPTS = (
    ("e25", TRAIN / "epoch=0025-val_loss=0.048.ckpt"),
    ("latest_1005", TRAIN / "latest_1005.ckpt"),
)
OUT = EVAL / "score_field_v2_add.json"
STEPS = (1, 6, 16, 64)


def _fmt(block, key):
    return "/".join(f"{block[key][str(step)]:.1f}" for step in STEPS)


def _public(pos, rot):
    return {"n": int(len(pos)), "pos_mm": heldout._mean(pos), "rot_deg": heldout._mean(rot)}


def _predict(policy, dataset, rows, device):
    out = []
    for n_done, (idx, _rgb, index) in enumerate(rows):
        obs, _sample = dataset.sampler.sample_sequence(idx)
        pred = np.asarray(predict_sparse_action(policy, obs, device), dtype=np.float64)
        pred = pred.reshape(-1, pred.shape[-1])[: heldout.H]
        if pred.shape[1] != 10:
            raise RuntimeError(f"field_v2 action {pred.shape}")
        out.append((index, pred))
        if n_done and n_done % 40 == 0:
            print(f"[fv2] {n_done}/{len(rows)}", flush=True)
    return out


def _both(pred_rows, demo, field):
    """Horizon is rollout_tool. One-step is pose_error. Same predictions."""
    h_pos, h_rot, o_pos, o_rot = [], [], [], []
    heldout._WALK_CHECKED = True
    for index, pred in pred_rows:
        i0 = int(index[0])
        dts = heldout._dts(demo, index)
        xyz = sc.sign_flip.fair.rollout_tool(
            field, pred, demo["foot"][i0], demo["frame"][i0], demo["tool_p"][i0], dts
        )
        _walk_pos, frames = heldout._walk(
            field, pred, demo["foot"][i0], demo["frame"][i0], demo["tool_p"][i0], dts
        )
        got = np.einsum("nij,njk->nik", frames, rot6d_rows_to_R(pred[:, 4:10]))
        tool_p = demo["tool_p"][index]
        tool_R = demo["tool_R"][index]
        h_pos.append(np.linalg.norm(xyz - tool_p, axis=1) * 1e3)
        h_rot.append(geodesic_deg(got, tool_R))
        p_mm, r_deg, _xyz = sc.pose_error("causal", pred, demo, index)
        o_pos.append(p_mm)
        o_rot.append(r_deg)
    return np.stack(h_pos), np.stack(h_rot), np.stack(o_pos), np.stack(o_rot)


def _sign_rule(dataset, demo, origin, axes):
    """field_v2 reorient + mold-local position. Not the test-time S flip."""
    frame = field_v2.carry_frames(
        field_v2.reorient(np.asarray(demo["frame"], dtype=np.float64), axes),
        np.asarray(demo["tool_R"], dtype=np.float64),
    )
    demo["frame"] = frame
    demo.pop("_heading", None)
    ep = next(iter(dataset.replay_buffer["data"].values()))
    geom = ep["obs"]["robot0_mold_geom"]
    if len(geom) != len(frame) or geom.shape[1] < 10:
        raise RuntimeError(f"mold geom {geom.shape} vs frames {len(frame)}")
    geom[:, 4:10] = so3_to_rot6d_rows(frame).astype(geom.dtype)
    pos = ep["obs"]["robot0_eef_pos"]
    ep["obs"]["robot0_eef_pos"] = mold_local(pos, origin, axes).astype(np.float32)


def _mesh(path, frames):
    field = FixBMoldField(str(path))
    frames = np.asarray(frames, dtype=np.float64).reshape(-1, 3, 3)
    if len(frames) != len(field.frames):
        raise RuntimeError(f"frames {len(frames)} vs mesh {len(field.frames)}")
    field.frames = frames
    return field


def _thin_plain(cfg, norm):
    wall = FixBMoldField(str(heldout.WALL))
    recordings, _axis = cw._recordings(wall)
    vertices, frames, apron = field_v2._load_field(field_v2.THIN_FIELD)
    origin, axes = field_v2.mold_frame(vertices, apron)
    mesh = _mesh(heldout.WALL, field_v2.reorient(frames, axes))
    parent = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.update(parent, "task.dataset._target_", "fixb_r4vt_dataset.UmiFTDatasetR4VT", merge=False)
    cache = {}
    for letter, rec in recordings.items():
        dataset, rows = cw._rows_for(parent, norm, rec, cache)
        kept, _detail, counts = cw._select(rec["demo"], rows)
        expect = 9 if letter == "C" else 94
        if counts["kept"] != expect:
            raise SystemExit(f"{letter} kept {counts['kept']}")
        rec["rows"] = sc._filter(rows, {int(index[0]) for _i, _r, index in kept})
        _sign_rule(dataset, rec["demo"], origin, axes)
        rec["ds"] = dataset
        print(f"[fv2] thin {letter} {len(rec['rows'])} no S flip", flush=True)
    return recordings, mesh


def _tilt_flip_check(device):
    """Thin flip does not touch tilt_a. Compare raw a, negated a, and t1-as-heading."""
    field = FixBMoldField(str(heldout.WALL))
    recordings, axis = sc._thin(field)
    field.frames = np.einsum("vij,jk->vik", field.frames, sc.sign_flip.S)
    path = sc.TRAIN / (
        "oct-v35-h4s-r4vt-tilt1_e200_bs64_valtopk_43467749/checkpoints/"
        "epoch=0025-val_loss=0.064.ckpt"
    )
    policy, _cfg = load_policy_and_cfg(str(path), device)
    report = {}
    for letter, rec in recordings.items():
        raw, neg, along = [], [], []
        dataset, rows = rec["causal_ds"], rec["rows"]
        for idx, _rgb, index in rows:
            obs, _sample = dataset.sampler.sample_sequence(idx)
            pred = np.asarray(predict_sparse_action(policy, obs, device), dtype=np.float64)
            pred = pred.reshape(-1, pred.shape[-1])[: heldout.H]
            _p, got, _xyz = sc.pose_error("tilt1", pred, rec["demo"], index, axis)
            raw.append(got[0])
            flipped = pred.copy()
            flipped[:, 4] *= -1.0
            _p, got_n, _xyz = sc.pose_error("tilt1", flipped, rec["demo"], index, axis)
            neg.append(got_n[0])
            direction, sides = sc._episode_heading(rec["demo"], axis)
            frame = np.asarray(rec["demo"]["frame"], dtype=np.float64)[index]
            n_out = -frame[:, :, 2]
            t1 = frame[:, :, 0]
            t1 = t1 / np.clip(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12, None)
            got_R = rotation_from_tilt1(
                n_out, t1, pred[:, 4], previous=float(sides[int(index[0])])
            )
            along.append(float(geodesic_deg(got_R, rec["demo"]["tool_R"][index])[0]))
            raw_axis = direction[int(index[0])]
            dot = float(np.dot(t1[0], raw_axis))
            report.setdefault("_ang", []).append(float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))))
            report.setdefault("_ang_abs", []).append(float(np.degrees(np.arccos(np.clip(abs(dot), -1.0, 1.0)))))
        report[letter] = {
            "n": len(rows),
            "step1_raw": round(float(np.mean(raw)), 1),
            "step1_neg_a": round(float(np.mean(neg)), 1),
            "step1_e_long_is_flipped_t1": round(float(np.mean(along)), 1),
        }
        print(f"[tilt] {letter} {report[letter]}", flush=True)
    report["t1_vs_axis_deg"] = round(float(np.mean(report.pop("_ang"))), 1)
    report["t1_vs_axis_acute_deg"] = round(float(np.mean(report.pop("_ang_abs"))), 1)
    del policy
    torch.cuda.empty_cache()
    return report


def _exists_remap(text: str) -> str:
    if Path(text).exists():
        return text
    swaps = (
        ("/opt/umift/data/", str(sc.DATA) + "/"),
        ("/data/", str(sc.DATA) + "/"),
        ("/opt/umift/UMIFT_Data/surface_frames_scripts/", str(FV) + "/"),
    )
    for old, new in swaps:
        if text.startswith(old):
            cand = Path(new + text[len(old):])
            if cand.exists():
                return str(cand)
    return text


def _point(cfg) -> None:
    zarr_path = sc.DATA / "r4vt_field_v2_k3000/acp_replay_buffer.zarr"
    OmegaConf.update(cfg, "task.dataset_path", str(zarr_path), merge=False)
    OmegaConf.update(cfg, "task.dataset.dataset_path", str(zarr_path), merge=False)
    OmegaConf.update(cfg, "task.dataset.action_normalizer_path", str(field_v2.NORM), merge=False)
    for key in ("long_dt_mask_path", "pose_jump_flag_path"):
        if key in cfg.task.dataset and cfg.task.dataset[key]:
            OmegaConf.update(
                cfg, f"task.dataset.{key}", _exists_remap(str(cfg.task.dataset[key])), merge=False
            )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tilt = _tilt_flip_check(device)
    print(f"[tilt] {json.dumps(tilt)}", flush=True)
    policy, cfg = load_policy_and_cfg(str(CKPTS[0][1]), "cpu")
    norm = field_v2.NORM
    if not norm.is_file():
        raise SystemExit(f"missing {norm}")
    _point(cfg)
    recordings, thin_mesh = _thin_plain(cfg, norm)
    del policy
    saved = np.load(sc.DATA / "r4vt_field_v2_k3000/frames_pca_r020.npz")["frames"]
    train_mesh = _mesh(field_v2.TRAIN_FIELD, saved)
    report = {"tilt1_e25_thin_flip": tilt, "field_v2": []}
    for which, path in CKPTS:
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        policy, cfg = load_policy_and_cfg(str(path), device)
        _point(cfg)
        epoch = sc._epoch(path)
        row = {"run": "field_v2", "which": which, "epoch": epoch, "flip": False}
        for letter, rec in recordings.items():
            preds = _predict(policy, rec["ds"], rec["rows"], device)
            h_pos, h_rot, o_pos, o_rot = _both(preds, rec["demo"], thin_mesh)
            row[letter] = {"horizon": _public(h_pos, h_rot), "onestep": _public(o_pos, o_rot)}
            print(
                f"[fv2] {which} e{epoch} {letter} "
                f"H pos {_fmt(row[letter]['horizon'], 'pos_mm')} rot {_fmt(row[letter]['horizon'], 'rot_deg')} "
                f"1 pos {_fmt(row[letter]['onestep'], 'pos_mm')} rot {_fmt(row[letter]['onestep'], 'rot_deg')}",
                flush=True,
            )
        val, zarr_path = sc._train_val(cfg, norm)
        demos, grouped, n_win = sc._train_rows(val, zarr_path)
        parts = [
            _both(_predict(policy, val, rows, device), demos[epi], train_mesh)
            for epi, rows in grouped.items()
        ]
        cat = [np.concatenate([part[i] for part in parts]) for i in range(4)]
        row["train"] = {
            "n": int(len(cat[0])),
            "windows": n_win,
            "horizon": _public(cat[0], cat[1]),
            "onestep": _public(cat[2], cat[3]),
        }
        print(
            f"[fv2] {which} train n {row['train']['n']} "
            f"H pos {_fmt(row['train']['horizon'], 'pos_mm')} rot {_fmt(row['train']['horizon'], 'rot_deg')} "
            f"1 pos {_fmt(row['train']['onestep'], 'pos_mm')} rot {_fmt(row['train']['onestep'], 'rot_deg')}",
            flush=True,
        )
        report["field_v2"].append(row)
        del policy, val, demos, grouped
        torch.cuda.empty_cache()
    OUT.write_text(json.dumps(report, indent=2))
    print(f"[fv2] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
