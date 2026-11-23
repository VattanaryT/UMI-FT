#!/usr/bin/env python3
"""Test-time flip of the thin tangent (t1, t2) = (-t1, -t2). Does not save anything."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "viewer"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402

import contact_wall as cw  # noqa: E402
import foot_overlay  # noqa: E402
import heldout  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import so3_to_rot6d_rows  # noqa: E402
from mold_aware_runtime import rot6d_rows_to_R  # noqa: E402
from run_policy_open_loop_val_export import load_policy_and_cfg  # noqa: E402

import compare_r1rel as fair  # noqa: E402

TRAIN_FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
TRAIN_ZARR = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_causal_foot/acp_replay_buffer.zarr"
)
S = np.diag([-1.0, -1.0, 1.0])
MODELS = (
    ("causal e199", cw.MODELS[0][2]),
    ("k3000 e100", cw.TRAIN / "oct-v35-h4s-r4vt-causal-foot-carry-k3000_e200_bs64_valtopk_43472139/checkpoints/latest.ckpt"),
)


def _check() -> None:
    R = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    back = rot6d_rows_to_R(so3_to_rot6d_rows((R @ S).reshape(1, 3, 3)))[0]
    if not np.allclose(back, R @ S, atol=1e-8):
        raise RuntimeError("tangent flip does not round-trip")


def _t1x(frames) -> np.ndarray:
    R = np.asarray(frames, dtype=np.float64).reshape(-1, 3, 3)
    return R[:, 0, 0]


def _sign_report(name: str, t1x: np.ndarray) -> None:
    t1x = np.asarray(t1x, dtype=np.float64).reshape(-1)
    print(
        f"[sign] {name} n={len(t1x)} "
        f"t1x>0 {(t1x > 0).mean():.4f} t1x<0 {(t1x < 0).mean():.4f} "
        f"min {t1x.min():.3f} max {t1x.max():.3f}",
        flush=True,
    )


def _field_sign(path: Path) -> None:
    field = np.load(path)
    frames = field["frames"]
    apron = np.asarray(field["is_apron"]).reshape(-1).astype(bool) if "is_apron" in field else None
    _sign_report(path.name, _t1x(frames))
    if apron is not None and apron.any() and (~apron).any():
        _sign_report(path.name + " body", _t1x(frames)[~apron])
        _sign_report(path.name + " apron", _t1x(frames)[apron])
    centered = field["vertices"] - field["vertices"].mean(axis=0)
    axis = np.linalg.svd(centered, full_matrices=False)[2][0]
    t1 = frames.reshape(-1, 3, 3)[:, :, 0]
    print(
        f"[sign] {path.name} pca_axis {np.round(axis, 3).tolist()} "
        f"mean t1·axis {float(np.mean(t1 @ axis)):.3f}",
        flush=True,
    )


def _zarr_sign(path: Path) -> None:
    root = zarr.open(path, "r")["data"]
    parts = [_t1x(np.asarray(root[key]["mold_R_field_0"])) for key in root.keys()]
    _sign_report(path.parent.name, np.concatenate(parts))


def _keep(demo, rows):
    i0 = np.array([int(index[0]) for _idx, _rgb, index in rows], dtype=np.int64)
    _foot, face, _z = foot_overlay._project(*foot_overlay._load(), demo["tool_p"][i0])
    kept = []
    for row, kind in zip(rows, face):
        _idx, _rgb, index = row
        j = heldout._nearest(demo["wrench_t"], demo["robot_t"][index])
        mag = np.linalg.norm(demo["smooth"][j], axis=1)
        if str(kind) not in ("cap", "back") and float(mag[0]) >= 8.0 and float(np.mean(mag >= 5.0)) >= 0.80:
            kept.append(row)
    return kept


def _flip_obs(dataset) -> None:
    for ep in dataset.replay_buffer["data"].values():
        geom = ep["obs"]["robot0_mold_geom"]
        R = rot6d_rows_to_R(np.asarray(geom[:, 4:10], dtype=np.float64))
        geom[:, 4:10] = so3_to_rot6d_rows(R @ S).astype(geom.dtype)


def _flip_demo(demo) -> None:
    demo["frame"] = np.einsum("nij,jk->nik", demo["frame"], S)


def main() -> None:
    _check()
    _field_sign(heldout.WALL)
    _field_sign(TRAIN_FIELD)
    _zarr_sign(TRAIN_ZARR)
    field = FixBMoldField(str(heldout.WALL))
    field.frames = np.einsum("vij,jk->vik", field.frames, S)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recordings, axis = cw._recordings(field)
    jobs = []
    cache = {}
    for name, path in MODELS:
        policy, cfg = load_policy_and_cfg(str(path), device)
        norm, _dim, decode, _sha = heldout._norm_of(cfg, "auto")
        if decode != "causal":
            raise SystemExit(f"{name} decodes as {decode}")
        jobs.append((name, policy, cfg, norm, fair.ckpt_epoch(path)))
    cfg, norm = jobs[0][2], jobs[0][3]
    for letter, rec in recordings.items():
        dataset, rows = cw._rows_for(cfg, norm, rec, cache)
        kept = _keep(rec["demo"], rows)
        _flip_obs(dataset)
        _flip_demo(rec["demo"])
        i0 = np.array([int(index[0]) for _idx, _rgb, index in kept])
        _sign_report(f"kept {letter} after flip", _t1x(rec["demo"]["frame"][i0]))
        print(f"[flip] {letter} n={len(kept)}", flush=True)
        for name, policy, _cfg, _norm, epoch in jobs:
            scored = heldout._score_one(
                "causal", policy, dataset, field, rec["demo"], kept, device, axis
            )
            print(
                f"[flip] {letter} {name} e{epoch} "
                f"pos {scored['pos_mm']} dn {scored['dn_mm']} rot {scored['rot_deg']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
