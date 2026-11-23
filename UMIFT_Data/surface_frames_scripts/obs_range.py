#!/usr/bin/env python3
"""Obs range: kept thin queries vs 200 training windows. New file, read-only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
HELD = Path(
    "/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/"
    "UMIFT_Data/surface_frames_scripts/eval/thin_heldout"
)
sys.path[:0] = [str(SCRIPTS), str(HELD), str(HELD.parent), str(ROOT / "PyriteML"), str(ROOT)]

import zarr  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import pose7_to_Rp, so3_to_rot6d_rows  # noqa: E402
from fixb_smooth_frame_labels import ACTION_DS, ACTION_H, EXCLUDE, LONG_DT  # noqa: E402
from fixb_tcp_track_clean import _queries  # noqa: E402
from diffusion_policy.common.sampler_umift import sparse_action_indices_from_rgb_timestamps  # noqa: E402

import field_v2  # noqa: E402

TRAIN = field_v2.OUT / "acp_replay_buffer.zarr"
CAUSAL = Path(
    "/mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_causal_foot/"
    "oct-v35-h4s-r4vt-causal-foot_e200_bs64_valtopk_43390127/checkpoints/latest.ckpt"
)
OUTWARD = np.diag([1.0, -1.0, -1.0])
ROT = [f"r{i}" for i in range(6)]
WRENCH_W = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
WRENCH_F = ["Ft1", "Ft2", "Fn", "Tt1", "Tt2", "Tn"]


def _check() -> None:
    eye = np.eye(3)
    got = so3_to_rot6d_rows(eye[None])[0]
    if not np.allclose(got, [1, 0, 0, 0, 1, 0]):
        raise RuntimeError("rot6d of identity")
    force = np.array([3.0, 0.0, 0.0])
    turned = eye @ force
    if not np.allclose(turned, force):
        raise RuntimeError("world wrench")


def _nearest(times: np.ndarray, query: float) -> int:
    j = int(np.clip(np.searchsorted(times, query), 1, len(times) - 1))
    if abs(query - times[j - 1]) <= abs(query - times[j]):
        return j - 1
    return j


def _wrench_key(ep) -> str:
    for key in ("wrench_time_stamps_left_0", "wrench_time_stamps_0"):
        if key in ep:
            return key
    raise KeyError("no wrench timestamps")


def _pack(tool_R: np.ndarray, wrench: np.ndarray, foot_R: np.ndarray) -> dict[str, np.ndarray]:
    """foot_R columns are t1, t2, outward normal. wrench is TCP 6D."""
    world = np.concatenate([tool_R @ wrench[:3], tool_R @ wrench[3:]])
    foot = np.concatenate([foot_R.T @ wrench[:3], foot_R.T @ wrench[3:]])
    return {
        "rot_world": so3_to_rot6d_rows(tool_R[None])[0],
        "wrench_world": world,
        "rot_foot": so3_to_rot6d_rows((foot_R.T @ tool_R)[None])[0],
        "wrench_foot": foot,
    }


def _at(ep, robot_i: int, foot_R: np.ndarray) -> dict[str, np.ndarray]:
    tool_R, _p = pose7_to_Rp(np.asarray(ep["ts_pose_fb_0"][robot_i : robot_i + 1], dtype=np.float64))
    robot_t = float(np.asarray(ep["robot_time_stamps_0"]).reshape(-1)[robot_i])
    wrench_t = np.asarray(ep[_wrench_key(ep)]).reshape(-1)
    wrench = np.asarray(ep["wrench_left_0"][_nearest(wrench_t, robot_t)], dtype=np.float64)
    return _pack(tool_R[0], wrench, foot_R)


def _train_windows() -> list[dict[str, np.ndarray]]:
    root = zarr.open(str(TRAIN), "r")["data"]
    blob = np.load(LONG_DT)
    long_idx: dict[int, set[int]] = {}
    for ep, idx in zip(blob["flagged_episode"], blob["flagged_index"]):
        long_idx.setdefault(int(ep), set()).add(int(idx))
    pairs = []
    for ep_id in range(50):
        if ep_id in EXCLUDE:
            continue
        ep = root[f"episode_{ep_id}"]
        robot_t, rgb_t, kept, bad = _queries(ep, ep_id, long_idx)
        for rgb_id in kept:
            index, _ = sparse_action_indices_from_rgb_timestamps(
                robot_t, rgb_t, int(rgb_id), ACTION_H, ACTION_DS
            )
            if len(index) == 0 or np.any(bad[np.clip(index, 0, len(bad) - 1)]):
                continue
            pairs.append((ep_id, int(index[0])))
    take = np.unique(np.linspace(0, len(pairs) - 1, 200).astype(int))
    rows = []
    for i in take:
        ep_id, robot_i = pairs[i]
        ep = root[f"episode_{ep_id}"]
        inward = np.asarray(ep["mold_R_field_0"][robot_i], dtype=np.float64).reshape(3, 3)
        rows.append(_at(ep, robot_i, inward @ OUTWARD))
    print(f"[obs] training windows {len(rows)} of {len(pairs)}", flush=True)
    return rows


def _thin_field():
    field = FixBMoldField(str(field_v2.THIN_FIELD))
    _origin, axes = field_v2.mold_frame(field.vertices, field_v2._load_field(field_v2.THIN_FIELD)[2])
    field.frames = field_v2.reorient(field.frames, axes)
    return field


def _thin_rows(field) -> list[tuple[str, dict[str, np.ndarray]]]:
    kept = json.loads((HELD / "contact_wall_v2/kept.json").read_text())
    out = []
    for letter, folder, key in (("C", "C_first", "0"), ("A", "A_full", "1")):
        ep = zarr.open(str(HELD / folder / "replay.zarr"), "r")["data"]["episode_0"]
        robot_t = np.asarray(ep["robot_time_stamps_0"]).reshape(-1)
        rgb0 = float(np.asarray(ep["rgb_time_stamps_0"]).reshape(-1)[0])
        foot = np.asarray(ep["mold_x_s_0"], dtype=np.float64)
        times = [round(float(t), 3) for t, keep in zip(kept[key]["t"], kept[key]["keep"]) if keep]
        for t in times:
            robot_i = int(np.argmin(np.abs(robot_t - rgb0 - t)))
            hit = field.query(foot[robot_i : robot_i + 1])
            out.append((letter, _at(ep, robot_i, hit["R"][0])))
    if [n for n, _ in out].count("C") != 9 or [n for n, _ in out].count("A") != 94:
        raise SystemExit("kept counts are not 9 and 94")
    return out


def _stack(rows: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    return np.stack([row[key] for row in rows])


def _band(x: np.ndarray) -> dict:
    return {
        "min": np.round(x.min(0), 3),
        "p1": np.round(np.percentile(x, 1, axis=0), 3),
        "median": np.round(np.median(x, axis=0), 3),
        "p99": np.round(np.percentile(x, 99, axis=0), 3),
        "max": np.round(x.max(0), 3),
    }


def _report(title: str, names: list[str], train: np.ndarray, thin: dict[str, np.ndarray]) -> np.ndarray:
    band = _band(train)
    print(f"\n[obs] {title}", flush=True)
    header = f"{'dim':6} {'train min':>10} {'p1':>10} {'median':>10} {'p99':>10} {'max':>10}"
    for letter in ("C", "A"):
        header += f"  {letter+' min':>10} {letter+' med':>10} {letter+' max':>10}"
    print(header, flush=True)
    flagged = []
    pooled = np.concatenate([thin["C"], thin["A"]])
    for i, name in enumerate(names):
        line = (
            f"{name:6} {band['min'][i]:10.3f} {band['p1'][i]:10.3f} {band['median'][i]:10.3f} "
            f"{band['p99'][i]:10.3f} {band['max'][i]:10.3f}"
        )
        for letter in ("C", "A"):
            col = thin[letter][:, i]
            line += f"  {col.min():10.3f} {np.median(col):10.3f} {col.max():10.3f}"
            if col.min() < band["p1"][i] or col.max() > band["p99"][i]:
                flagged.append((letter, name, float(col.min()), float(col.max()), float(band["p1"][i]), float(band["p99"][i])))
        outside = np.mean((pooled[:, i] < band["p1"][i]) | (pooled[:, i] > band["p99"][i]))
        print(f"{line}  outside {outside:.2f}", flush=True)
    for letter, name, lo, hi, p1, p99 in flagged:
        print(f"[obs] FLAG {letter} {name} thin [{lo:.3f}, {hi:.3f}] train p1–p99 [{p1:.3f}, {p99:.3f}]", flush=True)
    if not flagged:
        print("[obs] no dim outside training p1–p99", flush=True)
    return np.stack([band["p1"], band["median"], band["p99"]])


def _z(x: np.ndarray, band: np.ndarray) -> np.ndarray:
    scale = np.maximum(band[2] - band[0], 1e-6)
    return (x - band[1]) / scale


def _corr(name: str, dist: np.ndarray, err: np.ndarray) -> None:
    pr, pp = pearsonr(dist, err)
    sr, sp = spearmanr(dist, err)
    print(
        f"[obs] corr {name}  pearson {pr:.3f} p {pp:.3g}  spearman {sr:.3f} p {sp:.3g}",
        flush=True,
    )


def _step1() -> dict[str, np.ndarray]:
    import heldout
    from run_policy_open_loop_val_export import load_policy_and_cfg

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, cfg = load_policy_and_cfg(str(CAUSAL), device)
    norm, _dim, decode, _sha = heldout._norm_of(cfg, "causal")
    kept = json.loads((HELD / "contact_wall_v2/kept.json").read_text())
    out = {}
    for letter, folder, key in (("C", "C_first", "0"), ("A", "A_full", "1")):
        path = HELD / folder / "replay.zarr"
        demo = heldout._demo(path, None)
        dataset = heldout._dataset(cfg, path, norm)
        rows = heldout._queries(dataset, demo)
        want = {round(float(t), 3) for t, keep in zip(kept[key]["t"], kept[key]["keep"]) if keep}
        chosen = []
        for row in rows:
            t = round(float(demo["robot_t"][int(row[2][0])] - demo["rgb_t"][0]), 3)
            if t in want:
                chosen.append(row)
        if len(chosen) != len(want):
            raise SystemExit(f"{letter} scored {len(chosen)} kept {len(want)}")
        vertices, _frames, apron = field_v2._load_field(field_v2.THIN_FIELD)
        axis = field_v2.mold_frame(vertices, apron)[1][:, 0]
        field = heldout.FixBMoldField(str(heldout.WALL))
        block = heldout._score_one(decode, policy, dataset, field, demo, chosen, device, axis)
        err = block["_rot"][:, 0]
        out[letter] = err
        print(f"[obs] causal e199 {letter} step1 rot mean {err.mean():.2f} median {np.median(err):.2f}", flush=True)
        del dataset
    del policy
    return out


def main() -> None:
    _check()
    train = _train_windows()
    thin_rows = _thin_rows(_thin_field())
    groups = (
        ("current tool rot6d, fixture", ROT, "rot_world"),
        ("current wrench, fixture = R_tool @ tcp", WRENCH_W, "wrench_world"),
        ("proposed tool rot6d, foot frame", ROT, "rot_foot"),
        ("proposed wrench, foot frame", WRENCH_F, "wrench_foot"),
    )
    bands = {}
    for title, names, key in groups:
        thin_k = {letter: _stack([row for name, row in thin_rows if name == letter], key) for letter in ("C", "A")}
        bands[key] = _report(title, names, _stack(train, key), thin_k)
    err = _step1()
    rot_x = np.concatenate([_stack([row for name, row in thin_rows if name == letter], "rot_world") for letter in ("C", "A")])
    wr_x = np.concatenate([_stack([row for name, row in thin_rows if name == letter], "wrench_world") for letter in ("C", "A")])
    err_x = np.concatenate([err["C"], err["A"]])
    rot_d = np.linalg.norm(_z(rot_x, bands["rot_world"]), axis=1)
    wr_d = np.linalg.norm(_z(wr_x, bands["wrench_world"]), axis=1)
    _corr("rot6d distance", rot_d, err_x)
    _corr("wrench distance", wr_d, err_x)
    _corr("rot+wrench distance", np.hypot(rot_d, wr_d), err_x)


if __name__ == "__main__":
    main()
