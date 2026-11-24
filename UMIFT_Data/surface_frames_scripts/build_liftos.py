#!/usr/bin/env python3
"""New zarr: v3r absolute rot6d, original v2n δn. Also the lift-index sidecar.

Does not modify r4vt_field_v2n_v3r_lift or any normalizer. Does not submit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS), str(ROOT / "PyriteML"), str(ROOT / "PyriteUtility"), str(ROOT)]

import zarr  # noqa: E402

from diffusion_policy.common.sampler_umift import (  # noqa: E402
    filter_rgb_ids_without_action_gaps,
    get_val_mask,
    sparse_action_indices_from_rgb_timestamps,
)
from field_v2 import smooth_force  # noqa: E402
from field_v2n_v3r_liftos_dataset import expand_indices  # noqa: E402
from fixb_build_datasets import link_tree, replace_array  # noqa: E402
from fixb_labels import pose7_to_Rp  # noqa: E402


def isolated(contact, time_s, hold_s=0.5):
    contact = np.asarray(contact, dtype=bool).reshape(-1)
    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    ends = np.flatnonzero(contact[:-1] & ~contact[1:])
    keep = []
    for end in ends:
        later = np.flatnonzero(contact[end + 1:])
        if later.size == 0 or float(time_s[end + 1 + int(later[0])] - time_s[end]) > hold_s:
            keep.append(int(end))
    return np.asarray(keep, dtype=np.int64)

DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
V2N = DATA / "r4vt_field_v2n/acp_replay_buffer.zarr"
V3R = DATA / "r4vt_field_v3r/acp_replay_buffer.zarr"
OUT = DATA / "r4vt_field_v2n_v3r_liftos/acp_replay_buffer.zarr"
FORBIDDEN = DATA / "r4vt_field_v2n_v3r_lift/acp_replay_buffer.zarr"
NPZ = SCRIPTS / "assets/lift_isolated_ends.npz"
LONG_DT = DATA / "shared/long_dt_gt_100ms.npz"
HELD = {9, 12, 28, 37, 41}
EXCLUDE = {35, 44}
RGB_H, RGB_DS = 16, 4
ROBOT_H, ROBOT_DS = 51, 16
WRENCH_H, WRENCH_DS = 118, 1
ACTION_H, ACTION_DS = 64, 1
QUERY_DS = 2


def _fn(group):
    robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rotation, _ = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
    frame = np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
    n = min(len(robot_t), len(rotation), len(frame))
    robot_t, rotation, frame = robot_t[:n], rotation[:n], frame[:n]
    force = smooth_force(group, robot_t)
    world = np.einsum("nij,nj->ni", rotation, force[:, :3])
    outward = -frame[:, :, 2]
    return robot_t, np.einsum("ni,ni->n", world, outward)


def _queries(group):
    rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    wrench_t = np.asarray(group["wrench_time_stamps_0"], dtype=np.float64).reshape(-1)
    start = max(
        float(rgb_t[RGB_H * RGB_DS + 1]),
        float(robot_t[ROBOT_H * ROBOT_DS + 1]),
        float(wrench_t[WRENCH_H * WRENCH_DS + 1]),
    )
    end = min(float(rgb_t[-1]), float(robot_t[-1]), float(wrench_t[-1]))
    first = int(np.searchsorted(rgb_t, start, side="left"))
    last = int(np.searchsorted(rgb_t, end, side="right") - 1)
    candidate = first + np.arange(max(last - first, 0))
    candidate = candidate[::QUERY_DS]
    kept, _ = filter_rgb_ids_without_action_gaps(
        candidate, robot_t, rgb_t, ACTION_H, ACTION_DS, 3.0
    )
    return kept


def _long_idx():
    blob = np.load(LONG_DT)
    out: dict[int, set[int]] = {}
    for epi, idx in zip(blob["flagged_episode"], blob["flagged_index"]):
        out.setdefault(int(epi), set()).add(int(idx))
    return out


def _count(root):
    """Reproduce the kept-train query list well enough to check the 348."""
    keys = list(root.keys())
    val = get_val_mask(len(keys), 0.1, 42)
    held = {int(keys[i].split("_")[-1]) for i in np.flatnonzero(val)}
    if held != HELD:
        raise SystemExit(f"val episodes {sorted(held)} != {sorted(HELD)}")
    long_idx = _long_idx()
    ends = {}
    pairs = []
    lift_flags = []
    for i, key in enumerate(keys):
        epi = int(key.split("_")[-1])
        group = root[key]
        t, fn = _fn(group)
        contact = np.abs(fn) >= 5.0
        ends[epi] = isolated(contact, t)
        if val[i] or epi in EXCLUDE:
            continue
        span = np.zeros(len(t), dtype=bool)
        if "bad_span_0" in group:
            raw = np.asarray(group["bad_span_0"]).reshape(-1)
            span[: min(len(t), len(raw))] = raw[: len(t)] > 0
        dt = np.zeros(len(t), dtype=bool)
        for j in long_idx.get(epi, ()):
            if 0 <= j < len(t):
                dt[j] = True
        rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
        for rgb in _queries(group):
            idx, _ = sparse_action_indices_from_rgb_timestamps(
                t, rgb_t, int(rgb), ACTION_H, ACTION_DS
            )
            if idx.size == 0:
                continue
            if np.any(span[idx]) or np.any(dt[idx]):
                continue
            pairs.append((epi, int(rgb)))
            lift_flags.append(bool(np.intersect1d(idx, ends[epi]).size))
    return pairs, np.asarray(lift_flags), ends


def _write_zarr(v2n, v3r):
    if OUT.exists() or OUT == FORBIDDEN:
        raise SystemExit(f"refusing to write {OUT}")
    if FORBIDDEN.exists() and OUT.resolve() == FORBIDDEN.resolve():
        raise SystemExit("refusing to touch the lift zarr")
    link_tree(V3R, OUT)
    root = zarr.open(str(OUT), "a")
    root.attrs["label_version"] = "field_v2n_v3r_liftos_v1"
    root.attrs["delta_n"] = "original field_v2n blend"
    root.attrs["delta_R"] = "absolute tool rot6d; training label is R_tcp(t).T @ R_tcp(t+k)"
    guard = np.asarray(v3r["data/episode_0/fixb_delta_n_0"][:8]).copy()
    for epi in range(50):
        src = v2n["data"][f"episode_{epi}"]
        dst = root["data"][f"episode_{epi}"]
        dn = np.asarray(src["fixb_delta_n_0"])
        if len(dn) != len(dst["fixb_delta_n_0"]):
            raise SystemExit(f"episode {epi} δn length {len(dn)} != {len(dst['fixb_delta_n_0'])}")
        cmd = np.array(dst["ts_pose_command_0"])
        src_cmd = np.asarray(src["ts_pose_command_0"])
        if cmd.shape != src_cmd.shape:
            raise SystemExit(f"episode {epi} command shape {cmd.shape} != {src_cmd.shape}")
        cmd[:, 2] = src_cmd[:, 2]
        replace_array(dst, "fixb_delta_n_0", dn, dst["fixb_delta_n_0"])
        replace_array(dst, "ts_pose_command_0", cmd, dst["ts_pose_command_0"])
    if not np.array_equal(np.asarray(v3r["data/episode_0/fixb_delta_n_0"][:8]), guard):
        raise SystemExit("source v3r δn changed")
    written = np.asarray(root["data/episode_0/fixb_delta_n_0"][:8])
    if not np.array_equal(written, np.asarray(v2n["data/episode_0/fixb_delta_n_0"][:8])):
        raise SystemExit("written δn is not the v2n label")


def main():
    if not NPZ.exists():
        v2n = zarr.open(str(V2N), "r")
        _pairs, flags, ends = _count(v2n["data"])
        flat_ep, flat_i = [], []
        for epi, idx in ends.items():
            flat_ep.extend([epi] * len(idx))
            flat_i.extend(int(i) for i in idx)
        NPZ.parent.mkdir(parents=True, exist_ok=True)
        np.savez(NPZ, episode=np.asarray(flat_ep), index=np.asarray(flat_i))
        _expanded, plan = expand_indices(list(range(len(flags))), flags, 0.15)
        print(
            f"[liftos] kept {len(flags)} lift {int(flags.sum())} "
            f"weight {plan['weight']:.4f} fraction {plan['fraction']:.4f} "
            f"copies {plan['base']}+{plan['extra']} epoch {plan['epoch']}",
            flush=True,
        )
    v2n = zarr.open(str(V2N), "r")
    v3r = zarr.open(str(V3R), "r")
    _write_zarr(v2n, v3r)
    print(f"[liftos] zarr {OUT}", flush=True)


if __name__ == "__main__":
    main()
