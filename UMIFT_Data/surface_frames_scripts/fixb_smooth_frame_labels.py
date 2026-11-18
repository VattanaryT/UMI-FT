#!/usr/bin/env python3
"""Relabel October r4vt with the smooth orientation field. New zarr only.

Does not write the source zarr, r1rel, fixb_field.npz, or tag trajectories.
November recordings are not in the source tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

from diffusion_policy.common.sampler_umift import (
    filter_rgb_ids_without_action_gaps,
    sparse_action_indices_from_rgb_timestamps,
)
from fixb_build_datasets import MOLD_KEYS, link_tree, replace_array
from fixb_field import FixBMoldField
from fixb_labels import (
    delta_R_from_field_tool,
    fixb_pose7_command,
    interp_rows,
    next_control_times,
    pose7_to_Rp,
    rot_angle_deg,
    smooth_positions,
    so3_to_rot6d_rows,
    tangent_steps,
)
from fixb_smooth_field import SmoothFrameField, build_smooth_frames
from mold_aware_runtime import integrate_tcp_foot, rot6d_rows_to_R

SRC = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_frame_continuity_causal/acp_replay_buffer.zarr"
)
R1REL = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r1rel_world/acp_replay_buffer.zarr"
)
FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
SIDE = FIELD.with_name("fixb_smooth_frames.npz")
OTHER_SIDE = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb/shared/fixb_smooth_frames.npz"
)
OUT = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_smooth_frame"
)
LONG_DT = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/long_dt_gt_100ms.npz"
)
NORM = Path(
    "/home/boeing/umi_ft_ws/worktrees/UMI-FT-smooth_frame/UMIFT_Data/surface_frames_scripts/"
    "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_exbad.pkl"
)
EXCLUDE = {35, 44}
ACTION_H = 64
ACTION_DS = 1
QUERY_STRIDE = 2


def _geodesic_deg(R: np.ndarray) -> np.ndarray:
    tr = np.clip((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def _stats(angle: np.ndarray) -> dict:
    a = np.asarray(angle, dtype=np.float64).reshape(-1)
    return {
        "n": int(len(a)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()) if len(a) else 0.0,
    }


def _kept_robot(ep, ep_id: int, long_idx: dict) -> np.ndarray:
    robot_t = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rgb_t = np.asarray(ep["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    wrench_t = np.asarray(ep["wrench_time_stamps_0"], dtype=np.float64).reshape(-1)
    start_t = max(float(rgb_t[16 * 4 + 1]), float(robot_t[51 * 16 + 1]), float(wrench_t[118 + 1]))
    end_t = min(float(rgb_t[-1]), float(robot_t[-1]), float(wrench_t[-1]))
    first = int(np.searchsorted(rgb_t, start_t, side="left"))
    last = int(np.searchsorted(rgb_t, end_t, side="right") - 1)
    ids = np.arange(first, last + 1, dtype=np.int64)[::QUERY_STRIDE]
    kept, _ = filter_rgb_ids_without_action_gaps(ids, robot_t, rgb_t, ACTION_H, ACTION_DS, 3.0)
    bad = np.zeros(len(robot_t), dtype=bool)
    if "bad_span_0" in ep:
        bad |= np.asarray(ep["bad_span_0"]).reshape(-1)[: len(bad)] > 0
    for j in long_idx.get(ep_id, ()):
        if 0 <= j < len(bad):
            bad[j] = True
    use = np.zeros(len(robot_t), dtype=bool)
    for rgb_id in kept:
        idx, _ = sparse_action_indices_from_rgb_timestamps(robot_t, rgb_t, int(rgb_id), ACTION_H, ACTION_DS)
        if len(idx) == 0 or np.any(bad[np.clip(idx, 0, len(bad) - 1)]):
            continue
        use[np.clip(idx, 0, len(use) - 1)] = True
    return use


def _sensitivity(query_R, points: np.ndarray, step: float = 0.005) -> np.ndarray:
    worst = np.zeros(len(points))
    base = query_R(points)
    for sign in (1.0, -1.0):
        for axis in range(3):
            q = np.array(points, copy=True)
            q[:, axis] += sign * step
            worst = np.maximum(worst, rot_angle_deg(base, query_R(q)))
    return worst


def _chunked_sensitivity(query_R, points: np.ndarray, chunk: int = 8000) -> np.ndarray:
    out = np.zeros(len(points))
    for i in range(0, len(points), chunk):
        out[i : i + chunk] = _sensitivity(query_R, points[i : i + chunk])
        print(f"  sensitivity {min(i + chunk, len(points))}/{len(points)}", flush=True)
    return out


def _gt(field, p, R_tool, x_label, v, dn, dR_r6, dt) -> dict:
    n = len(p)
    H = 16
    if n <= H:
        return {}
    starts = np.linspace(0, n - H - 1, 8).astype(int)
    angs = []
    feet = []
    for i in starts:
        sl = slice(int(i), int(i) + H)
        dR = rot6d_rows_to_R(dR_r6[sl])
        dec = integrate_tcp_foot(field, p[int(i)], v[sl], dn[sl], dt[sl], hold_face=False)
        R_d = np.einsum("hij,hjk->hik", dec["R"], dR)
        angs.append(rot_angle_deg(R_d, R_tool[sl]))
        feet.append(np.linalg.norm(dec["x_s"] - x_label[sl], axis=1) * 1e3)
    ang = np.concatenate(angs)
    foot = np.concatenate(feet)
    return {
        "ori_median_deg": float(np.median(ang)),
        "ori_max_deg": float(ang.max()),
        "foot_median_mm": float(np.median(foot)),
        "foot_max_mm": float(foot.max()),
    }


def _normalizer_check(v_dn: np.ndarray) -> dict:
    """Dims 0:3 are v_par and δn. ΔR is not in this 21-d pickle."""
    from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

    with NORM.open("rb") as f:
        saved = pickle.load(f)
    src = saved.params_dict["action"]
    lo = src["input_stats"]["min"].detach().float().cpu().reshape(-1).numpy()
    hi = src["input_stats"]["max"].detach().float().cpu().reshape(-1).numpy()
    got_lo = np.asarray(v_dn.min(0), dtype=np.float64)
    got_hi = np.asarray(v_dn.max(0), dtype=np.float64)
    outside = (got_lo < lo[:3] - 1e-5) | (got_hi > hi[:3] + 1e-5)
    report = {
        "pickle": str(NORM),
        "pickle_min_0_3": lo[:3].tolist(),
        "pickle_max_0_3": hi[:3].tolist(),
        "new_min_0_3": got_lo.tolist(),
        "new_max_0_3": got_hi.tolist(),
        "outside": bool(np.any(outside)),
        "note": "r4 training pins ΔR dims 21:27 to scale 1. The pickle is 21-d and has no ΔR range.",
    }
    if not np.any(outside):
        report["refit"] = None
        return report
    span = np.maximum(got_hi - got_lo, 1e-4)
    scale = src["scale"].detach().float().cpu().reshape(-1).numpy().copy()
    offset = src["offset"].detach().float().cpu().reshape(-1).numpy().copy()
    scale[:3] = 2.0 / span
    offset[:3] = -1.0 - scale[:3] * got_lo
    stats = {
        "min": src["input_stats"]["min"].detach().float().cpu().reshape(-1).numpy().copy(),
        "max": src["input_stats"]["max"].detach().float().cpu().reshape(-1).numpy().copy(),
        "mean": src["input_stats"]["mean"].detach().float().cpu().reshape(-1).numpy().copy(),
        "std": src["input_stats"]["std"].detach().float().cpu().reshape(-1).numpy().copy(),
    }
    stats["min"][:3] = got_lo
    stats["max"][:3] = got_hi
    stats["mean"][:3] = v_dn.mean(0)
    stats["std"][:3] = v_dn.std(0)
    np_dtype = np.float32 if src["scale"].dtype == torch.float32 else np.float64
    shape = tuple(src["scale"].shape)
    pack = lambda v: np.asarray(v, dtype=np_dtype).reshape(shape)
    lin = LinearNormalizer()
    lin["action"] = SingleFieldLinearNormalizer.create_manual(
        scale=pack(scale),
        offset=pack(offset),
        input_stats_dict={k: pack(v) for k, v in stats.items()},
    )
    out = NORM.with_name("oct_v35_h4s_r4vt_action_normalizer_minmax_smooth.pkl")
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    with out.open("wb") as f:
        pickle.dump(lin, f)
    report["refit"] = str(out)
    report["refit_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    return report


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    t0 = time.time()
    src = zarr.open(str(SRC), "r")
    rel = zarr.open(str(R1REL), "r")["data"]
    keys = sorted((k for k in src["data"].keys() if str(k).startswith("episode_")), key=lambda s: int(s.split("_")[-1]))
    ids = [int(k.split("_")[-1]) for k in keys]
    if ids != list(range(50)):
        raise SystemExit(f"expected October episodes 0-49, found {ids[:5]}...{ids[-3:]}")
    long_blob = np.load(LONG_DT)
    long_idx: dict[int, set[int]] = {}
    for ep, idx in zip(long_blob["flagged_episode"], long_blob["flagged_index"]):
        long_idx.setdefault(int(ep), set()).add(int(idx))

    bare = FixBMoldField(FIELD)
    positions = []
    tool_x = []
    for key, ep_id in zip(keys, ids):
        if ep_id in EXCLUDE:
            continue
        R, p = pose7_to_Rp(src["data"][key]["ts_pose_fb_0"][::20])
        x_s, vid, w = bare.closest(p)
        n = np.einsum("qk,qkd->qd", w, bare.normals[vid])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        sd = np.abs(np.sum((p - x_s) * n, axis=1))
        keep = sd < 0.03
        positions.append(p[keep])
        tool_x.append(R[keep, :, 0])
    info = build_smooth_frames(
        FIELD,
        SIDE,
        tangent_samples=np.concatenate(positions),
        tool_x=np.concatenate(tool_x),
    )
    if not OTHER_SIDE.exists():
        os.link(SIDE, OTHER_SIDE)
    print("[field]", json.dumps(info), flush=True)
    field = SmoothFrameField(FIELD, SIDE)

    print("[sensitivity] vertices", len(field.vertices), flush=True)
    sens_before = _chunked_sensitivity(lambda p: bare.query(p)["R"], field.vertices)
    sens_after = _chunked_sensitivity(lambda p: field.query(p)["R"], field.vertices)

    curves = {}
    dR_before = []
    dR_after = []
    train_vd = []
    gt = {}
    built = []
    roundtrip = []
    for key, ep_id in zip(keys, ids):
        ep = src["data"][key]
        robot_t = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        rgb_t = np.asarray(ep["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
        R_tool, p = pose7_to_Rp(ep["ts_pose_fb_0"])
        p_s = smooth_positions(robot_t, p, 0.1)
        q = field.query(p_s, with_curvature=True)
        R = np.asarray(q["R"], dtype=np.float64)
        x = np.asarray(q["x_s"], dtype=np.float64)
        n_out = -R[:, :, 2]
        t_next = next_control_times(robot_t, rgb_t)
        dt = t_next - robot_t
        x_next = interp_rows(t_next, robot_t, x)
        v = tangent_steps(field, x, R, x_next)
        ok = dt > 1e-6
        v[ok] /= dt[ok, None]
        v[~ok] = 0.0
        v = np.clip(v, -1.0, 1.0)
        p_vt = pose7_to_Rp(rel[key]["ts_pose_virtual_target_0"])[1]
        m = min(len(p_vt), len(x))
        dn = np.sum((p_vt[:m] - x[:m]) * n_out[:m], axis=1).astype(np.float32)
        if m < len(x):
            pad = np.repeat(dn[-1:], len(x) - m)
            dn = np.concatenate([dn, pad])
        dR = delta_R_from_field_tool(R, R_tool)
        back = np.einsum("nij,njk->nik", R, dR)
        roundtrip.append(float(_geodesic_deg(np.einsum("nji,njk->nik", back, R_tool)).max()))
        r6 = so3_to_rot6d_rows(dR).astype(np.float32)
        cmd = fixb_pose7_command(v, dn).astype(np.float32)
        R_old = np.asarray(ep["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        old_r6 = np.asarray(ep["fixb_delta_R_rot6d_0"], dtype=np.float64).reshape(-1, 6)
        use = _kept_robot(ep, ep_id, long_idx) if ep_id not in EXCLUDE else np.zeros(len(R), dtype=bool)
        if ep_id not in EXCLUDE:
            dR_before.append(_geodesic_deg(rot6d_rows_to_R(old_r6[use])))
            dR_after.append(_geodesic_deg(dR[use]))
            train_vd.append(np.concatenate([v[use], dn[use, None]], axis=1))
        ang = rot_angle_deg(R_old, R)
        at = int(np.argmax(ang))
        ndot = np.clip(np.sum(R_old[:, :, 2] * n_out, axis=1), -1.0, 1.0)
        nang = np.degrees(np.arccos(ndot))
        curves[ep_id] = {
            "p50": float(np.median(ang)),
            "p95": float(np.percentile(ang, 95)),
            "max": float(ang.max()),
            "t_at_max_s": float(robot_t[at] - robot_t[0]),
            "np50": float(np.median(nang)),
            "np95": float(np.percentile(nang, 95)),
            "nmax": float(nang.max()),
            "t": (robot_t[::20] - robot_t[0]).astype(np.float32),
            "ang": ang[::20].astype(np.float32),
            "nang": nang[::20].astype(np.float32),
        }
        pack = {
            "mold_x_s_0": x.astype(np.float32),
            "mold_R_field_0": R.reshape(-1, 9).astype(np.float32),
            "mold_R_field_rot6d_0": so3_to_rot6d_rows(R).astype(np.float32),
            "mold_normal_0": n_out.astype(np.float32),
            "mold_signed_distance_0": np.asarray(q["sd"], dtype=np.float32).reshape(-1, 1),
            "mold_curvature_0": np.asarray(q["curv"], dtype=np.float32),
            "fixb_v_par_0": v.astype(np.float32),
            "fixb_delta_n_0": dn.reshape(-1, 1).astype(np.float32),
            "fixb_dt_next_0": dt.reshape(-1, 1).astype(np.float32),
        }
        built.append({"pack": pack, "cmd": cmd, "dR": r6})
        if ep_id in (0, 9):
            gt[str(ep_id)] = _gt(field, p_s, R_tool, x, v, dn, r6, dt)
        print(f"[{ep_id:02d}] {time.time() - t0:.0f}s frame p95 {curves[ep_id]['p95']:.1f}°", flush=True)

    focus = [0, 9]
    rest = sorted((i for i in curves if i not in focus), key=lambda i: curves[i]["p95"], reverse=True)
    focus.append(int(rest[0]))
    if str(focus[2]) not in gt:
        # Recompute that episode's GT from the stored pack. Poses are re-read.
        ep_id = focus[2]
        ep = src["data"][f"episode_{ep_id}"]
        robot_t = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        R_tool, p = pose7_to_Rp(ep["ts_pose_fb_0"])
        p_s = smooth_positions(robot_t, p, 0.1)
        pack = built[keys.index(f"episode_{ep_id}")]["pack"]
        gt[str(ep_id)] = _gt(
            field,
            p_s,
            R_tool,
            np.asarray(pack["mold_x_s_0"]),
            np.asarray(pack["fixb_v_par_0"]),
            np.asarray(pack["fixb_delta_n_0"]).reshape(-1),
            built[keys.index(f"episode_{ep_id}")]["dR"],
            np.asarray(pack["fixb_dt_next_0"]).reshape(-1),
        )

    norm = _normalizer_check(np.concatenate(train_vd))
    print("[normalizer]", json.dumps(norm), flush=True)

    _plot(field.vertices, sens_before, sens_after, curves, focus, OUT)
    # plot dir is created inside _plot before the zarr write; OUT must not exist yet.
    # _plot creates OUT. link_tree then fills the zarr under it.
    print("[write]", flush=True)
    link_tree(SRC, OUT / "acp_replay_buffer.zarr")
    root = zarr.open(str(OUT / "acp_replay_buffer.zarr"), "a")
    root.attrs["label_version"] = "r4vt_smooth_frame"
    root.attrs["october_episodes_only"] = 50
    root.attrs["november_excluded"] = "source tree is episodes 0-49"
    root.attrs["smooth_frames"] = str(SIDE)
    for key, item in zip(keys, built):
        ep = root["data"][key]
        for name in MOLD_KEYS:
            replace_array(ep, name, item["pack"][name], ep[name])
        replace_array(ep, "ts_pose_command_0", item["cmd"], ep["ts_pose_command_0"])
        replace_array(ep, "fixb_delta_R_rot6d_0", item["dR"], ep["fixb_delta_R_rot6d_0"])
        ep.attrs["label_version"] = "r4vt_smooth_frame"

    x = field.vertices[:, 0]
    interior = (x > x.min() + 0.04) & (x < x.max() - 0.04)
    dR_b = np.concatenate(dR_before)
    dR_a = np.concatenate(dR_after)
    report = {
        "field": info,
        "sensitivity_vertex_deg": {"before": _stats(sens_before), "after": _stats(sens_after)},
        "sensitivity_interior_40mm_deg": {
            "before": _stats(sens_before[interior]),
            "after": _stats(sens_after[interior]),
        },
        "delta_R_deg": {"before": _stats(dR_b), "after": _stats(dR_a)},
        "delta_R_modes": {
            "before": _modes(dR_b),
            "after": _modes(dR_a),
        },
        "roundtrip_max_deg": float(max(roundtrip)),
        "frame_angle_deg": {
            str(i): {k: curves[i][k] for k in ("p50", "p95", "max", "t_at_max_s", "np50", "np95", "nmax")}
            for i in focus
        },
        "gt_decode": gt,
        "normalizer": norm,
        "face_hold": "off for this field; 10 mm foot cap and 3° R_d slew stay",
        "zarr": str(OUT / "acp_replay_buffer.zarr"),
    }
    (OUT / "smooth_frame_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def _modes(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    return {"frac_lt_45": float(np.mean(a < 45.0)), "frac_gt_135": float(np.mean(a > 135.0))}


def _plot(vertices, before, after, curves, focus, out: Path) -> None:
    out.mkdir(parents=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    vmax = float(max(before.max(), after.max()))
    xy = vertices[:, [0, 2]]
    step = max(1, len(vertices) // 20000)
    for ax, val, title in zip(axes, (before, after), ("closest face", "smooth field")):
        sc = ax.scatter(
            xy[::step, 0],
            xy[::step, 1],
            c=val[::step],
            s=2,
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            linewidths=0,
        )
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nmax {val.max():.1f}°")
        ax.set_xlabel("fixture x [m]")
        ax.set_ylabel("fixture z [m]")
    fig.colorbar(sc, ax=axes, label="max frame rotation under ±5 mm [deg]", shrink=0.85)
    fig.savefig(out / "frame_sensitivity.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, constrained_layout=True)
    for ep_id in focus:
        c = curves[ep_id]
        axes[0].plot(c["t"], c["ang"], label=f"episode {ep_id}", lw=1.0)
        axes[1].plot(c["t"], c["nang"], label=f"episode {ep_id}", lw=1.0)
    axes[0].set_ylabel("full frame [deg]")
    axes[1].set_ylabel("outward normal [deg]")
    axes[1].set_xlabel("time in episode [s]")
    axes[0].legend(frameon=False)
    fig.savefig(out / "frame_angle_trajectories.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
