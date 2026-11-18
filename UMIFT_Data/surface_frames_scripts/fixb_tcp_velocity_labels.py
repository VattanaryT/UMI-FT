#!/usr/bin/env python3
"""Two offline replacements for the closest-point v1/v2 labels.

Option A (track): TCP velocity projected onto the smooth frame at a foot that
is searched near the previous foot.

Option B (dof): the same velocity, with the frame from a harmonic off-surface
field (Bilaloglu et al. 2026) instead of a foot.

Writes new zarr directories. Does not modify r4vt_smooth_frame, r1rel, the
field, or the normalizer pickle.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import zarr

from fixb_build_datasets import MOLD_KEYS, link_tree, replace_array
from fixb_diffused_field import build_diffused_field
from fixb_labels import (
    delta_R_from_field_tool,
    fixb_pose7_command,
    pose7_to_Rp,
    rot_angle_deg,
    so3_to_rot6d_rows,
)
from fixb_smooth_field import SmoothFrameField
from fixb_smooth_frame_labels import EXCLUDE, LONG_DT, NORM, _geodesic_deg, _kept_robot
from fixb_tcp_track import GATE_M, project_tcp_velocity, step_foot, track_local_feet
from mold_aware_runtime import rot6d_rows_to_R

ROOT = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
SMOOTH = ROOT / "r4vt_smooth_frame" / "acp_replay_buffer.zarr"
R1REL = ROOT / "r1rel_world" / "acp_replay_buffer.zarr"
FIELD = ROOT / "shared" / "fixb_field.npz"
SIDE = FIELD.with_name("fixb_smooth_frames.npz")
OUT_TRACK = ROOT / "r4vt_smooth_tcp_track"
OUT_DOF = ROOT / "r4vt_smooth_dof"
REPORT = ROOT / "r4vt_velocity_options_report.json"
STRIDE = 80
RAY_LIMIT_M = 0.08


def _query(field, p):
    q = field.query(np.asarray(p, dtype=np.float64))
    return q["x_s"].reshape(-1, 3), q["R"].reshape(-1, 3, 3)


def _dn(p_vt, x, n_out) -> np.ndarray:
    m = min(len(p_vt), len(x))
    dn = np.sum((p_vt[:m] - x[:m]) * n_out[:m], axis=1)
    if m < len(x):
        dn = np.concatenate([dn, np.repeat(dn[-1:], len(x) - m)])
    return dn.astype(np.float64)


def _ray_feet(scene, p: np.ndarray, inward: np.ndarray) -> tuple[np.ndarray, int]:
    """Surface point along the inward axis, then the outward axis if that misses."""
    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
    d = np.asarray(inward, dtype=np.float64).reshape(-1, 3)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)

    def cast(origin, direction):
        rays = np.concatenate([origin, direction], axis=1).astype(np.float32)
        ans = scene.cast_rays(o3d.core.Tensor(rays))
        return np.asarray(ans["t_hit"].numpy(), dtype=np.float64).reshape(-1)

    t = cast(p, d)
    miss = ~np.isfinite(t) | (t < 0.0) | (t > RAY_LIMIT_M)
    if np.any(miss):
        t2 = cast(p[miss], -d[miss])
        t = np.array(t, copy=True)
        d = np.array(d, copy=True)
        t[miss] = t2
        d[miss] *= -1.0
        miss = ~np.isfinite(t) | (t < 0.0) | (t > RAY_LIMIT_M)
    t_use = np.where(miss, 0.0, t)
    return p + t_use[:, None] * d, miss


def _motion_stats(v, x, dn) -> dict:
    speed = np.linalg.norm(np.asarray(v, dtype=np.float64).reshape(-1, 2), axis=1)
    jump = np.linalg.norm(np.diff(np.asarray(x, dtype=np.float64).reshape(-1, 3), axis=0), axis=1)
    ddn = np.abs(np.diff(np.asarray(dn, dtype=np.float64).reshape(-1)))
    return {
        "samples": int(len(speed)),
        "v_gt_0.5": int(np.sum(speed > 0.5)),
        "v_p50": float(np.median(speed)),
        "v_p95": float(np.percentile(speed, 95)),
        "v_max": float(speed.max()) if len(speed) else 0.0,
        "foot_gt_20mm": int(np.sum(jump > 0.020)),
        "foot_jump_mm_p95": float(np.percentile(jump, 95) * 1e3) if len(jump) else 0.0,
        "foot_jump_mm_max": float(jump.max() * 1e3) if len(jump) else 0.0,
        "ddn_mm_p50": float(np.median(ddn) * 1e3) if len(ddn) else 0.0,
        "ddn_mm_p95": float(np.percentile(ddn, 95) * 1e3) if len(ddn) else 0.0,
        "ddn_mm_max": float(ddn.max() * 1e3) if len(ddn) else 0.0,
        "ddn_gt_5mm": int(np.sum(ddn > 0.005)),
    }


def _dR_stats(R_field, R_tool, use) -> dict:
    ang = _geodesic_deg(delta_R_from_field_tool(R_field, R_tool))
    a = ang[use] if use is not None else ang
    if len(a) == 0:
        return {"n": 0}
    return {
        "n": int(len(a)),
        "median": float(np.median(a)),
        "frac_lt_45": float(np.mean(a < 45.0)),
        "frac_gt_135": float(np.mean(a > 135.0)),
    }


def _angles(a, b) -> np.ndarray:
    return np.asarray(rot_angle_deg(a, b), dtype=np.float64).reshape(-1)


def _stability_stats(angle: np.ndarray) -> dict:
    a = np.asarray(angle, dtype=np.float64).reshape(-1)
    if len(a) == 0:
        return {"n": 0}
    return {
        "n": int(len(a)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def _perturb_closest(field, p, R, stride: int) -> np.ndarray:
    idx = np.arange(0, len(p), stride)
    worst = np.zeros(len(idx))
    for sign in (1.0, -1.0):
        for axis in range(3):
            q = np.array(p[idx])
            q[:, axis] += sign * 0.005
            _, Rp = _query(field, q)
            worst = np.maximum(worst, _angles(R[idx], Rp))
    return worst


def _perturb_track(field, p, x, R, stride: int) -> np.ndarray:
    idx = np.arange(stride, len(p), stride)
    worst = np.zeros(len(idx))

    def query(pts):
        return _query(field, pts)

    for sign in (1.0, -1.0):
        for axis in range(3):
            q = np.array(p[idx])
            q[:, axis] += sign * 0.005
            xf, Rf = query(q)
            got = np.empty((len(idx), 3, 3))
            for j, i in enumerate(idx):
                _, Rh = step_foot(query, q[j], x[i - 1], R[i - 1], xf[j], Rf[j])
                got[j] = Rh
            worst = np.maximum(worst, _angles(R[idx], got))
    return worst


def _perturb_dof(dof, p, R, stride: int) -> np.ndarray:
    idx = np.arange(0, len(p), stride)
    worst = np.zeros(len(idx))
    for sign in (1.0, -1.0):
        for axis in range(3):
            q = np.array(p[idx])
            q[:, axis] += sign * 0.005
            worst = np.maximum(worst, _angles(R[idx], dof.rotations(q)))
    return worst


def _range_check(rows: np.ndarray) -> dict:
    with NORM.open("rb") as f:
        saved = pickle.load(f)
    src = saved.params_dict["action"]
    lo = src["input_stats"]["min"].detach().float().cpu().reshape(-1).numpy()
    hi = src["input_stats"]["max"].detach().float().cpu().reshape(-1).numpy()
    got_lo = rows.min(0)
    got_hi = rows.max(0)
    outside = (got_lo < lo[:3] - 1e-5) | (got_hi > hi[:3] + 1e-5)
    return {
        "pickle": str(NORM),
        "pickle_min_0_3": lo[:3].tolist(),
        "pickle_max_0_3": hi[:3].tolist(),
        "new_min_0_3": got_lo.tolist(),
        "new_max_0_3": got_hi.tolist(),
        "outside": bool(np.any(outside)),
        "refit": None,
    }


def _existing(root) -> dict:
    motion = {"v_gt_0.5": 0, "foot_gt_20mm": 0, "ddn_gt_5mm": 0, "samples": 0}
    jumps = []
    ddn = []
    speeds = []
    for i in range(50):
        ep = root["data"][f"episode_{i}"]
        v = np.asarray(ep["fixb_v_par_0"])
        x = np.asarray(ep["mold_x_s_0"])
        dn = np.asarray(ep["fixb_delta_n_0"]).reshape(-1)
        st = _motion_stats(v, x, dn)
        for key in ("v_gt_0.5", "foot_gt_20mm", "ddn_gt_5mm", "samples"):
            motion[key] += st[key]
        speeds.append(np.linalg.norm(v, axis=1))
        jumps.append(np.linalg.norm(np.diff(x, axis=0), axis=1))
        ddn.append(np.abs(np.diff(dn)))
    speed = np.concatenate(speeds)
    jump = np.concatenate(jumps)
    dd = np.concatenate(ddn)
    motion.update(
        {
            "v_p50": float(np.median(speed)),
            "v_p95": float(np.percentile(speed, 95)),
            "v_max": float(speed.max()),
            "foot_jump_mm_p95": float(np.percentile(jump, 95) * 1e3),
            "foot_jump_mm_max": float(jump.max() * 1e3),
            "ddn_mm_p50": float(np.median(dd) * 1e3),
            "ddn_mm_p95": float(np.percentile(dd, 95) * 1e3),
            "ddn_mm_max": float(dd.max() * 1e3),
        }
    )
    return motion


def _r1rel_note(root) -> dict:
    motion = _existing(root)
    cmd_big = 0
    normal_big = 0
    for i in range(50):
        ep = root["data"][f"episode_{i}"]
        cmd = np.asarray(ep["ts_pose_command_0"])[:, :3]
        cmd_big += int(np.sum(np.linalg.norm(np.diff(cmd, axis=0), axis=1) > 0.020))
        nrm = np.asarray(ep["mold_normal_0"])
        dots = np.clip(np.sum(nrm[1:] * nrm[:-1], axis=1), -1.0, 1.0)
        normal_big += int(np.sum(np.degrees(np.arccos(dots)) > 30.0))
    motion["command_xyz_steps_gt_20mm"] = cmd_big
    motion["normal_step_gt_30deg"] = normal_big
    motion["viewer"] = (
        "The r1rel button plots one RGB step of the original stringer TCP "
        "(dx, dy, dz in meters). It does not read fixb_v_par or this zarr."
    )
    motion["mold_geom"] = (
        "robot0_mold_geom is signed distance, outward normal, and R_field rot6d. "
        "The foot position is mold_x_s_0, not one of those 10 dims. "
        "The normal and R_field are the frame at that foot, so a foot hop is in the observation."
    )
    motion["action"] = (
        "The trained r1rel action is the relative TCP pose in ts_pose_command_0, not v1/v2."
    )
    return motion


def _pack(field, x, R, n_out, p, v, dn, dt) -> dict:
    q = field.query(x, with_curvature=True)
    return {
        "mold_x_s_0": x.astype(np.float32),
        "mold_R_field_0": R.reshape(-1, 9).astype(np.float32),
        "mold_R_field_rot6d_0": so3_to_rot6d_rows(R).astype(np.float32),
        "mold_normal_0": n_out.astype(np.float32),
        "mold_signed_distance_0": np.sum((p - x) * n_out, axis=1).astype(np.float32).reshape(-1, 1),
        "mold_curvature_0": np.asarray(q["curv"], dtype=np.float32),
        "fixb_v_par_0": v.astype(np.float32),
        "fixb_delta_n_0": dn.reshape(-1, 1).astype(np.float32),
        "fixb_dt_next_0": dt.reshape(-1, 1).astype(np.float32),
    }


def _write(src_zarr: Path, out_dir: Path, built: list, version: str) -> None:
    link_tree(src_zarr, out_dir / "acp_replay_buffer.zarr")
    root = zarr.open(str(out_dir / "acp_replay_buffer.zarr"), "a")
    root.attrs["label_version"] = version
    root.attrs["october_episodes_only"] = 50
    for item in built:
        ep = root["data"][item["key"]]
        for name in MOLD_KEYS:
            replace_array(ep, name, item["pack"][name], ep[name])
        replace_array(ep, "ts_pose_command_0", item["cmd"], ep["ts_pose_command_0"])
        replace_array(ep, "fixb_delta_R_rot6d_0", item["dR"], ep["fixb_delta_R_rot6d_0"])
        ep.attrs["label_version"] = version


def _episode(field, dof, rel_ep, ep, ep_id: int, long_idx) -> dict:
    robot_t = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rgb_t = np.asarray(ep["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    R_tool, p = pose7_to_Rp(ep["ts_pose_fb_0"])
    p_vt = pose7_to_Rp(rel_ep["ts_pose_virtual_target_0"])[1]
    use = _kept_robot(ep, ep_id, long_idx) if ep_id not in EXCLUDE else np.zeros(len(p), dtype=bool)

    x_track, R_track, held = track_local_feet(lambda pts: _query(field, pts), p)
    n_track = -R_track[:, :, 2]
    v_track, dt = project_tcp_velocity(R_track, p, robot_t, rgb_t)
    dn_track = _dn(p_vt, x_track, n_track)

    R_dof = dof.rotations(p)
    n_dof = -R_dof[:, :, 2]
    x_ray, miss = _ray_feet(field.scene, p, R_dof[:, :, 2])
    x_dof = np.array(x_ray, copy=True)
    # The frame comes from the field. The foot used for δn still cannot hop:
    # a far ray is replaced by the tracked foot, and either way the step is capped.
    for i in range(1, len(p)):
        propose = x_track[i] if miss[i] or float(np.linalg.norm(x_ray[i] - x_dof[i - 1])) > GATE_M else x_ray[i]
        delta = propose - x_dof[i - 1]
        dist = float(np.linalg.norm(delta))
        if dist > GATE_M:
            propose = x_dof[i - 1] + delta * (GATE_M / dist)
        x_dof[i] = propose
    missed = int(np.sum(miss))
    v_dof, _ = project_tcp_velocity(R_dof, p, robot_t, rgb_t)
    dn_dof = _dn(p_vt, x_dof, n_dof)

    _, R_free = _query(field, p)
    return {
        "use": use,
        "held": held,
        "ray_miss": missed,
        "track": (x_track, R_track, n_track, v_track, dn_track, dt),
        "dof": (x_dof, R_dof, n_dof, v_dof, dn_dof, dt),
        "closest_stab": _perturb_closest(field, p, R_free, STRIDE),
        "track_stab": _perturb_track(field, p, x_track, R_track, STRIDE),
        "dof_stab": _perturb_dof(dof, p, R_dof, STRIDE),
        "p": p,
        "R_tool": R_tool,
    }


def main() -> None:
    if OUT_TRACK.exists() or OUT_DOF.exists():
        raise SystemExit(f"refusing to overwrite {OUT_TRACK} or {OUT_DOF}")
    if REPORT.exists():
        raise SystemExit(f"refusing to overwrite {REPORT}")
    t0 = time.time()
    smooth = zarr.open(str(SMOOTH), "r")
    rel = zarr.open(str(R1REL), "r")
    keys = [f"episode_{i}" for i in range(50)]
    long_blob = np.load(LONG_DT)
    long_idx: dict[int, set[int]] = {}
    for ep, idx in zip(long_blob["flagged_episode"], long_blob["flagged_index"]):
        long_idx.setdefault(int(ep), set()).add(int(idx))

    field = SmoothFrameField(FIELD, SIDE)
    dof = build_diffused_field(field)
    track_built = []
    dof_built = []
    acc = {
        name: {"motion": [], "dR_all": [], "dR_kept": [], "rows": [], "stab": []}
        for name in ("track", "dof")
    }
    closest_stab = []
    held = 0
    missed = 0
    for i, key in enumerate(keys):
        ep = smooth["data"][key]
        got = _episode(field, dof, rel["data"][key], ep, i, long_idx)
        held += got["held"]
        missed += got["ray_miss"]
        closest_stab.append(got["closest_stab"])
        for name in ("track", "dof"):
            x, R, n_out, v, dn, dt = got[name]
            pack = _pack(field, x, R, n_out, got["p"], v, dn, dt)
            dR = delta_R_from_field_tool(R, got["R_tool"])
            r6 = so3_to_rot6d_rows(dR).astype(np.float32)
            cmd = fixb_pose7_command(v, dn).astype(np.float32)
            bucket = track_built if name == "track" else dof_built
            bucket.append({"key": key, "pack": pack, "cmd": cmd, "dR": r6})
            acc[name]["motion"].append(_motion_stats(v, x, dn))
            acc[name]["dR_all"].append(_dR_stats(R, got["R_tool"], np.ones(len(R), dtype=bool)))
            acc[name]["dR_kept"].append(_dR_stats(R, got["R_tool"], got["use"]))
            if np.any(got["use"]):
                acc[name]["rows"].append(np.concatenate([v[got["use"]], dn[got["use"], None]], axis=1))
            acc[name]["stab"].append(got[f"{name}_stab"])
        print(
            f"[{i:02d}] {time.time() - t0:.0f}s held {got['held']} ray_miss {got['ray_miss']}",
            flush=True,
        )

    def _sum_motion(parts: list) -> dict:
        out = {"v_gt_0.5": 0, "foot_gt_20mm": 0, "ddn_gt_5mm": 0, "samples": 0}
        for part in parts:
            for key in out:
                out[key] += part[key]
        return out

    report = {
        "tcp": "ts_pose_fb_0 in the odom-corrected October zarr (tag-fixture TCP)",
        "smooth_closest": _existing(smooth),
        "r1rel": _r1rel_note(rel),
        "closest_tcp_pm5mm_deg": _stability_stats(np.concatenate(closest_stab)),
        "track_refused_global_closest": held,
        "dof_ray_miss": missed,
        "stride": STRIDE,
    }
    for name, built, out_dir, version in (
        ("track", track_built, OUT_TRACK, "r4vt_smooth_tcp_track"),
        ("dof", dof_built, OUT_DOF, "r4vt_smooth_dof"),
    ):
        rows = np.concatenate(acc[name]["rows"]) if acc[name]["rows"] else np.zeros((0, 3))
        all_ang = []
        kept_ang = []
        for i, item in enumerate(built):
            ang = _geodesic_deg(rot6d_rows_to_R(item["dR"]))
            ep = smooth["data"][item["key"]]
            use = _kept_robot(ep, i, long_idx) if i not in EXCLUDE else np.zeros(len(ang), dtype=bool)
            all_ang.append(ang)
            kept_ang.append(ang[use])
        all_a = np.concatenate(all_ang)
        kept_a = np.concatenate(kept_ang) if kept_ang else np.zeros(0)
        report[name] = {
            "zarr": str(out_dir / "acp_replay_buffer.zarr"),
            "motion": _sum_motion(acc[name]["motion"]),
            "frame_pm5mm_deg": _stability_stats(np.concatenate(acc[name]["stab"])),
            "delta_R_all": {
                "n": int(len(all_a)),
                "median": float(np.median(all_a)),
                "frac_lt_45": float(np.mean(all_a < 45.0)),
                "frac_gt_135": float(np.mean(all_a > 135.0)),
            },
            "delta_R_kept": {
                "n": int(len(kept_a)),
                "median": float(np.median(kept_a)) if len(kept_a) else None,
                "frac_lt_45": float(np.mean(kept_a < 45.0)) if len(kept_a) else None,
                "frac_gt_135": float(np.mean(kept_a > 135.0)) if len(kept_a) else None,
            },
            "normalizer": _range_check(rows) if len(rows) else {"outside": None},
        }
        # Fill the distribution tails that _sum_motion dropped.
        speeds = []
        jumps = []
        ddns = []
        for item in built:
            speeds.append(np.linalg.norm(item["pack"]["fixb_v_par_0"], axis=1))
            jumps.append(np.linalg.norm(np.diff(item["pack"]["mold_x_s_0"], axis=0), axis=1))
            ddns.append(np.abs(np.diff(item["pack"]["fixb_delta_n_0"].reshape(-1))))
        speed = np.concatenate(speeds)
        jump = np.concatenate(jumps)
        dd = np.concatenate(ddns)
        report[name]["motion"].update(
            {
                "v_p50": float(np.median(speed)),
                "v_p95": float(np.percentile(speed, 95)),
                "v_max": float(speed.max()),
                "foot_jump_mm_p95": float(np.percentile(jump, 95) * 1e3),
                "foot_jump_mm_max": float(jump.max() * 1e3),
                "ddn_mm_p50": float(np.median(dd) * 1e3),
                "ddn_mm_p95": float(np.percentile(dd, 95) * 1e3),
                "ddn_mm_max": float(dd.max() * 1e3),
            }
        )
        print(f"[write] {name}", flush=True)
        _write(SMOOTH, out_dir, built, version)

    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
