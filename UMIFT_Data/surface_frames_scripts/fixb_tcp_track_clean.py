#!/usr/bin/env python3
"""Rebuild the track labels from a TCP pose with short T265 glitches filled in.

The raw pose in r4vt_smooth_tcp_track is not modified. Short glitches are
interpolated. Longer ones are written into bad_span_0 of the new zarr so a
training window that overlaps them is dropped. A new normalizer pickle is
written only when the kept rows fall outside the existing 21-d pickle.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import zarr

from diffusion_policy.common.sampler_umift import (
    filter_rgb_ids_without_action_gaps,
    sparse_action_indices_from_rgb_timestamps,
)
from fixb_build_datasets import MOLD_KEYS, link_tree, replace_array
from fixb_labels import (
    delta_R_from_field_tool,
    fixb_pose7_command,
    pose7_to_Rp,
    so3_to_rot6d_rows,
)
from fixb_smooth_field import SmoothFrameField
from fixb_smooth_frame_labels import ACTION_DS, ACTION_H, EXCLUDE, LONG_DT, NORM, QUERY_STRIDE
from fixb_tcp_glitch import (
    SHORT_MAX_S,
    SPEED_MAX_M_S,
    can_interpolate,
    clean_pose7,
    glitch_mask,
    glitch_spans,
)
from fixb_tcp_track import project_tcp_velocity, track_local_feet
from fixb_tcp_velocity_labels import R1REL, _dn, _query, _range_check

ROOT = R1REL.parents[1]
TRACK = ROOT / "r4vt_smooth_tcp_track" / "acp_replay_buffer.zarr"
FIELD = ROOT / "shared" / "fixb_field.npz"
SIDE = FIELD.with_name("fixb_smooth_frames.npz")
OUT = ROOT / "r4vt_smooth_tcp_track_clean"
NEW_NORM = NORM.with_name("oct_v35_h4s_r4vt_action_normalizer_minmax_track_clean.pkl")


def _bad(ep, ep_id: int, long_idx: dict, n: int) -> np.ndarray:
    bad = np.zeros(n, dtype=bool)
    if "bad_span_0" in ep:
        bad |= np.asarray(ep["bad_span_0"]).reshape(-1)[:n] > 0
    for j in long_idx.get(ep_id, ()):
        if 0 <= j < n:
            bad[j] = True
    return bad


def _queries(ep, ep_id: int, long_idx: dict):
    robot_t = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rgb_t = np.asarray(ep["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    wrench_t = np.asarray(ep["wrench_time_stamps_0"], dtype=np.float64).reshape(-1)
    start_t = max(float(rgb_t[16 * 4 + 1]), float(robot_t[51 * 16 + 1]), float(wrench_t[118 + 1]))
    end_t = min(float(rgb_t[-1]), float(robot_t[-1]), float(wrench_t[-1]))
    first = int(np.searchsorted(rgb_t, start_t, side="left"))
    last = int(np.searchsorted(rgb_t, end_t, side="right") - 1)
    ids = np.arange(first, last + 1, dtype=np.int64)[::QUERY_STRIDE]
    kept, _ = filter_rgb_ids_without_action_gaps(ids, robot_t, rgb_t, ACTION_H, ACTION_DS, 3.0)
    return robot_t, rgb_t, kept, _bad(ep, ep_id, long_idx, len(robot_t))


def _kept(ep, ep_id: int, long_idx: dict, extra: np.ndarray) -> np.ndarray:
    robot_t, rgb_t, kept, bad = _queries(ep, ep_id, long_idx)
    bad = bad | extra
    use = np.zeros(len(robot_t), dtype=bool)
    for rgb_id in kept:
        idx, _ = sparse_action_indices_from_rgb_timestamps(robot_t, rgb_t, int(rgb_id), ACTION_H, ACTION_DS)
        if len(idx) == 0 or np.any(bad[np.clip(idx, 0, len(bad) - 1)]):
            continue
        use[np.clip(idx, 0, len(use) - 1)] = True
    return use


def _r1rel_touch(ep, ep_id: int, long_idx: dict, glitch: np.ndarray) -> tuple[int, int, int]:
    """Kept r1rel queries, how many overlap a glitch in time, and those glitch rows."""
    robot_t, rgb_t, kept, bad = _queries(ep, ep_id, long_idx)
    queries = 0
    hit = 0
    touched = np.zeros(len(robot_t), dtype=bool)
    for rgb_id in kept:
        idx, _ = sparse_action_indices_from_rgb_timestamps(robot_t, rgb_t, int(rgb_id), ACTION_H, ACTION_DS)
        if len(idx) == 0 or np.any(bad[np.clip(idx, 0, len(bad) - 1)]):
            continue
        queries += 1
        idx = np.clip(idx, 0, len(robot_t) - 1)
        inside = glitch & (robot_t >= robot_t[idx[0]]) & (robot_t <= robot_t[idx[-1]])
        if np.any(inside):
            hit += 1
            touched |= inside
    return queries, hit, int(np.count_nonzero(touched))


def _rgb_times(rgb_t: np.ndarray, t0: float, t_a: float, t_b: float) -> list[float]:
    rel = rgb_t - t0
    inside = rel[(rel >= t_a - 1e-3) & (rel <= t_b + 1e-3)]
    if len(inside) == 0:
        inside = rel[np.argmin(np.abs(rel - 0.5 * (t_a + t_b))) : ][:1]
    return [round(float(v), 3) for v in inside]


def _refit(rows: np.ndarray) -> Path:
    """Copy the 21-d pickle and replace the v1, v2, delta_n range. Never the old file."""
    if NEW_NORM.exists():
        raise SystemExit(f"refusing to overwrite {NEW_NORM}")
    with NORM.open("rb") as f:
        saved = pickle.load(f)
    action = saved.params_dict["action"]
    stats = action["input_stats"]
    lo = rows.min(0)
    hi = rows.max(0)
    new_stats = {}
    for key in ("min", "max", "mean", "std"):
        new_stats[key] = stats[key].detach().cpu().clone()
    scale = action["scale"].detach().cpu().clone()
    offset = action["offset"].detach().cpu().clone()
    for i in range(3):
        new_stats["min"][i] = float(lo[i])
        new_stats["max"][i] = float(hi[i])
        new_stats["mean"][i] = float(rows[:, i].mean())
        new_stats["std"][i] = float(rows[:, i].std())
        span = max(float(hi[i] - lo[i]), 1e-7)
        scale[i] = 2.0 / span
        offset[i] = -1.0 - scale[i] * float(lo[i])
    for key, value in new_stats.items():
        stats[key] = value
        stats[key].requires_grad_(False)
    action["scale"] = scale
    action["offset"] = offset
    action["scale"].requires_grad_(False)
    action["offset"].requires_grad_(False)
    NEW_NORM.parent.mkdir(parents=True, exist_ok=True)
    with NEW_NORM.open("wb") as f:
        pickle.dump(saved, f)
    return NEW_NORM


def main() -> None:
    out_zarr = OUT / "acp_replay_buffer.zarr"
    if out_zarr.exists() or (OUT / "glitches.json").exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    long_blob = np.load(LONG_DT)
    long_idx: dict[int, set[int]] = {}
    for ep, idx in zip(long_blob["flagged_episode"], long_blob["flagged_index"]):
        long_idx.setdefault(int(ep), set()).add(int(idx))
    field = SmoothFrameField(str(FIELD), str(SIDE))
    src = zarr.open(str(TRACK), "r")
    rel = zarr.open(str(R1REL), "r")
    built = []
    listing = []
    glitch_json = {"speed_max_m_s": SPEED_MAX_M_S, "short_max_s": SHORT_MAX_S, "episodes": {}}
    rows = []
    v_hi = 0
    v_hi_kept = 0
    v_all = 0
    foot_hi = 0
    n_short = 0
    n_long = 0
    r1_queries = 0
    r1_hit = 0
    r1_steps = 0
    for ep_id in range(50):
        key = f"episode_{ep_id}"
        ep = src["data"][key]
        robot_t = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        rgb_t = np.asarray(ep["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
        pose = np.asarray(ep["ts_pose_fb_0"], dtype=np.float64)
        _, p_raw = pose7_to_Rp(pose)
        mask = glitch_mask(p_raw, robot_t)
        spans = glitch_spans(mask, robot_t)
        for span in spans:
            if not can_interpolate(p_raw, robot_t, span):
                span["kind"] = "long"
        cleaned = clean_pose7(pose, robot_t, spans)
        R_tool, p = pose7_to_Rp(cleaned)
        p_vt = pose7_to_Rp(rel["data"][key]["ts_pose_virtual_target_0"])[1]
        x, R, _held = track_local_feet(lambda pts: _query(field, pts), p)
        n_out = -R[:, :, 2]
        v, dt = project_tcp_velocity(R, p, robot_t, rgb_t)
        dn = _dn(p_vt, x, n_out)
        long = np.zeros(len(p), dtype=bool)
        t0 = float(rgb_t[0])
        ep_spans = []
        # A label step that still looks into a one-way jump is excluded too.
        long |= np.linalg.norm(v, axis=1) > SPEED_MAX_M_S
        for span in spans:
            if span["kind"] == "long":
                long[max(0, span["i0"] - 1) : span["i1"] + 1] = True
                n_long += 1
            else:
                n_short += 1
            item = {
                "i0": span["i0"],
                "i1": span["i1"],
                "t0": round(float(robot_t[span["i0"]] - t0), 3),
                "t1": round(float(robot_t[span["i1"]] - t0), 3),
                "kind": span["kind"],
                "rgb_s": _rgb_times(rgb_t, t0, float(robot_t[span["i0"]]), float(robot_t[span["i1"]])),
            }
            ep_spans.append(item)
            listing.append(
                f"ep {ep_id:02d}  {item['kind']:5s}  {item['t0']:8.3f}-{item['t1']:8.3f} s"
                f"  samples {span['i0']}-{span['i1']}  rgb {', '.join(f'{s:.3f}' for s in item['rgb_s'])}"
            )
        if ep_spans:
            glitch_json["episodes"][str(ep_id)] = ep_spans
        use = _kept(ep, ep_id, long_idx, long) if ep_id not in EXCLUDE else np.zeros(len(p), dtype=bool)
        speed = np.linalg.norm(v, axis=1)
        v_all += int(len(speed))
        v_hi += int(np.count_nonzero(speed > 0.5))
        v_hi_kept += int(np.count_nonzero(speed[use] > 0.5)) if np.any(use) else 0
        foot = np.linalg.norm(np.diff(x, axis=0), axis=1)
        foot_hi += int(np.count_nonzero(foot > 0.020))
        if np.any(use):
            rows.append(np.stack([v[use, 0], v[use, 1], dn[use]], axis=1))
        rel_ep = rel["data"][key]
        rel_t = np.asarray(rel_ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        _, p_rel = pose7_to_Rp(rel_ep["ts_pose_fb_0"])
        same = len(p_rel) == len(p_raw) and np.allclose(p_rel, p_raw)
        mask_rel = mask if same else glitch_mask(p_rel, rel_t)
        q, h, steps = _r1rel_touch(rel_ep, ep_id, long_idx, mask_rel)
        r1_queries += q
        r1_hit += h
        r1_steps += steps
        qcurv = field.query(x, with_curvature=True)
        pack = {
            "mold_x_s_0": x.astype(np.float32),
            "mold_R_field_0": R.reshape(-1, 9).astype(np.float32),
            "mold_R_field_rot6d_0": so3_to_rot6d_rows(R).astype(np.float32),
            "mold_normal_0": n_out.astype(np.float32),
            "mold_signed_distance_0": np.sum((p - x) * n_out, axis=1).astype(np.float32).reshape(-1, 1),
            "mold_curvature_0": np.asarray(qcurv["curv"], dtype=np.float32),
            "fixb_v_par_0": v.astype(np.float32),
            "fixb_delta_n_0": dn.reshape(-1, 1).astype(np.float32),
            "fixb_dt_next_0": dt.reshape(-1, 1).astype(np.float32),
        }
        built.append(
            {
                "key": key,
                "pack": pack,
                "cmd": fixb_pose7_command(v, dn).astype(np.float32),
                "dR": so3_to_rot6d_rows(delta_R_from_field_tool(R, R_tool)).astype(np.float32),
                "pose": cleaned.astype(np.float32),
                "long": long,
            }
        )
        print(f"  ep {ep_id:02d} spans {len(spans)} long {int(long.sum())}", flush=True)

    link_tree(TRACK, out_zarr)
    root = zarr.open(str(out_zarr), "a")
    root.attrs["label_version"] = "tcp_track_clean_v1"
    for item in built:
        ep = root["data"][item["key"]]
        for name in MOLD_KEYS:
            replace_array(ep, name, item["pack"][name], ep[name])
        replace_array(ep, "ts_pose_command_0", item["cmd"], ep["ts_pose_command_0"])
        replace_array(ep, "fixb_delta_R_rot6d_0", item["dR"], ep["fixb_delta_R_rot6d_0"])
        replace_array(ep, "ts_pose_fb_0", item["pose"], ep["ts_pose_fb_0"])
        if np.any(item["long"]):
            if "bad_span_0" in ep:
                prev = np.asarray(ep["bad_span_0"])
                bad = np.array(prev, dtype=np.float32, copy=True)
                flat = bad.reshape(-1)
                flat[item["long"][: len(flat)]] = 1.0
                replace_array(ep, "bad_span_0", bad, ep["bad_span_0"])
            else:
                bad = item["long"].astype(np.float32).reshape(-1, 1)
                ep["bad_span_0"] = bad
        ep.attrs["label_version"] = "tcp_track_clean_v1"

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "glitches.txt").write_text("\n".join(listing) + "\n")
    (OUT / "glitches.json").write_text(json.dumps(glitch_json))
    got = np.concatenate(rows) if rows else np.zeros((0, 3))
    check = _range_check(got) if len(got) else {"outside": True}
    refit = None
    if check.get("outside") and len(got):
        refit = str(_refit(got))
        check["refit"] = refit
    report = {
        "short_spans": n_short,
        "long_spans": n_long,
        "v_gt_0.5": v_hi,
        "v_gt_0.5_kept": v_hi_kept,
        "samples": v_all,
        "kept_rows": int(len(got)),
        "foot_gt_20mm": foot_hi,
        "normalizer": check,
        "r1rel_queries_kept": r1_queries,
        "r1rel_queries_hitting_glitch": r1_hit,
        "r1rel_action_rows_that_are_glitches": r1_steps,
        "v_abs_max_kept": float(np.max(np.abs(got[:, :2]))) if len(got) else None,
        "dn_min_kept": float(got[:, 2].min()) if len(got) else None,
        "dn_max_kept": float(got[:, 2].max()) if len(got) else None,
    }
    (OUT / "track_clean_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"[list] {OUT / 'glitches.txt'}", flush=True)


if __name__ == "__main__":
    main()
