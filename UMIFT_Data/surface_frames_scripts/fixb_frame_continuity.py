#!/usr/bin/env python3
"""Relabel October r4vt with face hysteresis. Writes a new zarr only.

Does not open the source zarr, r1rel, fixb_field.npz, or tag trajectories
for writing. November recordings are absent from the source r4vt tree.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from diffusion_policy.common.sampler_umift import (
    filter_rgb_ids_without_action_gaps,
    get_val_mask,
    sparse_action_indices_from_rgb_timestamps,
)
from fixb_build_datasets import MOLD_KEYS, link_tree, replace_array
from fixb_field import FixBMoldField
from fixb_labels import delta_R_from_field_tool, fixb_pose7_command, pose7_to_Rp, rot_angle_deg, so3_to_rot6d_rows
from fixb_labels_tracked import crown_wall_edges, episode_arrays_tracked, geodesic_jump, query_smoothed_frames
from fixb_r4_chunk_filter import episode_masks_from_zarr

SRC = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_world_fixb_tilt/acp_replay_buffer.zarr"
)
R1REL = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r1rel_world/acp_replay_buffer.zarr"
)
FIELD = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb/shared/fixb_field.npz")
OUT = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_frame_continuity")
LONG_DT = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/long_dt_gt_100ms.npz"
)

ACTION_H = 64
ACTION_DS = 1
QUERY_STRIDE = 2
GAP_MULT = 3.0
JUMP_DEG = 30.0
EXCLUDE = {35, 44}


def _queries(ep) -> np.ndarray:
    rgb_t = np.asarray(ep["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    robot_t = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    wrench_t = np.asarray(ep["wrench_time_stamps_0"], dtype=np.float64).reshape(-1)
    start_t = max(float(rgb_t[16 * 4 + 1]), float(robot_t[51 * 16 + 1]), float(wrench_t[118 + 1]))
    end_t = min(float(rgb_t[-1]), float(robot_t[-1]), float(wrench_t[-1]))
    first = int(np.searchsorted(rgb_t, start_t, side="left"))
    last = int(np.searchsorted(rgb_t, end_t, side="right") - 1)
    ids = np.arange(first, last + 1, dtype=np.int64)[::QUERY_STRIDE]
    kept, _ = filter_rgb_ids_without_action_gaps(ids, robot_t, rgb_t, ACTION_H, ACTION_DS, GAP_MULT)
    return kept


def _rot180(axis: np.ndarray) -> np.ndarray:
    a = axis / max(float(np.linalg.norm(axis)), 1e-12)
    return 2.0 * np.outer(a, a) - np.eye(3)


def _class_of(Rf, n, t1, foot_m, n_ang, along, lo, hi) -> str:
    ang = float(rot_angle_deg(np.eye(3)[None], Rf[None])[0])
    if ang > 150.0:
        for axis in (n, t1, np.cross(n, t1)):
            if float(rot_angle_deg(Rf[None], _rot180(axis)[None])[0]) < 25.0:
                return "sign"
    if foot_m > 0.02 and n_ang > 25.0:
        return "face"
    rv = Rotation.from_matrix(Rf).as_rotvec()
    axis = rv / max(float(np.linalg.norm(rv)), 1e-12)
    span = hi - lo
    at_end = along < lo + 0.06 * span or along > hi - 0.06 * span
    if abs(float(np.dot(axis, n))) > 0.85 and ang > 30.0 and at_end:
        return "end"
    return "other"


def _jump_class(R, x, along, lo, hi, trigger) -> np.ndarray:
    out = np.array([""] * len(R), dtype=object)
    for i in np.flatnonzero(trigger):
        Rf = R[i - 1].T @ R[i]
        n_ang = float(np.degrees(np.arccos(np.clip(np.dot(R[i - 1, :, 2], R[i, :, 2]), -1.0, 1.0))))
        out[i] = _class_of(Rf, R[i - 1, :, 2], R[i - 1, :, 0], float(np.linalg.norm(x[i] - x[i - 1])), n_ang, float(along[i]), lo, hi)
    return out


def _masked(ep_id, queries, jump_class, robot_t, rgb_t, bad, drop_excluded: bool) -> dict:
    buckets = {"face": 0, "end": 0, "sign": 0, "other": 0}
    n_steps = n_masked = 0
    if drop_excluded and ep_id in EXCLUDE:
        return {"steps": 0, "masked": 0, **buckets}
    for rgb_id in queries:
        idx, _ = sparse_action_indices_from_rgb_timestamps(robot_t, rgb_t, int(rgb_id), ACTION_H, ACTION_DS)
        if len(idx) == 0:
            continue
        if bad is not None and np.any(bad[np.clip(idx, 0, len(bad) - 1)]):
            continue
        a0 = int(idx[0])
        for t in range(ACTION_H):
            n_steps += 1
            if t == 0:
                continue
            a_t = int(idx[min(t, len(idx) - 1)])
            hit = next((c for c in jump_class[a0 + 1 : a_t + 1] if c), "")
            if hit:
                n_masked += 1
                buckets[hit] += 1
    return {"steps": n_steps, "masked": n_masked, **buckets}


def _body_axis(path: Path):
    blob = np.load(path)
    V = np.asarray(blob["vertices"], dtype=np.float64)
    F = np.asarray(blob["faces"], dtype=np.int64)
    parent = np.arange(len(V))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for tri in F:
        a = find(int(tri[0]))
        for b in tri[1:]:
            b = find(int(b))
            if a != b:
                parent[b] = a
    roots = np.array([find(i) for i in range(len(V))])
    root = np.bincount(roots).argmax()
    body = V[roots == root]
    center = body.mean(axis=0)
    _, _, vt = np.linalg.svd(body - center, full_matrices=False)
    axis = vt[0]
    if axis[0] < 0:
        axis = -axis
    s = (body - center) @ axis
    return center, axis, float(s.min()), float(s.max())


def _along(x, center, axis):
    return (np.asarray(x, dtype=np.float64).reshape(-1, 3) - center) @ axis


def _episode_view(ep) -> dict:
    return {
        "robot_time_stamps_0": np.asarray(ep["robot_time_stamps_0"]),
        "rgb_time_stamps_0": np.asarray(ep["rgb_time_stamps_0"]),
        "ts_pose_fb_0": np.asarray(ep["ts_pose_fb_0"]),
    }


def main(out: Path | None = None) -> None:
    out = OUT if out is None else Path(out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    src = zarr.open(str(SRC), "r")["data"]
    rel = zarr.open(str(R1REL), "r")["data"]
    keys = [k for k in src.keys() if str(k).startswith("episode_")]
    if len(keys) != 50:
        raise SystemExit(f"expected 50 October episodes, found {len(keys)}")
    val_mask = get_val_mask(len(keys), 0.1, seed=42)
    val_eps = {int(keys[i].split("_")[-1]) for i in np.flatnonzero(val_mask)}
    center, axis, lo, hi = _body_axis(FIELD)
    field = FixBMoldField(FIELD)
    long_blob = np.load(LONG_DT)
    long_idx: dict[int, set[int]] = {}
    for ep, idx in zip(long_blob["flagged_episode"], long_blob["flagged_index"]):
        long_idx.setdefault(int(ep), set()).add(int(idx))

    built = []
    totals = {}
    for name in ("before", "after"):
        totals[name] = {}
        for split in ("train", "val"):
            for stage in ("logged", "trained"):
                totals[name][f"{split}_{stage}"] = {
                    "steps": 0, "masked": 0, "face": 0, "end": 0, "sign": 0, "other": 0
                }
    switches = {
        "crown_wall": 0,
        "held_large_normal": 0,
        "margin_large_normal_tool_still": 0,
        "margin": 0,
        "rotation": 0,
        "hold": 0,
    }
    dn_old = [np.inf, -np.inf]
    dn_new = [np.inf, -np.inf]
    compare = {}
    t0 = time.time()

    for key in keys:
        ep_id = int(key.split("_")[-1])
        ep = src[key]
        view = _episode_view(ep)
        arr = episode_arrays_tracked(field, view, smooth_window_s=0.1, v_par_mode="foot", delta_n_source="fb")
        R = np.asarray(arr["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        x = np.asarray(arr["mold_x_s_0"], dtype=np.float64)
        n = R[:, :, 2]
        _, p_vt = pose7_to_Rp(rel[key]["ts_pose_virtual_target_0"])
        dn = np.sum((p_vt - x) * n, axis=1).astype(np.float32).reshape(-1, 1)
        R_tool, _ = pose7_to_Rp(view["ts_pose_fb_0"])
        dR = so3_to_rot6d_rows(delta_R_from_field_tool(R, R_tool)).astype(np.float32)
        cmd = fixb_pose7_command(arr["fixb_v_par_0"], dn).astype(np.float32)
        dn_new[0] = min(dn_new[0], float(dn.min()))
        dn_new[1] = max(dn_new[1], float(dn.max()))
        old_dn = np.asarray(ep["fixb_delta_n_0"], dtype=np.float64)
        dn_old[0] = min(dn_old[0], float(old_dn.min()))
        dn_old[1] = max(dn_old[1], float(old_dn.max()))

        R_old = np.asarray(ep["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        x_old = np.asarray(ep["mold_x_s_0"], dtype=np.float64)
        along = _along(x, center, axis)
        old_trigger = geodesic_jump(R_old, JUMP_DEG)
        new_ep = dict(view)
        new_ep["mold_R_field_0"] = R.reshape(-1, 9)
        new_ep["mold_x_s_0"] = x
        new_trigger = episode_masks_from_zarr(new_ep)["field_jump"]
        old_class = _jump_class(R_old, x_old, _along(x_old, center, axis), lo, hi, old_trigger)
        new_class = _jump_class(R, x, along, lo, hi, new_trigger)
        queries = _queries(ep)
        robot_t = np.asarray(view["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        rgb_t = np.asarray(view["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
        bad = np.zeros(len(robot_t), dtype=bool)
        if "bad_span_0" in ep:
            bad |= np.asarray(ep["bad_span_0"]).reshape(-1)[: len(bad)] > 0
        for j in long_idx.get(ep_id, ()):
            if 0 <= j < len(bad):
                bad[j] = True
        split = "val" if ep_id in val_eps else "train"
        for name, klass in (("before", old_class), ("after", new_class)):
            logged = _masked(ep_id, queries, klass, robot_t, rgb_t, None, False)
            trained = _masked(ep_id, queries, klass, robot_t, rgb_t, bad, True)
            for stage, part in (("logged", logged), ("trained", trained)):
                for k, v in part.items():
                    totals[name][f"{split}_{stage}"][k] += v

        reason = np.asarray(arr["face_reason_0"])
        free = query_smoothed_frames(field, robot_t, pose7_to_Rp(view["ts_pose_fb_0"])[1], R_tool, which="free")["free"]
        if len(R) > 1:
            n_ang = np.degrees(np.arccos(np.clip(np.sum(R[:-1, :, 2] * R[1:, :, 2], axis=1), -1.0, 1.0)))
            tool_ang = rot_angle_deg(R_tool[:-1], R_tool[1:])
            free_n = free["R"][:, :, 2]
            free_ang = np.degrees(np.arccos(np.clip(np.sum(R[:-1, :, 2] * free_n[1:], axis=1), -1.0, 1.0)))
            taken = reason[1:]

            def _events(mask):
                mask = np.asarray(mask, dtype=bool)
                prev = np.concatenate([[False], mask[:-1]])
                return int(np.sum(mask & ~prev))

            switches["margin"] += _events(taken == 1)
            switches["rotation"] += _events(taken == 2)
            switches["hold"] += _events(taken == 3)
            # Accepted closest-face changes. The output frame slews across them at <= 25 deg/step.
            switches["crown_wall"] += int(np.sum(crown_wall_edges(reason, free["R"])))
            switches["held_large_normal"] += _events((taken == 3) & (free_ang > 45.0) & (tool_ang < 5.0))
            switches["margin_large_normal_tool_still"] += _events((taken == 1) & (n_ang > 45.0) & (tool_ang < 1.0))

        if ep_id in (0, 9):
            summary, arrays = _compare(ep_id, x_old, R_old, x, R, old_trigger, new_trigger)
            compare[str(ep_id)] = summary
            compare[f"arrays_{ep_id}"] = arrays
        built.append({"arr": arr, "dn": dn, "dR": dR, "cmd": cmd})
        print(f"[{ep_id:02d}] {time.time() - t0:.0f}s", flush=True)

    print("[write] hardlink then replace mold arrays", flush=True)
    out.mkdir(parents=True)
    link_tree(SRC, out / "acp_replay_buffer.zarr")
    root = zarr.open(str(out / "acp_replay_buffer.zarr"), "a")
    root.attrs["label_version"] = "r4vt_frame_continuity_causal"
    root.attrs["query_smooth"] = "causal_savgol_0.1s_poly2"
    root.attrs["october_episodes_only"] = 50
    root.attrs["november_excluded"] = "source r4vt tree is episodes 0-49; November tag_traj is not in this zarr"
    for i, key in enumerate(keys):
        ep = root["data"][key]
        pack = built[i]
        for name in MOLD_KEYS:
            value = pack["dn"] if name == "fixb_delta_n_0" else pack["arr"][name]
            replace_array(ep, name, value, ep[name])
        replace_array(ep, "ts_pose_command_0", pack["cmd"], ep["ts_pose_command_0"])
        replace_array(ep, "fixb_delta_R_rot6d_0", pack["dR"], ep["fixb_delta_R_rot6d_0"])
        ep.attrs["label_version"] = "r4vt_frame_continuity_causal"

    replay = {k: v for k, v in compare.items() if not k.startswith("arrays_")}
    report = {
        "source": str(SRC),
        "output": str(out / "acp_replay_buffer.zarr"),
        "field_read_only": str(FIELD),
        "val_episodes_lexical_seed42": sorted(val_eps),
        "excluded_after_mask": sorted(EXCLUDE),
        "masked": totals,
        "switches": switches,
        "delta_n_old_m": dn_old,
        "delta_n_new_m": dn_new,
        "episodes_0_and_9": replay,
        "mask_note": "logged matches the dataset print (before episode/span drops). trained drops episodes 35 and 44, bad_span_0, and long-dt targets.",
    }
    (out / "frame_continuity_report.json").write_text(json.dumps(report, indent=2))
    flat = {}
    for ep in (0, 9):
        for k, v in compare[f"arrays_{ep}"].items():
            flat[f"ep{ep}_{k}"] = v
    np.savez_compressed(out / "ep0_ep9_frames.npz", **flat)
    print(json.dumps({"masked": totals, "switches": switches, "delta_n_new_m": dn_new}, indent=2))


def _compare(ep_id, x_old, R_old, x_new, R_new, old_jump, new_jump):
    ang = rot_angle_deg(R_old, R_new)
    foot = np.linalg.norm(x_new - x_old, axis=1)
    summary = {
        "episode": ep_id,
        "n": int(len(x_old)),
        "old_jumps": int(old_jump.sum()),
        "new_unexplained": int(new_jump.sum()),
        "old_jumps_still_unexplained": int(np.sum(old_jump & new_jump)),
        "foot_delta_p50_mm": round(float(np.median(foot)) * 1e3, 3),
        "foot_delta_p95_mm": round(float(np.percentile(foot, 95)) * 1e3, 3),
        "frame_angle_p50_deg": round(float(np.median(ang)), 3),
        "frame_angle_at_old_jumps_p50_deg": round(float(np.median(ang[old_jump])) if np.any(old_jump) else 0.0, 3),
    }
    arrays = {
        "x_old": x_old.astype(np.float32),
        "x_new": x_new.astype(np.float32),
        "R_old": R_old.astype(np.float32),
        "R_new": R_new.astype(np.float32),
        "old_jump": old_jump,
        "new_jump": new_jump,
        "frame_angle_deg": ang.astype(np.float32),
        "foot_delta_m": foot.astype(np.float32),
    }
    return summary, arrays


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    main(parser.parse_args().out)
