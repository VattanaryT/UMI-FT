#!/usr/bin/env python3
"""Check the thin inputs the scorer decodes. Does not rewrite the score table.

Ground-truth ΔR and heading go through the same walk as a prediction.
Identity / R_zero is that walk with a zero rotation. The training stringer
is the same measurement on its own field.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import heldout  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import next_control_times, pose7_to_Rp, so3_to_rot6d_rows  # noqa: E402
from heading_tilt import frame_zero, geodesic_deg, long_axis, rotation_from_tilt  # noqa: E402
from mold_aware_runtime import rot6d_rows_to_R  # noqa: E402

import zarr  # noqa: E402

OUT = HERE / "input_check.json"
TRAIN_ZARR = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_causal_foot/acp_replay_buffer.zarr"
)
TRAIN_FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
STEPS = (1, 6, 16, 64)
UP = np.array([0.0, -1.0, 0.0])


def _ang(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12, None)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def _mean(block) -> dict:
    return {
        "mean": {str(step): round(float(block[:, step - 1].mean()), 3) for step in STEPS},
        "median": {str(step): round(float(np.median(block[:, step - 1])), 3) for step in STEPS},
    }


def _mesh_normals(field) -> tuple[np.ndarray, np.ndarray]:
    tri = field.faces
    v = field.vertices
    fn = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])
    vn = field.normals[tri].mean(axis=1)
    fn[np.sum(fn * vn, axis=1) < 0] *= -1
    fn /= np.clip(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12, None)
    return fn, long_axis(field.vertices)


def _triangles(field, points) -> np.ndarray:
    q = o3d_closest(field, points)
    return q


def o3d_closest(field, points) -> np.ndarray:
    ans = field.scene.compute_closest_points(
        __import__("open3d").core.Tensor(np.asarray(points, dtype=np.float32).reshape(-1, 3))
    )
    tri = ans["primitive_ids"].numpy().astype(np.int64)
    return np.clip(tri, 0, len(field.faces) - 1)


def _face_kind(field) -> np.ndarray:
    """Cap vertices are the rounded ends appended by the wall build. Apron is the flange."""
    n_cap = json.loads(
        Path(str(heldout.WALL)).with_name("gap_report_vertical.json").read_text()
    )["trim"]["field"]["cap_vertices"]
    cap_v = np.zeros(len(field.vertices), dtype=bool)
    cap_v[-int(n_cap):] = True
    apron = np.load(heldout.WALL)["is_apron"].astype(bool)
    tri = field.faces
    kind = np.full(len(tri), "wall", dtype=object)
    kind[apron[tri].any(axis=1)] = "apron"
    kind[cap_v[tri].all(axis=1)] = "cap"
    return kind


def _foot_velocity(frame, foot, robot_t, rgb_t) -> np.ndarray:
    t_next = next_control_times(robot_t, rgb_t)
    dt = t_next - robot_t
    nxt = np.stack([np.interp(t_next, robot_t, foot[:, d]) for d in range(3)], axis=1)
    step = np.einsum("nia,ni->na", frame[:, :, :2], nxt - foot)
    v = np.zeros_like(step)
    ok = dt > 1e-6
    v[ok] = step[ok] / dt[ok, None]
    return v


def _walk(field, frame0, foot0, tcp0, dts, v):
    action = np.zeros((len(dts), 10))
    action[:, 0:2] = v
    return heldout._walk(field, action, foot0, frame0, tcp0, dts)


def _window_queries(letter: str):
    blob = np.load(heldout.EXPORT / "causal_e25" / ("episode_0.npz" if letter == "C" else "episode_1.npz"))
    return np.asarray(blob["action_robot_idx"], dtype=np.int64), np.asarray(blob["query_rgb_idx"], dtype=np.int64)


def _load(path: Path):
    ep = zarr.open(path, "r")["data"]["episode_0"]
    robot_t = np.asarray(ep["robot_time_stamps_0"]).reshape(-1)
    rgb_t = np.asarray(ep["rgb_time_stamps_0"]).reshape(-1)
    tool_R, tool_p = pose7_to_Rp(np.asarray(ep["ts_pose_fb_0"], dtype=np.float64))
    frame = np.asarray(ep["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
    foot = np.asarray(ep["mold_x_s_0"], dtype=np.float64).reshape(-1, 3)
    dR = np.asarray(ep["fixb_delta_R_rot6d_0"], dtype=np.float64).reshape(-1, 6)
    tilt = np.asarray(ep["fixb_heading_tilt_0"], dtype=np.float64).reshape(-1, 4)
    return {
        "robot_t": robot_t, "rgb_t": rgb_t, "tool_R": tool_R, "tool_p": tool_p,
        "frame": frame, "foot": foot, "dR": dR, "tilt": tilt,
    }


def _score_gt(field, demo, axis, queries):
    v_all = _foot_velocity(demo["frame"], demo["foot"], demo["robot_t"], demo["rgb_t"])
    ident = so3_to_rot6d_rows(np.repeat(np.eye(3)[None], 1, 0))[0]
    bags = {name: [] for name in ("pos", "tang", "rot", "ident", "rzero", "head")}
    for index in queries:
        i0 = int(index[0])
        dts = heldout._dts(demo, index)
        pos, frames = _walk(
            field, demo["frame"][i0], demo["foot"][i0], demo["tool_p"][i0], dts, v_all[index]
        )
        tcp = demo["tool_p"][index]
        normal = -demo["frame"][index][:, :, 2]
        err = pos - tcp
        along = np.einsum("ni,ni->n", err, normal)
        bags["pos"].append(np.linalg.norm(err, axis=1) * 1e3)
        bags["tang"].append(np.linalg.norm(err - along[:, None] * normal, axis=1) * 1e3)
        got = np.einsum("nij,njk->nik", frames, rot6d_rows_to_R(demo["dR"][index]))
        bags["rot"].append(geodesic_deg(got, demo["tool_R"][index]))
        got_i = np.einsum("nij,njk->nik", frames, rot6d_rows_to_R(np.repeat(ident[None], len(index), 0)))
        bags["ident"].append(geodesic_deg(got_i, demo["tool_R"][index]))
        n_out = -frames[:, :, 2]
        e_long = heldout._e_long(n_out, axis)
        bags["rzero"].append(geodesic_deg(frame_zero(e_long, n_out), demo["tool_R"][index]))
        got_h = rotation_from_tilt(n_out, e_long, demo["tilt"][index])
        bags["head"].append(geodesic_deg(got_h, demo["tool_R"][index]))
    stacked = {key: np.stack(val) for key, val in bags.items()}
    return {key: _mean(val) for key, val in stacked.items()} | {"n": int(len(queries))}


def _series(field, demo, fn, kind, axis, t_end):
    t0 = float(demo["rgb_t"][0])
    rgb_rel = demo["rgb_t"] - t0
    rgb_keep = rgb_rel <= t_end if t_end is not None else np.ones(len(rgb_rel), dtype=bool)
    robot_at = np.searchsorted(demo["robot_t"], demo["rgb_t"][rgb_keep])
    robot_at = np.clip(robot_at, 0, len(demo["robot_t"]) - 1)
    tri = o3d_closest(field, demo["foot"][robot_at])
    face = kind[tri]
    mesh_n = fn[tri]
    frame = demo["frame"][robot_at]
    tcp = demo["tool_p"][robot_at]
    gap = tcp - demo["foot"][robot_at]
    travel = np.gradient(tcp, demo["robot_t"][robot_at], axis=0)
    speed = np.linalg.norm(travel, axis=1)
    t1 = frame[:, :, 0]
    n_out = -frame[:, :, 2]
    e_long = heldout._e_long(n_out, axis)
    moving = speed > 0.01
    head = _ang(t1, travel)
    elong = _ang(e_long, travel)
    same = np.sum(mesh_n * gap, axis=1) >= 0
    return {
        "t": [round(float(v), 3) for v in rgb_rel[rgb_keep]],
        "foot_mm": [round(float(v), 2) for v in np.linalg.norm(gap, axis=1) * 1e3],
        "normal_mm": [round(float(v), 2) for v in np.abs(np.sum(gap * frame[:, :, 2], axis=1)) * 1e3],
        "frame_vs_out_deg": [round(float(v), 2) for v in _ang(frame[:, :, 2], mesh_n)],
        "rzero_deg": [round(float(v), 2) for v in geodesic_deg(frame_zero(e_long, n_out), demo["tool_R"][robot_at])],
        "ident_deg": [round(float(v), 2) for v in geodesic_deg(frame, demo["tool_R"][robot_at])],
        "t1_travel_deg": [round(float(v), 2) for v in head],
        "elong_travel_deg": [round(float(v), 2) for v in elong],
        "moving": [bool(v) for v in moving],
        "face": face.tolist(),
        "same_side": [bool(v) for v in same],
    }


def _summary(series, gt) -> dict:
    face = np.array(series["face"])
    moving = np.array(series["moving"])
    travel = np.array(series["t1_travel_deg"])
    elong = np.array(series["elong_travel_deg"])

    def med(key, mask=None):
        val = np.array(series[key], dtype=np.float64)
        if mask is not None:
            val = val[mask]
        return None if len(val) == 0 else round(float(np.median(val)), 2)

    counts = {name: int(np.sum(face == name)) for name in ("wall", "cap", "apron")}
    return {
        "n_rgb": len(face),
        "foot_mm_median": med("foot_mm"),
        "normal_mm_median": med("normal_mm"),
        "frame_z_vs_mesh_out_deg_median": med("frame_vs_out_deg"),
        "rzero_deg_median": med("rzero_deg"),
        "ident_deg_median": med("ident_deg"),
        "t1_vs_travel_deg_median": med("t1_travel_deg", moving),
        "elong_vs_travel_deg_median": med("elong_travel_deg", moving),
        "t1_reversed_frac": None if moving.sum() == 0 else round(float(np.mean(travel[moving] > 90)), 3),
        "elong_reversed_frac": None if moving.sum() == 0 else round(float(np.mean(elong[moving] > 90)), 3),
        "face_counts": counts,
        "back_face_frac": round(float(np.mean(~np.array(series["same_side"]))), 3),
        "gt": gt,
    }


def _train(field) -> dict:
    root = zarr.open(TRAIN_ZARR, "r")["data"]
    dots = []
    rzero = []
    ident = []
    travel = []
    axis = long_axis(field.vertices)
    fn, _axis = _mesh_normals(field)
    for name in list(root.keys())[:8]:
        ep = root[name]
        if "mold_R_field_0" not in ep:
            continue
        frame = np.asarray(ep["mold_R_field_0"], dtype=np.float64)
        if frame.ndim == 2:
            frame = frame.reshape(-1, 3, 3)
        foot = np.asarray(ep["mold_x_s_0"], dtype=np.float64).reshape(-1, 3)
        tool_R, tool_p = pose7_to_Rp(np.asarray(ep["ts_pose_fb_0"], dtype=np.float64))
        step = max(1, len(frame) // 400)
        sl = slice(0, len(frame), step)
        tri = o3d_closest(field, foot[sl])
        mesh_n = fn[tri]
        dots.append(_ang(frame[sl][:, :, 2], mesh_n))
        n_out = -frame[sl][:, :, 2]
        e_long = heldout._e_long(n_out, axis)
        rzero.append(geodesic_deg(frame_zero(e_long, n_out), tool_R[sl]))
        ident.append(geodesic_deg(frame[sl], tool_R[sl]))
        vel = np.gradient(tool_p[sl], axis=0)
        fast = np.linalg.norm(vel, axis=1) > 0.01
        travel.append(_ang(frame[sl][:, :, 0], vel)[fast])
    cat = lambda parts: np.concatenate(parts) if parts else np.zeros(0)
    def med(parts):
        val = cat(parts)
        return None if len(val) == 0 else round(float(np.median(val)), 2)
    moved = cat(travel)
    return {
        "episodes": 8,
        "frame_z_vs_mesh_out_deg_median": med(dots),
        "rzero_deg_median": med(rzero),
        "ident_deg_median": med(ident),
        "t1_vs_travel_deg_median": None if len(moved) == 0 else round(float(np.median(moved)), 2),
        "t1_reversed_frac": None if len(moved) == 0 else round(float(np.mean(moved > 90)), 3),
    }


def _round_series(series):
    """Page plot does not need every boolean twice."""
    return series


def main() -> None:
    if OUT.exists():
        OUT.unlink()
    field = FixBMoldField(str(heldout.WALL))
    fn, axis = _mesh_normals(field)
    t_mid = json.loads(
        (heldout.SCRIPTS / "eval/thin_variants/contact_half.json").read_text()
    )["episodes"]["C"]["t_mid_s"]
    report = {"train": _train(FixBMoldField(str(TRAIN_FIELD))), "recordings": {}}
    for letter, path, t_end in (
        ("C", heldout.HERE / "C_first" / "replay.zarr", float(t_mid)),
        ("A", heldout.HERE / "A_full" / "replay.zarr", None),
    ):
        demo = _load(path)
        queries, _rgb = _window_queries(letter)
        gt = _score_gt(field, demo | {"robot_t": demo["robot_t"]}, axis, queries)
        # stored compose, no walk: must be ~0 if the label matches its frame
        comp = np.einsum(
            "nij,njk->nik", demo["frame"], rot6d_rows_to_R(demo["dR"])
        )
        stored = float(np.median(geodesic_deg(comp, demo["tool_R"])))
        kind = _face_kind(field)
        series = _series(field, demo, fn, kind, axis, t_end)
        summary = _summary(series, gt)
        summary["stored_compose_deg_median"] = round(stored, 5)
        report["recordings"][letter] = {"summary": summary, "series": _round_series(series)}
        print(letter, json.dumps(summary), flush=True)
    print("train", json.dumps(report["train"]), flush=True)
    OUT.write_text(json.dumps(report))
    print(f"[input-check] {OUT}", flush=True)


if __name__ == "__main__":
    main()
