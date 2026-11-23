#!/usr/bin/env python3
"""Training exports and the thin stringer on one open-loop page.

Does not edit view_openloop.py. The thin mesh is the adjusted wall.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

SURF = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/UMIFT_Data/surface_frames_scripts")
ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
sys.path[:0] = [str(SURF), str(SURF.parent.parent / "PyriteEnvSuites/scripts"), str(ROOT)]

import view_openloop as viewer  # noqa: E402
import zarr  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "thin_variants"))
from contact_half import gap_payload, halves, load_variant  # noqa: E402
PORT = 8802
TRAIN_MESH = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
THIN_MESH = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/"
    "wall_fix/fixb_field_wall_vertical.npz"
)
TRAIN_RGB = (
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_causal_foot/acp_replay_buffer.zarr"
)
THIN_RGB = (
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/"
    "remap_to_training/acp_replay_buffer.zarr"
)
THIN_TAG = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/"
    "remap_to_training/tag_traj/2026-11-19-15.49.52_ee_master.npz"
)
TRAIN_EXPORTS = [
    Path("/mnt/data/umidata/ft/open_loop_exports/r4vt_e66520fae9ae"),
    Path("/mnt/data/umidata/ft/open_loop_exports/r4vt_e66520fae9ae_fixdt"),
    Path("/mnt/data/umidata/ft/open_loop_exports/r4vt_causal_e0030_43308311_release"),
    Path("/mnt/data/umidata/ft/open_loop_exports/r4vt_smooth_e0030_43335222_release"),
    Path("/mnt/data/umidata/ft/open_loop_exports/r4vt_track_clean_e0055_43351238"),
    Path("/mnt/data/umidata/ft/open_loop_exports/r1rel_df82d01f0d23_fixdt"),
]
THIN_EXPORT = ROOT / "UMIFT_Data/surface_frames_scripts/eval/thin_v2/export"
HELD_EXPORT = ROOT / "UMIFT_Data/surface_frames_scripts/eval/thin_heldout/export"
THIN_ZARR = Path("/mnt/data/umidata/ft/datasets/flat_stringer/zarr")
VARIANT_DIR = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/variant_resolve"
)
# Episode 0 stays recording C so the existing prediction export still binds.
THIN_EPS = (
    (0, "C 15.49.52 first half", "2026-11-19-15.49.52"),
    (1, "A 15.43.29 full", "2026-11-19-15.43.29"),
    (2, "B 15.45.09 no pred", "2026-11-19-15.45.09"),
    (3, "D 15.51.13 no pred", "2026-11-19-15.51.13"),
)
VARIANTS = ("A", "B", "C", "D")
HEADING_E25 = (
    "/mnt/data/umidata/ft/training_outputs/"
    "oct-v35-h4s-r4vt-heading-tilt_sc_e200_bs64_valtopk_43448962/checkpoints/"
    "epoch=0025-val_loss=0.059.ckpt"
)
CAUSAL_E199 = (
    "/mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_causal_foot/"
    "oct-v35-h4s-r4vt-causal-foot_e200_bs64_valtopk_43390127/checkpoints/latest.ckpt"
)
_CACHE = {}
_DN = {}
_VEL = {}
_CONTACT = {}
_FIELD = None
K_RUN = 3000.0
ALIGN_SAVE = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/"
    "manual_align/thin_manual_T.json"
)
_POS = ("path", "label_foot", "pred_foot", "pred_exec", "pred_full", "tcp_exec", "tcp_full", "q_xyz")
_DIR = ("label_n", "pred_n")


def _R(rpy_deg) -> np.ndarray:
    """Intrinsic XYZ, R = Rx(roll) @ Ry(pitch) @ Rz(yaw). Same as frame_conventions."""
    a, b, c = np.radians(np.asarray(rpy_deg, dtype=np.float64))
    cx, sx = np.cos(a), np.sin(a)
    cy, sy = np.cos(b), np.sin(b)
    cz, sz = np.cos(c), np.sin(c)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rx @ ry @ rz


def _apply(xyz: np.ndarray, mm, rpy) -> np.ndarray:
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    return ( _R(rpy) @ pts.T).T + np.asarray(mm, dtype=np.float64).reshape(3) / 1000.0


def _check() -> None:
    from scipy.spatial.transform import Rotation

    got = _apply([[0.0, 1.0, 0.0]], [0, 0, 0], [90, 0, 0])
    if np.linalg.norm(got - np.array([0.0, 0.0, 1.0])) > 1e-9:
        raise RuntimeError(f"roll is not Rx: {got}")
    e = [10.0, -20.0, 30.0]
    if not np.allclose(_R(e), Rotation.from_euler("XYZ", e, degrees=True).as_matrix()):
        raise RuntimeError("align rotation does not match intrinsic XYZ")
    foot = np.zeros((1, 2, 3))
    normal = np.zeros((1, 2, 3))
    normal[0, 0] = [0.0, 0.0, 1.0]
    normal[0, 1] = [1.0, 0.0, 0.0]
    start = np.array([[1.0, 0.0, 0.2]])
    got = _tool_from_velocity(foot, normal, start)
    if np.linalg.norm(got[0, 1] - np.array([0.0, 0.0, -1.0])) > 1e-6:
        raise RuntimeError(f"tangent was not carried with the normal: {got[0, 1]}")


def _rotate_from_to(vector, source, target):
    """Rotate ``vector`` (Q,3) by the map that takes ``source`` onto ``target`` (Q,H,3)."""
    source = source / np.clip(np.linalg.norm(source, axis=1, keepdims=True), 1e-12, None)
    target = target / np.clip(np.linalg.norm(target, axis=2, keepdims=True), 1e-12, None)
    base = np.broadcast_to(source[:, None, :], target.shape)
    cosine = np.einsum("qhi,qhi->qh", base, target)
    axis = np.cross(base, target)
    values = np.broadcast_to(vector[:, None, :], target.shape)
    once = np.cross(axis, values)
    twice = np.cross(axis, once)
    out = values + once + twice / np.maximum(1.0 + cosine, 1e-8)[..., None]
    out[cosine > 1.0 - 1e-8] = values[cosine > 1.0 - 1e-8]
    flipped = cosine < -1.0 + 1e-8
    if np.any(flipped):
        out[flipped] = -values[flipped]
    return out


def _tool_from_velocity(foot, normal, start):
    """Foot walked by v, plus the query TCP's tangent offset. δn is not added."""
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.clip(np.linalg.norm(normal, axis=2, keepdims=True), 1e-12, None)
    origin = normal[:, 0]
    offset = np.asarray(start, dtype=np.float64) - foot[:, 0]
    offset = offset - origin * np.einsum("qi,qi->q", offset, origin)[:, None]
    return np.asarray(foot, dtype=np.float64) + _rotate_from_to(offset, origin, normal)


def _k3000(dataset: str, episode: int):
    key = (dataset, int(episode))
    if key in _DN:
        return _DN[key]
    from PyriteUtility.data_pipeline.wrench_baseline_utils import time_based_moving_average
    from fixb_labels import pose7_to_Rp

    group = viewer._dataset_group(dataset, episode)
    robot_t = np.asarray(group["robot_time_stamps_0"]).reshape(-1)
    wrench_t = np.asarray(group["wrench_time_stamps_left_0"]).reshape(-1)
    wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
    smooth = time_based_moving_average(wrench, wrench_t, 1.0)
    j = np.clip(np.searchsorted(wrench_t, robot_t), 1, len(wrench_t) - 1)
    left = np.abs(robot_t - wrench_t[j - 1]) <= np.abs(robot_t - wrench_t[j])
    force = smooth[np.where(left, j - 1, j), :3]
    rotation, _tcp = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
    outward = np.asarray(group["mold_normal_0"], dtype=np.float64).reshape(-1, 3)
    fn = np.einsum("ni,ni->n", np.einsum("nij,nj->ni", rotation, force), outward)
    _DN[key] = -fn / K_RUN
    return _DN[key]


def _velocity(exp, episode: int):
    key = (str(exp["dir"]), int(episode))
    if key in _VEL:
        return _VEL[key]
    if exp["settings"].get("role") != "r4vt":
        _VEL[key] = None
        return None
    blob = exp["episodes"].get(episode)
    if blob is None:
        return None
    foot, normal = viewer.prediction_feet(blob, exp["dir"], episode)
    if foot is None:
        _VEL[key] = None
        return None
    tcp = np.asarray(blob["tcp_xyz"], dtype=np.float64)
    anchor = np.clip(viewer.query_anchor(blob), 0, len(tcp) - 1)
    _VEL[key] = _tool_from_velocity(foot, normal, tcp[anchor])
    return _VEL[key]


def _attach_split(out, exports, episode: int) -> None:
    by_label = {exp["label"]: exp for exp in exports}
    for run in out.get("runs") or []:
        exp = by_label.get(run.get("label"))
        if exp is None:
            continue
        blob = exp["episodes"].get(episode)
        pos = _velocity(exp, episode)
        if pos is not None and blob is not None:
            horizon = int(exp["settings"].get("execution_horizon") or 6)
            rgb = np.asarray(blob["query_rgb_idx"])
            n_valid = np.asarray(blob["n_valid"])
            run["tcp_exec"], run["tcp_exec_break"] = viewer._chunk_lines(pos, n_valid, rgb, horizon, False)
            run["tcp_full"], run["tcp_full_break"] = viewer._chunk_lines(pos, n_valid, rgb, horizon, True)
        dataset = exp["settings"].get("dataset")
        if exp["settings"].get("role") != "r4vt" or not dataset or blob is None or "q_i" not in run:
            continue
        if "action_robot_idx" not in blob.files:
            continue
        try:
            label = _k3000(dataset, episode)
        except Exception:
            continue
        idx = np.asarray(blob["action_robot_idx"], dtype=np.int64)[:, 0]
        values = np.full(len(idx), np.nan)
        known = (idx >= 0) & (idx < len(label))
        values[known] = label[idx[known]]
        run["dn_k"] = [
            None if not np.isfinite(values[int(i)]) else round(float(values[int(i)]), 5)
            for i in run["q_i"]
        ]


def _attach_cursor_split(payload, exports, episode: int, query: int, span: str) -> None:
    by_label = {exp["label"]: exp for exp in exports}
    for item in payload.get("runs") or []:
        exp = by_label.get(item.get("label"))
        if exp is None:
            continue
        pos = _velocity(exp, episode)
        blob = exp["episodes"].get(episode)
        if pos is None or blob is None or query >= len(pos):
            continue
        n = int(blob["n_valid"][query])
        if span != "full":
            n = min(n, int(exp["settings"].get("execution_horizon") or 6))
        n = int(np.clip(n, 0, pos.shape[1]))
        if n == 0:
            continue
        item["tcp_v"] = viewer.rounded(pos[query, :n], 4)


def _mesh(path: Path):
    field = np.load(path)
    vertices, faces = viewer.decimate_mesh(field["vertices"], field["faces"], field["is_apron"])
    return viewer.mesh_payload(vertices, faces)


def _jpeg(frame, size, quality):
    image = Image.fromarray(np.ascontiguousarray(frame), mode="RGB")
    if size:
        image = image.resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _variant(name) -> str:
    name = str(name or "A").upper()
    return name if name in VARIANTS else "A"


def _input_view(episode: int) -> dict:
    letter = {0: "C", 1: "A"}.get(int(episode))
    path = HERE.parent / "thin_heldout" / "input_check.json"
    if letter is None or not path.is_file():
        return {}
    blob = json.loads(path.read_text())
    rec = blob["recordings"][letter]
    series = rec["series"]
    summary = rec["summary"]
    train = blob["train"]
    rot = summary["gt"]["rot"]["median"]
    head = summary["gt"]["head"]["median"]
    faces = summary["face_counts"]
    return {
        "in_t": series["t"],
        "in_foot": series["foot_mm"],
        "in_face": series["face"],
        "in_ident": series["ident_deg"],
        "in_rzero": series["rzero_deg"],
        "in_normal": series["frame_vs_out_deg"],
        "in_head": series["t1_travel_deg"],
        "in_rot_note": (
            f"zero ΔR {summary['ident_deg_median']}°  R_zero {summary['rzero_deg_median']}°"
            f"  (train {train['ident_deg_median']}° / {train['rzero_deg_median']}°)"
            f"   GT ΔR median {rot['1']}/{rot['6']}/{rot['16']}/{rot['64']}°"
            f"   GT heading {head['1']}/{head['6']}/{head['16']}/{head['64']}°"
        ),
        "in_frame_note": (
            f"frame z vs mesh outward {summary['frame_z_vs_mesh_out_deg_median']}°"
            f"  (train {train['frame_z_vs_mesh_out_deg_median']}°)"
            f"   t1 vs travel {summary['t1_vs_travel_deg_median']}°"
            f"  (train {train['t1_vs_travel_deg_median']}°)"
        ),
        "in_foot_note": (
            f"foot–TCP median {summary['foot_mm_median']} mm"
            f"   wall {faces.get('wall', 0)}  cap {faces.get('cap', 0)}  apron {faces.get('apron', 0)}"
            f"   other side {summary['back_face_frac']}"
        ),
    }


def _wall_view(episode: int) -> dict:
    letter = {0: "C", 1: "A"}.get(int(episode))
    path = HERE.parent / "thin_heldout" / "wall_only" / "page.json"
    if letter is None or not path.is_file():
        return {}
    rec = json.loads(path.read_text())["recordings"][letter]
    return {
        "wall_t": rec["t"],
        "wall_mm": rec["wall_mm"],
        "wall_face": rec["full_face"],
        "wall_note": rec["note"],
        "ood_t": rec["ood_t"],
        "ood_ht": rec.get("ood_ht") or rec["ood_t"],
        "ood_causal": rec["causal_deg"],
        "ood_heading": rec["heading_deg"],
        "ood_names": rec["names"],
        "ood_y": rec["units"],
        "ood_note": rec["ood_note"],
    }


def _thin_row(episode: int):
    episode = int(episode)
    for row in THIN_EPS:
        if row[0] == episode:
            return row
    return THIN_EPS[0]


def _variant_pose(episode: int, variant: str):
    _ep, _label, stamp = _thin_row(episode)
    path = VARIANT_DIR / _variant(variant) / f"{stamp}_ee_master.npz"
    if not path.is_file():
        return None
    blob = np.load(path)
    T = np.asarray(blob["T_master_tcp"], dtype=np.float64)
    times = np.asarray(blob["timestamp_sec_unified"], dtype=np.float64)
    return times, T[:, :3, 3], T[:, :3, :3]


def _apply_variant(out: dict, episode: int, variant: str) -> None:
    got = _variant_pose(episode, variant)
    if got is None:
        out["variant"] = None
        return
    times, xyz, rot = got
    keep = viewer.stride_idx(len(times), viewer.PATH_CAP)
    t0 = float(times[0])
    out["path_t"] = viewer.rounded(times[keep] - t0, 3)
    out["path"] = viewer.rounded(xyz[keep], 4)
    out["tool_R"] = viewer.rounded(rot[keep], 4)
    out["path_break"] = [False] * max(len(keep) - 1, 0)
    out["variant"] = _variant(variant)


def _bare_payload(episode: int, variant: str) -> dict:
    got = _variant_pose(episode, variant)
    if got is None:
        raise KeyError(episode)
    times, xyz, rot = got
    keep = viewer.stride_idx(len(times), viewer.PATH_CAP)
    t0 = float(times[0])
    path_t = viewer.rounded(times[keep] - t0, 3)
    zeros = [0.0] * len(path_t)
    run = {
        "label": "no prediction export",
        "role": "r4vt",
        "split": "heldout",
        "q_i": [int(i) for i in keep],
        "q_t": path_t,
        "q_break": [False] * len(path_t),
        "gaps": [],
        "path": [],
        "path_break": [],
        "v1_p": zeros, "v1_l": zeros, "v2_p": zeros, "v2_l": zeros,
        "n_p": zeros, "n_l": zeros, "stiff_p": zeros, "stiff_l": zeros,
        "ang": zeros,
        "error": {"mae": {}},
        "exec_horizon": 16,
        "query_stride": 1,
        "epoch": None,
        "ckpt": "",
    }
    _ep, stamp_label, stamp = _thin_row(episode)
    wrench_t, force = _wrench_series(stamp, t0)
    return {
        "episode": int(episode),
        "split": stamp_label,
        "policy": False,
        "variant": _variant(variant),
        "path_t": path_t,
        "path": viewer.rounded(xyz[keep], 4),
        "path_break": [False] * max(len(keep) - 1, 0),
        "tool_R": viewer.rounded(rot[keep], 4),
        "force_t": wrench_t,
        "force": force,
        "force_break": [False] * max(len(wrench_t) - 1, 0),
        "shade": [],
        "q_index": [],
        "runs": [run],
    }


def _wrench_series(stamp: str, t0: float):
    g = zarr.open(THIN_ZARR / f"{stamp}_unified.zarr", "r")["sensors"]["ft_wrench"]
    times = np.asarray(g["timestamp_sec_unified"], dtype=np.float64)
    force = np.linalg.norm(
        np.column_stack([np.asarray(g[k], dtype=np.float64) for k in ("force_x", "force_y", "force_z")]),
        axis=1,
    )
    keep = viewer.stride_idx(len(times), viewer.SERIES_CAP)
    return viewer.rounded(times[keep] - t0, 3), viewer.rounded(force[keep], 2)


def _images(mold: str, episode: int):
    key = (mold, episode)
    if key in _CACHE:
        return _CACHE[key]
    tags = None
    if mold == "thin" and int(episode) != 0:
        _ep, _label, stamp = _thin_row(episode)
        color = zarr.open(THIN_ZARR / f"{stamp}_unified.zarr", "r")["camera"]["color"]
        rgb = np.asarray(color["frames"])
        rgb_t = np.asarray(color["timestamp_sec_unified"]).reshape(-1)
        side = VARIANT_DIR / "A" / f"{stamp}_ee_master.npz"
        if side.is_file():
            tags = np.asarray(np.load(side)["n_tags_visible"], dtype=np.int32).reshape(-1)
    else:
        path = TRAIN_RGB if mold == "training" else THIN_RGB
        group = zarr.open(path, "r")["data"][f"episode_{episode}"]
        rgb = np.asarray(group["rgb_0"])
        rgb_t = np.asarray(group["rgb_time_stamps_0"]).reshape(-1)
        if mold == "thin":
            side = np.load(THIN_TAG)
            tags = np.asarray(side["n_tags_visible"], dtype=np.int32).reshape(-1)
            if len(tags) != len(rgb):
                raise SystemExit("AprilTag sidecar length does not match rgb")
    t0 = float(rgb_t[0])
    if tags is None or len(tags) != len(rgb):
        tags = np.zeros(len(rgb), dtype=np.int32)
    thumbs = []
    full = []
    means = []
    for frame in rgb:
        frame = np.asarray(frame)
        means.append(round(float(frame.mean()), 1))
        thumbs.append("data:image/jpeg;base64," + base64.b64encode(_jpeg(frame, 56, 35)).decode())
        full.append(_jpeg(frame, 224 if frame.shape[1] > 224 else 0, 70))
    pack = {
        "rgb_t": [round(float(t - t0), 3) for t in rgb_t],
        "jpg": thumbs,
        "tag": [int(v) for v in tags],
        "pix_mean": means,
        "full": full,
    }
    _CACHE[key] = pack
    return pack


def _nums(args, names, default=0.0):
    out = []
    for name in names:
        try:
            out.append(float(args.get(name, default)))
        except (TypeError, ValueError):
            out.append(float(default))
    return out


def _contact_xyz(episode: int, variant: str = "A") -> np.ndarray:
    key = (int(episode), _variant(variant))
    if key in _CONTACT:
        return _CONTACT[key]
    got = _variant_pose(episode, variant)
    if got is None:
        _CONTACT[key] = np.zeros((0, 3))
        return _CONTACT[key]
    times, xyz, _rot = got
    _ep, _label, stamp = _thin_row(episode)
    g = zarr.open(THIN_ZARR / f"{stamp}_unified.zarr", "r")["sensors"]["ft_wrench"]
    wrench_t = np.asarray(g["timestamp_sec_unified"], dtype=np.float64)
    force = np.linalg.norm(
        np.column_stack([np.asarray(g[k], dtype=np.float64) for k in ("force_x", "force_y", "force_z")]),
        axis=1,
    )
    j = np.clip(np.searchsorted(wrench_t, times), 1, len(wrench_t) - 1)
    left = np.abs(times - wrench_t[j - 1]) <= np.abs(times - wrench_t[j])
    hit = force[np.where(left, j - 1, j)] >= 8.0
    _CONTACT[key] = xyz[hit]
    return _CONTACT[key]


def _field():
    global _FIELD
    if _FIELD is None:
        from fixb_field import FixBMoldField

        _FIELD = FixBMoldField(THIN_MESH)
    return _FIELD


def align_report(episode: int, mm, rpy, variant: str = "A") -> dict:
    pts = _apply(_contact_xyz(episode, variant), mm, rpy)
    if len(pts) == 0:
        stats = {"n": 0, "median_mm": None, "p90_mm": None}
    else:
        foot = np.asarray(_field().query(pts)["x_s"], dtype=np.float64).reshape(-1, 3)
        gap = np.linalg.norm(pts - foot, axis=1) * 1e3
        stats = {
            "n": int(len(gap)),
            "median_mm": round(float(np.median(gap)), 1),
            "p90_mm": round(float(np.percentile(gap, 90)), 1),
        }
    return {
        "xyz_mm": [round(float(v), 3) for v in mm],
        "rpy_deg": [round(float(v), 3) for v in rpy],
        "euler": "intrinsic XYZ, R = Rx(roll) @ Ry(pitch) @ Rz(yaw), p' = R @ p + t",
        "R": _R(rpy).round(6).tolist(),
        "t_m": [round(float(v) / 1000.0, 6) for v in mm],
        "mesh": str(THIN_MESH),
        "contact": "|F| >= 8 N on the demo TCP",
        **stats,
    }


def save_align(report: dict) -> tuple[int, dict]:
    if ALIGN_SAVE.exists():
        return 409, {"error": "exists", "path": str(ALIGN_SAVE)}
    ALIGN_SAVE.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2)
    try:
        fd = os.open(ALIGN_SAVE, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return 409, {"error": "exists", "path": str(ALIGN_SAVE)}
    with os.fdopen(fd, "w") as handle:
        handle.write(payload)
    report["path"] = str(ALIGN_SAVE)
    return 200, report


def _align_bar() -> str:
    bits = [
        '<div class="bar" id="alignBar" style="display:none">',
        '<span>variant</span><select id="variant">',
        '<option value="A" selected>A recorded</option>',
        '<option value="B">B new-mount relever</option>',
        '<option value="C">C fixture t=0</option>',
        '<option value="D">D both</option>',
        "</select>",
    ]
    for name, unit, lo, hi, step in (
        ("x", "mm", -500, 500, 0.5),
        ("y", "mm", -500, 500, 0.5),
        ("z", "mm", -500, 500, 0.5),
        ("roll", "deg", -180, 180, 0.1),
        ("pitch", "deg", -180, 180, 0.1),
        ("yaw", "deg", -180, 180, 0.1),
    ):
        bits.append(
            f'<label>{name}<input id="a_{name}" type="range" min="{lo}" max="{hi}" '
            f'step="{step}" value="0"><input id="n_{name}" type="number" min="{lo}" '
            f'max="{hi}" step="{step}" value="0">{unit}</label>'
        )
    bits.append('<span id="alignGap">contact |F|≥8 —</span><button id="alignSave">Save</button>')
    bits.append('<span id="alignMsg"></span></div>')
    return "".join(bits)


_ALIGN_JS = r"""
const ALIGN_POS = ["path","label_foot","pred_foot","pred_exec","pred_full","tcp_exec","tcp_full","q_xyz","future","label_xyz","tcp_v"];
const ALIGN_DIR = ["label_n","pred_n"];
let alignRaw = null;
let alignTimer = 0;
function variantName() {
  const el = document.getElementById("variant");
  return el ? el.value : "A";
}
function alignMul(A, B) {
  const C = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    C[i][j] = A[i][0]*B[0][j] + A[i][1]*B[1][j] + A[i][2]*B[2][j];
  return C;
}
function alignA() {
  const d = Math.PI / 180;
  const a = +document.getElementById("n_roll").value * d;
  const b = +document.getElementById("n_pitch").value * d;
  const c = +document.getElementById("n_yaw").value * d;
  const cx = Math.cos(a), sx = Math.sin(a);
  const cy = Math.cos(b), sy = Math.sin(b);
  const cz = Math.cos(c), sz = Math.sin(c);
  const R = alignMul(
    [[1,0,0],[0,cx,-sx],[0,sx,cx]],
    alignMul([[cy,0,sy],[0,1,0],[-sy,0,cy]], [[cz,-sz,0],[sz,cz,0],[0,0,1]])
  );
  return {R, t: ["x","y","z"].map(n => +document.getElementById("n_"+n).value / 1000)};
}
function poseList(pts, A, dirs) {
  if (!pts) return pts;
  const t = dirs ? [0,0,0] : A.t, R = A.R;
  return pts.map(p => {
    if (!p || p.length < 3 || typeof p[0] !== "number") return p;
    return [
      R[0][0]*p[0] + R[0][1]*p[1] + R[0][2]*p[2] + t[0],
      R[1][0]*p[0] + R[1][1]*p[1] + R[1][2]*p[2] + t[1],
      R[2][0]*p[0] + R[2][1]*p[1] + R[2][2]*p[2] + t[2]
    ];
  });
}
function poseMats(mats, A) {
  if (!mats) return mats;
  return mats.map(M => {
    if (!M) return M;
    const out = [[0,0,0],[0,0,0],[0,0,0]], R = A.R;
    for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++)
      out[r][c] = R[r][0]*M[0][c] + R[r][1]*M[1][c] + R[r][2]*M[2][c];
    return out;
  });
}
function poseRun(src, dst, A) {
  ALIGN_POS.forEach(k => { if (src[k]) dst[k] = poseList(src[k], A, false); });
  ALIGN_DIR.forEach(k => { if (src[k]) dst[k] = poseList(src[k], A, true); });
}
function poseData() {
  if (mold !== "thin" || !alignRaw) return;
  const A = alignA();
  data.path = poseList(alignRaw.path, A, false);
  if (alignRaw.tool_R) data.tool_R = poseMats(alignRaw.tool_R, A);
  data.runs.forEach((run, i) => poseRun(alignRaw.runs[i], run, A));
}
function poseCursor(cursor) {
  if (mold !== "thin" || !cursor) return;
  const A = alignA();
  (cursor.runs || []).forEach(item => {
    poseRun(item, item, A);
    if (item.future_R) item.future_R = poseMats(item.future_R, A);
  });
  if (cursor.frame) {
    ["x_old","x_new"].forEach(k => {
      if (cursor.frame[k]) cursor.frame[k] = poseList([cursor.frame[k]], A, false)[0];
    });
    ["R_old","R_new"].forEach(k => {
      if (cursor.frame[k]) cursor.frame[k] = poseMats([cursor.frame[k]], A)[0];
    });
  }
}
function stashAlign() {
  if (mold !== "thin" || !data) { alignRaw = null; return; }
  alignRaw = {
    path: JSON.parse(JSON.stringify(data.path)),
    tool_R: data.tool_R ? JSON.parse(JSON.stringify(data.tool_R)) : null,
    runs: data.runs.map(r => JSON.parse(JSON.stringify(r)))
  };
  poseData();
  refreshGap();
}
function scheduleAlign() {
  clearTimeout(alignTimer);
  alignTimer = setTimeout(() => { poseData(); paintCursor(); refreshGap(); }, 60);
}
async function refreshGap() {
  if (mold !== "thin" || !data) return;
  const q = ["x","y","z","roll","pitch","yaw"].map(n => n + "=" + document.getElementById("n_"+n).value).join("&");
  const gap = await (await fetch("/api/align?mold=thin&ep=" + data.episode + "&v=" + variantName() + "&" + q)).json();
  const box = document.getElementById("alignGap");
  if (box) box.textContent = "contact |F|≥8  n=" + gap.n + "  med " + gap.median_mm + " mm  p90 " + gap.p90_mm + " mm";
}
async function saveAlign() {
  const body = {ep: data ? data.episode : 0, v: variantName()};
  ["x","y","z","roll","pitch","yaw"].forEach(n => { body[n] = +document.getElementById("n_"+n).value; });
  const res = await fetch("/api/align_save", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const msg = document.getElementById("alignMsg");
  msg.textContent = res.status === 409 ? "already saved, not overwritten" : res.ok ? "saved" : "save failed";
}
["x","y","z","roll","pitch","yaw"].forEach(name => {
  const r = document.getElementById("a_" + name);
  const n = document.getElementById("n_" + name);
  if (!r || !n) return;
  r.oninput = () => { n.value = r.value; scheduleAlign(); };
  n.oninput = () => { r.value = n.value; scheduleAlign(); };
});
document.getElementById("alignSave").onclick = saveAlign;
function paintGap() {
  const box = document.getElementById("gapPlot");
  if (!box) return;
  if (typeof mold === "undefined" || mold !== "thin" || !data || !data.gap_t) {
    box.style.display = "none";
    return;
  }
  box.style.display = "block";
  const tagT = [], tagY = [];
  data.gap_t.forEach((t, i) => { if (data.gap_tag[i]) { tagT.push(t); tagY.push(data.gap_mm[i]); } });
  const shapes = [];
  if (data.gap_window != null) {
    shapes.push({type: "rect", xref: "x", yref: "paper", x0: 0, x1: data.gap_window, y0: 0, y1: 1,
      fillcolor: "rgba(255,196,80,0.18)", line: {width: 0}});
  }
  Plotly.react(box, [
    {x: data.gap_t, y: data.gap_mm, mode: "lines", name: "distance", line: {color: "#4aa3ff", width: 1.5}},
    {x: tagT, y: tagY, mode: "markers", name: "tag", marker: {color: "#3ddc97", size: 8}}
  ], {
    margin: {l: 48, r: 16, t: 28, b: 36}, height: 190,
    paper_bgcolor: "#111", plot_bgcolor: "#111", font: {color: "#ccc", size: 11},
    title: {text: data.gap_note || "", font: {size: 12}},
    xaxis: {title: "s", gridcolor: "#333"}, yaxis: {title: "mm", gridcolor: "#333"},
    legend: {orientation: "h", y: 1.15},
    shapes: shapes
  }, {displayModeBar: false, responsive: true});
}
function paintInput() {
  ["inRot","inFrame","inFoot"].forEach(id => {
    const box = document.getElementById(id);
    if (!box) return;
    if (typeof mold === "undefined" || mold !== "thin" || !data || !data.in_t) {
      box.style.display = "none";
      return;
    }
    box.style.display = "block";
  });
  if (typeof mold === "undefined" || mold !== "thin" || !data || !data.in_t) return;
  const dark = {paper_bgcolor: "#111", plot_bgcolor: "#111", font: {color: "#ccc", size: 11},
    margin: {l: 48, r: 16, t: 28, b: 28}, height: 160, legend: {orientation: "h", y: 1.18},
    xaxis: {title: "s", gridcolor: "#333"}, yaxis: {gridcolor: "#333"}};
  const shapes = [];
  if (data.gap_window != null) {
    shapes.push({type: "rect", xref: "x", yref: "paper", x0: 0, x1: data.gap_window, y0: 0, y1: 1,
      fillcolor: "rgba(255,196,80,0.18)", line: {width: 0}});
  }
  const lay = (title, extra) => Object.assign({}, dark, {title: {text: title, font: {size: 12}}, shapes: shapes}, extra || {});
  Plotly.react("inRot", [
    {x: data.in_t, y: data.in_ident, name: "zero ΔR", line: {color: "#ffb020", width: 1.4}},
    {x: data.in_t, y: data.in_rzero, name: "R_zero", line: {color: "#e06c75", width: 1.4}}
  ], lay(data.in_rot_note || "zero action vs demo tool"), {displayModeBar: false, responsive: true});
  Plotly.react("inFrame", [
    {x: data.in_t, y: data.in_normal, name: "frame z vs outward", line: {color: "#8bd5ff", width: 1.4}},
    {x: data.in_t, y: data.in_head, name: "t1 vs travel", line: {color: "#c3e88d", width: 1.4}}
  ], lay(data.in_frame_note || "field frame"), {displayModeBar: false, responsive: true});
  const color = data.in_face.map(f => f === "cap" ? "#e06c75" : f === "apron" ? "#ffb020" : "#4aa3ff");
  Plotly.react("inFoot", [
    {x: data.in_t, y: data.in_foot, name: "foot to TCP", mode: "markers",
      marker: {color: color, size: 4}, text: data.in_face, hovertemplate: "%{x}s %{y} mm %{text}"}
  ], lay(data.in_foot_note || "foot face: blue wall, red cap, gold apron"), {displayModeBar: false, responsive: true});
}
function paintWall() {
  ["wallFoot", "oodPlot"].forEach(id => {
    const box = document.getElementById(id);
    if (!box) return;
    const show = typeof mold !== "undefined" && mold === "thin" && data && data.wall_t;
    box.style.display = show ? "block" : "none";
  });
  if (typeof mold === "undefined" || mold !== "thin" || !data || !data.wall_t) return;
  const dark = {paper_bgcolor: "#111", plot_bgcolor: "#111", font: {color: "#ccc", size: 11},
    margin: {l: 48, r: 48, t: 36, b: 28}, height: 180, legend: {orientation: "h", y: 1.22},
    xaxis: {title: "s", gridcolor: "#333"}, yaxis: {gridcolor: "#333"}};
  const shapes = [];
  if (data.gap_window != null) {
    shapes.push({type: "rect", xref: "x", yref: "paper", x0: 0, x1: data.gap_window, y0: 0, y1: 1,
      fillcolor: "rgba(255,196,80,0.18)", line: {width: 0}});
  }
  const lay = (title, extra) => Object.assign({}, dark, {title: {text: title, font: {size: 12}}, shapes: shapes}, extra || {});
  const color = (data.wall_face || []).map(f => f === "cap" ? "#e06c75" : f === "apron" ? "#ffb020" : "#4aa3ff");
  Plotly.react("wallFoot", [
    {x: data.wall_t, y: data.wall_mm, name: "wall foot", mode: "markers",
      marker: {color: color, size: 4}, text: data.wall_face, hovertemplate: "%{x}s %{y} mm %{text}"}
  ], lay(data.wall_note || "wall-only foot"), {displayModeBar: false, responsive: true});
  const traces = [
    {x: data.ood_t, y: data.ood_causal, name: "causal |ΔR|", mode: "lines", line: {color: "#8bd5ff", width: 1.4}},
    {x: data.ood_ht || data.ood_t, y: data.ood_heading, name: "heading |R|", mode: "lines", line: {color: "#e06c75", width: 1.4}}
  ];
  (data.ood_names || []).forEach((name, i) => {
    traces.push({x: data.ood_t, y: data.ood_y[i], name: name, yaxis: "y2", mode: "lines", line: {width: 1}});
  });
  Plotly.react("oodPlot", traces, lay(data.ood_note || "predicted angle", {
    yaxis: {title: "deg", gridcolor: "#333"},
    yaxis2: {title: "train range", overlaying: "y", side: "right", range: [-0.15, 1.4], gridcolor: "#333", showgrid: false}
  }), {displayModeBar: false, responsive: true});
}
function footTraces(traces, t) {
  if (typeof mold === "undefined" || mold !== "thin" || !data || !data.foot || !data.foot.length) return;
  const colors = {wall: "#4aa3ff", cap: "#e06c75", back: "#c792ea", apron: "#ffb020"};
  const groups = {wall: [], cap: [], back: [], apron: []};
  data.foot.forEach((p, i) => {
    const name = (data.foot_face && data.foot_face[i]) || "wall";
    (groups[name] || groups.wall).push(p);
  });
  Object.keys(colors).forEach(name => {
    const pts = groups[name];
    if (!pts.length) return;
    traces.push({type: "scatter3d", mode: "markers", name: name,
      x: pts.map(p => p[0]), y: pts.map(p => p[1]), z: pts.map(p => p[2]),
      marker: {size: 3, color: colors[name]}});
  });
  const lx = [], ly = [], lz = [], ax = [], ay = [], az = [];
  for (let i = 0; i < data.foot.length; i++) {
    const a = data.path[i], b = data.foot[i], v = data.foot_z[i];
    if (!a || !b) continue;
    lx.push(a[0], b[0], null); ly.push(a[1], b[1], null); lz.push(a[2], b[2], null);
    if (!v) continue;
    ax.push(b[0], b[0] + 0.02 * v[0], null);
    ay.push(b[1], b[1] + 0.02 * v[1], null);
    az.push(b[2], b[2] + 0.02 * v[2], null);
  }
  traces.push({type: "scatter3d", mode: "lines", name: "TCP–foot", showlegend: false,
    x: lx, y: ly, z: lz, line: {color: "rgba(180,180,180,0.35)", width: 1}});
  traces.push({type: "scatter3d", mode: "lines", name: "frame z",
    x: ax, y: ay, z: az, line: {color: "rgba(125,211,252,0.55)", width: 2}});
  const i = nearest(data.path_t, t);
  const a = data.path[i], b = data.foot[i], v = data.foot_z[i];
  if (!a || !b) return;
  traces.push({type: "scatter3d", mode: "markers", name: "TCP now", showlegend: false,
    x: [a[0]], y: [a[1]], z: [a[2]], marker: {size: 8, color: "#ffffff"}});
  traces.push({type: "scatter3d", mode: "markers", name: "foot now", showlegend: false,
    x: [b[0]], y: [b[1]], z: [b[2]], marker: {size: 8, color: "#ffe14a"}});
  traces.push({type: "scatter3d", mode: "lines", name: "now", showlegend: false,
    x: [a[0], b[0]], y: [a[1], b[1]], z: [a[2], b[2]], line: {color: "#ffe14a", width: 6}});
  if (v) traces.push({type: "scatter3d", mode: "lines", name: "z now", showlegend: false,
    x: [b[0], b[0] + 0.04 * v[0]], y: [b[1], b[1] + 0.04 * v[1]], z: [b[2], b[2] + 0.04 * v[2]],
    line: {color: "#7dd3fc", width: 6}});
}
"""


def _patch_script(page: str) -> str:
    old = """(async function init() {
  const meta = await (await fetch("/api/meta")).json();"""
    new = """let mold = "training";
document.getElementById("mold").onchange = () => { mold = document.getElementById("mold").value; init(); };
async function init() {
  const bar = document.getElementById("alignBar");
  if (bar) bar.style.display = mold === "thin" ? "flex" : "none";
  const variant = document.getElementById("variant");
  if (variant && !variant._bound) { variant._bound = true; variant.onchange = () => show(); }
  const meta = await (await fetch("/api/meta?mold=" + mold)).json();"""
    if old not in page:
        raise SystemExit("init() patch missed")
    page = page.replace(old, new, 1)
    page = page.replace("  show();\n})();", "  show();\n}\ninit();", 1)
    page = page.replace(
        "</script>\n</body>",
        r"""
function paintWrist(t) {
  const img = document.getElementById("wristImg");
  const cap = document.getElementById("wristCap");
  const strip = document.getElementById("strip");
  if (!img || !data || !data.jpg || !data.jpg.length) return;
  let i = 0, best = 1e9;
  (data.rgb_t || []).forEach((rt, k) => { const d = Math.abs(rt - t); if (d < best) { best = d; i = k; } });
  img.src = "/api/rgb?mold=" + mold + "&ep=" + data.episode + "&i=" + i;
  const n = (data.tag && data.tag[i]) || 0;
  const mean = data.pix_mean ? data.pix_mean[i] : "";
  cap.textContent = (data.rgb_t[i] || 0).toFixed(2) + " s  mean " + mean
    + (n > 0 ? "  AprilTag " + n : "  no AprilTag");
  if (strip._n !== data.jpg.length) {
    strip.innerHTML = data.jpg.map((src, k) =>
      "<img data-k='" + k + "' src='" + src + "'" + (data.tag[k] > 0 ? " class='tag'" : "") + ">").join("");
    strip._n = data.jpg.length;
    strip.onclick = (ev) => {
      const k = ev.target && ev.target.dataset ? +ev.target.dataset.k : NaN;
      if (!Number.isFinite(k) || !data.path_t) return;
      fi = nearest(data.path_t, data.rgb_t[k]);
      document.getElementById("time").value = fi;
      paintCursor();
    };
  }
  strip.querySelectorAll("img").forEach((el, k) => el.classList.toggle("on", k === i));
  const run = (data.runs || []).find(r => r.a_p);
  const box = document.getElementById("tilt");
  if (!run || !box) return;
  box.style.display = "block";
  const mark = {type: "line", xref: "x", yref: "y domain", x0: t, x1: t, y0: 0, y1: 1, line: {color: "#fff", width: 1}};
  Plotly.react("tilt", [
    {x: run.q_t, y: run.a_l, name: "a label", mode: "lines", line: {color: "#4aa3ff"}},
    {x: run.q_t, y: run.a_p, name: "a pred", mode: "lines", line: {color: "#ff4d6a", dash: "dash"}},
    {x: run.q_t, y: run.b_l, name: "b label", mode: "lines", xaxis: "x2", yaxis: "y2", line: {color: "#3ddc97"}},
    {x: run.q_t, y: run.b_p, name: "b pred", mode: "lines", xaxis: "x2", yaxis: "y2", line: {color: "#ffd23f", dash: "dash"}}
  ], {
    paper_bgcolor: "#111", plot_bgcolor: "#1c1c1c", font: {color: "#ddd"},
    margin: {l: 48, r: 12, t: 18, b: 24}, legend: {orientation: "h", y: 1.2},
    shapes: [mark, Object.assign({}, mark, {xref: "x2", yref: "y2 domain"})],
    xaxis: {domain: [0, 1]}, yaxis: {title: "a deg", domain: [0.55, 1]},
    xaxis2: {anchor: "y2", matches: "x", domain: [0, 1]},
    yaxis2: {title: "b deg", domain: [0, 0.45]}
  }, {responsive: true});
}
</script>
</body></html>
""",
        1,
    )
    return page


def main() -> None:
    _check()
    page = _patch_script(viewer.PAGE)
    page = page.replace(
        "#view3d { height:42vh; }",
        "#view3d { height:42vh; }\n"
        "#wrist { display:flex; gap:8px; align-items:flex-start; padding:4px 12px; }\n"
        "#wristImg { width:224px; height:224px; background:#222; object-fit:contain; }\n"
        "#strip { display:flex; gap:2px; overflow-x:auto; max-width:calc(100vw - 260px); }\n"
        "#strip img { height:48px; border:2px solid #333; }\n"
        "#strip img.tag { border-color:#3ddc97; }\n"
        "#strip img.on { border-color:#ffd23f; }\n"
        "#tilt { height:18vh; }\n"
        "#alignBar { flex-wrap:wrap; gap:8px; }\n"
        "#alignBar label { display:flex; align-items:center; gap:4px; }\n"
        "#alignBar input[type=range] { width:120px; }\n"
        "#alignBar input[type=number] { width:72px; }\n"
        "#gapPlot { height:190px; }\n"
        "#inRot, #inFrame, #inFoot { height:160px; }\n"
        "#wallFoot, #oodPlot { height:180px; }\n",
        1,
    )
    page = page.replace(
        '<span>episode</span><select id="ep"></select>',
        '<span>mold</span><select id="mold"><option value="training">training stringer</option>'
        '<option value="thin">thin stringer</option></select>\n'
        '  <span>episode</span><select id="ep"></select>',
        1,
    )
    page = page.replace(
        '<div id="view3d"></div>',
        '<div id="wrist"><img id="wristImg" alt="D405"><div><div id="wristCap"></div>'
        '<div id="strip"></div></div></div><div id="tilt"></div>'
        + _align_bar()
        + '<div id="gapPlot" style="display:none"></div>'
        + '<div id="inRot" style="display:none"></div>'
        + '<div id="inFrame" style="display:none"></div>'
        + '<div id="inFoot" style="display:none"></div>'
        + '<div id="wallFoot" style="display:none"></div>'
        + '<div id="oodPlot" style="display:none"></div>'
        + '<div id="view3d"></div>',
        1,
    )
    page = page.replace(
        'fetch("/api/episode?ep="',
        'fetch("/api/episode?mold=" + mold + "&v=" + variantName() + "&ep="',
    )
    page = page.replace(
        'fetch("/api/cursor?ep="',
        'fetch("/api/cursor?mold=" + mold + "&v=" + variantName() + "&ep="',
    )
    page = page.replace(
        'fetch("/api/heading?ep="',
        'fetch("/api/heading?mold=" + mold + "&v=" + variantName() + "&ep="',
    )
    page = page.replace(
        "const cam = liveCamera();\n  const cursor = await",
        "const cam = liveCamera();\n  paintWrist(t);\n  const cursor = await",
        1,
    )
    page = page.replace(
        '<button id="geomPoint">virtual target</button>',
        '<button id="geomPoint">virtual target</button>\n'
        '  <button id="geomSplit">split score</button>',
        1,
    )
    page = page.replace(
        "function setGeom(next) {\n"
        "  geomMode = next;\n"
        '  document.getElementById("geomFoot").classList.toggle("on", geomMode === "foot");\n'
        '  document.getElementById("geomPoint").classList.toggle("on", geomMode === "point");\n'
        "  if (data) paintCursor();\n"
        "}\n"
        'document.getElementById("geomFoot").onclick = () => setGeom("foot");\n'
        'document.getElementById("geomPoint").onclick = () => setGeom("point");',
        "function setGeom(next) {\n"
        "  geomMode = next;\n"
        '  document.getElementById("geomFoot").classList.toggle("on", geomMode === "foot");\n'
        '  document.getElementById("geomPoint").classList.toggle("on", geomMode === "point");\n'
        '  document.getElementById("geomSplit").classList.toggle("on", geomMode === "split");\n'
        "  if (data) paintCursor();\n"
        "}\n"
        'document.getElementById("geomFoot").onclick = () => setGeom("foot");\n'
        'document.getElementById("geomPoint").onclick = () => setGeom("point");\n'
        'document.getElementById("geomSplit").onclick = () => setGeom("split");',
        1,
    )
    page = page.replace(
        '  const footMode = geomMode === "foot" && drawn.some(r => r.label_foot && r.pred_foot);\n'
        "  const traces = [];\n"
        "  if (footMode) {",
        '  const splitMode = geomMode === "split" && picked && picked.tcp_exec;\n'
        '  const footMode = !splitMode && geomMode === "foot" && drawn.some(r => r.label_foot && r.pred_foot);\n'
        "  const traces = [];\n"
        "  if (splitMode) {\n"
        '    const demo = scatter(data.path, "demo TCP", "#4aa3ff", 5, data.path_break, data.path_t);\n'
        "    demo.showlegend = true;\n"
        "    traces.push(demo);\n"
        '    const predPts = span === "full" && picked.tcp_full ? picked.tcp_full : picked.tcp_exec;\n'
        '    const predCut = span === "full" && picked.tcp_full ? picked.tcp_full_break : picked.tcp_exec_break;\n'
        '    const vel = scatter(predPts, "v only", "#ff4d6a", 5, predCut);\n'
        "    vel.showlegend = true;\n"
        "    traces.push(vel);\n"
        '    if (item && item.label_xyz) traces.push(scatter(item.label_xyz, "demo chunk", "#9ad", 6));\n'
        '    if (item && item.tcp_v) traces.push(scatter(item.tcp_v, "this chunk", "#ffe14a", 8));\n'
        "  } else if (footMode) {",
        1,
    )
    page = page.replace(
        "  if (!footMode && picked && shown[picked.label] !== false) {",
        "  if (!footMode && !splitMode && picked && shown[picked.label] !== false) {",
        1,
    )
    page = page.replace(
        'line(run.q_t, run.n_l, c + " label", DIM.n.label, "x3", null, br),',
        'line(run.q_t, (geomMode === "split" && run.dn_k) ? run.dn_k : run.n_l,'
        ' (geomMode === "split" && run.dn_k) ? "δn k=3000" : c + " label", DIM.n.label, "x3", null, br),',
        1,
    )
    page = page.replace(
        'const cursor = await (await fetch("/api/cursor?mold=" + mold + "&v=" + variantName() + "&ep=" + data.episode + "&q=" + j + "&span=" + span)).json();',
        'const cursor = await (await fetch("/api/cursor?mold=" + mold + "&v=" + variantName() + "&ep=" + data.episode + "&q=" + j + "&span=" + span)).json();\n'
        "  poseCursor(cursor);",
        1,
    )
    page = page.replace(
        'document.getElementById("time").max = Math.max(0, data.path_t.length - 1);\n  fi = 0;',
        'document.getElementById("time").max = Math.max(0, data.path_t.length - 1);\n  fi = 0;\n  stashAlign();\n  paintGap();\n  paintInput();\n  paintWall();',
        1,
    )
    page = page.replace(
        'await Plotly.react("view3d", traces, sceneLayout(), {responsive: true});',
        'footTraces(traces, t);\n  await Plotly.react("view3d", traces, sceneLayout(), {responsive: true});',
        1,
    )
    page = page.replace("</script>\n</body>", _ALIGN_JS + "\n</script>\n</body>", 1)
    if "alignBar" not in page or "stashAlign" not in page or "poseCursor" not in page or "variantName" not in page:
        raise SystemExit("align panel did not apply")
    if "geomSplit" not in page or "tcp_exec" not in page or "δn k=3000" not in page:
        raise SystemExit("split-score patch did not apply")
    if "paintWrist" not in page or "mold=" not in page or "paintGap" not in page or "gapPlot" not in page or "paintInput" not in page or "paintWall" not in page or "footTraces" not in page:
        raise SystemExit("page patch did not apply")
    viewer.PAGE = page
    original = viewer.episode_payload
    ab = json.loads((THIN_EXPORT / "ab.json").read_text())

    def episode_payload(exports, episode, frames=None):
        mold = getattr(viewer.Handler, "mold", "training")
        variant = _variant(getattr(viewer.Handler, "variant", "A"))
        try:
            out = original(exports, episode, frames=frames)
        except KeyError:
            if mold != "thin":
                raise
            out = _bare_payload(episode, variant)
        else:
            _attach_split(out, exports, int(episode))
            if mold == "thin":
                _apply_variant(out, int(episode), variant)
                for run in out.get("runs") or []:
                    if str(run.get("label", "")).startswith("r4vt ep25"):
                        run.update(ab)
        if mold == "thin":
            out.update(gap_payload(int(episode), variant))
            if int(episode) == 0:
                if variant == "A":
                    mid = json.loads(
                        (HERE.parent / "thin_variants" / "contact_half.json").read_text()
                    )["episodes"]["C"]["t_mid_s"]
                else:
                    _ep, _label, stamp = _thin_row(episode)
                    mid = halves(load_variant(stamp, variant))["t_mid_s"]
                out["gap_window"] = mid
                out["gap_note"] += "  | variant A predictions, scored 0 to first-half contact"
            elif int(episode) == 1:
                out["gap_note"] += "  | variant A predictions, scored full recording"
            out.update(_input_view(int(episode)))
            out.update(_wall_view(int(episode)))
            if out.get("path"):
                import foot_overlay
                out.update(foot_overlay.overlay(out["path"]))
        try:
            images = _images(mold, int(episode))
        except Exception as exc:
            out["image_error"] = str(exc)
            return out
        out.update({k: images[k] for k in ("rgb_t", "jpg", "tag", "pix_mean")})
        return out

    viewer.episode_payload = episode_payload

    class Handler(viewer.Handler):
        mold = "training"
        molds = {}

        def do_GET(self):
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            args = dict(part.split("=", 1) for part in query.split("&") if "=" in part)
            mold = args.get("mold", "training")
            if mold not in self.molds:
                mold = "training"
            type(self).mold = mold
            type(self).variant = _variant(args.get("v", "A"))
            viewer.Handler.exports = self.molds[mold]["exports"]
            viewer.Handler.mesh = self.molds[mold]["mesh"]
            path = self.path.split("?", 1)[0]
            if path == "/api/meta" and mold == "thin":
                body = {
                    "episodes": [ep for ep, _label, _stamp in THIN_EPS],
                    "splits": {str(ep): label for ep, label, _stamp in THIN_EPS},
                    "mesh": self.mesh,
                    "heading": self.heading is not None,
                }
                self._send(200, json.dumps(body).encode(), "application/json")
                return
            if path == "/api/rgb":
                try:
                    images = _images(mold, int(args.get("ep", "0")))
                    frame = images["full"][int(args.get("i", "0"))]
                except Exception:
                    self._send(404, b"", "text/plain")
                    return
                self._send(200, frame, "image/jpeg")
                return
            if path == "/api/heading" and mold == "thin":
                self._send(200, b'{"t":[]}', "application/json")
                return
            if path == "/api/align":
                mm = _nums(args, ("x", "y", "z"))
                rpy = _nums(args, ("roll", "pitch", "yaw"))
                try:
                    episode = int(args.get("ep", "0"))
                except ValueError:
                    episode = 0
                self._send(
                    200,
                    json.dumps(align_report(episode, mm, rpy, type(self).variant)).encode(),
                    "application/json",
                )
                return
            if path == "/api/cursor":
                try:
                    episode = int(args.get("ep", "0"))
                    query_i = int(args.get("q", "0"))
                except ValueError:
                    self._send(400, b"{}", "application/json")
                    return
                try:
                    body = viewer.cursor_payload(
                        self.exports, episode, query_i, frames=getattr(self, "frames", None), span=args.get("span", "exec")
                    )
                except KeyError:
                    body = {"runs": []}
                else:
                    _attach_cursor_split(body, self.exports, episode, query_i, args.get("span", "exec"))
                self._send(200, json.dumps(body).encode(), "application/json")
                return
            super().do_GET()

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path != "/api/align_save":
                self._send(404, b"", "text/plain")
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._send(400, b"{}", "application/json")
                return
            mm = [float(body.get(k, 0.0)) for k in ("x", "y", "z")]
            rpy = [float(body.get(k, 0.0)) for k in ("roll", "pitch", "yaw")]
            try:
                episode = int(body.get("ep", 0))
            except (TypeError, ValueError):
                episode = 0
            status, report = save_align(align_report(episode, mm, rpy, _variant(body.get("v", "A"))))
            self._send(status, json.dumps(report).encode(), "application/json")

    viewer.Handler = Handler
    print("[both-view] loading exports", flush=True)
    Handler.molds = {
        "training": {
            "exports": [viewer.load_export(path) for path in TRAIN_EXPORTS],
            "mesh": _mesh(TRAIN_MESH),
        },
        "thin": {
            "exports": [
                viewer.load_export(path)
                for path in sorted(HELD_EXPORT.iterdir())
                if path.is_dir() and (path / "settings.json").is_file()
            ] if HELD_EXPORT.is_dir() else [],
            "mesh": _mesh(THIN_MESH),
        },
    }
    for pack in Handler.molds.values():
        labels = [exp["label"] for exp in pack["exports"]]
        if len(set(labels)) != len(labels):
            for exp in pack["exports"]:
                exp["label"] = f"{exp['label']} [{exp['dir'].name}]"
    print("[both-view] training", [e["label"] for e in Handler.molds["training"]["exports"]], flush=True)
    from heading_tilt_layer import HeadingLayer

    Handler.heading = HeadingLayer(HEADING_E25, CAUSAL_E199)
    Handler.exports = Handler.molds["training"]["exports"]
    Handler.mesh = Handler.molds["training"]["mesh"]
    # One payload proves the image stats are attached and not black.
    sample = _images("thin", 0)
    if sample["pix_mean"][50] < 20:
        raise SystemExit(f"thin frame 50 mean {sample['pix_mean'][50]} is dark")
    print(f"[both-view] http://127.0.0.1:{PORT}  thin frame50 mean {sample['pix_mean'][50]}", flush=True)
    from http.server import ThreadingHTTPServer

    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
