#!/usr/bin/env python3
"""Closed-loop hardware replay viewer. Reads rollout logs. Does not write them.

  python UMIFT_Data/surface_frames_scripts/view_rollout.py

Binds 127.0.0.1 on 8810, then 8811, then 8812. Does not use 8766, 8802, or 8803.
"""

from __future__ import annotations

import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path("/mnt/data/umidata/ft/vtevy/rollouts")
EXTRINSICS = (
    "/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/"
    "PyriteConfig/tasks/umift/layup_workcell_extrinsics_20261116_orange.yaml"
)
DEMO_ZARR = (
    "/mnt/data/umidata/ft/mold_aware_datasets/"
    "stringer_v3.5_fixb_odom_corr/r4vt_field_v2n_v3r_liftos/acp_replay_buffer.zarr"
)
LIFTOS_CKPT = (
    "/mnt/data/umidata/ft/training_outputs/"
    "oct_v35_h4s_r4vt_field_v2n_v3r_liftos/latest.ckpt"
)
ROLL_INDEX = 6016
DEMO_START = 1220
PORTS = (8810, 8811, 8812)

os.environ.setdefault("UMIFT_WORKCELL_EXTRINSICS", EXTRINSICS)

_LOCK = threading.Lock()
_RUNS: dict[str, dict] = {}
_POLICY = None
_RUNTIME = None
_DEMO: dict = {"ready": False, "by_h": {}, "error": ""}
_LIVE: dict = {}
_JPEG: dict[tuple, bytes] = {}
_JPEG_LOCK = threading.Lock()
_MESH: dict | None = None
FIELD_NPZ = (
    "/mnt/data/umidata/ft/mold_aware_datasets/"
    "stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
MESH_TRI_CAP = 4000


def _round(value):
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return round(number, 5)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _twist(field_R, tool_R) -> float:
    from scipy.spatial.transform import Rotation

    relative = np.asarray(field_R, dtype=np.float64).reshape(3, 3).T @ np.asarray(
        tool_R, dtype=np.float64
    ).reshape(3, 3)
    return float(np.degrees(Rotation.from_matrix(relative).as_rotvec()[2]))


PINNED = (
    "2026-11-24_liftos_hw_pe_guard",
    "2026-11-24_liftos_hw_pe_dnbias",
    "2026-11-24_liftos_hw_pe_dnbias_i5600",
    "2026-11-24_liftos_hw_pose_ensemble",
    "2026-11-24_liftos_hw_pe_sd6",
)


def _rollout_dirs() -> list[Path]:
    found = {}
    for path in sorted(ROOT.glob("2026-11-24_*")):
        if not path.is_dir():
            continue
        if (path / "horizon_0").exists() or any(path.glob("*/horizon_0")):
            found[path.name] = path
    ordered = [found.pop(name) for name in PINNED if name in found]
    ordered.extend(found[name] for name in sorted(found))
    return ordered


def _log_root(path: Path) -> Path:
    if (path / "horizon_0").exists():
        return path
    nested = sorted(path.glob("*/horizon_0"))
    return nested[0].parent if nested else path


def _horizon_names(group) -> list[str]:
    names = [key for key in group.keys() if str(key).startswith("horizon_")]
    return sorted(names, key=lambda name: int(str(name).split("_")[1]))


def _load_robot(folder: str):
    path = Path(folder) / "robot_data_0.json"
    if not path.is_file():
        return None
    text = re.sub(r",(\s*[}\]])", r"\1", path.read_text())
    rows = json.loads(text)
    time_s = np.array([row["robot_time_stamps"] for row in rows], dtype=np.float64) / 1000.0
    pose = np.array([row["ts_pose_fb"] for row in rows], dtype=np.float64)
    wrench = np.array(
        [row.get("robot_wrench") or [0, 0, 0, 0, 0, 0] for row in rows],
        dtype=np.float64,
    )
    return {"t": time_s, "pose": pose, "wrench": wrench}


def _nearest(times: np.ndarray, stamp: float) -> int:
    index = int(np.clip(np.searchsorted(times, stamp), 1, len(times) - 1))
    if abs(times[index - 1] - stamp) < abs(times[index] - stamp):
        index -= 1
    return index


def _base_to_fix(points, rotation_rows, T):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    fixed = points @ T[:3, :3].T + T[:3, 3]
    rotations = None
    if rotation_rows is not None:
        rotations = np.einsum("ij,njk->nik", T[:3, :3], rotation_rows)
    return fixed, rotations


def _pose_bundle(pose, T):
    from fixb_labels import pose7_to_Rp

    rotation, position = pose7_to_Rp(np.asarray(pose, dtype=np.float64).reshape(-1, 7))
    fixed, fixed_R = _base_to_fix(position, rotation, T)
    return fixed, fixed_R


def _demo_path():
    import zarr
    from fixb_labels import pose7_to_Rp

    episode = zarr.open_group(DEMO_ZARR, mode="r")["data/episode_9"]
    pose = np.asarray(episode["ts_pose_fb_0"])
    _, position = pose7_to_Rp(pose)
    step = max(1, len(position) // 500)
    index = np.arange(0, len(position), step)
    if index[-1] != ROLL_INDEX and ROLL_INDEX < len(position):
        index = np.append(index, ROLL_INDEX)
    return {
        "xyz": position[index],
        "roll": position[ROLL_INDEX],
        "pose": pose,
        "time": np.asarray(episode["robot_time_stamps_0"]).reshape(-1),
        "rgb_t": np.asarray(episode["rgb_time_stamps_0"]).reshape(-1),
        "rgb": episode["rgb_0"],
        "wrench": np.asarray(episode["wrench_left_0"]),
        "wrench_t": np.asarray(episode["wrench_time_stamps_0"]).reshape(-1),
        "sd": np.asarray(episode["mold_signed_distance_0"]).reshape(-1),
        "normal": np.asarray(episode["mold_normal_0"]),
        "rot6d": np.asarray(episode["mold_R_field_rot6d_0"]),
        "field": np.asarray(episode["mold_R_field_0"]).reshape(-1, 3, 3),
    }


def _series(path: Path) -> dict:
    import zarr
    from PyriteConfig.tasks.umift.workcell_frames import load_workcell_extrinsics
    from fixb_labels import pose7_to_Rp
    from rot6d import rot6d_rows_to_R

    group = zarr.open_group(str(_log_root(path)), mode="r")
    names = _horizon_names(group)
    T = load_workcell_extrinsics().T_fixture_from_right_base
    robot = _load_robot(str(group.attrs.get("episode_data_folder", "")))
    rows = []
    for name in names:
        action = group[name]["action"]
        stamps = np.asarray(action["timestamps_s"]).reshape(-1)
        stamp = float(stamps[0])
        virtual = np.asarray(action["ts_virtual_targets_0"])
        foot = np.asarray(action["foot"]) if "foot" in action else virtual[:, :3]
        if foot.ndim == 2 and foot.shape[1] >= 3:
            foot_xyz = foot[:, :3]
        else:
            foot_xyz = virtual[:, :3]
        foot_fix, _ = _base_to_fix(foot_xyz[:1], None, T)
        cmd_fix, cmd_R = _pose_bundle(virtual, T)
        nominal = (
            np.asarray(action["ts_nominal_targets_0"])
            if "ts_nominal_targets_0" in action
            else None
        )
        if nominal is not None:
            nom_fix, _ = _pose_bundle(nominal, T)
            dn_target = foot_fix[0] + (cmd_fix[0] - nom_fix[0])
        else:
            dn_target = None
        measured_path = []
        if robot is not None:
            stop = float(stamps[-1]) + 0.05
            take = (robot["t"] >= stamp - 0.02) & (robot["t"] <= stop)
            if np.any(take):
                measured_path, _ = _pose_bundle(robot["pose"][take][::8], T)
                measured_path = measured_path.tolist()
        geom = np.asarray(group[name]["obs_policy"]["robot0_mold_geom"])
        field_R = rot6d_rows_to_R(geom[-1, 4:10])
        normal = geom[-1, 1:4]
        scheduled = float(np.asarray(action["delta_n"]).reshape(-1)[0])
        measured_twist = None
        force = None
        if robot is not None:
            index = _nearest(robot["t"], stamp)
            meas_fix, meas_R = _pose_bundle(robot["pose"][index], T)
            measured_twist = _twist(field_R, meas_R[0])
            force_fix = meas_R[0] @ robot["wrench"][index, :3]
            force = abs(float(np.dot(force_fix, normal)))
        pos = np.asarray(group[name]["obs_policy"]["robot0_eef_pos"])
        rows.append(
            {
                "h": int(str(name).split("_")[1]),
                "t": stamp,
                "cmd": cmd_fix[0],
                "cmd_path": cmd_fix.tolist(),
                "meas_path": measured_path,
                "cmd_twist": _twist(field_R, cmd_R[0]),
                "meas_twist": measured_twist,
                "dn_sched": scheduled * 1e3,
                "sd": float(geom[-1, 0]) * 1e3,
                "fn": force,
                "foot": foot_fix[0],
                "dn_target": None if dn_target is None else dn_target,
                "vt": cmd_fix[0],
                "pos": pos[-1],
            }
        )
    t0 = rows[0]["t"] if rows else 0.0
    video_t0 = float(robot["t"][0]) if robot is not None else t0
    cameras = _log_root(path) / "cameras"
    demo = _demo_path()
    start_pos = rows[0]["pos"] if rows else np.zeros(3)
    _, demo_pos = pose7_to_Rp(demo["pose"][DEMO_START : DEMO_START + 1])
    from field_v2_dataset import mold_local
    import zarr as _zarr

    spec = _zarr.open_group(DEMO_ZARR, mode="r").attrs["field_v2_axes"]
    demo_local = mold_local(
        demo_pos, np.asarray(spec["origin"]), np.asarray(spec["axes"])
    )[0]
    overlay = bool(np.linalg.norm(start_pos - demo_local) < 0.015)
    return {
        "name": path.name,
        "ckpt": str(group.attrs.get("ckpt_path", "")),
        "exec_h": int(group.attrs.get("sparse_execution_horizon", 6)),
        "slow": float(group.attrs.get("slow_down_factor", 2.0)),
        "decode_dt": float(group.attrs.get("decode_dt_s", 1.0 / 15.0)),
        "overlay": overlay,
        "video_t0": video_t0,
        "wrist": (cameras / "wrist_d405.mp4").is_file(),
        "center": (cameras / "orbbec_center.mp4").is_file(),
        "logged_wrist": "rgb_0" in group["horizon_0"]["obs_policy"],
        "h": [row["h"] for row in rows],
        "t": [row["t"] - t0 for row in rows],
        "cmd_twist": [row["cmd_twist"] for row in rows],
        "meas_twist": [row["meas_twist"] for row in rows],
        "dn_sched": [row["dn_sched"] for row in rows],
        "sd": [row["sd"] for row in rows],
        "fn": [row["fn"] for row in rows],
        "foot": [row["foot"].tolist() for row in rows],
        "dn_target": [
            None if row["dn_target"] is None else row["dn_target"].tolist() for row in rows
        ],
        "vt": [row["vt"].tolist() for row in rows],
        "cmd": [row["cmd"].tolist() for row in rows],
        "cmd_path": [row["cmd_path"] for row in rows],
        "meas_path": [row["meas_path"] for row in rows],
        "demo": demo["xyz"][::1].tolist(),
        "roll": demo["roll"].tolist(),
    }


def _get_run(name: str) -> dict:
    path = ROOT / name
    if not path.is_dir() or path.parent != ROOT:
        raise KeyError(name)
    with _LOCK:
        cached = _RUNS.get(name)
    if cached is None:
        cached = _series(path)
        cached["path"] = path
        with _LOCK:
            _RUNS[name] = cached
            start = not cached.get("_bg")
            cached["_bg"] = True
        if start:
            threading.Thread(target=_fill_live, args=(name,), daemon=True).start()
            threading.Thread(target=_prefetch_images, args=(name,), daemon=True).start()
            if cached.get("overlay") and not _DEMO.get("started"):
                _DEMO["started"] = True
                threading.Thread(
                    target=_fill_demo,
                    args=(int(cached["exec_h"]), len(cached["h"])),
                    daemon=True,
                ).start()
    return cached


def _policy():
    global _POLICY
    if _POLICY is not None:
        return _POLICY
    import importlib.util
    import torch

    script = (
        Path(__file__).resolve().parents[2]
        / "PyriteEnvSuites/scripts/run_policy_open_loop_val_export.py"
    )
    spec = importlib.util.spec_from_file_location("ol", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, _ = module.load_policy_and_cfg(LIFTOS_CKPT, device)
    _POLICY = (policy, device)
    return _POLICY


def _runtime():
    global _RUNTIME
    if _RUNTIME is None:
        from mold_aware_runtime import MoldAwareRuntime

        runtime = MoldAwareRuntime.from_assets("r4_world_fixb_tilt", dry_run=False)
        runtime.tcp_relative_rotation = True
        _RUNTIME = runtime
    return _RUNTIME


def _decode(action, pose_base, T, dt: float, field0=None):
    from fixb_labels import pose7_to_Rp

    runtime = _runtime()
    runtime.reset()
    rotation, position = pose7_to_Rp(np.asarray(pose_base, dtype=np.float64).reshape(1, 7))
    position_fix, rotation_fix = _base_to_fix(position, rotation, T)
    runtime.tcp_relative_rotation = True
    decoded = runtime.decode_chunk(
        np.asarray(action, dtype=np.float64),
        position_fix[0],
        rotation_fix[0],
        np.full(len(action), float(dt)),
        apply_safety=True,
    )
    twists = [
        _twist(decoded["R_field"][i], decoded["R_d"][i]) for i in range(len(action))
    ]
    if field0 is not None:
        twists[0] = _twist(field0, decoded["R_d"][0])
    raw = np.asarray(action, dtype=np.float64)
    foot = np.asarray(decoded["x_s"], dtype=np.float64)
    normal = np.asarray(decoded["n_out"], dtype=np.float64)
    delta = np.asarray(decoded["delta_n"], dtype=np.float64).reshape(-1)
    return {
        "xyz": decoded["x_d"],
        "twist": twists,
        "dn": raw[:, 2] * 1e3,
        "foot": foot,
        "dn_target": foot + delta[:, None] * normal,
        "vt": np.asarray(decoded["x_d"], dtype=np.float64),
    }


def _predict_logged(path: Path, horizon: int):
    import torch
    import zarr
    from PyriteConfig.tasks.umift.workcell_frames import load_workcell_extrinsics

    group = zarr.open_group(str(_log_root(path)), mode="r")
    name = f"horizon_{horizon}"
    obs = {
        key: np.asarray(group[name]["obs_policy"][key])
        for key in group[name]["obs_policy"].keys()
        if key != "rgb_0" or True
    }
    policy, device = _policy()
    batched = {
        "sparse": {
            key: torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device).unsqueeze(0)
            for key, value in obs.items()
        }
    }
    robot = _load_robot(str(group.attrs.get("episode_data_folder", "")))
    stamp = float(np.asarray(group[name]["action"]["timestamps_s"]).reshape(-1)[0])
    T = load_workcell_extrinsics().T_fixture_from_right_base
    if robot is None:
        raise RuntimeError("no measured TCP for this rollout")
    pose = robot["pose"][_nearest(robot["t"], stamp)]
    dt = float(group.attrs.get("decode_dt_s", 1.0 / 15.0))
    from rot6d import rot6d_rows_to_R

    field0 = rot6d_rows_to_R(obs["robot0_mold_geom"][-1, 4:10])
    with _LOCK:
        torch.manual_seed(0)
        with torch.no_grad():
            action = policy.predict_action(batched)["sparse"][0].detach().cpu().numpy()
        return _decode(action, pose, T, dt, field0)


def _demo_obs(demo: dict, index: int) -> dict:
    from field_v2_dataset import mold_local
    from fixb_labels import pose7_to_Rp
    from rot6d import so3_to_rot6d_rows
    import zarr

    spec = zarr.open_group(DEMO_ZARR, mode="r").attrs["field_v2_axes"]
    origin = np.asarray(spec["origin"])
    axes = np.asarray(spec["axes"])
    pose_index = np.clip(index - np.arange(50, -1, -1) * 16, 0, len(demo["pose"]) - 1)
    rotation, position = pose7_to_Rp(demo["pose"][pose_index])
    rgb_index = int(np.clip(np.searchsorted(demo["rgb_t"], demo["time"][index]), 0, len(demo["rgb_t"]) - 1))
    rgb_ids = np.clip(rgb_index - np.arange(15, -1, -1) * 4, 0, len(demo["rgb_t"]) - 1)
    wrench_index = int(
        np.clip(np.searchsorted(demo["wrench_t"], demo["time"][index]), 0, len(demo["wrench_t"]) - 1)
    )
    wrench_ids = np.clip(wrench_index - np.arange(117, -1, -1), 0, len(demo["wrench"]) - 1)
    images = np.asarray(demo["rgb"][rgb_ids])
    return {
        "rgb_0": np.moveaxis(images, -1, 1).astype(np.float32) / 255.0,
        "robot0_eef_pos": mold_local(position, origin, axes).astype(np.float32),
        "robot0_eef_rot_axis_angle": so3_to_rot6d_rows(rotation).astype(np.float32),
        "robot0_eef_wrench_left": demo["wrench"][wrench_ids, :6].astype(np.float32),
        "robot0_eef_wrench_right": np.zeros((118, 6), np.float32),
        "robot0_mold_geom": np.concatenate(
            [
                demo["sd"][pose_index, None],
                demo["normal"][pose_index],
                demo["rot6d"][pose_index],
            ],
            axis=1,
        ).astype(np.float32),
    }


def _fill_demo(exec_h: int, n_horizons: int) -> None:
    import torch

    try:
        demo = _demo_path()
        dt = float(np.median(np.diff(demo["time"])))
        stride = max(1, int(round(exec_h * (1.0 / 15.0) / dt)))
        policy, device = _policy()
        for horizon in range(n_horizons):
            with _LOCK:
                if horizon in _DEMO["by_h"]:
                    continue
            index = DEMO_START + horizon * stride
            if index >= len(demo["pose"]) - 2:
                break
            obs = _demo_obs(demo, index)
            batched = {
                "sparse": {
                    key: torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
                    for key, value in obs.items()
                }
            }
            with _LOCK:
                if horizon in _DEMO["by_h"]:
                    continue
                torch.manual_seed(0)
                with torch.no_grad():
                    action = policy.predict_action(batched)["sparse"][0].detach().cpu().numpy()
                decoded = _decode(
                    action, demo["pose"][index], np.eye(4), 1.0 / 15.0, demo["field"][index]
                )
                _DEMO["by_h"][horizon] = {
                    "xyz": decoded["xyz"].tolist(),
                    "twist0": decoded["twist"][0],
                    "dn0": float(decoded["dn"][0]),
                }
        with _LOCK:
            _DEMO["ready"] = True
    except Exception as exc:
        with _LOCK:
            _DEMO["error"] = str(exc)


def _fill_live(name: str) -> None:
    with _LOCK:
        run = _RUNS[name]
    try:
        for horizon in run["h"]:
            with _LOCK:
                if horizon in _LIVE.get(name, {}):
                    continue
            predicted = _predict_logged(run["path"], int(horizon))
            with _LOCK:
                _LIVE.setdefault(name, {})[int(horizon)] = {
                    "dn0": float(predicted["dn"][0]),
                    "twist0": float(predicted["twist"][0]),
                    "xyz": predicted["xyz"].tolist(),
                    "foot": np.asarray(predicted["foot"][0]).tolist(),
                    "dn_target": np.asarray(predicted["dn_target"][0]).tolist(),
                    "vt": np.asarray(predicted["vt"][0]).tolist(),
                }
        with _LOCK:
            _LIVE.setdefault(name, {})["ready"] = True
    except Exception as exc:
        with _LOCK:
            _LIVE.setdefault(name, {})["error"] = str(exc)


def _chunk(name: str, horizon: int) -> dict:
    run = _get_run(name)
    import zarr
    from PyriteConfig.tasks.umift.workcell_frames import load_workcell_extrinsics

    path = run["path"]
    group = zarr.open_group(str(_log_root(path)), mode="r")
    predicted = _predict_logged(path, horizon)
    T = load_workcell_extrinsics().T_fixture_from_right_base
    virtual = np.asarray(group[f"horizon_{horizon}"]["action"]["ts_virtual_targets_0"])
    commanded, _ = _pose_bundle(virtual, T)
    robot = _load_robot(str(group.attrs.get("episode_data_folder", "")))
    stamps = np.asarray(group[f"horizon_{horizon}"]["action"]["timestamps_s"]).reshape(-1)
    measured = []
    if robot is not None:
        stop = float(stamps[-1]) + 0.05
        take = (robot["t"] >= float(stamps[0]) - 0.02) & (robot["t"] <= stop)
        if np.any(take):
            measured, _ = _pose_bundle(robot["pose"][take][::5], T)
            measured = measured.tolist()
    demo_row = None
    with _LOCK:
        demo_row = _DEMO["by_h"].get(horizon) if run["overlay"] else None
        if run["overlay"] and not _DEMO.get("started"):
            _DEMO["started"] = True
            threading.Thread(
                target=_fill_demo,
                args=(int(run["exec_h"]), len(run["h"])),
                daemon=True,
            ).start()
        if not _LIVE.get(name, {}).get("started"):
            _LIVE.setdefault(name, {})["started"] = True
    exec_h = min(int(run["exec_h"]), len(predicted["xyz"]))
    return {
        "exec_h": exec_h,
        "pred": predicted["xyz"].tolist(),
        "pred_twist": predicted["twist"],
        "pred_dn": predicted["dn"].tolist(),
        "cmd": commanded.tolist(),
        "meas": measured,
        "demo": demo_row,
    }


def mold_mesh() -> dict:
    """Fixture-frame mesh that fixb_labels queries. Traces use the same frame."""
    global _MESH
    if _MESH is not None:
        return _MESH
    import open3d as o3d

    field = np.load(FIELD_NPZ)
    vertices = np.asarray(field["vertices"], dtype=np.float64)
    faces = np.asarray(field["faces"], dtype=np.int32)
    if "is_apron" in field.files:
        apron = np.asarray(field["is_apron"], dtype=bool)
        faces = faces[~apron[faces].any(axis=1)]
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    vertices = vertices[used]
    faces = remap[faces]
    if len(faces) > MESH_TRI_CAP:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=MESH_TRI_CAP)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.triangles, dtype=np.int32)
    _MESH = {
        "x": np.round(vertices[:, 0], 4).tolist(),
        "y": np.round(vertices[:, 1], 4).tolist(),
        "z": np.round(vertices[:, 2], 4).tolist(),
        "i": faces[:, 0].astype(int).tolist(),
        "j": faces[:, 1].astype(int).tolist(),
        "k": faces[:, 2].astype(int).tolist(),
    }
    print(
        f"[view_rollout] mold mesh {len(vertices)} verts {len(faces)} tris",
        flush=True,
    )
    return _MESH


def _downsample(frame: np.ndarray, width: int = 360) -> np.ndarray:
    import cv2

    height, got = frame.shape[:2]
    if got <= width:
        return frame
    return cv2.resize(
        frame,
        (width, max(1, int(round(height * width / got)))),
        interpolation=cv2.INTER_AREA,
    )


def _encode_bgr(frame: np.ndarray) -> bytes:
    import cv2

    ok, encoded = cv2.imencode(
        ".jpg", _downsample(frame), [int(cv2.IMWRITE_JPEG_QUALITY), 60]
    )
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return encoded.tobytes()


def _jpeg_rgb(image: np.ndarray) -> bytes:
    import cv2

    if image.ndim == 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    return _encode_bgr(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _video_jpeg(path: Path, time_s: float) -> bytes:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 15.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        index = int(np.clip(round(max(time_s, 0.0) * fps), 0, max(frame_count - 1, 0)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"no frame at {time_s:.2f}s in {path.name}")
        return _encode_bgr(frame)
    finally:
        capture.release()


def _image(name: str, camera: str, horizon: int) -> bytes:
    key = (name, camera, int(horizon))
    with _JPEG_LOCK:
        cached = _JPEG.get(key)
    if cached is not None:
        return cached
    with _LOCK:
        run = _RUNS.get(name)
    if run is None:
        run = _get_run(name)
    cameras = _log_root(run["path"]) / "cameras"
    import zarr

    group = zarr.open_group(str(_log_root(run["path"])), mode="r")
    absolute = float(np.asarray(group[f"horizon_{horizon}"]["action"]["timestamps_s"]).reshape(-1)[0])
    video_s = absolute - float(run["video_t0"])
    if camera == "center" and run["center"]:
        blob = _video_jpeg(cameras / "orbbec_center.mp4", video_s)
    elif camera == "wrist" and run["wrist"]:
        blob = _video_jpeg(cameras / "wrist_d405.mp4", video_s)
    elif camera == "wrist" and run["logged_wrist"]:
        frame = np.asarray(group[f"horizon_{horizon}"]["obs_policy"]["rgb_0"][-1])
        blob = _jpeg_rgb(frame)
    else:
        raise FileNotFoundError(camera)
    with _JPEG_LOCK:
        _JPEG[key] = blob
    return blob


def _prefetch_images(name: str) -> None:
    with _LOCK:
        run = _RUNS[name]
    for horizon in run["h"]:
        for camera in ("wrist", "center"):
            try:
                _image(name, camera, int(horizon))
            except Exception:
                continue


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Closed-loop replay</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { margin: 0; font: 14px/1.4 sans-serif; background: #16181d; color: #e8eaed; }
  header { display: flex; gap: 12px; align-items: center; padding: 10px 14px; background: #22262e; }
  select, input { background: #111; color: #eee; border: 1px solid #444; padding: 4px 6px; }
  main { display: grid; grid-template-columns: 1fr; gap: 8px; padding: 8px; }
  .full { grid-column: 1 / -1; }
  .plots { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
  figure { margin: 0; background: #1c2028; min-height: 260px; }
  #view { height: 520px; min-height: 520px; }
  .cams { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .cams img { width: 100%; background: #000; min-height: 180px; object-fit: contain; }
  .note { color: #9aa0a6; padding: 0 14px 10px; }
</style>
<header>
  <label>Rollout <select id="run"></select></label>
  <label>Horizon <input id="h" type="range" min="0" max="0" value="0"></label>
  <span id="readout">—</span>
</header>
<main>
  <figure id="view"></figure>
  <figure id="foot"></figure>
  <div class="plots full">
    <figure id="twist"></figure>
    <figure id="dn"></figure>
    <figure id="force"></figure>
  </div>
  <div class="cams full">
    <img id="wrist" alt="wrist">
    <img id="center" alt="center">
  </div>
</main>
<p class="note" id="note"></p>
<script>
const axis = {color: "#9aa0a6", gridcolor: "#2c313a", zerolinecolor: "#444"};
const layout3 = {paper_bgcolor: "#1c2028", plot_bgcolor: "#1c2028", font: {color: "#ddd"},
  scene: {xaxis: {title: "x"}, yaxis: {title: "y"}, zaxis: {title: "z"}, aspectmode: "data"},
  margin: {l: 0, r: 0, t: 30, b: 0}, legend: {orientation: "h"}};
const layout2 = {paper_bgcolor: "#1c2028", plot_bgcolor: "#1c2028", font: {color: "#ddd"},
  margin: {l: 48, r: 12, t: 30, b: 36}, xaxis: axis, yaxis: axis, legend: {orientation: "h"}};
let series = null;
let mesh = null;
let preds = {};
let demoTimer = null;
let scrubTimer = null;
let drawn = false;
const VIEW = {mesh: 0, demo: 1, roll: 2, foot: 3, later: 4, exec: 5, cmd: 6, meas: 7, here: 8,
  predDn: 9, predVt: 10, logDn: 11, logVt: 12};

function line(name, x, y, color) {
  return {type: "scatter", mode: "lines", name, x, y, line: {color, width: 2}};
}
function xyzOf(rows) {
  const pts = rows || [];
  if (!pts.length) return {x: [null], y: [null], z: [null]};
  return {x: pts.map(p => p[0]), y: pts.map(p => p[1]), z: pts.map(p => p[2])};
}
function scatter3(name, xyz, color, width) {
  const cols = xyzOf(xyz);
  return {type: "scatter3d", mode: "lines", name,
    x: cols.x, y: cols.y, z: cols.z,
    line: {color, width: width || 4}};
}
function marker3(name, color, size) {
  return {type: "scatter3d", mode: "markers", name,
    x: [null], y: [null], z: [null], marker: {size: size || 6, color}};
}
function arrow3(name, color) {
  return {type: "scatter3d", mode: "lines+markers", name,
    x: [null, null], y: [null, null], z: [null, null],
    line: {color, width: 6}, marker: {size: 4, color}};
}
function arrow2(name, color) {
  return {type: "scatter", mode: "lines+markers", name,
    x: [null, null], y: [null, null],
    line: {color, width: 2}, marker: {size: 7, color}};
}
function seg3(a, b) {
  if (!a || !b) return {x: [null, null], y: [null, null], z: [null, null]};
  return {x: [a[0], b[0]], y: [a[1], b[1]], z: [a[2], b[2]]};
}
function seg2(a, b) {
  if (!a || !b) return {x: [null, null], y: [null, null]};
  return {x: [a[0], b[0]], y: [a[2], b[2]]};
}

async function loadRuns() {
  const runs = await (await fetch("/api/rollouts")).json();
  const select = document.getElementById("run");
  select.innerHTML = runs.map(name => `<option>${name}</option>`).join("");
  select.onchange = () => loadSeries(select.value);
  if (runs.length) loadSeries(runs[0]);
}

async function loadSeries(name) {
  series = await (await fetch("/api/series?run=" + encodeURIComponent(name))).json();
  const slider = document.getElementById("h");
  slider.max = String(Math.max(0, series.h.length - 1));
  slider.value = "0";
  slider.oninput = () => {
    clearTimeout(scrubTimer);
    scrubTimer = setTimeout(() => scrub(+slider.value), 100);
  };
  preds = {};
  drawn = false;
  if (!mesh) mesh = await (await fetch("/api/mesh")).json();
  drawSeries();
  scrub(0);
  if (demoTimer) clearInterval(demoTimer);
  demoTimer = setInterval(refreshDemo, 2000);
  refreshDemo();
}

function horizonMarker(color) {
  return {type: "scatter", mode: "markers", name: "now", x: [null], y: [null],
    marker: {size: 11, color, symbol: "diamond", line: {width: 1, color: "#fff"}}};
}
function drawSeries() {
  const h = series.h;
  const foot = series.foot;
  const meshTrace = {type: "mesh3d", name: "mold", x: mesh.x, y: mesh.y, z: mesh.z,
    i: mesh.i, j: mesh.j, k: mesh.k, opacity: 0.35, color: "#8aa4b8",
    flatshading: true, hoverinfo: "skip", lighting: {ambient: 0.75, diffuse: 0.5}};
  Plotly.newPlot("view", [
    meshTrace,
    scatter3("demo ep9", series.demo, "#5dade2", 2),
    marker3("demo roll", "#e74c3c", 8),
    scatter3("foot path", foot, "#ff9f1a", 4),
    scatter3("predicted, later steps", [], "#f5cba7", 3),
    scatter3("predicted, executed", [], "#ff9f1a", 8),
    scatter3("commanded", [], "#2ecc71", 6),
    scatter3("measured", [], "#e8eaed", 4),
    marker3("this horizon", "#ffffff", 7),
    arrow3("predicted δn target", "#f5b041"),
    arrow3("predicted virtual target", "#af7ac5"),
    arrow3("logged δn target", "#f4d03f"),
    arrow3("logged virtual target", "#58d68d"),
  ], {...layout3, title: "fixture frame"});
  const roll = xyzOf([series.roll]);
  Plotly.restyle("view", {x: [roll.x], y: [roll.y], z: [roll.z]}, [VIEW.roll]);
  Plotly.newPlot("twist", [
    line("commanded", h, series.cmd_twist, "#2ecc71"),
    line("measured", h, series.meas_twist, "#e8eaed"),
    horizonMarker("#ffffff"),
  ], {...layout2, title: "Twist vs field (deg)"});
  Plotly.newPlot("dn", [
    line("scheduled δn", h, series.dn_sched, "#f4d03f"),
    horizonMarker("#ffffff"),
  ], {...layout2, title: "δn (mm)"});
  Plotly.newPlot("force", [
    line("signed distance", h, series.sd, "#3498db"),
    line("|F·n|", h, series.fn, "#e74c3c"),
    horizonMarker("#ffffff"),
  ], {...layout2, title: "sd (mm) and |F·n| (N)"});
  Plotly.newPlot("foot", [
    {type: "scatter", mode: "lines", name: "demo ep9", x: series.demo.map(p => p[0]), y: series.demo.map(p => p[2]), line: {color: "#5dade2", width: 1}},
    {type: "scatter", mode: "lines", name: "this foot", x: foot.map(p => p[0]), y: foot.map(p => p[2]), line: {color: "#ff9f1a", width: 2}},
    {type: "scatter", mode: "markers", name: "demo roll", x: [series.roll[0]], y: [series.roll[2]], marker: {size: 12, color: "#e74c3c", symbol: "x"}},
    {type: "scatter", mode: "markers", name: "now", x: [foot[0][0]], y: [foot[0][2]], marker: {size: 10, color: "#ffffff"}},
    arrow2("predicted δn target", "#f5b041"),
    arrow2("predicted virtual target", "#af7ac5"),
    arrow2("logged δn target", "#f4d03f"),
    arrow2("logged virtual target", "#58d68d"),
  ], {...layout2, title: "Foot path, x vs z", xaxis: {...axis, title: "x"}, yaxis: {...axis, title: "z", scaleanchor: "x"}});
  drawn = true;
  document.getElementById("note").textContent =
    "Scrub moves markers on traces built once. Predicted chunk is a seed-0 resample, filled in the background. " +
    "The mesh is the fixture-frame mold fixb_labels queries. " +
    (series.overlay ? "Demo-obs overlay is aligned from ep9 index 1220." : "No demo-obs overlay for this run.");
}

async function refreshDemo() {
  const extra = await (await fetch("/api/demo?run=" + encodeURIComponent(series.name))).json();
  if (!(extra.live_h && extra.live_h.length) && !(extra.h && extra.h.length)) return;
  const twist = [
    line("commanded", series.h, series.cmd_twist, "#2ecc71"),
    line("measured", series.h, series.meas_twist, "#e8eaed"),
  ];
  const dn = [line("scheduled δn", series.h, series.dn_sched, "#f4d03f")];
  if (extra.live_h && extra.live_h.length) {
    extra.live_h.forEach((horizon, i) => {
      if (extra.pred && extra.pred[i]) {
        preds[horizon] = {
          xyz: extra.pred[i],
          foot: extra.live_foot ? extra.live_foot[i] : null,
          dn_target: extra.live_dn_target ? extra.live_dn_target[i] : null,
          vt: extra.live_vt ? extra.live_vt[i] : null,
        };
      }
    });
    twist.push(line("predicted", extra.live_h, extra.live_twist, "#ff9f1a"));
    dn.push(line("predicted δn", extra.live_h, extra.live_dn, "#ff9f1a"));
  }
  if (extra.h && extra.h.length) {
    twist.push(line("demo-obs open loop", extra.h, extra.twist, "#bb8fce"));
    dn.push(line("demo-obs δn", extra.h, extra.dn, "#bb8fce"));
  }
  const marker = horizonMarker("#ffffff");
  twist.push(marker);
  dn.push(horizonMarker("#ffffff"));
  Plotly.react("twist", twist, {...layout2, title: "Twist vs field (deg)"});
  Plotly.react("dn", dn, {...layout2, title: "δn (mm)"});
  if (drawn) scrub(+document.getElementById("h").value);
  if (extra.ready && Object.keys(preds).length >= series.h.length && demoTimer) clearInterval(demoTimer);
}

function scrub(index) {
  if (!series || !drawn) return;
  const horizon = series.h[index];
  const pred = preds[horizon] || {};
  const path = pred.xyz || [];
  const execN = series.exec_h || 6;
  const foot = series.foot[index];
  const arrows = [
    seg3(pred.foot, pred.dn_target),
    seg3(pred.foot, pred.vt),
    seg3(foot, series.dn_target[index]),
    seg3(foot, series.vt ? series.vt[index] : series.cmd[index]),
  ];
  const sets = [xyzOf(path.slice(execN)), xyzOf(path.slice(0, execN)),
    xyzOf(series.cmd_path[index]), xyzOf(series.meas_path[index]),
    xyzOf(foot ? [foot] : [])].concat(arrows);
  Plotly.restyle("view", {
    x: sets.map(s => s.x), y: sets.map(s => s.y), z: sets.map(s => s.z),
  }, [VIEW.later, VIEW.exec, VIEW.cmd, VIEW.meas, VIEW.here, VIEW.predDn, VIEW.predVt, VIEW.logDn, VIEW.logVt]);
  const flat = [
    seg2(pred.foot, pred.dn_target),
    seg2(pred.foot, pred.vt),
    seg2(foot, series.dn_target[index]),
    seg2(foot, series.vt ? series.vt[index] : series.cmd[index]),
  ];
  Plotly.restyle("foot", {
    x: [[foot[0]]].concat(flat.map(s => s.x)),
    y: [[foot[2]]].concat(flat.map(s => s.y)),
  }, [3, 4, 5, 6, 7]);
  const mark = (div, y) => Plotly.restyle(div, {x: [[horizon]], y: [[y]]}, [-1]);
  mark("twist", series.cmd_twist[index]);
  mark("dn", series.dn_sched[index]);
  mark("force", series.sd[index]);
  document.getElementById("readout").textContent =
    "h" + horizon + "  t " + series.t[index].toFixed(1) + " s";
  const name = encodeURIComponent(series.name);
  document.getElementById("wrist").src = `/img?run=${name}&cam=wrist&h=${horizon}`;
  document.getElementById("center").src = `/img?run=${name}&cam=center&h=${horizon}`;
}
loadRuns();
document.addEventListener("keydown", (ev) => {
  if (ev.key !== "ArrowRight" && ev.key !== "ArrowLeft") return;
  const slider = document.getElementById("h");
  slider.value = String(Math.max(0, Math.min(+slider.max, +slider.value + (ev.key === "ArrowRight" ? 1 : -1))));
  scrub(+slider.value);
});
</script>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[view_rollout]", fmt % args, flush=True)

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, code: int = 200) -> None:
        body = json.dumps(_round(payload)).encode()
        self._send(code, body, "application/json")

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/rollouts":
                self._json([path.name for path in _rollout_dirs()])
                return
            if parsed.path == "/api/mesh":
                self._json(mold_mesh())
                return
            name = query.get("run", [""])[0]
            if parsed.path == "/api/series":
                run = _get_run(name)
                payload = {
                    key: value
                    for key, value in run.items()
                    if key != "path" and not str(key).startswith("_")
                }
                self._json(payload)
                return
            if parsed.path == "/api/chunk":
                self._json(_chunk(name, int(query.get("h", ["0"])[0])))
                return
            if parsed.path == "/api/demo":
                run = _get_run(name)
                with _LOCK:
                    rows = dict(_DEMO["by_h"])
                    ready = bool(_DEMO["ready"])
                    error = _DEMO["error"]
                ordered = sorted(rows)
                live = _LIVE.get(name, {})
                live_h = sorted(key for key in live if isinstance(key, int))
                self._json(
                    {
                        "ready": ready,
                        "error": error,
                        "h": ordered,
                        "twist": [rows[key]["twist0"] for key in ordered],
                        "dn": [rows[key]["dn0"] for key in ordered],
                        "overlay": bool(run["overlay"]),
                        "live_h": live_h,
                        "live_twist": [live[key]["twist0"] for key in live_h],
                        "live_dn": [live[key]["dn0"] for key in live_h],
                        "pred": [live[key].get("xyz") for key in live_h],
                        "live_foot": [live[key].get("foot") for key in live_h],
                        "live_dn_target": [live[key].get("dn_target") for key in live_h],
                        "live_vt": [live[key].get("vt") for key in live_h],
                    }
                )
                return
            if parsed.path == "/img":
                blob = _image(name, query.get("cam", ["wrist"])[0], int(query.get("h", ["0"])[0]))
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(blob)))
                self.send_header("Cache-Control", "max-age=86400")
                self.end_headers()
                self.wfile.write(blob)
                return
            self._send(404, b"not found", "text/plain")
        except Exception as exc:
            self._json({"error": str(exc)}, 500)


def main() -> None:
    server = None
    port = None
    for candidate in PORTS:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), Handler)
            port = candidate
            break
        except OSError as exc:
            print(f"[view_rollout] {candidate} unavailable ({exc})", flush=True)
    if server is None or port is None:
        raise SystemExit("8810, 8811, and 8812 are all in use")
    print(f"[view_rollout] http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
