#!/usr/bin/env python3
"""Kinematic sim for the smooth_frame runner. Does not copy that code.

The decode, foot walk, jump guard, D405 corners, and flange plane are the
functions in the smooth_frame worktree (UMIFT_SMOOTH_FRAME, default
~/umi_ft_ws/worktrees/UMI-FT-smooth_frame). This file only calls them.

  sim_harness.py label_check --episode 40 --bench
  sim_harness.py replay --episode 40 --bench --lag 0
  sim_harness.py rollout --episode 40 --seconds 60 --ckpt <path>

Same env vars as scripts/run_rollout.sh. --bench sets UMIFT_ENABLE_DELTA_N_FLOOR.
Measured TCP is the last commanded pose. --lag 1 reports the command before that.
RGB and wrench on a rollout are the episode tape; the pose is the fake arm.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SMOOTH = Path(
    os.environ.get(
        "UMIFT_SMOOTH_FRAME",
        "/home/boeing/umi_ft_ws/worktrees/UMI-FT-smooth_frame",
    )
)
WS = Path("/home/boeing/umi_ft_ws")
DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
FIELD = DATA / "shared" / "fixb_field.npz"
TRACK_ZARR = DATA / "r4vt_smooth_tcp_track_clean" / "acp_replay_buffer.zarr"
NORM = (
    SMOOTH
    / "UMIFT_Data/surface_frames_scripts/assets"
    / "oct_v35_h4s_r4vt_action_normalizer_minmax_track_clean.pkl"
)
NORM_SHA = "29a48f69ff51807df631b1e350087d6991091cba9739f0be88b30e2c85cf3bd4"
DT = 1.0 / 15.0
EXEC_H = 6
DECAY = 0.01

_BOOTED = False
runner_decode = None


def boot() -> None:
    """Import the live smooth_frame tree, not this worktree's old copy."""
    global _BOOTED, runner_decode
    if _BOOTED:
        return
    os.environ.setdefault("UMIFT_R4VT_ZARR", str(TRACK_ZARR))
    os.environ.setdefault("UMIFT_R4VT_NORMALIZER", str(NORM))
    os.environ.setdefault("UMIFT_R4VT_NORMALIZER_SHA", NORM_SHA)
    os.environ.setdefault("UMIFT_ACTION_NORMALIZER", str(NORM))
    os.environ.setdefault("UMIFT_ACTION_NORMALIZER_SHA", NORM_SHA)
    scripts = SMOOTH / "UMIFT_Data" / "surface_frames_scripts"
    ordered = [
        scripts,
        SMOOTH / "PyriteML",
        SMOOTH / "PyriteUtility",
        SMOOTH,
        WS,
    ]
    for path in reversed(ordered):
        text = str(path)
        while text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    # Tests may have imported this worktree's copy before boot.
    own = Path(__file__).resolve().parents[2]
    for name, mod in list(sys.modules.items()):
        if name in ("sim_harness", "__main__") or name.startswith("sim_harness."):
            continue
        file = getattr(mod, "__file__", None)
        if file and own in Path(file).resolve().parents:
            del sys.modules[name]
    import mold_aware_runtime as runtime

    if "UMI-FT-smooth_frame" not in Path(runtime.__file__).resolve().parts:
        raise RuntimeError(f"mold runtime is not smooth_frame: {runtime.__file__}")
    runner_decode = runtime.MoldAwareRuntime.decode_chunk
    _BOOTED = True


class FakeArm:
    """Measured TCP is the last command. lag=1 is the command before that.

    contact=False reports the command itself.
    contact=True reports the closest surface point to the commanded position
    and keeps the commanded rotation. ``project`` maps that position to the surface.
    """

    def __init__(self, pose7: np.ndarray, lag: int = 0, contact: bool = False, project=None):
        self.lag = int(lag)
        self.contact = bool(contact)
        self.project = project
        self._sent = [np.asarray(pose7, dtype=np.float64).reshape(7).copy()]

    def measure(self) -> np.ndarray:
        index = max(0, len(self._sent) - 1 - self.lag)
        return self._sent[index].copy()

    def command(self, pose7: np.ndarray) -> None:
        pose = np.asarray(pose7, dtype=np.float64).reshape(7).copy()
        if self.contact:
            pose[:3] = np.asarray(self.project(pose[:3]), dtype=np.float64).reshape(3)
        self._sent.append(pose)


class PoseLog:
    """Piecewise-constant commanded poses, for the policy's pose history."""

    def __init__(self, t0: float, pose7: np.ndarray):
        self.t = [float(t0)]
        self.p = [np.asarray(pose7, dtype=np.float64).reshape(7).copy()]

    def add(self, t: float, pose7: np.ndarray) -> None:
        self.t.append(float(t))
        self.p.append(np.asarray(pose7, dtype=np.float64).reshape(7).copy())

    def at(self, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        idx = np.searchsorted(np.asarray(self.t), times, side="right") - 1
        idx = np.clip(idx, 0, len(self.p) - 1)
        return np.stack([self.p[i] for i in idx])


def _stats(values) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "median": None, "max": None}
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def _geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    rel = np.einsum("nji,njk->nik", a, b)
    cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def write_horizon(log, index: int, t0: float, dt: float, decoded: dict, wrench) -> None:
    """One executed slice, the fields plot_rollout_quick.series reads."""
    from mold_aware_runtime import Rp_to_SE3

    n = int(np.asarray(decoded["x_d"]).shape[0])
    group = log.create_group(f"horizon_{int(index)}")
    action = group.create_group("action")
    action.create_dataset("timestamps_s", data=float(t0) + np.arange(n) * float(dt))
    action.create_dataset("delta_n", data=np.asarray(decoded["delta_n"], dtype=np.float64)[:n])
    action.create_dataset("v_par", data=np.asarray(decoded["v_par"], dtype=np.float64)[:n])
    action.create_dataset("foot", data=np.asarray(decoded["x_s"], dtype=np.float64)[:n])
    action.create_dataset("wrench", data=np.asarray(wrench, dtype=np.float64).reshape(-1)[:6])
    r_d = np.asarray(decoded["R_d"], dtype=np.float64)[:n]
    x_s = np.asarray(decoded["x_s"], dtype=np.float64)[:n]
    n_out = np.asarray(decoded["n_out"], dtype=np.float64)[:n]
    delta_n = np.asarray(decoded["delta_n"], dtype=np.float64).reshape(-1)[:n]
    action.create_dataset("ts_nominal_targets_0", data=Rp_to_SE3(r_d, x_s))
    action.create_dataset(
        "ts_virtual_targets_0",
        data=Rp_to_SE3(r_d, x_s + delta_n[:, None] * n_out),
    )


def _runner_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(SMOOTH), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _apply_bench(bench: bool) -> float:
    if bench:
        os.environ["UMIFT_ENABLE_DELTA_N_FLOOR"] = "1"
    lift = os.environ.get("UMIFT_BENCH_LIFT_M", "").strip()
    return float(lift) if lift else 0.0


def make_runtime(bench: bool):
    """Same construction as umift_env_runner.main for an r4vt checkpoint."""
    boot()
    from mold_aware_runtime import (
        MoldAwareRuntime,
        camera_table_from_flange,
        r4_safety_limits,
    )
    from PyriteConfig.tasks.umift.workcell_frames import load_workcell_extrinsics

    abort = os.environ.get("UMIFT_ABORT_ON_MEASURED_MESH_DISTANCE", "").strip().lower()
    floor = os.environ.get("UMIFT_ENABLE_DELTA_N_FLOOR", "").strip().lower()
    limits = r4_safety_limits(
        abort_on_measured_mesh_distance=abort in ("1", "true", "yes"),
        enable_delta_n_floor=floor in ("1", "true", "yes"),
    )
    runtime = MoldAwareRuntime.from_assets(
        run_type="r4_world_fixb_tilt",
        field_path=FIELD,
        dry_run=False,
        limits=limits,
    )
    extrinsics = load_workcell_extrinsics()
    normal, offset, _base_y = camera_table_from_flange(
        FIELD, extrinsics.T_fixture_from_right_base
    )
    runtime.set_table_plane(normal, offset)
    runtime._plane = (np.asarray(normal, dtype=np.float64), float(offset))
    return runtime, extrinsics.T_fixture_from_right_base


def _episode(episode: int) -> dict:
    boot()
    import zarr
    from fixb_labels import pose7_to_Rp

    group = zarr.open(os.environ["UMIFT_R4VT_ZARR"], mode="r")["data"][f"episode_{episode}"]
    pose = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    rotation, position = pose7_to_Rp(pose)
    return {
        "pose": pose,
        "R": rotation,
        "p": position,
        "robot_t": np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1),
        "rgb": group["rgb_0"],
        "rgb_t": np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1),
        "wrench": np.asarray(group["wrench_left_0"], dtype=np.float64),
        "wrench_t": np.asarray(group["wrench_time_stamps_0"], dtype=np.float64).reshape(-1),
        "foot": np.asarray(group["mold_x_s_0"], dtype=np.float64),
        "frame": np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3),
        "normal": np.asarray(group["mold_normal_0"], dtype=np.float64),
        "v": np.asarray(group["fixb_v_par_0"], dtype=np.float64),
        "dn": np.asarray(group["fixb_delta_n_0"], dtype=np.float64).reshape(-1),
        "dR": np.asarray(group["fixb_delta_R_rot6d_0"], dtype=np.float64),
    }


def _nearest_wrench(ep: dict, t_abs: float) -> np.ndarray:
    k = int(np.argmin(np.abs(ep["wrench_t"] - t_abs)))
    return np.asarray(ep["wrench"][k], dtype=np.float64).reshape(-1)[:6]


def _clearance_m(runtime, rotation: np.ndarray, position: np.ndarray) -> float | None:
    from mold_aware_runtime import d405_housing_corners_in_tcp

    rotation = np.asarray(rotation, dtype=np.float64).reshape(-1, 3, 3)
    position = np.asarray(position, dtype=np.float64).reshape(-1, 3)
    corners = d405_housing_corners_in_tcp()
    pts = np.einsum("nij,kj->nki", rotation, corners) + position[:, None, :]
    clearance = runtime._clearance(pts.reshape(-1, 3))
    if clearance is None:
        return None
    return float(np.min(clearance))


def _poses_clear(runtime, pose7: np.ndarray) -> bool:
    """Runner approach check. Poses are already in the fixture frame."""
    from PyriteEnvSuites.env_runners.umift_env_runner import _camera_poses_clear

    normal, offset = runtime._plane
    return bool(_camera_poses_clear(pose7, np.eye(4), normal, offset))


def _tilt_from_normal(rotation, normal) -> np.ndarray:
    """Degrees between tool +z and −n. Same angle as _clamp_tilt_to_normal."""
    z = np.asarray(rotation, dtype=np.float64).reshape(-1, 3, 3)[:, :, 2]
    n = np.asarray(normal, dtype=np.float64).reshape(-1, 3)
    n = n / np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9, None)
    cos = np.clip(np.sum(z * (-n), axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _tilt_regions(tilt_deg, normal) -> dict:
    from openloop_lib import REGION_NAMES, region_code

    code = region_code(normal)
    out = {}
    tilt_deg = np.asarray(tilt_deg, dtype=np.float64).reshape(-1)
    for index, name in enumerate(REGION_NAMES):
        picked = tilt_deg[code.reshape(-1) == index]
        out[name] = {
            "n": int(picked.size),
            "median_deg": None if picked.size == 0 else float(np.median(picked)),
        }
    return out


def _trim(decoded: dict, n: int, full: int) -> dict:
    out = dict(decoded)
    for key, value in decoded.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (full,):
            out[key] = value[:n]
    return out


def runner_step(runtime, ensembler, guard, step_index, raw, pose7, label_R, lift_m):
    """One r4vt replan: ensemble, decode_chunk, flange hold, jump guard.

    ponytail: the jump guard runs in the fixture frame. Step size is unchanged
    by the rigid base transform the runner applies first.
    """
    from fixb_labels import Rp_to_pose7, pose7_to_Rp
    from mold_aware_runtime import fixb_decode_dt, shift_fixture_up

    rotation, position = pose7_to_Rp(np.asarray(pose7, dtype=np.float64).reshape(1, 7))
    ensembler.push(int(step_index) * EXEC_H, np.asarray(raw, dtype=np.float64))
    blended = ensembler.pop(EXEC_H)
    if label_R is not None:
        label_R = np.asarray(label_R, dtype=np.float64)[: blended.shape[0]]
    decoded = runtime.decode_chunk(
        blended,
        position[0],
        rotation[0],
        fixb_decode_dt(blended.shape[0], 1, DT),
        label_R=label_R,
    )
    if lift_m:
        decoded["x_s"] = shift_fixture_up(decoded["x_s"], lift_m)
        decoded["x_d"] = shift_fixture_up(decoded["x_d"], lift_m)
    hard = list(runtime.last_hard_triggers)
    full = int(decoded["x_d"].shape[0])
    r_before = np.asarray(decoded["R_d"], dtype=np.float64).copy()
    x_g, r_g, field_g, aborted = guard.apply(
        decoded["x_d"], decoded["R_d"], R_field=decoded["R_field"]
    )
    n_keep = int(x_g.shape[0])
    guards = []
    for row in guard.holds:
        if row.get("_seen"):
            continue
        row["_seen"] = True
        guards.append(
            {
                "name": "jump_guard_abort" if row["abort"] else "jump_guard_hold",
                "step": int(row["step"]),
                "pos_mm": float(row["pos_mm"]),
                "rot_deg": float(row["rot_deg"]),
            }
        )
    if n_keep == 0:
        return {
            "decoded": None,
            "hard": hard + ["jump_guard_abort"],
            "decode_hard": list(hard),
            "jump_abort": True,
            "guards": guards,
            "clearance_m": None,
            "pose7": None,
            "waypoints": None,
        }
    decoded = _trim(decoded, n_keep, full)
    decoded["x_d"] = x_g
    decoded["R_d"] = r_g
    decoded["R_field"] = field_g
    # The runner stores the pre-guard rotation of the last kept step.
    runtime.note_commanded_rd(r_before[n_keep - 1])
    pose = Rp_to_pose7(decoded["R_d"], decoded["x_d"])
    clearance = _clearance_m(runtime, decoded["R_d"], decoded["x_d"])
    meas_c = _clearance_m(runtime, rotation, position)
    if meas_c is not None:
        clearance = meas_c if clearance is None else min(clearance, meas_c)
    clear_ok = _poses_clear(runtime, pose)
    if not clear_ok:
        hard.append("camera_poses_clear")
    if aborted:
        hard.append("jump_guard_abort")
    # Decode hard-stops are not scheduled. A jump abort still sends the kept steps.
    decode_hard = [name for name in hard if name not in ("jump_guard_abort", "camera_poses_clear")]
    return {
        "decoded": decoded,
        "hard": hard,
        "decode_hard": decode_hard,
        "jump_abort": bool(aborted) or n_keep == 0,
        "guards": guards,
        "clearance_m": clearance,
        "pose7": pose,
        "waypoints": pose,
    }


def _open_log(mode: str, tag: str):
    import zarr

    root = Path(os.environ.get("PYRITE_CONTROL_LOG_FOLDERS", str(Path.home() / "data" / "control_log")))
    root.mkdir(parents=True, exist_ok=True)
    folder = root / f"sim_{mode}_{tag}_{int(time.time())}"
    log = zarr.open(zarr.DirectoryStore(str(folder)), mode="w")
    log.attrs["sim"] = mode
    log.attrs["runner_sha"] = _runner_sha()
    log.attrs["decode_dt_s"] = DT
    return folder, log


def _finish(folder: Path, summary: dict) -> None:
    path = folder.parent / f"{folder.name}.json"
    path.write_text(json.dumps(summary, indent=2))
    summary["summary_json"] = str(path)
    summary["control_log"] = str(folder)
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "steps"}, indent=2))


def _note(guards: list, name: str, t: float) -> None:
    if name and not any(row["name"] == name for row in guards):
        guards.append({"name": name, "t_s": float(t)})


def decode_vpar_enabled() -> bool:
    """UMIFT_DECODE_VPAR=1 adds tangential velocity to the commanded TCP."""
    return os.environ.get("UMIFT_DECODE_VPAR", "").strip().lower() in ("1", "true", "yes", "on")


def command_tangential(foot, frame, v_par, delta_n, dt) -> tuple[np.ndarray, np.ndarray]:
    """foot + (v1·t1 + v2·t2)·dt + δn·n_out. Frame z is the inward normal."""
    frame = np.asarray(frame, dtype=np.float64).reshape(3, 3)
    tang = frame[:, :2] @ np.asarray(v_par, dtype=np.float64).reshape(2) * float(dt)
    outward = -frame[:, 2]
    commanded = np.asarray(foot, dtype=np.float64).reshape(3) + tang + float(delta_n) * outward
    return commanded, tang


def install_causal_foot():
    """Point the runner's foot walk at causal_foot. The smooth_frame file is not edited."""
    import importlib.util

    import mold_aware_runtime as runtime_mod

    path = Path(__file__).resolve().with_name("causal_foot.py")
    spec = importlib.util.spec_from_file_location("causal_foot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    foot = module.CausalFoot(FIELD, FIELD.with_name("fixb_smooth_frames.npz"))
    runtime_mod._harness_tcp = None
    runtime_mod._harness_tcp_disp = None

    def integrate(field, p0, v_par, delta_n, dt, prev_x=None, prev_R=None, **_kwargs):
        del field
        count = int(np.asarray(delta_n).reshape(-1).shape[0])
        tcp = np.asarray(p0, dtype=np.float64).reshape(3)
        prev = None
        if prev_x is not None and prev_R is not None:
            prev = (
                np.asarray(prev_x, dtype=np.float64).reshape(3),
                np.asarray(prev_R, dtype=np.float64).reshape(3, 3),
            )
        xs = np.zeros((count, 3), dtype=np.float64)
        frames = np.zeros((count, 3, 3), dtype=np.float64)
        depth = np.asarray(delta_n, dtype=np.float64).reshape(-1)
        runtime_mod._harness_formula_xd = None
        if decode_vpar_enabled():
            # v·dt is the TCP's tangent, not the foot step. On the crown the
            # closest-point foot hops when the TCP moves along the normal, which
            # v does not contain. Carry the TCP by its full step and let the
            # tracker read that TCP. x_d still adds the tangent once.
            velocity = np.asarray(v_par, dtype=np.float64).reshape(count, 2)
            dts = np.asarray(dt, dtype=np.float64).reshape(count)
            disp = getattr(runtime_mod, "_harness_tcp_disp", None)
            if disp is None or len(np.asarray(disp)) < count:
                disp = None
            else:
                disp = np.asarray(disp, dtype=np.float64).reshape(-1, 3)[:count]
            carried = getattr(runtime_mod, "_harness_tcp", None)
            if prev is None or carried is None:
                tcp_state = tcp.copy()
            else:
                tcp_state = np.asarray(carried, dtype=np.float64).reshape(3).copy()
            commanded = np.zeros((count, 3), dtype=np.float64)
            for k in range(count):
                x, rotation = foot.step(tcp_state, prev)
                xs[k] = x
                frames[k] = rotation
                commanded[k], _tang = command_tangential(
                    x, rotation, velocity[k], depth[k], dts[k]
                )
                if disp is not None:
                    tcp_state = tcp_state + disp[k]
                prev = (x, rotation)
            runtime_mod._harness_tcp = tcp_state
            runtime_mod._harness_formula_xd = commanded
            outward = -frames[:, :, 2]
            x_d = commanded
        else:
            for k in range(count):
                xs[k], frames[k] = foot.step(tcp, prev)
                prev = (xs[k], frames[k])
            outward = -frames[:, :, 2]
            x_d = xs + depth[:, None] * outward
        quiet = np.zeros(count, dtype=bool)
        return {
            "x_s": xs,
            "x_d": x_d,
            "R": frames,
            "n_out": outward,
            "face_held": quiet,
            "hop_hold": quiet,
            "foot_reanchor": quiet,
            "foot_cap": quiet,
            "hop_streak": np.array(0),
            "reject_streak": np.array(0),
            "step_reject": quiet,
            "ray_recovery": quiet,
        }

    runtime_mod.integrate_tcp_foot = integrate
    if not getattr(runtime_mod.MoldAwareRuntime.decode_chunk, "_vpar_wrap", False):
        original = runtime_mod.MoldAwareRuntime.decode_chunk

        def decode_chunk(self, raw_action, p0, R0, dt, apply_safety=True, frame_seed=None, label_R=None):
            runtime_mod._harness_formula_xd = None
            out = original(
                self, raw_action, p0, R0, dt,
                apply_safety=apply_safety, frame_seed=frame_seed, label_R=label_R,
            )
            commanded = getattr(runtime_mod, "_harness_formula_xd", None)
            if commanded is not None:
                # The VT limiter rebuilds x_d as foot + δn·n and drops the sum.
                n = int(out["x_d"].shape[0])
                out["x_d"] = np.asarray(commanded, dtype=np.float64)[:n]
                out["delta_n"] = np.sum((out["x_d"] - out["x_s"]) * out["n_out"], axis=1)
            return out

        decode_chunk._vpar_wrap = True
        runtime_mod.MoldAwareRuntime.decode_chunk = decode_chunk
    return foot


def label_parity(runtime, ep: dict, label_x: np.ndarray, label_R: np.ndarray) -> dict:
    """Safety off. The runner foot and the label foot see the same demo TCP series."""
    import mold_aware_runtime as runtime_mod

    runtime.release_command_foot()
    prev_x = None
    prev_R = None
    foot_err = []
    frame_err = []
    for i, tcp in enumerate(ep["p"]):
        dec = runtime_mod.integrate_tcp_foot(
            runtime.field,
            tcp,
            ep["v"][i],
            ep["dn"][i : i + 1],
            np.array([DT]),
            prev_x=prev_x,
            prev_R=prev_R,
        )
        prev_x, prev_R = dec["x_s"][0], dec["R"][0]
        foot_err.append(np.linalg.norm(dec["x_s"][0] - label_x[i]))
        frame_err.append(_geodesic_deg(dec["R"][0:1], label_R[i : i + 1])[0])
        if i and i % 2000 == 0:
            print(f"[label_check] parity {i}/{len(ep['p'])}", flush=True)
    probe = runtime.decode_chunk(
        __import__("openloop_lib").r4vt_action_row(ep["v"][0], ep["dn"][0], ep["dR"][0]),
        ep["p"][0],
        ep["R"][0],
        np.array([DT]),
        apply_safety=False,
    )
    return {
        "foot_mm": _stats(np.asarray(foot_err) * 1e3),
        "frame_deg": _stats(frame_err),
        "decode_matches_first_mm": float(np.linalg.norm(probe["x_s"][0] - label_x[0]) * 1e3),
        "n": len(foot_err),
    }


def demo_tilt(ep: dict) -> dict:
    k = np.clip(np.searchsorted(ep["robot_t"], ep["rgb_t"]), 0, len(ep["robot_t"]) - 1)
    return _tilt_regions(_tilt_from_normal(ep["R"][k], ep["normal"][k]), ep["normal"][k])


def causal_source(ep: dict):
    """Label chunks from the shared tracker. Same windows the runner pushes."""
    import zarr
    from fixb_labels import (
        delta_R_from_field_tool,
        interp_rows,
        next_control_times,
        pose7_to_Rp,
        so3_to_rot6d_rows,
    )
    from fixb_tcp_track import project_tcp_velocity
    from fixb_tcp_velocity_labels import _dn
    from openloop_lib import r4vt_action_row

    import replay_gt

    foot = install_causal_foot()
    x, frame = foot.track(ep["p"])
    outward = -frame[:, :, 2]
    velocity, _dt = project_tcp_velocity(frame, ep["p"], ep["robot_t"], ep["rgb_t"])
    group = zarr.open(os.environ["UMIFT_R4VT_ZARR"], mode="r")["data"][f"episode_{ep['_episode']}"]
    virtual = pose7_to_Rp(np.asarray(group["ts_pose_virtual_target_0"], dtype=np.float64))[1]
    stiffness = np.asarray(group["stiffness_0"], dtype=np.float64).reshape(-1)
    depth = _dn(virtual, x, outward)
    delta_r = so3_to_rot6d_rows(delta_R_from_field_tool(frame, ep["R"]))
    windows = replay_gt._windows(ep["robot_t"], ep["rgb_t"], EXEC_H)
    actions = [
        r4vt_action_row(velocity[idx], depth[idx], delta_r[idx], stiffness=stiffness[idx])
        for idx in windows
    ]

    class _Source:
        def __init__(self):
            self.actions = actions
            self.windows = windows
            self.k = 0
            self.demo = ep["pose"]
            self.tcp_disp = None

        def remaining(self):
            return self.k < len(self.actions)

        def current(self):
            return self.actions[self.k], self.windows[self.k]

        def advance(self):
            self.k += 1

    source = _Source()
    arrived = next_control_times(ep["robot_t"], ep["rgb_t"])
    source.tcp_disp = interp_rows(arrived, ep["robot_t"], ep["p"]) - ep["p"]
    source.foot = x
    source.frame = frame
    source.normal = outward
    return source


def run_gt(ep, runtime, pose_fn, on_command, lift_m, log, guards, clearance, foot_err, tilt, normal, stop_on_hard=True, source=None):
    """GT chunks through the real decode. pose_fn(gt) is the measured TCP."""
    from fixb_labels import pose7_to_Rp
    from jump_guard import TargetJumpGuard
    from mold_aware_runtime import FixBActionEnsembler

    if source is None:
        import replay_gt

        replay_gt.R4VT_ZARR = Path(os.environ["UMIFT_R4VT_ZARR"])
        from gt_replay import GtReplay

        gt = GtReplay("r4vt", int(ep["_episode"]), EXEC_H)
    else:
        gt = source
    ens = FixBActionEnsembler(decay_m=DECAY)
    guard = TargetJumpGuard()
    t0 = float(ep["robot_t"][0])
    horizons = 0
    last_t = 0.0
    stop = "episode_end"
    while gt.remaining():
        window = gt.windows[gt.k]
        head = int(window[0])
        t_abs = float(ep["robot_t"][min(head, len(ep["robot_t"]) - 1)])
        last_t = t_abs - t0
        pose7 = np.asarray(pose_fn(gt), dtype=np.float64).reshape(7)
        obs = {"ts_pose_fb_0": pose7.reshape(1, 7), "robot_time_stamps_0": np.array([t_abs])}
        runtime.fill_obs_raw(obs)
        obs_hard = list(runtime.last_hard_triggers)
        for name in obs_hard:
            _note(guards, name, last_t)
        wrench = _nearest_wrench(ep, t_abs)
        decision = runtime.wrench_hold_update(float(np.linalg.norm(wrench[:3])))
        if decision == "hold":
            _note(guards, "wrench_hold", last_t)
            continue
        if decision == "abort":
            _note(guards, "wrench_abort", last_t)
            stop = "wrench_abort"
            break
        if obs_hard and stop_on_hard:
            stop = obs_hard[0]
            break
        raw, _window = gt.current()
        label_R, _p = pose7_to_Rp(gt.demo[np.asarray(window[:EXEC_H], dtype=np.int64)])
        disp = getattr(gt, "tcp_disp", None)
        if disp is not None:
            import mold_aware_runtime as runtime_mod

            taken = np.asarray(window[:EXEC_H], dtype=np.int64)
            runtime_mod._harness_tcp_disp = np.asarray(disp, dtype=np.float64)[taken]
        step = runner_step(runtime, ens, guard, horizons, raw, pose7, label_R, lift_m)
        for row in step["guards"]:
            _note(guards, row["name"], last_t)
        for name in step["hard"]:
            _note(guards, name, last_t)
        if step["clearance_m"] is not None:
            clearance.append(step["clearance_m"])
        decoded = step["decoded"]
        if decoded is not None:
            write_horizon(log, horizons, last_t, DT, decoded, wrench)
            horizons += 1
            used = np.asarray(window[: decoded["x_s"].shape[0]], dtype=np.int64)
            used = used[used < len(ep["foot"])]
            n = len(used)
            if n:
                foot_err.append(np.linalg.norm(decoded["x_s"][:n] - ep["foot"][used], axis=1))
                tilt.append(_tilt_from_normal(decoded["R_d"][:n], decoded["n_out"][:n]))
                normal.append(np.asarray(decoded["n_out"][:n], dtype=np.float64))
            if not step["decode_hard"]:
                on_command(step["waypoints"], last_t, decoded)
        gt.advance()
        if horizons == 1 or horizons % 20 == 0:
            print(f"[sim] h={horizons} t={last_t:.1f}s", flush=True)
        if stop_on_hard and (step["decode_hard"] or step["jump_abort"]):
            stop = (step["decode_hard"] or step["hard"] or ["aborted"])[0]
            break
    return {"t_s": float(last_t), "horizons": horizons, "stop": stop}


def _summary_motion(foot_err, tilt, normal, clearance, guards, motion) -> dict:
    foot = np.concatenate(foot_err) if foot_err else np.zeros(0)
    tilt_v = np.concatenate(tilt) if tilt else np.zeros(0)
    normal_v = np.concatenate(normal) if normal else np.zeros((0, 3))
    clear = np.asarray(clearance, dtype=np.float64)
    return {
        "t_s": motion["t_s"],
        "horizons": motion["horizons"],
        "stop": motion["stop"],
        "guards": guards,
        "min_camera_clearance_mm": None if clear.size == 0 else float(np.min(clear) * 1e3),
        "foot_error_mm": _stats(foot * 1e3),
        "tilt_deg": _tilt_regions(tilt_v, normal_v) if tilt_v.size else {},
    }


def label_check(episode: int, bench: bool) -> dict:
    ep = _episode(episode)
    ep["_episode"] = episode
    lift = _apply_bench(bench)
    runtime, _t = make_runtime(bench)
    foot = install_causal_foot()
    label_x, label_R = foot.track(ep["p"])
    parity = label_parity(runtime, ep, label_x, label_R)
    runtime.reset()
    folder, log = _open_log("label_check", f"ep{episode}")
    guards, clearance, foot_err, tilt, normal = [], [], [], [], []

    def at_demo(gt):
        return gt.demo[int(gt.windows[gt.k][0])]

    motion = run_gt(
        ep, runtime, at_demo, lambda _pose, _t, _decoded=None: None, lift, log, guards, clearance, foot_err, tilt, normal
    )
    summary = {
        "mode": "label_check",
        "episode": episode,
        "bench": bench,
        "foot_climb": os.environ.get("UMIFT_FOOT_CLIMB", ""),
        "runner_sha": _runner_sha(),
        "runner_vs_label_safety_off": parity,
        "demo_tilt_deg": demo_tilt(ep),
        "open_loop": _summary_motion(foot_err, tilt, normal, clearance, guards, motion),
    }
    _finish(folder, summary)
    return summary


def _rgb_tilt(ep: dict, normal: np.ndarray) -> dict:
    k = np.clip(np.searchsorted(ep["robot_t"], ep["rgb_t"]), 0, len(ep["robot_t"]) - 1)
    return _tilt_regions(_tilt_from_normal(ep["R"][k], normal[k]), normal[k])


def _surface_point(runtime, position: np.ndarray) -> np.ndarray:
    """Closest mesh point to a commanded TCP. Rotation is left to the caller."""
    surface, _vid, _w = runtime.field.closest(np.asarray(position, dtype=np.float64).reshape(1, 3))
    return np.asarray(surface, dtype=np.float64).reshape(3)


def replay(episode: int, bench: bool, lag: int, contact: bool = False) -> dict:
    ep = _episode(episode)
    ep["_episode"] = episode
    lift = _apply_bench(bench)
    runtime, _t = make_runtime(bench)
    source = causal_source(ep)
    ep["foot"] = source.foot
    from mold_aware_runtime import shift_fixture_up

    # The TCP step is relative to the first label index. pose[0] can be earlier.
    start_index = int(source.windows[0][0]) if decode_vpar_enabled() else 0
    start = ep["pose"][start_index].copy()
    if lift:
        start = shift_fixture_up(start.reshape(1, 7), lift).reshape(7)
    def project(position):
        return _surface_point(runtime, position)

    arm = FakeArm(start, lag=lag, contact=contact, project=project if contact else None)
    log_pose = PoseLog(float(ep["robot_t"][0]), start)
    folder, log = _open_log("replay", f"ep{episode}")
    guards, clearance, foot_err, tilt, normal = [], [], [], [], []
    tang_cmd, tang_demo = [], []
    previous = {"x": None, "index": None}

    def on_command(waypoints, t_s, decoded=None):
        if waypoints is None:
            return
        t0 = float(ep["robot_t"][0]) + t_s
        last = len(waypoints) - 1
        for i, pose in enumerate(waypoints):
            logged = np.asarray(pose, dtype=np.float64).reshape(7).copy()
            if contact:
                logged[:3] = project(logged[:3])
            log_pose.add(t0 + i * DT, logged)
        arm.command(waypoints[last])
        if decoded is None:
            return
        window = source.windows[source.k]
        commanded = np.asarray(decoded["x_d"], dtype=np.float64)
        outward = np.asarray(decoded["n_out"], dtype=np.float64)
        for i in range(commanded.shape[0]):
            index = int(window[i])
            if previous["x"] is not None:
                normal_i = outward[i]
                normal_i = normal_i / max(float(np.linalg.norm(normal_i)), 1e-12)
                step = commanded[i] - previous["x"]
                tang_cmd.append(float(np.linalg.norm(step - normal_i * np.dot(step, normal_i))))
                demo_step = ep["p"][index] - ep["p"][previous["index"]]
                tang_demo.append(float(np.linalg.norm(demo_step - normal_i * np.dot(demo_step, normal_i))))
            previous["x"] = commanded[i].copy()
            previous["index"] = index

    motion = run_gt(
        ep,
        runtime,
        lambda _gt: arm.measure(),
        on_command,
        lift,
        log,
        guards,
        clearance,
        foot_err,
        tilt,
        normal,
        source=source,
    )
    summary = {
        "mode": "replay",
        "episode": episode,
        "bench": bench,
        "lag": lag,
        "contact": contact,
        "integrate_v_par": decode_vpar_enabled(),
        "tangential_step_mm": {
            "commanded": _stats(np.asarray(tang_cmd) * 1e3),
            "demo": _stats(np.asarray(tang_demo) * 1e3),
        },
        "foot_climb": os.environ.get("UMIFT_FOOT_CLIMB", ""),
        "label_tilt_deg": {
            "causal": _rgb_tilt(ep, source.normal),
            "track_clean": _rgb_tilt(ep, ep["normal"]),
        },
        "runner_sha": _runner_sha(),
        "demo_tilt_deg": demo_tilt(ep),
        **_summary_motion(foot_err, tilt, normal, clearance, guards, motion),
    }
    _finish(folder, summary)
    return summary


def _tail(values, n: int) -> np.ndarray:
    values = np.asarray(values)
    if len(values) >= n:
        return values[-n:]
    pad = np.repeat(values[:1], n - len(values), axis=0)
    return np.concatenate([pad, values], axis=0)


def policy_obs(shape_meta, ep, pose_log, now_abs: float, runtime) -> dict:
    """Full-rate tails ending at now. The runner's MPC strides these."""
    import PyriteUtility.spatial_math.spatial_utilities as su

    robot_k = int(np.searchsorted(ep["robot_t"], now_abs, side="right"))
    robot_k = max(1, min(robot_k, len(ep["robot_t"])))
    times = ep["robot_t"][:robot_k]
    poses = pose_log.at(times)
    obs_raw = {
        "ts_pose_fb_0": poses,
        "robot_time_stamps_0": times,
    }
    runtime.fill_obs_raw(obs_raw)
    sparse = shape_meta["sample"]["obs"]["sparse"]
    obs = {}
    for key, attr in sparse.items():
        need = (int(attr["horizon"]) - 1) * int(attr["down_sample_steps"]) + 1
        if key == "rgb_0":
            k = int(np.searchsorted(ep["rgb_t"], now_abs, side="right"))
            k = max(1, min(k, len(ep["rgb_t"])))
            obs[key] = _tail(np.asarray(ep["rgb"][max(0, k - need) : k]), need)
        elif "depth" in key:
            obs[key] = np.zeros((need, 224, 224, 3), dtype=np.float16)
        elif key in ("robot0_eef_pos", "robot0_eef_rot_axis_angle"):
            pose9 = su.SE3_to_pose9(su.pose7_to_SE3(poses))
            part = pose9[:, :3] if key.endswith("pos") else pose9[:, 3:]
            obs[key] = _tail(part, need)
        elif key == "robot0_mold_geom":
            geom = np.concatenate(
                [
                    obs_raw["mold_signed_distance_0"],
                    obs_raw["mold_normal_0"],
                    obs_raw["mold_R_field_rot6d_0"],
                ],
                axis=-1,
            )
            obs[key] = _tail(geom, need)
        elif "wrench_left" in key:
            k = int(np.searchsorted(ep["wrench_t"], now_abs, side="right"))
            k = max(1, min(k, len(ep["wrench_t"])))
            obs[key] = _tail(ep["wrench"][max(0, k - need) : k], need)
        elif "wrench_right" in key:
            obs[key] = np.zeros((need, 6), dtype=np.float32)
        else:
            raise RuntimeError(f"sim has no tape for obs key {key}")
    return obs


def _policy_action(policy, controller, shape_meta, device, ep, pose_log, now_abs: float, runtime):
    obs = policy_obs(shape_meta, ep, pose_log, now_abs, runtime)
    obs_hard = list(runtime.last_hard_triggers)
    controller.compute_one_horizon_action(obs, device)
    return np.asarray(controller.last_raw_action, dtype=np.float64), obs_hard


def rollout(episode: int, ckpt: str, seconds: float, lag: int) -> dict:
    """Closed-loop policy at 15 Hz. ponytail: rgb and wrench stay on the episode tape."""
    import torch
    from PyriteUtility.planning_control.mpc import ModelPredictiveController
    from PyriteUtility.pytorch_utils.model_io import load_policy
    from jump_guard import TargetJumpGuard
    from mold_aware_runtime import FixBActionEnsembler, shift_fixture_up

    ep = _episode(episode)
    ep["_episode"] = episode
    lift = _apply_bench(False)
    runtime, _t = make_runtime(False)
    folders = os.environ.get("PYRITE_CHECKPOINT_FOLDERS", "/mnt/data/umidata/ft/training_outputs")
    path = ckpt if os.path.isfile(ckpt) else str(Path(folders) / ckpt.lstrip("/"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, shape_meta = load_policy(path, device)
    controller = ModelPredictiveController(
        shape_meta=shape_meta,
        id_list=[0],
        policy=policy,
        action_to_trajectory=lambda *args, **kwargs: None,
        execution_horizon=EXEC_H,
        fix_orientation=False,
    )
    controller.verbose_level = -1
    start = ep["pose"][0].copy()
    if lift:
        start = shift_fixture_up(start.reshape(1, 7), lift).reshape(7)
    arm = FakeArm(start, lag=lag)
    pose_log = PoseLog(float(ep["robot_t"][0]), start)
    ens = FixBActionEnsembler(decay_m=DECAY)
    guard = TargetJumpGuard()
    folder, log = _open_log("rollout", Path(path).parent.parent.name[:40])
    guards, clearance, foot_err, tilt, normal = [], [], [], [], []
    t0 = float(ep["robot_t"][0])
    horizons = 0
    last_t = 0.0
    stop = "duration"
    while last_t < float(seconds) - 1e-9:
        t_abs = t0 + last_t
        pose7 = arm.measure()
        wrench = _nearest_wrench(ep, min(t_abs, float(ep["wrench_t"][-1])))
        decision = runtime.wrench_hold_update(float(np.linalg.norm(wrench[:3])))
        if decision == "hold":
            _note(guards, "wrench_hold", last_t)
            last_t += EXEC_H * DT
            continue
        if decision == "abort":
            _note(guards, "wrench_abort", last_t)
            stop = "wrench_abort"
            break
        raw, obs_hard = _policy_action(
            policy, controller, shape_meta, device, ep, pose_log, t_abs, runtime
        )
        for name in obs_hard:
            _note(guards, name, last_t)
        if obs_hard:
            stop = obs_hard[0]
            break
        step = runner_step(runtime, ens, guard, horizons, raw, pose7, None, lift)
        for row in step["guards"]:
            _note(guards, row["name"], last_t)
        for name in step["hard"]:
            _note(guards, name, last_t)
        if step["clearance_m"] is not None:
            clearance.append(step["clearance_m"])
        decoded = step["decoded"]
        if decoded is not None:
            write_horizon(log, horizons, last_t, DT, decoded, wrench)
            if not step["decode_hard"]:
                for i, pose in enumerate(step["waypoints"]):
                    pose_log.add(t_abs + i * DT, pose)
                arm.command(step["waypoints"][-1])
            k = int(np.clip(np.searchsorted(ep["robot_t"], t_abs), 0, len(ep["foot"]) - 1))
            n = min(decoded["x_s"].shape[0], len(ep["foot"]) - k)
            if n > 0:
                foot_err.append(np.linalg.norm(decoded["x_s"][:n] - ep["foot"][k : k + n], axis=1))
                tilt.append(_tilt_from_normal(decoded["R_d"][:n], decoded["n_out"][:n]))
                normal.append(np.asarray(decoded["n_out"][:n], dtype=np.float64))
            horizons += 1
            if horizons == 1 or horizons % 20 == 0:
                print(f"[rollout] h={horizons} t={last_t:.1f}s", flush=True)
        last_t += EXEC_H * DT
        if step["decode_hard"] or step["jump_abort"]:
            stop = (step["decode_hard"] or step["hard"] or ["aborted"])[0]
            break
    summary = {
        "mode": "rollout",
        "episode": episode,
        "ckpt": path,
        "seconds": seconds,
        "lag": lag,
        "device": str(device),
        "foot_climb": os.environ.get("UMIFT_FOOT_CLIMB", ""),
        "runner_sha": _runner_sha(),
        "vision": "episode tape; pose is the fake arm",
        **_summary_motion(
            foot_err, tilt, normal, clearance, guards, {"t_s": last_t, "horizons": horizons, "stop": stop}
        ),
    }
    _finish(folder, summary)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--episode", type=int, default=40)
    common.add_argument("--bench", action="store_true")
    common.add_argument("--lag", type=int, default=0)
    common.add_argument(
        "--contact",
        action="store_true",
        help="report the closest surface point to the commanded TCP; keep its rotation",
    )
    common.add_argument(
        "--integrate-v-par",
        action="store_true",
        help="set UMIFT_DECODE_VPAR so x_d includes the tangential sum",
    )
    check = sub.add_parser("label_check", parents=[common])
    play = sub.add_parser("replay", parents=[common])
    roll = sub.add_parser("rollout", parents=[common])
    roll.add_argument("--ckpt", required=True)
    roll.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    boot()
    if args.mode == "label_check":
        label_check(args.episode, args.bench)
    elif args.mode == "replay":
        if args.integrate_v_par:
            os.environ["UMIFT_DECODE_VPAR"] = "1"
        replay(args.episode, args.bench, args.lag, args.contact)
    else:
        rollout(args.episode, args.ckpt, args.seconds, args.lag)
    del check, play


if __name__ == "__main__":
    main()
