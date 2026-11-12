#!/usr/bin/env python3
"""Replay October action labels through the normalizer and the runtime decode.

No checkpoint and no robot. Scores episodes offline and writes
/mnt/data/umidata/ft/replay_exports/gt_{r4vt,r1rel}/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np

from diffusion_policy.common.sampler_umift import (
    get_val_mask,
    sparse_action_indices_from_rgb_timestamps,
)
from PyriteConfig.tasks.umift.umift_type_conversions import (
    action21_postprocess,
    action21_to_action_sample,
    raw_to_action21,
)
from PyriteUtility.planning_control.temporal_ensemble import (
    ensemble_chunks_offline,
    resolve_execution_horizon,
)
import PyriteUtility.spatial_math.spatial_utilities as su
from jump_guard import MAX_STEP_DEG, MAX_STEP_MM, TargetJumpGuard
from mold_aware_runtime import (
    FixBActionEnsembler,
    blend_action_rows,
    fixb_decode_dt,
    mean_rotation,
)

DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
R4VT_ZARR = DATA / "r4vt_world_fixb_tilt" / "acp_replay_buffer.zarr"
R1REL_ZARR = DATA / "r1rel_world" / "acp_replay_buffer.zarr"
FIELD_NPZ = DATA / "shared" / "fixb_field.npz"
LONG_DT = DATA / "shared" / "long_dt_gt_100ms.npz"
ASSETS = Path(__file__).resolve().parent / "assets"
R4VT_NORMALIZER = ASSETS / "oct_v35_h4s_r4vt_action_normalizer_minmax_exbad.pkl"
R1REL_NORMALIZER = ASSETS / "oct_v35_h4s_r1rel_odom_action_pose_normalizer_2c20fba56851.pkl"
R4VT_SHA = "ac13d3fd7b647541d093688896ff43f1edb3a025397b6037da01dfffb6707bee"
R1REL_SHA = "2c20fba56851f59ec7d1a652d4e08a979ec9f7a8ea46368737df984475a0cf05"
EXPORT_ROOT = Path("/mnt/data/umidata/ft/replay_exports")

ACTION_HORIZON = 64
EXEC_HORIZONS = (6, 32)
RUNNER_DT_S = 1.0 / 15.0  # fixb_decode_dt; slow-down stays on the waypoint clock
DECAY_M = 0.01
PRESS_FLOOR_M = -0.005
CONTACT_N = 8.0
EXCLUDED_R4VT = (35, 44)
VAL_SEED = 42
VAL_RATIO = 0.1

# Configured execution horizon is 6. raise_exec_horizon_64_to_32 defaults off,
# so a 64-step checkpoint stays at 6. Turn the key on to restore the raise to 32.

_FIELD = None
_SCALE = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    raw = np.ascontiguousarray(values)
    return hashlib.sha256(raw.view(np.uint8)).hexdigest()


def geodesic_deg(ra: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """Angle of R_pred^T R_label, in degrees."""
    rel = np.einsum("...ji,...jk->...ik", ra, rb)
    tr = np.trace(rel, axis1=-2, axis2=-1)
    cos = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def load_action_scale(path: Path, expected_sha: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    digest = sha256_file(path)
    if digest != expected_sha:
        raise RuntimeError(
            f"action normalizer sha256 {digest} != {expected_sha} ({path})"
        )
    key = str(path)
    if key not in _SCALE:
        with path.open("rb") as handle:
            saved = pickle.load(handle)
        src = saved.params_dict["action"]
        scale = src["scale"].detach().float().cpu().numpy().reshape(-1)
        offset = src["offset"].detach().float().cpu().numpy().reshape(-1)
        _SCALE[key] = (scale.astype(np.float64), offset.astype(np.float64))
    return _SCALE[key]


def roundtrip_action(action: np.ndarray, scale: np.ndarray, offset: np.ndarray) -> np.ndarray:
    out = np.array(action, dtype=np.float64, copy=True)
    width = int(scale.shape[0])
    if out.shape[-1] < width:
        raise ValueError(f"action dim {out.shape[-1]} < normalizer dim {width}")
    got = out[..., :width]
    normed = got * scale + offset
    out[..., :width] = (normed - offset) / scale
    return out


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"n": 0, "median": None, "max": None}
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def stiffness_buckets(err, contact, bad, long_dt) -> dict:
    """Abs stiffness error. Long-dt rows are counted and left out of every bucket."""
    err = np.asarray(err, dtype=np.float64).reshape(-1)
    contact = np.asarray(contact, dtype=bool).reshape(-1)
    bad = np.asarray(bad, dtype=bool).reshape(-1)
    long_dt = np.asarray(long_dt, dtype=bool).reshape(-1)
    keep = ~long_dt
    return {
        "clean_contact": _stats(err[keep & ~bad & contact]),
        "clean_free": _stats(err[keep & ~bad & ~contact]),
        "bad_contact": _stats(err[keep & bad & contact]),
        "bad_free": _stats(err[keep & bad & ~contact]),
        "n_long_dt": int(np.sum(long_dt)),
    }


def blend_executed(chunks, execution_horizon: int, decay_m: float = DECAY_M) -> dict:
    return ensemble_chunks_offline(
        chunks,
        execution_horizon=int(execution_horizon),
        pose_keys=("pose7", "nominal_pose7"),
        linear_keys=("stiffness",),
        weight_mode="exp",
        decay_m=float(decay_m),
    )


def _pose_R(pose7: np.ndarray) -> np.ndarray:
    mats = su.pose7_to_SE3(np.asarray(pose7, dtype=np.float64).reshape(-1, 7))
    return mats[..., :3, :3]


def _inplane(points: np.ndarray, foot: np.ndarray, normal: np.ndarray) -> np.ndarray:
    delta = np.asarray(points, dtype=np.float64) - np.asarray(foot, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.clip(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12, None)
    along = np.sum(delta * normal, axis=-1, keepdims=True)
    return np.linalg.norm(delta - along * normal, axis=-1)


def _err_columns(decoded: np.ndarray, ensembled: np.ndarray, mask: np.ndarray) -> dict:
    decoded = np.asarray(decoded, dtype=np.float64).reshape(-1)
    ensembled = np.asarray(ensembled, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    left = decoded[mask]
    right = ensembled[mask]
    if left.size == 0:
        empty = None
        return {
            "n": 0,
            "decode_median": empty,
            "decode_max": empty,
            "ensemble_median": empty,
            "ensemble_max": empty,
            "added_median": empty,
            "added_max": empty,
        }
    added = right - left
    return {
        "n": int(left.size),
        "decode_median": float(np.median(left)),
        "decode_max": float(np.max(left)),
        "ensemble_median": float(np.median(right)),
        "ensemble_max": float(np.max(right)),
        "added_median": float(np.median(added)),
        "added_max": float(np.max(added)),
    }


def episode_ids_in_key_order(zarr_path: Path) -> list[int]:
    """Episode ids in the order zarr yields keys: string sort, not numeric."""
    import zarr

    keys = sorted(
        k for k in zarr.open(str(zarr_path), "r")["data"].keys() if str(k).startswith("episode_")
    )
    return [int(str(k).split("_")[-1]) for k in keys]


def val_episode_ids(zarr_path: Path) -> list[int]:
    ids = episode_ids_in_key_order(zarr_path)
    mask = get_val_mask(len(ids), VAL_RATIO, seed=VAL_SEED)
    return sorted(ep for ep, flag in zip(ids, mask) if flag)


def split_name(episode: int, zarr_path: Path) -> str:
    ids = episode_ids_in_key_order(zarr_path)
    mask = get_val_mask(len(ids), VAL_RATIO, seed=VAL_SEED)
    return "val" if bool(mask[ids.index(int(episode))]) else "train"


def _open_episode(zarr_path: Path, episode: int):
    import zarr

    return zarr.open(str(zarr_path), "r")["data"][f"episode_{episode}"]


def _action_rows(group, indices: np.ndarray, with_delta_r: bool) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    robot_t = np.asarray(group["robot_time_stamps_0"][idx], dtype=np.float64).reshape(-1)
    raw = {
        "ts_pose_command_0": np.asarray(group["ts_pose_command_0"][idx]),
        "ts_pose_virtual_target_0": np.asarray(group["ts_pose_virtual_target_0"][idx]),
        "stiffness_0": np.asarray(group["stiffness_0"][idx], dtype=np.float64).reshape(-1),
        "gripper_0": np.asarray(group["gripper_0"][idx], dtype=np.float64).reshape(-1, 1),
        "wrench_left_0": np.zeros((len(idx), 6), dtype=np.float64),
        "wrench_right_0": np.zeros((len(idx), 6), dtype=np.float64),
        "wrench_time_stamps_0": robot_t,
        "robot_time_stamps_0": robot_t,
    }
    if with_delta_r:
        raw["fixb_delta_R_rot6d_0"] = np.asarray(group["fixb_delta_R_rot6d_0"][idx])
    episode = {}
    raw_to_action21(raw, episode, [0])
    return np.asarray(episode["action"], dtype=np.float64)


def _tcp_at(group, index: int) -> np.ndarray:
    return np.asarray(group["ts_pose_fb_0"][index], dtype=np.float64).reshape(7)


def r1rel_chunk_errors(episode: int = 0, rgb_id: int = 100) -> dict:
    """One relative chunk, normalizer round trip, stock decode, no ensemble."""
    group = _open_episode(R1REL_ZARR, episode)
    robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    indices, _times = sparse_action_indices_from_rgb_timestamps(
        robot_t, rgb_t, int(rgb_id), ACTION_HORIZON, 1
    )
    if len(indices) < ACTION_HORIZON:
        raise RuntimeError(f"episode {episode} rgb {rgb_id} has {len(indices)} action steps")
    indices = indices[:ACTION_HORIZON]
    scale, offset = load_action_scale(R1REL_NORMALIZER, R1REL_SHA)
    absolute = _action_rows(group, indices, with_delta_r=False)
    base = su.pose7_to_SE3(_tcp_at(group, int(indices[0])).reshape(1, 7))[0]
    relative = action21_to_action_sample(absolute, np.zeros((0,)), [0], [base])["sparse"]
    relative = roundtrip_action(relative, scale, offset)
    nominal, _vt, _stiff, _eoat = action21_postprocess(relative, [base], [0])
    decoded = su.SE3_to_pose7(nominal[0])
    demo = np.asarray(group["ts_pose_fb_0"][indices], dtype=np.float64)
    pos = np.linalg.norm(decoded[:, :3] - demo[:, :3], axis=1)
    ang = geodesic_deg(_pose_R(decoded), _pose_R(demo))
    return {"pos_m": float(np.max(pos)), "rot_deg": float(np.max(ang))}


def _mold_field():
    global _FIELD
    if _FIELD is None:
        from fixb_field import FixBMoldField

        _FIELD = FixBMoldField(FIELD_NPZ)
    return _FIELD


def _r4_runtime():
    from mold_aware_runtime import MoldAwareRuntime

    return MoldAwareRuntime(
        field=_mold_field(),
        R_off=np.eye(3),
        run_type="r4_world_fixb_tilt",
        dry_run=False,
    )


def r4vt_label_step_errors(episode: int = 0, index: int = 6314) -> dict:
    """One label step. Anchor is the stored foot, dt is fixb_dt_next_0."""
    group = _open_episode(R4VT_ZARR, episode)
    index = int(index)
    scale, offset = load_action_scale(R4VT_NORMALIZER, R4VT_SHA)
    action = roundtrip_action(
        _action_rows(group, np.array([index, index]), with_delta_r=True),
        scale,
        offset,
    )
    robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    dt = float(np.asarray(group["fixb_dt_next_0"][index]).reshape(-1)[0])
    foot = np.asarray(group["mold_x_s_0"][index], dtype=np.float64).reshape(3)
    tool_r = _pose_R(_tcp_at(group, index))[0]
    decoded = _r4_runtime().decode_chunk(
        action, foot, tool_r, np.array([dt, dt], dtype=np.float64), apply_safety=False
    )
    nxt = int(np.clip(np.searchsorted(robot_t, robot_t[index] + dt), 0, len(robot_t) - 1))
    stored_dn = float(np.asarray(group["fixb_delta_n_0"][index]).reshape(-1)[0])
    next_foot = np.asarray(group["mold_x_s_0"][nxt], dtype=np.float64).reshape(3)
    ang = geodesic_deg(decoded["R_d"][0], tool_r)
    return {
        "depth_m": abs(float(np.asarray(decoded["sd"][0]).reshape(-1)[0]) - stored_dn),
        "foot_m": float(np.linalg.norm(decoded["x_s"][0] - foot)),
        "foot_next_m": float(np.linalg.norm(decoded["x_s"][1] - next_foot)),
        "rot_deg": float(np.asarray(ang).reshape(-1)[0]),
    }


def _force_on_robot(group, robot_t: np.ndarray) -> np.ndarray:
    wrench_t = np.asarray(group["wrench_time_stamps_0"], dtype=np.float64).reshape(-1)
    force = np.linalg.norm(np.asarray(group["wrench_left_0"], dtype=np.float64)[:, :3], axis=1)
    order = np.argsort(wrench_t, kind="mergesort")
    wrench_t, force = wrench_t[order], force[order]
    keep = np.ones(len(wrench_t), dtype=bool)
    keep[1:] = np.diff(wrench_t) > 1e-9
    return np.interp(robot_t, wrench_t[keep], force[keep])


def _long_dt_mask(episode: int, n_robot: int) -> np.ndarray:
    blob = np.load(LONG_DT)
    mask = np.zeros(n_robot, dtype=bool)
    for ep, idx in zip(blob["flagged_episode"], blob["flagged_index"]):
        if int(ep) == int(episode) and 0 <= int(idx) < n_robot:
            mask[int(idx)] = True
    return mask


def safety_floor_report(delta_n: np.ndarray) -> dict:
    delta_n = np.asarray(delta_n, dtype=np.float64).reshape(-1)
    hit = delta_n < PRESS_FLOOR_M
    cut = PRESS_FLOOR_M - delta_n[hit]
    return {
        "threshold_m": PRESS_FLOOR_M,
        "n": int(delta_n.size),
        "n_clamped": int(np.sum(hit)),
        "median_cut_m": None if cut.size == 0 else float(np.median(cut)),
        "max_cut_m": None if cut.size == 0 else float(np.max(cut)),
    }


def _windows(robot_t, rgb_t, exec_h: int):
    windows = []
    for rgb_id in range(0, len(rgb_t), exec_h):
        indices, _times = sparse_action_indices_from_rgb_timestamps(
            robot_t, rgb_t, rgb_id, ACTION_HORIZON, 1
        )
        if len(indices) < ACTION_HORIZON:
            break
        windows.append(np.asarray(indices[:ACTION_HORIZON], dtype=np.int64))
    return windows


def pushed_action_chunks(arm: str, episode: int, exec_h: int):
    """Actions the runner ensembler receives for one episode.

    r4vt: round-tripped action27, one (64, 27) array per window. These are the
    arrays ``FixBActionEnsembler.push`` gets. r1rel: relative action21 against
    the demo TCP at the window start, the array ``action21_postprocess`` gets
    before the pose ensembler. Windows step by ``exec_h`` RGB frames.
    """
    if arm == "r4vt":
        zarr_path, norm_path, sha = R4VT_ZARR, R4VT_NORMALIZER, R4VT_SHA
        with_delta_r = True
    elif arm == "r1rel":
        zarr_path, norm_path, sha = R1REL_ZARR, R1REL_NORMALIZER, R1REL_SHA
        with_delta_r = False
    else:
        raise ValueError(arm)
    group = _open_episode(zarr_path, episode)
    scale, offset = load_action_scale(norm_path, sha)
    robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    demo = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    actions = []
    windows = _windows(robot_t, rgb_t, exec_h)
    for indices in windows:
        action = roundtrip_action(
            _action_rows(group, indices, with_delta_r=with_delta_r), scale, offset
        )
        if arm == "r1rel":
            base = su.pose7_to_SE3(demo[int(indices[0])].reshape(1, 7))[0]
            relative = action21_to_action_sample(action, np.zeros((0,)), [0], [base])["sparse"]
            action = roundtrip_action(relative, scale, offset)
        actions.append(np.asarray(action, dtype=np.float64))
    return actions, windows


def _nominal_pose(action_or_pose, base, kind: str, decoded=None) -> np.ndarray:
    if kind == "r1rel":
        nominal, _vt, stiff, _eoat = action21_postprocess(action_or_pose, [base], [0])
        return su.SE3_to_pose7(nominal[0]), np.asarray(stiff[0], dtype=np.float64).reshape(-1)
    from mold_aware_runtime import Rp_to_SE3

    pose = su.SE3_to_pose7(Rp_to_SE3(decoded["R_d"], decoded["x_d"]))
    return pose, np.asarray(decoded["stiffness"], dtype=np.float64).reshape(-1)


def replay_horizon(group, arm: str, episode: int, exec_h: int, scale, offset, runtime) -> dict:
    robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    demo = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    label_stiff = np.asarray(group["stiffness_0"], dtype=np.float64).reshape(-1)
    bad = np.asarray(group["bad_span_0"]).reshape(-1) > 0
    long_dt = _long_dt_mask(episode, len(robot_t))
    force = _force_on_robot(group, robot_t)
    contact = force > CONTACT_N
    windows = _windows(robot_t, rgb_t, exec_h)
    chunks = []
    dec_pos = []
    dec_rot = []
    dec_stiff = []
    exec_idx = []
    drift_end = []
    drift_peak = []
    r4 = arm == "r4vt"
    stored_foot = np.asarray(group["mold_x_s_0"], dtype=np.float64) if r4 else None
    stored_n = np.asarray(group["mold_normal_0"], dtype=np.float64) if r4 else None
    stored_dn = np.asarray(group["fixb_delta_n_0"], dtype=np.float64).reshape(-1) if r4 else None

    for indices in windows:
        action = roundtrip_action(
            _action_rows(group, indices, with_delta_r=r4), scale, offset
        )
        head = int(indices[0])
        base = su.pose7_to_SE3(demo[head].reshape(1, 7))[0]
        if r4:
            tool_r = _pose_R(demo[head])[0]
            decoded = runtime.decode_chunk(
                action,
                demo[head, :3],
                tool_r,
                fixb_decode_dt(ACTION_HORIZON, 1, RUNNER_DT_S),
                apply_safety=False,
            )
            pose, stiff = _nominal_pose(None, base, "r4vt", decoded)
            sl = slice(0, exec_h)
            foot_err = _inplane(decoded["x_s"][sl], stored_foot[indices[sl]], stored_n[indices[sl]])
            depth_err = np.abs(np.asarray(decoded["sd"][sl], dtype=np.float64) - stored_dn[indices[sl]])
            dec_pos.append(np.stack([depth_err, foot_err], axis=1))
            drift_end.append(float(foot_err[-1]))
            drift_peak.append(float(np.max(foot_err)))
            dec_rot.append(geodesic_deg(decoded["R_d"][sl], _pose_R(demo[indices[sl]])))
        else:
            relative = action21_to_action_sample(action, np.zeros((0,)), [0], [base])["sparse"]
            relative = roundtrip_action(relative, scale, offset)
            pose, stiff = _nominal_pose(relative, base, "r1rel")
            dec_pos.append(np.linalg.norm(pose[:exec_h, :3] - demo[indices[:exec_h], :3], axis=1))
            dec_rot.append(geodesic_deg(_pose_R(pose[:exec_h]), _pose_R(demo[indices[:exec_h]])))
        used = indices[:exec_h]
        dec_stiff.append(np.abs(stiff[:exec_h] - label_stiff[used]))
        exec_idx.append(used)
        chunks.append(
            {
                "pose7": pose,
                "nominal_pose7": pose,
                "stiffness": stiff,
            }
        )

    if not chunks:
        raise RuntimeError(f"{arm} episode {episode} produced no {exec_h}-step windows")
    if arm == "r1rel":
        from PyriteUtility.planning_control.temporal_ensemble import (
            ensemble_pose_chunks,
        )

        blended = ensemble_pose_chunks(chunks, exec_h, DECAY_M)
    else:
        blended = blend_executed(chunks, exec_h)
    idx = np.concatenate(exec_idx)
    clean = ~(bad[idx] | long_dt[idx])
    ens_pose = blended["nominal_pose7"]
    ens_stiff = np.abs(np.asarray(blended["stiffness"], dtype=np.float64).reshape(-1) - label_stiff[idx])
    ens_rot = geodesic_deg(_pose_R(ens_pose), _pose_R(demo[idx]))
    if r4:
        hard = np.concatenate(dec_pos, axis=0)
        queried = _mold_field().query(ens_pose[:, :3])
        ens_depth = np.abs(queried["sd"] - stored_dn[idx])
        ens_foot = _inplane(queried["x_s"], stored_foot[idx], stored_n[idx])
        position = {
            "depth_m": _err_columns(hard[:, 0], ens_depth, clean),
            "foot_inplane_m": _err_columns(hard[:, 1], ens_foot, clean),
        }
        drift = {
            "end_median_m": float(np.median(drift_end)),
            "end_max_m": float(np.max(drift_end)),
            "peak_median_m": float(np.median(drift_peak)),
            "peak_max_m": float(np.max(drift_peak)),
            "n_windows": len(drift_end),
        }
    else:
        hard = np.concatenate(dec_pos, axis=0)
        ens_pos = np.linalg.norm(ens_pose[:, :3] - demo[idx, :3], axis=1)
        position = {"tcp_m": _err_columns(hard, ens_pos, clean)}
        drift = None
    return {
        "execution_horizon": int(exec_h),
        "runner_dt_s": RUNNER_DT_S,
        "n_windows": len(windows),
        "n_executed": int(len(idx)),
        "n_clean": int(np.sum(clean)),
        "position": position,
        "orientation_deg": _err_columns(np.concatenate(dec_rot), ens_rot, clean),
        "stiffness": stiffness_buckets(
            np.concatenate(dec_stiff), contact[idx], bad[idx], long_dt[idx]
        ),
        "stiffness_ensemble": stiffness_buckets(
            ens_stiff, contact[idx], bad[idx], long_dt[idx]
        ),
        "foot_drift_before_reanchor": drift,
    }


def replay_episode(arm: str, episode: int, exec_horizons=EXEC_HORIZONS) -> dict:
    if arm == "r4vt" and int(episode) in EXCLUDED_R4VT:
        raise RuntimeError(f"r4vt does not replay episode {episode}")
    if arm == "r4vt":
        zarr_path, norm_path, sha = R4VT_ZARR, R4VT_NORMALIZER, R4VT_SHA
        runtime = _r4_runtime()
    elif arm == "r1rel":
        zarr_path, norm_path, sha = R1REL_ZARR, R1REL_NORMALIZER, R1REL_SHA
        runtime = None
    else:
        raise ValueError(arm)
    group = _open_episode(zarr_path, episode)
    scale, offset = load_action_scale(norm_path, sha)
    report = {
        "arm": arm,
        "episode": int(episode),
        "split": split_name(episode, zarr_path),
        "apply_safety": False,
        "contact_n": CONTACT_N,
        "horizons": {},
    }
    if arm == "r4vt":
        delta_n = np.asarray(group["fixb_delta_n_0"], dtype=np.float64).reshape(-1)
        report["safety_floor"] = safety_floor_report(delta_n)
        report["lowdim_sha256"] = {
            "fixb_delta_n_0": sha256_array(delta_n),
            "fixb_v_par_0": sha256_array(np.asarray(group["fixb_v_par_0"])),
            "ts_pose_fb_0": sha256_array(np.asarray(group["ts_pose_fb_0"])),
        }
    else:
        report["safety_floor"] = None
        report["lowdim_sha256"] = {
            "ts_pose_fb_0": sha256_array(np.asarray(group["ts_pose_fb_0"])),
            "stiffness_0": sha256_array(np.asarray(group["stiffness_0"])),
        }
    for exec_h in exec_horizons:
        report["horizons"][str(exec_h)] = replay_horizon(
            group, arm, episode, exec_h, scale, offset, runtime
        )
    return report


def _settings(arm: str, episodes) -> dict:
    root = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    norm = R4VT_NORMALIZER if arm == "r4vt" else R1REL_NORMALIZER
    expected = R4VT_SHA if arm == "r4vt" else R1REL_SHA
    return {
        "arm": arm,
        "git": sha,
        "episodes": [int(ep) for ep in episodes],
        "val_episode_ids": val_episode_ids(R4VT_ZARR if arm == "r4vt" else R1REL_ZARR),
        "action_horizon": ACTION_HORIZON,
        "execution_horizons": list(EXEC_HORIZONS),
        "runner_dt_s": RUNNER_DT_S,
        "slow_down_factor": 2,
        "ensemble_decay_m": DECAY_M,
        "apply_safety": False,
        "press_floor_m": PRESS_FLOOR_M,
        "contact_n": CONTACT_N,
        "anchor": "demo TCP at the chunk start",
        "r4vt_label_test_anchor": "stored mold_x_s_0 and fixb_dt_next_0",
        "raise_exec_horizon_64_to_32": False,
        "resolved_execution_horizon": resolve_execution_horizon(action_horizon=ACTION_HORIZON),
        "sha256": {
            "normalizer": sha256_file(norm),
            "normalizer_expected": expected,
            "fixb_field": sha256_file(FIELD_NPZ),
            "long_dt_gt_100ms": sha256_file(LONG_DT),
        },
    }


def _mm(value) -> str:
    if value is None:
        return "n/a"
    return f"{value * 1e3:.3f}"


def format_report(report: dict) -> str:
    lines = [
        f"{report['arm']} episode {report['episode']} ({report['split']})",
    ]
    floor = report.get("safety_floor")
    if floor is not None:
        lines.append(
            "  safety floor -5 mm: "
            f"{floor['n_clamped']}/{floor['n']} steps, "
            f"cut median {_mm(floor['median_cut_m'])} mm, "
            f"max {_mm(floor['max_cut_m'])} mm"
        )
    for key, block in report["horizons"].items():
        lines.append(
            f"  horizon {key}: windows {block['n_windows']}, "
            f"clean {block['n_clean']}/{block['n_executed']}"
        )
        for name, cols in block["position"].items():
            lines.append(
                f"    {name}: decode med/max {_mm(cols['decode_median'])}/"
                f"{_mm(cols['decode_max'])} mm, "
                f"ensemble med/max {_mm(cols['ensemble_median'])}/"
                f"{_mm(cols['ensemble_max'])} mm, "
                f"added med {_mm(cols['added_median'])} mm"
            )
        ori = block["orientation_deg"]
        lines.append(
            "    orientation deg: decode med/max "
            f"{ori['decode_median']:.4f}/{ori['decode_max']:.4f}, "
            f"ensemble med/max {ori['ensemble_median']:.4f}/{ori['ensemble_max']:.4f}, "
            f"added med {ori['added_median']:.4f}"
        )
        drift = block["foot_drift_before_reanchor"]
        if drift is not None:
            lines.append(
                "    foot drift end med/max "
                f"{_mm(drift['end_median_m'])}/{_mm(drift['end_max_m'])} mm, "
                f"peak med/max {_mm(drift['peak_median_m'])}/{_mm(drift['peak_max_m'])} mm"
            )
        for bucket, stats in block["stiffness"].items():
            if not isinstance(stats, dict):
                continue
            ens = block["stiffness_ensemble"][bucket]
            lines.append(
                f"    stiffness {bucket}: n {stats['n']} "
                f"decode med/max {stats['median']}/{stats['max']} "
                f"ensemble med/max {ens['median']}/{ens['max']}"
            )
    return "\n".join(lines)


def write_exports(episodes=(0, 9)) -> list[Path]:
    written = []
    for arm in ("r4vt", "r1rel"):
        out = EXPORT_ROOT / f"gt_{arm}"
        settings_path = out / "settings.json"
        if settings_path.exists():
            raise SystemExit(f"refusing to overwrite {settings_path}")
        out.mkdir(parents=True, exist_ok=True)
        for episode in episodes:
            report = replay_episode(arm, int(episode))
            path = out / f"episode_{int(episode)}.json"
            path.write_text(json.dumps(report, indent=2) + "\n")
            written.append(path)
            print(format_report(report), flush=True)
        settings = _settings(arm, episodes)
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
        written.append(settings_path)
    return written


def _frame_flags(r_dec: np.ndarray, r_stored: np.ndarray) -> tuple[bool, bool]:
    n_dot = float(np.dot(r_dec[:, 2], r_stored[:, 2]))
    t_dot = float(np.dot(r_dec[:, 0], r_stored[:, 0]))
    return n_dot > 0.5 and t_dot < 0.0, n_dot < 0.0


def _rgb_aligned_dt(group) -> dict:
    robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    label_dt = np.asarray(group["fixb_dt_next_0"], dtype=np.float64).reshape(-1)
    indices, _times = sparse_action_indices_from_rgb_timestamps(
        robot_t, rgb_t, 0, len(rgb_t), 1
    )
    at_rgb = label_dt[indices]
    p5, p50, p95 = np.percentile(at_rgb, [5, 50, 95])
    return {
        "rgb_period_s": float(np.median(np.diff(rgb_t))),
        "all_robot_p50_s": float(np.median(label_dt)),
        "rgb_aligned_p5_s": float(p5),
        "rgb_aligned_p50_s": float(p50),
        "rgb_aligned_p95_s": float(p95),
        "n_rgb": int(len(at_rgb)),
    }


def _column(values, mask) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.asarray(mask, dtype=bool).reshape(-1)]
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def action_blend_episode(episode: int, exec_h: int, dt_s: float = 1.0 / 15.0) -> dict:
    """Ensemble off, pose blend, and action blend at one dt. Re-anchors every window."""
    group = _open_episode(R4VT_ZARR, episode)
    runtime = _r4_runtime()
    robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    demo = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    bad = np.asarray(group["bad_span_0"]).reshape(-1) > 0
    long_dt = _long_dt_mask(episode, len(robot_t))
    stored_foot = np.asarray(group["mold_x_s_0"], dtype=np.float64)
    stored_n = np.asarray(group["mold_normal_0"], dtype=np.float64)
    stored_dn = np.asarray(group["fixb_delta_n_0"], dtype=np.float64).reshape(-1)
    stored_r = np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
    actions, windows = pushed_action_chunks("r4vt", episode, exec_h)
    chunks = []
    foot = {"off": [], "pose": [], "action": []}
    depth = {"off": [], "pose": [], "action": []}
    rot = {"off": [], "pose": [], "action": []}
    used_idx = []
    kept_idx = []
    outliers = []
    field = _mold_field()
    ensembler = FixBActionEnsembler(decay_m=DECAY_M)
    jump_guard = TargetJumpGuard()
    dt_chunk = fixb_decode_dt(ACTION_HORIZON, 1, RUNNER_DT_S)
    dt_exec = fixb_decode_dt(exec_h, 1, RUNNER_DT_S)
    for k, indices in enumerate(windows):
        head = int(indices[0])
        decoded = runtime.decode_chunk(
            actions[k],
            demo[head, :3],
            _pose_R(demo[head])[0],
            dt_chunk,
            apply_safety=False,
        )
        pose, stiff = _nominal_pose(None, None, "r4vt", decoded)
        chunks.append({"pose7": pose, "nominal_pose7": pose, "stiffness": stiff})
        sl = slice(0, exec_h)
        used = indices[sl]
        tool_r = _pose_R(demo[used])
        foot_off = _inplane(decoded["x_s"][sl], stored_foot[used], stored_n[used])
        rot_off = geodesic_deg(decoded["R_d"][sl], tool_r)
        foot["off"].append(foot_off)
        depth["off"].append(np.abs(np.asarray(decoded["sd"][sl]) - stored_dn[used]))
        rot["off"].append(rot_off)
        for j in np.flatnonzero(rot_off > 90.0):
            t1_flip, n_flip = _frame_flags(decoded["R_field"][j], stored_r[int(used[j])])
            outliers.append(
                {
                    "robot_index": int(used[j]),
                    "step_in_window": int(j),
                    "rot_deg": float(rot_off[j]),
                    "foot_m": float(foot_off[j]),
                    "t1_flip": t1_flip,
                    "n_flip": n_flip,
                }
            )
        ensembler.push(k * int(exec_h), actions[k])
        action_decoded = runtime.decode_chunk(
            ensembler.pop(exec_h),
            demo[head, :3],
            _pose_R(demo[head])[0],
            dt_exec,
            apply_safety=False,
        )
        x_g, R_g, _field_g, aborted = jump_guard.apply(
            action_decoded["x_d"],
            action_decoded["R_d"],
            robot_index=used,
            R_field=action_decoded["R_field"],
        )
        n_keep = int(x_g.shape[0])
        if n_keep == 0:
            chunks.pop()
            foot["off"].pop()
            depth["off"].pop()
            rot["off"].pop()
            break
        if n_keep < len(used):
            foot["off"][-1] = foot["off"][-1][:n_keep]
            depth["off"][-1] = depth["off"][-1][:n_keep]
            rot["off"][-1] = rot["off"][-1][:n_keep]
            used = used[:n_keep]
            tool_r = tool_r[:n_keep]
            action_decoded["x_s"] = action_decoded["x_s"][:n_keep]
            action_decoded["sd"] = action_decoded["sd"][:n_keep]
        action_decoded["x_d"] = x_g
        action_decoded["R_d"] = R_g
        kept_idx.append(used)
        foot["action"].append(
            _inplane(action_decoded["x_s"], stored_foot[used], stored_n[used])
        )
        depth["action"].append(np.abs(np.asarray(action_decoded["sd"]) - stored_dn[used]))
        rot["action"].append(geodesic_deg(action_decoded["R_d"], tool_r))
        if n_keep < exec_h:
            chunks.pop()
        else:
            used_idx.append(used)
        if aborted:
            break
    kept = np.concatenate(kept_idx) if kept_idx else np.zeros(0, dtype=np.int64)
    idx = np.concatenate(used_idx) if used_idx else np.zeros(0, dtype=np.int64)
    clean_kept = ~(bad[kept] | long_dt[kept]) if kept.size else np.zeros(0, dtype=bool)
    clean = ~(bad[idx] | long_dt[idx]) if idx.size else np.zeros(0, dtype=bool)
    blended = blend_executed(chunks, exec_h) if chunks else None
    if blended is not None and idx.size:
        queried = field.query(blended["nominal_pose7"][:, :3])
        foot["pose"].append(_inplane(queried["x_s"], stored_foot[idx], stored_n[idx]))
        depth["pose"].append(np.abs(queried["sd"] - stored_dn[idx]))
        rot["pose"].append(geodesic_deg(_pose_R(blended["nominal_pose7"]), _pose_R(demo[idx])))
    else:
        foot["pose"].append(np.zeros(0))
        depth["pose"].append(np.zeros(0))
        rot["pose"].append(np.zeros(0))
    report = {
        "episode": int(episode),
        "split": split_name(episode, R4VT_ZARR),
        "dt_s": float(dt_s),
        "execution_horizon": int(exec_h),
        "label_dt": _rgb_aligned_dt(group),
        "decode_dt_s": float(dt_chunk[0]),
        "rotation_gt_90_ensemble_off": outliers,
        "jump_guard": {
            "max_step_mm": float(MAX_STEP_MM),
            "max_step_deg": float(MAX_STEP_DEG),
            "n_holds": int(sum(1 for row in jump_guard.holds if not row["abort"])),
            "aborted": bool(jump_guard.aborted),
            "holds": jump_guard.holds,
        },
    }
    clean_for = {"off": clean_kept, "action": clean_kept, "pose": clean}
    for name in ("off", "pose", "action"):
        report[name] = {
            "foot_m": _column(np.concatenate(foot[name]), clean_for[name]),
            "depth_m": _column(np.concatenate(depth[name]), clean_for[name]),
            "rotation_deg": _column(np.concatenate(rot[name]), clean_for[name]),
        }
    return report


def write_runtime_replay(episodes=(0, 9)) -> None:
    """Episodes 0 and 9 through the runner's dt and FixBActionEnsembler."""
    out = EXPORT_ROOT / "gt_r4_runtime_fix"
    settings = out / "settings.json"
    if settings.exists():
        raise SystemExit(f"refusing to overwrite {settings}")
    out.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        for exec_h in EXEC_HORIZONS:
            report = action_blend_episode(int(episode), int(exec_h))
            path = out / f"episode_{int(episode)}_h{int(exec_h)}.json"
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(_format_action_blend(report), flush=True)
            big = report["rotation_gt_90_ensemble_off"]
            print(f"  rot>90 ensemble off: {len(big)}", flush=True)
            for row in big:
                print(
                    f"    idx {row['robot_index']} step {row['step_in_window']} "
                    f"rot {row['rot_deg']:.1f} foot {row['foot_m'] * 1e3:.1f} mm "
                    f"t1_flip {row['t1_flip']} n_flip {row['n_flip']}",
                    flush=True,
                )
    settings.write_text(
        json.dumps(
            {
                "episodes": [int(ep) for ep in episodes],
                "decode": "fixb_decode_dt(down_sample_steps=1, raw_time_step_s=1/15)",
                "ensemble": "FixBActionEnsembler",
                "execution_horizons": list(EXEC_HORIZONS),
            },
            indent=2,
        )
        + "\n"
    )


def write_action_blend(episodes=(0, 9)) -> None:
    out = EXPORT_ROOT / "gt_action_blend"
    settings = out / "settings.json"
    if settings.exists():
        raise SystemExit(f"refusing to overwrite {settings}")
    out.mkdir(parents=True, exist_ok=True)
    force_rows = []
    for episode in episodes:
        group = _open_episode(R4VT_ZARR, episode)
        robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        force = _force_on_robot(group, robot_t)
        contact = force >= 8.0
        p5, p50, p95, p99 = np.percentile(force[contact], [5, 50, 95, 99])
        force_rows.append(
            {
                "episode": int(episode),
                "n_contact": int(np.sum(contact)),
                "p5_n": float(p5),
                "p50_n": float(p50),
                "p95_n": float(p95),
                "p99_n": float(p99),
                "max_n": float(np.max(force[contact])),
            }
        )
    rows = []
    for episode in episodes:
        for exec_h in EXEC_HORIZONS:
            report = action_blend_episode(int(episode), int(exec_h), 1.0 / 15.0)
            rows.append(report)
            path = out / f"episode_{int(episode)}_h{int(exec_h)}.json"
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(_format_action_blend(report), flush=True)
    settings.write_text(
        json.dumps(
            {
                "episodes": [int(ep) for ep in episodes],
                "dt_s": 1.0 / 15.0,
                "execution_horizons": list(EXEC_HORIZONS),
                "decay_m": DECAY_M,
                "apply_safety": False,
                "contact_force_n": force_rows,
            },
            indent=2,
        )
        + "\n"
    )
    return rows


def _format_action_blend(report: dict) -> str:
    dt = report["label_dt"]
    lines = [
        f"episode {report['episode']} ({report['split']}) h {report['execution_horizon']}",
        (
            "  label dt all-robot p50 "
            f"{1e3 * dt['all_robot_p50_s']:.1f} ms, rgb-aligned p5/50/95 "
            f"{1e3 * dt['rgb_aligned_p5_s']:.1f}/"
            f"{1e3 * dt['rgb_aligned_p50_s']:.1f}/"
            f"{1e3 * dt['rgb_aligned_p95_s']:.1f} ms, "
            f"rgb period {1e3 * dt['rgb_period_s']:.1f} ms"
        ),
    ]
    for name in ("off", "pose", "action"):
        block = report[name]
        lines.append(
            f"  {name}: foot {_mm(block['foot_m']['median'])}/{_mm(block['foot_m']['max'])} "
            f"depth {_mm(block['depth_m']['median'])}/{_mm(block['depth_m']['max'])} "
            f"rot {block['rotation_deg']['median']:.3f}/{block['rotation_deg']['max']:.3f}"
        )
    guard = report.get("jump_guard")
    if guard:
        lines.append(
            f"  jump guard {guard['max_step_mm']:.2f} mm / {guard['max_step_deg']:.2f} deg, "
            f"holds {guard['n_holds']}, abort {guard['aborted']}"
        )
        for row in guard["holds"]:
            kind = "ABORT" if row["abort"] else "hold"
            lines.append(
                f"    {kind} idx {row['robot_index']} step {row['step']} "
                f"pos {row['pos_mm']:.1f} mm rot {row['rot_deg']:.1f} deg run {row['run']}"
            )
    return "\n".join(lines)


def write_runner_replay(out: Path, episodes, horizon: int) -> None:
    """r4vt action blend and r1rel pose blend, the runner's H-step path."""
    out = Path(out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    r4 = out / "r4vt"
    r1 = out / "r1rel"
    r4.mkdir(parents=True)
    r1.mkdir(parents=True)
    for episode in episodes:
        report = action_blend_episode(int(episode), int(horizon))
        path = r4 / f"episode_{int(episode)}_h{int(horizon)}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(_format_action_blend(report), flush=True)
        report = replay_episode("r1rel", int(episode), exec_horizons=(int(horizon),))
        path = r1 / f"episode_{int(episode)}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(format_report(report), flush=True)
    print(f"wrote {out}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("exports", "runner"), default="exports")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 9])
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.mode == "runner":
        if args.out is None:
            raise SystemExit("runner mode needs --out")
        write_runner_replay(args.out, args.episodes, args.horizon)
        return
    write_exports(tuple(args.episodes))


if __name__ == "__main__":
    main()
