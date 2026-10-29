#!/usr/bin/env python3
"""Shared chunk index and one action normalizer for October stringer_v3.5 Fix B.

Builds the same SequenceSamplerUmiFT index training uses (no image copy).
Fits one 21-d action normalizer on the r2 train chunks (identical to r4 dims
0:21). Range on 0:3, 9:12, 18:21. Identity on rot 3:9 and 12:18.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import zarr

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.sampler_umift import (
    SequenceSamplerUmiFT,
    get_val_mask,
    sparse_action_indices_from_rgb_timestamps,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from PyriteConfig.tasks.umift.umift_type_conversions import (
    action21_to_action_sample,
    raw_to_action21,
    raw_to_obs,
)

from fixb_dataset import apply_shared_action_normalizer
from fixb_r4_chunk_filter import FIELD_JUMP_DEG, build_episode_mask_table

ROOT = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb")
ASSETS = Path(__file__).resolve().parent / "assets"
RUNS = ("r1_world_world", "r2_world_fixb", "r4_world_fixb_tilt")
H = 24
STRIDE = 1
QUERY_DS = 2
VAL_RATIO = 0.1
SEED = 42


def shape_meta(action_dim: int, with_delta_r: bool) -> dict:
    raw = {
        "rgb_0": {"shape": [3, 224, 224], "type": "rgb"},
        "ts_pose_fb_0": {"shape": [7], "type": "low_dim"},
        "ts_pose_command_0": {"shape": [7], "type": "low_dim"},
        "ts_pose_virtual_target_0": {"shape": [7], "type": "low_dim"},
        "gripper_0": {"shape": [1], "type": "low_dim"},
        "stiffness_0": {"shape": [1], "type": "low_dim"},
        "wrench_left_0": {"shape": [6], "type": "low_dim"},
        "wrench_right_0": {"shape": [6], "type": "low_dim"},
        "rgb_time_stamps_0": {"shape": [1], "type": "timestamp"},
        "robot_time_stamps_0": {"shape": [1], "type": "timestamp"},
        "wrench_time_stamps_0": {"shape": [1], "type": "timestamp"},
    }
    if with_delta_r:
        raw["fixb_delta_R_rot6d_0"] = {"shape": [6], "type": "low_dim"}
    return {
        "pose_frames": {"obs": "absolute", "action": "absolute"},
        "id_list": [0],
        "raw": raw,
        "obs": {
            "rgb_0": {"shape": [3, 224, 224], "type": "rgb"},
            "robot0_eef_pos": {"shape": [3], "type": "low_dim"},
            "robot0_eef_rot_axis_angle": {"shape": [6], "type": "low_dim", "rotation_rep": "rotation_6d"},
            "robot0_eef_wrench_left": {"shape": [6], "type": "low_dim"},
            "robot0_eef_wrench_right": {"shape": [6], "type": "low_dim"},
            "rgb_time_stamps_0": {"shape": [1], "type": "timestamp"},
            "robot_time_stamps_0": {"shape": [1], "type": "timestamp"},
            "wrench_time_stamps_0": {"shape": [1], "type": "timestamp"},
        },
        "action": {"shape": [action_dim], "rotation_rep": "rotation_6d"},
        "sample": {
            "obs": {
                "sparse": {
                    "rgb_0": {"horizon": 5, "down_sample_steps": 4},
                    "robot0_eef_pos": {"horizon": 5, "down_sample_steps": 16},
                    "robot0_eef_rot_axis_angle": {"horizon": 5, "down_sample_steps": 16},
                    "robot0_eef_wrench_left": {"horizon": 32, "down_sample_steps": 1},
                    "robot0_eef_wrench_right": {"horizon": 32, "down_sample_steps": 1},
                }
            },
            "action": {"sparse": {"horizon": H, "down_sample_steps": STRIDE}},
            "training_duration_per_sparse_query": 5.0,
        },
    }


def fit_action_normalizer(data: np.ndarray) -> LinearNormalizer:
    """Same split as UmiFTDataset.get_normalizer for action21 or action27."""
    al = int(data.shape[-1])
    if al not in (21, 27):
        raise RuntimeError(f"action dim {al}")
    parts = [
        get_range_normalizer_from_stat(array_to_stats(data[:, 0:3])),
        get_identity_normalizer_from_stat(array_to_stats(data[:, 3:9])),
        get_range_normalizer_from_stat(array_to_stats(data[:, 9:12])),
        get_identity_normalizer_from_stat(array_to_stats(data[:, 12:18])),
        get_range_normalizer_from_stat(array_to_stats(data[:, 18:19])),
        get_range_normalizer_from_stat(array_to_stats(data[:, 19:20])),
        get_range_normalizer_from_stat(array_to_stats(data[:, 20:21])),
    ]
    if al == 27:
        parts.append(get_identity_normalizer_from_stat(array_to_stats(data[:, 21:27])))
    lin = LinearNormalizer()
    lin["action"] = concatenate_normalizer(parts)
    return lin


def load_replay(zpath: Path, meta_shape: dict):
    src = zarr.open(str(zpath), "r")
    keys = [str(k) for k in src["data"].keys()]
    data = {}
    for i, key in enumerate(keys, start=1):
        ep = {}
        raw_to_obs(src["data"][key], ep, meta_shape)
        raw_to_action21(src["data"][key], ep, [0])
        data[key] = ep
        if i == 1 or i % 10 == 0 or i == len(keys):
            print(f"  converted {i}/{len(keys)} {key} action {ep['action'].shape}", flush=True)
    meta = {k: np.asarray(src["meta"][k]) for k in src["meta"].keys()}
    n_meta = int(len(meta["episode_robot0_len"]))
    if n_meta != len(keys):
        raise RuntimeError(f"episode_robot0_len {n_meta} != data keys {len(keys)}")
    return {"data": data, "meta": meta}, keys


def make_sampler(replay, meta_shape, episode_mask):
    return SequenceSamplerUmiFT(
        shape_meta=meta_shape,
        replay_buffer=replay,
        sparse_query_frequency_down_sample_steps=QUERY_DS,
        episode_mask=episode_mask,
        action_padding=True,
        obs_to_obs_sample=lambda *a, **k: None,
        action_to_action_sample=lambda *a, **k: None,
        id_list=[0],
        action_gap_threshold_mult=3.0,
        scan_pre_first_fill="zeros",
        max_abs_pos_m=2.0,
    )


def gather_actions(sampler) -> np.ndarray:
    eye = np.eye(4)
    empty = np.zeros((0,))
    chunks = []
    for n, (epi_id, rgb_id) in enumerate(sampler.indices):
        ep = sampler.replay_buffer["data"][f"episode_{int(epi_id)}"]
        action_times = np.squeeze(np.asarray(ep["action_time_stamps"], dtype=np.float64))
        rgb_times = np.squeeze(np.asarray(ep["obs"]["rgb_time_stamps_0"], dtype=np.float64))
        action_idx, _ = sparse_action_indices_from_rgb_timestamps(
            action_times=action_times,
            rgb_times=rgb_times,
            rgb_id=int(rgb_id),
            action_horizon=H,
            rgb_frame_stride=STRIDE,
        )
        act = np.asarray(ep["action"][action_idx], dtype=np.float32)
        if act.shape[0] < H:
            pad = np.repeat(act[-1:], H - act.shape[0], axis=0)
            act = np.concatenate([act, pad], axis=0)
        proc = action21_to_action_sample(act, empty, [0], [eye])["sparse"]
        chunks.append(np.asarray(proc, dtype=np.float64))
        if n and n % 5000 == 0:
            print(f"  gathered {n}/{len(sampler.indices)}", flush=True)
    return np.concatenate(chunks, axis=0)


def mask_fraction(indices, jumps) -> dict:
    n_steps = 0
    n_masked = 0
    for epi_id, rgb_id in indices:
        jump = jumps[f"episode_{int(epi_id)}"]
        ep_key = f"episode_{int(epi_id)}"
        # timestamps live on the r4 replay; caller passes action lookup via jumps only.
        # Recompute from the jump array indexed by the same action indices the dataset uses.
        # The caller stores action_idx lookup in `lookup`.
        action_idx = lookup_action_idx(ep_key, int(rgb_id))
        n_steps += H
        if len(action_idx) == 0:
            continue
        a0 = int(action_idx[0])
        for t in range(H):
            a_t = int(action_idx[-1] if t >= len(action_idx) else action_idx[t])
            if t == 0:
                continue
            lo, hi = a0 + 1, a_t + 1
            if lo < hi and np.any(jump[lo:hi]):
                n_masked += 1
    return {
        "n_chunks": len(indices),
        "n_action_steps": n_steps,
        "n_delta_R_steps_masked": n_masked,
        "frac_delta_R_steps_masked": float(n_masked / max(n_steps, 1)),
        "field_jump_deg": FIELD_JUMP_DEG,
    }


# Filled by main before mask_fraction. Episode key -> (action_times, rgb_times).
_TIME = {}


def lookup_action_idx(ep_key: str, rgb_id: int) -> np.ndarray:
    action_times, rgb_times = _TIME[ep_key]
    action_idx, _ = sparse_action_indices_from_rgb_timestamps(
        action_times=action_times,
        rgb_times=rgb_times,
        rgb_id=int(rgb_id),
        action_horizon=H,
        rgb_frame_stride=STRIDE,
    )
    return np.asarray(action_idx, dtype=np.int64)


def scale_offset(lin: LinearNormalizer):
    p = lin.params_dict["action"]
    s = p["scale"].detach().float().cpu().reshape(-1).numpy()
    o = p["offset"].detach().float().cpu().reshape(-1).numpy()
    return s, o


def main() -> int:
    global _TIME
    report = {}
    replays = {}
    shapes = {
        "r1_world_world": shape_meta(21, False),
        "r2_world_fixb": shape_meta(21, False),
        "r4_world_fixb_tilt": shape_meta(27, True),
    }
    keys_by_run = {}
    for run in RUNS:
        zpath = ROOT / run / "acp_replay_buffer.zarr"
        print(f"[load] {run}", flush=True)
        replay, keys = load_replay(zpath, shapes[run])
        replays[run] = replay
        keys_by_run[run] = keys
    if not (keys_by_run["r1_world_world"] == keys_by_run["r2_world_fixb"] == keys_by_run["r4_world_fixb_tilt"]):
        raise RuntimeError("zarr episode key order differs across r1/r2/r4")
    keys = keys_by_run["r2_world_fixb"]
    lexical = sorted(keys, key=lambda s: (len(s), s))
    report["episode_key_order"] = keys
    report["key_order_is_lexical"] = keys == lexical
    n_ep = len(keys)
    val_mask = get_val_mask(n_episodes=n_ep, val_ratio=VAL_RATIO, seed=SEED)
    train_mask = ~val_mask
    val_ids = [int(k.split("_")[-1]) for k, m in zip(keys, val_mask) if m]
    train_ids = [int(k.split("_")[-1]) for k, m in zip(keys, train_mask) if m]
    report["n_episodes"] = n_ep
    report["val_episode_ids"] = val_ids
    report["train_episode_ids"] = train_ids
    report["val_ratio"] = VAL_RATIO
    report["seed"] = SEED
    print(f"[split] val episodes {val_ids}", flush=True)

    indices = {}
    train_samplers = {}
    for run in RUNS:
        print(f"[index] {run}", flush=True)
        tr = make_sampler(replays[run], shapes[run], train_mask)
        va = make_sampler(replays[run], shapes[run], val_mask)
        train_samplers[run] = tr
        indices[run] = {
            "train": [(int(a), int(b)) for a, b in tr.indices],
            "val": [(int(a), int(b)) for a, b in va.indices],
        }
        report.setdefault("counts", {})[run] = {
            "train": len(tr.indices),
            "val": len(va.indices),
        }
        print(f"  {run} train {len(tr.indices)} val {len(va.indices)}", flush=True)

    same_train = indices["r1_world_world"]["train"] == indices["r2_world_fixb"]["train"] == indices["r4_world_fixb_tilt"]["train"]
    same_val = indices["r1_world_world"]["val"] == indices["r2_world_fixb"]["val"] == indices["r4_world_fixb_tilt"]["val"]
    report["identical_train_chunks"] = bool(same_train)
    report["identical_val_chunks"] = bool(same_val)
    if not (same_train and same_val):
        raise RuntimeError("r1/r2/r4 chunk index sets differ")

    print("[fit] r2 train actions", flush=True)
    r2_actions = gather_actions(train_samplers["r2_world_fixb"])
    print("[fit] r4 train actions (dims 0:21 check)", flush=True)
    r4_actions = gather_actions(train_samplers["r4_world_fixb_tilt"])
    print("[fit] r1 train actions (dims 18:21 check)", flush=True)
    r1_actions = gather_actions(train_samplers["r1_world_world"])

    d21 = float(np.max(np.abs(r2_actions[:, :21] - r4_actions[:, :21])))
    d1821 = float(np.max(np.abs(r1_actions[:, 18:21] - r2_actions[:, 18:21])))
    report["max_abs_r2_vs_r4_dims_0_21"] = d21
    report["max_abs_r1_vs_r2_dims_18_21"] = d1821
    if d21 > 1e-5 or d1821 > 1e-5:
        raise RuntimeError(f"shared dims differ r2/r4 {d21} r1/r2 18:21 {d1821}")

    shared = fit_action_normalizer(r2_actions[:, :21])
    r1_online = fit_action_normalizer(r1_actions[:, :21])
    s_shared, o_shared = scale_offset(shared)
    s_r1, o_r1 = scale_offset(r1_online)
    if not np.allclose(s_shared[3:9], 1) or not np.allclose(s_shared[12:18], 1):
        raise RuntimeError("rot dims are not identity scale")
    if not np.allclose(o_shared[3:9], 0) or not np.allclose(o_shared[12:18], 0):
        raise RuntimeError("rot dims are not identity offset")
    report["shared_scale"] = s_shared.tolist()
    report["shared_offset"] = o_shared.tolist()
    report["r1_online_scale_18_21"] = s_r1[18:21].tolist()
    report["shared_scale_18_21"] = s_shared[18:21].tolist()
    report["r1_pose_scale_0_18_max_abs_diff_vs_shared"] = float(
        np.max(np.abs(s_r1[:18] - s_shared[:18]))
    )
    report["r1_dims_18_21_scale_max_abs_diff_vs_shared"] = float(
        np.max(np.abs(s_r1[18:21] - s_shared[18:21]))
    )
    report["n_train_action_rows"] = int(r2_actions.shape[0])
    report["action_row_note"] = (
        "Each train chunk contributes 24 rows. Short end-of-episode windows "
        "repeat the last action, matching action_padding=True."
    )

    ASSETS.mkdir(parents=True, exist_ok=True)
    tmp = ASSETS / "oct_v35_r2r4_action_normalizer.tmp.pkl"
    with tmp.open("wb") as f:
        pickle.dump(shared, f)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    final = ASSETS / f"oct_v35_r2r4_action_normalizer_{digest[:12]}.pkl"
    tmp.replace(final)
    shared_copy = ROOT / "shared" / final.name
    shared_copy.write_bytes(final.read_bytes())
    report["normalizer_path"] = str(final)
    report["normalizer_sha256"] = digest
    report["normalizer_shared_copy"] = str(shared_copy)

    # Pin smoke: r2 replaces 21, r4 keeps ΔR identity, r1 only 18:21.
    dummy_r4 = fit_action_normalizer(np.concatenate([r2_actions[:, :21], np.zeros((len(r2_actions), 6))], axis=1))
    apply_shared_action_normalizer(dummy_r4, str(final), digest, "r4")
    s4, o4 = scale_offset(dummy_r4)
    if not np.allclose(s4[:21], s_shared) or not np.allclose(s4[21:], 1) or not np.allclose(o4[21:], 0):
        raise RuntimeError("r4 pin failed")
    apply_shared_action_normalizer(r1_online, str(final), digest, "r1")
    s1b, _ = scale_offset(r1_online)
    if float(np.max(np.abs(s1b[18:21] - s_shared[18:21]))) != 0:
        raise RuntimeError("r1 pin of dims 18:21 failed")
    if float(np.max(np.abs(s1b[:18] - s_r1[:18]))) != 0:
        raise RuntimeError("r1 pin changed pose dims 0:18")

    print("[mask] ΔR", flush=True)
    table = build_episode_mask_table(ROOT / "r4_world_fixb_tilt" / "acp_replay_buffer.zarr")
    jumps = {k: v["field_jump"] for k, v in table.items()}
    n_jumps = int(sum(int(np.sum(v)) for v in jumps.values()))
    n_steps = int(sum(len(v) for v in jumps.values()))
    report["n_field_jumps_gt_30deg"] = n_jumps
    report["n_robot_steps"] = n_steps
    report["field_jump_fraction_of_steps"] = float(n_jumps / max(n_steps, 1))
    for key, ep in replays["r4_world_fixb_tilt"]["data"].items():
        _TIME[key] = (
            np.squeeze(np.asarray(ep["action_time_stamps"], dtype=np.float64)),
            np.squeeze(np.asarray(ep["obs"]["rgb_time_stamps_0"], dtype=np.float64)),
        )
    report["delta_R_mask"] = {
        "train": mask_fraction(indices["r4_world_fixb_tilt"]["train"], jumps),
        "val": mask_fraction(indices["r4_world_fixb_tilt"]["val"], jumps),
    }

    out = ROOT / "shared" / "oct_v35_chunk_normalizer.json"
    out.write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        ROOT / "shared" / "oct_v35_chunk_index.npz",
        train=np.asarray(indices["r2_world_fixb"]["train"], dtype=np.int32),
        val=np.asarray(indices["r2_world_fixb"]["val"], dtype=np.int32),
        val_episode_ids=np.asarray(val_ids, dtype=np.int32),
    )
    print(json.dumps({
        "identical_train_chunks": report["identical_train_chunks"],
        "counts": report["counts"],
        "val_episode_ids": val_ids,
        "sha256": digest,
        "path": str(final),
        "r1_pose_scale_diff": report["r1_pose_scale_0_18_max_abs_diff_vs_shared"],
        "r1_18_21_scale_diff": report["r1_dims_18_21_scale_max_abs_diff_vs_shared"],
        "delta_R_mask": report["delta_R_mask"],
        "n_jumps": n_jumps,
    }, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
