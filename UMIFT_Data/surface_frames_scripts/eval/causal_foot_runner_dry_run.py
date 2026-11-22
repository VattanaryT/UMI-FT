#!/usr/bin/env python3
"""Offline replay of recorded episodes through the runner's r4 decode.

Loads the causal-foot checkpoint, blends chunks, and calls decode_chunk.
No robot and no plant. Episodes 9, 28, and 41.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [
    str(SCRIPTS),
    str(SCRIPTS / "eval"),
    str(ROOT / "PyriteEnvSuites/scripts"),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT),
]

from causal_foot import CausalFoot
from fixb_smooth_field import SmoothFrameField
from heading_tilt import geodesic_deg
from heading_tilt_vs_causal_foot import CAUSAL_NORM, CAUSAL_ZARR, build_dataset
from mold_aware_runtime import (
    FixBActionEnsembler,
    MoldAwareRuntime,
    RD_SLEW_AFTER_RAMP_DEG,
    VT_RAMP_S,
    _causal_foot_for,
    fixb_decode_dt,
    r4_safety_limits,
)
from run_policy_open_loop_val_export import load_policy_and_cfg, predict_sparse_action

FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
CKPT = Path(
    "/mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_causal_foot/"
    "oct-v35-h4s-r4vt-causal-foot_e200_bs64_valtopk_43390127/checkpoints/latest.ckpt"
)
OUT = SCRIPTS / "eval/causal_foot_runner_dry_run.json"
EPISODES = (9, 28, 41)
RAW_DT_S = 1.0 / 15.0
EXEC_H = 6


def _checks(runtime):
    side = FIELD.with_name("fixb_smooth_frames.npz")
    foot = _causal_foot_for(runtime.field)
    return {
        "sidecar": str(side),
        "sidecar_exists": side.is_file(),
        "field_class": type(runtime.field).__name__,
        "causal_foot": type(foot).__name__ if foot is not None else None,
        "lead_fallback_m": 0.020,
        "used_lead_fallback": foot is None,
        "slew_during_ramp_deg": float(runtime.limits.max_rd_slew_deg),
        "slew_enabled": bool(runtime.limits.enable_rd_slew),
        "ramp_s": float(VT_RAMP_S),
        "slew_after_ramp_deg": float(RD_SLEW_AFTER_RAMP_DEG),
        "dry_run": bool(runtime.dry_run),
    }


def _episode_indices(dataset, episode):
    rows = [(idx, int(rgb)) for idx, (epi, rgb) in enumerate(dataset.sampler.indices) if int(epi) == episode]
    rows.sort(key=lambda item: item[1])
    return rows


def replay_episode(runtime, policy, dataset, episode, down, device):
    from mold_aware_runtime import rot6d_rows_to_R

    runtime.reset()
    ensembler = FixBActionEnsembler(decay_m=0.01)
    chosen = []
    last = None
    for idx, rgb in _episode_indices(dataset, episode):
        if last is None or rgb - last >= EXEC_H * down:
            chosen.append((idx, rgb))
            last = rgb
    xs, Rs, speeds, slews, elapsed = [], [], [], [], []
    hard = []
    n_out_ok = True
    finite = True
    for count, (idx, _rgb) in enumerate(chosen):
        obs, _action = dataset.sampler.sample_sequence(idx)
        pred = np.asarray(predict_sparse_action(policy, obs, device), dtype=np.float64).reshape(-1, 10)
        ensembler.push(count * EXEC_H, pred)
        raw = ensembler.pop(EXEC_H)
        sparse = obs["sparse"]
        p0 = np.asarray(sparse["robot0_eef_pos"][-1], dtype=np.float64).reshape(3)
        r0 = rot6d_rows_to_R(np.asarray(sparse["robot0_eef_rot_axis_angle"][-1], dtype=np.float64))
        dt = fixb_decode_dt(raw.shape[0], down, RAW_DT_S)
        before = float(runtime._vt_elapsed_s)
        decoded = runtime.decode_chunk(raw, p0, r0, dt, apply_safety=True)
        after = float(runtime._vt_elapsed_s)
        n_out = np.asarray(decoded["n_out"])
        inward = np.asarray(decoded["R_field"])[:, :, 2]
        if not np.allclose(n_out, -inward, atol=1e-6):
            n_out_ok = False
        block = [np.asarray(decoded[key], dtype=np.float64) for key in ("x_d", "R_d", "v_par")]
        if not all(np.isfinite(part).all() for part in block):
            finite = False
        xs.append(block[0])
        Rs.append(block[1])
        speeds.append(np.linalg.norm(block[2], axis=1))
        for step in range(raw.shape[0]):
            at = before + (step + 1) * float(dt[step])
            slews.append(3.0 if at < VT_RAMP_S else RD_SLEW_AFTER_RAMP_DEG)
            elapsed.append(at)
        if runtime.last_hard_triggers:
            hard.extend(runtime.last_hard_triggers)
        del after
    pos = np.concatenate(xs)
    rot = np.concatenate(Rs)
    speed = np.concatenate(speeds)
    step_m = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    step_deg = geodesic_deg(rot[:-1], rot[1:])
    cap = np.asarray(slews[1:], dtype=np.float64)
    return {
        "n_replans": len(chosen),
        "n_steps": int(len(pos)),
        "dt_s": RAW_DT_S * down,
        "finite": finite,
        "n_out_is_minus_frame_z": n_out_ok,
        "hard_triggers": sorted(set(hard)),
        "max_speed_mps": float(speed.max()) if len(speed) else None,
        "speed_limit_mps": float(runtime.limits.max_tangential_speed_mps),
        "max_step_mm": float(step_m.max() * 1e3) if len(step_m) else None,
        "step_limit_mm": float(runtime.limits.max_step_m * 1e3),
        "max_rot_step_deg": float(step_deg.max()) if len(step_deg) else None,
        "rot_steps_over_cap": int(np.sum(step_deg > cap + 0.05)),
        "field_class": type(runtime.field).__name__,
    }


def main():
    if not CKPT.is_file():
        raise SystemExit(f"checkpoint missing: {CKPT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = MoldAwareRuntime.from_assets(
        "r4_world_fixb_tilt",
        field_path=FIELD,
        dry_run=False,
        limits=r4_safety_limits(),
    )
    info = _checks(runtime)
    if not isinstance(runtime.field, SmoothFrameField) or not isinstance(_causal_foot_for(runtime.field), CausalFoot):
        raise SystemExit(f"sidecar foot was not attached: {info}")
    policy, cfg = load_policy_and_cfg(str(CKPT), device)
    sha = str(cfg.task.dataset.action_normalizer_sha256)
    dataset = build_dataset(cfg, CAUSAL_ZARR, CAUSAL_NORM, "14308b72", 10).get_validation_dataset()
    down = int(cfg.task.shape_meta.sample.action.sparse.down_sample_steps)
    episodes = {}
    for episode in EPISODES:
        episodes[str(episode)] = replay_episode(runtime, policy, dataset, episode, down, device)
        print(episode, episodes[str(episode)], flush=True)
    report = {
        "ckpt": str(CKPT),
        "normalizer": str(CAUSAL_NORM),
        "normalizer_sha_prefix": sha[:8],
        "setup": info,
        "down_sample_steps": down,
        "episodes": episodes,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
