#!/usr/bin/env python3
"""Where the causal-foot δn label comes from, and what step 1 would be if it matched the TCP."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial import cKDTree

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [
    str(SCRIPTS / "eval"),
    str(SCRIPTS),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT),
]

import compare_r1rel as fair
from fixb_labels import pose7_to_Rp
from fixb_smooth_field import SmoothFrameField

OUT = Path(__file__).resolve().parent
DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
ZARR = DATA / "r4vt_causal_foot/acp_replay_buffer.zarr"
FIELD = DATA / "shared/fixb_field.npz"
SMOOTH = DATA / "shared/fixb_smooth_frames.npz"
VAL = (9, 12, 28, 37, 41)
UP = np.array([0.0, -1.0, 0.0])
REGIONS = ("crown", "fillet", "wall", "flange")
K_MAX = 3000.0
K_MIN = 200.0
F_LOW = 2.0
F_HIGH = 8.0


def _check() -> None:
    # Step 1 of the fair decoder is exactly δn minus the TCP's component on frame z.
    dn = -0.075
    tcp_along = -0.021
    assert abs((dn - tcp_along) * 1e3 - (-54.0)) < 1e-6


def k_of(force_norm: np.ndarray) -> np.ndarray:
    f = np.asarray(force_norm, dtype=np.float64)
    k = np.empty(f.shape, dtype=np.float64)
    k[f < F_LOW] = K_MAX
    k[f > F_HIGH] = K_MIN
    mid = (f >= F_LOW) & (f <= F_HIGH)
    k[mid] = K_MAX - (K_MAX - K_MIN) * (f[mid] - F_LOW) / (F_HIGH - F_LOW)
    return k


def region_of(normal, foot, apron) -> str:
    n = normal / max(float(np.linalg.norm(normal)), 1e-12)
    tilt = float(np.degrees(np.arccos(np.clip(n @ UP, -1.0, 1.0))))
    if float(apron.query(foot)[0]) < 0.015:
        return "flange"
    if tilt < 25.0:
        return "crown"
    if tilt < 55.0:
        return "fillet"
    return "wall"


def nearest(times, query):
    j = np.searchsorted(times, query)
    j = np.clip(j, 1, len(times) - 1)
    left = np.abs(query - times[j - 1]) <= np.abs(query - times[j])
    return np.where(left, j - 1, j)


def mean(rows, key):
    if not rows:
        return None
    return round(float(np.mean([r[key] for r in rows])), 2)


def std(rows, key):
    if len(rows) < 2:
        return None
    return round(float(np.std([r[key] for r in rows])), 2)


def fair_rgb() -> set[tuple[int, int]]:
    """The same validation queries as the fair rollout. Datasets are freed after."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    cfg_dir = ROOT / "PyriteML/diffusion_policy/config"
    tasks = SCRIPTS / "config"
    chosen = []
    for task, zarr_path in (
        ("umift_oct_fixb_r4vt_causal_foot", ZARR),
        ("umift_oct_fixb_r1rel_odom_h4s", DATA / "r1rel_world/acp_replay_buffer.zarr"),
    ):
        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=str(cfg_dir), version_base=None)
        cfg = compose(
            config_name="train_umift_workspace",
            overrides=[
                f"hydra.searchpath=[file://{tasks}]",
                f"task={task}",
                "task.dataset_path=" + str(zarr_path),
                "task.dataset.dataset_path=" + str(zarr_path),
                "task.dataset.exclusion_stats_path=null",
                "task.dataset.pose_jump_flag_path=null",
            ],
        )
        dataset = fair.hydra_instantiate(OmegaConf.to_container(cfg.task.dataset, resolve=True))
        chosen.append(dataset.get_validation_dataset())
    queries = fair.common_queries(chosen[0], chosen[1], set(VAL))
    del chosen
    return {(int(ep), int(rgb)) for ep, rgb, _c, _r in queries}


def summarize(rows) -> dict:
    out = {}
    for name in REGIONS:
        group = [r for r in rows if r["region"] == name]
        out[name] = {"n": len(group)}
        if not group:
            continue
        out[name].update({
            "stored_dn_mm": mean(group, "stored_dn_mm"),
            "tcp_along_inward_mm": mean(group, "tcp_in_mm"),
            "gap_mm": mean(group, "gap_mm"),
            "vt_minus_tcp_out_mm": mean(group, "vt_tcp_out_mm"),
            "foot_off_mesh_mm": mean(group, "foot_off_mm"),
            "tcp_mesh_out_mm": mean(group, "tcp_mesh_out_mm"),
            "k": mean(group, "k"),
            "F_dot_out_N": mean(group, "fn"),
            "minus_Fn_over_k_mm": mean(group, "fn_over_k_mm"),
            "minus_Fn_over_3000_mm": mean(group, "fn_over_3000_mm"),
            "force_offset_out_mm": mean(group, "recon_out_mm"),
            "step1_stored_mm": mean(group, "err_stored_mm"),
            "step1_tcp_axis_mm": mean(group, "err_tcp_mm"),
            "step1_mesh_axis_mm": mean(group, "err_mesh_mm"),
            "step1_vt_inward_mm": mean(group, "err_vt_in_mm"),
            "step1_Fn_over_k_mm": mean(group, "err_fnk_mm"),
            "step1_Fn_over_3000_mm": mean(group, "err_3000_mm"),
            "gap_std_mm": std(group, "gap_mm"),
        })
    return out


def main() -> None:
    _check()
    field = SmoothFrameField(FIELD, SMOOTH)
    blob = np.load(FIELD)
    apron = cKDTree(field.vertices[np.asarray(blob["is_apron"]).astype(bool)])
    wanted = fair_rgb()
    root = zarr.open(ZARR, "r")["data"]
    rows = []
    formula_err = []
    for episode in VAL:
        group = root[f"episode_{episode}"]
        robot_t = np.asarray(group["robot_time_stamps_0"]).reshape(-1)
        rgb_t = np.asarray(group["rgb_time_stamps_0"]).reshape(-1)
        tcp_R, tcp = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
        _vt_R, vt = pose7_to_Rp(np.asarray(group["ts_pose_virtual_target_0"], dtype=np.float64))
        foot = np.asarray(group["mold_x_s_0"], dtype=np.float64).reshape(-1, 3)
        frame = np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        dn = np.asarray(group["fixb_delta_n_0"], dtype=np.float64).reshape(-1)
        stiff = np.asarray(group["stiffness_0"], dtype=np.float64).reshape(-1)
        wrench_t = np.asarray(group["wrench_time_stamps_left_0"]).reshape(-1)
        force = np.asarray(group["wrench_left_0"], dtype=np.float64)[:, :3]
        for rgb in sorted(rgb for ep, rgb in wanted if ep == episode):
            index = fair.target_index(robot_t, rgb_t, rgb, 1, fair.HORIZON)
            if len(index) < fair.HORIZON:
                continue
            i = int(np.clip(np.searchsorted(robot_t, rgb_t[rgb]), 0, len(tcp) - 1))
            n_in = frame[i, :, 2]
            n_out = -n_in
            hit, vid, w = field.closest(tcp[i].reshape(1, 3))
            raw = np.asarray(w[0] @ field.normals[vid[0]], dtype=np.float64)
            raw /= max(float(np.linalg.norm(raw)), 1e-12)
            foot_hit = np.asarray(field.closest(foot[i].reshape(1, 3))[0], dtype=np.float64).reshape(3)
            mesh = np.asarray(hit, dtype=np.float64).reshape(3)
            j = int(nearest(wrench_t, np.array([robot_t[i]]))[0])
            f_tool = force[j]
            f_norm = float(np.linalg.norm(f_tool))
            k = float(k_of(np.array([f_norm]))[0])
            fn = float(f_tool @ (tcp_R[i].T @ n_out))
            offset = np.zeros(3) if f_norm < F_LOW else tcp_R[i] @ (-f_tool / max(k, 1.0))
            recon_out = float(offset @ n_out)
            vt_out = float((vt[i] - foot[i]) @ n_out)
            tcp_in = float((tcp[i] - foot[i]) @ n_in)
            tcp_out = float((tcp[i] - foot[i]) @ n_out)
            mesh_in = float((tcp[i] - mesh) @ n_in)
            stored = float(dn[i])
            formula_err.append(abs(stored - vt_out) * 1e3)
            # Step-1 |error| under the current decoder: |δn - (tcp-foot)·inward|.
            def err(value, tcp_in=tcp_in):
                return abs(value - tcp_in) * 1e3

            rows.append({
                "region": region_of(raw, mesh, apron),
                "stored_dn_mm": stored * 1e3,
                "tcp_in_mm": tcp_in * 1e3,
                "gap_mm": (stored - tcp_in) * 1e3,
                "vt_tcp_out_mm": float((vt[i] - tcp[i]) @ n_out) * 1e3,
                "foot_off_mm": float(np.linalg.norm(foot[i] - foot_hit)) * 1e3,
                "tcp_mesh_out_mm": float((tcp[i] - mesh) @ raw) * 1e3,
                "k": float(stiff[i]),
                "k_from_force": k,
                "fn": fn,
                "fn_over_k_mm": (-fn / max(float(stiff[i]), 1.0)) * 1e3,
                "fn_over_3000_mm": (-fn / K_MAX) * 1e3,
                "recon_out_mm": recon_out * 1e3,
                "err_stored_mm": err(stored),
                "err_tcp_mm": err(tcp_in),
                "err_mesh_mm": err(mesh_in),
                "err_vt_in_mm": err(-vt_out),
                "err_fnk_mm": err((-fn / max(float(stiff[i]), 1.0))),
                "err_3000_mm": err(-fn / K_MAX),
            })
    report = {
        "formula": (
            "fixb_causal_foot_zarr writes δn = (p_virtual_target - foot) · n_out. "
            "n_out = -R_field[:,2]. The virtual target is the TCP plus the tool-frame "
            "offset from VirtualTargetEstimator: f=-F[:3], k=3000 below 2 N, k=200 above 8 N, "
            "linear between, offset=f/k. It is not foot-to-TCP and not a fixed 3000."
        ),
        "queries": len(rows),
        "stored_matches_vt_dot_out_mm_max": round(float(np.max(formula_err)), 3) if formula_err else None,
        "by_region": summarize(rows),
    }
    (OUT / "dn_label.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
