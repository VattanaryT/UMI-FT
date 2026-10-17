#!/usr/bin/env python3
"""Sanity-check the mold-aware surface-frame labels on one training episode.

For each variant zarr (world_world / world_surface / surface_surface):
  * plots the action labels over time (surface deltas for the surface variants),
  * plots the angle between consecutive surface frames [t1, t2, n],
  * compares each delta dimension's std to the pose-tracking noise level,
  * flags spikes, tangent sign flips, and dimensions that are mostly noise,
  * checks that surface_surface observations differ from world_surface's.

Usage:
  python mold_aware_data_check.py \
    --dataset_root /mnt/data/umidata/ft/mold_aware_datasets/stringer_data_v2_full \
    --episode episode_0 \
    --out_dir /mnt/data/umidata/ft/training_outputs/mold_aware/eval/data_check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

VARIANTS = ("world_world", "world_surface", "surface_surface")
AXES = ("t1", "t2", "n")


def frames_from_normals(normals: np.ndarray) -> np.ndarray:
    """Same construction as build_mold_aware_datasets._frames_from_normals."""
    n = np.asarray(normals, dtype=float)
    n = n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-9, None)
    helper = np.zeros_like(n)
    helper[:, 0] = 1.0
    swap = np.abs(n[:, 0]) > 0.9
    helper[swap] = (0.0, 1.0, 0.0)
    t1 = np.cross(helper, n)
    t1 = t1 / np.clip(np.linalg.norm(t1, axis=1, keepdims=True), 1e-9, None)
    t2 = np.cross(n, t1)
    return np.stack([t1, t2, n], axis=-1)


def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(-1, 4)
    return Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


def consecutive_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def rot_step_deg(R: np.ndarray) -> np.ndarray:
    rel = np.einsum("nji,njk->nik", R[:-1], R[1:])
    return np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=1))


def highpass_std(x: np.ndarray, window: int) -> np.ndarray:
    smooth = savgol_filter(x, window_length=window, polyorder=2, axis=0)
    return np.std(x - smooth, axis=0)


def load_episode(root: Path, variant: str, episode: str) -> dict:
    ep = zarr.open(str(root / variant / "acp_replay_buffer.zarr"), "r")["data"][episode]
    return {
        "t": np.asarray(ep["robot_time_stamps_0"], dtype=float).reshape(-1),
        "fb": np.asarray(ep["ts_pose_fb_0"], dtype=float),
        "cmd": np.asarray(ep["ts_pose_command_0"], dtype=float),
        "vt": np.asarray(ep["ts_pose_virtual_target_0"], dtype=float),
        "normal": np.asarray(ep["mold_normal_0"], dtype=float),
        "rot6d": np.asarray(ep["mold_R_field_rot6d_0"], dtype=float),
        "sd": np.asarray(ep["mold_signed_distance_0"], dtype=float).reshape(-1),
        "x_s": np.asarray(ep["mold_x_s_0"], dtype=float),
        "rgb_t": np.asarray(ep["rgb_time_stamps_0"], dtype=float).reshape(-1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_data_v2_full"),
    )
    ap.add_argument("--episode", default="episode_0")
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/mnt/data/umidata/ft/training_outputs/mold_aware/eval/data_check"),
    )
    ap.add_argument("--spike_deg", type=float, default=10.0)
    ap.add_argument("--noise_ratio_flag", type=float, default=0.5)
    ap.add_argument(
        "--highpass_window",
        type=int,
        default=41,
        help="Savitzky-Golay window in robot samples (~0.2 s at 200 Hz).",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.dataset_root / "shared" / "meta.json").read_text())
    scale = np.asarray(meta["surface_delta_scale"], dtype=float)
    data = {v: load_episode(args.dataset_root, v, args.episode) for v in VARIANTS}
    ww, ws, ss = data["world_world"], data["world_surface"], data["surface_surface"]
    t = ww["t"] - ww["t"][0]
    fb_w = ww["fb"]
    R_fb_w = quat_wxyz_to_R(fb_w[:, 3:])
    R_field = frames_from_normals(ww["normal"])
    report: dict = {"episode": args.episode, "surface_delta_scale_m": scale.tolist()}
    flags: list[str] = []

    # --- label reconstruction / equality checks --------------------------------
    same_labels = all(
        np.array_equal(ws[k], ss[k]) for k in ("cmd", "vt")
    )
    report["world_surface_and_surface_surface_action_labels_identical"] = bool(same_labels)
    rot6d_from_normals = R_field[:, :2, :].reshape(-1, 6)
    report["stored_rot6d_vs_normals_max_abs"] = float(
        np.abs(rot6d_from_normals - ww["rot6d"]).max()
    )
    d_vt = ws["vt"][:, :3] * scale  # unnormalized surface delta (m)
    vt_recon = fb_w[:, :3] + np.einsum("nij,nj->ni", R_field, d_vt)
    report["vt_roundtrip_max_err_mm"] = float(
        np.linalg.norm(vt_recon - ww["vt"][:, :3], axis=1).max() * 1e3
    )
    R_vt_local = quat_wxyz_to_R(ws["vt"][:, 3:])
    R_vt_recon = np.einsum("nij,njk->nik", R_field, R_vt_local)
    R_vt_w = quat_wxyz_to_R(ww["vt"][:, 3:])
    rel = np.einsum("nji,njk->nik", R_vt_recon, R_vt_w)
    report["vt_rot_roundtrip_max_err_deg"] = float(
        np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=1)).max()
    )

    cmd_minus_fb_mm = np.linalg.norm(ww["cmd"][:, :3] - fb_w[:, :3], axis=1) * 1e3
    report["world_cmd_minus_fb_max_mm"] = float(cmd_minus_fb_mm.max())
    surf_cmd_xyz_std = ws["cmd"][:, :3].std(axis=0)
    report["surface_cmd_xyz_label_std_normalized"] = surf_cmd_xyz_std.tolist()
    if np.all(surf_cmd_xyz_std < 1e-9):
        flags.append(
            "Surface nominal-command translation labels are identically zero "
            "(source ts_pose_command_0 == ts_pose_fb_0); those 3 action dims carry no "
            "information and the decoded command position is only defined given future fb."
        )

    # --- frame continuity ------------------------------------------------------
    step_deg = {
        ax: consecutive_angle_deg(R_field[:-1, :, i], R_field[1:, :, i])
        for i, ax in enumerate(AXES)
    }
    t1_flip = np.sum(R_field[:-1, :, 0] * R_field[1:, :, 0], axis=-1) < 0
    n_flip = np.sum(R_field[:-1, :, 2] * R_field[1:, :, 2], axis=-1) < 0
    swap = np.abs(ww["normal"][:, 0] / np.linalg.norm(ww["normal"], axis=1)) > 0.9
    helper_switches = int(np.sum(swap[1:] != swap[:-1]))
    fb_rot_step = rot_step_deg(R_fb_w)
    frame_rot_step = rot_step_deg(R_field)
    cont = {}
    for ax in AXES:
        s = step_deg[ax]
        cont[ax] = {
            "p50_deg": float(np.percentile(s, 50)),
            "p99_deg": float(np.percentile(s, 99)),
            "max_deg": float(s.max()),
            f"n_steps_over_{args.spike_deg:g}deg": int(np.sum(s > args.spike_deg)),
            "frac_steps_nonzero": float(np.mean(s > 1e-6)),
        }
    cont["t1_sign_flips"] = int(t1_flip.sum())
    cont["n_sign_flips"] = int(n_flip.sum())
    cont["helper_axis_switches"] = helper_switches
    cont["field_rot_step_p99_deg"] = float(np.percentile(frame_rot_step, 99))
    cont["tool_rot_step_p99_deg"] = float(np.percentile(fb_rot_step, 99))
    report["frame_continuity"] = cont
    n_spikes = int(np.sum(frame_rot_step > args.spike_deg))
    if n_spikes:
        flags.append(
            f"{n_spikes} robot-rate steps where R_field rotates > {args.spike_deg:g} deg "
            f"(max {frame_rot_step.max():.1f} deg) while the tool itself rotates "
            f"p99 {cont['tool_rot_step_p99_deg']:.2f} deg/step."
        )
    if cont["t1_sign_flips"] or cont["n_sign_flips"]:
        flags.append(
            f"Sign flips between consecutive frames: t1={cont['t1_sign_flips']}, "
            f"n={cont['n_sign_flips']} (helper-axis switches={helper_switches})."
        )

    # surface-label jumps caused by frame flicker (vs tool motion)
    dlabel = np.linalg.norm(np.diff(d_vt, axis=0), axis=1) * 1e3
    dvt_world = np.linalg.norm(np.diff(ww["vt"][:, :3] - fb_w[:, :3], axis=0), axis=1) * 1e3
    report["vt_label_step_mm"] = {
        "surface_p99": float(np.percentile(dlabel, 99)),
        "surface_max": float(dlabel.max()),
        "world_p99": float(np.percentile(dvt_world, 99)),
        "world_max": float(dvt_world.max()),
    }

    # --- per-dim signal vs noise ----------------------------------------------
    w = args.highpass_window
    fb_noise_world = fb_w[:, :3] - savgol_filter(fb_w[:, :3], w, 2, axis=0)
    fb_noise_surface = np.einsum("nji,nj->ni", R_field, fb_noise_world)
    tracking_noise_mm = np.std(fb_noise_surface, axis=0) * 1e3
    label_signal_mm = np.std(d_vt, axis=0) * 1e3
    label_hp_noise_mm = highpass_std(d_vt, w) * 1e3
    world_delta = ww["vt"][:, :3] - fb_w[:, :3]
    world_signal_mm = np.std(world_delta, axis=0) * 1e3
    world_hp_noise_mm = highpass_std(world_delta, w) * 1e3

    # Label noise at the policy's action rate (RGB grid ~14 Hz): step-to-step change
    # that is not explained by a smooth trend.
    rgb_idx = np.clip(np.searchsorted(ww["t"], ww["rgb_t"]), 0, len(ww["t"]) - 1)
    d_rgb = d_vt[rgb_idx]
    rgb_hp_mm = highpass_std(d_rgb, 7) * 1e3
    noise_dims = {}
    for i, ax in enumerate(AXES):
        ratio = float(label_hp_noise_mm[i] / max(label_signal_mm[i], 1e-9))
        noise_dims[ax] = {
            "label_std_mm": float(label_signal_mm[i]),
            "label_highpass_std_mm": float(label_hp_noise_mm[i]),
            "label_highpass_std_rgb_rate_mm": float(rgb_hp_mm[i]),
            "fb_tracking_noise_std_mm": float(tracking_noise_mm[i]),
            "noise_to_signal": ratio,
        }
        if ratio > args.noise_ratio_flag:
            flags.append(
                f"Surface VT delta dim {ax}: high-pass noise {label_hp_noise_mm[i]:.2f} mm "
                f"vs label std {label_signal_mm[i]:.2f} mm (ratio {ratio:.2f})."
            )
    report["vt_delta_signal_vs_noise"] = noise_dims
    report["world_vt_delta_signal_vs_noise_xyz"] = {
        "label_std_mm": world_signal_mm.tolist(),
        "label_highpass_std_mm": world_hp_noise_mm.tolist(),
    }
    R_local_vt = quat_wxyz_to_R(ws["vt"][:, 3:])
    report["rotation_label_step_deg_p99"] = {
        "surface_vt_R_local": float(np.percentile(rot_step_deg(R_local_vt), 99)),
        "world_vt_R": float(np.percentile(rot_step_deg(R_vt_w), 99)),
    }

    # --- tool axis that faces the mold ----------------------------------------
    near = np.abs(ww["sd"]) < 0.02
    n_unit = ww["normal"] / np.linalg.norm(ww["normal"], axis=1, keepdims=True)
    axis_align = {
        name: float(np.mean(np.sum(R_fb_w[near, :, i] * n_unit[near], axis=-1)))
        for i, name in enumerate(("x", "y", "z"))
    }
    report["tool_axis_mean_dot_mesh_normal_within_20mm"] = axis_align
    report["frac_samples_within_20mm"] = float(np.mean(near))

    # --- surface_surface vs world_surface observations ------------------------
    fb_ss = ss["fb"]
    diff_mm = np.linalg.norm(fb_ss[:, :3] - ws["fb"][:, :3], axis=1) * 1e3
    corr_n_sd = float(np.corrcoef(fb_ss[:, 2], ww["sd"])[0, 1])
    obs_cmp = {
        "pos_diff_mm_median": float(np.median(diff_mm)),
        "ss_pos_std_mm_t1_t2_n": (fb_ss[:, :3].std(axis=0) * 1e3).tolist(),
        "ws_pos_std_mm_xyz": (ws["fb"][:, :3].std(axis=0) * 1e3).tolist(),
        "ss_pos_abs_max_mm_t1_t2_n": (np.abs(fb_ss[:, :3]).max(axis=0) * 1e3).tolist(),
        "ss_n_component_vs_mold_signed_distance_corr": corr_n_sd,
        "ss_n_component_minus_signed_distance_max_mm": float(
            np.abs(fb_ss[:, 2] - ww["sd"]).max() * 1e3
        ),
    }
    R_ss = quat_wxyz_to_R(fb_ss[:, 3:])
    rel = np.einsum("nji,njk->nik", R_ss, R_fb_w)
    obs_cmp["rot_diff_deg_median"] = float(
        np.median(np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=1)))
    )
    obs_cmp["ss_rot_step_p99_deg"] = float(np.percentile(rot_step_deg(R_ss), 99))
    obs_cmp["ws_rot_step_p99_deg"] = float(np.percentile(fb_rot_step, 99))
    report["surface_surface_vs_world_surface_obs"] = obs_cmp
    if obs_cmp["pos_diff_mm_median"] < 1e-6:
        flags.append("surface_surface fb obs are identical to world_surface's.")
    if max(obs_cmp["ss_pos_std_mm_t1_t2_n"][:2]) < 5.0:
        flags.append(
            "surface_surface obs position has almost no tangential content "
            f"(t1/t2 std {obs_cmp['ss_pos_std_mm_t1_t2_n'][0]:.2f}/"
            f"{obs_cmp['ss_pos_std_mm_t1_t2_n'][1]:.2f} mm); its n component is the "
            f"signed distance (corr {corr_n_sd:.3f}), already in robot0_mold_geom. "
            "The policy sees no global position except through images."
        )
    report["flags"] = flags

    # --- figures ---------------------------------------------------------------
    fig, axs = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    for i, c in enumerate("xyz"):
        axs[0].plot(t, world_delta[:, i] * 1e3, lw=0.7, label=f"vt-fb {c}")
    axs[0].set_ylabel("world_world\nvt - fb [mm]")
    axs[0].set_title(f"{args.episode}: action labels (world_world absolute; vt shown as offset)")
    for i, ax in enumerate(AXES):
        axs[1].plot(t, ws["vt"][:, i], lw=0.7, label=f"vt d_{ax} / scale")
    axs[1].set_ylabel("surface VT delta\n(normalized)")
    for i, ax in enumerate(AXES):
        axs[2].plot(t, ws["cmd"][:, i], lw=0.7, label=f"cmd d_{ax}")
    axs[2].set_ylabel("surface CMD delta\n(normalized)")
    rv = Rotation.from_matrix(R_local_vt).as_rotvec()
    for i, c in enumerate("xyz"):
        axs[3].plot(t, np.degrees(rv[:, i]), lw=0.7, label=f"R_local rotvec {c}")
    axs[3].set_ylabel("surface VT\nR_local [deg]")
    axs[4].plot(t, ww["sd"] * 1e3, lw=0.7, color="k", label="signed distance")
    axs[4].set_ylabel("signed dist [mm]")
    axs[4].set_xlabel("time [s]")
    for a in axs:
        a.legend(loc="upper right", fontsize=7, ncol=3)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "labels_over_time.png", dpi=110)
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax in AXES:
        axs[0].semilogy(t[1:], np.maximum(step_deg[ax], 1e-4), lw=0.6, label=ax)
    axs[0].axhline(args.spike_deg, color="r", ls="--", lw=0.8, label=f"{args.spike_deg:g} deg")
    flip_t = t[1:][t1_flip | n_flip]
    for ft in flip_t:
        axs[0].axvline(ft, color="m", lw=0.4, alpha=0.5)
    axs[0].set_ylabel("angle between consecutive\nframe axes [deg]")
    axs[0].set_title("Surface frame continuity (magenta = sign flip)")
    axs[0].legend(fontsize=8)
    axs[1].semilogy(t[1:], np.maximum(frame_rot_step, 1e-4), lw=0.6, label="R_field step")
    axs[1].semilogy(t[1:], np.maximum(fb_rot_step, 1e-4), lw=0.6, label="tool R_fb step")
    axs[1].set_ylabel("rotation per step [deg]")
    axs[1].set_xlabel("time [s]")
    axs[1].legend(fontsize=8)
    for a in axs:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(args.out_dir / "frame_continuity.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(3)
    wbar = 0.2
    ax.bar(xs - 1.5 * wbar, label_signal_mm, wbar, label="label std (signal)")
    ax.bar(xs - 0.5 * wbar, label_hp_noise_mm, wbar, label="label high-pass std @200 Hz")
    ax.bar(xs + 0.5 * wbar, rgb_hp_mm, wbar, label="label high-pass std @RGB rate")
    ax.bar(xs + 1.5 * wbar, tracking_noise_mm, wbar, label="fb tracking noise")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"d_{a}" for a in AXES])
    ax.set_yscale("log")
    ax.set_ylabel("mm")
    ax.set_title("Surface VT delta: signal vs noise per dimension")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both", axis="y")
    fig.tight_layout()
    fig.savefig(args.out_dir / "delta_signal_vs_noise.png", dpi=110)
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for i, c in enumerate("xyz"):
        axs[0].plot(t, ws["fb"][:, i] * 1e3, lw=0.7, label=f"world_surface fb {c}")
    axs[0].set_ylabel("world_surface obs pos [mm]")
    for i, a in enumerate(AXES):
        axs[1].plot(t, fb_ss[:, i] * 1e3, lw=0.7, label=f"surface_surface fb {a}")
    axs[1].plot(t, ww["sd"] * 1e3, "k--", lw=0.6, label="mold signed distance")
    axs[1].set_ylabel("surface_surface obs pos [mm]")
    axs[1].set_xlabel("time [s]")
    for a in axs:
        a.legend(fontsize=7, ncol=4)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "obs_surface_vs_world.png", dpi=110)
    plt.close(fig)

    (args.out_dir / "data_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
