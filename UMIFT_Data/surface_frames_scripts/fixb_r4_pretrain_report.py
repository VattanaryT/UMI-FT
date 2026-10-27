#!/usr/bin/env python3
"""r4 pre-train report: twist-about-n histogram + field-jump locations on the mesh.

  python fixb_r4_pretrain_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from fixb_labels import pose7_to_Rp
from mold_aware_runtime import rot6d_rows_to_R
from fixb_r4_chunk_filter import FIELD_JUMP_DEG, OFF_EDGE_M, rot_angle_deg

ROOT = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb")
R4 = ROOT / "r4_world_fixb_tilt" / "acp_replay_buffer.zarr"
OUT = ROOT / "r4_world_fixb_tilt" / "checks"
CONTACT_SD = 0.003


def twist_about_n_deg(dR: np.ndarray) -> np.ndarray:
    """Signed twist (deg) of ΔR about field-n (local +z), in (-180, 180]."""
    rv = Rotation.from_matrix(np.asarray(dR, dtype=np.float64).reshape(-1, 3, 3)).as_rotvec()
    # ΔR lives in the field frame; n = ê_z.
    ang = np.degrees(rv[:, 2])
    return (ang + 180.0) % 360.0 - 180.0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(R4), "r")["data"]
    names = sorted(
        (k for k in root.keys() if k.startswith("episode_")),
        key=lambda s: int(s.split("_")[-1]),
    )

    twists_contact = []
    jump_xyz = []
    jump_n = []

    for name in names:
        ep = root[name]
        R_tool, p = pose7_to_Rp(ep["ts_pose_fb_0"])
        R_f = np.asarray(ep["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        x = np.asarray(ep["mold_x_s_0"], dtype=np.float64).reshape(-1, 3)
        n = R_f[:, :, 2]
        sd = np.sum((p - x) * n, axis=-1)
        contact = np.abs(sd) < CONTACT_SD
        dR = rot6d_rows_to_R(
            np.asarray(ep["fixb_delta_R_rot6d_0"], dtype=np.float64).reshape(-1, 6)
        )
        tw = twist_about_n_deg(dR)
        twists_contact.append(tw[contact])

        if len(R_f) > 1:
            jumps = rot_angle_deg(R_f[1:], R_f[:-1]) > FIELD_JUMP_DEG
            idx = np.flatnonzero(jumps) + 1  # sample after the jump
            if len(idx):
                jump_xyz.append(x[idx])
                jump_n.append(n[idx])

    tw = np.concatenate(twists_contact) if twists_contact else np.array([])
    # Two peaks ~180° apart: cluster by sign of |twist| near 0 vs near ±180,
    # or simply split at |twist| > 90.
    near_0 = np.abs(tw) <= 90.0
    near_180 = ~near_0
    report = {
        "contact_sd_m": CONTACT_SD,
        "n_contact_steps": int(len(tw)),
        "twist_about_n_deg": {
            "median": float(np.median(tw)) if len(tw) else None,
            "mean": float(np.mean(tw)) if len(tw) else None,
            "std": float(np.std(tw)) if len(tw) else None,
            "frac_abs_le_90": float(near_0.mean()) if len(tw) else None,
            "frac_abs_gt_90": float(near_180.mean()) if len(tw) else None,
            "peak_mode": (
                "bimodal_approx_180"
                if len(tw) and near_0.mean() > 0.15 and near_180.mean() > 0.15
                else "unimodal_or_skewed"
            ),
        },
        "field_jumps": {
            "n_jump_samples": int(sum(len(a) for a in jump_xyz)),
            "jump_deg_threshold": FIELD_JUMP_DEG,
        },
    }

    # Histogram
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(tw, bins=72, range=(-180, 180), color="steelblue", edgecolor="none")
    ax.axvline(0, color="k", lw=0.8, alpha=0.5)
    ax.axvline(180, color="k", lw=0.8, alpha=0.3, ls="--")
    ax.axvline(-180, color="k", lw=0.8, alpha=0.3, ls="--")
    ax.set_xlabel("ΔR twist about n (deg)")
    ax.set_ylabel("count (in-contact steps)")
    ax.set_title(
        f"r4 contact twist about n  "
        f"(|φ|≤90: {near_0.mean()*100:.1f}%  |φ|>90: {near_180.mean()*100:.1f}%)"
    )
    fig.tight_layout()
    fig.savefig(OUT / "twist_about_n_contact.png", dpi=130)
    plt.close(fig)

    # Jump locations on mesh (XY top-down in fixture; also XZ)
    if jump_xyz:
        pts = np.concatenate(jump_xyz, axis=0)
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].scatter(pts[:, 0], pts[:, 2], s=4, c="crimson", alpha=0.35, linewidths=0)
        axs[0].set_xlabel("x (m)")
        axs[0].set_ylabel("z (m)")
        axs[0].set_title(f"field jumps >{FIELD_JUMP_DEG:.0f}° (x–z)")
        axs[0].set_aspect("equal", adjustable="datalim")
        axs[1].scatter(pts[:, 0], pts[:, 1], s=4, c="crimson", alpha=0.35, linewidths=0)
        axs[1].set_xlabel("x (m)")
        axs[1].set_ylabel("y (m)")
        axs[1].set_title("field jumps (x–y)")
        axs[1].set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        fig.savefig(OUT / "field_jump_locations.png", dpi=130)
        plt.close(fig)
        # save points for overlay
        np.savez_compressed(OUT / "field_jump_xyz.npz", xyz=pts.astype(np.float32))

    (OUT / "twist_and_jumps_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {OUT}/twist_about_n_contact.png")
    print(f"wrote {OUT}/field_jump_locations.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
