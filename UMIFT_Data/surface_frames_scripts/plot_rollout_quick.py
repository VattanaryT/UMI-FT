#!/usr/bin/env python3
"""One page for a rollout control log: force, commanded δn, v1/v2, and the foot.

Reads the zarr written by umift_env_runner. δn and v1/v2 come from the executed
chunk logged as action/delta_n and action/v_par. Older logs fall back to the
virtual-target offset for δn and leave v blank.

  python plot_rollout_quick.py /path/to/control_log_episode
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)


def horizons(log):
    keys = [k for k in log.keys() if str(k).startswith("horizon_")]
    return sorted(keys, key=lambda k: int(str(k).split("_", 1)[1]))


def _force(group) -> float:
    action = group["action"]
    if "wrench" in action:
        wrench = np.asarray(action["wrench"], dtype=np.float64).reshape(-1)
        if wrench.size >= 3:
            return float(np.linalg.norm(wrench[:3]))
    return np.nan


def series(log) -> dict:
    t, force, dn, v1, v2, foot = [], [], [], [], [], []
    for key in horizons(log):
        group = log[key]
        action = group["action"]
        times = np.asarray(action["timestamps_s"], dtype=np.float64).reshape(-1)
        if times.size == 0:
            continue
        t.append(float(times[0]))
        force.append(_force(group))
        if "delta_n" in action:
            dn.append(float(np.asarray(action["delta_n"]).reshape(-1)[0]) * 1e3)
        else:
            vt = np.asarray(action["ts_virtual_targets_0"][0, :3], dtype=np.float64)
            nom = np.asarray(action["ts_nominal_targets_0"][0, :3], dtype=np.float64)
            dn.append(float(np.linalg.norm(vt - nom)) * 1e3)
        if "v_par" in action:
            vel = np.asarray(action["v_par"], dtype=np.float64).reshape(-1, 2)[0]
            v1.append(float(vel[0]))
            v2.append(float(vel[1]))
        else:
            v1.append(np.nan)
            v2.append(np.nan)
        if "foot" in action:
            foot.append(np.asarray(action["foot"], dtype=np.float64).reshape(-1, 3)[0])
        else:
            foot.append(np.asarray(action["ts_nominal_targets_0"][0, :3], dtype=np.float64))
    t = np.asarray(t, dtype=np.float64)
    if t.size:
        t = t - t[0]
    return {
        "t": t,
        "force": np.asarray(force, dtype=np.float64),
        "dn_mm": np.asarray(dn, dtype=np.float64),
        "v1": np.asarray(v1, dtype=np.float64),
        "v2": np.asarray(v2, dtype=np.float64),
        "foot": np.asarray(foot, dtype=np.float64).reshape(-1, 3),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args(argv)
    log = zarr.open(store=zarr.DirectoryStore(str(args.log)), mode="r")
    data = series(log)
    if data["t"].size == 0:
        raise SystemExit(f"no horizons in {args.log}")

    fig = plt.figure(figsize=(12, 8))
    ax_f = fig.add_subplot(2, 2, 1)
    ax_d = fig.add_subplot(2, 2, 2)
    ax_v = fig.add_subplot(2, 2, 3)
    ax3 = fig.add_subplot(2, 2, 4, projection="3d")
    ax_f.plot(data["t"], data["force"], color="#222222")
    ax_f.set_ylabel("force N")
    ax_d.plot(data["t"], data["dn_mm"], color="#3ddc97")
    ax_d.set_ylabel("commanded δn mm")
    ax_v.plot(data["t"], data["v1"], label="v1", color="#4aa3ff")
    ax_v.plot(data["t"], data["v2"], label="v2", color="#ff4d6a")
    ax_v.legend()
    ax_v.set_ylabel("m/s")
    ax_v.set_xlabel("time s")
    ax_f.set_xlabel("time s")
    ax_d.set_xlabel("time s")
    if FIELD.is_file():
        mesh = np.load(FIELD)["vertices"]
        step = max(1, len(mesh) // 4000)
        pts = mesh[::step]
        ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c="#bbbbbb", depthshade=False)
    foot = data["foot"]
    ax3.plot(foot[:, 0], foot[:, 1], foot[:, 2], color="#ff4d6a", linewidth=2)
    ax3.set_title("foot")
    fig.suptitle(args.log.name)
    fig.tight_layout()
    if args.save is not None:
        fig.savefig(args.save, dpi=120)
        print(args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
