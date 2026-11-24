#!/usr/bin/env python3
"""field_v2 labels with air δn replaced by TCP clearance. Does not submit training.

Contact (|F·n| > 5 N) keeps field_v2's -(F·n)/3000. Air (|F·n| < 2 N) is the
arrival TCP's clearance along that row's outward frame normal (positive = away),
from the closest point on the mold mesh. The two blend from 2 N to 5 N.
v1, v2, and ΔR are not rewritten.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS), str(ROOT / "PyriteML"), str(ROOT / "PyriteConfig"), str(ROOT / "PyriteUtility"), str(ROOT)]

import zarr  # noqa: E402

from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer  # noqa: E402
from fixb_build_datasets import link_tree, replace_array  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import interp_rows, next_control_times, pose7_to_Rp  # noqa: E402
from fixb_smooth_frame_labels import EXCLUDE, LONG_DT  # noqa: E402
from fixb_tcp_track_clean import _kept  # noqa: E402

DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
SRC = DATA / "r4vt_field_v2_k3000/acp_replay_buffer.zarr"
OUT = DATA / "r4vt_field_v2n"
FIELD = DATA / "shared/fixb_field.npz"
NORM = SCRIPTS / "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_field_v2n.pkl"
PLOT = SCRIPTS / "eval/field_v2n_dn_ep0_12_43.png"
YAML = SCRIPTS / "config/task/umift_oct_fixb_r4vt_field_v2n.yaml"
SBATCH = SCRIPTS / "fixb_train_oct_v35_r4vt_field_v2n_e200_klone.sbatch"
K = 3000.0
LO = 2.0
HI = 5.0
PLOT_EPS = (0, 12, 43)


def blend(force_dn: np.ndarray, clearance: np.ndarray, fn: np.ndarray) -> np.ndarray:
    """1 at contact (|F·n| >= 5 N), 0 in the air (|F·n| <= 2 N)."""
    weight = np.clip((np.abs(fn) - LO) / (HI - LO), 0.0, 1.0)
    return (1.0 - weight) * clearance + weight * force_dn


def target_clearance(position, frame, robot_t, rgb_t, field) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arrival TCP, its mesh foot, and clearance along this row's outward normal."""
    t_next = next_control_times(robot_t, rgb_t)
    arrived = interp_rows(t_next, robot_t, np.asarray(position, dtype=np.float64))
    foot, _vid, _w = field.closest(arrived)
    outward = -np.asarray(frame, dtype=np.float64).reshape(-1, 3, 3)[:, :, 2]
    clearance = np.sum((arrived - foot) * outward, axis=1)
    return arrived, foot, clearance


def _span(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "n": int(values.size),
        "min_mm": round(float(values.min()) * 1e3, 3),
        "median_mm": round(float(np.median(values)) * 1e3, 3),
        "max_mm": round(float(values.max()) * 1e3, 3),
    }


def _fit(rows: np.ndarray) -> str:
    if NORM.exists():
        raise SystemExit(f"refusing to overwrite {NORM}")
    lo = rows.min(0).astype(np.float32)
    hi = rows.max(0).astype(np.float32)
    span = np.maximum(hi - lo, np.float32(1e-7))
    scale = (np.float32(2.0) / span).astype(np.float32)
    offset = (np.float32(-1.0) - scale * lo).astype(np.float32)
    # A constant 3000 overflows the float32 offset. Same pin as the k3000 normalizer.
    scale[3] = np.float32(2.0e7)
    offset[3] = np.float32(-1.0) - scale[3] * np.float32(K)
    stats = {
        "min": lo, "max": hi,
        "mean": rows.mean(0).astype(np.float32),
        "std": rows.std(0).astype(np.float32),
    }
    lin = LinearNormalizer()
    lin["action"] = SingleFieldLinearNormalizer.create_manual(scale, offset, stats)
    value = float(scale[3] * K + offset[3])
    if abs(value) > 1e-4 or not np.all(rows[:, 3] == K):
        raise SystemExit(f"stiffness did not pin to 3000 ({value})")
    NORM.parent.mkdir(parents=True, exist_ok=True)
    with NORM.open("wb") as handle:
        pickle.dump(lin, handle)
    digest = hashlib.sha256(NORM.read_bytes()).hexdigest()
    for path in (YAML, SBATCH):
        text = path.read_text()
        if "SHA_HERE" not in text:
            raise SystemExit(f"{path} has no SHA_HERE")
        path.write_text(text.replace("SHA_HERE", digest))
    return digest


def _plot(series: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(series), 1, figsize=(8.2, 7.2), sharey=True)
    for ax, (ep_id, t, dn, fn) in zip(axes, series):
        air = np.abs(fn) <= LO
        mm = dn * 1e3
        ax.plot(t, mm, color="0.75", lw=0.5, label="contact and blend")
        ax.plot(np.where(air, t, np.nan), np.where(air, mm, np.nan), color="C0", lw=0.8, label="air")
        if ep_id == 43:
            ax.axvline(59.8, color="C3", lw=0.8, label="59.8 s")
        ax.set_ylabel("δn mm")
        ax.set_title(f"episode {ep_id}")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time s (robot clock, rgb start = 0)")
    fig.tight_layout()
    fig.savefig(PLOT, dpi=120)
    plt.close(fig)
    print(f"[field_v2n] {PLOT}", flush=True)


def _commands() -> None:
    tar = "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_field_v2n.tar"
    remote = "/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v3.5_fixb_odom_corr"
    code = "/gscratch/scrubbed/vtevy/UMI-FT-causal_foot/UMIFT_Data/surface_frames_scripts"
    host = "vtevy@klone.hyak.uw.edu"
    print("\n".join([
        f"tar -C /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr -cf {tar} r4vt_field_v2n",
        f"rsync -P {tar} {host}:{remote}/",
        f"ssh {host} 'tar -xf {remote}/r4vt_field_v2n.tar -C {remote} && rm {remote}/r4vt_field_v2n.tar'",
        f"rm {tar}",
        f"rsync -P {SCRIPTS}/field_v2n_dataset.py {host}:{code}/",
        f"rsync -P {NORM} {host}:{code}/assets/",
        f"rsync -P {YAML} {host}:{code}/config/task/",
        f"rsync -P {SBATCH} {host}:{code}/",
        f"ssh {host} 'sbatch {code}/fixb_train_oct_v35_r4vt_field_v2n_e200_klone.sbatch'",
    ]), flush=True)


def _self_check() -> None:
    force = np.array([-30.0 / K, -3.0 / K, -1.0 / K])
    fn = -force * K
    clearance = np.full(3, 0.02)
    got = blend(force, clearance, fn)
    if got[0] != force[0] or got[2] != clearance[2]:
        raise RuntimeError("blend endpoints moved")
    weight = (3.0 - LO) / (HI - LO)
    mid = (1.0 - weight) * 0.02 + weight * force[1]
    if abs(got[1] - mid) > 1e-12:
        raise RuntimeError("blend midpoint is not linear")


def build() -> None:
    out_zarr = OUT / "acp_replay_buffer.zarr"
    if out_zarr.exists() or NORM.exists() or PLOT.exists():
        raise SystemExit(f"refusing to overwrite {out_zarr} or {NORM} or {PLOT}")
    field = FixBMoldField(str(FIELD))
    long_blob = np.load(LONG_DT)
    long_idx: dict[int, set[int]] = {}
    for ep, idx in zip(long_blob["flagged_episode"], long_blob["flagged_index"]):
        long_idx.setdefault(int(ep), set()).add(int(idx))
    OUT.mkdir(parents=True, exist_ok=True)
    link_tree(SRC, out_zarr)
    root = zarr.open(str(out_zarr), "a")
    root.attrs["label_version"] = "field_v2n_v1"
    root.attrs["delta_n"] = (
        "air |F·n|<=2: arrival TCP clearance along the row outward normal; "
        "contact |F·n|>=5: -(F·n)/3000; linear blend between"
    )
    air_dn, contact_dn = [], []
    series = []
    rows = []
    for ep_id in range(50):
        ep = root["data"][f"episode_{ep_id}"]
        robot_t = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        rgb_t = np.asarray(ep["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
        _rotation, position = pose7_to_Rp(np.asarray(ep["ts_pose_fb_0"], dtype=np.float64))
        frame = np.asarray(ep["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        stored = np.asarray(ep["fixb_delta_n_0"], dtype=np.float32).reshape(-1)
        fn = -stored.astype(np.float64) * K
        _arrived, _foot, clearance = target_clearance(position, frame, robot_t, rgb_t, field)
        depth = blend(stored.astype(np.float64), clearance, fn).astype(np.float32)
        contact = np.abs(fn) >= HI
        if not np.array_equal(depth[contact], stored[contact]):
            raise SystemExit(f"episode {ep_id} contact δn changed")
        like = ep["fixb_delta_n_0"]
        replace_array(ep, "fixb_delta_n_0", depth.reshape(-1, 1), like)
        command = np.array(np.asarray(ep["ts_pose_command_0"]), copy=True)
        command[:, 2] = depth
        replace_array(ep, "ts_pose_command_0", command, ep["ts_pose_command_0"])
        air = np.abs(fn) <= LO
        if np.any(air):
            air_dn.append(depth[air])
        if np.any(contact):
            contact_dn.append(depth[contact])
        if ep_id not in EXCLUDE:
            use = _kept(ep, ep_id, long_idx, np.zeros(len(position), dtype=bool))
            if np.any(use):
                velocity = np.asarray(ep["fixb_v_par_0"], dtype=np.float64)
                delta_r = np.asarray(ep["fixb_delta_R_rot6d_0"], dtype=np.float64)
                stiff = np.asarray(ep["stiffness_0"], dtype=np.float64).reshape(-1)
                rows.append(np.column_stack([
                    velocity[use, 0], velocity[use, 1], depth[use], stiff[use], delta_r[use],
                ]))
        if ep_id in PLOT_EPS:
            series.append((ep_id, robot_t - rgb_t[0], depth.astype(np.float64), fn))
        print(f"[field_v2n] ep {ep_id:02d} air {int(air.sum())} contact {int(contact.sum())}", flush=True)
    report = {"air": _span(np.concatenate(air_dn)), "contact": _span(np.concatenate(contact_dn))}
    print("[field_v2n] δn mm", json.dumps(report), flush=True)
    _plot(series)
    got = np.concatenate(rows).astype(np.float32)
    sha = _fit(got)
    print(f"[field_v2n] normalizer {NORM} {sha} rows {len(got)}", flush=True)
    _commands()


def main() -> None:
    _self_check()
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        print("field_v2n check ok")
        return
    build()


if __name__ == "__main__":
    main()
