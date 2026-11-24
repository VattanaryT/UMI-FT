#!/usr/bin/env python3
"""Sweep the air/contact gate. Does not write a zarr or a normalizer.

A confirmed lift is the 1 s |F·n| >= 5 N release that stays open for 0.5 s,
on every episode except the five held out of training. Clearance rise is the
first time clearance is 2 mm above its minimum in the second before that release.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V2N = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-field_v2n/UMIFT_Data/surface_frames_scripts")
sys.path[:0] = [str(ROOT / "UMIFT_Data/surface_frames_scripts"), str(V2N), str(ROOT / "PyriteUtility"), str(ROOT)]

import zarr  # noqa: E402
from field_v2n import target_clearance  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import pose7_to_Rp  # noqa: E402
from lift_gate import crossings, delta_n, hysteresis, isolated  # noqa: E402
from PyriteUtility.data_pipeline.wrench_baseline_utils import time_based_moving_average  # noqa: E402

ZARR = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_field_v2n/acp_replay_buffer.zarr"
)
FIELD = ZARR.parents[1] / "shared/fixb_field.npz"
HELD = {9, 12, 28, 37, 41}
WINDOWS = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
GATES = ((5.0, 5.0), (5.0, 4.0), (5.0, 3.0), (6.0, 3.0), (4.0, 2.0))
OUT = Path(__file__).resolve().parent / "eval/lift_sweep"
CACHE = OUT / "cache"


def _force(group, robot_t, rotation, frame, window):
    wrench_t = np.asarray(group["wrench_time_stamps_left_0"], dtype=np.float64).reshape(-1)
    wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
    smooth = wrench if window <= 0 else time_based_moving_average(wrench, wrench_t, window)
    j = np.clip(np.searchsorted(wrench_t, robot_t), 1, len(wrench_t) - 1)
    left = np.abs(robot_t - wrench_t[j - 1]) <= np.abs(robot_t - wrench_t[j])
    force = smooth[np.where(left, j - 1, j), :3]
    n = min(len(robot_t), len(rotation), len(frame), len(force))
    outward = -frame[:n, :, 2]
    world = np.einsum("nij,nj->ni", rotation[:n], force[:n])
    return np.einsum("ni,ni->n", world, outward)


def _load():
    if (CACHE / "done").exists():
        episodes = {}
        for path in sorted(CACHE.glob("ep_*.npz")):
            epi = int(path.stem.split("_")[1])
            blob = np.load(path)
            episodes[epi] = {k: blob[k] for k in blob.files}
            episodes[epi]["confirmed"] = blob["confirmed"].astype(bool)
        return episodes
    CACHE.mkdir(parents=True, exist_ok=True)
    field = FixBMoldField(str(FIELD))
    root = zarr.open_group(str(ZARR), mode="r")["data"]
    episodes = {}
    for epi in range(50):
        group = root[f"episode_{epi}"]
        robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
        rotation, position = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
        frame = np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        n = min(len(robot_t), len(rotation), len(frame))
        robot_t, rotation, position, frame = robot_t[:n], rotation[:n], position[:n], frame[:n]
        fn = {f"{w:.2f}": _force(group, robot_t, rotation, frame, w) for w in WINDOWS}
        _arrived, _foot, clearance = target_clearance(position, frame, robot_t, rgb_t, field)
        confirmed = np.abs(fn["1.00"]) >= 5.0
        payload = {"t": robot_t, "clearance": clearance, "confirmed": confirmed, **{f"fn_{k}": v for k, v in fn.items()}}
        np.savez(CACHE / f"ep_{epi:02d}.npz", **payload)
        episodes[epi] = payload
        print(f"[sweep] cached {epi:02d}", flush=True)
    (CACHE / "done").write_text("ok\n")
    return episodes


def _rise_time(t, clearance, end):
    """First time clearance is 2 mm above its minimum in the previous second."""
    t0 = t[end]
    i0 = int(np.searchsorted(t, t0 - 1.0))
    pre = clearance[i0:end + 1]
    if pre.size == 0:
        return None
    base = float(pre.min())
    imin = int(pre.argmin())
    i1 = int(np.searchsorted(t, t0 + 0.3))
    after = clearance[i0 + imin:i1]
    hit = np.flatnonzero(after >= base + 0.002)
    if hit.size == 0:
        return None
    return float(t[i0 + imin + int(hit[0])])


def _switch_time(t, contact, t_rise):
    i = int(np.clip(np.searchsorted(t, t_rise), 0, len(t) - 1))
    if not contact[i]:
        prev = np.flatnonzero(contact[:i + 1])
        if prev.size == 0:
            return None
        end = int(prev[-1])
        if end + 1 < len(contact) and not contact[end + 1]:
            return float(t[end])
        return None
    later = np.flatnonzero(~contact[i:])
    if later.size == 0:
        return None
    return float(t[i + int(later[0]) - 1]) if i + int(later[0]) else float(t[i])


def _false_flips(t, contact, clearance):
    """Air gaps of <= 0.2 s while clearance is still within 2 mm of the press."""
    n = 0
    for end in crossings(contact):
        later = np.flatnonzero(contact[end + 1:])
        if later.size == 0:
            continue
        reopen = end + 1 + int(later[0])
        if t[reopen] - t[end] > 0.2:
            continue
        i0 = int(np.searchsorted(t, t[end] - 0.5))
        base = float(np.min(clearance[i0:end + 1]))
        gap = clearance[end + 1:reopen]
        if gap.size and float(np.max(gap)) < base + 0.002:
            n += 1
    return n


def _score(episodes, window, on_n, off_n):
    key = f"fn_{window:.2f}"
    detected = flips = contact_n = called_air = 0
    lags = []
    post = []
    own = 0
    for epi, ep in episodes.items():
        if epi in HELD:
            continue
        contact = hysteresis(np.abs(ep[key]), on_n, off_n)
        own += int(isolated(contact, ep["t"]).size)
        confirmed = ep["confirmed"]
        ends = isolated(confirmed, ep["t"])
        for end in ends:
            t_sw = None
            near = crossings(contact)
            if near.size:
                dt = np.abs(ep["t"][near] - ep["t"][end])
                if float(dt.min()) <= 0.5:
                    detected += 1
            t_rise = _rise_time(ep["t"], ep["clearance"], int(end))
            if t_rise is not None:
                t_sw = _switch_time(ep["t"], contact, t_rise)
                if t_sw is not None and abs(t_sw - t_rise) <= 1.5:
                    lags.append(t_sw - t_rise)
                sel = (ep["t"] > t_rise) & (ep["t"] <= t_rise + 0.5)
                if np.any(sel):
                    post.append(delta_n(ep[key], ep["clearance"], contact)[sel])
        flips += _false_flips(ep["t"], contact, ep["clearance"])
        both = confirmed
        contact_n += int(both.sum())
        called_air += int(np.sum(both & ~contact))
    lags = np.asarray(lags, dtype=np.float64)
    post = np.concatenate(post) * 1e3 if post else np.zeros(0)
    n_conf = sum(int(isolated(ep["confirmed"], ep["t"]).size) for epi, ep in episodes.items() if epi not in HELD)
    return {
        "window_s": window,
        "on_n": on_n,
        "off_n": off_n,
        "confirmed": n_conf,
        "detected": detected,
        "own_isolated": own,
        "false_flips": flips,
        "lag_n": int(lags.size),
        "lag_median_s": None if lags.size == 0 else round(float(np.median(lags)), 3),
        "lag_p90_s": None if lags.size == 0 else round(float(np.percentile(lags, 90)), 3),
        "frac_contact_called_air": None if contact_n == 0 else round(called_air / contact_n, 4),
        "post_dn_median_mm": None if post.size == 0 else round(float(np.median(post)), 3),
        "post_frac_positive": None if post.size == 0 else round(float(np.mean(post > 0)), 4),
    }


def _pick(rows):
    """Quiet cluster: false flips <= 30. Nothing in the grid is zero.

    Lowest lag in that cluster, then the fewest-flip cell.
    """
    quiet = [
        r for r in rows
        if r["false_flips"] <= 30 and r["detected"] >= 0.9 * r["confirmed"] and r["lag_median_s"] is not None
    ]
    winner = min(quiet, key=lambda r: (r["lag_median_s"], r["false_flips"], r["lag_p90_s"]))
    quietest = min(rows, key=lambda r: (r["false_flips"], r["lag_median_s"]))
    return winner, quietest


def _plot(episodes, first, second):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True)
    for ax, spec in zip(axes, (first, second)):
        key = f"fn_{spec['window_s']:.2f}"
        curves_c, curves_l = [], []
        for epi in sorted(HELD):
            ep = episodes[epi]
            contact = hysteresis(np.abs(ep[key]), spec["on_n"], spec["off_n"])
            label = delta_n(ep[key], ep["clearance"], contact)
            for end in isolated(ep["confirmed"], ep["t"]):
                t_rise = _rise_time(ep["t"], ep["clearance"], int(end))
                if t_rise is None:
                    continue
                sel = (ep["t"] >= t_rise - 0.5) & (ep["t"] <= t_rise + 0.5)
                curves_c.append((ep["t"][sel] - t_rise, ep["clearance"][sel] * 1e3))
                curves_l.append((ep["t"][sel] - t_rise, label[sel] * 1e3))
        for t, y in curves_c:
            ax.plot(t, y, color="0.75", lw=0.6)
        for t, y in curves_l:
            ax.plot(t, y, color="C0", lw=0.7, alpha=0.8)
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("mm")
        ax.set_title(
            f"{spec['window_s']:g} s, on {spec['on_n']:g} / off {spec['off_n']:g}"
            f"  lag {spec['lag_median_s']} s  flips {spec['false_flips']}"
        )
    axes[1].set_xlabel("seconds after clearance starts rising")
    axes[0].plot([], [], color="0.4", lw=2, label="clearance")
    axes[0].plot([], [], color="C0", lw=2, label="new δn")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "top2_lift.png", dpi=140)
    plt.close(fig)


def main() -> None:
    episodes = _load()
    n_conf = sum(int(isolated(ep["confirmed"], ep["t"]).size) for epi, ep in episodes.items() if epi not in HELD)
    print(f"[sweep] confirmed lifts excluding held-out = {n_conf}", flush=True)
    rows = []
    for window in WINDOWS:
        for on_n, off_n in GATES:
            row = _score(episodes, window, on_n, off_n)
            rows.append(row)
            print(
                f"{window:4.2f}s {on_n:.0f}/{off_n:.0f}  "
                f"det {row['detected']}/{row['confirmed']} own {row['own_isolated']} "
                f"flips {row['false_flips']} lag {row['lag_median_s']}/{row['lag_p90_s']} "
                f"air|contact {row['frac_contact_called_air']} "
                f"post {row['post_dn_median_mm']} {row['post_frac_positive']}",
                flush=True,
            )
    first, second = _pick(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table.json").write_text(json.dumps({"rows": rows, "pick": first, "second": second}, indent=2) + "\n")
    _plot(episodes, first, second)
    print("[sweep] pick", json.dumps(first), flush=True)
    print("[sweep] second", json.dumps(second), flush=True)


if __name__ == "__main__":
    main()
