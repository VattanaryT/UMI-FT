#!/usr/bin/env python3
"""Clearance latch on top of the 1 s |F·n| blend. Read-only. Does not write a zarr."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lift_gate import K, crossings, isolated
from sweep_lift_gate import HELD, _false_flips, _load, _rise_time, _switch_time

LO = 2.0
HI = 5.0
C_ON_MM = (0.5, 1.0, 2.0)
RISE_S = 0.05
OUT = Path(__file__).resolve().parent / "eval/geom_lift"


def _rising(t, clearance, dt=RISE_S):
    j = np.clip(np.searchsorted(t, t - dt), 0, len(t) - 1)
    return clearance > clearance[j]


def latch_clearance(clearance, rising, c_on):
    """Stay on clearance until it falls below c_on/2. Starts off."""
    c_off = 0.5 * c_on
    state = False
    out = np.zeros(len(clearance), dtype=bool)
    for i, (c, up) in enumerate(zip(clearance, rising)):
        if state:
            if c < c_off:
                state = False
        elif c > c_on and up:
            state = True
        out[i] = state
    return out


def _blend(fn, clearance):
    force = -np.asarray(fn, dtype=np.float64) / K
    weight = np.clip((np.abs(fn) - LO) / (HI - LO), 0.0, 1.0)
    label = (1.0 - weight) * clearance + weight * force
    return label


def _label(fn, clearance, latched):
    label = _blend(fn, clearance)
    label = np.array(label, copy=True)
    label[latched] = clearance[latched]
    return label


def _noise(episodes):
    level = []
    delta = []
    per_std = []
    near_level = []
    near_delta = []
    for epi, ep in episodes.items():
        if epi in HELD:
            continue
        t = ep["t"]
        c = ep["clearance"]
        steady = ep["confirmed"].copy()
        for end in isolated(ep["confirmed"], t):
            steady &= (t < t[int(end)] - 1.0) | (t > t[int(end)] + 0.3)
        if int(steady.sum()) < 20:
            continue
        j = np.clip(np.searchsorted(t, t - RISE_S), 0, len(t) - 1)
        d = c - c[j]
        level.append(c[steady])
        delta.append(d[steady])
        per_std.append(float(np.std(c[steady])))
        near = steady & (np.abs(c) < 0.003)
        paired = near & near[j]
        if int(near.sum()) >= 20:
            near_level.append(c[near])
            near_delta.append(d[paired])
    level = np.concatenate(level)
    delta = np.concatenate(delta)
    near_level = np.concatenate(near_level) if near_level else np.zeros(0)
    near_delta = np.concatenate(near_delta) if near_delta else np.zeros(0)
    return {
        "steady_n": int(level.size),
        "clearance_median_mm": round(float(np.median(level)) * 1e3, 3),
        "clearance_std_mm": round(float(np.std(level)) * 1e3, 3),
        "clearance_p10_mm": round(float(np.percentile(level, 10)) * 1e3, 3),
        "clearance_p90_mm": round(float(np.percentile(level, 90)) * 1e3, 3),
        "frac_above_2mm": round(float(np.mean(level > 0.002)), 4),
        "delta_50ms_std_mm": round(float(np.std(delta)) * 1e3, 3),
        "episode_std_median_mm": round(float(np.median(per_std)) * 1e3, 3),
        "near_surface_n": int(near_level.size),
        "near_surface_std_mm": None if near_level.size == 0 else round(float(np.std(near_level)) * 1e3, 3),
        "near_surface_delta_50ms_std_mm": None if near_delta.size == 0 else round(float(np.std(near_delta)) * 1e3, 3),
    }


def _score(episodes, c_on):
    detected = flips = contact_n = called_air = 0
    approach = bumps = 0
    lags = []
    post = []
    for epi, ep in episodes.items():
        if epi in HELD:
            continue
        t = ep["t"]
        c = ep["clearance"]
        fn = ep["fn_1.00"]
        latched = latch_clearance(c, _rising(t, c), c_on)
        contact = ep["confirmed"] & ~latched
        ends = isolated(ep["confirmed"], t)
        for end in ends:
            near = crossings(contact)
            if near.size and float(np.min(np.abs(t[near] - t[int(end)]))) <= 0.5:
                detected += 1
            t_rise = _rise_time(t, c, int(end))
            if t_rise is None:
                continue
            t_sw = _switch_time(t, contact, t_rise)
            if t_sw is not None and abs(t_sw - t_rise) <= 1.5:
                lags.append(t_sw - t_rise)
            sel = (t > t_rise) & (t <= t_rise + 0.5)
            if np.any(sel):
                post.append(_label(fn, c, latched)[sel])
        flips += _false_flips(t, contact, c)
        contact_n += int(ep["confirmed"].sum())
        called_air += int(np.sum(ep["confirmed"] & latched))
        entries = np.flatnonzero(~latched[:-1] & latched[1:]) + 1
        j = np.clip(np.searchsorted(t, t[entries] - 0.2), 0, len(t) - 1)
        descending = c[entries] < c[j]
        approach += int(np.sum(descending))
        for i in entries:
            i0 = int(np.searchsorted(t, t[i] - 0.5))
            base = float(np.min(c[i0:i + 1]))
            if ep["confirmed"][i] and c[i] < base + 0.002:
                bumps += 1
    lags = np.asarray(lags, dtype=np.float64)
    post = np.concatenate(post) * 1e3 if post else np.zeros(0)
    n_conf = sum(int(isolated(ep["confirmed"], ep["t"]).size) for epi, ep in episodes.items() if epi not in HELD)
    return {
        "c_on_mm": c_on * 1e3,
        "c_off_mm": c_on * 0.5e3,
        "confirmed": n_conf,
        "detected": detected,
        "false_flips": flips,
        "midstroke_entries": bumps,
        "descending_entries": approach,
        "lag_n": int(lags.size),
        "lag_median_s": None if lags.size == 0 else round(float(np.median(lags)), 3),
        "lag_p90_s": None if lags.size == 0 else round(float(np.percentile(lags, 90)), 3),
        "frac_contact_called_air": None if contact_n == 0 else round(called_air / contact_n, 4),
        "post_dn_median_mm": None if post.size == 0 else round(float(np.median(post)), 3),
        "post_frac_positive": None if post.size == 0 else round(float(np.mean(post > 0)), 4),
    }


def _plot(episodes, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True)
    for ax, row in zip(axes, rows):
        c_on = row["c_on_mm"] * 1e-3
        for epi in sorted(HELD):
            ep = episodes[epi]
            latched = latch_clearance(ep["clearance"], _rising(ep["t"], ep["clearance"]), c_on)
            label = _label(ep["fn_1.00"], ep["clearance"], latched)
            for end in isolated(ep["confirmed"], ep["t"]):
                t_rise = _rise_time(ep["t"], ep["clearance"], int(end))
                if t_rise is None:
                    continue
                sel = (ep["t"] >= t_rise - 0.5) & (ep["t"] <= t_rise + 0.5)
                ax.plot(ep["t"][sel] - t_rise, ep["clearance"][sel] * 1e3, color="0.75", lw=0.6)
                ax.plot(ep["t"][sel] - t_rise, label[sel] * 1e3, color="C0", lw=0.7, alpha=0.8)
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("mm")
        ax.set_title(
            f"c_on {row['c_on_mm']:g} mm  lag {row['lag_median_s']} s"
            f"  flips {row['false_flips']}  post {row['post_dn_median_mm']} mm"
        )
    axes[1].set_xlabel("seconds after clearance starts rising")
    axes[0].plot([], [], color="0.4", lw=2, label="clearance")
    axes[0].plot([], [], color="C0", lw=2, label="new δn")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "top2_lift.png", dpi=140)
    plt.close(fig)


def _check():
    c = np.array([0.0, 0.0004, 0.0012, 0.0010, 0.0004, 0.0002, 0.003, 0.0002])
    up = np.array([False, True, True, False, False, False, True, False])
    got = latch_clearance(c, up, 0.001)
    expect = np.array([False, False, True, True, False, False, True, False])
    if not np.array_equal(got, expect):
        raise RuntimeError(f"latch {got} != {expect}")


def main():
    _check()
    episodes = _load()
    noise = _noise(episodes)
    print("[geom] steady-contact clearance", json.dumps(noise), flush=True)
    rows = [_score(episodes, mm * 1e-3) for mm in C_ON_MM]
    for row in rows:
        print(
            f"c_on {row['c_on_mm']:g} mm  det {row['detected']}/{row['confirmed']}"
            f"  flips {row['false_flips']}  bumps {row['midstroke_entries']}"
            f"  descend {row['descending_entries']}"
            f"  lag {row['lag_median_s']}/{row['lag_p90_s']}"
            f"  air|contact {row['frac_contact_called_air']}"
            f"  post {row['post_dn_median_mm']} {row['post_frac_positive']}",
            flush=True,
        )
    ordered = sorted(rows, key=lambda r: (
        not (r["post_dn_median_mm"] is not None and r["post_dn_median_mm"] > 0 and r["false_flips"] <= 5),
        -(r["post_dn_median_mm"] or -999),
        r["false_flips"],
        r["lag_median_s"] if r["lag_median_s"] is not None else 99,
    ))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table.json").write_text(json.dumps({"noise": noise, "rows": rows, "top": ordered[:2]}, indent=2) + "\n")
    _plot(episodes, ordered[:2])
    print("[geom] top", json.dumps(ordered[0]), flush=True)


if __name__ == "__main__":
    main()
