#!/usr/bin/env python3
"""Subtract a force-anchored clearance offset. Read-only. Does not write a zarr."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from geom_lift_gate import _label, _rising, _score, latch_clearance
from lift_gate import isolated
from sweep_lift_gate import HELD, _load, _rise_time

FIRM_N = 8.0
FIRM_S = 0.1
HOLD_S = 1.0
C_ON_MM = (1.0, 2.0)
OUT = Path(__file__).resolve().parent / "eval/fuse_clearance"
ZARR = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_field_v2n/acp_replay_buffer.zarr"
)
FIELD = ZARR.parents[1] / "shared/fixb_field.npz"
LARGE_STD_MM = 2.0


def _centerline(field, origin, axes):
    v = field["vertices"]
    body = ~field["is_apron"].astype(bool)
    local = (v[body] - origin) @ axes
    nm = field["normals"][body] @ axes
    wall = np.abs(nm[:, 2]) >= np.abs(nm[:, 1])
    pts = local[wall] if wall.sum() > 100 else local
    lo, hi = np.quantile(pts[:, 0], [0.01, 0.99])
    edges = np.linspace(lo, hi, 61)
    cen = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (pts[:, 0] >= a) & (pts[:, 0] < b)
        if m.sum() < 20:
            continue
        cen.append(np.median(pts[m], axis=0))
    cen = np.asarray(cen)
    step = np.linalg.norm(np.diff(cen, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(step)])
    return cen[:, 0], s


def _firm_runs(t, fn):
    above = np.abs(fn) > FIRM_N
    cuts = np.flatnonzero(above[1:] != above[:-1]) + 1
    runs = np.split(np.arange(len(above)), cuts)
    kept = []
    for idx in runs:
        if not above[idx[0]]:
            continue
        if float(t[idx[-1]] - t[idx[0]]) >= FIRM_S:
            kept.append((int(idx[0]), int(idx[-1])))
    return kept


def _offset_a(t, clearance, runs):
    out = np.array(clearance, copy=True)
    for i, (a, b) in enumerate(runs):
        off = float(np.median(clearance[a:b + 1]))
        stop = float(t[b] + HOLD_S)
        end = int(np.searchsorted(t, stop, side="right"))
        if i + 1 < len(runs):
            end = min(end, runs[i + 1][0])
        out[a:end] -= off
    return out


def _offset_b(s, y, clearance, firm):
    out = np.array(clearance, copy=True)
    idx = np.flatnonzero(firm)
    if idx.size < 40 or float(s[idx].max() - s[idx].min()) < 0.05:
        if idx.size:
            out -= float(np.median(clearance[idx]))
        return out
    sc = s[idx] - float(s[idx].mean())
    yc = y[idx] - float(y[idx].mean())
    design = np.column_stack([np.ones(idx.size), sc, yc, sc ** 2])
    coef, *_ = np.linalg.lstsq(design, clearance[idx], rcond=None)
    s0 = float(s[idx].mean())
    y0 = float(y[idx].mean())
    sc = s - s0
    yc = y - y0
    out -= coef[0] + coef[1] * sc + coef[2] * yc + coef[3] * sc ** 2
    return out


def _kalman(t, clearance, firm, q_per_s, r):
    idx = np.flatnonzero(firm)
    if idx.size == 0:
        return np.array(clearance, copy=True)
    x = float(clearance[idx[0]])
    p = r
    offset = np.empty(len(t))
    offset[: idx[0] + 1] = x
    for i in range(idx[0] + 1, len(t)):
        dt = max(float(t[i] - t[i - 1]), 1e-3)
        p = p + q_per_s * dt
        if firm[i]:
            k = p / (p + r)
            x = x + k * (float(clearance[i]) - x)
            p = (1.0 - k) * p
        offset[i] = x
    return clearance - offset


def _firm_mask(n, runs):
    mask = np.zeros(n, dtype=bool)
    for a, b in runs:
        mask[a:b + 1] = True
    return mask


def _residual_stats(residual, firm):
    v = residual[firm] * 1e3
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "median_mm": round(float(np.median(v)), 3),
        "p10_mm": round(float(np.percentile(v, 10)), 3),
        "p90_mm": round(float(np.percentile(v, 90)), 3),
        "std_mm": round(float(np.std(v)), 3),
    }


def _slopes(t, s, residual, runs):
    time_mm_s = []
    pos_mm_m = []
    for a, b in runs:
        if b - a < 15 or float(t[b] - t[a]) < 0.3:
            continue
        tt = t[a:b + 1] - t[a]
        time_mm_s.append(float(np.polyfit(tt, residual[a:b + 1], 1)[0]) * 1e3)
        ss = s[a:b + 1]
        if float(ss.max() - ss.min()) > 0.03:
            pos_mm_m.append(float(np.polyfit(ss, residual[a:b + 1], 1)[0]) * 1e3)
    def pack(vals):
        if not vals:
            return None
        arr = np.asarray(vals)
        return {
            "n": int(arr.size),
            "median": round(float(np.median(arr)), 3),
            "p90_abs": round(float(np.percentile(np.abs(arr), 90)), 3),
        }
    return {"vs_time_mm_per_s": pack(time_mm_s), "vs_s_mm_per_m": pack(pos_mm_m)}


def _noise_std(t, corrected, firm):
    j = np.clip(np.searchsorted(t, t - 0.05), 0, len(t) - 1)
    paired = firm & firm[j]
    if int(paired.sum()) < 20:
        return None
    return round(float(np.std(corrected[paired] - corrected[j[paired]])) * 1e3, 3)


def _surface():
    import zarr
    field = np.load(FIELD)
    root = zarr.open_group(str(ZARR), mode="r")
    spec = root.attrs["field_v2_axes"]
    origin = np.asarray(spec["origin"], dtype=np.float64)
    axes = np.asarray(spec["axes"], dtype=np.float64)
    cx, cs = _centerline(field, origin, axes)
    return root, origin, axes, cx, cs


def _attach_surface(episodes):
    root, origin, axes, cx, cs = _surface()
    for epi, ep in episodes.items():
        group = root["data"][f"episode_{epi}"]
        robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        n = len(ep["t"])
        if n > len(robot_t) or float(np.max(np.abs(robot_t[:n] - ep["t"]))) > 1e-3:
            raise SystemExit(f"episode {epi} cache time does not match the zarr")
        foot = (np.asarray(group["mold_x_s_0"], dtype=np.float64).reshape(-1, 3)[:n] - origin) @ axes
        ep["s"] = np.interp(foot[:, 0], cx, cs)
        ep["y"] = foot[:, 1]
        ep["runs"] = _firm_runs(ep["t"], ep["fn_0.00"])
        ep["firm"] = _firm_mask(n, ep["runs"])


def _pack(vals):
    if not vals:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "median": round(float(np.median(arr)), 3),
        "p90_abs": round(float(np.percentile(np.abs(arr), 90)), 3),
    }


def _method_row(usable, key):
    pooled = np.concatenate([ep[key][ep["firm"]] for ep in usable.values()])
    time_slopes, pos_slopes, noises = [], [], []
    for ep in usable.values():
        noises.append(_noise_std(ep["t"], ep[key], ep["firm"]))
        for a, b in ep["runs"]:
            if b - a < 15 or float(ep["t"][b] - ep["t"][a]) < 0.3:
                continue
            tt = ep["t"][a:b + 1] - ep["t"][a]
            time_slopes.append(float(np.polyfit(tt, ep[key][a:b + 1], 1)[0]) * 1e3)
            ss = ep["s"][a:b + 1]
            if float(ss.max() - ss.min()) > 0.03:
                pos_slopes.append(float(np.polyfit(ss, ep[key][a:b + 1], 1)[0]) * 1e3)
    noise_vals = [v for v in noises if v is not None]
    return {
        "residual": _residual_stats(pooled, np.ones(pooled.size, dtype=bool)),
        "noise_50ms_std_mm": None if not noise_vals else round(float(np.median(noise_vals)), 3),
        "drift": {"vs_time_mm_per_s": _pack(time_slopes), "vs_s_mm_per_m": _pack(pos_slopes)},
    }


def _q_from_slopes(drift):
    pack = drift["vs_time_mm_per_s"]
    slope = 0.5 if pack is None else max(pack["p90_abs"], 0.5)
    return (slope * 1e-3) ** 2


def _plot(episodes, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 1, figsize=(8.2, 3.1 * len(rows)), sharex=True)
    if len(rows) == 1:
        axes = [axes]
    for ax, row in zip(axes, rows):
        c_on = row["c_on_mm"] * 1e-3
        key = row["clearance_key"]
        for epi in sorted(HELD):
            ep = episodes[epi]
            c = ep[key]
            latched = latch_clearance(c, _rising(ep["t"], c), c_on)
            label = _label(ep["fn_1.00"], c, latched)
            for end in isolated(ep["confirmed"], ep["t"]):
                t_rise = _rise_time(ep["t"], c, int(end))
                if t_rise is None:
                    continue
                sel = (ep["t"] >= t_rise - 0.5) & (ep["t"] <= t_rise + 0.5)
                ax.plot(ep["t"][sel] - t_rise, c[sel] * 1e3, color="0.75", lw=0.6)
                ax.plot(ep["t"][sel] - t_rise, label[sel] * 1e3, color="C0", lw=0.7, alpha=0.8)
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("mm")
        ax.set_title(
            f"{row['method']}  c_on {row['c_on_mm']:g} mm  "
            f"lag {row['lag_median_s']} s  flips {row['false_flips']}  "
            f"post {row['post_dn_median_mm']} mm"
        )
    axes[-1].set_xlabel("seconds after clearance starts rising")
    axes[0].plot([], [], color="0.4", lw=2, label="corrected clearance")
    axes[0].plot([], [], color="C0", lw=2, label="δn")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "top2_lift.png", dpi=140)
    plt.close(fig)


def _check():
    t = np.linspace(0, 2.4, 481)
    c = np.full_like(t, 0.005)
    fn = np.zeros_like(t)
    fn[(t >= 0.2) & (t <= 1.0)] = 10.0
    runs = _firm_runs(t, fn)
    got = _offset_a(t, c, runs)
    if abs(float(np.median(got[(t >= 0.2) & (t <= 1.0)]))) > 1e-9:
        raise RuntimeError("stroke offset did not zero firm contact")
    if abs(float(got[np.searchsorted(t, 1.5)])) > 1e-9:
        raise RuntimeError("offset was dropped inside the 1 s hold")
    if abs(float(got[np.searchsorted(t, 2.05)] - 0.005)) > 1e-9:
        raise RuntimeError("offset was held past 1 s")


def main():
    _check()
    episodes = _load()
    _attach_surface(episodes)
    usable = {i: ep for i, ep in episodes.items() if i not in HELD}
    n_strokes = sum(len(ep["runs"]) for ep in usable.values())
    firm_n = sum(int(ep["firm"].sum()) for ep in usable.values())
    print(f"[fuse] strokes {n_strokes} firm samples {firm_n}", flush=True)

    for epi, ep in episodes.items():
        ep["clearance_raw"] = ep["clearance"]
        ep["clearance_A"] = _offset_a(ep["t"], ep["clearance"], ep["runs"])
        ep["clearance_B"] = _offset_b(ep["s"], ep["y"], ep["clearance"], ep["firm"])
    report = {"strokes": n_strokes, "firm_samples": firm_n, "methods": {}}
    methods = {}
    for name, key in (("raw", "clearance_raw"), ("A", "clearance_A"), ("B", "clearance_B")):
        row = _method_row(usable, key)
        report["methods"][name] = row
        methods[name] = key
        print(f"[fuse] {name} {json.dumps(row)}", flush=True)

    best_std = min(report["methods"]["A"]["residual"]["std_mm"], report["methods"]["B"]["residual"]["std_mm"])
    if best_std > LARGE_STD_MM:
        # Slow walk, 1 mm/s. The p90 stroke slope is a jump, not a drift the filter should follow.
        noise_mm = np.median([
            _noise_std(ep["t"], ep["clearance_raw"], ep["firm"]) or 1.0
            for ep in usable.values()
        ])
        r = (float(noise_mm) * 1e-3 / np.sqrt(2.0)) ** 2
        q = (1.0e-3) ** 2
        report["kalman"] = {"q_m2_per_s": q, "r_m2": r}
        for ep in episodes.values():
            ep["clearance_C"] = _kalman(ep["t"], ep["clearance"], ep["firm"], q, max(r, 1e-8))
        # stats filled below by reusing the loop body once
        print(f"[fuse] Kalman q {q:.3e} r {r:.3e}", flush=True)
    else:
        print(f"[fuse] skip Kalman, best firm residual std {best_std} mm", flush=True)

    if "clearance_C" in next(iter(episodes.values())):
        row = _method_row(usable, "clearance_C")
        report["methods"]["C"] = row
        methods["C"] = "clearance_C"
        print(f"[fuse] C {json.dumps(row)}", flush=True)

    latch_rows = []
    for name, key in methods.items():
        if name == "raw":
            continue
        swapped = {i: dict(ep, clearance=ep[key]) for i, ep in episodes.items()}
        for mm in C_ON_MM:
            row = _score(swapped, mm * 1e-3)
            row["method"] = name
            row["clearance_key"] = key
            latch_rows.append(row)
            print(
                f"[latch] {name} c_on {mm:g}  det {row['detected']}/{row['confirmed']}"
                f"  flips {row['false_flips']}  bumps {row['midstroke_entries']}"
                f"  descend {row['descending_entries']}"
                f"  lag {row['lag_median_s']}/{row['lag_p90_s']}"
                f"  air|contact {row['frac_contact_called_air']}"
                f"  post {row['post_dn_median_mm']} {row['post_frac_positive']}",
                flush=True,
            )
    ordered = sorted(latch_rows, key=lambda r: (
        r["false_flips"],
        -(r["post_dn_median_mm"] or -999),
        r["frac_contact_called_air"] or 1,
    ))
    report["latch"] = latch_rows
    report["best"] = ordered[:2]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table.json").write_text(json.dumps(report, indent=2) + "\n")
    _plot(episodes, ordered[:2])
    print("[fuse] best", json.dumps(ordered[0]), flush=True)


if __name__ == "__main__":
    main()
