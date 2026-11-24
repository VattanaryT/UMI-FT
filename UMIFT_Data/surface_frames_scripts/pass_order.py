#!/usr/bin/env python3
"""Pass order on the stringer, and whether checkpoint A fails at a half switch.

s is arc length along the wall centerline, 0 at the end farther from the
right-arm base. Far half is s below the midpoint, base half is above it.
A pass is one monotonic sweep of at least 50 mm. Does not write a normalizer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V2N = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-field_v2n/UMIFT_Data/surface_frames_scripts")
sys.path[:0] = [
    str(V2N),
    str(ROOT / "UMIFT_Data/surface_frames_scripts"),
    str(ROOT / "PyriteEnvSuites"),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT),
]

ZARR = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_field_v2n/acp_replay_buffer.zarr"
)
FIELD = ZARR.parents[1] / "shared/fixb_field.npz"
CKPT = Path("/mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_field_v2n/latest.ckpt")
HELD = (9, 12, 28, 37, 41)
EXCLUDED = (35, 44)
OUT = Path(__file__).resolve().parent / "eval/v2n_pass"
MIN_PASS = 0.05
AIR = 0.015


def _smooth(t, x, win):
    n = max(3, int(round(win / np.median(np.diff(t)))))
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


def _runs(labels):
    cuts = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    return np.split(np.arange(len(labels)), cuts)


def _geodesic(ra, rb):
    r = ra.T @ rb
    c = np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def _centerline(field, origin, axes):
    v = field["vertices"]
    n = field["normals"]
    body = ~field["is_apron"].astype(bool)
    local = (v[body] - origin) @ axes
    nm = n[body] @ axes
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


def _episode(group, origin, axes, cx, cs):
    t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    foot = (np.asarray(group["mold_x_s_0"], dtype=np.float64).reshape(-1, 3) - origin) @ axes
    normal = np.asarray(group["mold_normal_0"], dtype=np.float64).reshape(-1, 3) @ axes
    sd = np.asarray(group["mold_signed_distance_0"], dtype=np.float64).reshape(-1)
    n = min(len(t), len(foot), len(normal), len(sd))
    t, foot, normal, sd = t[:n], foot[:n], normal[:n], sd[:n]
    s = np.interp(foot[:, 0], cx, cs)
    wall = np.abs(normal[:, 2]) >= np.abs(normal[:, 1])
    return {"t": t, "s": s, "foot": foot, "wall": wall, "sd": sd, "pose": np.asarray(group["ts_pose_fb_0"])[:n]}


def _passes(ep, mid):
    """One visit per half. A 4 cm trip past the midpoint starts the other half."""
    s = _smooth(ep["t"], ep["s"], 0.5)
    state = 0 if s[0] < mid else 1
    mark = np.empty(len(s), dtype=np.int8)
    arm = 0.0
    for i, v in enumerate(s):
        if state == 0 and v > mid:
            arm = max(arm, v - mid)
            if arm >= 0.04:
                state, arm = 1, 0.0
        elif state == 1 and v < mid:
            arm = max(arm, mid - v)
            if arm >= 0.04:
                state, arm = 0, 0.0
        else:
            arm = 0.0
        mark[i] = state
    passes = []
    for idx in _runs(mark):
        if (ep["sd"][idx] > AIR).mean() > 0.5:
            continue
        if ep["t"][idx[-1]] - ep["t"][idx[0]] < 0.8:
            continue
        if float(s[idx].max() - s[idx].min()) < MIN_PASS:
            continue
        passes.append(idx)
    return s, mark, passes


def _half_name(s, mid):
    return "far" if s < mid else "base"


def _summarize(ep, s, passes, mid):
    order = []
    count = {"far": 0, "base": 0}
    for idx in passes:
        name = "far" if s[idx].mean() < mid else "base"
        count[name] += 1
        surface = "wall" if ep["wall"][idx].mean() >= 0.5 else "flange"
        order.append(f"{surface}:{name}")
    return count, order


def _at(ep, s, i, future):
    vel = float(np.gradient(s, ep["t"])[i])
    return {
        "ep": ep["id"],
        "i": i,
        "s": float(s[i]),
        "half": None,
        "wall": bool(ep["wall"][i]),
        "yz": ep["foot"][i, 1:],
        "sd": float(ep["sd"][i]),
        "vel": vel,
        "future": future,
        "pose": ep["pose"][i],
    }


def _decision_points(episodes, smoothed, passes, mid):
    """Points just before a half switch, and just before a turn that stays on the half."""
    points = []
    for epi, ep in episodes.items():
        ep["id"] = epi
        s = smoothed[epi]
        for n, idx in enumerate(passes[epi]):
            i = int(idx[max(0, len(idx) - int(0.4 / np.median(np.diff(ep["t"]))))])
            if ep["sd"][i] > AIR:
                continue
            nxt = passes[epi][n + 1] if n + 1 < len(passes[epi]) else None
            if nxt is not None and (s[nxt].mean() < mid) != (s[idx].mean() < mid):
                row = _at(ep, s, i, "other_half")
                row["half"] = _half_name(float(s[idx].mean()), mid)
                points.append(row)
        # Turnarounds of at least 8 cm that do not cross the midpoint within 1.5 s.
        vel = np.gradient(s, ep["t"])
        sign = np.sign(vel)
        sign[np.abs(vel) < 0.01] = 0
        dt = float(np.median(np.diff(ep["t"])))
        for cut in np.flatnonzero((sign[:-1] != 0) & (sign[1:] != 0) & (sign[:-1] != sign[1:])) + 1:
            j = min(len(s) - 1, cut + int(1.5 / dt))
            if abs(float(s[j] - s[cut])) < 0.08:
                continue
            i = max(0, cut - int(0.4 / np.median(np.diff(ep["t"]))))
            later = (ep["t"] > ep["t"][i]) & (ep["t"] <= ep["t"][i] + 1.5)
            if later.sum() and ((s[later] < mid).mean() > 0.2) and ((s[later] >= mid).mean() > 0.2):
                continue
            if ep["sd"][i] > AIR:
                continue
            if min(abs(float(s[i]) - mid), float(s[i]), 0.436 - float(s[i])) > 0.08:
                continue
            row = _at(ep, s, i, "second_pass")
            row["half"] = _half_name(float(s[i]), mid)
            points.append(row)
    return points


def _match(a, b, rot):
    if a["ep"] == b["ep"] or a["wall"] != b["wall"]:
        return False
    if abs(a["s"] - b["s"]) > 0.03:
        return False
    if np.linalg.norm(a["yz"] - b["yz"]) > 0.02:
        return False
    if abs(a["sd"] - b["sd"]) > 0.008:
        return False
    if np.sign(a["vel"]) != np.sign(b["vel"]) or abs(a["vel"] - b["vel"]) > 0.03:
        return False
    return rot < 15.0


def main() -> None:
    import zarr
    from field_v2 import mold_local
    from fixb_labels import pose7_to_Rp

    field = np.load(FIELD)
    root = zarr.open_group(str(ZARR), mode="r")
    spec = root.attrs["field_v2_axes"]
    origin = np.asarray(spec["origin"], dtype=np.float64)
    axes = np.asarray(spec["axes"], dtype=np.float64)
    cx, cs = _centerline(field, origin, axes)
    mid = 0.5 * (float(cs[0]) + float(cs[-1]))
    episodes = {}
    smoothed, passes = {}, {}
    rows = []
    for key in sorted(root["data"].keys(), key=lambda k: int(k.split("_")[-1])):
        epi = int(key.split("_")[-1])
        ep = _episode(root["data"][key], origin, axes, cx, cs)
        episodes[epi] = ep
        s, _mark, got = _passes(ep, mid)
        smoothed[epi] = s
        passes[epi] = got
        count, order = _summarize(ep, s, got, mid)
        role = "heldout" if epi in HELD else "excluded" if epi in EXCLUDED else "train"
        rows.append({
            "episode": epi,
            "role": role,
            "passes_far": count["far"],
            "passes_base": count["base"],
            "order": " ".join(order),
            "flange_fraction": round(float((~ep["wall"]).mean()), 3),
        })

    def tally(role):
        sub = [r for r in rows if r["role"] == role]
        out = {}
        for half in ("far", "base"):
            vals = [r[f"passes_{half}"] for r in sub]
            out[half] = {
                "n_demos": len(vals),
                "one_pass": int(sum(v == 1 for v in vals)),
                "two_passes": int(sum(v == 2 for v in vals)),
                "other": int(sum(v not in (1, 2) for v in vals)),
                "median": None if not vals else float(np.median(vals)),
            }
        return out

    points = _decision_points(episodes, smoothed, passes, mid)
    rots = {}
    for p in points:
        rots[id(p)] = pose7_to_Rp(p["pose"][None])[0][0]
    ambiguous = []
    opposite = {("other_half", "second_pass"), ("second_pass", "other_half")}
    for i, a in enumerate(points):
        hits = []
        for b in points[i + 1:]:
            ang = _geodesic(rots[id(a)], rots[id(b)])
            if not _match(a, b, ang):
                continue
            if (a["future"], b["future"]) in opposite:
                hits.append({"ep": b["ep"], "future": b["future"], "rot_deg": round(ang, 1)})
        if hits:
            ambiguous.append({"ep": a["ep"], "s": round(a["s"], 3), "half": a["half"], "future": a["future"], "matches": hits})

    # Obs content check: mold-local TCP x versus arc length, and the keys the policy sees.
    tcp = np.concatenate([pose7_to_Rp(ep["pose"])[1] for ep in episodes.values()])
    ss = np.concatenate([ep["s"] for ep in episodes.values()])
    local = mold_local(tcp, origin, axes)
    corr = [round(float(np.corrcoef(local[:, k], ss)[0, 1]), 4) for k in range(3)]

    report = {
        "s_definition": "arc length along the wall centerline, meters. 0 is the end farther from the right-arm base.",
        "centerline_length_m": round(float(cs[-1]), 3),
        "half_split_m": round(mid, 3),
        "obs": {
            "robot0_eef_pos": "absolute mold PCA xyz. x is the long axis, correlated with s.",
            "eef_pos_corr_with_s": {"x": corr[0], "y": corr[1], "z": corr[2]},
            "robot0_eef_rot_axis_angle": "absolute tool rot6d, 4 s history",
            "robot0_mold_geom": "signed distance, outward normal, local field rot6d. No arc length.",
            "wrench": "TCP wrench, 4 s history. Right wrench is zeros.",
            "pass_or_progress": "none. The 4 s position history shows the current direction, not an earlier pass.",
        },
        "train": tally("train"),
        "heldout": tally("heldout"),
        "demos": rows,
        "order_counts_train": {
            k: int(v) for k, v in sorted(
                ((r["order"], sum(1 for q in rows if q["role"] == "train" and q["order"] == r["order"])) for r in rows if r["role"] == "train"),
                key=lambda kv: -kv[1],
            )
        },
        "decision_class": {
            name: int(sum(p["future"] == name for p in points))
            for name in ("other_half", "second_pass")
        },
        "decision_points": len(points),
        "ambiguous_decision_points": len(ambiguous),
        "ambiguous_fraction": None if not points else round(len(ambiguous) / len(points), 4),
        "ambiguous": ambiguous[:30],
    }
    _open_loop(episodes, smoothed, passes, mid, report)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numbers.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in report if k not in ("demos", "ambiguous", "open_loop_rows")}, indent=2))


def _open_loop(episodes, smoothed, passes, mid, report) -> None:
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from diffusion_policy.common.drop_depth_inputs import apply_drop_depth_inputs
    from field_v2_dataset import UmiFTDatasetFieldV2
    from fixb_labels import pose7_to_Rp
    from mold_aware_runtime import rot6d_rows_to_R
    from PyriteEnvSuites.scripts.run_policy_open_loop_val_export import (
        load_policy_and_cfg,
        predict_sparse_action,
    )

    GlobalHydra.instance().clear()
    train_cfg = ROOT / "PyriteML/diffusion_policy/config"
    with initialize_config_dir(version_base=None, config_dir=str(train_cfg)):
        cfg = compose(
            config_name="train_umift_workspace",
            overrides=[
                f"hydra.searchpath=[file://{V2N / 'config'}]",
                "task=umift_oct_fixb_r4vt_field_v2n",
                "task.dataset.exclusion_stats_path=/tmp/v2n_pass_exclusion.json",
                "task.drop_depth_inputs=true",
                "hydra.job.chdir=false",
            ],
        )
    apply_drop_depth_inputs(cfg)
    OmegaConf.resolve(cfg)
    kw = OmegaConf.to_container(cfg.task.dataset, resolve=True)
    kw.pop("_target_", None)
    train = UmiFTDatasetFieldV2(**kw)
    val = train.get_validation_dataset()
    index = {}
    for i, (epi, rgb) in enumerate(val.sampler.indices):
        index.setdefault(int(epi), []).append((int(rgb), i))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, _ = load_policy_and_cfg(str(CKPT), device)
    rows = []
    for epi in HELD:
        ep = episodes[epi]
        rgb_t = np.asarray(train.replay_buffer["data"][f"episode_{epi}"]["obs"]["rgb_time_stamps_0"]).reshape(-1)
        switches = []
        s = smoothed[epi]
        half = np.array([_half_name(float(v), mid) for v in s])
        for cut in np.flatnonzero(half[1:] != half[:-1]) + 1:
            if abs(float(s[cut] - s[cut - 1])) < 0.01:
                switches.append(float(ep["t"][cut]))
        # A real half crossing moves across the midpoint, so keep cuts near mid.
        switches = [float(ep["t"][i]) for i in np.flatnonzero(half[1:] != half[:-1]) + 1 if abs(float(s[i]) - mid) < 0.03]
        wanted = set()
        last = -1e9
        for rgb, t in enumerate(rgb_t):
            near = any(abs(t - sw) <= 0.5 for sw in switches)
            if near or t - last >= 0.5:
                wanted.add(rgb)
                last = t
        for rgb, idx in index.get(epi, []):
            if rgb not in wanted:
                continue
            t = float(rgb_t[rgb])
            k = int(np.clip(np.searchsorted(ep["t"], t), 1, len(ep["t"]) - 1))
            if abs(ep["t"][k - 1] - t) < abs(ep["t"][k] - t):
                k -= 1
            ahead = [sw - t for sw in switches if sw > t]
            sample = val[idx]
            obs = {key: value.numpy() for key, value in sample["obs"]["sparse"].items()}
            pred = np.asarray(predict_sparse_action(policy, {"sparse": obs}, device), dtype=np.float64)
            gt = sample["action"]["sparse"].numpy()
            dt = float(np.median(np.diff(rgb_t)))
            frame = np.asarray(train.replay_buffer["data"][f"episode_{epi}"]["action"]).shape
            del frame
            R = np.asarray(zarr_frame(epi, k))
            p0 = pose7_to_Rp(ep["pose"][k:k + 1])[1][0]
            p1 = p0 + R[:, :2] @ (pred[0, :2] * dt)
            k2 = int(np.clip(np.searchsorted(ep["t"], t + dt), 1, len(ep["t"]) - 1))
            p_gt = pose7_to_Rp(ep["pose"][k2:k2 + 1])[1][0]
            rows.append({
                "ep": epi,
                "s": float(s[k]),
                "half": half[k],
                "t_to_switch": None if not ahead else round(min(ahead), 3),
                "pos_mm": float(np.linalg.norm(p1 - p_gt) * 1e3),
                "dR_deg": _geodesic(rot6d_rows_to_R(pred[0:1, 4:10])[0], rot6d_rows_to_R(gt[0:1, 4:10])[0]),
                "dn_mm": float((pred[0, 2] - gt[0, 2]) * 1e3),
            })
            if len(rows) % 50 == 0:
                print(f"[pass] {len(rows)}", flush=True)
    report["open_loop"] = _bins(rows)
    report["open_loop_n"] = len(rows)
    _plot(rows, OUT / "pass_error.png")


def zarr_frame(epi, k):
    import zarr
    g = zarr.open_group(str(ZARR), mode="r")["data"][f"episode_{epi}"]
    return np.asarray(g["mold_R_field_0"][k], dtype=np.float64).reshape(3, 3)


def _bins(rows):
    if not rows:
        return {}
    edges = np.array([0.10, 0.18, 0.26, 0.34, 0.45])
    out = {"by_s": [], "by_time_to_half_switch": []}
    for a, b in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if a <= r["s"] <= b] if b == edges[-1] else [r for r in rows if a <= r["s"] < b]
        out["by_s"].append({"s_lo": round(float(a), 3), "s_hi": round(float(b), 3), **_med(sel)})
    bands = [("under_0.5s", 0.0, 0.5), ("0.5_to_2s", 0.5, 2.0), ("over_2s", 2.0, 1e9), ("no_switch_ahead", None, None)]
    for name, lo, hi in bands:
        if lo is None:
            sel = [r for r in rows if r["t_to_switch"] is None]
        else:
            sel = [r for r in rows if r["t_to_switch"] is not None and lo <= r["t_to_switch"] < hi]
        out["by_time_to_half_switch"].append({"band": name, **_med(sel)})
    return out


def _med(rows):
    if not rows:
        return {"n": 0}
    def m(key):
        v = np.array([r[key] for r in rows])
        return round(float(np.median(np.abs(v) if key == "dn_mm" else v)), 2)
    return {"n": len(rows), "pos_mm": m("pos_mm"), "dR_deg": m("dR_deg"), "abs_dn_mm": m("dn_mm")}


def _plot(rows, path: Path) -> None:
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    s = np.array([r["s"] for r in rows])
    order = np.argsort(s)
    s = s[order]
    for ax, key, title in (
        (axes[0], "s", "vs arc length"),
        (axes[1], "t", "vs time to half switch"),
    ):
        pass
    def series(key):
        return np.array([r[key] for r in rows])[order]
    axes[0].scatter(s, series("pos_mm"), s=8, label="pos mm")
    axes[0].scatter(s, series("dR_deg"), s=8, label="ΔR deg")
    axes[0].scatter(s, np.abs(series("dn_mm")), s=8, label="|δn| mm")
    axes[0].set_xlabel("s (m)")
    tt = np.array([np.nan if r["t_to_switch"] is None else r["t_to_switch"] for r in rows])
    m = np.isfinite(tt)
    axes[1].scatter(tt[m], np.array([r["pos_mm"] for r in rows])[m], s=8, label="pos mm")
    axes[1].scatter(tt[m], np.array([r["dR_deg"] for r in rows])[m], s=8, label="ΔR deg")
    axes[1].scatter(tt[m], np.abs(np.array([r["dn_mm"] for r in rows])[m]), s=8, label="|δn| mm")
    axes[1].set_xlabel("seconds until the foot crosses the midpoint")
    axes[1].set_xlim(0, 8)
    for ax in axes:
        ax.legend(fontsize=7)
        ax.set_ylabel("error")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
