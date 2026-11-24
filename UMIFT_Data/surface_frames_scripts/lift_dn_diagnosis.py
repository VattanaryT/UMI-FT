#!/usr/bin/env python3
"""Why field_v2n under-predicts clearance after contact. Reads the zarr and checkpoint A.

Contact is |F·n| >= 5 N, air is |F·n| <= 2 N, matching the δn blend.
A lift-off is the action step where contact ends. Post-contact steps are the
next 0.5 s. Does not write the normalizer or the checkpoint.
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

from diffusion_policy.common.drop_depth_inputs import apply_drop_depth_inputs  # noqa: E402
from diffusion_policy.common.sampler_umift import (  # noqa: E402
    sparse_action_indices_from_rgb_timestamps,
)
from field_v2 import smooth_force  # noqa: E402
from field_v2n import LO, HI, target_clearance  # noqa: E402
from field_v2_dataset import UmiFTDatasetFieldV2  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import pose7_to_Rp  # noqa: E402
from fixb_r4vt_dataset import UmiFTDatasetR4VT  # noqa: E402

ZARR = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_field_v2n/acp_replay_buffer.zarr"
)
FIELD = ZARR.parents[1] / "shared/fixb_field.npz"
CKPT = Path("/mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_field_v2n/latest.ckpt")
PLOT_EPS = (9, 12, 28, 37, 41)
WINDOW = 0.5
OUT = Path(__file__).resolve().parent / "eval/v2n_lift_dn"


class Capture(UmiFTDatasetFieldV2):
    def _apply(self, split):
        pre = [(int(a), int(b)) for a, b in self.sampler.indices]
        stats = UmiFTDatasetR4VT._apply(self, split)
        self._capture = {
            "split": split,
            "pre": pre,
            "post": [(int(a), int(b)) for a, b in self.sampler.indices],
            "stats": stats,
        }
        return stats


def _fn(group):
    robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rotation, _p = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
    frame = np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
    n = min(len(robot_t), len(rotation), len(frame))
    outward = -frame[:n, :, 2]
    force = smooth_force(group, robot_t[:n])
    world = np.einsum("nij,nj->ni", rotation[:n], force)
    return robot_t[:n], np.einsum("ni,ni->n", world, outward), frame[:n], rotation[:n]


def _episode(group, action_n):
    robot_t, fn, frame, _rotation = _fn(group)
    n = min(action_n, len(fn))
    robot_t, fn, frame = robot_t[:n], fn[:n], frame[:n]
    contact = np.abs(fn) >= HI
    air = np.abs(fn) <= LO
    ends = np.flatnonzero(contact[:-1] & ~contact[1:])
    isolated = []
    for end in ends:
        later = np.flatnonzero(contact[end + 1:])
        if later.size == 0 or float(robot_t[end + 1 + later[0]] - robot_t[end]) > WINDOW:
            isolated.append(int(end))
    isolated = np.asarray(isolated, dtype=np.int64)
    since = np.full(n, np.inf)
    for end in isolated:
        dt = robot_t - robot_t[end]
        take = (dt > 0.0) & (dt <= WINDOW) & (dt < since)
        since[take] = dt[take]
    post = np.isfinite(since) & ~contact
    dn = np.asarray(group["fixb_delta_n_0"], dtype=np.float64).reshape(-1)[:n]
    return {
        "t": robot_t,
        "fn": fn,
        "frame": frame,
        "contact": contact,
        "air": air,
        "ends": ends,
        "isolated": isolated,
        "post": post,
        "dn": dn,
        "position": pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))[1][:n],
    }


def _chunk(ep, action_idx):
    idx = np.asarray(action_idx, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < len(ep["contact"]))]
    if idx.size == 0:
        return False, np.zeros(0, dtype=np.int64)
    in_horizon = np.zeros(len(ep["contact"]), dtype=bool)
    in_horizon[idx] = True
    transition = bool(np.any(in_horizon[ep["isolated"]])) if ep["isolated"].size else False
    return transition, idx[ep["post"][idx]]


def _horizon(replay, epi, rgb, horizon, stride):
    ep = replay[f"episode_{epi}"]
    idx, _times = sparse_action_indices_from_rgb_timestamps(
        action_times=np.asarray(ep["action_time_stamps"]),
        rgb_times=np.asarray(ep["obs"]["rgb_time_stamps_0"]),
        rgb_id=rgb,
        action_horizon=horizon,
        rgb_frame_stride=stride,
    )
    return idx


def _fraction(hits, total):
    return {
        "n": int(hits),
        "of": int(total),
        "fraction": None if total == 0 else round(hits / total, 4),
    }


def _clearance(ep, group, index, field):
    if index.size == 0:
        return np.zeros(0)
    rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    _arrived, _foot, clearance = target_clearance(
        ep["position"], ep["frame"], ep["t"], rgb_t, field
    )
    return clearance[index]


def main() -> None:
    import hydra
    import torch
    import zarr
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
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
                "task.dataset.exclusion_stats_path=/tmp/v2n_lift_exclusion.json",
                "task.drop_depth_inputs=true",
                "hydra.job.chdir=false",
            ],
        )
    apply_drop_depth_inputs(cfg)
    OmegaConf.resolve(cfg)
    kw = OmegaConf.to_container(cfg.task.dataset, resolve=True)
    kw.pop("_target_", None)
    train = Capture(**kw)
    cap = train._capture
    stats = cap["stats"]
    expect = {"before": 19125, "dropped_bad_span": 5888, "dropped_long_dt": 916, "kept": 11127}
    for key, value in expect.items():
        if stats[key] != value:
            raise SystemExit(f"train filter {key}={stats[key]}, expected {value}")

    root = zarr.open_group(str(ZARR), mode="r")["data"]
    field = FixBMoldField(str(FIELD))
    episodes = {}
    for epi, _rgb in cap["pre"]:
        if epi not in episodes:
            action_n = int(np.asarray(train.replay_buffer["data"][f"episode_{epi}"]["action"]).shape[0])
            episodes[epi] = _episode(root[f"episode_{epi}"], action_n)

    shape = train.sampler.shape_meta
    horizon = int(shape["sample"]["action"]["sparse"]["horizon"])
    stride = int(shape["sample"]["action"]["sparse"]["down_sample_steps"])
    kept_set = set(cap["post"])
    reason = {}
    pre_set = set(cap["pre"])
    # Reclassify with the same priority as drop_excluded_queries.
    span, long = {}, {}
    from fixb_r4vt_dataset import _robot_masks

    n_action = {
        epi: int(np.asarray(train.replay_buffer["data"][f"episode_{epi}"]["action_time_stamps"]).shape[0])
        for epi in episodes
    }
    span, long = _robot_masks(str(ZARR), train._long_dt_mask_path, n_action)

    buckets = {"kept": [], "bad_span": [], "long_dt": [], "both": [], "episode": []}
    post_steps = {name: [] for name in buckets}
    for epi, rgb in cap["pre"]:
        idx = _horizon(train.replay_buffer["data"], epi, rgb, horizon, stride)
        transition, post_idx = _chunk(episodes[epi], idx)
        hit_span = bool(np.any(span[epi][idx])) if idx.size else False
        hit_dt = bool(np.any(long[epi][idx])) if idx.size else False
        if epi in set(train._exclude):
            name = "episode"
        elif (epi, rgb) in kept_set:
            name = "kept"
        elif hit_span and hit_dt:
            name = "both"
        elif hit_span:
            name = "bad_span"
        elif hit_dt:
            name = "long_dt"
        else:
            name = "episode"
        buckets[name].append(transition)
        post_steps[name].append((epi, post_idx))

    def pack(name):
        flags = np.asarray(buckets[name], dtype=bool)
        steps = post_steps[name]
        occ = int(sum(len(idx) for _epi, idx in steps))
        unique = {(epi, int(i)) for epi, idx in steps for i in idx}
        return {
            "chunks": _fraction(int(flags.sum()), int(flags.size)),
            "post_contact_step_occurrences": occ,
            "post_contact_unique_steps": len(unique),
        }

    # Unique post-contact rows that appear in a kept chunk, with label vs clearance.
    kept_rows = {}
    for epi, post_idx in post_steps["kept"]:
        kept_rows.setdefault(epi, set()).update(int(i) for i in post_idx)
    label_mm, clear_mm, fn_n, blend_n, air_n = [], [], [], [], []
    for epi, rows in kept_rows.items():
        ep = episodes[epi]
        index = np.array(sorted(rows), dtype=np.int64)
        clearance = _clearance(ep, root[f"episode_{epi}"], np.arange(len(ep["t"])), field)
        label_mm.append(ep["dn"][index] * 1e3)
        clear_mm.append(clearance[index] * 1e3)
        fn_n.append(np.abs(ep["fn"][index]))
        blend_n.append(int(np.sum((np.abs(ep["fn"][index]) > LO) & (np.abs(ep["fn"][index]) < HI))))
        air_n.append(int(np.sum(ep["air"][index])))
    label_mm = np.concatenate(label_mm) if label_mm else np.zeros(0)
    clear_mm = np.concatenate(clear_mm) if clear_mm else np.zeros(0)
    fn_n = np.concatenate(fn_n) if fn_n else np.zeros(0)

    # Events and model overlay on the five val episodes.
    val = train.get_validation_dataset()
    val_index = {(int(ep), int(rgb)): i for i, (ep, rgb) in enumerate(val.sampler.indices)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, _pcfg = load_policy_and_cfg(str(CKPT), device)
    curves = []
    pred_rel, pred_mm, label_at_pred = [], [], []
    for epi in PLOT_EPS:
        group = root[f"episode_{epi}"]
        action_n = int(np.asarray(train.replay_buffer["data"][f"episode_{epi}"]["action"]).shape[0])
        ep = episodes.get(epi) or _episode(group, action_n)
        episodes[epi] = ep
        clearance = _clearance(ep, group, np.arange(len(ep["t"])), field)
        rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
        for end in ep["isolated"]:
            t0 = float(ep["t"][end])
            sel = (ep["t"] >= t0 - WINDOW) & (ep["t"] <= t0 + WINDOW)
            curves.append({
                "episode": int(epi),
                "t": (ep["t"][sel] - t0).astype(np.float32),
                "label_mm": (ep["dn"][sel] * 1e3).astype(np.float32),
                "clearance_mm": (clearance[sel] * 1e3).astype(np.float32),
                "fn": ep["fn"][sel].astype(np.float32),
            })
            for rgb, t in enumerate(rgb_t):
                rel = float(t - t0)
                if not -WINDOW <= rel <= WINDOW:
                    continue
                key = (epi, rgb)
                if key not in val_index:
                    continue
                sample = val[val_index[key]]
                obs = {k: v.numpy() for k, v in sample["obs"]["sparse"].items()}
                pred = float(predict_sparse_action(policy, {"sparse": obs}, device)[0, 2]) * 1e3
                pred_rel.append(rel)
                pred_mm.append(pred)
                label_at_pred.append(float(sample["action"]["sparse"][0, 2]) * 1e3)

    if pack("bad_span")["chunks"]["n"] and pack("bad_span")["chunks"]["of"] != stats["dropped_bad_span"]:
        raise SystemExit(f"bad_span bucket {pack('bad_span')['chunks']['of']} != {stats['dropped_bad_span']}")
    if pack("long_dt")["chunks"]["of"] != stats["dropped_long_dt"]:
        raise SystemExit(f"long_dt bucket {pack('long_dt')['chunks']['of']} != {stats['dropped_long_dt']}")
    # Open-loop window: raw |F| > 5 N is contact. The 0.5 s after that phase ends.
    raw_rows = []
    by_ep = {}
    for (epi, rgb), idx in val_index.items():
        if epi not in PLOT_EPS:
            continue
        t = float(np.asarray(root[f"episode_{epi}"]["rgb_time_stamps_0"]).reshape(-1)[rgb])
        by_ep.setdefault(epi, []).append((t, idx))
    for epi, rows in by_ep.items():
        rows.sort()
        group = root[f"episode_{epi}"]
        wt = np.asarray(group["wrench_time_stamps_0"], dtype=np.float64).reshape(-1)
        wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
        ep = episodes[epi]
        clearance = _clearance(ep, group, np.arange(len(ep["t"])), field)
        forces, clears = [], []
        for t, _idx in rows:
            j = int(np.clip(np.searchsorted(wt, t), 1, len(wt) - 1))
            if abs(wt[j - 1] - t) <= abs(wt[j] - t):
                j -= 1
            forces.append(float(np.linalg.norm(wrench[j, :3])))
            k = int(np.clip(np.searchsorted(ep["t"], t), 1, len(ep["t"]) - 1))
            if abs(ep["t"][k - 1] - t) <= abs(ep["t"][k] - t):
                k -= 1
            clears.append(max(float(clearance[k]), 0.0))
        phase = "approach"
        labels = []
        for force, gap in zip(forces, clears):
            if force > 5.0:
                phase = "contact"
            elif force < 2.0 and gap > 0.010:
                phase = "approach"
            elif phase == "contact":
                phase = "lift-off"
            labels.append(phase)
        ends = [rows[i][0] for i, label in enumerate(labels) if label == "contact" and (i + 1 == len(labels) or labels[i + 1] != "contact")]
        for (t, idx), _label in zip(rows, labels):
            if any(0.0 < t - end <= WINDOW for end in ends):
                raw_rows.append(idx)
    raw_pred, raw_label = [], []
    for idx in raw_rows:
        sample = val[idx]
        obs = {k: v.numpy() for k, v in sample["obs"]["sparse"].items()}
        raw_pred.append(float(predict_sparse_action(policy, {"sparse": obs}, device)[0, 2]) * 1e3)
        raw_label.append(float(sample["action"]["sparse"][0, 2]) * 1e3)
    raw_pred = np.asarray(raw_pred)
    raw_label = np.asarray(raw_label)
    after = np.asarray(pred_rel) > 0
    pred_after = np.asarray(pred_mm)[after] if len(pred_mm) else np.zeros(0)
    label_after = np.asarray(label_at_pred)[after] if len(label_at_pred) else np.zeros(0)
    report = {
        "definitions": {
            "contact": "|F·n| >= 5 N",
            "air": "|F·n| <= 2 N",
            "transition": "the horizon includes a lift-off: contact ends and does not return within 0.5 s",
            "post_contact": "0 < t - t_end <= 0.5 s after that lift-off, and not back in contact",
        },
        "filter_counts": stats,
        "kept": pack("kept"),
        "dropped_bad_span": pack("bad_span"),
        "dropped_long_dt": pack("long_dt"),
        "dropped_both": pack("both"),
        "kept_post_contact_label": {
            "n_unique_steps": int(label_mm.size),
            "median_label_mm": None if label_mm.size == 0 else round(float(np.median(label_mm)), 3),
            "frac_label_positive": None if label_mm.size == 0 else round(float(np.mean(label_mm > 0)), 4),
            "median_clearance_mm": None if clear_mm.size == 0 else round(float(np.median(clear_mm)), 3),
            "frac_clearance_positive": None if clear_mm.size == 0 else round(float(np.mean(clear_mm > 0)), 4),
            "median_abs_fn_n": None if fn_n.size == 0 else round(float(np.median(fn_n)), 3),
            "n_still_in_blend": int(sum(blend_n)),
            "n_full_air": int(sum(air_n)),
        },
        "plot_episodes": list(PLOT_EPS),
        "plot_events": len(curves),
        "isolated_lifts_in_train_episodes": int(sum(len(ep["isolated"]) for ep in episodes.values())),
        "crossings_in_train_episodes": int(sum(len(ep["ends"]) for ep in episodes.values())),
        "model_queries_in_window": len(pred_mm),
        "model_after_contact": {
            "n": int(pred_after.size),
            "median_pred_mm": None if pred_after.size == 0 else round(float(np.median(pred_after)), 3),
            "median_label_mm": None if label_after.size == 0 else round(float(np.median(label_after)), 3),
            "frac_pred_positive": None if pred_after.size == 0 else round(float(np.mean(pred_after > 0)), 4),
            "frac_label_positive": None if label_after.size == 0 else round(float(np.mean(label_after > 0)), 4),
        },
        "raw_force_window_val": {
            "n": int(raw_pred.size),
            "median_pred_mm": None if raw_pred.size == 0 else round(float(np.median(raw_pred)), 3),
            "median_label_mm": None if raw_label.size == 0 else round(float(np.median(raw_label)), 3),
            "frac_pred_positive": None if raw_pred.size == 0 else round(float(np.mean(raw_pred > 0)), 4),
            "frac_label_positive": None if raw_label.size == 0 else round(float(np.mean(raw_label > 0)), 4),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numbers.json").write_text(json.dumps(report, indent=2) + "\n")
    pieces_t, pieces_l, pieces_c, pieces_f, pieces_e = [], [], [], [], []
    for c in curves:
        pieces_t.append(c["t"])
        pieces_l.append(c["label_mm"])
        pieces_c.append(c["clearance_mm"])
        pieces_f.append(c["fn"])
        pieces_e.append(np.full(c["t"].shape, c["episode"]))
    np.savez(
        OUT / "curves.npz",
        t=np.concatenate(pieces_t),
        episode=np.concatenate(pieces_e),
        label_mm=np.concatenate(pieces_l),
        clearance_mm=np.concatenate(pieces_c),
        fn=np.concatenate(pieces_f),
        pred_rel=np.asarray(pred_rel, dtype=np.float32),
        pred_mm=np.asarray(pred_mm, dtype=np.float32),
        label_at_pred=np.asarray(label_at_pred, dtype=np.float32),
    )
    _plot(curves, np.asarray(pred_rel), np.asarray(pred_mm), OUT / "lift_dn.png")
    print(json.dumps(report, indent=2))


def _plot(curves, pred_rel, pred_mm, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True)
    ax, af = axes
    for c in curves:
        ax.plot(c["t"], c["clearance_mm"], color="0.75", lw=0.6)
        ax.plot(c["t"], c["label_mm"], color="C0", lw=0.7, alpha=0.7)
        af.plot(c["t"], c["fn"], color="0.65", lw=0.6)
    grid = np.linspace(-WINDOW, WINDOW, 41)
    med_l, med_c = [], []
    for t in grid:
        lab, clr = [], []
        for c in curves:
            m = (c["t"] >= t - 0.02) & (c["t"] < t + 0.02)
            if np.any(m):
                lab.append(np.median(c["label_mm"][m]))
                clr.append(np.median(c["clearance_mm"][m]))
        med_l.append(np.median(lab) if lab else np.nan)
        med_c.append(np.median(clr) if clr else np.nan)
    ax.plot(grid, med_c, color="0.15", lw=2.0, label="clearance median")
    ax.plot(grid, med_l, color="C0", lw=2.0, label="δn label median")
    if pred_rel.size:
        ax.scatter(pred_rel, pred_mm, s=16, color="C3", zorder=3, label="A predicted δn")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6, ls="--")
    ax.set_ylabel("mm")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Isolated lift-off, episodes 9, 12, 28, 37, 41. t=0 is |F·n| falling through 5 N")
    af.axhline(HI, color="C3", lw=0.8, label="5 N contact")
    af.axhline(LO, color="C2", lw=0.8, label="2 N air")
    af.axhline(0, color="k", lw=0.6)
    af.axvline(0, color="k", lw=0.6, ls="--")
    af.set_ylabel("F·n (N)")
    af.set_xlabel("seconds after contact ends")
    af.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
