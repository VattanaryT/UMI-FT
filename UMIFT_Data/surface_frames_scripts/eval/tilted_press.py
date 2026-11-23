#!/usr/bin/env python3
"""Low-force moments: does δn deepen over the next 0.4 s with tilt a held?

The window is the compare setup's chunk: 6 control steps at 1/15 s, RGB stride 8,
bad spans and long-dt targets dropped. Demos use every episode except 35 and 44.
Causal-foot e199 is scored on the seed-42 held-out episodes, at the low-force
queries only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [
    str(SCRIPTS),
    str(SCRIPTS / "eval"),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT / "PyriteEnvSuites/scripts"),
    str(ROOT),
]

import frame_variance as fv  # noqa: E402
import openloop_causal_foot as cf  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import pose7_to_Rp  # noqa: E402
from heading_tilt import e_long_sequence, long_axis, pack_tilt  # noqa: E402
from mold_aware_runtime import integrate_tcp_foot, rot6d_rows_to_R  # noqa: E402

HEAD = zarr.open(fv.ZARR, "r")["data"]
CKPT = (
    "/mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_causal_foot/"
    "oct-v35-h4s-r4vt-causal-foot_e200_bs64_valtopk_43390127/checkpoints/latest.ckpt"
)
OUT = SCRIPTS / "eval/tilted_press.json"
MD = SCRIPTS / "eval/tilted_press.md"
PNG = SCRIPTS / "eval/tilted_press.png"
DEEPEN_M = 0.002
HOLD_DEG = 3.0


def _force(robot_t, rotation, normal, group):
    wrench_t = np.asarray(group["wrench_time_stamps_0"], np.float64).reshape(-1)
    wrench = np.asarray(group["wrench_left_0"], np.float64)[:, :3]
    order = np.argsort(wrench_t)
    wrench_t, wrench = wrench_t[order], wrench[order]
    keep = np.ones(len(wrench_t), dtype=bool)
    keep[1:] = np.diff(wrench_t) > 1e-9
    wrench_t, wrench = wrench_t[keep], wrench[keep]
    force = np.column_stack([np.interp(robot_t, wrench_t, wrench[:, i]) for i in range(3)])
    world = np.einsum("nij,nj->ni", rotation, force)
    return np.sum(world * normal, axis=1)


def _episode(episode):
    group = HEAD[f"episode_{episode}"]
    robot_t = np.asarray(group["robot_time_stamps_0"], np.float64).reshape(-1)
    rgb_t = np.asarray(group["rgb_time_stamps_0"], np.float64).reshape(-1)
    pose = np.asarray(group["ts_pose_fb_0"], np.float64)
    tilt = np.asarray(group["fixb_heading_tilt_0"], np.float64)
    dn = np.asarray(group["fixb_delta_n_0"], np.float64).reshape(-1)
    normal = np.asarray(group["mold_normal_0"], np.float64)
    foot = np.asarray(group["mold_x_s_0"], np.float64)
    n = min(len(robot_t), len(pose), len(tilt), len(dn), len(normal), len(foot))
    rotation, _p = pose7_to_Rp(pose[:n])
    bad = (
        np.asarray(group["bad_span_0"], bool).reshape(-1)[:n]
        if "bad_span_0" in group
        else np.zeros(n, dtype=bool)
    )
    return {
        "robot_t": robot_t[:n],
        "rgb_t": rgb_t,
        "bad": bad,
        "a": np.degrees(tilt[:n, 2]),
        "dn": dn[:n],
        "normal": normal[:n],
        "foot": foot[:n],
        "fn": _force(robot_t[:n], rotation, normal[:n], group),
    }


def _outcome(dn0, dn1, a0, a1, sign):
    deepen = sign * (dn1 - dn0) > DEEPEN_M
    hold = abs(a1 - a0) <= HOLD_DEG
    toward = abs(a1) < abs(a0) - HOLD_DEG
    return bool(deepen), bool(hold), bool(toward)


def _rates(rows, prefix=""):
    key = (lambda name: prefix + name) if prefix else (lambda name: name)
    rows = [row for row in rows if key("deepen") in row]
    if not rows:
        return {"n": 0}
    deepen = np.array([row[key("deepen")] for row in rows])
    hold = np.array([row[key("hold")] for row in rows])
    toward = np.array([row[key("toward0")] for row in rows])
    da_key = prefix + "da" if prefix else "da"
    da = np.array([row.get(da_key, np.nan) for row in rows], dtype=np.float64)
    return {
        "n": int(len(rows)),
        "deepen_and_hold": round(float(np.mean(deepen & hold)), 3),
        "deepen_and_toward0": round(float(np.mean(deepen & toward)), 3),
        "deepen_and_toward0_under_45deg": round(float(np.mean(deepen & toward & (np.abs(da) < 45.0))), 3),
        "deepen": round(float(np.mean(deepen)), 3),
        "hold": round(float(np.mean(hold)), 3),
        "toward0": round(float(np.mean(toward)), 3),
    }


def _predict_chunk(field, axis, action, p0, r0):
    action = np.asarray(action, np.float64)
    if action.shape[-1] == 27:
        v = action[:, 0:2]
        dn = action[:, 2]
        dR = rot6d_rows_to_R(action[:, 21:27])
    else:
        v = action[:, 0:2]
        dn = action[:, 2]
        dR = rot6d_rows_to_R(action[:, 4:10])
    dt = np.full(len(action), cf.DT_S)
    foot = integrate_tcp_foot(field, p0, v, dn, dt, R_tool=r0)
    rotation = np.einsum("nij,njk->nik", foot["R"], dR)
    e_long, _angle = e_long_sequence(foot["n_out"], axis)
    packed = pack_tilt(rotation, foot["n_out"], e_long)
    return dn, np.degrees(packed[:, 2])


def _span(ep, event):
    seg = ep["a"][event["i0"] : event["i1"] + 1]
    return float(np.max(seg) - np.min(seg))


def _examples(events, episodes):
    hold = [
        row for row in events
        if row["deepen"] and row["hold"] and _span(episodes[row["episode"]], row) < 15.0
    ]
    toward = [
        row for row in events
        if row["deepen"] and row["toward0"] and abs(row["da"]) < 45.0
    ]
    picked = []
    for pool in (hold, toward):
        choice = next((row for row in pool if row.get("pred_a") is not None), None)
        if choice is None and pool:
            choice = pool[0]
        if choice is not None:
            picked.append(choice)
    if len(picked) < 2:
        for row in events:
            if row not in picked and row["deepen"]:
                picked.append(row)
            if len(picked) == 2:
                break
    return picked[:2]


def _plot(examples, episodes):
    fig, axes = plt.subplots(3, len(examples), figsize=(5 * len(examples), 7), squeeze=False)
    labels = ("F·n N", "δn mm", "a deg")
    for col, event in enumerate(examples):
        ep = episodes[event["episode"]]
        t0 = ep["robot_t"][event["i0"]]
        window = (ep["robot_t"] >= t0 - 1.0) & (ep["robot_t"] <= t0 + 0.6)
        series = (ep["fn"][window], ep["dn"][window] * 1e3, ep["a"][window])
        pred_t = t0 + np.arange(cf.EXEC_H) * cf.DT_S
        pred = (
            (None, event.get("pred_dn"), event.get("pred_a"))
            if event.get("pred_a") is not None
            else (None, None, None)
        )
        for row, (name, values) in enumerate(zip(labels, series)):
            ax = axes[row, col]
            ax.plot(ep["robot_t"][window] - t0, values, color="#4aa3ff", lw=0.8)
            ax.axvline(0.0, color="#888", lw=0.6)
            if pred[row] is not None:
                scale = 1e3 if row == 1 else 1.0
                ax.plot(pred_t - t0, np.asarray(pred[row]) * scale, color="#ff4d6a", lw=1.2)
            ax.set_ylabel(name)
        kind = "hold" if event["hold"] else "toward 0"
        axes[0, col].set_title(f"ep {event['episode']} {event['region']} {kind}")
    axes[-1, 0].set_xlabel("time from the low-force sample, s")
    fig.tight_layout()
    fig.savefig(PNG, dpi=120)
    plt.close(fig)


def _md(report):
    lines = [
        "# Tilted pressing after a low normal force",
        "",
        "A moment counts when F·n is more than one standard deviation under that",
        "region's median. F·n is the TCP force rotated into the fixture frame and",
        "dotted with the outward normal. The next 0.4 s is six steps at 1/15 s.",
        f"δn deepens when it moves {report['deepen_sign']:+d} by more than 2 mm,",
        "the sign of corr(δn, F·n). Hold means |Δa| ≤ 3°. Toward 0 means |a| drops by more than 3°.",
        "Causal-foot e199 is decoded like the compare setup, at these same moments.",
        "Toward 0 under 45° drops branch cuts, where the change in a is at least 45°.",
        "",
        "| region | n | deepen+hold | deepen+toward 0 | toward 0 under 45 | pred n | pred deepen+hold | pred deepen+toward 0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for region in ("all",) + fv.REGIONS:
        demo = report["demo"].get(region, {"n": 0})
        pred = report["pred"].get(region, {"n": 0})
        lines.append(
            f"| {region} | {demo.get('n', 0)} | {demo.get('deepen_and_hold', '')} | "
            f"{demo.get('deepen_and_toward0', '')} | {demo.get('deepen_and_toward0_under_45deg', '')} | "
            f"{pred.get('n', 0)} | {pred.get('deepen_and_hold', '')} | {pred.get('deepen_and_toward0', '')} |"
        )
    lines.append("")
    MD.write_text("\n".join(lines))


def main():
    episodes = {}
    feet = []
    fn_parts = []
    region_parts = []
    for episode in range(50):
        if episode in fv.SKIP or f"episode_{episode}" not in HEAD:
            continue
        ep = _episode(episode)
        episodes[episode] = ep
        sl = slice(None, None, fv.STRIDE)
        feet.append(ep["foot"][sl])
        fn_parts.append(ep["fn"][sl])
        print(f"demo {episode}", flush=True)
    feet = np.concatenate(feet)
    bottom = float(np.percentile(feet[:, 1], 90))
    hat = fv.across_hat(feet)
    for episode, ep in episodes.items():
        ep["region"] = fv.regions_of(ep["normal"], ep["foot"], bottom, hat)
        sl = slice(None, None, fv.STRIDE)
        region_parts.append(ep["region"][sl])
    fn = np.concatenate(fn_parts)
    region = np.concatenate(region_parts)
    dn = np.concatenate([ep["dn"][:: fv.STRIDE] for ep in episodes.values()])
    sign = 1.0 if np.corrcoef(dn, fn)[0, 1] >= 0 else -1.0
    level = {}
    for name in fv.REGIONS:
        values = fn[region == name]
        level[name] = (float(np.median(values)), float(values.std()))
    long = cf._long_dt()
    events = []
    for episode, ep in episodes.items():
        queries = cf.query_indices(
            {"robot_t": ep["robot_t"], "rgb_t": ep["rgb_t"], "bad": ep["bad"]},
            episode,
            long.get(episode, set()),
        )
        for rgb_id, index in queries:
            i0, i1 = int(index[0]), int(index[-1])
            name = str(ep["region"][i0])
            median, std = level[name]
            if ep["fn"][i0] >= median - std:
                continue
            deepen, hold, toward = _outcome(ep["dn"][i0], ep["dn"][i1], ep["a"][i0], ep["a"][i1], sign)
            events.append(
                {
                    "episode": episode,
                    "rgb": int(rgb_id),
                    "i0": i0,
                    "i1": i1,
                    "region": name,
                    "fn": float(ep["fn"][i0]),
                    "a0": float(ep["a"][i0]),
                    "da": float(ep["a"][i1] - ep["a"][i0]),
                    "deepen": deepen,
                    "hold": hold,
                    "toward0": toward,
                }
            )
    pred_events = events
    print(f"events {len(events)} pred candidates {len(pred_events)} sign {sign:+.0f}", flush=True)
    _score_predictions(pred_events, sign)
    demo_rows = {"all": _rates(events)}
    pred_rows = {"all": _rates(pred_events, "pred_")}
    for name in fv.REGIONS:
        demo_rows[name] = _rates([row for row in events if row["region"] == name])
        pred_rows[name] = _rates([row for row in pred_events if row["region"] == name], "pred_")
    report = {
        "deepen_sign": int(sign),
        "corr_dn_fn": round(float(np.corrcoef(dn, fn)[0, 1]), 3),
        "force_level": {name: {"median": round(med, 2), "std": round(std, 2)} for name, (med, std) in level.items()},
        "demo": demo_rows,
        "pred": pred_rows,
        "ckpt": CKPT,
    }
    OUT.write_text(json.dumps(report, indent=2))
    _md(report)
    chosen = _examples(events, episodes)
    _plot(chosen, episodes)
    print(json.dumps({"demo": demo_rows, "pred": pred_rows}, indent=2), flush=True)


def _score_predictions(events, sign):
    if not events:
        return
    import torch

    import export_openloop as exporter
    import score_causal_foot as score

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    policy, stock, device, val = score._load_val(
        Path(CKPT), cf.CAUSAL_ZARR, cf.CAUSAL_NORM, cf.CAUSAL_SHA, True, device_name
    )
    field = FixBMoldField(cf.FIELD)
    axis = long_axis(field.vertices)
    for done, event in enumerate(events, start=1):
        try:
            obs, _action = score._sample_rgb(val.sampler, event["episode"], event["rgb"])
        except Exception as exc:
            event["pred_error"] = str(exc)
            continue
        pred = np.asarray(
            exporter.predict_one(policy, obs, device, stock, "r4vt", event["rgb"]),
            dtype=np.float64,
        )[: cf.EXEC_H]
        sparse = obs["sparse"]
        p0 = np.asarray(sparse["robot0_eef_pos"][-1], np.float64).reshape(3)
        r0 = rot6d_rows_to_R(np.asarray(sparse["robot0_eef_rot_axis_angle"][-1], np.float64))
        dn, tilt_a = _predict_chunk(field, axis, pred, p0, r0)
        deepen, hold, toward = _outcome(dn[0], dn[-1], tilt_a[0], tilt_a[-1], sign)
        event["pred_dn"] = [round(float(v), 5) for v in dn]
        event["pred_a"] = [round(float(v), 2) for v in tilt_a]
        event["pred_da"] = float(tilt_a[-1] - tilt_a[0])
        event["pred_deepen"] = deepen
        event["pred_hold"] = hold
        event["pred_toward0"] = toward
        if done % 20 == 0 or done == len(events):
            print(f"pred {done}/{len(events)}", flush=True)


if __name__ == "__main__":
    main()
