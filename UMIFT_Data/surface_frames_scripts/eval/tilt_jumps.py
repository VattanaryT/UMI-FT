#!/usr/bin/env python3
"""Jumps in the stored tilt chart versus the tool rotation.

A neighbor step is two consecutive robot samples, dt under 0.1 s.
|Δψ| is wrapped. |Δa| and |Δb| are not. The tool geodesic is the angle of
R_i^T R_{i+1} from ts_pose_fb_0. A jump over 45° whose geodesic stays at or
under 45° is a chart artifact.
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
from scipy.spatial.transform import Rotation

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / "eval")]

import frame_variance as fv  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import pose7_to_Rp  # noqa: E402
from heading_tilt import e_long_sequence, long_axis, pack_tilt  # noqa: E402

OUT = SCRIPTS / "eval/tilt_jumps.json"
MD = SCRIPTS / "eval/tilt_jumps.md"
PNG = SCRIPTS / "eval/tilt_jumps.png"
LIMIT = 45.0
NAMES = ("psi", "a", "b")


def _wrap(delta):
    return (delta + 180.0) % 360.0 - 180.0


def _geodesic(rotation):
    rel = np.einsum("nji,njk->nik", rotation[:-1], rotation[1:])
    return np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=1))


def _angles(tilt):
    psi = np.degrees(np.arctan2(tilt[:, 0], tilt[:, 1]))
    ab = np.degrees(tilt[:, 2:4])
    return psi, ab[:, 0], ab[:, 1]


def _deltas(psi, a, b):
    return np.column_stack([_wrap(np.diff(psi)), np.diff(a), np.diff(b)])


def _true_normal(field, position):
    _foot, vid, weight = field.closest(position)
    normal = np.einsum("qk,qkd->qd", weight, field.normals[vid])
    return normal / np.clip(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12, None)


def _carry_normal(normal, geodesic):
    """Hold the previous normal when the closest point hops and the tool does not."""
    out = np.array(normal, copy=True)
    for i in range(1, len(out)):
        if float(np.dot(out[i], out[i - 1])) < 0.0:
            out[i] *= -1.0
        ang = np.degrees(np.arccos(np.clip(float(np.dot(out[i], out[i - 1])), -1.0, 1.0)))
        if ang > 20.0 and geodesic[i - 1] < 20.0:
            out[i] = out[i - 1]
    return out


def _episode(store, episode, field, axis, bottom, hat):
    group = store[f"episode_{episode}"]
    tilt = np.asarray(group["fixb_heading_tilt_0"], np.float64)
    pose = np.asarray(group["ts_pose_fb_0"], np.float64)
    normal = np.asarray(group["mold_normal_0"], np.float64)
    foot = np.asarray(group["mold_x_s_0"], np.float64)
    t = np.asarray(group["robot_time_stamps_0"], np.float64).reshape(-1)
    n = min(len(tilt), len(pose), len(normal), len(foot), len(t))
    tilt, pose, normal, foot, t = tilt[:n], pose[:n], normal[:n], foot[:n], t[:n]
    rotation, position = pose7_to_Rp(pose)
    psi, a, b = _angles(tilt)
    delta = _deltas(psi, a, b)
    geo = _geodesic(rotation)
    dt = np.diff(t)
    neighbor = (dt > 1e-4) & (dt < 0.1)
    if "fixb_heading_tilt_keep_0" in group:
        keep = np.asarray(group["fixb_heading_tilt_keep_0"], np.uint8).reshape(-1)[:n] > 0
        neighbor = neighbor & keep[:-1] & keep[1:]
    queried = _true_normal(field, position)
    carried = _carry_normal(queried, np.r_[geo, 0.0])
    repack = pack_tilt(rotation, carried, e_long_sequence(carried, axis)[0])
    fixed_delta = _deltas(*_angles(repack))
    raw_n = queried.copy()
    for i in range(1, len(raw_n)):
        if float(np.dot(raw_n[i], raw_n[i - 1])) < 0.0:
            raw_n[i] *= -1.0
    normal_step = np.degrees(np.arccos(np.clip(np.sum(raw_n[:-1] * raw_n[1:], axis=1), -1.0, 1.0)))
    region = fv.regions_of(normal, foot, bottom, hat)[:-1]
    return {
        "t": t,
        "psi": psi,
        "a": a,
        "b": b,
        "delta": delta,
        "fixed_delta": fixed_delta,
        "geo": geo,
        "neighbor": neighbor,
        "region": region,
        "normal_step": normal_step,
    }


def _empty():
    return {name: {"jumps": 0, "artifact": 0, "real": 0} for name in NAMES + ("any",)}


def _add(bucket, jump, artifact):
    for i, name in enumerate(NAMES):
        bucket[name]["jumps"] += int(jump[:, i].sum())
        bucket[name]["artifact"] += int((jump[:, i] & artifact).sum())
        bucket[name]["real"] += int((jump[:, i] & ~artifact).sum())
    any_jump = jump.any(1)
    bucket["any"]["jumps"] += int(any_jump.sum())
    bucket["any"]["artifact"] += int((any_jump & artifact).sum())
    bucket["any"]["real"] += int((any_jump & ~artifact).sum())


def _cause(row, mask):
    art = mask & (row["geo"] <= LIMIT)
    hopped = row["normal_step"] > 20.0
    return {
        "artifact": int(art.sum()),
        "normal_hop": int((art & hopped).sum()),
        "no_normal_hop": int((art & ~hopped).sum()),
    }


def _examples(rows):
    found = []
    for episode, row in rows:
        jump = np.abs(row["delta"]) > LIMIT
        art = jump.any(1) & (row["geo"] <= LIMIT) & row["neighbor"]
        if not art.any():
            continue
        score = np.abs(row["delta"]).sum(1)
        score[~art] = -1.0
        i = int(np.argmax(score))
        found.append((float(score[i]), episode, i, row))
    found.sort(reverse=True)
    picked = []
    seen = set()
    for _score, episode, i, row in found:
        region = str(row["region"][i])
        if region in seen:
            continue
        seen.add(region)
        picked.append((episode, i, row))
        if len(picked) == 3:
            break
    return picked


def _plot(examples, png):
    fig, axes = plt.subplots(3, 3, figsize=(11, 7), sharex="col")
    series = ("psi", "a", "b")
    for col, (episode, i, row) in enumerate(examples):
        lo = max(0, i - 40)
        hi = min(len(row["psi"]), i + 41)
        sl = slice(lo, hi)
        time = row["t"][sl] - row["t"][i]
        for r, name in enumerate(series):
            ax = axes[r, col]
            ax.plot(time, row[name][sl], color="#4aa3ff", lw=0.9)
            ax.axvline(0.0, color="#888", lw=0.6)
            ax.set_ylabel(name + " deg")
        which = [name for k, name in enumerate(NAMES) if abs(row["delta"][i, k]) > LIMIT]
        axes[0, col].set_title(
            f"ep {episode} {row['region'][i]}\n{', '.join(which)}  geo {row['geo'][i]:.1f}°"
        )
        axes[-1, col].set_xlabel("time s")
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


def _line(stats):
    return f"{stats['jumps']} jumps, {stats['artifact']} artifact, {stats['real']} real"


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else fv.ZARR
    if len(sys.argv) > 2:
        stem = Path(sys.argv[2])
        out, md, png = stem.with_suffix(".json"), stem.with_suffix(".md"), stem.with_suffix(".png")
    else:
        out, md, png = OUT, MD, PNG
    store = zarr.open(source, "r")["data"]
    field = FixBMoldField(
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
    )
    axis = long_axis(field.vertices)
    feet = []
    for episode in range(50):
        if f"episode_{episode}" not in store:
            continue
        foot = np.asarray(store[f"episode_{episode}"]["mold_x_s_0"][::20], np.float64)
        feet.append(foot)
    feet = np.concatenate(feet)
    bottom = float(np.percentile(feet[:, 1], 90))
    hat = fv.across_hat(feet)
    rows = []
    region_stats = {name: _empty() for name in fv.REGIONS}
    episode_stats = {}
    cause = {"artifact": 0, "normal_hop": 0, "no_normal_hop": 0}
    fixed_left = 0
    leftover_geo = []
    steps = 0
    masked = 0
    if "fixb_heading_tilt_keep_0" in store["episode_0"]:
        for episode in range(50):
            if f"episode_{episode}" not in store:
                continue
            keep = np.asarray(store[f"episode_{episode}"]["fixb_heading_tilt_keep_0"]).reshape(-1)
            masked += int((keep == 0).sum())
    for episode in range(50):
        if f"episode_{episode}" not in store:
            continue
        row = _episode(store, episode, field, axis, bottom, hat)
        rows.append((episode, row))
        use = row["neighbor"]
        steps += int(use.sum())
        jump = (np.abs(row["delta"]) > LIMIT) & use[:, None]
        artifact = row["geo"] <= LIMIT
        bucket = _empty()
        _add(bucket, jump, artifact)
        episode_stats[str(episode)] = {name: bucket[name]["artifact"] for name in ("any",) + NAMES}
        episode_stats[str(episode)]["steps"] = int(use.sum())
        for name in fv.REGIONS:
            mask = use & (row["region"] == name)
            part = _empty()
            _add(part, jump & mask[:, None], artifact)
            for key, stats in part.items():
                for item, value in stats.items():
                    region_stats[name][key][item] += value
        counted = _cause(row, jump.any(1) & use)
        for key, value in counted.items():
            cause[key] += value
        fixed_jump = (np.abs(row["fixed_delta"]) > LIMIT) & use[:, None]
        residual = fixed_jump.any(1) & artifact & use
        fixed_left += int(residual.sum())
        if residual.any():
            leftover_geo.append(row["geo"][residual])
        print(f"episode {episode} artifacts {bucket['any']['artifact']}", flush=True)
    total = _empty()
    for stats in region_stats.values():
        for key, block in stats.items():
            for item, value in block.items():
                total[key][item] += value
    report = {
        "steps": steps,
        "masked_samples": masked,
        "limit_deg": LIMIT,
        "total": total,
        "by_region": region_stats,
        "by_episode": episode_stats,
        "artifact_cause": cause,
        "artifacts_after_normal_carry": fixed_left,
        "leftover_geodesic_deg": (
            {
                "median": round(float(np.median(np.concatenate(leftover_geo))), 2),
                "max": round(float(np.max(np.concatenate(leftover_geo))), 2),
            }
            if leftover_geo
            else None
        ),
        "fix": (
            "Before pack_tilt, keep the previous closest-point normal when it turns "
            "more than 20° and the tool geodesic on that step is under 20°. "
            "A sign flip is not enough: the hop is often about 70°, not 180°. "
            "Unwrap does not help, because ψ barely moves. Masking only the jump "
            "samples leaves the hopped chart in place until the normal hops back."
        ),
    }
    out.write_text(json.dumps(report, indent=2))
    lines = [
        "# Tilt-chart jumps",
        "",
        "Neighbor samples are consecutive robot rows with dt under 0.1 s.",
        "A jump is |Δψ|, |Δa|, or |Δb| over 45°. ψ is wrapped to ±180°.",
        "Artifact means the tool geodesic on that same step is at most 45°.",
        "",
        f"Steps {steps}. Masked samples {masked}. A masked sample is not a neighbor.",
        "",
        "| | jumps | artifact | real |",
        "|---|---:|---:|---:|",
    ]
    for name in ("any",) + NAMES:
        block = total[name]
        lines.append(f"| {name} | {block['jumps']} | {block['artifact']} | {block['real']} |")
    lines += ["", "## By region", "", "| region | any artifact | ψ | a | b |", "|---|---:|---:|---:|---:|"]
    for name in fv.REGIONS:
        block = region_stats[name]
        lines.append(
            f"| {name} | {block['any']['artifact']} | {block['psi']['artifact']} | "
            f"{block['a']['artifact']} | {block['b']['artifact']} |"
        )
    lines += [
        "",
        "## Why the chart jumped",
        "",
        f"Artifacts {cause['artifact']}. Closest-point normal turned more than 20° on "
        f"{cause['normal_hop']} of them, and on {cause['no_normal_hop']} it did not.",
        f"After carrying that normal across hops, artifacts left: {fixed_left}.",
        "",
        report["fix"],
        "",
        "## Episodes",
        "",
    ]
    ranked = sorted(episode_stats.items(), key=lambda item: item[1]["any"], reverse=True)
    for episode, stats in ranked:
        if stats["any"] == 0:
            continue
        lines.append(
            f"- episode {episode}: {stats['any']} artifact steps "
            f"(ψ {stats['psi']}, a {stats['a']}, b {stats['b']}) of {stats['steps']}"
        )
    lines.append("")
    md.write_text("\n".join(lines))
    _plot(_examples(rows), png)
    print(json.dumps({"total": total, "cause": cause, "fixed_left": fixed_left}, indent=2), flush=True)


if __name__ == "__main__":
    main()
