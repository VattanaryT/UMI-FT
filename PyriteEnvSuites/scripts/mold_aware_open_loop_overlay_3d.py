#!/usr/bin/env python3
"""3D overlay of expert vs the three mold-aware runs for one val episode.

Each run's executed open-loop trajectory (first exec_h steps of every chunk,
decoded to fixture frame) is drawn with markers colored by signed distance to
the mold (consistently oriented field, mm). The expert tool path (fb) and the
expert virtual target are drawn for reference, plus a subsampled mold cloud.

Usage:
  PYTHONPATH=<UMI-FT>/PyriteUtility python mold_aware_open_loop_overlay_3d.py \
    --episode episode_38 --ckpt latest --target vt --decode oracle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from mold_aware_eval_common import (
    RUNS,
    MoldField,
    decode_actions,
    load_expert_episode,
    load_export,
    load_surface_scale,
)

RUN_COLORS = {
    "world_world": "#1f77b4",
    "world_surface": "#2ca02c",
    "surface_surface": "#d62728",
}


def executed_path(T: np.ndarray, rgb_id: np.ndarray, exec_h: int) -> np.ndarray:
    """(N, 3) positions with NaN rows where consecutive queries are not contiguous."""
    pts = []
    for q in range(T.shape[0]):
        if q > 0 and rgb_id[q] - rgb_id[q - 1] != exec_h:
            pts.append(np.full((1, 3), np.nan))
        pts.append(T[q, :exec_h, :3, 3])
    return np.concatenate(pts, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--ckpt", default="latest")
    ap.add_argument("--target", choices=("vt", "cmd"), default="vt")
    ap.add_argument("--decode", choices=("oracle", "anchored"), default="oracle")
    ap.add_argument(
        "--exports_root",
        type=Path,
        default=Path("/mnt/data/umidata/ft/training_outputs/mold_aware/eval/exports"),
    )
    ap.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_data_v2_full"),
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/mnt/data/umidata/ft/training_outputs/mold_aware/eval/overlay"),
    )
    ap.add_argument("--sd_range_mm", type=float, default=60.0)
    ap.add_argument("--mold_points", type=int, default=5000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    field = MoldField(args.dataset_root / "shared" / "merged_fixture.ply")
    scale = load_surface_scale(args.dataset_root)
    expert = load_expert_episode(args.dataset_root, args.episode)
    color_kw = dict(
        coloraxis="coloraxis",
        size=3,
    )

    fig = go.Figure()
    rng = np.random.default_rng(0)
    sel = rng.choice(len(field.points), min(args.mold_points, len(field.points)), replace=False)
    mp = field.points[sel] * 1e3
    fig.add_trace(
        go.Scatter3d(
            x=mp[:, 0], y=mp[:, 1], z=mp[:, 2], mode="markers",
            marker=dict(size=1.5, color="lightgray", opacity=0.35), name="mold",
        )
    )

    span = None
    for run in RUNS:
        ck_dir = args.exports_root / run / args.ckpt
        if not (ck_dir / f"{args.episode}.npz").exists():
            print(f"[overlay] missing {ck_dir}/{args.episode}.npz")
            continue
        exec_h = int(json.loads((ck_dir / "meta.json").read_text())["sparse_execution_horizon"])
        d = load_export(ck_dir / f"{args.episode}.npz")
        dec = decode_actions(
            d["raw_action"], run, expert, d["action_robot_idx"], d["query_robot_idx"],
            scale, args.decode,
        )
        p = executed_path(dec[args.target], d["rgb_id"], exec_h)
        sd, _ = field.query(np.nan_to_num(p))
        sd_mm = np.where(np.isnan(p[:, 0]), np.nan, sd * 1e3)
        pm = p * 1e3
        fig.add_trace(
            go.Scatter3d(
                x=pm[:, 0], y=pm[:, 1], z=pm[:, 2], mode="lines+markers",
                line=dict(color=RUN_COLORS[run], width=3),
                marker=dict(color=sd_mm, **color_kw),
                name=f"{run} {args.target}",
                hovertemplate="sd=%{marker.color:.1f} mm<extra>" + run + "</extra>",
            )
        )
        if span is None:
            idx = d["action_robot_idx"]
            span = (int(idx.min()), int(idx.max()))

    lo, hi = span if span else (0, len(expert["t"]) - 1)
    stride = 10
    for name, key, color in (("expert tool (fb)", "fb", "black"), (f"expert {args.target}", args.target, "orange")):
        if key == "fb" and args.target == "cmd":
            continue
        pe = expert[key][lo : hi + 1 : stride, :3, 3]
        sd, _ = field.query(pe)
        pem = pe * 1e3
        fig.add_trace(
            go.Scatter3d(
                x=pem[:, 0], y=pem[:, 1], z=pem[:, 2], mode="lines+markers",
                line=dict(color=color, width=4),
                marker=dict(color=sd * 1e3, **color_kw),
                name=name,
                hovertemplate="sd=%{marker.color:.1f} mm<extra>" + name + "</extra>",
            )
        )

    fig.update_layout(
        title=(
            f"{args.episode}: {args.target} executed open-loop paths, {args.ckpt}, "
            f"{args.decode} decode (color = signed distance to mold, mm)"
        ),
        coloraxis=dict(
            colorscale="RdBu", cmin=-args.sd_range_mm, cmax=args.sd_range_mm,
            colorbar=dict(title="SD [mm]"),
        ),
        scene=dict(aspectmode="data", xaxis_title="x [mm]", yaxis_title="y [mm]", zaxis_title="z [mm]"),
        legend=dict(itemsizing="constant"),
        height=850,
    )
    out = args.out_dir / f"{args.episode}_{args.ckpt}_{args.target}_{args.decode}.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"[overlay] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
