#!/usr/bin/env python3
"""Compare layup eval JSONs: zone mean bars + median-episode heatmaps.

Usage:
  python plot_layup_eval_compare.py
  python plot_layup_eval_compare.py \\
    --json combined=path/to/a_layup_eval.json \\
    --json teleop=path/to/b_layup_eval.json \\
    --out /tmp/layup_compare.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]  # umi_ft_ws
DEFAULT_RUNS: list[tuple[str, Path]] = [
    (
        "combined",
        ROOT
        / "data/training_outputs/data_source_test/combined/epoch=0095-train_loss=0.014_layup_eval.json",
    ),
    (
        "teleop",
        ROOT
        / "data/training_outputs/data_source_test/teleop/epoch=0155-train_loss=0.013_layup_eval.json",
    ),
    (
        "umi",
        ROOT
        / "data/training_outputs/data_source_test/umi/epoch=0170-train_loss=0.012_layup_eval.json",
    ),
]

METRICS = ("overall", "left", "center", "right")
ZONE_KEYS = ("left", "center", "right")


def parse_json_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"expected LABEL=PATH, got {value!r}"
        )
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"empty label in {value!r}")
    return label, Path(path).expanduser()


def load_run(label: str, path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    episodes = data.get("episodes") or []
    if not episodes:
        raise ValueError(f"{path}: no episodes")

    overall = np.array([float(ep["mean_mm"]) for ep in episodes], dtype=float)
    zones = {
        z: np.array(
            [float(ep["zones"][z]["mean_mm"]) for ep in episodes], dtype=float
        )
        for z in ZONE_KEYS
    }

    order = np.argsort(overall)
    median_idx = int(order[(len(order) - 1) // 2])  # lower-middle if even
    median_ep = episodes[median_idx]

    return {
        "label": label,
        "path": path,
        "n": len(episodes),
        "means": {
            "overall": float(overall.mean()),
            **{z: float(zones[z].mean()) for z in ZONE_KEYS},
        },
        "stds": {
            "overall": float(overall.std(ddof=0)),
            **{z: float(zones[z].std(ddof=0)) for z in ZONE_KEYS},
        },
        "median_ep": median_ep,
        "median_mean_mm": float(median_ep["mean_mm"]),
        "heatmap_png": median_ep.get("heatmap_png"),
    }


def plot_compare(
    runs: list[dict[str, Any]],
    *,
    title: str | None = None,
    out: Path | None = None,
    show: bool = False,
) -> None:
    n = len(runs)
    fig = plt.figure(figsize=(4.2 * max(n, 2), 8.5), layout="constrained")
    gs = GridSpec(2, n, figure=fig, height_ratios=[1.15, 1.0], hspace=0.12, wspace=0.08)

    ax_bar = fig.add_subplot(gs[0, :])
    x = np.arange(len(METRICS))
    width = min(0.8 / n, 0.28)
    offsets = (np.arange(n) - (n - 1) / 2.0) * width

    for i, run in enumerate(runs):
        means = [run["means"][m] for m in METRICS]
        stds = [run["stds"][m] for m in METRICS]
        bars = ax_bar.bar(
            x + offsets[i],
            means,
            width,
            yerr=stds,
            capsize=3,
            label=f"{run['label']} (n={run['n']})",
            error_kw={"linewidth": 1.0},
        )
        for bar, mean in zip(bars, means):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(METRICS)
    ax_bar.set_ylabel("distance mean (mm)")
    ax_bar.set_title(title or "Layup eval comparison (mean ± std over episodes)")
    ax_bar.legend(frameon=False, loc="upper right")
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.set_axisbelow(True)

    for i, run in enumerate(runs):
        ax = fig.add_subplot(gs[1, i])
        ax.set_xticks([])
        ax.set_yticks([])
        ep_id = run["median_ep"].get("episode_id", "?")
        caption = (
            f"{run['label']}  median ep={ep_id}  "
            f"mean={run['median_mean_mm']:.2f} mm"
        )
        heatmap_path = run["heatmap_png"]
        if not heatmap_path:
            ax.text(
                0.5,
                0.5,
                "no heatmap_png in JSON",
                ha="center",
                va="center",
                wrap=True,
            )
            ax.set_title(caption, fontsize=9)
            continue

        path = Path(heatmap_path)
        if not path.is_file():
            ax.text(
                0.5,
                0.5,
                f"missing heatmap\n{path}",
                ha="center",
                va="center",
                fontsize=8,
                wrap=True,
            )
            ax.set_title(caption, fontsize=9)
            print(f"[skip] {run['label']}: heatmap not found: {path}")
            continue

        img = np.asarray(Image.open(path))
        ax.imshow(img)
        ax.set_title(caption, fontsize=9)

    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")
    if show or out is None:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare layup eval JSONs (bars + median heatmaps)."
    )
    parser.add_argument(
        "--json",
        dest="jsons",
        action="append",
        type=parse_json_arg,
        metavar="LABEL=PATH",
        help="Eval JSON to include (repeatable). Default: combined/teleop/umi.",
    )
    parser.add_argument("--title", default=None, help="Figure title.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Save PNG here (default: show interactively).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display the figure when --out is set.",
    )
    args = parser.parse_args()

    specs = args.jsons if args.jsons else DEFAULT_RUNS
    runs = [load_run(label, path) for label, path in specs]
    plot_compare(runs, title=args.title, out=args.out, show=args.show)


if __name__ == "__main__":
    main()
