#!/usr/bin/env python3
"""Variant A contact distance and tool-tilt jitter, first half vs second half.

Half is the time midpoint between the first and last |F|>=8 sample.
Jitter is the frame-to-frame tool rotation on successive frames that are both in contact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CMP = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
try:
    from fixb_field import FixBMoldField
except ImportError:
    sys.path.insert(0, str(CMP / "UMIFT_Data/surface_frames_scripts"))
    from fixb_field import FixBMoldField

import zarr

WALL = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/"
    "wall_fix/fixb_field_wall_vertical.npz"
)
VARIANT_DIR = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/variant_resolve"
)
ZARR_DIR = Path("/mnt/data/umidata/ft/datasets/flat_stringer/zarr")
HERE = Path(__file__).resolve().parent
EPISODES = (
    ("C", "2026-11-19-15.49.52"),
    ("A", "2026-11-19-15.43.29"),
    ("B", "2026-11-19-15.45.09"),
    ("D", "2026-11-19-15.51.13"),
)
FORCE_N = 8.0
_FIELD = None


def _check() -> None:
    turned = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    step = _step_deg(np.eye(3)[None], turned[None])
    if abs(float(step[0]) - 90.0) > 1e-6:
        raise RuntimeError("frame-to-frame angle is not the geodesic")


def _field():
    global _FIELD
    if _FIELD is None:
        _FIELD = FixBMoldField(str(WALL))
    return _FIELD


def _step_deg(r0: np.ndarray, r1: np.ndarray) -> np.ndarray:
    rel = np.einsum("nji,njk->nik", r0, r1)
    trace = np.clip((rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2] - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def _stats(mm: np.ndarray) -> dict:
    if len(mm) == 0:
        return {"n": 0, "median_mm": None, "p90_mm": None}
    return {
        "n": int(len(mm)),
        "median_mm": round(float(np.median(mm)), 1),
        "p90_mm": round(float(np.percentile(mm, 90)), 1),
    }


def _jitter(deg: np.ndarray) -> dict:
    if len(deg) == 0:
        return {"n": 0, "median_deg": None, "p90_deg": None}
    return {
        "n": int(len(deg)),
        "median_deg": round(float(np.median(deg)), 2),
        "p90_deg": round(float(np.percentile(deg, 90)), 2),
    }


def load_variant(stamp: str, variant: str = "A"):
    blob = np.load(VARIANT_DIR / variant / f"{stamp}_ee_master.npz")
    T = np.asarray(blob["T_master_tcp"], dtype=np.float64)
    times = np.asarray(blob["timestamp_sec_unified"], dtype=np.float64).reshape(-1)
    tag = np.asarray(blob["n_tags_visible"]).astype(int) >= 1
    g = zarr.open(ZARR_DIR / f"{stamp}_unified.zarr", "r")["sensors"]["ft_wrench"]
    wrench_t = np.asarray(g["timestamp_sec_unified"], dtype=np.float64)
    force = np.linalg.norm(
        np.column_stack([np.asarray(g[k]) for k in ("force_x", "force_y", "force_z")]),
        axis=1,
    )
    j = np.clip(np.searchsorted(wrench_t, times), 1, len(wrench_t) - 1)
    left = np.abs(times - wrench_t[j - 1]) <= np.abs(times - wrench_t[j])
    mag = force[np.where(left, j - 1, j)]
    dist = np.linalg.norm(T[:, :3, 3] - _field().closest(T[:, :3, 3])[0], axis=1) * 1e3
    return {
        "t": times - float(times[0]),
        "dist": dist,
        "tag": tag,
        "contact": mag >= FORCE_N,
        "R": T[:, :3, :3],
    }


def halves(row: dict) -> dict:
    hit = row["contact"]
    t = row["t"]
    if not np.any(hit):
        empty = {"n": 0, "median_mm": None, "p90_mm": None}
        return {"t_mid_s": None, "first": empty, "second": empty,
                "jitter_first": _jitter(np.zeros(0)), "jitter_second": _jitter(np.zeros(0))}
    t_mid = 0.5 * (float(t[hit].min()) + float(t[hit].max()))
    first = hit & (t < t_mid)
    second = hit & (t >= t_mid)
    step = _step_deg(row["R"][:-1], row["R"][1:])
    both_first = first[:-1] & first[1:]
    both_second = second[:-1] & second[1:]
    return {
        "t_mid_s": round(t_mid, 2),
        "first": _stats(row["dist"][first]),
        "second": _stats(row["dist"][second]),
        "jitter_first": _jitter(step[both_first]),
        "jitter_second": _jitter(step[both_second]),
    }


def gap_payload(episode: int, variant: str = "A") -> dict:
    """JSON series for the 8802 page. episode ids match the page selector."""
    ids = {0: "C", 1: "A", 2: "B", 3: "D"}
    letter = ids[int(episode)]
    stamp = dict(EPISODES)[letter]
    row = load_variant(stamp, variant)
    stat = halves(row)
    a, b = stat["first"], stat["second"]
    note = (
        f"{letter} variant {variant}  contact |F|≥8  "
        f"first med {a['median_mm']} / p90 {a['p90_mm']} mm (n={a['n']})  "
        f"second med {b['median_mm']} / p90 {b['p90_mm']} mm (n={b['n']})"
    )
    return {
        "gap_t": [round(float(v), 3) for v in row["t"]],
        "gap_mm": [round(float(v), 2) for v in row["dist"]],
        "gap_tag": [bool(v) for v in row["tag"]],
        "gap_note": note,
    }


def _plot(rows: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=False)
    for ax, (letter, _stamp) in zip(axes, EPISODES):
        row = rows[letter]
        stat = halves(row)
        ax.plot(row["t"], row["dist"], color="#4aa3ff", lw=1)
        ax.scatter(row["t"][row["tag"]], row["dist"][row["tag"]], c="#3ddc97", s=18, zorder=3, label="tag")
        if stat["t_mid_s"] is not None:
            ax.axvline(stat["t_mid_s"], color="#888", lw=0.8)
        a, b = stat["first"], stat["second"]
        ax.set_ylabel("mm")
        ax.set_title(
            f"{letter}  first {a['median_mm']}/{a['p90_mm']} mm   "
            f"second {b['median_mm']}/{b['p90_mm']} mm",
            loc="left", fontsize=9,
        )
    axes[-1].set_xlabel("seconds from start")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "contact_half.png", dpi=120)
    plt.close(fig)


def main() -> None:
    _check()
    out = {}
    loaded = {}
    for letter, stamp in EPISODES:
        row = load_variant(stamp, "A")
        loaded[letter] = row
        out[letter] = halves(row)
        out[letter]["contact_all"] = _stats(row["dist"][row["contact"]])
    if not (14.0 <= out["C"]["contact_all"]["median_mm"] <= 16.0):
        raise RuntimeError(f"recording C contact median drifted: {out['C']['contact_all']}")
    path = HERE / "contact_half.json"
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.write_text(json.dumps({"variant": "A", "half": "time midpoint of contact", "episodes": out}, indent=2))
    _plot(loaded)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
