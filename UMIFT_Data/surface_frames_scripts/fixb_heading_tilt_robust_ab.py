#!/usr/bin/env python3
"""Robust a,b normalizer. Does not touch e93b8756, the zarr, or the labels.

Same 34112 kept rows as the heading-tilt fit. Dims other than a and b are
copied from that pickle. a and b map their 1st and 99th percentiles to [-1, 1].
The stock LinearNormalizer has no clip, so the check below clips those two
channels after normalize. Samples outside the percentiles do not round-trip.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import zarr

from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from fixb_field import FixBMoldField
from fixb_labels import pose7_to_Rp
from fixb_smooth_frame_labels import EXCLUDE, LONG_DT
from fixb_tcp_track_clean import _kept
from heading_tilt import e_long_sequence, geodesic_deg, long_axis, rotation_from_tilt

ROOT = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
ZARR = ROOT / "r4vt_heading_tilt" / "acp_replay_buffer.zarr"
FIELD = ROOT / "shared" / "fixb_field.npz"
ASSETS = Path(__file__).resolve().parent / "assets"
PARENT = ASSETS / "oct_v35_h4s_r4vt_action_normalizer_minmax_heading_tilt_swing_sc.pkl"
PARENT_SHA = "e93b87561e762c2d7f1549a883eb18d45e6d50d929df463c43fce4817444e2f5"
OUT = ASSETS / "oct_v35_h4s_r4vt_action_normalizer_minmax_heading_tilt_swing_sc_robust_ab.pkl"
STAMP = OUT.with_suffix(".json")
EPISODES = (0, 9, 31)
KEPT = 34112


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vec(params, key):
    return params[key].detach().float().cpu().reshape(-1).numpy()


def kept_rows() -> np.ndarray:
    long_blob = np.load(LONG_DT)
    long_idx: dict[int, set[int]] = {}
    for ep, idx in zip(long_blob["flagged_episode"], long_blob["flagged_index"]):
        long_idx.setdefault(int(ep), set()).add(int(idx))
    root = zarr.open(str(ZARR), mode="r")
    rows = []
    for ep_id in range(50):
        if ep_id in EXCLUDE:
            continue
        group = root["data"][f"episode_{ep_id}"]
        tilt = np.asarray(group["fixb_heading_tilt_0"], dtype=np.float64)
        use = _kept(group, ep_id, long_idx, np.zeros(len(tilt), dtype=bool))
        if not np.any(use):
            continue
        command = np.asarray(group["ts_pose_command_0"], dtype=np.float64)
        stiff = np.asarray(group["stiffness_0"], dtype=np.float64).reshape(-1)
        rows.append(
            np.column_stack(
                [command[use, 0], command[use, 1], command[use, 2], stiff[use], tilt[use]]
            )
        )
    return np.concatenate(rows)


def _fit_ab(rows: np.ndarray, parent) -> LinearNormalizer:
    src = parent.params_dict["action"]
    scale = _vec(src, "scale").copy()
    offset = _vec(src, "offset").copy()
    stats = {key: _vec(src["input_stats"], key).copy() for key in ("min", "max", "mean", "std")}
    got_min = rows.min(0).astype(np.float32)
    got_max = rows.max(0).astype(np.float32)
    if not np.allclose(got_min, stats["min"], atol=1e-5) or not np.allclose(got_max, stats["max"], atol=1e-5):
        raise SystemExit(
            "kept-row min/max does not match e93b8756\n"
            f"rows {got_min.tolist()}\nfile {stats['min'].tolist()}\n"
            f"rows {got_max.tolist()}\nfile {stats['max'].tolist()}"
        )
    if not (scale[4] == 1 and scale[5] == 1 and offset[4] == 0 and offset[5] == 0):
        raise SystemExit("sin/cos are not identity on e93b8756")
    lo = np.percentile(rows[:, 6:8], 1, axis=0).astype(np.float32)
    hi = np.percentile(rows[:, 6:8], 99, axis=0).astype(np.float32)
    span = np.maximum(hi - lo, np.float32(1e-7))
    scale[6:8] = np.float32(2.0) / span
    offset[6:8] = np.float32(-1.0) - scale[6:8] * lo
    stats["min"][6:8] = lo
    stats["max"][6:8] = hi
    lin = LinearNormalizer()
    lin["action"] = SingleFieldLinearNormalizer.create_manual(scale, offset, stats)
    return lin, lo, hi


def _clip_back(action, scale, offset):
    norm = action * scale + offset
    outside = (norm[:, 6] < -1.0) | (norm[:, 6] > 1.0) | (norm[:, 7] < -1.0) | (norm[:, 7] > 1.0)
    norm = norm.copy()
    norm[:, 6:8] = np.clip(norm[:, 6:8], -1.0, 1.0)
    back = (norm - offset) / scale
    return back, outside


def _round_trip(scale, offset) -> dict:
    field = FixBMoldField(FIELD)
    axis = long_axis(field.vertices)
    root = zarr.open(str(ZARR), mode="r")
    report = {}
    for ep_id in EPISODES:
        group = root["data"][f"episode_{ep_id}"]
        rotation, position = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
        foot, vid, weight = field.closest(position)
        normal = np.einsum("qk,qkd->qd", weight, field.normals[vid])
        normal /= np.clip(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12, None)
        direction, _angle = e_long_sequence(normal, axis)
        tilt = np.asarray(group["fixb_heading_tilt_0"], dtype=np.float64)
        command = np.asarray(group["ts_pose_command_0"], dtype=np.float64)
        stiff = np.asarray(group["stiffness_0"], dtype=np.float64).reshape(-1)
        action = np.column_stack([command[:, 0], command[:, 1], command[:, 2], stiff, tilt])
        back, outside = _clip_back(action, scale, offset)
        plain = rotation_from_tilt(normal, direction, tilt)
        clipped = rotation_from_tilt(normal, direction, back[:, 4:8])
        base = geodesic_deg(rotation, plain)
        err = geodesic_deg(rotation, clipped)
        if float(np.median(base)) > 1e-3:
            raise SystemExit(f"episode {ep_id} unclipped round trip median {np.median(base)}")
        report[str(ep_id)] = {
            "n": int(err.size),
            "clipped": int(np.count_nonzero(outside)),
            "median_deg": float(np.median(err)),
            "max_deg": float(np.max(err)),
        }
    return report


def _range_used(rows, scale) -> dict:
    out = {}
    for index, name in ((6, "a"), (7, "b")):
        p05, p95 = np.percentile(rows[:, index], [5, 95])
        out[name] = float(scale[index] * (p95 - p05) / 2.0)
    return out


def main() -> None:
    if _sha(PARENT) != PARENT_SHA:
        raise SystemExit("e93b8756 sha changed; not reading it as the parent")
    if OUT.exists() or STAMP.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    rows = kept_rows()
    if len(rows) != KEPT:
        raise SystemExit(f"kept rows {len(rows)}, expected {KEPT}")
    with PARENT.open("rb") as handle:
        parent = pickle.load(handle)
    lin, lo, hi = _fit_ab(rows, parent)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("wb") as handle:
        pickle.dump(lin, handle)
    digest = _sha(OUT)
    src = lin.params_dict["action"]
    scale = _vec(src, "scale")
    offset = _vec(src, "offset")
    parent_scale = _vec(parent.params_dict["action"], "scale")
    before = _range_used(rows, parent_scale)
    after = _range_used(rows, scale)
    outside = (rows[:, 6] < lo[0]) | (rows[:, 6] > hi[0])
    outside_b = (rows[:, 7] < lo[1]) | (rows[:, 7] > hi[1])
    batch = torch.tensor(rows[: 64 * 64].reshape(64, 64, 8), dtype=torch.float32)
    normed = lin["action"].normalize(batch)
    if normed.shape != (64, 64, 8) or not torch.isfinite(normed).all():
        raise SystemExit(f"batch failed shape={tuple(normed.shape)} finite={bool(torch.isfinite(normed).all())}")
    trips = _round_trip(scale, offset)
    stamp = {
        "sha256": digest,
        "parent_sha256": PARENT_SHA,
        "kept_rows": KEPT,
        "episodes_excluded": [35, 44],
        "a": {
            "p01_rad": float(lo[0]),
            "p99_rad": float(hi[0]),
            "p01_deg": float(np.degrees(lo[0])),
            "p99_deg": float(np.degrees(hi[0])),
            "clip_fraction": float(np.mean(outside)),
        },
        "b": {
            "p01_rad": float(lo[1]),
            "p99_rad": float(hi[1]),
            "p01_deg": float(np.degrees(lo[1])),
            "p99_deg": float(np.degrees(hi[1])),
            "clip_fraction": float(np.mean(outside_b)),
        },
        "middle90_fraction_of_normalized_range": {"before": before, "after": after},
        "round_trip_clip": trips,
        "batch_64_64_8_finite": True,
    }
    STAMP.write_text(json.dumps(stamp, indent=2) + "\n")
    print(json.dumps(stamp, indent=2), flush=True)


if __name__ == "__main__":
    main()
