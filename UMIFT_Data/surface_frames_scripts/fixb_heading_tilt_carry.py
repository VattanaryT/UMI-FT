#!/usr/bin/env python3
"""New heading-tilt zarr. Does not modify r4vt_heading_tilt or its pickles.

Hardlinks that zarr, then replaces fixb_heading_tilt_0. The closest-point
normal is held when it turns more than 20° and the tool geodesic is under 20°.
pack_tilt runs on that normal. Steps that still jump more than 45° while the
tool turns at most 45° are marked 0 in fixb_heading_tilt_keep_0.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from fixb_build_datasets import link_tree, replace_array
from fixb_field import FixBMoldField
from fixb_labels import pose7_to_Rp
from fixb_smooth_frame_labels import EXCLUDE, LONG_DT
from fixb_tcp_track_clean import _kept
from heading_tilt import e_long_sequence, geodesic_deg, long_axis, pack_tilt, rotation_from_tilt

ROOT = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
SRC = ROOT / "r4vt_heading_tilt" / "acp_replay_buffer.zarr"
FIELD = ROOT / "shared" / "fixb_field.npz"
OUT = ROOT / "r4vt_heading_tilt_carry"
NORM = (
    Path(__file__).resolve().parent
    / "assets"
    / "oct_v35_h4s_r4vt_action_normalizer_minmax_heading_tilt_carry.pkl"
)
STAMP = NORM.with_suffix(".json")
LIMIT = 45.0


def _self_check() -> None:
    base = np.array([0.0, 0.0, 1.0])
    hopped = np.array([np.sin(np.radians(70.0)), 0.0, np.cos(np.radians(70.0))])
    normal = np.stack([base, hopped, base])
    held = _carry_normal(normal, np.zeros(3))
    assert np.allclose(held[1], held[0])
    slight = np.array([np.sin(np.radians(10.0)), 0.0, np.cos(np.radians(10.0))])
    moved = _carry_normal(np.stack([base, slight]), np.array([0.0, 0.0]))
    assert np.allclose(moved[1], slight)
    kept = _carry_normal(normal, np.array([30.0, 0.0, 0.0]))
    assert not np.allclose(kept[1], kept[0])


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _true_normal(field: FixBMoldField, position: np.ndarray):
    foot, vid, weight = field.closest(position)
    normal = np.einsum("qk,qkd->qd", weight, field.normals[vid])
    normal /= np.clip(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12, None)
    return foot, normal


def _carry_normal(normal: np.ndarray, geodesic: np.ndarray) -> np.ndarray:
    """Hold the previous normal when the closest point hops and the tool does not."""
    out = np.array(normal, copy=True)
    for i in range(1, len(out)):
        if float(np.dot(out[i], out[i - 1])) < 0.0:
            out[i] *= -1.0
        ang = np.degrees(np.arccos(np.clip(float(np.dot(out[i], out[i - 1])), -1.0, 1.0)))
        if ang > 20.0 and geodesic[i - 1] < 20.0:
            out[i] = out[i - 1]
    return out


def _geodesic(rotation: np.ndarray) -> np.ndarray:
    rel = np.einsum("nji,njk->nik", rotation[:-1], rotation[1:])
    return np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=1))


def _angles(tilt: np.ndarray):
    psi = np.degrees(np.arctan2(tilt[:, 0], tilt[:, 1]))
    return psi, np.degrees(tilt[:, 2]), np.degrees(tilt[:, 3])


def _wrap(delta: np.ndarray) -> np.ndarray:
    return (delta + 180.0) % 360.0 - 180.0


def _leftover(tilt: np.ndarray, rotation: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Destination sample of a >45° chart step whose tool geodesic is <=45°."""
    psi, a, b = _angles(tilt)
    delta = np.column_stack([_wrap(np.diff(psi)), np.diff(a), np.diff(b)])
    dt = np.diff(time)
    neighbor = (dt > 1e-4) & (dt < 0.1)
    bad = neighbor & (np.abs(delta).max(1) > LIMIT) & (_geodesic(rotation) <= LIMIT)
    keep = np.ones(len(tilt), dtype=bool)
    keep[1:][bad] = False
    return keep


def _fit(rows: np.ndarray) -> None:
    if NORM.exists() and NORM.stat().st_size > 0:
        raise SystemExit(f"refusing to overwrite {NORM}")
    lo = rows.min(0).astype(np.float32)
    hi = rows.max(0).astype(np.float32)
    span = np.maximum(hi - lo, np.float32(1e-7))
    scale = (np.float32(2.0) / span).astype(np.float32)
    offset = (np.float32(-1.0) - scale * lo).astype(np.float32)
    stats = {
        "min": lo,
        "max": hi,
        "mean": rows.mean(0).astype(np.float32),
        "std": rows.std(0).astype(np.float32),
    }
    lin = LinearNormalizer()
    lin["action"] = SingleFieldLinearNormalizer.create_manual(scale, offset, stats)
    NORM.parent.mkdir(parents=True, exist_ok=True)
    with NORM.open("wb") as handle:
        pickle.dump(lin, handle)


def main() -> None:
    _self_check()
    out_zarr = OUT / "acp_replay_buffer.zarr"
    if out_zarr.exists() or NORM.exists() or STAMP.exists():
        raise SystemExit(f"refusing to overwrite {OUT} or {NORM}")
    field = FixBMoldField(FIELD)
    axis = long_axis(field.vertices)
    source = zarr.open(str(SRC), "r")
    packed = []
    crown_y = []
    for ep_id in range(50):
        group = source["data"][f"episode_{ep_id}"]
        rotation, position = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
        foot, normal = _true_normal(field, position)
        high = foot[:, 1] < np.percentile(foot[:, 1], 15)
        crown_y.append(normal[high, 1])
        time = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        command = np.asarray(group["ts_pose_command_0"], dtype=np.float64)
        stiff = np.asarray(group["stiffness_0"], dtype=np.float64).reshape(-1)
        if not (len(rotation) == len(time) == len(command) == len(stiff)):
            raise SystemExit(f"episode {ep_id} length mismatch")
        packed.append((rotation, normal, time, command, stiff))
        print(f"  closest ep {ep_id:02d}", flush=True)
    flip = float(np.mean(np.concatenate(crown_y))) >= 0.0
    built = []
    masked_steps = 0
    for rotation, normal, time, command, stiff in packed:
        if flip:
            normal = -normal
        carried = _carry_normal(normal, np.r_[_geodesic(rotation), 0.0])
        direction, _angle = e_long_sequence(carried, axis)
        tilt = pack_tilt(rotation, carried, direction)
        error = geodesic_deg(rotation, rotation_from_tilt(carried, direction, tilt))
        if float(np.median(error)) > 1.0:
            raise SystemExit("carried pack does not decode to the tool; zarr not written")
        keep = _leftover(tilt, rotation, time)
        masked_steps += int((~keep).sum())
        built.append((tilt, keep, error, command, stiff))
    print(f"masked samples {masked_steps}", flush=True)
    link_tree(SRC, out_zarr)
    root = zarr.open(str(out_zarr), "a")
    long_blob = np.load(LONG_DT)
    long_idx: dict[int, set[int]] = {}
    for ep, idx in zip(long_blob["flagged_episode"], long_blob["flagged_index"]):
        long_idx.setdefault(int(ep), set()).add(int(idx))
    rows = []
    decode = {}
    for ep_id, (tilt, keep, error, command, stiff) in enumerate(built):
        group = root["data"][f"episode_{ep_id}"]
        like = group["fixb_heading_tilt_0"]
        replace_array(group, "fixb_heading_tilt_0", tilt.astype(np.float32), like)
        replace_array(group, "fixb_heading_tilt_keep_0", keep.astype(np.uint8).reshape(-1, 1))
        decode[str(ep_id)] = {
            "median_deg": float(np.median(error)),
            "max_deg": float(np.max(error)),
            "masked": int((~keep).sum()),
        }
        if ep_id not in EXCLUDE:
            use = _kept(group, ep_id, long_idx, ~keep)
            if np.any(use):
                rows.append(
                    np.column_stack(
                        [command[use, 0], command[use, 1], command[use, 2], stiff[use], tilt[use]]
                    )
                )
        print(f"  wrote ep {ep_id:02d}", flush=True)
    got = np.concatenate(rows)
    _fit(got)
    digest = _sha(NORM)
    report = {
        "source": str(SRC),
        "zarr": str(out_zarr),
        "crown_flip": flip,
        "masked_samples": masked_steps,
        "kept_normalizer_rows": int(len(got)),
        "decode_deg": decode,
        "normalizer": str(NORM),
        "sha256": digest,
        "columns": ["v1", "v2", "delta_n", "stiffness", "sin_psi", "cos_psi", "a", "b"],
    }
    root.attrs["heading_tilt_carry"] = {
        "version": "heading_tilt_carry_v1",
        "normal_carry_deg": 20.0,
        "tool_still_deg": 20.0,
        "mask": "fixb_heading_tilt_keep_0 is 0 on the later sample of a >45° chart step whose tool geodesic is <=45°",
        "normalizer_sha256": digest,
    }
    STAMP.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("masked_samples", "kept_normalizer_rows", "sha256")}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
