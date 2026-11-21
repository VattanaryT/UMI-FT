#!/usr/bin/env python3
"""Add fixb_heading_tilt_0 beside the causal-foot zarr. Does not modify that zarr.

Hardlinks every existing chunk. The only new array is (sin ψ, cos ψ, a, b).
(a, b) is the tilt rotvec in the frame after Rz(ψ). Labels are not clipped.
The normalizer and the p99.5 runtime clip are fit on the kept training rows.
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

from fixb_build_datasets import link_tree
from fixb_field import FixBMoldField
from fixb_labels import pose7_to_Rp
from fixb_smooth_frame_labels import EXCLUDE, LONG_DT
from fixb_tcp_track_clean import _kept
from heading_tilt import (
    COLUMNS,
    _self_check,
    e_long_sequence,
    geodesic_deg,
    long_axis,
    pack_tilt,
    rotation_from_tilt,
)
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)

ROOT = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
SRC = ROOT / "r4vt_causal_foot" / "acp_replay_buffer.zarr"
FIELD = ROOT / "shared" / "fixb_field.npz"
OUT = ROOT / "r4vt_heading_tilt"
NORM = (
    Path(__file__).resolve().parent
    / "assets"
    / "oct_v35_h4s_r4vt_action_normalizer_minmax_heading_tilt.pkl"
)
STAMP = NORM.with_suffix(".json")
YAML = Path(__file__).resolve().parent / "config" / "task" / "umift_oct_fixb_r4vt_heading_tilt.yaml"
SBATCH = Path(__file__).resolve().parent / "fixb_train_oct_v35_r4vt_heading_tilt_e200_klone.sbatch"
OLD_NORM = NORM.with_name("oct_v35_h4s_r4vt_action_normalizer_minmax_causal_foot.pkl")
OLD_SHA = "14308b72d55600b4359dfa67d10aedab378e42ad26f2bbcac988eed4d97a1ac2"
GT_EPISODES = (0, 9, 31)
ABORT_MEDIAN_DEG = 1.0


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


def _patch_sha(digest: str) -> None:
    for path in (YAML, SBATCH):
        text = path.read_text()
        if "PENDING" not in text:
            raise SystemExit(f"{path} has no PENDING sha")
        path.write_text(text.replace("PENDING", digest, 1))


def _round_trip(rotation, normal, direction, tilt) -> tuple[np.ndarray, np.ndarray]:
    back = rotation_from_tilt(normal, direction, tilt)
    error = geodesic_deg(rotation, back)
    residual = np.einsum("nji,njk->nik", back, rotation)
    twist = np.abs(Rotation.from_matrix(residual).as_rotvec()[:, 2])
    return error, np.degrees(twist)


def _summarize(error: np.ndarray, twist: np.ndarray) -> dict:
    return {
        "n": int(error.size),
        "geodesic_deg_median": float(np.median(error)),
        "geodesic_deg_p90": float(np.percentile(error, 90)),
        "geodesic_deg_max": float(np.max(error)),
        "dropped_twist_deg_median": float(np.median(twist)),
        "dropped_twist_deg_p90": float(np.percentile(twist, 90)),
        "dropped_twist_deg_max": float(np.max(twist)),
    }


def _rewrite_heading(out_zarr: Path) -> None:
    """Replace fixb_heading_tilt_0 with the swing-twist pack. Leaves the normalizer."""
    if not out_zarr.exists():
        raise SystemExit(f"missing {out_zarr}")
    if _sha(OLD_NORM) != OLD_SHA:
        raise SystemExit("causal-foot normalizer sha changed; not touching it")
    field = FixBMoldField(FIELD)
    axis = long_axis(field.vertices)
    root = zarr.open(str(out_zarr), "a")
    crown_y = []
    rows = []
    for ep_id in range(50):
        group = root["data"][f"episode_{ep_id}"]
        rotation, position = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
        foot, normal = _true_normal(field, position)
        high = foot[:, 1] < np.percentile(foot[:, 1], 15)
        crown_y.append(normal[high, 1])
        rows.append((rotation, normal, foot))
    flip = float(np.mean(np.concatenate(crown_y))) >= 0.0
    table = {}
    for ep_id, (rotation, normal, _foot) in enumerate(rows):
        if flip:
            normal = -normal
        direction, _angle = e_long_sequence(normal, axis)
        tilt = pack_tilt(rotation, normal, direction)
        group = root["data"][f"episode_{ep_id}"]
        old = group["fixb_heading_tilt_0"]
        if "fixb_delta_R_rot6d_0" not in group:
            raise SystemExit(f"episode {ep_id} lost delta_R")
        compressor = old.compressor
        del group["fixb_heading_tilt_0"]
        group.create_dataset(
            "fixb_heading_tilt_0",
            data=tilt.astype(np.float32),
            chunks=tilt.shape,
            dtype=np.float32,
            compressor=compressor,
        )
        if ep_id in GT_EPISODES:
            stored = np.asarray(group["fixb_heading_tilt_0"], dtype=np.float64)
            table[ep_id] = _summarize(*_round_trip(rotation, normal, direction, stored))
    if STAMP.exists():
        stamp = json.loads(STAMP.read_text())
        stamp["gt_decode_no_clip"] = {str(k): v for k, v in table.items()}
        stamp["pack"] = "swing_twist"
        STAMP.write_text(json.dumps(stamp, indent=2))
    print(json.dumps({str(k): v for k, v in table.items()}, indent=2), flush=True)


def main() -> None:
    _self_check()
    out_zarr = OUT / "acp_replay_buffer.zarr"
    if "--rewrite-heading" in sys.argv:
        _rewrite_heading(out_zarr)
        return
    if out_zarr.exists() or NORM.exists() or STAMP.exists():
        raise SystemExit(f"refusing to overwrite {OUT} or {NORM}")
    if _sha(OLD_NORM) != OLD_SHA:
        raise SystemExit("causal-foot normalizer sha changed; not touching it")
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
        command = np.asarray(group["ts_pose_command_0"], dtype=np.float64)
        stiff = np.asarray(group["stiffness_0"], dtype=np.float64).reshape(-1)
        if not (len(rotation) == len(command) == len(stiff)):
            raise SystemExit(f"episode {ep_id} length mismatch")
        packed.append(
            {
                "rotation": rotation,
                "normal": normal,
                "command": command,
                "stiff": stiff,
            }
        )
        print(f"  closest ep {ep_id:02d}", flush=True)
    crown_mean_y = float(np.mean(np.concatenate(crown_y)))
    flip = crown_mean_y >= 0.0
    built = []
    carried = 0
    for item in packed:
        normal = -item["normal"] if flip else item["normal"]
        direction, angle = e_long_sequence(normal, axis)
        carried += int(np.count_nonzero(angle < 20.0))
        tilt = pack_tilt(item["rotation"], normal, direction)
        error, twist = _round_trip(item["rotation"], normal, direction, tilt)
        built.append((tilt, error, twist, normal, direction))
    for ep_id in GT_EPISODES:
        pilot = _summarize(built[ep_id][1], built[ep_id][2])
        print(f"episode {ep_id} before write", json.dumps(pilot), flush=True)
        if pilot["geodesic_deg_median"] > ABORT_MEDIAN_DEG:
            raise SystemExit(f"episode {ep_id} round trip is not near 0; zarr not written")
    link_tree(SRC, out_zarr)
    root = zarr.open(str(out_zarr), "a")
    rows = []
    magnitudes = []
    long_blob = np.load(LONG_DT)
    long_idx: dict[int, set[int]] = {}
    for ep, idx in zip(long_blob["flagged_episode"], long_blob["flagged_index"]):
        long_idx.setdefault(int(ep), set()).add(int(idx))
    gt = {}
    for ep_id, (tilt, error, twist, _normal, _direction) in enumerate(built):
        group = root["data"][f"episode_{ep_id}"]
        if "fixb_heading_tilt_0" in group:
            raise SystemExit(f"episode {ep_id} already has fixb_heading_tilt_0")
        like = group["fixb_delta_R_rot6d_0"]
        group.create_dataset(
            "fixb_heading_tilt_0",
            data=tilt.astype(np.float32),
            chunks=tilt.shape,
            dtype=np.float32,
            compressor=like.compressor,
        )
        if ep_id in GT_EPISODES:
            gt[str(ep_id)] = _summarize(error, twist)
        if ep_id not in EXCLUDE:
            use = _kept(group, ep_id, long_idx, np.zeros(len(tilt), dtype=bool))
            if np.any(use):
                command = packed[ep_id]["command"]
                stiff = packed[ep_id]["stiff"]
                rows.append(
                    np.column_stack(
                        [
                            command[use, 0],
                            command[use, 1],
                            command[use, 2],
                            stiff[use],
                            tilt[use],
                        ]
                    )
                )
                magnitudes.append(np.linalg.norm(tilt[use, 2:4], axis=1))
        print(f"  wrote ep {ep_id:02d}", flush=True)
    got = np.concatenate(rows)
    magnitude = np.concatenate(magnitudes)
    p995 = float(np.percentile(magnitude, 99.5))
    _fit(got)
    digest = _sha(NORM)
    if _sha(OLD_NORM) != OLD_SHA:
        raise SystemExit("causal-foot normalizer changed during the write")
    report = {
        "columns": list(COLUMNS),
        "kept_rows": int(len(got)),
        "carried_e_long_samples": carried,
        "crown_normal_y": crown_mean_y,
        "mesh_normals_outward_on_crown": not flip,
        "long_axis": axis.tolist(),
        "tilt_p995_rad": p995,
        "tilt_p995_deg": float(np.degrees(p995)),
        "labels_clipped": False,
        "runtime_clip_rad": p995,
        "runtime_slew_deg": 3.0,
        "gt_decode_no_clip": gt,
        "normalizer": str(NORM),
        "sha256": digest,
        "source": str(SRC),
        "min": got.min(0).tolist(),
        "max": got.max(0).tolist(),
    }
    root.attrs["heading_tilt"] = {
        "version": "heading_tilt_v1",
        "columns": ["sin_psi", "cos_psi", "tilt_a", "tilt_b"],
        "psi_zero": "mold long axis toward +X, projected onto the true outward normal",
        "tilt_frame": "rotvec (a, b, 0) in the tool frame after Rz(psi); tool z is inward",
        "tilt_p995_rad": p995,
        "tilt_p995_deg": float(np.degrees(p995)),
        "labels_clipped": False,
        "runtime_slew_deg": 3.0,
        "normalizer_sha256": digest,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    STAMP.write_text(json.dumps(report, indent=2))
    _patch_sha(digest)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    sys.exit(main())
