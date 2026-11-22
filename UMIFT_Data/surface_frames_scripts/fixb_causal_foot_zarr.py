#!/usr/bin/env python3
"""Causal-foot labels next to track_clean. Does not modify that zarr or its pickle.

Hardlinks the cleaned October zarr, then replaces the foot, tangent velocity,
press depth, and ΔR. The 10 training dims are v1, v2, δn, stiffness, ΔR(6).
The other 17 action slots stay the constants the dataset already writes.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import zarr

from causal_foot import CausalFoot
from fixb_build_datasets import MOLD_KEYS, link_tree, replace_array
from fixb_labels import (
    delta_R_from_field_tool,
    fixb_pose7_command,
    pose7_to_Rp,
    so3_to_rot6d_rows,
)
from fixb_smooth_frame_labels import EXCLUDE, LONG_DT
from fixb_tcp_track import project_tcp_velocity
from fixb_tcp_track_clean import TRACK, _kept
from fixb_tcp_velocity_labels import _dn
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)

ROOT = TRACK.parents[1]
SRC = ROOT / "r4vt_smooth_tcp_track_clean" / "acp_replay_buffer.zarr"
FIELD = ROOT / "shared" / "fixb_field.npz"
OUT = ROOT / "r4vt_causal_foot"
NORM = (
    Path(__file__).resolve().parent
    / "assets"
    / "oct_v35_h4s_r4vt_action_normalizer_minmax_causal_foot.pkl"
)
COLS = ("v1", "v2", "delta_n", "stiffness", "dR0", "dR1", "dR2", "dR3", "dR4", "dR5")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _kept_rows(root) -> np.ndarray:
    long_blob = np.load(LONG_DT)
    long_idx: dict[int, set[int]] = {}
    for ep, idx in zip(long_blob["flagged_episode"], long_blob["flagged_index"]):
        long_idx.setdefault(int(ep), set()).add(int(idx))
    rows = []
    for ep_id in range(50):
        if ep_id in EXCLUDE:
            continue
        ep = root["data"][f"episode_{ep_id}"]
        n = int(np.asarray(ep["ts_pose_fb_0"]).shape[0])
        use = _kept(ep, ep_id, long_idx, np.zeros(n, dtype=bool))
        if not np.any(use):
            continue
        velocity = np.asarray(ep["fixb_v_par_0"], dtype=np.float64)
        depth = np.asarray(ep["fixb_delta_n_0"], dtype=np.float64).reshape(-1)
        stiff = np.asarray(ep["stiffness_0"], dtype=np.float64).reshape(-1)
        delta_r = np.asarray(ep["fixb_delta_R_rot6d_0"], dtype=np.float64)
        rows.append(
            np.column_stack(
                [velocity[use, 0], velocity[use, 1], depth[use], stiff[use], delta_r[use]]
            )
        )
    return np.concatenate(rows) if rows else np.zeros((0, 10))


def _write_report(got: np.ndarray) -> None:
    digest = _sha(NORM)
    report = {
        "kept_rows": int(len(got)),
        "columns": list(COLS),
        "min": got.min(0).tolist(),
        "max": got.max(0).tolist(),
        "normalizer": str(NORM),
        "sha256": digest,
        "source": str(SRC),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "causal_foot_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    out_zarr = OUT / "acp_replay_buffer.zarr"
    if "--fit-only" in sys.argv:
        if not out_zarr.exists():
            raise SystemExit(f"missing {out_zarr}")
        got = _kept_rows(zarr.open(str(out_zarr), "r"))
        _fit(got)
        _write_report(got)
        return
    if out_zarr.exists() or NORM.exists():
        raise SystemExit(f"refusing to overwrite {OUT} or {NORM}")
    long_blob = np.load(LONG_DT)
    long_idx: dict[int, set[int]] = {}
    for ep, idx in zip(long_blob["flagged_episode"], long_blob["flagged_index"]):
        long_idx.setdefault(int(ep), set()).add(int(idx))
    tracker = CausalFoot(FIELD, FIELD.with_name("fixb_smooth_frames.npz"))
    link_tree(SRC, out_zarr)
    root = zarr.open(str(out_zarr), "a")
    root.attrs["label_version"] = "causal_foot_v1"
    rows = []
    for ep_id in range(50):
        key = f"episode_{ep_id}"
        ep = root["data"][key]
        robot_t = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        rgb_t = np.asarray(ep["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
        rotation, position = pose7_to_Rp(np.asarray(ep["ts_pose_fb_0"], dtype=np.float64))
        virtual = pose7_to_Rp(np.asarray(ep["ts_pose_virtual_target_0"], dtype=np.float64))[1]
        foot, frame = tracker.track(position)
        outward = -frame[:, :, 2]
        velocity, dt = project_tcp_velocity(frame, position, robot_t, rgb_t)
        depth = _dn(virtual, foot, outward)
        delta_r = so3_to_rot6d_rows(delta_R_from_field_tool(frame, rotation))
        curved = tracker.field.query(foot, with_curvature=True)
        pack = {
            "mold_x_s_0": foot.astype(np.float32),
            "mold_R_field_0": frame.reshape(-1, 9).astype(np.float32),
            "mold_R_field_rot6d_0": so3_to_rot6d_rows(frame).astype(np.float32),
            "mold_normal_0": outward.astype(np.float32),
            "mold_signed_distance_0": np.sum((position - foot) * outward, axis=1).astype(np.float32).reshape(-1, 1),
            "mold_curvature_0": np.asarray(curved["curv"], dtype=np.float32),
            "fixb_v_par_0": velocity.astype(np.float32),
            "fixb_delta_n_0": depth.reshape(-1, 1).astype(np.float32),
            "fixb_dt_next_0": dt.reshape(-1, 1).astype(np.float32),
        }
        for name in MOLD_KEYS:
            replace_array(ep, name, pack[name], ep[name])
        replace_array(ep, "ts_pose_command_0", fixb_pose7_command(velocity, depth).astype(np.float32), ep["ts_pose_command_0"])
        replace_array(ep, "fixb_delta_R_rot6d_0", delta_r.astype(np.float32), ep["fixb_delta_R_rot6d_0"])
        ep.attrs["label_version"] = "causal_foot_v1"
        if ep_id not in EXCLUDE:
            use = _kept(ep, ep_id, long_idx, np.zeros(len(position), dtype=bool))
            if np.any(use):
                stiff = np.asarray(ep["stiffness_0"], dtype=np.float64).reshape(-1)
                rows.append(
                    np.column_stack(
                        [velocity[use, 0], velocity[use, 1], depth[use], stiff[use], delta_r[use]]
                    )
                )
        print(f"  ep {ep_id:02d}", flush=True)
    got = np.concatenate(rows) if rows else np.zeros((0, 10))
    _fit(got)
    _write_report(got)


if __name__ == "__main__":
    main()
