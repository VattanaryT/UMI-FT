#!/usr/bin/env python3
"""stringer_v4 through the liftos label path at action horizon 64.

Reads the combined tag-fixture zarr. Does not open it for writing. Does not
touch the v3.5 liftos zarr, its normalizer, the training frames file, or the
earlier horizon-24 sv4 outputs.
"""

from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [
    str(SCRIPTS),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT),
]

import zarr  # noqa: E402

from causal_foot import CausalFoot  # noqa: E402
from diffusion_policy.common.sampler_umift import sparse_action_indices_from_rgb_timestamps  # noqa: E402
from diffusion_policy.model.common.normalizer import (  # noqa: E402
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from field_v2 import carry_frames, mold_frame, reorient, smooth_force  # noqa: E402
from field_v2n import blend, target_clearance  # noqa: E402
from field_v2n_v3r_liftos_dataset import action_liftos_to_action_sample  # noqa: E402
from field_v3r_dataset import action_v3r_to_action_sample  # noqa: E402
from fixb_build_datasets import MOLD_KEYS, link_tree, replace_array  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import fixb_pose7_command, pose7_to_Rp, so3_to_rot6d_rows  # noqa: E402
from fixb_tcp_track import project_tcp_velocity  # noqa: E402
from fixb_tcp_track_clean import _queries  # noqa: E402
from build_liftos import isolated  # noqa: E402

SRC = Path("/mnt/data/umidata/ft/datasets/stringer_v4/acp_replay_buffer.zarr")
DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v4_fixb")
OUT = DATA / "r4vt_field_v2n_v3r_liftos_sv4b/acp_replay_buffer.zarr"
FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
TRAIN_FRAMES = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_field_v2_k3000/frames_pca_r020.npz"
)
LIFTOS = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_field_v2n_v3r_liftos/acp_replay_buffer.zarr"
)
NORM = SCRIPTS / "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_field_v2n_v3r_liftos_sv4b.pkl"
LIFT_NPZ = SCRIPTS / "assets/lift_isolated_ends_sv4b.npz"
LONG_NPZ = DATA / "shared/long_dt_gt_100ms_sv4b.npz"
SV4_OUT = DATA / "r4vt_field_v2n_v3r_liftos_sv4/acp_replay_buffer.zarr"
K = 3000.0
ACTION_H = 64
ACTION_DS = 1
DROP = {7, 14, 30}
PROTECTED = {
    "ce85fae23a90d3d7fc29d2bc357693b17254fbadea1af716ba2f10af9be394af",
    "4d7c4febbafb93293cc2c470960ddeec6dc02731cebd90ea4ee0b11785f73ac2",
    "55a99fbffed1c47e018a4638e044208021bf833b92c10f98d22c957e5eae9f30",
    "743c83c80bcb7d5a85bb3441d80e4e4adc8b3010070f84378de40ba3c828040f",
    "c93f000b9616cb2409a1c6cc8dfeac43db231824f951776c14e239a83645cdb4",
}


def _long_table(root) -> dict[int, set[int]]:
    flagged_ep = []
    flagged_i = []
    flagged_dt = []
    out: dict[int, set[int]] = {}
    for name in root["data"].keys():
        ep_id = int(name.split("_")[-1])
        t = np.asarray(root["data"][name]["robot_time_stamps_0"]).reshape(-1)
        dt = np.diff(t, prepend=t[0])
        hit = np.flatnonzero(dt > 0.1)
        out[ep_id] = set(int(i) for i in hit)
        for i in hit:
            flagged_ep.append(ep_id)
            flagged_i.append(int(i))
            flagged_dt.append(float(dt[i]))
    LONG_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        LONG_NPZ,
        threshold_s=np.float64(0.1),
        flagged_episode=np.asarray(flagged_ep, dtype=np.int64),
        flagged_index=np.asarray(flagged_i, dtype=np.int64),
        flagged_dt_s=np.asarray(flagged_dt, dtype=np.float64),
    )
    return out


def _fit(raw: np.ndarray, delta: np.ndarray) -> str:
    if NORM.exists():
        raise SystemExit(f"refusing to overwrite {NORM}")
    def pack(rows):
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
        return scale, offset, stats
    scale, offset, stats = pack(raw)
    scale[3] = np.float32(2.0e7)
    offset[3] = np.float32(-1.0) - scale[3] * np.float32(K)
    d_scale, d_offset, d_stats = pack(delta)
    scale[4:10] = d_scale[4:10]
    offset[4:10] = d_offset[4:10]
    for key in stats:
        stats[key][4:10] = d_stats[key][4:10]
    value = float(scale[3] * K + offset[3])
    if abs(value) > 1e-4:
        raise SystemExit(f"stiffness did not pin ({value})")
    lin = LinearNormalizer()
    lin["action"] = SingleFieldLinearNormalizer.create_manual(scale, offset, stats)
    NORM.parent.mkdir(parents=True, exist_ok=True)
    with NORM.open("wb") as handle:
        pickle.dump(lin, handle)
    digest = hashlib.sha256(NORM.read_bytes()).hexdigest()
    if digest in PROTECTED:
        NORM.unlink()
        raise SystemExit(f"sha collided with a protected normalizer {digest}")
    return digest


def _windows(ep, rotation, long_idx, ep_id):
    robot_t, rgb_t, kept, bad = _queries(ep, ep_id, long_idx)
    v = np.asarray(ep["fixb_v_par_0"], dtype=np.float32)
    dn = np.asarray(ep["fixb_delta_n_0"], dtype=np.float32).reshape(-1)
    stiff = np.asarray(ep["stiffness_0"], dtype=np.float32).reshape(-1)
    raw_blocks = []
    delta_blocks = []
    for rgb_id in kept:
        idx, _ = sparse_action_indices_from_rgb_timestamps(
            robot_t, rgb_t, int(rgb_id), ACTION_H, ACTION_DS
        )
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) < ACTION_H or np.any(bad[np.clip(idx, 0, len(bad) - 1)]):
            continue
        idx = idx[:ACTION_H]
        action = np.column_stack(
            [v[idx], dn[idx], stiff[idx], np.asarray(ep["fixb_delta_R_rot6d_0"][idx])]
        ).astype(np.float32)
        raw_blocks.append(action)
        base = np.eye(4)
        base[:3, :3] = rotation[int(idx[0])]
        delta_blocks.append(
            action_v3r_to_action_sample(action, np.zeros((0,)), [0], [base])["sparse"]
        )
    return raw_blocks, delta_blocks


def frames_path_of(out: Path) -> Path:
    return out.parent / "frames_pca_r020.npz"


def _frame_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r = np.einsum("nji,njk->nik", a, b)
    cos = np.clip((np.trace(r, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def main() -> None:
    if OUT.exists() or NORM.exists() or LIFT_NPZ.exists():
        raise SystemExit(f"refusing to overwrite {OUT} or {NORM}")
    if not SRC.is_dir():
        raise SystemExit(f"missing source {SRC}")
    if not TRAIN_FRAMES.is_file():
        raise SystemExit(f"missing training frames {TRAIN_FRAMES}")
    if frames_path_of(OUT) == TRAIN_FRAMES:
        raise SystemExit("refusing to write the training frames file")
    link_tree(SRC, OUT)
    root = zarr.open(str(OUT), "a")
    spec = zarr.open_group(str(LIFTOS), "r").attrs["field_v2_axes"]
    origin = np.asarray(spec["origin"], dtype=np.float64)
    axes = np.asarray(spec["axes"], dtype=np.float64)
    trained = np.load(TRAIN_FRAMES)
    frames = np.asarray(trained["frames"], dtype=np.float64)
    again = reorient(frames, axes)
    # The training file is already PCA-reoriented. A second pass should be a no-op.
    moved = _frame_angle_deg(frames, again)
    print(
        f"[sv4b] reorient-again median {np.median(moved):.3f} deg "
        f"max {moved.max():.3f} deg",
        flush=True,
    )
    frames_path = frames_path_of(OUT)
    np.savez(
        frames_path,
        frames=frames.astype(np.float32),
        radius_m=np.float32(trained["radius_m"]) if "radius_m" in trained.files else np.float32(0.02),
    )
    # Same PCA as the October field. Confirm it matches a fresh fit of that mesh.
    field_np = np.load(FIELD)
    apron = (
        np.asarray(field_np["is_apron"]).astype(bool)
        if "is_apron" in field_np.files
        else np.zeros(len(field_np["vertices"]), dtype=bool)
    )
    fit_origin, fit_axes = mold_frame(np.asarray(field_np["vertices"]), apron)
    if np.linalg.norm(fit_origin - origin) > 1e-6 or np.linalg.norm(fit_axes - axes) > 1e-5:
        raise SystemExit("liftos PCA does not match a fresh fit of the mold mesh")
    root.attrs["label_version"] = "field_v2n_v3r_liftos_sv4b_v1"
    root.attrs["delta_n"] = "field_v2n blend, K=3000"
    root.attrs["delta_R"] = "absolute tool rot6d; training label is R_tcp(t).T @ R_tcp(t+k)"
    root.attrs["source_zarr"] = str(SRC)
    root.attrs["field_v2_axes"] = {
        "origin": origin.tolist(),
        "axes": axes.tolist(),
        "end": "right_base",
    }
    root.attrs["dropped_episodes"] = {
        "7": "review stringer_v4/7 recording 20.58.25",
        "14": "review stringer_v4/14 recording 21.09.43 corrupted RGB",
        "30": "review stringer_v4/30 recording 21.53.07",
    }
    tracker = CausalFoot(FIELD, frames_path)
    mesh = FixBMoldField(FIELD)
    long_idx = _long_table(root)
    raw_rows = []
    delta_rows = []
    lift_ep = []
    lift_i = []
    names = sorted(root["data"].keys(), key=lambda s: int(s.split("_")[-1]))
    for name in names:
        ep_id = int(name.split("_")[-1])
        ep = root["data"][name]
        robot_t = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        rgb_t = np.asarray(ep["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
        rotation, position = pose7_to_Rp(np.asarray(ep["ts_pose_fb_0"], dtype=np.float64))
        foot, frame = tracker.track(position)
        frame = carry_frames(reorient(frame, axes), rotation)
        outward = -frame[:, :, 2]
        force = smooth_force(ep, robot_t)
        fn = np.einsum("ni,ni->n", np.einsum("nij,nj->ni", rotation, force), outward)
        force_dn = -fn / K
        _arrived, _foot, clearance = target_clearance(position, frame, robot_t, rgb_t, mesh)
        depth = blend(force_dn, clearance, fn)
        velocity, dt = project_tcp_velocity(frame, position, robot_t, rgb_t)
        absolute = so3_to_rot6d_rows(rotation).astype(np.float32)
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
        for key in MOLD_KEYS:
            replace_array(ep, key, pack[key])
        replace_array(ep, "ts_pose_command_0", fixb_pose7_command(velocity, depth).astype(np.float32))
        replace_array(ep, "fixb_delta_R_rot6d_0", absolute)
        stored = ep["stiffness_0"]
        replace_array(ep, "stiffness_0", np.full(stored.shape, np.float32(K)))
        ep.attrs["label_version"] = "field_v2n_v3r_liftos_sv4b_v1"
        t_fn, fn_row = robot_t, fn
        ends = isolated(np.abs(fn_row) >= 5.0, t_fn)
        for index in ends:
            lift_ep.append(ep_id)
            lift_i.append(int(index))
        if ep_id not in DROP:
            raw, delta = _windows(ep, rotation, long_idx, ep_id)
            raw_rows.extend(raw)
            delta_rows.extend(delta)
        print(f"[sv4b] ep {ep_id:02d} n {len(position)} lifts {len(ends)}", flush=True)
    if not raw_rows:
        raise SystemExit("no training rows")
    # The liftos sample keeps v and δn and rewrites only the rotation block.
    probe = action_liftos_to_action_sample(
        raw_rows[0][:2], np.zeros((0,)), [0], [np.eye(4)]
    )["sparse"]
    if not np.array_equal(probe[:, :4], raw_rows[0][:2, :4]):
        raise SystemExit("liftos sample moved v or δn")
    digest = _fit(np.concatenate(raw_rows), np.concatenate(delta_rows))
    np.savez(
        LIFT_NPZ,
        episode=np.asarray(lift_ep, dtype=np.int64),
        index=np.asarray(lift_i, dtype=np.int64),
    )
    print(f"[sv4b] zarr {OUT}", flush=True)
    print(f"[sv4b] frames {frames_path}", flush=True)
    print(f"[sv4b] normalizer {NORM} {digest} rows {sum(len(r) for r in raw_rows)}", flush=True)
    print(f"[sv4b] lifts {len(lift_i)} long_dt {LONG_NPZ}", flush=True)
    print(f"[sv4b] left untouched {SV4_OUT}", flush=True)


if __name__ == "__main__":
    main()
