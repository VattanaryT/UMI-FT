"""1-tilt action: v1, v2, δn, stiffness, a.

Heading is the field axis snapped to the near side of the stringer:

    R = R_zero(e_long, n) @ Rz(0 or π) @ exp([a, 0, 0])

0 when n_out points toward table +Z, π when it points away. The side holds
until n·Z crosses ±SIDE_Z, and an episode starts on its first clear sample.
b is 0. a is recomputed in that frame. +a leans tool z toward the flange.
"""

from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from heading_tilt import _Rz, e_long_sequence, frame_zero, long_axis, rotation_from_tilt

COLUMNS = ("v1", "v2", "delta_n", "stiffness", "tilt_a")
# Operator side of the table. The stringer runs along +X, so the walls face ±Z.
NEAR = np.array([0.0, 0.0, 1.0])
# Hold the side through the crown. 0.4 has the lowest crown p90; walls do not move.
SIDE_Z = 0.4
ZARR = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_causal_foot/acp_replay_buffer.zarr"
)
FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)


def heading_half(normal_out, previous=None) -> np.ndarray:
    """(N,) heading, 0 or π. Hold the side until n·Z crosses ±SIDE_Z.

    ``previous`` is the side carried in from the last step. None starts the
    episode on its first clear sample, including the samples before that.
    """
    z = np.asarray(normal_out, dtype=np.float64).reshape(-1, 3) @ NEAR
    side = None if previous is None else float(previous)
    if side is None:
        clear = np.flatnonzero(np.abs(z) > SIDE_Z)
        side = np.pi if len(clear) and z[int(clear[0])] < 0.0 else 0.0
    half = np.empty(len(z), dtype=np.float64)
    for i, value in enumerate(z):
        if value > SIDE_Z:
            side = 0.0
        elif value < -SIDE_Z:
            side = np.pi
        half[i] = side
    return half


def pack_tilt1(rotation, normal_out, e_long, previous=None) -> np.ndarray:
    """(N,) lean in the snapped frame. b is dropped."""
    half = heading_half(normal_out, previous)
    zero = frame_zero(e_long, normal_out)
    relative = np.einsum(
        "nji,njk->nik", zero, np.asarray(rotation, dtype=np.float64).reshape(-1, 3, 3)
    )
    swung = np.einsum("nji,njk->nik", _Rz(half), relative)
    return np.arctan2(swung[:, 2, 1] - swung[:, 1, 2], swung[:, 1, 1] + swung[:, 2, 2])


def rotation_from_tilt1(normal_out, e_long, tilt_a, previous=None) -> np.ndarray:
    """R_zero @ Rz(0 or π) @ exp([a, 0, 0]). Same side rule as the labels."""
    angle = np.asarray(tilt_a, dtype=np.float64).reshape(-1)
    half = heading_half(normal_out, previous)
    tilt = np.column_stack([np.sin(half), np.cos(half), angle, np.zeros(len(angle))])
    return rotation_from_tilt(normal_out, e_long, tilt)


def _pose7_R(pose7: np.ndarray) -> np.ndarray:
    """Same quat order as fixb_labels.pose7_to_Rp. Numpy only, so training does not import open3d."""
    p = np.asarray(pose7, dtype=np.float64).reshape(-1, 7)
    return Rotation.from_quat(p[:, [4, 5, 6, 3]]).as_matrix()


def action_tilt1_from_group(group, axis) -> np.ndarray:
    """(T, 5) from a causal-foot episode. Same rows as ts_pose_command_0."""
    rotation = _pose7_R(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
    normal = np.asarray(group["mold_normal_0"], dtype=np.float64)
    direction, _angle = e_long_sequence(normal, axis)
    command = np.asarray(group["ts_pose_command_0"], dtype=np.float64)
    stiff = np.asarray(group["stiffness_0"], dtype=np.float64).reshape(-1)
    lean = pack_tilt1(rotation, normal, direction)
    n = min(len(command), len(stiff), len(lean))
    return np.column_stack(
        [command[:n, 0], command[:n, 1], command[:n, 2], stiff[:n], lean[:n]]
    ).astype(np.float32)


def action_tilt1_to_action_sample(action_sparse, action_dense, id_list, base_pose):
    """Pass the 5-d vector through. It is not an SE3 pose."""
    del id_list, base_pose
    if np.asarray(action_sparse).shape[-1] != 5:
        raise ValueError(f"tilt1 action dim {np.asarray(action_sparse).shape[-1]}, expected 5")
    if len(action_dense) > 0:
        raise NotImplementedError
    return {"sparse": action_sparse, "dense": {}}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kept_rows(zarr_path=ZARR, field_path=FIELD) -> np.ndarray:
    """Training rows: episodes 35 and 44, bad spans, and long dt dropped."""
    from fixb_smooth_frame_labels import EXCLUDE, LONG_DT
    from fixb_tcp_track_clean import _kept

    import zarr

    axis = long_axis(np.load(field_path)["vertices"])
    root = zarr.open(str(zarr_path), "r")
    blob = np.load(LONG_DT)
    long_idx: dict[int, set[int]] = {}
    for ep, idx in zip(blob["flagged_episode"], blob["flagged_index"]):
        long_idx.setdefault(int(ep), set()).add(int(idx))
    rows = []
    for ep_id in range(50):
        if ep_id in EXCLUDE:
            continue
        group = root["data"][f"episode_{ep_id}"]
        action = action_tilt1_from_group(group, axis)
        use = _kept(group, ep_id, long_idx, np.zeros(len(action), dtype=bool))
        if np.any(use):
            rows.append(action[use])
    return np.concatenate(rows) if rows else np.zeros((0, 5), dtype=np.float32)


def fit_normalizer(rows: np.ndarray, path: Path) -> str:
    """Min-max to [-1, 1], same as the causal-foot pickle. Refuses to overwrite."""
    from diffusion_policy.model.common.normalizer import (
        LinearNormalizer,
        SingleFieldLinearNormalizer,
    )

    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        raise SystemExit(f"refusing to overwrite {path}")
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(lin, handle)
    return _sha(path)


def _patch_sha(digest: str) -> None:
    root = Path(__file__).resolve().parent
    for path in (
        root / "config" / "task" / "umift_oct_fixb_r4vt_tilt1.yaml",
        root / "fixb_train_oct_v35_r4vt_tilt1_e200_klone.sbatch",
    ):
        text = path.read_text()
        if "PENDING" not in text:
            raise SystemExit(f"{path} has no PENDING sha")
        path.write_text(text.replace("PENDING", digest, 1))


def main() -> None:
    out = (
        Path(__file__).resolve().parent
        / "assets"
        / "oct_v35_h4s_r4vt_action_normalizer_minmax_tilt1.pkl"
    )
    rows = kept_rows()
    digest = fit_normalizer(rows, out)
    _patch_sha(digest)
    print(
        f"rows={len(rows)} sha256={digest} "
        f"min={rows.min(0).tolist()} max={rows.max(0).tolist()}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
