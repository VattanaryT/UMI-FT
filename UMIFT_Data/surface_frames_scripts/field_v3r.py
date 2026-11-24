#!/usr/bin/env python3
"""Field v2 inputs and position. Rotation is the query-TCP delta, as in r1rel.

The zarr stores each row's absolute tool rot6d. The dataset turns a horizon into
ΔR = R_tcp(t).T @ R_tcp(t+k). v1, v2, δn, and stiffness are the field_v2 arrays.
Does not submit training.
"""

from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS), str(ROOT / "PyriteML"), str(ROOT / "PyriteConfig"), str(ROOT / "PyriteUtility"), str(ROOT)]

import zarr  # noqa: E402

from diffusion_policy.common.sampler_umift import sparse_action_indices_from_rgb_timestamps  # noqa: E402
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer  # noqa: E402
from field_v3r_dataset import action_v3r_to_action_sample  # noqa: E402
from fixb_build_datasets import link_tree, replace_array  # noqa: E402
from fixb_labels import pose7_to_Rp, so3_to_rot6d_rows  # noqa: E402
from fixb_smooth_frame_labels import ACTION_DS, ACTION_H, EXCLUDE, LONG_DT  # noqa: E402
from fixb_tcp_track_clean import _queries  # noqa: E402
from mold_aware_runtime import rot6d_rows_to_R  # noqa: E402

DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
SRC = DATA / "r4vt_field_v2_k3000/acp_replay_buffer.zarr"
OUT = DATA / "r4vt_field_v3r"
NORM = SCRIPTS / "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_field_v3r.pkl"
YAML = SCRIPTS / "config/task/umift_oct_fixb_r4vt_field_v3r.yaml"
SBATCH = SCRIPTS / "fixb_train_oct_v35_r4vt_field_v3r_e200_klone.sbatch"
K = 3000.0
SAME = ("fixb_v_par_0", "fixb_delta_n_0", "stiffness_0", "ts_pose_command_0")


def _long_idx() -> dict:
    blob = np.load(LONG_DT)
    out: dict[int, set[int]] = {}
    for ep, idx in zip(blob["flagged_episode"], blob["flagged_index"]):
        out.setdefault(int(ep), set()).add(int(idx))
    return out


def _fit(rows: np.ndarray) -> str:
    if NORM.exists():
        raise SystemExit(f"refusing to overwrite {NORM}")
    lo = rows.min(0).astype(np.float32)
    hi = rows.max(0).astype(np.float32)
    span = np.maximum(hi - lo, np.float32(1e-7))
    scale = (np.float32(2.0) / span).astype(np.float32)
    offset = (np.float32(-1.0) - scale * lo).astype(np.float32)
    scale[3] = np.float32(2.0e7)
    offset[3] = np.float32(-1.0) - scale[3] * np.float32(K)
    stats = {
        "min": lo, "max": hi,
        "mean": rows.mean(0).astype(np.float32),
        "std": rows.std(0).astype(np.float32),
    }
    lin = LinearNormalizer()
    lin["action"] = SingleFieldLinearNormalizer.create_manual(scale, offset, stats)
    if abs(float(scale[3] * K + offset[3])) > 1e-4 or not np.all(rows[:, 3] == K):
        raise SystemExit("stiffness did not pin to 3000")
    NORM.parent.mkdir(parents=True, exist_ok=True)
    with NORM.open("wb") as handle:
        pickle.dump(lin, handle)
    digest = hashlib.sha256(NORM.read_bytes()).hexdigest()
    return digest


def _stamp(sha: str) -> None:
    for path in (YAML, SBATCH):
        text = path.read_text().replace("SHA_HERE", sha)
        if sha not in text:
            raise SystemExit(f"{path} did not take the normalizer sha")
        path.write_text(text)


def _commands() -> None:
    tar = str(DATA / "r4vt_field_v3r.tar")
    remote = "/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v3.5_fixb_odom_corr"
    code = "/gscratch/scrubbed/vtevy/UMI-FT-causal_foot/UMIFT_Data/surface_frames_scripts"
    host = "vtevy@klone.hyak.uw.edu"
    print("\n".join([
        f"tar -C {DATA} -cf {tar} r4vt_field_v3r",
        f"rsync -P {tar} {host}:{remote}/",
        f"ssh {host} 'tar -xf {remote}/r4vt_field_v3r.tar -C {remote} && rm {remote}/r4vt_field_v3r.tar'",
        f"rm {tar}",
        f"rsync -P {SCRIPTS}/field_v3r_dataset.py {host}:{code}/",
        f"rsync -P {NORM} {host}:{code}/assets/",
        f"rsync -P {YAML} {host}:{code}/config/task/",
        f"rsync -P {SBATCH} {host}:{code}/",
        f"ssh {host} 'sbatch {code}/fixb_train_oct_v35_r4vt_field_v3r_e200_klone.sbatch'",
    ]), flush=True)


def _windows(ep, ep_id: int, long_idx: dict, rotation: np.ndarray):
    robot_t, rgb_t, kept, bad = _queries(ep, ep_id, long_idx)
    v = np.asarray(ep["fixb_v_par_0"], dtype=np.float32)
    dn = np.asarray(ep["fixb_delta_n_0"], dtype=np.float32).reshape(-1)
    stiff = np.asarray(ep["stiffness_0"], dtype=np.float32).reshape(-1)
    blocks = []
    for rgb_id in kept:
        idx, _ = sparse_action_indices_from_rgb_timestamps(
            robot_t, rgb_t, int(rgb_id), ACTION_H, ACTION_DS
        )
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) < ACTION_H or np.any(bad[np.clip(idx, 0, len(bad) - 1)]):
            continue
        idx = idx[:ACTION_H]
        action = np.column_stack([v[idx], dn[idx], stiff[idx], np.asarray(ep["fixb_delta_R_rot6d_0"][idx])])
        base = np.eye(4)
        base[:3, :3] = rotation[int(idx[0])]
        blocks.append(action_v3r_to_action_sample(action, np.zeros((0,)), [0], [base])["sparse"])
    return blocks


def build() -> None:
    out_zarr = OUT / "acp_replay_buffer.zarr"
    if out_zarr.exists() or NORM.exists():
        raise SystemExit(f"refusing to overwrite {out_zarr} or {NORM}")
    link_tree(SRC, out_zarr)
    root = zarr.open(str(out_zarr), "a")
    src = zarr.open(str(SRC), "r")
    root.attrs["label_version"] = "field_v3r_v1"
    root.attrs["delta_R"] = "absolute tool rot6d; training label is R_tcp(t).T @ R_tcp(t+k)"
    long_idx = _long_idx()
    blocks = []
    for ep_id in range(50):
        ep = root["data"][f"episode_{ep_id}"]
        src_ep = src["data"][f"episode_{ep_id}"]
        rotation, _position = pose7_to_Rp(np.asarray(ep["ts_pose_fb_0"], dtype=np.float64))
        absolute = so3_to_rot6d_rows(rotation).astype(np.float32)
        replace_array(ep, "fixb_delta_R_rot6d_0", absolute, src_ep["fixb_delta_R_rot6d_0"])
        for key in SAME:
            if not np.array_equal(np.asarray(ep[key]), np.asarray(src_ep[key])):
                raise SystemExit(f"episode {ep_id} {key} changed")
        ep.attrs["label_version"] = "field_v3r_v1"
        if ep_id not in EXCLUDE:
            blocks.extend(_windows(ep, ep_id, long_idx, rotation))
        print(f"[field_v3r] ep {ep_id:02d}", flush=True)
    rows = np.concatenate(blocks).astype(np.float32)
    sha = _fit(rows)
    print(f"[field_v3r] normalizer {NORM} {sha} rows {len(rows)}", flush=True)
    _stamp(sha)
    _commands()


def _self_check() -> None:
    yaw = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    absolute = so3_to_rot6d_rows(np.stack([np.eye(3), yaw])).astype(np.float32)
    action = np.zeros((2, 10), dtype=np.float32)
    action[:, 4:10] = absolute
    base = np.eye(4)
    got = action_v3r_to_action_sample(action, np.zeros((0,)), [0], [base])["sparse"]
    if not np.array_equal(got[:, :4], action[:, :4]):
        raise RuntimeError("position columns moved")
    back = rot6d_rows_to_R(got[:, 4:10].astype(np.float64))
    if not np.allclose(back[1], yaw, atol=1e-5):
        raise RuntimeError("query delta did not decode")


if __name__ == "__main__":
    _self_check()
    build()
