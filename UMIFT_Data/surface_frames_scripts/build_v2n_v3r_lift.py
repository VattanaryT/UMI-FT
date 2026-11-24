#!/usr/bin/env python3
"""Write the 0.5 s / 6 N / 3 N δn into a new zarr and fit a new normalizer.

Hardlinks r4vt_field_v3r, then replaces fixb_delta_n_0 and ts_pose_command_0.
Does not submit training.
"""

from __future__ import annotations

import hashlib
import json
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

from diffusion_policy.common.sampler_umift import sparse_action_indices_from_rgb_timestamps  # noqa: E402
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer  # noqa: E402
from field_v2n_v3r_lift_dataset import action_lift_to_action_sample  # noqa: E402
from fixb_build_datasets import link_tree, replace_array  # noqa: E402
from fixb_labels import pose7_to_Rp  # noqa: E402
from fixb_smooth_frame_labels import ACTION_DS, ACTION_H, EXCLUDE, LONG_DT  # noqa: E402
from fixb_tcp_track_clean import _queries  # noqa: E402
from lift_gate import K, OFF_N, ON_N, WINDOW_S, delta_n, hysteresis, isolated  # noqa: E402
from sweep_lift_gate import _rise_time, _switch_time  # noqa: E402
from rot6d import rot6d_rows_to_R  # noqa: E402

DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
SRC = DATA / "r4vt_field_v3r/acp_replay_buffer.zarr"
OUT = DATA / "r4vt_field_v2n_v3r_lift/acp_replay_buffer.zarr"
NORM = SCRIPTS / "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_field_v2n_v3r_lift.pkl"
NPZ = SCRIPTS / "eval/v2n_v3r_lift/delta_n.npz"
TABLE = SCRIPTS / "eval/lift_sweep/table.json"
CACHE = SCRIPTS / "eval/lift_sweep/cache"
YAML = SCRIPTS / "config/task/umift_oct_fixb_r4vt_field_v2n_v3r_lift.yaml"
SBATCH = SCRIPTS / "fixb_train_oct_v35_r4vt_field_v2n_v3r_lift_e200_klone.sbatch"
PROTECTED = {
    "4d7c4febbafb93293cc2c470960ddeec6dc02731cebd90ea4ee0b11785f73ac2",
    "14308b72d55600b4359dfa67d10aedab378e42ad26f2bbcac988eed4d97a1ac2",
    "71bd4cdc7be5a066e786d0b1fb9be526e01d6a6ff1e4ad7dd9c6280ac3709f84",
    "557e4d25a584ea026c253c95134f983dec0d5624dd1dd90652b78cd679630215",
    "e93b87561e762c2d7f1549a883eb18d45e6d50d929df463c43fce4817444e2f5",
    "3f9b16b1b06e4e2c27cba06053da3152c5f12322b35cd5cb94ff744f478d1c41",
    "55a99fbffed1c47e018a4638e044208021bf833b92c10f98d22c957e5eae9f30",
}


def _long_idx() -> dict:
    blob = np.load(LONG_DT)
    out: dict[int, set[int]] = {}
    for ep, idx in zip(blob["flagged_episode"], blob["flagged_index"]):
        out.setdefault(int(ep), set()).add(int(idx))
    return out


def _angle_deg(stored, rotation) -> np.ndarray:
    got = rot6d_rows_to_R(np.asarray(stored, dtype=np.float64))
    tr = np.einsum("nii->n", np.einsum("nji,njk->nik", got, rotation))
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def _labels():
    table = json.loads(TABLE.read_text())
    pick = table["pick"]
    if (pick["window_s"], pick["on_n"], pick["off_n"]) != (WINDOW_S, ON_N, OFF_N):
        raise SystemExit(f"table pick {pick} is not {WINDOW_S} s {ON_N}/{OFF_N}")
    labels = {}
    post = []
    for epi in range(50):
        blob = np.load(CACHE / f"ep_{epi:02d}.npz")
        fn = blob["fn_0.50"]
        clearance = blob["clearance"]
        contact = hysteresis(np.abs(fn), ON_N, OFF_N)
        depth = delta_n(fn, clearance, contact).astype(np.float32)
        labels[epi] = depth
        t = blob["t"]
        for end in isolated(blob["confirmed"].astype(bool), t):
            t_rise = _rise_time(t, clearance, int(end))
            if t_rise is None:
                continue
            t_switch = _switch_time(t, contact, t_rise)
            if t_switch is None:
                continue
            sel = (t > t_switch) & (t <= t_switch + 0.5)
            post.append(depth[sel])
    pooled = np.concatenate(post) if post else np.zeros(0)
    print(
        f"[lift] after gate opens, 0.5 s δn median {np.median(pooled) * 1e3:.3f} mm"
        f" frac>0 {np.mean(pooled > 0):.4f} n {pooled.size}",
        flush=True,
    )
    return labels


def _windows(ep, ep_id, long_idx, rotation, depth):
    robot_t, rgb_t, kept, bad = _queries(ep, ep_id, long_idx)
    v = np.asarray(ep["fixb_v_par_0"], dtype=np.float32)
    stiff = np.asarray(ep["stiffness_0"], dtype=np.float32).reshape(-1)
    rot6 = np.asarray(ep["fixb_delta_R_rot6d_0"], dtype=np.float32)
    blocks = []
    for rgb_id in kept:
        idx, _ = sparse_action_indices_from_rgb_timestamps(
            robot_t, rgb_t, int(rgb_id), ACTION_H, ACTION_DS
        )
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) < ACTION_H or np.any(bad[np.clip(idx, 0, len(bad) - 1)]):
            continue
        idx = idx[:ACTION_H]
        if int(idx.max()) >= len(depth):
            raise SystemExit(f"episode {ep_id} action index past the new δn")
        action = np.column_stack([v[idx], depth[idx], stiff[idx], rot6[idx]])
        base = np.eye(4)
        base[:3, :3] = rotation[int(idx[0])]
        blocks.append(action_lift_to_action_sample(action, np.zeros((0,)), [0], [base])["sparse"])
    return blocks


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
    if abs(float(scale[3] * K + offset[3])) > 1e-4 or not np.all(rows[:, 3] == K):
        raise SystemExit("stiffness did not pin to 3000")
    stats = {
        "min": lo, "max": hi,
        "mean": rows.mean(0).astype(np.float32),
        "std": rows.std(0).astype(np.float32),
    }
    lin = LinearNormalizer()
    lin["action"] = SingleFieldLinearNormalizer.create_manual(scale, offset, stats)
    NORM.parent.mkdir(parents=True, exist_ok=True)
    with NORM.open("wb") as handle:
        pickle.dump(lin, handle)
    digest = hashlib.sha256(NORM.read_bytes()).hexdigest()
    if digest in PROTECTED:
        NORM.unlink()
        raise SystemExit(f"new normalizer collided with {digest}")
    print(
        f"[lift] δn scale {float(scale[2]):.8e} offset {float(offset[2]):.8e}"
        f" rows {len(rows)} sha {digest}",
        flush=True,
    )
    return digest


def _stamp(sha: str) -> None:
    for path in (YAML, SBATCH):
        text = path.read_text().replace("SHA_HERE", sha)
        if text.count(sha) < 1 or "SHA_HERE" in text:
            raise SystemExit(f"{path} did not take the sha")
        path.write_text(text)


def main() -> None:
    if OUT.exists() or NORM.exists() or NPZ.exists():
        raise SystemExit(f"refusing to overwrite {OUT} or {NORM} or {NPZ}")
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    labels = _labels()
    src = zarr.open(str(SRC), "r")
    if "field_v2_axes" not in src.attrs:
        raise SystemExit("v3r zarr has no field_v2_axes")
    guard = np.asarray(src["data/episode_0/fixb_delta_n_0"][:8]).copy()
    long_idx = _long_idx()
    blocks = []
    for ep_id, depth in labels.items():
        ep = src["data"][f"episode_{ep_id}"]
        if len(depth) != len(ep["fixb_delta_n_0"]):
            raise SystemExit(
                f"episode {ep_id} δn {len(depth)} != zarr {len(ep['fixb_delta_n_0'])}"
            )
        rotation, _ = pose7_to_Rp(np.asarray(ep["ts_pose_fb_0"], dtype=np.float64))
        ang = _angle_deg(ep["fixb_delta_R_rot6d_0"], rotation[: len(depth)])
        if float(np.median(ang)) > 1.0:
            raise SystemExit(f"episode {ep_id} stored rot6d is not the tool rotation")
        if ep_id not in EXCLUDE:
            blocks.extend(_windows(ep, ep_id, long_idx, rotation, depth))
        print(f"[lift] windows {ep_id:02d}", flush=True)
    sha = _fit(np.concatenate(blocks).astype(np.float32))
    NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(NPZ, **{f"episode_{i:02d}": labels[i] for i in labels}, on_n=ON_N, off_n=OFF_N, window_s=WINDOW_S)
    link_tree(SRC, OUT)
    root = zarr.open(str(OUT), "a")
    root.attrs["label_version"] = "field_v2n_v3r_lift_v1"
    root.attrs["delta_n"] = "0.5 s |F·n|, contact on >6 N off <3 N, else clearance"
    for ep_id, depth in labels.items():
        ep = root["data"][f"episode_{ep_id}"]
        src_ep = src["data"][f"episode_{ep_id}"]
        cmd = np.array(src_ep["ts_pose_command_0"])
        cmd[:, 2] = depth
        replace_array(ep, "fixb_delta_n_0", depth, src_ep["fixb_delta_n_0"])
        replace_array(ep, "ts_pose_command_0", cmd, src_ep["ts_pose_command_0"])
        ep.attrs["label_version"] = "field_v2n_v3r_lift_v1"
    if not np.array_equal(np.asarray(src["data/episode_0/fixb_delta_n_0"][:8]), guard):
        raise SystemExit("source v3r δn changed")
    written = np.asarray(root["data/episode_0/fixb_delta_n_0"][:8])
    if not np.array_equal(written, labels[0][:8]):
        raise SystemExit("written δn does not match the label")
    _stamp(sha)
    print(f"[lift] zarr {OUT}", flush=True)


if __name__ == "__main__":
    main()
