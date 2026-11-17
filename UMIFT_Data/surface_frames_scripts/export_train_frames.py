#!/usr/bin/env python3
"""Overlay frames for the training episodes, plus the crown-wall and near-cap lists.

Reads the October zarrs and the mold field. Does not write them.
Does not touch ep0_ep9_frames.npz. Refuses if the output file already exists.

The event track uses the causal 0.1 s window, the same one the labels and the live
query use. The new frames are read from r4vt_frame_continuity_causal. The centered
overlay in r4vt_frame_continuity/ is left as it was.

Training episodes are the seed-42 lexical train split, without episodes 35 and 44.
An episode outside that set is included when it contains a crown-wall switch or a
near-cap step, so those rows can be opened in the viewer.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from diffusion_policy.common.sampler_umift import get_val_mask
from fixb_field import FixBMoldField
from fixb_labels import pose7_to_Rp
from fixb_labels_tracked import (
    crown_wall_edges,
    geodesic_jump,
    near_cap_steps,
    query_smoothed_frames,
    unexplained_jump,
)

OLD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr"
    "/r4vt_world_fixb_tilt/acp_replay_buffer.zarr"
)
NEW = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr"
    "/r4vt_frame_continuity_causal/acp_replay_buffer.zarr"
)
FIELD = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb/shared/fixb_field.npz")
OUT = NEW.parent / "train_episode_frames.npz"
EXCLUDE = {35, 44}


def main(out: Path | None = None) -> None:
    out = OUT if out is None else Path(out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    old_root = zarr.open(str(OLD), "r")["data"]
    new_root = zarr.open(str(NEW), "r")["data"]
    keys = [k for k in old_root.keys() if str(k).startswith("episode_")]
    if len(keys) != 50:
        raise SystemExit(f"expected 50 episodes, found {len(keys)}")
    val = {int(keys[i].split("_")[-1]) for i in np.flatnonzero(get_val_mask(len(keys), 0.1, seed=42))}
    train = []
    for key in keys:
        ep_id = int(key.split("_")[-1])
        if ep_id not in val and ep_id not in EXCLUDE:
            train.append(ep_id)
    field = FixBMoldField(FIELD)
    switches = []
    nearcap = []
    overlay_ids = set(train)
    pending = {}
    t0 = time.time()
    for key in keys:
        ep_id = int(key.split("_")[-1])
        ep = old_root[key]
        robot_t = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        rgb_t = np.asarray(ep["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
        pose = np.asarray(ep["ts_pose_fb_0"], dtype=np.float64)
        R_tool, p = pose7_to_Rp(pose)
        pack = query_smoothed_frames(field, robot_t, p, R_tool, align="causal", which="both")
        tracked = pack["tracked"]
        R = tracked["R"]
        if len(R) < 2:
            continue
        crown = crown_wall_edges(tracked["reason"], pack["free"]["R"])
        cap, out_step, tool_step = near_cap_steps(R, R_tool)
        x_new = np.asarray(new_root[key]["mold_x_s_0"], dtype=np.float64).reshape(-1, 3)
        for k in np.flatnonzero(crown):
            sample = int(k) + 1
            switches.append(
                (ep_id, sample, float(robot_t[sample]), float(out_step[k]), float(tool_step[k]), x_new[sample])
            )
        for k in np.flatnonzero(cap):
            sample = int(k) + 1
            nearcap.append(
                (ep_id, sample, float(robot_t[sample]), float(out_step[k]), float(tool_step[k]))
            )
        if ep_id in train or np.any(crown) or np.any(cap):
            overlay_ids.add(ep_id)
            pending[ep_id] = (key, robot_t, float(rgb_t[0]), pose, R_tool)
        print(f"ep {ep_id:02d} switches={len(switches)} nearcap={len(nearcap)} {time.time() - t0:.0f}s", flush=True)

    flat = {}
    for ep_id in sorted(overlay_ids):
        key, robot_t, t0_s, pose, R_tool = pending[ep_id]
        old = old_root[key]
        new = new_root[key]
        R_old = np.asarray(old["mold_R_field_0"], dtype=np.float32).reshape(-1, 3, 3)
        R_new = np.asarray(new["mold_R_field_0"], dtype=np.float32).reshape(-1, 3, 3)
        x_old = np.asarray(old["mold_x_s_0"], dtype=np.float32).reshape(-1, 3)
        x_new = np.asarray(new["mold_x_s_0"], dtype=np.float32).reshape(-1, 3)
        prefix = f"ep{ep_id}_"
        flat[prefix + "x_old"] = x_old
        flat[prefix + "x_new"] = x_new
        flat[prefix + "R_old"] = R_old
        flat[prefix + "R_new"] = R_new
        flat[prefix + "old_jump"] = geodesic_jump(R_old)
        flat[prefix + "new_jump"] = unexplained_jump(R_new, R_tool)
        flat[prefix + "robot_time_s"] = robot_t.astype(np.float64)
        flat[prefix + "t0"] = np.array([t0_s], dtype=np.float64)
        flat[prefix + "tcp_xyz"] = pose[:, :3].astype(np.float32)
        quat = pose[:, [4, 5, 6, 3]]
        flat[prefix + "tcp_R"] = Rotation.from_quat(quat).as_matrix().astype(np.float32)
        if ep_id in EXCLUDE:
            split = "excluded"
        elif ep_id in val:
            split = "val"
        else:
            split = "train"
        flat[prefix + "split"] = np.array(split)

    flat["switch_episode"] = np.array([row[0] for row in switches], dtype=np.int16)
    flat["switch_index"] = np.array([row[1] for row in switches], dtype=np.int32)
    flat["switch_time_s"] = np.array([row[2] for row in switches], dtype=np.float64)
    flat["switch_frame_deg"] = np.array([row[3] for row in switches], dtype=np.float32)
    flat["switch_tool_deg"] = np.array([row[4] for row in switches], dtype=np.float32)
    flat["switch_xyz"] = np.stack([np.asarray(row[5], dtype=np.float32) for row in switches])
    flat["nearcap_episode"] = np.array([row[0] for row in nearcap], dtype=np.int16)
    flat["nearcap_index"] = np.array([row[1] for row in nearcap], dtype=np.int32)
    flat["nearcap_time_s"] = np.array([row[2] for row in nearcap], dtype=np.float64)
    flat["nearcap_frame_deg"] = np.array([row[3] for row in nearcap], dtype=np.float32)
    flat["nearcap_tool_deg"] = np.array([row[4] for row in nearcap], dtype=np.float32)
    flat["train_episodes"] = np.array(sorted(train), dtype=np.int16)
    flat["val_episodes"] = np.array(sorted(val), dtype=np.int16)
    np.savez_compressed(out, **flat)
    print(
        f"wrote {out} episodes={sorted(overlay_ids)} train={sorted(train)} "
        f"switches={len(switches)} nearcap={len(nearcap)}",
        flush=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    main(parser.parse_args().out)
