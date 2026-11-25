#!/usr/bin/env python3
"""Sweep foot-face hysteresis on label replay. No robot and no checkpoint.

Uses the runner's decode_chunk, including the camera-table check and the 8°
slew. Replay binds the stored mold foot, the same path as UMIFT_FOOT_FACE_HYST
on a label replay.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "UMIFT_Data" / "surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS), str(ROOT)]

os.environ.setdefault("PYRITE_CHECKPOINT_FOLDERS", "/mnt/data/umidata/ft/training_outputs")
os.environ["UMIFT_FOOT_WALK"] = "geodesic"
os.environ["UMIFT_FOOT_FACE_HYST"] = "1"

import zarr
from fixb_labels import pose7_to_Rp
from foot_face_hyst import reset_episode
from gt_replay import GtReplay
from mold_aware_runtime import (
    MoldAwareRuntime,
    camera_table_from_flange,
    expand_r4_action,
    fixb_decode_dt,
    r4_safety_limits,
)
from PyriteConfig.tasks.umift.workcell_frames import load_workcell_extrinsics
from sliding_foot import NAMES, face_tags

DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
FIELD = DATA / "shared/fixb_field.npz"
ZARR = DATA / "r4vt_causal_foot/acp_replay_buffer.zarr"
EXTRINSICS = ROOT / "PyriteConfig/tasks/umift/layup_workcell_extrinsics_20261116_orange.yaml"
os.environ["UMIFT_WORKCELL_EXTRINSICS"] = str(EXTRINSICS)
EPS = (9, 12, 28, 37, 41)
EXEC = 6


def _rot_deg(a, b) -> float:
    from scipy.spatial.transform import Rotation

    rel = np.asarray(a).T @ np.asarray(b)
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec())))


def _tags(field, tags, points):
    import open3d as o3d

    q = field.scene.compute_closest_points(
        o3d.core.Tensor(np.asarray(points, dtype=np.float32).reshape(-1, 3))
    )
    tri = np.clip(q["primitive_ids"].numpy().astype(np.int64), 0, len(tags) - 1)
    return tags[tri]


def _one(runtime, gt, mold_x, mold_R, tri, field, bench: bool) -> dict:
    reset_episode()
    runtime._cmd_foot = None
    runtime._cmd_R = None
    runtime._cmd_tcp = None
    runtime._cmd_vt = None
    runtime._vt_elapsed_s = 0.0
    runtime.limits.enable_delta_n_floor = bench
    feet, frames, rd, slew, dn, stiff, hard = [], [], [], [], [], [], []
    demo_p, demo_R = [], []
    while gt.remaining():
        action, window = gt.current()
        action = np.asarray(action, dtype=np.float64)
        if action.shape[-1] == 10:
            action = expand_r4_action(action)
        n = min(EXEC, action.shape[0], len(window))
        idx = np.asarray(window[:n], dtype=np.int64)
        label_R, label_p = pose7_to_Rp(gt.demo[idx])
        runtime._tcp_step = gt.tcp_step(n)
        runtime._demo_feet = mold_x[idx]
        runtime._demo_frames = mold_R[idx]
        R0, p0 = pose7_to_Rp(gt.demo[idx[:1]])
        dt = fixb_decode_dt(n, 1, 1.0 / 15.0)
        dec = runtime.decode_chunk(action[:n], p0[0], R0[0], dt, label_R=label_R)
        feet.append(dec["x_s"])
        frames.append(dec["R_field"])
        rd.append(dec["R_d"])
        slew.append(np.asarray(dec["rd_slew"], dtype=bool))
        dn.append(np.asarray(dec["delta_n"]).reshape(-1))
        stiff.append(np.asarray(dec["stiffness"]).reshape(-1))
        hard.extend(list(runtime.last_hard_triggers))
        demo_p.append(mold_x[idx])
        demo_R.append(label_R)
        gt.advance()
    gt.reset()
    foot = np.concatenate(feet)
    frame = np.concatenate(frames)
    got_R = np.concatenate(rd)
    slew_b = np.concatenate(slew)
    delta_n = np.concatenate(dn)
    k = np.concatenate(stiff)
    demo_foot = np.concatenate(demo_p)
    demo_tool = np.concatenate(demo_R)
    got_tag = _tags(field, tri, foot)
    demo_tag = _tags(field, tri, demo_foot)
    wall = demo_tag == 0
    dist = np.linalg.norm(foot - demo_foot, axis=1) * 1e3
    stored_R = []
    gt.reset()
    while gt.remaining():
        _a, window = gt.current()
        n = min(EXEC, len(window))
        stored_R.append(mold_R[np.asarray(window[:n], dtype=np.int64)])
        gt.advance()
    gt.reset()
    stored_R = np.concatenate(stored_R)
    stored_n = -stored_R[:, :, 2]
    got_n = -frame[:, :, 2]
    ndeg = np.degrees(np.arccos(np.clip(np.sum(got_n * stored_n, axis=1), -1, 1)))
    odeg = np.array([_rot_deg(got_R[i], demo_tool[i]) for i in range(len(got_R))])
    # Longest run of steps sitting on the 8° cap.
    run = best = 0
    for flag in slew_b:
        run = run + 1 if flag else 0
        best = max(best, run)
    # After the biggest demo orientation step, the later error.
    step = np.array([_rot_deg(demo_tool[i], demo_tool[i + 1]) for i in range(len(demo_tool) - 1)])
    at = int(np.argmax(step)) if len(step) else 0
    after = odeg[at:at + 30]
    spring = np.minimum(25.0, k * np.maximum(-delta_n, 0.0))
    requested = k * np.maximum(-delta_n, 0.0)
    return {
        "steps": int(len(foot)),
        "wall_n": int(wall.sum()),
        "wall_on_wall": round(float(np.mean(got_tag[wall] == 0)) if wall.any() else 1.0, 3),
        "wall_mm_median": round(float(np.median(dist[wall])) if wall.any() else 0.0, 2),
        "wall_mm_p95": round(float(np.percentile(dist[wall], 95)) if wall.any() else 0.0, 2),
        "wall_normal_median": round(float(np.median(ndeg[wall])) if wall.any() else 0.0, 2),
        "ori_after_median": round(float(np.median(after)) if len(after) else 0.0, 2),
        "ori_after_min": round(float(after.min()) if len(after) else 0.0, 2),
        "slew_longest": int(best),
        "slew_n": int(slew_b.sum()),
        "hard": sorted(set(hard)),
        "spring25_max": round(float(spring.max()) if len(spring) else 0.0, 2),
        "spring_requested_max": round(float(requested.max()) if len(requested) else 0.0, 2),
        "crown_same": round(float(np.mean(got_tag[demo_tag == 1] == 1)) if np.any(demo_tag == 1) else 1.0, 3),
        "flange_same": round(float(np.mean(got_tag[demo_tag == 3] == 3)) if np.any(demo_tag == 3) else 1.0, 3),
    }


def main() -> None:
    os.environ["UMIFT_R4VT_ZARR"] = str(ZARR)
    field_path = FIELD
    T = load_workcell_extrinsics().T_fixture_from_right_base
    tri = face_tags(FIELD)
    root = zarr.open_group(str(ZARR), mode="r")["data"]
    rows = []
    for ep in EPS:
        gt = GtReplay("r4vt", ep, EXEC, normalizer=None)
        group = root[f"episode_{ep}"]
        mold_x = np.asarray(group["mold_x_s_0"], dtype=np.float64)
        mold_R = np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        runtime = MoldAwareRuntime.from_assets(
            run_type="r4_world_fixb_tilt",
            field_path=field_path,
            dry_run=False,
            limits=r4_safety_limits(enable_delta_n_floor=False),
        )
        nrm, off, _y = camera_table_from_flange(field_path, T)
        runtime.set_table_plane(nrm, off)
        for n_step, d_mm in ((1, 5), (4, 5), (8, 5), (4, 2), (4, 10)):
            os.environ["UMIFT_FOOT_HYST_N"] = str(n_step)
            os.environ["UMIFT_FOOT_HYST_D_MM"] = str(d_mm)
            for bench in (True, False):
                if ep != 9 and bench:
                    continue
                got = _one(runtime, gt, mold_x, mold_R, tri, runtime.field, bench)
                got.update({"ep": ep, "N": n_step, "d_mm": d_mm, "bench": bench})
                rows.append(got)
                print(json.dumps(got), flush=True)
    passed = [
        r for r in rows
        if r["wall_on_wall"] >= 0.90
        and r["wall_mm_p95"] <= 10
        and r["wall_normal_median"] <= 10
        and r["ori_after_min"] <= 10
        and r["slew_longest"] <= 6
        and not r["hard"]
        and r["spring25_max"] <= 25
    ]
    print(json.dumps({"n_rows": len(rows), "n_pass": len(passed), "pass": passed}, indent=2))


if __name__ == "__main__":
    main()
