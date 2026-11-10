#!/usr/bin/env python3
"""Remove one-sample T265 teleports and write a new fixture-pose sidecar.

Does not modify *_ee_master.npz or the training zarrs. Output:
  /mnt/data/umidata/ft/datasets/stringer_v3.5/tag_traj_odom_corr/<name>_odom_corr.npz
  .../summary.json

A step is a candidate when the pose jumps more than 15 cm or 5 degrees in one
sample of the 200 Hz stream. It is removed only when the reported odom
velocity explains less than 1 cm and less than 1 degree over that sample.
Corrections are applied in time order on the corrected stream. The written
pose integrates that odom from each good tag fix and blends the closure onto
the next good tag, matching densify_poses_with_t265. Episode 47, 31–47 s on
the recording clock, is stored as a bad span and is not used as an anchor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, "/home/boeing/umi_ft_ws/ee_tag_traj")
from ee_tag_traj.config import load_config
from ee_tag_traj.detect import detect_tags, estimate_T_cam_tag_all_candidates
from ee_tag_traj.geometry import (
    average_SE3,
    exp_se3,
    inv_T,
    log_se3,
    mul_T,
    T_from_pose7_wxyz,
)
TAG_DIR = Path("/mnt/data/umidata/ft/datasets/stringer_v3.5/tag_traj")
UNIFIED_DIR = Path("/mnt/data/umidata/ft/datasets/stringer_v3.5/zarr")
OUT_DIR = Path("/mnt/data/umidata/ft/datasets/stringer_v3.5/tag_traj_odom_corr")
CFG_PATH = Path("/home/boeing/umi_ft_ws/ee_tag_traj/configs/example_stringer.yaml")

STEP_M = 0.15
ROT_DEG = 5.0
DT_MAX_S = 0.05
# Velocity-predicted motion over the sample. The teleports report a few cm/s,
# which is under 2 mm in a 5 ms sample. 1 cm/s itself would keep every jump.
VEL_EXPLAIN_M = 0.01
W_EXPLAIN_DEG = 1.0
REPROJ_MAX_PX = 2.0
AMBIGUITY_RATIO = 1.6
# Recording clock (seconds after the first RGB stamp), not the viewer's +5 s pad.
BAD_SPANS = {47: [(31.0, 47.0)]}

CFG = load_config(CFG_PATH)
ALLOWED = set(CFG.tags.T_master_tag)
T_FIX = CFG.extrinsics.T_fixture_master_tag
T_D405_TCP = CFG.extrinsics.T_d405_tcp


def names() -> list[str]:
    return [p.name.replace("_ee_master.npz", "") for p in sorted(TAG_DIR.glob("*_ee_master.npz"))]


def poses_from_pq(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    R = Rotation.from_quat(quat_xyzw).as_matrix()
    T = np.zeros((len(pos), 4, 4), dtype=np.float64)
    T[:, :3, :3] = R
    T[:, :3, 3] = pos
    T[:, 3, 3] = 1.0
    return T


def rel_step(Ta: np.ndarray, Tb: np.ndarray) -> tuple[float, float]:
    d = mul_T(inv_T(Ta), Tb)
    trans = float(np.linalg.norm(d[:3, 3]))
    c = float(np.clip((np.trace(d[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
    rot = float(np.degrees(np.arccos(c)))
    return trans, rot


def se3_err(T_pred: np.ndarray, T_meas: np.ndarray) -> tuple[float, float]:
    xi = log_se3(mul_T(inv_T(T_pred), T_meas))
    return float(np.linalg.norm(xi[3:])), float(np.degrees(np.linalg.norm(xi[:3])))


def load_odom(name: str):
    root = zarr.open(str(UNIFIED_DIR / f"{name}_unified.zarr"), "r")
    odom = root["sensors"]["t265_odom"]
    frames = root["camera"]["color"]["frames"]

    def col(key: str) -> np.ndarray:
        return np.asarray(odom[key], dtype=np.float64).reshape(-1)

    t = col("timestamp_sec_unified")
    pos = np.column_stack([col("pos_x"), col("pos_y"), col("pos_z")])
    quat = np.column_stack([col("orient_x"), col("orient_y"), col("orient_z"), col("orient_w")])
    vel = np.column_stack([col("vel_x"), col("vel_y"), col("vel_z")])
    ang = np.column_stack([col("angvel_x"), col("angvel_y"), col("angvel_z")])
    keep = np.ones(len(t), dtype=bool)
    keep[1:] = np.diff(t) > 1e-9
    return (
        t[keep],
        poses_from_pq(pos[keep], quat[keep]),
        vel[keep],
        ang[keep],
        frames,
    )


def correct_odom(T: np.ndarray, t: np.ndarray, vel: np.ndarray, ang: np.ndarray, t0: float):
    """Return corrected T, removed steps, rejected big steps. Walks in time order."""
    T = np.array(T, copy=True)
    removed = []
    rejected = []
    i = 1
    while i < len(T):
        dt = float(t[i] - t[i - 1])
        trans, rot = rel_step(T[i - 1], T[i])
        trans_big = trans > STEP_M
        rot_big = rot > ROT_DEG
        if not (trans_big or rot_big):
            i += 1
            continue
        v = float(np.linalg.norm(0.5 * (vel[i - 1] + vel[i])))
        wdeg = float(np.degrees(np.linalg.norm(0.5 * (ang[i - 1] + ang[i]))))
        explained_m = v * dt
        explained_deg = wdeg * dt
        one_sample = 0.0 < dt < DT_MAX_S
        trans_tiny = (not trans_big) or explained_m < VEL_EXPLAIN_M
        rot_tiny = (not rot_big) or explained_deg < W_EXPLAIN_DEG
        rec = {
            "t": round(float(t[i] - t0), 3),
            "dt_ms": round(dt * 1e3, 2),
            "cm": round(trans * 100, 2),
            "deg": round(rot, 2),
            "v_cm_s": round(v * 100, 2),
            "v_dt_mm": round(explained_m * 1e3, 2),
            "w_dt_deg": round(explained_deg, 3),
        }
        if one_sample and trans_tiny and rot_tiny:
            G = mul_T(T[i - 1], inv_T(T[i]))
            T[i:] = np.einsum("ij,njk->nik", G, T[i:])
            removed.append(rec)
        else:
            rec["reason"] = "not a 200 Hz sample" if not one_sample else "velocity explains the step"
            rejected.append(rec)
        i += 1
    return T, removed, rejected


def interp_poses(ot: np.ndarray, T: np.ndarray, tq: np.ndarray) -> np.ndarray:
    rot = Rotation.from_matrix(T[:, :3, :3])
    slerp = Slerp(ot, rot)
    tc = np.clip(tq, float(ot[0]), float(ot[-1]))
    Rq = slerp(tc).as_matrix()
    pq = np.column_stack([np.interp(tc, ot, T[:, j, 3]) for j in range(3)])
    out = np.zeros((len(tq), 4, 4), dtype=np.float64)
    out[:, :3, :3] = Rq
    out[:, :3, 3] = pq
    out[:, 3, 3] = 1.0
    return out


def tag_pose_at(frames, index: int) -> tuple[np.ndarray | None, float | None, bool]:
    """Fixture TCP from the tags in this RGB frame, max reproj, ambiguous flag."""
    img = np.asarray(frames[int(index)])[:, :, ::-1]
    dets = [d for d in detect_tags(img, CFG.tags.family) if d.tag_id in ALLOWED]
    cams = []
    reproj = []
    ambiguous = False
    for det in dets:
        scored = estimate_T_cam_tag_all_candidates(
            det.corners_px, CFG.tags.tag_size_m, CFG.camera.K, CFG.camera.dist
        )
        if not scored:
            continue
        err0 = float(scored[0][1])
        if len(scored) > 1 and err0 > 0.05 and scored[1][1] / max(err0, 1e-9) < AMBIGUITY_RATIO:
            ambiguous = True
            continue
        if err0 > REPROJ_MAX_PX:
            continue
        cams.append(mul_T(CFG.tags.T_master_tag[det.tag_id], inv_T(scored[0][0])))
        reproj.append(err0)
    if not cams:
        return None, None, ambiguous
    X = cams[0] if len(cams) == 1 else average_SE3(cams)
    tcp = mul_T(T_FIX, mul_T(X, T_D405_TCP))
    return tcp, float(max(reproj)), ambiguous


def propagate(T_anchor: np.ndarray, To_a: np.ndarray, To_q: np.ndarray) -> np.ndarray:
    return mul_T(T_anchor, mul_T(inv_T(To_a), To_q))


def poses_to_pose7(T: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(T[:, :3, :3]).as_quat()
    pose7 = np.empty((len(T), 7), dtype=np.float64)
    pose7[:, :3] = T[:, :3, 3]
    pose7[:, 3] = quat_xyzw[:, 3]
    pose7[:, 4:] = quat_xyzw[:, :3]
    return pose7


def build_pose(st: np.ndarray, To_corr: np.ndarray, anchors: list[dict]) -> np.ndarray:
    """Fixture TCP on the RGB clock. Blend each closure onto the next good tag."""
    N = len(st)
    out = np.zeros((N, 4, 4), dtype=np.float64)
    if not anchors:
        out[:] = np.eye(4)
        return out
    ids = [a["i"] for a in anchors]
    for a in anchors:
        out[a["i"]] = a["T"]
    for k, a in enumerate(anchors[:-1]):
        b = anchors[k + 1]
        i0, i1 = a["i"], b["i"]
        Tend = propagate(a["T"], To_corr[i0], To_corr[i1])
        xi = log_se3(mul_T(inv_T(Tend), b["T"]))
        span = float(st[i1] - st[i0])
        for j in range(i0 + 1, i1):
            alpha = 0.0 if span <= 1e-9 else float((st[j] - st[i0]) / span)
            out[j] = mul_T(propagate(a["T"], To_corr[i0], To_corr[j]), exp_se3(alpha * xi))
        out[i1] = b["T"]
    first, last = anchors[0], anchors[-1]
    for j in range(0, first["i"]):
        out[j] = propagate(first["T"], To_corr[first["i"]], To_corr[j])
    for j in range(last["i"] + 1, N):
        out[j] = propagate(last["T"], To_corr[last["i"]], To_corr[j])
    return out


def closures(anchors: list[dict], To: np.ndarray) -> list[dict]:
    rows = []
    for a, b in zip(anchors, anchors[1:]):
        pred = propagate(a["T"], To[a["i"]], To[b["i"]])
        cm, deg = se3_err(pred, b["T"])
        rows.append({"t0": a["t"], "t1": b["t"], "cm": cm * 100, "deg": deg})
    return rows


def process(ep: int, name: str) -> dict:
    side = np.load(TAG_DIR / f"{name}_ee_master.npz", allow_pickle=True)
    st = np.asarray(side["timestamp_sec_unified"], dtype=np.float64).reshape(-1)
    n_vis = np.asarray(side["n_tags_visible"], dtype=np.int32)
    t0 = float(st[0])
    ot, T_raw, vel, ang, frames = load_odom(name)
    T_corr, removed, rejected = correct_odom(T_raw, ot, vel, ang, t0)
    To_raw = interp_poses(ot, T_raw, st)
    To_corr = interp_poses(ot, T_corr, st)

    bad = BAD_SPANS.get(ep, [])
    anchors = []
    for i in np.flatnonzero(n_vis > 0):
        tr = float(st[i] - t0)
        if any(lo <= tr <= hi for lo, hi in bad):
            continue
        T_tag, reproj, ambiguous = tag_pose_at(frames, int(i))
        if T_tag is None or ambiguous:
            continue
        anchors.append({"i": int(i), "t": round(tr, 3), "T": T_tag, "reproj": reproj})

    before = closures(anchors, To_raw)
    after = closures(anchors, To_corr)
    pose_anchors = anchors
    if not pose_anchors:
        pose_anchors = [{"i": 0, "T": T_from_pose7_wxyz(np.asarray(side["pose7"][0]))}]
    pose = build_pose(st, To_corr, pose_anchors)
    dt = np.diff(ot)
    step_cm = np.linalg.norm(np.diff(T_corr[:, :3, 3], axis=0), axis=1) * 100
    max_step_after = float(step_cm[dt < DT_MAX_S].max()) if np.any(dt < DT_MAX_S) else 0.0
    bad_mask = np.zeros(len(st), dtype=np.uint8)
    rel = st - t0
    for lo, hi in bad:
        bad_mask[(rel >= lo) & (rel <= hi)] = 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / f"{name}_odom_corr.npz",
        timestamp_sec_unified=st,
        pose7=poses_to_pose7(pose),
        bad=bad_mask,
        bad_span_s=np.asarray(bad, dtype=np.float64).reshape(-1, 2) if bad else np.zeros((0, 2)),
        removed_t_s=np.asarray([r["t"] for r in removed], dtype=np.float64),
    )
    jump_rows = []
    for rec in removed:
        nxt = next((c for c in zip(before, after) if c[0]["t0"] < rec["t"] <= c[0]["t1"]), None)
        jump_rows.append(
            {
                **rec,
                "next_tag_t": None if nxt is None else nxt[0]["t1"],
                "before_cm": None if nxt is None else round(nxt[0]["cm"], 2),
                "before_deg": None if nxt is None else round(nxt[0]["deg"], 2),
                "after_cm": None if nxt is None else round(nxt[1]["cm"], 2),
                "after_deg": None if nxt is None else round(nxt[1]["deg"], 2),
            }
        )
    return {
        "episode": ep,
        "name": name,
        "n_removed": len(removed),
        "n_rejected": len(rejected),
        "removed": jump_rows,
        "rejected": rejected,
        "max_before_cm": None if not before else round(max(c["cm"] for c in before), 2),
        "max_after_cm": None if not after else round(max(c["cm"] for c in after), 2),
        "bad_span_s": bad,
        "n_anchors": len(anchors),
        "max_step_cm_after": round(max_step_after, 2),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", default="", help="comma-separated episode indices; default all")
    args = parser.parse_args()
    wanted = None if not args.episodes else {int(s) for s in args.episodes.split(",")}
    rows = []
    rejected = []
    for ep, name in enumerate(names()):
        if wanted is not None and ep not in wanted:
            continue
        zpath = UNIFIED_DIR / f"{name}_unified.zarr"
        if not zpath.exists():
            print(f"ep {ep} {name} missing unified zarr", flush=True)
            continue
        print(f"ep {ep} {name}", flush=True)
        row = process(ep, name)
        rows.append(row)
        rejected.extend({**r, "episode": ep} for r in row["rejected"])
        if row["n_removed"] or row["n_rejected"]:
            print(
                f"  removed {row['n_removed']}  rejected {row['n_rejected']}  "
                f"max closure {row['max_before_cm']} -> {row['max_after_cm']} cm",
                flush=True,
            )
            for rec in row["removed"]:
                print(
                    f"    t={rec['t']} {rec['cm']} cm  next {rec['next_tag_t']}  "
                    f"{rec['before_cm']} -> {rec['after_cm']} cm",
                    flush=True,
                )
    summary = {"episodes": rows, "rejected_big_steps": rejected, "bad_spans": BAD_SPANS}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
