#!/usr/bin/env python3
"""Summarize one mocked runner log: force, foot face, orientation, safety."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "UMIFT_Data" / "surface_frames_scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from fixb_labels import pose7_to_Rp
from sliding_foot import NAMES, face_tags

FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
ZARR = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_causal_foot/acp_replay_buffer.zarr"
)
SPRING_CAP_N = 40.0
PROPOSED_CAP_N = 25.0


def _log_dir(path: Path) -> Path:
    path = Path(path)
    kids = sorted(p for p in path.iterdir() if p.is_dir() and (p / "horizon_0").is_dir())
    if kids:
        return kids[-1]
    if (path / "horizon_0").is_dir():
        return path
    raise SystemExit(f"no control log under {path}")


def _closest_face(points, vertices, faces, centroids, tree):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    _, nn = tree.query(points, k=12)
    out = np.zeros(len(points), dtype=np.int64)
    for i, ids in enumerate(nn):
        best = 1e9
        chosen = int(ids[0])
        for face_id in ids:
            tri = vertices[faces[int(face_id)]]
            # barycentric closest point on the triangle
            a, b, c = tri
            ab, ac, ap = b - a, c - a, points[i] - a
            d00 = float(ab @ ab)
            d01 = float(ab @ ac)
            d11 = float(ac @ ac)
            d20 = float(ap @ ab)
            d21 = float(ap @ ac)
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-18:
                dist = float(np.linalg.norm(ap))
            else:
                v = (d11 * d20 - d01 * d21) / denom
                w = (d00 * d21 - d01 * d20) / denom
                u = 1.0 - v - w
                u, v, w = np.clip([u, v, w], 0.0, 1.0)
                s = u + v + w
                u, v, w = u / s, v / s, w / s
                dist = float(np.linalg.norm(points[i] - (u * a + v * b + w * c)))
            if dist < best:
                best = dist
                chosen = int(face_id)
        out[i] = chosen
    return out


def _angle(Ra, Rb) -> np.ndarray:
    rel = np.einsum("nji,njk->nik", Ra, Rb)
    return np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=1))


def _tool_z(Ra, Rb) -> np.ndarray:
    """Rotation of Rb relative to Ra, expressed in Ra's axes, degrees."""
    rel = np.einsum("nji,njk->nik", Ra, Rb)
    return np.degrees(Rotation.from_matrix(rel).as_rotvec())


def main() -> None:
    import zarr
    from PyriteConfig.tasks.umift.workcell_frames import load_workcell_extrinsics
    from PyriteEnvSuites.env_runners.dataset_start_pose import pose7_change_frame

    folder = _log_dir(Path(sys.argv[1]))
    root = zarr.open_group(str(folder), mode="r")
    names = sorted(
        (name for name in root.keys() if name.startswith("horizon_")),
        key=lambda name: int(name.split("_")[1]),
    )
    vt, nominal, demo, dn, k, foot, stamps = [], [], [], [], [], [], []
    for name in names:
        action = root[name]["action"]
        vt.append(np.asarray(action["ts_virtual_targets_0"]))
        nominal.append(np.asarray(action["ts_nominal_targets_0"]))
        dn.append(np.asarray(action["delta_n"]).reshape(-1))
        k.append(np.asarray(action["stiffness_raw_0"]).reshape(-1))
        foot.append(np.asarray(action["foot"]))
        stamps.append(np.asarray(action["timestamps_s"]).reshape(-1))
        if "ts_demo_0" in action:
            demo.append(np.asarray(action["ts_demo_0"]))
    vt = np.concatenate(vt)
    nominal = np.concatenate(nominal)
    dn = np.concatenate(dn)
    k = np.concatenate(k)
    foot = np.concatenate(foot)
    stamps = np.concatenate(stamps)
    demo = np.concatenate(demo) if demo else None
    T = load_workcell_extrinsics().T_fixture_from_right_base
    mesh = np.load(FIELD)
    vertices = mesh["vertices"].astype(np.float64)
    faces = mesh["faces"].astype(np.int64)
    tags = face_tags(FIELD)
    centroids = vertices[faces].mean(1)
    tree = cKDTree(centroids)
    foot_fix = pose7_change_frame(
        np.concatenate([foot, np.tile([1.0, 0, 0, 0], (len(foot), 1))], axis=1), T
    )[:, :3]
    cmd_face = tags[_closest_face(foot_fix, vertices, faces, centroids, tree)]
    R_cmd, _ = pose7_to_Rp(vt)
    step_deg = _angle(R_cmd[:-1], R_cmd[1:])
    spring = np.minimum(SPRING_CAP_N, k * np.maximum(-dn, 0.0))
    spring25 = np.minimum(PROPOSED_CAP_N, spring)
    journal = {}
    journal_path = folder.parent / "rollout_log.jsonl"
    if not journal_path.is_file():
        journal_path = folder / "rollout_log.jsonl"
    if journal_path.is_file():
        lines = journal_path.read_text().strip().splitlines()
        if lines:
            journal = json.loads(lines[-1])
    group = zarr.open_group(str(ZARR), mode="r")["data"]["episode_9"]
    demo_foot = np.asarray(group["mold_x_s_0"], dtype=np.float64)
    demo_R = np.asarray(group["mold_R_field_0"], dtype=np.float64)
    demo_pose = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    demo_tags = tags[_closest_face(demo_foot, vertices, faces, centroids, tree)]
    if demo is not None:
        demo_fix = pose7_change_frame(demo, T)
        dpos = demo_pose[:, :3]
        nn = np.zeros(len(demo_fix), dtype=np.int64)
        # chunked to keep memory down
        for i in range(0, len(demo_fix), 64):
            block = demo_fix[i : i + 64, :3]
            dist = np.linalg.norm(dpos[:, None, :] - block[None, :, :], axis=2)
            nn[i : i + 64] = np.argmin(dist, axis=0)
        demo_face = demo_tags[nn]
        R_demo, _ = pose7_to_Rp(demo)
    else:
        pose_t = np.asarray(group["robot_time_stamps_0"]).reshape(-1)
        origin = float(pose_t[1220])
        demo_clock = origin + (stamps - stamps[0]) / 2.0
        nn = np.clip(np.searchsorted(pose_t, demo_clock), 0, len(pose_t) - 1)
        demo_face = demo_tags[nn]
        R_demo, _ = pose7_to_Rp(pose7_change_frame(demo_pose[nn], np.linalg.inv(T)))
        demo = pose7_change_frame(demo_pose[nn], np.linalg.inv(T))
    wall = demo_face == 0
    same = cmd_face == demo_face
    ori = _angle(R_demo, R_cmd)
    tool = _tool_z(R_demo, R_cmd)
    counts = {}
    for tag, name in enumerate(NAMES):
        m = demo_face == tag
        if not np.any(m):
            continue
        got = {NAMES[int(t)]: int(np.sum(cmd_face[m] == t)) for t in np.unique(cmd_face[m])}
        counts[name] = {"n": int(m.sum()), "commanded": got, "same": int(np.sum(same[m]))}
    # Side switch: largest demo orientation step.
    demo_step = _angle(R_demo[:-1], R_demo[1:])
    switch = int(np.argmax(demo_step))
    window = slice(max(0, switch - 6), min(len(ori), switch + 12))
    report = {
        "log": str(folder),
        "n_steps": int(len(dn)),
        "n_horizons": len(names),
        "journal": {
            key: journal.get(key)
            for key in (
                "mode",
                "enable_delta_n_floor",
                "bench_dn_floor_mm",
                "wrench_hold_n",
                "n_safety_stops",
                "n_holds",
                "jump_aborted",
                "enable_rot_rate_clamp",
                "enable_tilt_clamp",
                "abort_on_measured_mesh_distance",
                "max_tcp_penetration_m",
            )
        },
        "delta_n_mm": {
            "median": float(np.median(dn) * 1e3),
            "min": float(np.min(dn) * 1e3),
            "max": float(np.max(dn) * 1e3),
            "frac_at_minus_5": float(np.mean(np.abs(dn + 0.005) < 1.5e-4)),
        },
        "stiffness": {
            "median": float(np.median(k)),
            "frac_near_200": float(np.mean(np.abs(k - 200) < 1)),
            "frac_near_3000": float(np.mean(k > 2500)),
        },
        "spring_n_cap40": {
            "median": float(np.median(spring)),
            "max": float(np.max(spring)),
            "frac_at_40": float(np.mean(spring >= 39.5)),
        },
        "spring_n_cap25": {"max": float(np.max(spring25)), "frac_at_25": float(np.mean(spring25 >= 24.5))},
        "faces": counts,
        "wall": {
            "n": int(wall.sum()),
            "same_face": int(np.sum(same & wall)),
            "commanded_crown": int(np.sum(wall & (cmd_face == 1))),
            "commanded_flange": int(np.sum(wall & (cmd_face == 3))),
            "ori_deg_median": float(np.median(ori[wall])) if np.any(wall) else None,
            "tool_z_deg_median": float(np.median(tool[wall, 2])) if np.any(wall) else None,
        },
        "orientation": {
            "vs_demo_median": float(np.median(ori)),
            "vs_demo_max": float(np.max(ori)),
            "step_deg_p95": float(np.percentile(step_deg, 95)),
            "step_deg_max": float(np.max(step_deg)),
            "steps_at_8": int(np.sum((step_deg > 7.9) & (step_deg < 8.2))),
        },
        "side_switch": {
            "step": int(switch),
            "demo_step_deg": float(demo_step[switch]),
            "tool_xyz_deg_median": [float(v) for v in np.median(tool[window], axis=0)],
            "ori_max": float(np.max(ori[window])),
            "cmd_steps_at_8": int(np.sum((step_deg[window] > 7.9) & (step_deg[window] < 8.2)))
            if window.stop - window.start > 1
            else 0,
        },
    }
    out = folder.parent / "report.json"
    if folder.parent.name.startswith("horizon"):
        out = folder / "report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
