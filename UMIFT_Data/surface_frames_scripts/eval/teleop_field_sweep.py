#!/usr/bin/env python3
"""Sweep main's vector-heat t_spatial on decimated thin and training meshes.

Read-only on geodude-teleoperation. No 8 mm snap, no temporal blend.
t1 is flipped so it follows the mold long axis. Run in the geodude-teleop env.
"""

from __future__ import annotations

import json
import subprocess
import types
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
import zarr
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

THIN = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/"
    "wall_fix/fixb_field_wall_vertical.npz"
)
TRAIN = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
TRAIN_ZARR = TRAIN.parents[1] / "r4vt_causal_foot" / "acp_replay_buffer.zarr"
HELD = Path(
    "/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/"
    "UMIFT_Data/surface_frames_scripts/eval/thin_heldout"
)
REPO = Path("/home/boeing/ros2_ws/src/geodude-teleoperation")
FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
TARGET_VERTS = 15000
OUT = Path(__file__).resolve().parent / "teleop_field_sweep.json"


def long_axis(vertices: np.ndarray) -> np.ndarray:
    centered = np.asarray(vertices, dtype=np.float64) - np.mean(vertices, axis=0)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    axis = np.array(vt[0], dtype=np.float64, copy=True)
    if axis[int(np.argmax(np.abs(axis)))] < 0.0:
        axis = -axis
    return axis / np.linalg.norm(axis)


def _manifold_counts(faces: np.ndarray, n_vertices: int) -> tuple[int, int]:
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    key = edges[:, 0].astype(np.int64) * n_vertices + edges[:, 1]
    _unique, counts = np.unique(key, return_counts=True)
    return int(np.sum(counts > 2)), int(np.sum(counts == 1))


def decimate_to(vertices: np.ndarray, faces: np.ndarray, target: int = TARGET_VERTS):
    """Quadric decimation. Returns the largest piece only when it stays manifold."""
    if len(vertices) <= target * 1.1:
        return vertices, faces
    original = (np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32))
    target_f = max(int(target * len(faces) / len(vertices)), 100)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces)
    )
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_f)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int32)
    piece = trimesh.Trimesh(verts, tris, process=False)
    parts = piece.split(only_watertight=False)
    piece = max(parts, key=lambda part: len(part.faces))
    verts = np.asarray(piece.vertices, dtype=np.float64)
    tris = np.asarray(piece.faces, dtype=np.int32)
    bad, _boundary = _manifold_counts(tris, len(verts))
    if bad or not (0.7 * target <= len(verts) <= 1.2 * target):
        print(f"[decimate] keep {len(original[0])} (got {len(verts)}, nonmanifold {bad})", flush=True)
        return original
    return verts, tris


def body_parts(path: Path):
    data = np.load(path)
    vertices = np.asarray(data["vertices"], dtype=np.float64)
    faces = np.asarray(data["faces"], dtype=np.int32)
    normals = np.asarray(data["normals"], dtype=np.float64)
    if "is_apron" in data.files:
        apron = np.asarray(data["is_apron"]).astype(bool)
        faces = faces[~apron[faces].any(axis=1)]
        used = np.unique(faces)
        remap = np.full(len(vertices), -1, dtype=np.int32)
        remap[used] = np.arange(len(used), dtype=np.int32)
        vertices, normals, faces = vertices[used], normals[used], remap[faces]
    mesh = trimesh.Trimesh(vertices, faces, vertex_normals=normals, process=False)
    return [part for part in mesh.split(only_watertight=False) if len(part.vertices) >= 400]


def _align_normals(part: trimesh.Trimesh, reference: np.ndarray, reference_normals: np.ndarray):
    normals = np.array(part.vertex_normals, dtype=np.float64, copy=True)
    _dist, index = cKDTree(reference).query(np.asarray(part.vertices))
    flip = np.einsum("ij,ij->i", normals, reference_normals[index]) < 0.0
    normals[flip] *= -1.0
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    part.vertex_normals = normals / np.maximum(length, 1e-12)
    return part


def load_main():
    source = subprocess.check_output(
        ["git", "-C", str(REPO), "show", "main:src/geodude_teleoperation/virtual_fixture.py"],
        text=True,
    )
    module = types.ModuleType("vf_main")
    exec(compile(source, "virtual_fixture_main.py", "exec"), module.__dict__)
    return module


def build_fields(path: Path, module):
    """One vector-heat field per manifold piece, each at about 15k vertices."""
    data = np.load(path)
    reference = np.asarray(data["vertices"], dtype=np.float64)
    reference_normals = np.asarray(data["normals"], dtype=np.float64)
    axis = long_axis(reference)
    fields = []
    for part in body_parts(path):
        vertices, faces = decimate_to(np.asarray(part.vertices), np.asarray(part.faces))
        piece = _align_normals(
            trimesh.Trimesh(vertices, faces, process=False), reference, reference_normals
        )
        vertices = np.ascontiguousarray(piece.vertices, dtype=np.float64)
        faces = np.ascontiguousarray(piece.faces, dtype=np.int32)
        normals = np.asarray(piece.vertex_normals, dtype=np.float64)
        bad, _boundary = _manifold_counts(faces, len(vertices))
        if bad:
            raise RuntimeError(f"{path.name} piece has {bad} non-manifold edges")
        def _laplacian(mesh_v, mesh_f):
            complex_ = module.gafropy.SimplicialComplex.build_from_faces(
                len(mesh_v), [[int(i) for i in row] for row in mesh_f]
            )
            return module.gafropy.ConnectionLaplacian.build(
                complex_, [tuple(map(float, row)) for row in mesh_v]
            )

        try:
            connection = _laplacian(vertices, faces)
        except Exception as exc:
            raw_v = np.asarray(part.vertices, dtype=np.float64)
            raw_f = np.asarray(part.faces, dtype=np.int32)
            print(
                f"[build] {path.name} decimated V={len(vertices)} failed ({exc}); retry {len(raw_v)}",
                flush=True,
            )
            if len(raw_v) == len(vertices):
                print(f"[build] skip piece V={len(vertices)}", flush=True)
                continue
            piece = _align_normals(
                trimesh.Trimesh(raw_v, raw_f, process=False), reference, reference_normals
            )
            vertices = np.ascontiguousarray(piece.vertices, dtype=np.float64)
            faces = np.ascontiguousarray(piece.faces, dtype=np.int32)
            normals = np.asarray(piece.vertex_normals, dtype=np.float64)
            try:
                connection = _laplacian(vertices, faces)
            except Exception as exc2:
                print(f"[build] skip piece V={len(vertices)} ({exc2})", flush=True)
                continue
        source = int(np.argmax(vertices[:, 2]))
        seed = module._tangent_direction(piece, source, axis)
        diffused = np.asarray(
            connection.diffuse_vector(
                seed, source, 5.0 * float(connection.get_mean_edge_length_squared())
            )
        )
        magnitude = np.linalg.norm(diffused, axis=1)
        tangent = np.where(
            magnitude[:, None] < 1e-9, seed, diffused / np.maximum(magnitude, 1e-30)[:, None]
        )
        tangent = np.array(tangent, dtype=np.float64, copy=True)
        flip = tangent @ axis < 0.0
        tangent[flip] *= -1.0
        rotors = module._continuous_boundary_rotors(piece, tangent, normals)
        field = module.gafropy.RotorFieldDiffusion(
            [tuple(map(float, row)) for row in vertices], rotors
        )
        edge = float(trimesh.Trimesh(vertices, faces, process=False).edges_unique_length.mean())
        fields.append({"field": field, "vertices": vertices, "edge": edge, "flipped": int(flip.sum())})
        print(
            f"[build] {path.name} V={len(vertices)} F={len(faces)} "
            f"flipped={int(flip.sum())} mean_edge={edge:.4f}",
            flush=True,
        )
    points = np.vstack([item["vertices"] for item in fields])
    owners = np.concatenate(
        [np.full(len(item["vertices"]), i, dtype=np.int32) for i, item in enumerate(fields)]
    )
    return fields, cKDTree(points), owners, axis


def basis_at(module, field, position, t_spatial: float) -> np.ndarray:
    rotor = field.diffuse([float(x) for x in position], float(t_spatial), max_iterations=50)
    matrix = np.array(module._fiber_rotor_to_rotation_matrix(rotor), dtype=np.float64, copy=True)
    matrix[:, 1] *= -1.0
    matrix[:, 2] *= -1.0
    return matrix


def geodesic(a: np.ndarray, b: np.ndarray) -> float:
    relative = np.asarray(a).T @ np.asarray(b)
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec())))


def _band(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "median": round(float(np.median(values)), 2),
        "p99": round(float(np.percentile(values, 99)), 2),
        "max": round(float(np.max(values)), 2),
    }


def jumps(frames: np.ndarray, times: np.ndarray) -> dict:
    angles, flips = [], 0
    for i in range(len(frames) - 1):
        if not np.isfinite(frames[i]).all() or not np.isfinite(frames[i + 1]).all():
            continue
        if times[i + 1] - times[i] > 0.3:
            continue
        angles.append(geodesic(frames[i], frames[i + 1]))
        if float(np.dot(frames[i, :, 0], frames[i + 1, :, 0])) < 0.0:
            flips += 1
    out = _band(angles)
    out["heading_flips"] = int(flips)
    return out


def query(module, fields, tree, owners, positions, factor: float) -> np.ndarray:
    _dist, index = tree.query(positions)
    which = owners[index]
    out = np.full((len(positions), 3, 3), np.nan)
    for piece in np.unique(which):
        take = np.flatnonzero(which == piece)
        item = fields[int(piece)]
        t_spatial = factor * item["edge"] ** 2
        for row in take:
            try:
                out[row] = basis_at(module, item["field"], positions[row], t_spatial)
            except RuntimeError:
                pass
    return out


def kept_tcp():
    kept = json.loads((HELD / "contact_wall_v2/kept.json").read_text())
    rows = {}
    for letter, key, zarr_path in (
        ("C", "0", HELD / "C_first/replay.zarr"),
        ("A", "1", HELD / "A_full/replay.zarr"),
    ):
        episode = zarr.open(zarr_path, "r")["data"]["episode_0"]
        robot_t = np.asarray(episode["robot_time_stamps_0"]).reshape(-1)
        rgb0 = float(np.asarray(episode["rgb_time_stamps_0"]).reshape(-1)[0])
        pose = np.asarray(episode["ts_pose_fb_0"], dtype=np.float64)
        frame = np.asarray(episode["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        times, tcp, current = [], [], []
        for stamp, keep in zip(kept[key]["t"], kept[key]["keep"]):
            if not keep:
                continue
            i = int(np.argmin(np.abs(robot_t - (rgb0 + float(stamp)))))
            times.append(float(stamp))
            tcp.append(pose[i, :3])
            current.append(frame[i])
        rows[letter] = (np.asarray(times), np.asarray(tcp), np.stack(current))
        print(f"[tcp] {letter} {len(times)}", flush=True)
    return rows


def training_windows(n_windows: int = 200, length: int = 16):
    root = zarr.open(TRAIN_ZARR, "r")["data"]
    pool = []
    for episode in range(50):
        if episode in (35, 44) or f"episode_{episode}" not in root:
            continue
        group = root[f"episode_{episode}"]
        robot_t = np.asarray(group["robot_time_stamps_0"]).reshape(-1)
        rgb_t = np.asarray(group["rgb_time_stamps_0"]).reshape(-1)
        pose = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
        frame = np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        index = np.clip(np.searchsorted(robot_t, rgb_t), 1, len(robot_t) - 1)
        left = index - 1
        use_left = np.abs(robot_t[left] - rgb_t) <= np.abs(robot_t[index] - rgb_t)
        index = np.where(use_left, left, index)
        for start in range(0, len(rgb_t) - length, 64):
            sl = slice(start, start + length)
            pool.append((rgb_t[sl] - rgb_t[start], pose[index[sl], :3], frame[index[sl]]))
    pick = np.unique(np.linspace(0, len(pool) - 1, n_windows).astype(int))
    chosen = [pool[i] for i in pick]
    print(f"[tcp] training windows {len(chosen)} of {len(pool)}", flush=True)
    return chosen


def sequence_stats(frames_list, times_list, foot_list) -> dict:
    jump_angles, flips, angles = [], 0, []
    foot_angles, foot_flips = [], 0
    for frames, times, foot in zip(frames_list, times_list, foot_list):
        for i in range(len(frames) - 1):
            if times[i + 1] - times[i] > 0.3:
                continue
            if np.isfinite(frames[i]).all() and np.isfinite(frames[i + 1]).all():
                jump_angles.append(geodesic(frames[i], frames[i + 1]))
                if float(np.dot(frames[i, :, 0], frames[i + 1, :, 0])) < 0.0:
                    flips += 1
            if np.isfinite(foot[i]).all() and np.isfinite(foot[i + 1]).all():
                foot_angles.append(geodesic(foot[i], foot[i + 1]))
                if float(np.dot(foot[i, :, 0], foot[i + 1, :, 0])) < 0.0:
                    foot_flips += 1
        for frame, target in zip(frames, foot):
            if np.isfinite(frame).all():
                angles.append(geodesic(frame, target))
    field = _band(jump_angles)
    field["heading_flips"] = int(flips)
    foot = _band(foot_angles)
    foot["heading_flips"] = int(foot_flips)
    return {"jump": field, "foot_jump": foot, "angle_to_foot": _band(angles)}


def main() -> None:
    module = load_main()
    print("[build] training", flush=True)
    train_fields, train_tree, train_owners, train_axis = build_fields(TRAIN, module)
    print("[build] thin", train_axis.round(3), flush=True)
    thin_fields, thin_tree, thin_owners, thin_axis = build_fields(THIN, module)
    print("[build] thin axis", thin_axis.round(3), flush=True)
    kept = kept_tcp()
    windows = training_windows()
    report = {"factors": {}}
    foot_done = False
    for factor in FACTORS:
        print(f"[query] {factor}", flush=True)
        thin_frames, thin_times, thin_foot = [], [], []
        per_letter = {}
        for letter, (times, tcp, foot) in kept.items():
            frames = query(module, thin_fields, thin_tree, thin_owners, tcp, factor)
            per_letter[letter] = sequence_stats([frames], [times], [foot])
            thin_frames.append(frames)
            thin_times.append(times)
            thin_foot.append(foot)
        train_frames, train_times, train_foot = [], [], []
        for times, tcp, foot in windows:
            train_frames.append(query(module, train_fields, train_tree, train_owners, tcp, factor))
            train_times.append(times)
            train_foot.append(foot)
        block = {
            "C": per_letter["C"],
            "A": per_letter["A"],
            "thin": sequence_stats(thin_frames, thin_times, thin_foot),
            "train": sequence_stats(train_frames, train_times, train_foot),
        }
        if not foot_done:
            report["foot"] = {
                "C": per_letter["C"]["foot_jump"],
                "A": per_letter["A"]["foot_jump"],
                "thin": block["thin"]["foot_jump"],
                "train": block["train"]["foot_jump"],
            }
            foot_done = True
        report["factors"][str(factor)] = block
        thin_jump = block["thin"]["jump"]
        train_jump = block["train"]["jump"]
        print(
            f"  thin p99 {thin_jump.get('p99')} train p99 {train_jump.get('p99')} "
            f"thin angle {block['thin']['angle_to_foot'].get('median')}",
            flush=True,
        )
    choices = []
    for factor, block in report["factors"].items():
        thin_p = block["thin"]["jump"].get("p99")
        train_p = block["train"]["jump"].get("p99")
        if thin_p is not None and train_p is not None and thin_p <= train_p:
            choices.append((thin_p, block["thin"]["angle_to_foot"].get("median", 1e9), float(factor)))
    if choices:
        _p99, _angle, factor = min(choices)
        foot_p = report["foot"]["thin"].get("p99")
        report["pick"] = {
            "factor": factor,
            "thin_p99": report["factors"][str(factor)]["thin"]["jump"]["p99"],
            "train_p99": report["factors"][str(factor)]["train"]["jump"]["p99"],
            "beats_foot_jump": bool(
                foot_p is not None and report["factors"][str(factor)]["thin"]["jump"]["p99"] < foot_p
            ),
        }
    else:
        report["pick"] = None
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({"foot": report["foot"], "pick": report["pick"]}, indent=2))
    print(f"[wrote] {OUT}", flush=True)


if __name__ == "__main__":
    main()
