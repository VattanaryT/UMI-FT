#!/usr/bin/env python3
"""Frame pictures and a teleop-field query at the kept thin TCPs. New files only."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
import zarr  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

OUT = Path(__file__).resolve().parent
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
ELEV, AZIM = 22, -58
ARROW = 0.028


def _frames(path: Path):
    data = np.load(path)
    frames = np.asarray(data["frames"], dtype=np.float64).reshape(-1, 3, 3)
    vertices = np.asarray(data["vertices"], dtype=np.float64)
    apron = np.asarray(data["is_apron"]).astype(bool) if "is_apron" in data.files else np.zeros(len(vertices), bool)
    return vertices, frames, apron, np.asarray(data["faces"]), np.asarray(data["normals"], dtype=np.float64)


def _fps(points: np.ndarray, n: int) -> np.ndarray:
    chosen = [int(np.argmax(points[:, 0]))]
    nearest = np.full(len(points), np.inf)
    for _ in range(n - 1):
        nearest = np.minimum(nearest, np.linalg.norm(points - points[chosen[-1]], axis=1))
        chosen.append(int(np.argmax(nearest)))
    return np.asarray(chosen)


def _stroke(points: np.ndarray, n: int, gap: float) -> np.ndarray:
    kept = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[kept[-1]]) >= gap:
            kept.append(i)
    take = np.linspace(0, len(kept) - 1, min(n, len(kept))).astype(int)
    return np.asarray(kept)[take]


def _equal(ax, points: np.ndarray) -> None:
    center = points.mean(axis=0)
    radius = float((points.max(0) - points.min(0)).max()) / 2 * 1.05
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(c - radius, c + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=ELEV, azim=AZIM)


def _draw(ax, vertices, frames, index, title: str, length: float, view: np.ndarray) -> None:
    cloud = view[::12]
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=1.6, c="#4d5c68", alpha=0.55, linewidths=0)
    origin = vertices[index]
    for column, color in ((0, "#d32020"), (1, "#1f8a3b"), (2, "#1d4ed8")):
        direction = frames[index, :, column]
        ax.quiver(
            origin[:, 0], origin[:, 1], origin[:, 2],
            direction[:, 0], direction[:, 1], direction[:, 2],
            length=length, normalize=True, color=color, arrow_length_ratio=0.28, linewidth=1.1,
        )
    _equal(ax, view)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def _pictures() -> None:
    thin_v, thin_f, thin_apron, _, _ = _frames(THIN)
    train_v, train_f, train_apron, _, _ = _frames(TRAIN)
    body = lambda v, a: np.flatnonzero(~a) if a.any() else np.arange(len(v))
    thin_body, train_body = body(thin_v, thin_apron), body(train_v, train_apron)

    c_pose = np.asarray(zarr.open(HELD / "C_first/replay.zarr", "r")["data"]["episode_0"]["ts_pose_fb_0"])
    t_pose = np.asarray(zarr.open(TRAIN_ZARR, "r")["data"]["episode_0"]["ts_pose_fb_0"])
    thin_tree, train_tree = cKDTree(thin_v), cKDTree(train_v)

    def on_stroke(pose, tree, n=36):
        idx = _stroke(np.asarray(pose[:, :3], dtype=np.float64), n, 0.025)
        _dist, vert = tree.query(pose[idx, :3])
        return vert

    panels = (
        ("thin_stringer.png", thin_v, thin_f, _fps(thin_v[thin_body], 30), on_stroke(c_pose, thin_tree), "thin stringer"),
        ("training_stringer.png", train_v, train_f, _fps(train_v[train_body], 30), on_stroke(t_pose, train_tree), "training stringer"),
    )
    for name, vertices, frames, surface, stroke, title in panels:
        fig = plt.figure(figsize=(11.5, 5.2))
        left, right = fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")
        shown = vertices[thin_body] if name.startswith("thin") else vertices[train_body]
        picks = thin_body[surface] if name.startswith("thin") else train_body[surface]
        _draw(left, vertices, frames, picks, f"{title}: 30 surface points", ARROW, shown)
        _draw(right, vertices, frames, stroke, f"{title}: demo stroke", ARROW, shown)
        fig.tight_layout()
        fig.savefig(OUT / name, dpi=130)
        plt.close(fig)
        print(f"[snap] {OUT / name}", flush=True)

    cap = np.arange(len(thin_v) - 513, len(thin_v))
    fig = plt.figure(figsize=(6.2, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    _draw(ax, thin_v, thin_f, cap[::16], "thin end cap", 0.012, thin_v[cap])
    center = thin_v[cap].mean(0)
    radius = float((thin_v[cap].max(0) - thin_v[cap].min(0)).max()) / 1.4
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(c - radius, c + radius)
    fig.tight_layout()
    fig.savefig(OUT / "thin_end_cap.png", dpi=130)
    plt.close(fig)
    print(f"[snap] {OUT / 'thin_end_cap.png'}", flush=True)


def _geodesic(a: np.ndarray, b: np.ndarray) -> float:
    rel = a.T @ b
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec())))


def _jump(frames: np.ndarray, times: np.ndarray) -> dict:
    angles, flips = [], 0
    for i in range(len(frames) - 1):
        if times[i + 1] - times[i] > 0.3:
            continue
        angles.append(_geodesic(frames[i], frames[i + 1]))
        if float(np.dot(frames[i, :, 0], frames[i + 1, :, 0])) < 0.0:
            flips += 1
    angles = np.asarray(angles)
    if len(angles) == 0:
        return {"n": 0}
    return {
        "n": int(len(angles)),
        "median": round(float(np.median(angles)), 2),
        "p99": round(float(np.percentile(angles, 99)), 2),
        "max": round(float(np.max(angles)), 2),
        "heading_flips": int(flips),
    }


def _band(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "median": round(float(np.median(values)), 2),
        "p99": round(float(np.percentile(values, 99)), 2),
        "max": round(float(np.max(values)), 2),
    }


def _kept_tcp():
    kept = json.loads((HELD / "contact_wall_v2/kept.json").read_text())
    rows = {}
    for letter, key, zarr_path in (
        ("C", "0", HELD / "C_first/replay.zarr"),
        ("A", "1", HELD / "A_full/replay.zarr"),
    ):
        ep = zarr.open(zarr_path, "r")["data"]["episode_0"]
        robot_t = np.asarray(ep["robot_time_stamps_0"]).reshape(-1)
        rgb0 = float(np.asarray(ep["rgb_time_stamps_0"]).reshape(-1)[0])
        pose = np.asarray(ep["ts_pose_fb_0"], dtype=np.float64)
        frame = np.asarray(ep["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        times, tcp, current = [], [], []
        for t, keep in zip(kept[key]["t"], kept[key]["keep"]):
            if not keep:
                continue
            i = int(np.argmin(np.abs(robot_t - (rgb0 + float(t)))))
            times.append(float(t))
            tcp.append(pose[i, :3])
            current.append(frame[i])
        rows[letter] = (np.asarray(times), np.asarray(tcp), np.stack(current))
    return rows


def _main_field(mesh):
    source = subprocess.check_output(
        ["git", "-C", str(REPO), "show", "FETCH_HEAD:src/geodude_teleoperation/virtual_fixture.py"],
        text=True,
    )
    module = types.ModuleType("vf_main")
    exec(compile(source, "virtual_fixture_main.py", "exec"), module.__dict__)
    # Main keeps every vertex unless asked. 2048 is the local query cap.
    return module.OrientationField(
        mesh, key_direction=np.array([1.0, 0.0, 0.0]), num_boundary_samples=2048
    ), module


def _matrix_from(module, field, position):
    rotor = field._field.diffuse(position, field._t_spatial, max_iterations=field._max_iterations)
    return module._fiber_rotor_to_rotation_matrix(rotor)


def _teleop(rows) -> None:
    sys.path.insert(0, str(REPO / "src"))
    from geodude_teleoperation.virtual_fixture import OrientationField

    vertices, _frames_v, _apron, faces, normals = _frames(THIN)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.vertex_normals = normals
    print("[vf] building local length-tangent field", flush=True)
    local = OrientationField(mesh)
    print("[vf] building main vector-heat field", flush=True)
    try:
        main, main_mod = _main_field(mesh)
    except MemoryError as exc:
        main, main_mod = None, None
        print(f"[vf] main field skipped: {exc}", flush=True)
    tree = cKDTree(vertices)
    report = {}
    for letter, (times, tcp, current) in rows.items():
        dist, _idx = tree.query(tcp)
        local_mod = sys.modules["geodude_teleoperation.virtual_fixture"]
        local_raw, local_approach, main_raw = [], [], []
        for point in tcp:
            local_raw.append(_matrix_from(local_mod, local, point))
            local_approach.append(np.asarray(local.approach_basis_at(point), dtype=np.float64))
            if main is not None:
                main_raw.append(_matrix_from(main_mod, main, point))
        local_raw = np.stack(local_raw)
        local_approach = np.stack(local_approach)
        report[letter] = {
            "n": int(len(tcp)),
            "tcp_to_vertex_mm": _band(dist * 1e3),
            "within_8mm": int(np.sum(dist <= 0.008)),
            "current_jump": _jump(current, times),
            "local_diffuse_jump": _jump(local_raw, times),
            "local_approach_jump": _jump(local_approach, times),
            "local_diffuse_vs_current": _band([_geodesic(a, b) for a, b in zip(local_raw, current)]),
            "local_approach_vs_current": _band([_geodesic(a, b) for a, b in zip(local_approach, current)]),
            "t_spatial_local": float(local._t_spatial),
        }
        if main_raw:
            main_raw = np.stack(main_raw)
            report[letter]["main_diffuse_jump"] = _jump(main_raw, times)
            report[letter]["main_diffuse_vs_current"] = _band(
                [_geodesic(a, b) for a, b in zip(main_raw, current)]
            )
        print(f"[vf] {letter}", flush=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    _pictures()
    _teleop(_kept_tcp())
