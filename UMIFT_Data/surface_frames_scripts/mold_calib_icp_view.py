#!/usr/bin/env python3
"""Point-to-surface ICP of the scan-only mold onto the recorded TCP touches.

The eight TCP points stay where the current extrinsics put them. The mesh
moves. The synthetic flange apron is not part of the surface. Does not
command the robot.

  python UMIFT_Data/surface_frames_scripts/mold_calib_icp_view.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation

_SURFACE = Path(__file__).resolve().parent
if str(_SURFACE) not in sys.path:
    sys.path.insert(0, str(_SURFACE))

from PyriteConfig.tasks.umift.workcell_frames import load_workcell_extrinsics

CSV_PATH = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/mold_calib_train_20261116_145442.csv")
FIELD_PATH = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
OUT_YAML = Path(
    "/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/PyriteConfig/tasks/umift/layup_workcell_extrinsics_20261116.yaml"
)
SHIFT_M = 0.030
RMS_WRITE_M = 0.003


def load_tcp(path: Path) -> tuple[list[str], np.ndarray]:
    rows = list(csv.DictReader(path.open()))
    labels = [row["label"] for row in rows]
    pts = np.array(
        [[float(row["tcp_x_mm"]), float(row["tcp_y_mm"]), float(row["tcp_z_mm"])] for row in rows],
        dtype=np.float64,
    ) / 1000.0
    return labels, pts


def scan_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vertices, faces, and vertex normals of the scan, apron triangles removed."""
    data = np.load(path)
    vertices = np.asarray(data["vertices"], dtype=np.float64)
    faces = np.asarray(data["faces"], dtype=np.int64)
    normals = np.asarray(data["normals"], dtype=np.float64)
    apron = np.asarray(data["is_apron"], dtype=bool)
    keep = ~apron[faces].any(axis=1)
    faces = faces[keep]
    used = np.unique(faces)
    remap = -np.ones(len(vertices), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[faces], normals[used]


class Surface:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray):
        self.vertices = vertices
        self.faces = faces
        self.normals = normals
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(vertices.astype(np.float32))
        mesh.triangle.indices = o3d.core.Tensor(faces.astype(np.int32))
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(mesh)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Closest point, interpolated normal, and unsigned distance."""
        query = o3d.core.Tensor(np.asarray(points, dtype=np.float32).reshape(-1, 3))
        ans = self.scene.compute_closest_points(query)
        tri = ans["primitive_ids"].numpy().astype(np.int64)
        uv = ans["primitive_uvs"].numpy().astype(np.float64)
        weights = np.stack([1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]], axis=-1)
        vid = self.faces[tri]
        foot = np.einsum("qk,qkd->qd", weights, self.vertices[vid])
        normal = np.einsum("qk,qkd->qd", weights, self.normals[vid])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        dist = np.linalg.norm(points - foot, axis=1)
        return foot, normal, dist


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """dst ≈ R @ src + t."""
    center_src = src.mean(axis=0)
    center_dst = dst.mean(axis=0)
    cov = (src - center_src).T @ (dst - center_dst)
    u_mat, _s, vt = np.linalg.svd(cov)
    rot = vt.T @ u_mat.T
    if np.linalg.det(rot) < 0.0:
        vt = vt.copy()
        vt[-1] *= -1.0
        rot = vt.T @ u_mat.T
    return rot, center_dst - rot @ center_src


def _plane_step(points: np.ndarray, foot: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Damped point-to-plane step, capped so one iteration cannot leave the mold."""
    rows = []
    rhs = []
    for point, target, nrm in zip(points, foot, normal):
        rows.append(np.concatenate([np.cross(point, nrm), nrm]))
        rhs.append(float(np.dot(nrm, target - point)))
    design = np.vstack(rows)
    damp = 1e-3 * np.eye(6)
    twist = np.linalg.solve(design.T @ design + damp, design.T @ np.asarray(rhs))
    rotvec = twist[:3]
    trans = twist[3:]
    angle = float(np.linalg.norm(rotvec))
    cap = np.deg2rad(5.0)
    if angle > cap:
        rotvec *= cap / angle
        trans *= cap / angle
    move = float(np.linalg.norm(trans))
    if move > 0.015:
        trans *= 0.015 / move
    return Rotation.from_rotvec(rotvec).as_matrix(), trans


def icp(surface: Surface, points: np.ndarray, t0: np.ndarray, iters: int = 50) -> dict:
    """Move `points` onto `surface`. p_on_mesh ≈ R @ p + t.

    The first passes pull each touch to its closest point. Once that stalls,
    point-to-plane steps slide the points along the surface.
    """
    current = points + np.asarray(t0, dtype=np.float64)
    rot = np.eye(3)
    trans = np.asarray(t0, dtype=np.float64).copy()
    mode = "point"
    prev = None
    for _ in range(iters):
        foot, normal, dist = surface.project(current)
        rms = float(np.sqrt(np.mean(dist ** 2)))
        if mode == "point" and prev is not None and abs(prev - rms) < 1e-5:
            mode = "plane"
        if mode == "plane" and prev is not None and abs(prev - rms) < 1e-6:
            break
        prev = rms
        if mode == "point":
            dR, dv = _kabsch(current, foot)
        else:
            dR, dv = _plane_step(current, foot, normal)
        current = (dR @ current.T).T + dv
        rot = dR @ rot
        trans = dR @ trans + dv
    foot, normal, dist = surface.project(current)
    return {
        "R": rot,
        "t": trans,
        "dist": dist,
        "normal": normal,
        "rms": float(np.sqrt(np.mean(dist ** 2))),
    }


def starts() -> list[tuple[str, np.ndarray]]:
    out = [("current yaml", np.zeros(3))]
    for axis, name in enumerate("xyz"):
        for sign in (1.0, -1.0):
            shift = np.zeros(3)
            shift[axis] = sign * SHIFT_M
            out.append((f"{shift[axis] * 1e3:+.0f} mm {name}", shift))
    return out


def mat_rows(T: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in np.asarray(T, dtype=np.float64)]


def write_yaml(T_right_from_fixture: np.ndarray, T_scanner_from_fixture: np.ndarray, T_right_from_scanner: np.ndarray, rms_mm: float) -> None:
    if OUT_YAML.exists():
        raise SystemExit(f"refusing to overwrite {OUT_YAML}")
    extent = load_workcell_extrinsics().scan_extent()
    header = (
        "# Layup workcell extrinsics (ICP 2026-11-16).\n"
        "#\n"
        "# Point-to-surface fit of the scan-only mold (apron removed) onto the\n"
        f"# eight TCP touches in {CSV_PATH.name}. Labels were not used.\n"
        f"# RMS distance to the surface {rms_mm:.2f} mm. The October yaml is unchanged.\n"
        "# T_right_base_from_scanner_base is the same CAD link as that file.\n"
        "#\n"
        "# Convention: column vectors, v_a = T_a<-b @ v_b; units in meters.\n"
    )
    body = {
        "T_right_base_from_scanner_base": mat_rows(T_right_from_scanner),
        "T_right_base_from_fixture": mat_rows(T_right_from_fixture),
        "T_scanner_base_from_fixture": mat_rows(T_scanner_from_fixture),
        "scan_heatmap_extent_stringer_scanner_v2": {k: float(extent[k]) for k in ("x_min", "x_max", "z_min", "z_max")},
    }
    OUT_YAML.write_text(header + yaml.safe_dump(body, sort_keys=False))


def shifted_mesh(vertices: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Mesh moved so the unmoved TCP points lie on it. Inverse of p' = R p + t."""
    return (rot.T @ (vertices - trans).T).T


def main() -> None:
    labels, points = load_tcp(CSV_PATH)
    vertices, faces, normals = scan_mesh(FIELD_PATH)
    surface = Surface(vertices, faces, normals)
    print(f"scan-only mesh {len(vertices)} verts, {len(faces)} faces")
    best = None
    for name, shift in starts():
        fit = icp(surface, points, shift)
        ang = float(np.degrees(np.linalg.norm(Rotation.from_matrix(fit["R"]).as_rotvec())))
        print(
            f"{name:16} rms {fit['rms'] * 1e3:6.2f} mm  "
            f"rot {ang:5.2f} deg  |t| {np.linalg.norm(fit['t']) * 1e3:6.1f} mm"
        )
        if best is None or fit["rms"] < best["rms"]:
            best = {**fit, "start": name}
    rotvec = Rotation.from_matrix(best["R"]).as_rotvec()
    ang = float(np.degrees(np.linalg.norm(rotvec)))
    axis = rotvec / max(np.linalg.norm(rotvec), 1e-12)
    print(f"\nbest start: {best['start']}")
    print(f"RMS {best['rms'] * 1e3:.2f} mm")
    print(f"rotation {ang:.2f} deg about {axis[0]:+.3f} {axis[1]:+.3f} {axis[2]:+.3f}")
    print(f"translation mm {best['t'][0] * 1e3:+.2f} {best['t'][1] * 1e3:+.2f} {best['t'][2] * 1e3:+.2f}")
    for label, dist, nrm in zip(labels, best["dist"], best["normal"]):
        print(f"  {label:16} {dist * 1e3:6.2f} mm  n {nrm[0]:+.2f} {nrm[1]:+.2f} {nrm[2]:+.2f}")

    ext = load_workcell_extrinsics()
    T_old = ext.T_fixture_from_right_base
    delta = np.eye(4)
    delta[:3, :3] = best["R"]
    delta[:3, 3] = best["t"]
    T_fixture_from_base = delta @ T_old
    T_right_from_fixture = np.linalg.inv(T_fixture_from_base)
    T_scanner_from_fixture = np.linalg.inv(ext.T_right_base_from_scanner_base) @ T_right_from_fixture
    modest = ang < 20.0 and float(np.linalg.norm(best["t"])) < 0.08
    if best["rms"] < RMS_WRITE_M and modest:
        write_yaml(T_right_from_fixture, T_scanner_from_fixture, ext.T_right_base_from_scanner_base, best["rms"] * 1e3)
        print(f"wrote {OUT_YAML}")
    else:
        why = "RMS is over 3 mm" if best["rms"] >= RMS_WRITE_M else "the rigid move is too large to trust"
        print(f"not writing a yaml: {why}")

    import trimesh
    import viser

    server = viser.ViserServer(host="127.0.0.1", port=8082)
    moved = shifted_mesh(vertices, best["R"], best["t"])
    mesh = trimesh.Trimesh(moved, faces, process=False)
    mesh.visual.face_colors = np.tile(np.array([180, 180, 180, 255], dtype=np.uint8), (len(faces), 1))
    server.scene.add_mesh_trimesh("/mesh", mesh)
    for i, (label, point) in enumerate(zip(labels, points)):
        server.scene.add_icosphere(
            f"/touch/{i}", radius=0.006, color=(210, 40, 40), position=tuple(point.tolist())
        )
        server.scene.add_label(f"/touch/{i}/label", label, position=tuple((point + np.array([0.0, -0.02, 0.0])).tolist()))
    server.scene.add_frame("/fixture", axes_length=0.08, axes_radius=0.003)
    print(f"viser http://127.0.0.1:{server.get_port()}")
    print("Red dots are the recorded TCP points. The mesh is the scan-only mold moved onto them.")
    try:
        while True:
            import time
            time.sleep(1.0)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
