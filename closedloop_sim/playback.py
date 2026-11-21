#!/usr/bin/env python3
"""Closed-loop traces on the existing viser page. No second server and no policy.

Reads closedloop_sim/out/policy_*_fixb_ep40.npz and demo_ep40.npz. The mold is
fixb_field.npz. Penetration colors the rollout path and the tool marker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

OUT = Path(__file__).resolve().parent / "out"
FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
RUNS = ("fk60", "e40", "r1rel")
MESH_TRI_CAP = 8000


def _pose_wxyz(pose7: np.ndarray) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(7)
    return pose7[3:7]


def _segments(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 2:
        zero = np.zeros((1, 2, 3), dtype=np.float64)
        return zero
    return np.stack([points[:-1], points[1:]], axis=1)


def _stride(n: int, cap: int) -> np.ndarray:
    step = max(1, int(np.ceil(n / cap)))
    return np.arange(0, n, step)


def _pen_rgb(penetration_m: np.ndarray) -> np.ndarray:
    """0 mm is green. 15 mm is red."""
    level = np.clip(np.asarray(penetration_m, dtype=np.float64) * 1e3 / 15.0, 0.0, 1.0)
    color = np.zeros((len(level), 3), dtype=np.uint8)
    color[:, 0] = (40 + 200 * level).astype(np.uint8)
    color[:, 1] = (180 * (1.0 - level)).astype(np.uint8)
    color[:, 2] = 40
    return color


def _load_mesh(path: Path) -> trimesh.Trimesh:
    data = np.load(path)
    vertices = np.asarray(data["vertices"], dtype=np.float64)
    faces = np.asarray(data["faces"], dtype=np.int64)
    if "is_apron" in data.files:
        apron = np.asarray(data["is_apron"], dtype=bool)
        faces = faces[~apron[faces].any(axis=1)]
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    vertices = vertices[used]
    faces = remap[faces]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if len(mesh.faces) > MESH_TRI_CAP:
        try:
            import open3d as o3d

            cloud = o3d.geometry.TriangleMesh()
            cloud.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
            cloud.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
            cloud = cloud.simplify_quadric_decimation(target_number_of_triangles=MESH_TRI_CAP)
            mesh = trimesh.Trimesh(
                vertices=np.asarray(cloud.vertices),
                faces=np.asarray(cloud.triangles),
                process=False,
            )
        except Exception:
            mesh = mesh.simplify_quadric_decimation(MESH_TRI_CAP)
    mesh.visual.face_colors = np.tile(
        np.array([176, 176, 176, 255], dtype=np.uint8), (len(mesh.faces), 1)
    )
    return mesh


def _load_run(path: Path) -> dict:
    blob = np.load(path)
    position = np.asarray(blob["position"], dtype=np.float64)
    normal = np.asarray(blob["normal"], dtype=np.float64)
    delta_n = np.asarray(blob["delta_n_m"], dtype=np.float64).reshape(-1)
    if "virtual_target" in blob.files:
        virtual = np.asarray(blob["virtual_target"], dtype=np.float64)
    else:
        virtual = position + delta_n[:, None] * normal
    if "force" in blob.files:
        force = np.asarray(blob["force"], dtype=np.float64)
    else:
        contact = np.asarray(blob["contact_n"], dtype=np.float64).reshape(-1)
        force = contact[:, None] * normal
    return {
        "t_s": np.asarray(blob["t_s"], dtype=np.float64).reshape(-1),
        "position": position,
        "pose7": np.asarray(blob["pose7"], dtype=np.float64),
        "virtual_target": virtual,
        "force": force,
        "penetration_m": np.asarray(blob["penetration_m"], dtype=np.float64).reshape(-1),
        "speed_mps": np.asarray(blob["speed_mps"], dtype=np.float64).reshape(-1),
    }


def _load_demo(path: Path) -> dict | None:
    if not path.is_file():
        return None
    blob = np.load(path)
    return {
        "t_s": np.asarray(blob["t_s"], dtype=np.float64).reshape(-1),
        "position": np.asarray(blob["position"], dtype=np.float64),
    }


class Playback:
    def __init__(self, server, trace_dir: Path = OUT, field_path: Path = FIELD):
        self.server = server
        self.trace_dir = Path(trace_dir)
        self.field_path = Path(field_path)
        self.runs = {}
        self.demo = None
        self._visible = True
        zero = np.zeros((1, 2, 3), dtype=np.float64)
        self.mold = server.scene.add_mesh_trimesh("/closedloop/mold", _load_mesh(self.field_path))
        self.demo_path = server.scene.add_line_segments(
            "/closedloop/demo", points=zero, colors=(150, 150, 150), thickness=0.0016
        )
        self.roll_path = server.scene.add_line_segments(
            "/closedloop/rollout", points=zero, colors=(40, 180, 60), thickness=0.0024
        )
        self.tool = server.scene.add_frame(
            "/closedloop/tool",
            axes_length=0.04,
            axes_radius=0.002,
            origin_radius=0.006,
            origin_color=(40, 180, 60),
        )
        self.target = server.scene.add_icosphere(
            "/closedloop/virtual_target",
            radius=0.005,
            color=(0, 190, 210),
            subdivisions=1,
        )
        self.force = server.scene.add_line_segments(
            "/closedloop/force", points=zero, colors=(210, 40, 40), thickness=0.003
        )
        self.which = server.gui.add_dropdown("Run", options=["fk60"], initial_value="fk60")
        self.slider = server.gui.add_slider(
            "Time (s)", min=0.0, max=60.0, step=0.05, initial_value=0.0
        )
        self.status = server.gui.add_markdown("Closed-loop playback.")
        server.gui.add_markdown(
            "Gray path is the episode 40 demo. The colored path is the rollout: "
            "green is no penetration, red is 15 mm. Cyan is the virtual target. "
            "The red arrow is the contact force, 2 mm per newton."
        )
        reload = server.gui.add_button("Reload traces")
        self.which.on_update(lambda _event: self._draw())
        self.slider.on_update(lambda _event: self._draw())
        reload.on_click(lambda _event: self.reload())
        self.reload()

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        for handle in (
            self.mold,
            self.demo_path,
            self.roll_path,
            self.tool,
            self.target,
            self.force,
        ):
            handle.visible = self._visible

    def reload(self) -> None:
        found = {}
        for name in RUNS:
            path = self.trace_dir / f"policy_{name}_fixb_ep40.npz"
            if path.is_file() and "position" in np.load(path, mmap_mode="r").files:
                found[name] = _load_run(path)
        self.runs = found
        self.demo = _load_demo(self.trace_dir / "demo_ep40.npz")
        options = list(found) or ["none"]
        self.which.options = options
        if self.which.value not in options:
            self.which.value = options[0]
        self._draw_demo()
        self._draw()

    def _draw_demo(self) -> None:
        if self.demo is None:
            self.demo_path.points = np.zeros((1, 2, 3))
            return
        index = _stride(len(self.demo["position"]), 900)
        self.demo_path.points = _segments(self.demo["position"][index])
        self.demo_path.visible = self._visible

    def _draw(self) -> None:
        name = str(self.which.value)
        run = self.runs.get(name)
        if run is None:
            self.status.content = "No closed-loop trace in closedloop_sim/out."
            return
        t = float(self.slider.value)
        row = int(np.argmin(np.abs(run["t_s"] - t)))
        shown = _stride(row + 1, 900)
        points = run["position"][shown]
        segments = _segments(points)
        self.roll_path.points = segments
        colors = _pen_rgb(run["penetration_m"][shown])
        self.roll_path.colors = colors[:-1] if len(points) >= 2 else np.array([[40, 180, 60]], dtype=np.uint8)
        pose = run["pose7"][row]
        self.tool.position = pose[:3]
        self.tool.wxyz = _pose_wxyz(pose)
        pen_mm = float(run["penetration_m"][row] * 1e3)
        pen_color = _pen_rgb(np.array([run["penetration_m"][row]]))[0]
        self.tool.origin_color = tuple(int(v) for v in pen_color)
        self.target.position = run["virtual_target"][row]
        force = run["force"][row]
        scale = 0.002
        self.force.points = np.stack([pose[:3], pose[:3] + force * scale])[None, :, :]
        demo_mm = None
        if self.demo is not None:
            demo_row = int(np.argmin(np.abs(self.demo["t_s"] - run["t_s"][row])))
            demo_mm = float(np.linalg.norm(run["position"][row] - self.demo["position"][demo_row]) * 1e3)
        away = "" if demo_mm is None else f" · {demo_mm:.0f} mm from demo"
        self.status.content = (
            f"**{name}** t={run['t_s'][row]:.2f} s · penetration {pen_mm:.1f} mm · "
            f"force {np.linalg.norm(force):.1f} N · speed {run['speed_mps'][row]:.3f} m/s"
            f"{away}"
        )
        for handle in (self.roll_path, self.tool, self.target, self.force, self.demo_path):
            handle.visible = self._visible


def attach_playback(server, trace_dir: Path = OUT, field_path: Path = FIELD) -> Playback:
    return Playback(server, trace_dir, field_path)
