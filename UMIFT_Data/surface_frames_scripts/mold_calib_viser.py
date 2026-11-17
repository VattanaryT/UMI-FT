#!/usr/bin/env python3
"""Read-only viser view of the mold, the touch targets, and the live TCP.

One RTDE receive connection on the right arm (and, if it answers, the left).
This process does not command either arm. Do not also run mold_calib_check.py;
a second receive client on the same controller is dropped.

Enter records the highlighted target into the CSV, using the same row format
as mold_calib_check.py. q finishes and prints the rigid-offset summary.

Run with the geodude venv, which has viser:

  source /home/boeing/ros2_ws/src/geodude-teleoperation/.venv/bin/activate
  cd /home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames
  export PYTHONPATH="$PWD:$PWD/UMIFT_Data/surface_frames_scripts:$PYTHONPATH"
  python UMIFT_Data/surface_frames_scripts/mold_calib_viser.py --mold train \
    --points UMIFT_Data/surface_frames_scripts/mold_calib_points_working.yaml

  python UMIFT_Data/surface_frames_scripts/mold_calib_viser.py --mold train --verify \
    PyriteConfig/tasks/umift/layup_workcell_extrinsics.yaml

  python UMIFT_Data/surface_frames_scripts/mold_calib_viser.py --free 6 --no-left

Open http://127.0.0.1:8081
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import viser

_SURFACE_SCRIPTS = Path(__file__).resolve().parent
if str(_SURFACE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SURFACE_SCRIPTS))

from fixb_labels import Rp_to_pose7
from mold_aware_runtime import DEFAULT_META, MoldAwareRuntime
from mold_calib_check import (
    CSV_FIELDS,
    expected_xyz,
    fit_rigid_offset,
    load_points,
    measure_contact,
    point_passed,
    pose7_base_to_fixture,
    print_sample,
    print_summary,
    resolve_field,
    rotvec_to_pose7,
    row_from_sample,
    spot_distance_mm,
    tag_poses_in_fixture,
)
from PyriteConfig.tasks.umift.workcell_frames import load_workcell_extrinsics

DEFAULT_PORT = 8081
DEFAULT_TAGS = Path("/home/boeing/umi_ft_ws/ee_tag_traj/configs/example_stringer.yaml")
DEFAULT_LEFT_IP = "192.168.1.3"
STREAM_HZ = 30.0
TRAIL_LENGTH = 45
PASS_COLOR = (40, 170, 70)
FAIL_COLOR = (210, 40, 40)
NEXT_COLOR = (255, 120, 0)
TARGET_COLOR = (230, 190, 40)
TAG_COLOR = (70, 170, 210)


def mesh_from_field(field_path: Path) -> trimesh.Trimesh:
    data = np.load(field_path)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(data["vertices"], dtype=np.float64),
        faces=np.asarray(data["faces"], dtype=np.int64),
        process=False,
    )
    mesh.visual.face_colors = np.tile(
        np.array([180, 180, 180, 255], dtype=np.uint8), (len(mesh.faces), 1)
    )
    return mesh


def scan_only_arrays(field_path: Path):
    """Vertices, faces, and normals with the synthetic apron removed."""
    data = np.load(field_path)
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


def boundary_segments(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, set]:
    """Boundary edge segments and the set of those edges as sorted vertex pairs."""
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(
        np.ascontiguousarray(edges).view(np.dtype((np.void, edges.dtype.itemsize * 2))).ravel(),
        return_counts=True,
    )
    boundary = uniq[counts == 1].view(edges.dtype).reshape(-1, 2)
    segs = np.stack([vertices[boundary[:, 0]], vertices[boundary[:, 1]]], axis=1)
    return segs, set(map(tuple, boundary.tolist()))


class _ScanSurface:
    """Closest points on the scan-only mesh. The apron is not loaded."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray, boundary: set):
        import open3d as o3d

        self.vertices = vertices
        self.faces = faces
        self.normals = normals
        self.boundary = boundary
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(vertices.astype(np.float32))
        mesh.triangle.indices = o3d.core.Tensor(faces.astype(np.int32))
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(mesh)

    def project(self, point: np.ndarray) -> dict:
        import open3d as o3d

        query = o3d.core.Tensor(np.asarray(point, dtype=np.float32).reshape(1, 3))
        ans = self.scene.compute_closest_points(query)
        uv = ans["primitive_uvs"].numpy().astype(np.float64)[0]
        weights = np.array([1.0 - uv[0] - uv[1], uv[0], uv[1]])
        vid = self.faces[int(ans["primitive_ids"].numpy()[0])]
        foot = weights @ self.vertices[vid]
        normal = weights @ self.normals[vid]
        normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
        delta = np.asarray(point, dtype=np.float64).reshape(3) - foot
        dist = float(np.linalg.norm(delta))
        sd = float(np.dot(delta, normal))
        past = float(np.sqrt(max(dist ** 2 - sd ** 2, 0.0)))
        on_edge = False
        for i in range(3):
            if weights[i] <= 1e-3:
                a, b = int(vid[(i + 1) % 3]), int(vid[(i + 2) % 3])
                edge = (a, b) if a < b else (b, a)
                if edge in self.boundary:
                    on_edge = True
        return {"foot": foot, "normal": normal, "sd": sd, "past_edge": past, "on_edge": on_edge}


def read_touch_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _wxyz_position(R: np.ndarray, p: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    pose = Rp_to_pose7(
        np.asarray(R, dtype=np.float64).reshape(1, 3, 3),
        np.asarray(p, dtype=np.float64).reshape(1, 3),
    )[0]
    return tuple(pose[:3].tolist()), tuple(pose[3:].tolist())


def _add_pose_frame(server, name: str, R: np.ndarray, p: np.ndarray, axes_length: float, label: str):
    position, wxyz = _wxyz_position(R, p)
    frame = server.scene.add_frame(
        name, axes_length=axes_length, axes_radius=axes_length * 0.06,
        position=position, wxyz=wxyz,
    )
    text = server.scene.add_label(
        name + "/label",
        label,
        position=(position[0], position[1] - 0.02, position[2]),
    )
    return frame, text


class _TcpStream:
    """One RTDE receive process. Prints a pose line; never opens a control interface."""

    def __init__(self, python: Path, robot_ip: str, hz: float = STREAM_HZ):
        period = 1.0 / float(hz)
        code = (
            "from rtde_receive import RTDEReceiveInterface\n"
            "import time\n"
            f"r = RTDEReceiveInterface(hostname={robot_ip!r})\n"
            "while True:\n"
            "    p = r.getActualTCPPose()\n"
            "    print(','.join(str(x) for x in p), flush=True)\n"
            f"    time.sleep({period})\n"
        )
        self._proc = subprocess.Popen(
            [str(python), "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def readline(self) -> Optional[np.ndarray]:
        if self._proc.stdout is None:
            return None
        line = self._proc.stdout.readline()
        if not line:
            err = ""
            if self._proc.stderr is not None:
                err = self._proc.stderr.read()
            raise RuntimeError(f"RTDE receive stopped: {err.strip()}")
        vals = [float(x) for x in line.strip().split(",") if x.strip()]
        if len(vals) != 6:
            return None
        return np.asarray(vals, dtype=np.float64)

    def close(self) -> None:
        self._proc.terminate()


def _fit_from_rows(rows: list[dict], catalog: dict) -> Optional[dict]:
    if len(rows) < 3:
        return None
    feet = np.array(
        [[float(r["foot_x_mm"]), float(r["foot_y_mm"]), float(r["foot_z_mm"])] for r in rows]
    ) / 1000.0
    tcp = np.array(
        [[float(r["tcp_x_mm"]), float(r["tcp_y_mm"]), float(r["tcp_z_mm"])] for r in rows]
    ) / 1000.0
    return fit_rigid_offset(
        feet, tcp, catalog["stringer_axis"], catalog["across_axis"], catalog["crown_normal"]
    )


def _free_row(index: int, tcp_m: np.ndarray, hit: dict) -> dict:
    foot = hit["foot"] * 1e3
    tcp = np.asarray(tcp_m, dtype=np.float64).reshape(3) * 1e3
    offset = tcp - foot
    normal = hit["normal"]
    on_edge = bool(hit["on_edge"] and hit["past_edge"] > 0.002)
    return {
        "label": f"free_{index}",
        "unix_s": f"{datetime.now().timestamp():.3f}",
        "sd_mm": f"{hit['sd'] * 1e3:.4f}",
        "foot_x_mm": f"{foot[0]:.3f}",
        "foot_y_mm": f"{foot[1]:.3f}",
        "foot_z_mm": f"{foot[2]:.3f}",
        "n_x": f"{normal[0]:.6f}",
        "n_y": f"{normal[1]:.6f}",
        "n_z": f"{normal[2]:.6f}",
        "angle_n_toolz_deg": "0.000",
        "normal_error_deg": "0.000",
        "tcp_x_mm": f"{tcp[0]:.3f}",
        "tcp_y_mm": f"{tcp[1]:.3f}",
        "tcp_z_mm": f"{tcp[2]:.3f}",
        "offset_x_mm": f"{offset[0]:.3f}",
        "offset_y_mm": f"{offset[1]:.3f}",
        "offset_z_mm": f"{offset[2]:.3f}",
        "passed": "no" if on_edge else "yes",
        "spot_warn": "yes" if on_edge else "no",
        "spot_dist_mm": f"{hit['past_edge'] * 1e3:.2f}",
    }


def free_session(args) -> int:
    """Record unlabeled TCP touches on the scan-only mesh. Does not command the arm."""
    field_path = resolve_field(args.mold, args.field)
    if not field_path.is_file():
        print(f"Mold field not found: {field_path}", file=sys.stderr)
        return 1
    extrinsics = load_workcell_extrinsics()
    vertices, faces, normals = scan_only_arrays(field_path)
    segs, boundary = boundary_segments(vertices, faces)
    surface = _ScanSurface(vertices, faces, normals, boundary)
    count = int(args.free)
    csv_path = args.csv
    if csv_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Path.cwd() / f"mold_calib_free_{stamp}.csv"
    csv_path = Path(csv_path).expanduser().resolve()
    if csv_path.exists():
        print(f"refusing to overwrite {csv_path}", file=sys.stderr)
        return 1
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.add_frame("/fixture", axes_length=0.08, axes_radius=0.003)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.visual.face_colors = np.tile(np.array([180, 180, 180, 255], dtype=np.uint8), (len(faces), 1))
    server.scene.add_mesh_trimesh("/mold", mesh)
    server.scene.add_line_segments("/edge", points=segs, colors=(255, 140, 0), thickness=0.0015)
    tcp_frame = server.scene.add_frame("/tcp", axes_length=0.05, axes_radius=0.002)
    foot_sphere = server.scene.add_icosphere("/foot", radius=0.004, color=(30, 120, 220), position=(0.0, 0.0, 0.0))
    guide = server.gui.add_markdown(
        f"Record **{count}** touches on the gray surface: 2 on the crown, 2 on each wall, "
        "mostly mid-length. Stay off the orange edges and off the flanges. "
        "Click **Record touch** while the tool is in contact. This does not move the arm."
    )
    status = server.gui.add_markdown("Waiting for TCP.")
    record_btn = server.gui.add_button("Record touch", color=(210, 40, 40))
    undo_btn = server.gui.add_button("Undo last")
    lock = threading.Lock()
    write_lock = threading.Lock()
    state = {"quit": False, "tcp": None, "hit": None, "line": "waiting for TCP", "n": 0}
    touch_handles: list = []

    def refresh() -> None:
        for handle in touch_handles:
            handle.remove()
        touch_handles.clear()
        rows = read_touch_csv(csv_path)
        state["n"] = len(rows)
        record_btn.disabled = len(rows) >= count
        for i, row in enumerate(rows):
            pos = (
                float(row["tcp_x_mm"]) / 1000.0,
                float(row["tcp_y_mm"]) / 1000.0,
                float(row["tcp_z_mm"]) / 1000.0,
            )
            color = FAIL_COLOR if row.get("spot_warn") == "yes" else PASS_COLOR
            sphere = server.scene.add_icosphere(f"/touches/{i}", radius=0.006, color=color, position=pos)
            touch_handles.append(sphere)
        guide.content = (
            f"Recorded **{len(rows)}/{count}**. "
            + (
                f"Saved `{csv_path.name}`."
                if len(rows) >= count
                else "2 on the crown, 2 on each wall, mid-length, off the orange edges."
            )
        )

    def append_touch() -> None:
        with lock:
            tcp = None if state["tcp"] is None else np.array(state["tcp"], dtype=np.float64)
            hit = None if state["hit"] is None else dict(state["hit"])
        if tcp is None or hit is None:
            return
        with write_lock:
            n = len(read_touch_csv(csv_path))
            if n >= count:
                return
            row = _free_row(n + 1, tcp, hit)
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
        where = "EDGE" if row["spot_warn"] == "yes" else "surface"
        print(
            f"recorded {row['label']}  tcp mm {row['tcp_x_mm']} {row['tcp_y_mm']} {row['tcp_z_mm']}  "
            f"sd {row['sd_mm']} mm  {where}",
            flush=True,
        )
        refresh()

    def undo_touch() -> None:
        with write_lock:
            rows = read_touch_csv(csv_path)
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                if rows:
                    writer.writerows(rows[:-1])
        print(f"undo, {max(len(rows) - 1, 0)} touches left", flush=True)
        refresh()

    @record_btn.on_click
    def _record(_event) -> None:
        append_touch()

    @undo_btn.on_click
    def _undo(_event) -> None:
        undo_touch()

    def live_loop() -> None:
        stream = None
        try:
            stream = _TcpStream(args.rtde_python, args.robot_ip)
            while not state["quit"]:
                tcp6 = stream.readline()
                if tcp6 is None:
                    continue
                pose_fix = pose7_base_to_fixture(
                    rotvec_to_pose7(tcp6), extrinsics.T_fixture_from_right_base
                ).reshape(7)
                tcp = np.asarray(pose_fix[:3], dtype=np.float64)
                hit = surface.project(tcp)
                tcp_frame.position = tuple(tcp.tolist())
                tcp_frame.wxyz = tuple(pose_fix[3:].tolist())
                foot_sphere.position = tuple(hit["foot"].tolist())
                beyond = hit["on_edge"] and hit["past_edge"] > 0.002
                line = (
                    f"sd={hit['sd'] * 1e3:+.1f} mm  "
                    + (
                        f"EDGE, {hit['past_edge'] * 1e3:.1f} mm past the scan"
                        if beyond
                        else "on the scan surface"
                    )
                )
                with lock:
                    state["tcp"] = tcp
                    state["hit"] = hit
                    state["line"] = line
        except Exception as exc:
            with lock:
                state["line"] = f"RTDE receive stopped: {exc}"
        finally:
            if stream is not None:
                stream.close()

    threading.Thread(target=live_loop, name="mold-calib-tcp", daemon=True).start()
    refresh()
    print(f"mesh: {field_path} (scan only, apron removed)")
    print(f"extrinsics: {extrinsics.path}")
    print(f"csv: {csv_path}")
    print(f"viser http://127.0.0.1:{int(server.get_port())}")
    print("Read-only. This process does not move the arm.")
    print(f"Record {count} free touches with the Record touch button.")
    try:
        while not state["quit"]:
            with lock:
                line = state["line"]
                n = state["n"]
            status.content = f"**{n}/{count}**  {line}"
            time.sleep(0.5)
    except KeyboardInterrupt:
        state["quit"] = True
    print(f"csv: {csv_path}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mold", choices=("train", "heldout"), default="train")
    parser.add_argument(
        "--free",
        type=int,
        default=0,
        help="Record this many unlabeled touches on the scan-only mesh. No target labels.",
    )
    parser.add_argument("--field", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--robot-ip", default="192.168.1.2")
    parser.add_argument("--left-ip", default=DEFAULT_LEFT_IP)
    parser.add_argument("--no-left", action="store_true", help="Do not open a receive connection on the left arm.")
    parser.add_argument("--tags", type=Path, default=DEFAULT_TAGS)
    parser.add_argument(
        "--points",
        type=Path,
        default=None,
        help="Touch-point yaml. Default is mold_calib_points.yaml.",
    )
    parser.add_argument(
        "--rtde-python",
        type=Path,
        default=Path("/home/boeing/keyence_scan_ws/.venv/bin/python"),
    )
    parser.add_argument(
        "--verify",
        type=Path,
        default=None,
        help="Extrinsics yaml. Print live signed distance and tilt. Does not record or refit.",
    )
    parser.add_argument("--sd-mm", type=float, default=3.0)
    parser.add_argument("--normal-deg", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.free:
        if args.verify is not None:
            print("--free does not take --verify", file=sys.stderr)
            return 1
        return free_session(args)

    field_path = resolve_field(args.mold, args.field)
    if not field_path.is_file():
        print(f"Mold field not found: {field_path}", file=sys.stderr)
        return 1
    verify = args.verify is not None
    if verify:
        extrinsics = load_workcell_extrinsics(str(args.verify))
    else:
        extrinsics = load_workcell_extrinsics()
    catalog = load_points(args.points) if args.points is not None else load_points()
    points = catalog["points"]
    meta_path = field_path.with_name("fixb_meta.json")
    if not meta_path.is_file():
        meta_path = DEFAULT_META
    runtime = MoldAwareRuntime.from_assets(
        run_type="r1_world_world",
        field_path=field_path,
        meta_path=meta_path,
        dry_run=True,
    )
    runtime.reset()

    csv_path = None
    if not verify:
        csv_path = args.csv
        if csv_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = Path.cwd() / f"mold_calib_{args.mold}_{stamp}.csv"
        csv_path = csv_path.expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not csv_path.is_file():
            with open(csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.add_frame("/fixture", axes_length=0.08, axes_radius=0.003)
    server.scene.add_mesh_trimesh("/mold", mesh_from_field(field_path))
    tcp_frame = server.scene.add_frame("/tcp", axes_length=0.05, axes_radius=0.002)
    left_frame = server.scene.add_frame("/left_tcp", axes_length=0.05, axes_radius=0.002, visible=False)
    foot_sphere = server.scene.add_icosphere(
        "/foot", radius=0.004, color=(30, 120, 220), position=(0.0, 0.0, 0.0)
    )
    normal_line = server.scene.add_line_segments(
        "/normal",
        points=np.zeros((1, 2, 3)),
        colors=(30, 120, 220),
        thickness=0.002,
    )
    to_target = server.scene.add_line_segments(
        "/to_target",
        points=np.zeros((1, 2, 3)),
        colors=NEXT_COLOR,
        thickness=0.002,
    )
    trail = server.scene.add_point_cloud(
        "/tcp_trail",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.array([255, 160, 40], dtype=np.uint8),
        point_size=0.005,
        point_shape="circle",
    )
    trail_pts: deque[np.ndarray] = deque(maxlen=TRAIL_LENGTH)

    _add_pose_frame(
        server, "/right_base",
        extrinsics.T_fixture_from_right_base[:3, :3],
        extrinsics.T_fixture_from_right_base[:3, 3],
        0.12, "right_base",
    )
    T_fixture_from_scanner = np.linalg.inv(extrinsics.T_scanner_base_from_fixture)
    _add_pose_frame(
        server, "/scanner_base",
        T_fixture_from_scanner[:3, :3],
        T_fixture_from_scanner[:3, 3],
        0.12, "scanner_base",
    )

    target_xyz = []
    normal_segs = []
    for point in points:
        xyz = np.asarray(expected_xyz(point, args.mold), dtype=np.float64).reshape(3)
        queried = runtime.field.query(xyz)
        R = np.asarray(queried["R"], dtype=np.float64).reshape(3, 3)
        normal = R[:, 2]
        normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
        target_xyz.append(xyz)
        normal_segs.append(np.stack([xyz, xyz + 0.04 * normal]))
        _add_pose_frame(
            server, f"/targets/{point['label']}", R, xyz, 0.03, point["label"],
        )
    target_xyz = np.stack(target_xyz)
    server.scene.add_line_segments(
        "/target_normals",
        points=np.stack(normal_segs),
        colors=TARGET_COLOR,
        thickness=0.002,
    )
    next_marker = server.scene.add_icosphere(
        "/next", radius=0.012, color=NEXT_COLOR, position=tuple(target_xyz[0].tolist()),
    )

    if args.tags.is_file():
        for tag in tag_poses_in_fixture(args.tags):
            _add_pose_frame(
                server, f"/tags/tag{tag['id']}", tag["R"], tag["xyz_m"], 0.04, f"tag{tag['id']}",
            )
    else:
        print(f"Tag config not found: {args.tags}", file=sys.stderr)

    show_shift = server.gui.add_checkbox("Show rigid translation arrow", initial_value=False)
    status = server.gui.add_markdown("Waiting for TCP.")
    shift_arrow = server.scene.add_arrows(
        "/shift",
        points=np.zeros((1, 2, 3)),
        colors=(230, 160, 20),
        shaft_radius=0.002,
        head_radius=0.005,
        head_length=0.012,
        visible=False,
    )
    touch_handles: list = []
    lock = threading.Lock()
    state = {
        "line": "waiting for TCP",
        "left": "left arm not started",
        "sample": None,
        "tcp_xyz": None,
        "next": len(read_touch_csv(csv_path)) if csv_path is not None else 0,
        "quit": False,
        "rows": read_touch_csv(csv_path) if csv_path is not None else [],
    }

    def place_next() -> None:
        index = int(state["next"])
        if index >= len(target_xyz):
            next_marker.visible = False
            return
        next_marker.visible = True
        next_marker.position = tuple(target_xyz[index].tolist())

    def refresh_touches() -> None:
        for handle in touch_handles:
            handle.remove()
        touch_handles.clear()
        rows = read_touch_csv(csv_path) if csv_path is not None else []
        state["rows"] = rows
        for i, row in enumerate(rows):
            pos = (
                float(row["tcp_x_mm"]) / 1000.0,
                float(row["tcp_y_mm"]) / 1000.0,
                float(row["tcp_z_mm"]) / 1000.0,
            )
            color = PASS_COLOR if row.get("passed") == "yes" else FAIL_COLOR
            name = f"/touches/{i}_{row['label']}"
            sphere = server.scene.add_icosphere(name, radius=0.007, color=color, position=pos)
            text = f"{row['label']}  sd={float(row['sd_mm']):+.1f} mm"
            label = server.scene.add_label(
                name + "/label",
                text,
                position=(pos[0], pos[1] - 0.015, pos[2]),
            )
            touch_handles.append(sphere)
            touch_handles.append(label)

            def _on_click(_event, text=text) -> None:
                state["touch"] = text

            sphere.on_click(_on_click)
        fit = _fit_from_rows(rows, catalog)
        if fit is None:
            shift_arrow.visible = False
            return
        start = fit["center_m"]
        end = start + fit["t_m"]
        shift_arrow.points = np.stack([start, end])[None, :, :]
        shift_arrow.visible = bool(show_shift.value)

    @show_shift.on_update
    def _toggle(_event) -> None:
        refresh_touches()

    state["touch"] = "click a recorded marker for its signed distance"
    place_next()

    def live_loop() -> None:
        stream = None
        try:
            stream = _TcpStream(args.rtde_python, args.robot_ip)
            while not state["quit"]:
                tcp6 = stream.readline()
                if tcp6 is None:
                    continue
                pose7 = rotvec_to_pose7(tcp6)
                sample = measure_contact(
                    runtime, extrinsics.T_fixture_from_right_base, pose7
                )
                pose_fix = pose7_base_to_fixture(
                    pose7, extrinsics.T_fixture_from_right_base
                ).reshape(7)
                tcp_xyz = np.asarray(pose_fix[:3], dtype=np.float64)
                tcp_frame.position = tuple(tcp_xyz.tolist())
                tcp_frame.wxyz = tuple(pose_fix[3:].tolist())
                foot = np.asarray(sample["foot_m"], dtype=np.float64)
                foot_sphere.position = tuple(foot.tolist())
                n = np.asarray(sample["normal"], dtype=np.float64)
                n = n / max(float(np.linalg.norm(n)), 1e-12)
                normal_line.points = np.stack([foot, foot + 0.03 * n])[None, :, :]
                trail_pts.append(tcp_xyz)
                trail.points = np.stack(trail_pts).astype(np.float32)
                index = int(state["next"])
                if verify:
                    line = (
                        f"sd={sample['sd_m'] * 1e3:+.1f} mm  "
                        f"tilt={sample['normal_error_deg']:.1f} deg"
                    )
                elif index < len(target_xyz):
                    goal = target_xyz[index]
                    dist_mm = float(np.linalg.norm(tcp_xyz - goal) * 1e3)
                    to_target.points = np.stack([tcp_xyz, goal])[None, :, :]
                    to_target.visible = True
                    name = points[index]["label"]
                    line = (
                        f"next {index + 1}/{len(points)} {name}  to_target={dist_mm:.1f} mm  "
                        f"sd={sample['sd_m'] * 1e3:+.1f} mm  "
                        f"tilt={sample['normal_error_deg']:.1f} deg"
                    )
                else:
                    to_target.visible = False
                    line = (
                        f"all {len(points)} targets recorded  "
                        f"sd={sample['sd_m'] * 1e3:+.1f} mm  "
                        f"tilt={sample['normal_error_deg']:.1f} deg"
                    )
                with lock:
                    state["sample"] = sample
                    state["tcp_xyz"] = tcp_xyz
                    state["line"] = line
        except Exception as exc:
            with lock:
                state["line"] = f"RTDE receive stopped: {exc}"
        finally:
            if stream is not None:
                stream.close()

    def left_loop() -> None:
        stream = None
        try:
            stream = _TcpStream(args.rtde_python, args.left_ip)
            left_frame.visible = True
            while not state["quit"]:
                tcp6 = stream.readline()
                if tcp6 is None:
                    continue
                pose_fix = pose7_base_to_fixture(
                    rotvec_to_pose7(tcp6), T_fixture_from_scanner
                ).reshape(7)
                left_frame.position = tuple(pose_fix[:3].tolist())
                left_frame.wxyz = tuple(pose_fix[3:].tolist())
                with lock:
                    state["left"] = "left TCP live"
        except Exception as exc:
            with lock:
                state["left"] = f"left arm receive stopped: {exc}"
            left_frame.visible = False
        finally:
            if stream is not None:
                stream.close()

    def record_loop() -> None:
        while not state["quit"]:
            try:
                typed = input("record> ").strip()
            except EOFError:
                state["quit"] = True
                return
            if typed.lower() in {"q", "quit", "exit"}:
                state["quit"] = True
                return
            index = int(state["next"])
            if index >= len(points):
                print(f"All {len(points)} targets are already recorded.")
                continue
            with lock:
                sample = state["sample"]
            if sample is None:
                print("No TCP yet.")
                continue
            point = points[index]
            ok = point_passed(
                sample["sd_m"] * 1e3,
                sample["normal_error_deg"],
                args.sd_mm,
                args.normal_deg,
            )
            dist_mm = spot_distance_mm(sample["foot_m"], target_xyz[index])
            warn = dist_mm > catalog["spot_warn_mm"]
            row = row_from_sample(point["label"], sample, ok, dist_mm, warn)
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
            print()
            print_sample(point["label"], row, ok)
            if warn:
                print(
                    f"  WARN foot is {dist_mm:.1f} mm from the expected spot "
                    f"(limit {catalog['spot_warn_mm']:.0f} mm). Likely the wrong corner."
                )
            state["next"] = index + 1
            place_next()
            refresh_touches()

    threading.Thread(target=live_loop, name="mold-calib-tcp", daemon=True).start()
    if not args.no_left:
        threading.Thread(target=left_loop, name="mold-calib-left", daemon=True).start()
    if not verify:
        threading.Thread(target=record_loop, name="mold-calib-record", daemon=True).start()
    refresh_touches()
    port = int(server.get_port())
    print(f"mesh: {field_path}")
    print(f"extrinsics: {extrinsics.path}")
    print(f"viser http://127.0.0.1:{port}")
    print("Read-only. This process does not move either arm.")
    if verify:
        print("Verify only. No recording and no refit.")
        print("Live line is signed distance and tilt at the TCP.")
    else:
        print(f"csv: {csv_path}")
        print("Do not run mold_calib_check.py in another shell. It would open a second receive client.")
        print("Orange marker is the next target. Enter records it. q finishes.")
        if state["next"] < len(points):
            nxt = points[state["next"]]
            print(f"Next {state['next'] + 1}/{len(points)}  {nxt['label']}")
            print(f"  {nxt['description']}")
    try:
        while not state["quit"]:
            with lock:
                line = state["line"]
                left = state["left"]
                touch = state["touch"]
            status.content = f"**Live** {line}\n\n{left}\n\n{touch}"
            print(line, flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        state["quit"] = True
    print()
    if not verify:
        print_summary(state["rows"], catalog)
        print(f"csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
