#!/usr/bin/env python3
"""Move the scan-only mesh against the 12 recorded touches.

The touches stay put. Sliders start at the approximate 6-point extrinsics and
move the mesh in the fixture frame. Does not command the robot.

  source /home/boeing/ros2_ws/src/geodude-teleoperation/.venv/bin/activate
  export PYTHONPATH="$PWD:$PWD/UMIFT_Data/surface_frames_scripts:$PYTHONPATH"
  python UMIFT_Data/surface_frames_scripts/mold_calib_manual_view.py
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import mold_calib_icp_view as icp
from PyriteConfig.tasks.umift.workcell_frames import load_workcell_extrinsics

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
OCTOBER = ROOT / "PyriteConfig/tasks/umift/layup_workcell_extrinsics.yaml"
APPROX = ROOT / "PyriteConfig/tasks/umift/layup_workcell_extrinsics_20261116_approx.yaml"
MANUAL = ROOT / "PyriteConfig/tasks/umift/layup_workcell_extrinsics_20261116_manual.yaml"
FREE_CSV = ROOT / "mold_calib_free_20261116_160630.csv"
DROP = {"flange_l_near", "flange_r_far"}
EARLY_COLOR = (40, 110, 230)
FREE_COLOR = (230, 120, 30)


def twelve() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Labels, points, and a bool mask that is True for the 14:54 session."""
    labels_a, pts_a = icp.load_tcp(icp.CSV_PATH)
    labels_b, pts_b = icp.load_tcp(FREE_CSV)
    labels, pts, early = [], [], []
    for lab, p in zip(labels_a, pts_a):
        if lab in DROP:
            continue
        labels.append(lab)
        pts.append(p)
        early.append(True)
    for lab, p in zip(labels_b, pts_b):
        labels.append(lab)
        pts.append(p)
        early.append(False)
    return labels, np.stack(pts), np.asarray(early, dtype=bool)


def delta_from_yaml() -> np.ndarray:
    """Point map of the approx yaml: p_on_mesh = delta @ p_october_fixture."""
    old = load_workcell_extrinsics(str(OCTOBER))
    approx = load_workcell_extrinsics(str(APPROX))
    return np.linalg.inv(approx.T_right_base_from_fixture) @ old.T_right_base_from_fixture


def slider_rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Roll about +X, then pitch about +Y, then yaw about +Z."""
    return Rotation.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()


def query_points(
    points: np.ndarray,
    delta: np.ndarray,
    rot: np.ndarray,
    trans_m: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    """Map fixed touches into the original mesh frame for a closest-point query."""
    inv = (rot.T @ (points - trans_m - center).T).T + center
    return (delta[:3, :3] @ inv.T).T + delta[:3, 3]


def manual_right_base(delta: np.ndarray, rot: np.ndarray, trans_m: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extrinsics after the slider motion S of the approx-aligned mesh."""
    # S @ v = rot @ v + (center + trans - rot @ center)
    trans_s = center + trans_m - rot @ center
    s_inv_rot = rot.T
    s_inv_trans = -rot.T @ trans_s
    s_inv = np.eye(4)
    s_inv[:3, :3] = s_inv_rot
    s_inv[:3, 3] = s_inv_trans
    delta_total = delta @ s_inv
    old = load_workcell_extrinsics(str(OCTOBER))
    # T_fixture_from_base = delta_total @ T_october, so the right-base
    # transform is T_october_right @ inv(delta_total).
    right = old.T_right_base_from_fixture @ np.linalg.inv(delta_total)
    scanner = np.linalg.inv(old.T_right_base_from_scanner_base) @ right
    return right, scanner


def _flow(T: np.ndarray) -> str:
    lines = []
    for row in np.asarray(T, dtype=np.float64):
        lines.append("  - [" + ", ".join(f"{float(v):.8f}" for v in row) + "]")
    return "\n".join(lines)


def write_manual(
    right: np.ndarray,
    scanner: np.ndarray,
    slider_mm: np.ndarray,
    slider_deg: np.ndarray,
    rms_mm: float,
) -> None:
    if MANUAL.exists():
        raise FileExistsError(MANUAL)
    october = load_workcell_extrinsics(str(OCTOBER))
    extent = october.raw["scan_heatmap_extent_stringer_scanner_v2"]
    text = (
        "# Layup workcell extrinsics (manual slider pose, 2026-11-16).\n"
        "#\n"
        "# Starting point: layup_workcell_extrinsics_20261116_approx.yaml.\n"
        "# Sliders move the scan-only mesh in the fixture frame, about the\n"
        "# centroid of the 12 touches. Roll is about +X, then pitch about +Y,\n"
        "# then yaw about +Z.\n"
        f"# Slider translation mm: {slider_mm[0]:+.2f} {slider_mm[1]:+.2f} {slider_mm[2]:+.2f}\n"
        f"# Slider rotation deg: roll {slider_deg[0]:+.2f}  pitch {slider_deg[1]:+.2f}  "
        f"yaw {slider_deg[2]:+.2f}\n"
        f"# RMS of all 12 touches to the scan-only surface: {rms_mm:.2f} mm.\n"
        "# The October yaml and the approx yaml are unchanged.\n"
        "# T_right_base_from_scanner_base is the same CAD link.\n"
        "#\n"
        "# Convention: column vectors, v_a = T_a<-b @ v_b; units in meters.\n"
        "\n"
        "T_right_base_from_scanner_base:\n"
        f"{_flow(october.T_right_base_from_scanner_base)}\n"
        "\n"
        "T_right_base_from_fixture:\n"
        f"{_flow(right)}\n"
        "\n"
        "T_scanner_base_from_fixture:\n"
        f"{_flow(scanner)}\n"
        "\n"
        "scan_heatmap_extent_stringer_scanner_v2:\n"
        f"  x_min: {extent['x_min']}\n"
        f"  x_max: {extent['x_max']}\n"
        f"  z_min: {extent['z_min']}\n"
        f"  z_max: {extent['z_max']}\n"
    )
    MANUAL.write_text(text)


def main() -> None:
    labels, points, early = twelve()
    delta = delta_from_yaml()
    center = points.mean(axis=0)
    right0, _ = manual_right_base(delta, np.eye(3), np.zeros(3), center)
    approx = load_workcell_extrinsics(str(APPROX)).T_right_base_from_fixture
    if not np.allclose(right0, approx, atol=1e-8):
        raise SystemExit("zero sliders do not reproduce the approx yaml")

    vertices, faces, _normals = icp.scan_mesh(icp.FIELD_PATH)
    surface = icp.Surface(vertices, faces, _normals)
    aligned = icp.shifted_mesh(vertices, delta[:3, :3], delta[:3, 3])

    import trimesh
    import viser

    server = viser.ViserServer(host="0.0.0.0", port=8081)
    server.scene.add_frame("/fixture", axes_length=0.08, axes_radius=0.003)
    mesh = trimesh.Trimesh(aligned, faces, process=False)
    mesh.visual.face_colors = np.tile(np.array([180, 180, 180, 255], dtype=np.uint8), (len(faces), 1))
    mesh_handle = server.scene.add_mesh_trimesh("/mesh", mesh)
    spheres = []
    texts = []
    for i, (label, point, is_early) in enumerate(zip(labels, points, early)):
        color = EARLY_COLOR if is_early else FREE_COLOR
        spheres.append(
            server.scene.add_icosphere(
                f"/touch/{i}",
                radius=0.006,
                color=color,
                position=tuple(float(v) for v in point),
            )
        )
        texts.append(
            server.scene.add_label(
                f"/touch/{i}/label",
                label,
                position=tuple(float(v) for v in point + np.array([0.0, -0.015, 0.0])),
            )
        )

    server.gui.add_markdown(
        "Blue dots are the six non-flange touches from `mold_calib_train_20261116_145442.csv`. "
        "Orange dots are the six free touches. The gray mesh is the scan only. "
        "Sliders move the mesh from the approx yaml. Rotation is about the centroid of the 12 dots. "
        "This does not move the arm."
    )
    sliders = {
        "x": server.gui.add_slider("x mm", min=-60.0, max=60.0, step=0.5, initial_value=0.0),
        "y": server.gui.add_slider("y mm", min=-60.0, max=60.0, step=0.5, initial_value=0.0),
        "z": server.gui.add_slider("z mm", min=-60.0, max=60.0, step=0.5, initial_value=0.0),
        "roll": server.gui.add_slider("roll deg", min=-15.0, max=15.0, step=0.1, initial_value=0.0),
        "pitch": server.gui.add_slider("pitch deg", min=-15.0, max=15.0, step=0.1, initial_value=0.0),
        "yaw": server.gui.add_slider("yaw deg", min=-15.0, max=15.0, step=0.1, initial_value=0.0),
    }
    readout = server.gui.add_markdown("RMS")
    save_btn = server.gui.add_button("Save manual yaml")
    status = server.gui.add_markdown("")
    lock = threading.Lock()

    def current():
        trans_mm = np.array([sliders["x"].value, sliders["y"].value, sliders["z"].value], dtype=np.float64)
        deg = np.array(
            [sliders["roll"].value, sliders["pitch"].value, sliders["yaw"].value],
            dtype=np.float64,
        )
        rot = slider_rotation(deg[0], deg[1], deg[2])
        trans_m = trans_mm / 1000.0
        queried = query_points(points, delta, rot, trans_m, center)
        _foot, _normal, dist = surface.project(queried)
        rms_mm = float(np.sqrt(np.mean(dist ** 2)) * 1e3)
        return trans_mm, deg, rot, trans_m, dist, rms_mm

    def refresh(_event=None) -> None:
        with lock:
            trans_mm, deg, rot, trans_m, dist, rms_mm = current()
            mesh_handle.wxyz = tuple(float(v) for v in Rotation.from_matrix(rot).as_quat()[[3, 0, 1, 2]])
            mesh_handle.position = tuple(float(v) for v in center + trans_m - rot @ center)
            lines = [f"**RMS 12 points: {rms_mm:.2f} mm**", ""]
            for label, is_early, d, text in zip(labels, early, dist, texts):
                session = "14:54" if is_early else "free"
                mm = d * 1e3
                text.text = f"{label}  {mm:.1f} mm"
                lines.append(f"{label} ({session})  {mm:.1f} mm")
            readout.content = "\n\n".join(lines)

    for slider in sliders.values():
        slider.on_update(refresh)

    @save_btn.on_click
    def _save(_event) -> None:
        with lock:
            if MANUAL.exists():
                status.content = f"Refusing to overwrite `{MANUAL.name}`."
                print(f"refusing to overwrite {MANUAL}", flush=True)
                return
            trans_mm, deg, rot, trans_m, _dist, rms_mm = current()
            right, scanner = manual_right_base(delta, rot, trans_m, center)
            try:
                write_manual(right, scanner, trans_mm, deg, rms_mm)
            except FileExistsError:
                status.content = f"Refusing to overwrite `{MANUAL.name}`."
                return
        status.content = f"Saved `{MANUAL.name}`  RMS {rms_mm:.2f} mm."
        print(f"wrote {MANUAL}  rms {rms_mm:.2f} mm", flush=True)

    refresh()
    print(f"viser http://127.0.0.1:{int(server.get_port())}", flush=True)
    print("Blue = 14:54 non-flange. Orange = free touches. Sliders move the mesh.", flush=True)
    try:
        while True:
            import time
            time.sleep(1.0)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
