#!/usr/bin/env python3
"""Read-only viser view of a mold calibration check.

Shows the fixb mesh in the fixture frame, the live TCP (RTDE receive only),
its foot and normal from MoldAwareRuntime, and CSV touches. Does not load a
policy and does not send motion.

Run with the geodude venv, which has viser. Point PYTHONPATH at this worktree:

  source /home/boeing/ros2_ws/src/geodude-teleoperation/.venv/bin/activate
  cd /home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames
  export PYTHONPATH="$PWD:$PWD/UMIFT_Data/surface_frames_scripts:$PYTHONPATH"
  python UMIFT_Data/surface_frames_scripts/mold_calib_viser.py --mold train --csv mold_calib_train_YYYYMMDD_HHMMSS.csv

Open http://127.0.0.1:8081
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import viser

_SURFACE_SCRIPTS = Path(__file__).resolve().parent
if str(_SURFACE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SURFACE_SCRIPTS))

from PyriteConfig.tasks.umift.workcell_frames import load_workcell_extrinsics
from mold_aware_runtime import DEFAULT_META, MoldAwareRuntime
from mold_calib_check import (
    expected_xyz,
    fit_rigid_offset,
    load_points,
    measure_contact,
    resolve_field,
    rotvec_to_pose7,
)

DEFAULT_PORT = 8081
PASS_COLOR = (40, 170, 70)
FAIL_COLOR = (210, 40, 40)


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


def read_touch_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class _TcpStream:
    """One RTDE receive process. Prints a pose line; never opens a control interface."""

    def __init__(self, python: Path, robot_ip: str):
        code = (
            "from rtde_receive import RTDEReceiveInterface\n"
            "import time\n"
            f"r = RTDEReceiveInterface(hostname={robot_ip!r})\n"
            "while True:\n"
            "    p = r.getActualTCPPose()\n"
            "    print(','.join(str(x) for x in p), flush=True)\n"
            "    time.sleep(0.05)\n"
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mold", choices=("train", "heldout"), default="train")
    parser.add_argument("--field", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--robot-ip", default="192.168.1.2")
    parser.add_argument(
        "--rtde-python",
        type=Path,
        default=Path("/home/boeing/keyence_scan_ws/.venv/bin/python"),
    )
    args = parser.parse_args(argv)

    field_path = resolve_field(args.mold, args.field)
    if not field_path.is_file():
        print(f"Mold field not found: {field_path}", file=sys.stderr)
        return 1
    extrinsics = load_workcell_extrinsics()
    catalog = load_points()
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

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.add_frame("/fixture", axes_length=0.08, axes_radius=0.003)
    server.scene.add_mesh_trimesh("/mold", mesh_from_field(field_path))
    tcp_frame = server.scene.add_frame("/tcp", axes_length=0.04, axes_radius=0.002)
    foot_sphere = server.scene.add_icosphere(
        "/foot", radius=0.004, color=(30, 120, 220), position=(0.0, 0.0, 0.0)
    )
    normal_line = server.scene.add_line_segments(
        "/normal",
        points=np.zeros((1, 2, 3)),
        colors=(30, 120, 220),
        thickness=0.002,
    )
    for point in catalog["points"]:
        xyz = expected_xyz(point, args.mold)
        server.scene.add_icosphere(
            f"/expected/{point['label']}",
            radius=0.006,
            color=(160, 160, 160),
            position=tuple(xyz.tolist()),
            opacity=0.45,
        )

    show_shift = server.gui.add_checkbox("Show rigid translation arrow", initial_value=False)
    status = server.gui.add_markdown("Waiting for TCP and CSV.")
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

    def refresh_touches() -> None:
        for handle in touch_handles:
            handle.remove()
        touch_handles.clear()
        rows = read_touch_csv(args.csv) if args.csv is not None else []
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

    state = {"tcp": "no pose yet", "touch": "click a marker for its signed distance"}

    def live_loop() -> None:
        from mold_calib_check import pose7_base_to_fixture

        stream = None
        try:
            stream = _TcpStream(args.rtde_python, args.robot_ip)
            while True:
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
                tcp_frame.position = tuple(pose_fix[:3].tolist())
                tcp_frame.wxyz = tuple(pose_fix[3:].tolist())
                foot = sample["foot_m"]
                foot_sphere.position = tuple(np.asarray(foot).tolist())
                n = np.asarray(sample["normal"], dtype=np.float64)
                n = n / max(float(np.linalg.norm(n)), 1e-12)
                normal_line.points = np.stack([foot, foot + 0.03 * n])[None, :, :]
                state["tcp"] = (
                    f"sd={sample['sd_m'] * 1e3:+.1f} mm  "
                    f"normal_error={sample['normal_error_deg']:.1f} deg"
                )
        except Exception as exc:
            state["tcp"] = f"RTDE receive stopped: {exc}"
        finally:
            if stream is not None:
                stream.close()

    threading.Thread(target=live_loop, name="mold-calib-tcp", daemon=True).start()
    refresh_touches()
    port = int(server.get_port())
    print(f"mesh: {field_path}")
    print(f"extrinsics: {extrinsics.path}")
    print(f"viser http://127.0.0.1:{port}")
    print("Read-only. This process does not move the arm.")
    csv_mtime = 0.0
    try:
        while True:
            if args.csv is not None and args.csv.is_file():
                mtime = args.csv.stat().st_mtime
                if mtime != csv_mtime:
                    csv_mtime = mtime
                    refresh_touches()
            status.content = f"**Live TCP** {state['tcp']}\n\n{state['touch']}"
            time.sleep(0.2)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
