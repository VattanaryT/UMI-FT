"""Play rollouts in the calibrated geodude scene.

Run with the geodude venv so the server is the one you tunnel to:
  MUJOCO_GL=egl python closedloop_sim/scene_view.py
Listens on 0.0.0.0:8811.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import viser
from geodude_teleoperation.ee_viz import (
    CHAMFER_MESH,
    CHAMFER_MESH_SCALE,
    chamfer_tcp_from_flange,
    flange_from_wrist3,
    seat_mesh_on_xy_origin,
)

from robot_viz import (  # noqa: E402
    OUT,
    arm_and_table,
    fixture_in_world,
    seat,
    _model,
)

FIXB_FIELD = (
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
THIN_FIELD = (
    "/home/boeing/umi_ft_ws/data/held_out_mold/thin_stringer/transform_check/"
    "wall_fix/fixb_field_wall_vertical.npz"
)

PORT = 8811


def _R(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(9))
    return q


def _sub(path: Path, T):
    blob = np.load(path, allow_pickle=True)
    pos = np.asarray(blob["position"], dtype=np.float64)
    idx = np.linspace(0, len(pos) - 1, min(200, len(pos))).astype(int)
    world = (T[:3, :3] @ pos[idx].T).T + T[:3, 3]
    force = np.linalg.norm(blob["force"][idx], axis=1) if "force" in blob.files else np.zeros(len(idx))
    if "pose7" in blob.files:
        tool_z = np.stack([T[:3, :3] @ _R(blob["pose7"][i, 3:])[:, 2] for i in idx])
    else:
        tool_z = np.tile(T[:3, 2], (len(idx), 1))
    return world, np.asarray(force, dtype=np.float64), tool_z


def _field_mesh(path, T, color):
    blob = np.load(path)
    mesh = trimesh.Trimesh(vertices=blob["vertices"], faces=blob["faces"], process=False)
    mesh.apply_transform(T)
    mesh.visual.face_colors = color
    return mesh


def _chamfer_mesh():
    raw = np.asarray(trimesh.load(str(CHAMFER_MESH), force="mesh").vertices)
    seated = seat_mesh_on_xy_origin(raw * CHAMFER_MESH_SCALE)
    tcp = chamfer_tcp_from_flange()
    # Vertices in the TCP frame, tip at the origin. Pose is wrist → flange → taught TCP.
    verts = (tcp[:3, :3].T @ (seated - tcp[:3, 3]).T).T
    mesh = trimesh.Trimesh(vertices=verts, faces=trimesh.load(str(CHAMFER_MESH), force="mesh").faces, process=False)
    mesh.visual.face_colors = [170, 170, 170, 255]
    return mesh


def _wrist_T(model, data):
    bid = model.body("right_ur5e/wrist_3_link").id
    T = np.eye(4)
    T[:3, :3] = data.xmat[bid].reshape(3, 3)
    T[:3, 3] = data.xpos[bid]
    return T


def _colors(force):
    if force.max() <= 1e-9:
        return np.full((len(force), 3), 180, dtype=np.uint8)
    t = np.clip(force / np.percentile(force, 95), 0, 1)
    c = np.zeros((len(force), 3), dtype=np.uint8)
    c[:, 0] = (40 + 215 * t).astype(np.uint8)
    c[:, 2] = (180 * (1 - t)).astype(np.uint8)
    return c


def main():
    T = fixture_in_world()
    model = _model(T)
    data = mujoco.MjData(model)
    arm, table = arm_and_table(model)
    names = sorted(p.stem for p in OUT.glob("policy_*_fixb_ep40.npz"))
    names = [n for n in names if (OUT / f"robot_q_{n}.npz").is_file()]
    if not names:
        raise SystemExit("no robot_q_*.npz; run robot_viz.py first")

    server = viser.ViserServer(host="0.0.0.0", port=PORT)
    mold_fixb = server.scene.add_mesh_trimesh("/mold/fixb", _field_mesh(FIXB_FIELD, T, [90, 110, 130, 180]))
    mold_thin = server.scene.add_mesh_trimesh("/mold/thin", _field_mesh(THIN_FIELD, T, [150, 120, 70, 180]))
    tool_from_wrist = flange_from_wrist3() @ chamfer_tcp_from_flange()
    chamfer = server.scene.add_mesh_trimesh("/chamfer", _chamfer_mesh())
    seat(model, data)
    gid = int(table)
    box = trimesh.creation.box(extents=2 * model.geom_size[gid])
    box.visual.face_colors = [180, 180, 180, 80]
    server.scene.add_mesh_trimesh("/table", box, position=data.geom_xpos[gid], wxyz=_quat(data.geom_xmat[gid]))

    handles = []
    for i in arm:
        size = model.geom_size[i]
        if int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            mesh = trimesh.creation.capsule(radius=float(size[0]), height=float(2 * size[1]))
        else:
            mesh = trimesh.creation.cylinder(radius=float(size[0]), height=float(2 * size[1]))
        mesh.visual.face_colors = [70, 110, 160, 255]
        handles.append((int(i), server.scene.add_mesh_trimesh(f"/arm/{i}", mesh)))

    which = server.gui.add_dropdown("rollout", options=names, initial_value=names[0])
    play = server.gui.add_checkbox("play", initial_value=False)
    scrub = server.gui.add_slider("scrub", min=0.0, max=1.0, step=0.001, initial_value=0.0)
    show_demo = server.gui.add_checkbox("demo", initial_value=True)
    show_policy = server.gui.add_checkbox("policy", initial_value=True)
    force_on = server.gui.add_checkbox("force-colored path", initial_value=True)
    normal_on = server.gui.add_checkbox("tool normal", initial_value=True)
    paths = {}
    demo_world, _, demo_z = _sub(OUT / "demo_ep40.npz", T)

    def _show_mold():
        thin = "thin" in which.value
        mold_fixb.visible = not thin
        mold_thin.visible = thin

    def refresh_paths():
        _show_mold()
        world = paths[which.value]["world"]
        force = paths[which.value]["force"]
        color = _colors(force) if force_on.value else np.full((len(world), 3), 220, dtype=np.uint8)
        if show_policy.value:
            server.scene.add_point_cloud("/path/policy", world, colors=color, point_size=0.008)
        else:
            server.scene.add_point_cloud("/path/policy", world[:1], colors=np.zeros((1, 3), np.uint8), point_size=0.001)
        if show_demo.value:
            server.scene.add_point_cloud("/path/demo", demo_world[::2], colors=np.full((len(demo_world[::2]), 3), (40, 160, 80), np.uint8), point_size=0.006)
        else:
            server.scene.add_point_cloud("/path/demo", demo_world[:1], colors=np.zeros((1, 3), np.uint8), point_size=0.001)

    for name in names:
        q = np.load(OUT / f"robot_q_{name}.npz")
        world, force, tool_z = _sub(OUT / f"{name}.npz", T)
        paths[name] = {"q": q["qpos"], "world": world, "force": force, "tool_z": tool_z}
        err = []
        for i in range(len(paths[name]["q"])):
            data.qpos[:] = paths[name]["q"][i]
            mujoco.mj_forward(model, data)
            tip = (_wrist_T(model, data) @ tool_from_wrist)[:3, 3]
            err.append(np.linalg.norm(tip - world[i]))
        err = np.asarray(err)
        print(f"tip-to-path {name} max {err.max()*1e3:.1f} mm median {np.median(err)*1e3:.1f} mm", flush=True)
    refresh_paths()
    for handle in (which, show_demo, show_policy, force_on):
        handle.on_update(lambda _: refresh_paths())

    normal = server.scene.add_line_segments(
        "/tool_normal", np.zeros((1, 2, 3)), colors=(255, 220, 40), thickness=0.004
    )

    def draw(i):
        pack = paths[which.value]
        i = int(np.clip(i, 0, len(pack["q"]) - 1))
        data.qpos[:] = pack["q"][i]
        mujoco.mj_forward(model, data)
        for gid, handle in handles:
            handle.position = data.geom_xpos[gid]
            handle.wxyz = _quat(data.geom_xmat[gid])
        tool = _wrist_T(model, data) @ tool_from_wrist
        chamfer.position = tool[:3, 3]
        chamfer.wxyz = _quat(tool[:3, :3])
        tip = pack["world"][i]
        if normal_on.value:
            seg = np.stack([tip, tip + 0.08 * pack["tool_z"][i]])
            normal.points = seg.reshape(1, 2, 3)
        else:
            normal.points = np.zeros((1, 2, 3))

    def loop():
        while True:
            n = len(paths[which.value]["q"])
            if play.value:
                scrub.value = float((scrub.value + 1.0 / n) % 1.0)
            draw(int(scrub.value * (n - 1)))
            time.sleep(0.05)

    threading.Thread(target=loop, daemon=True).start()
    print(f"viewer http://0.0.0.0:{PORT}")
    threading.Event().wait()


if __name__ == "__main__":
    main()
