"""IK the right UR5e in the geodude model onto existing plant TCP traces.

Run with the teleop venv and MUJOCO_GL=egl. Writes mp4 + html under out/.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from geodude_teleoperation.ee_viz import CHAMFER_TCP_FROM_FLANGE_XYZ

GEODUDE = Path("/home/boeing/ros2_ws/src/geodude-teleoperation/assets/robots/geodude/geodude.xml")
VENTION = Path(
    "/home/boeing/ros2_ws/src/geodude-teleoperation/src/geodude_teleoperation/data/frames/T_right_base_vention.json"
)
OUT = Path(__file__).resolve().parent / "out"
MOLD = OUT / "m1_ep40.obj"
# UR/RTDE base <- fixture. layup_workcell_extrinsics_20261116_orange.yaml
T_RTDE_FROM_FIXTURE = np.array(
    [
        [0.99892556, 0.02654420, -0.03799192, 0.12329489],
        [-0.02965165, 0.99604925, -0.08370661, 0.87052723],
        [0.03562013, 0.08474299, 0.99576572, 0.41172177],
        [0, 0, 0, 1],
    ]
)
# URDF vention_base.urdf.xacro mount origins. The mjcf copies are 0.53 m low.
RIGHT_MOUNT = "0.34 -0.858 1.65"
LEFT_MOUNT = "-0.33 -0.858 1.65"
JOINTS = [
    "right_ur5e/shoulder_pan_joint",
    "right_ur5e/shoulder_lift_joint",
    "right_ur5e/elbow_joint",
    "right_ur5e/wrist_1_joint",
    "right_ur5e/wrist_2_joint",
    "right_ur5e/wrist_3_joint",
]
LINKS = [
    "right_ur5e/base",
    "right_ur5e/shoulder_link",
    "right_ur5e/upper_arm_link",
    "right_ur5e/forearm_link",
    "right_ur5e/wrist_1_link",
    "right_ur5e/wrist_2_link",
    "right_ur5e/wrist_3_link",
]
# Taught chamfer TCP in the flange frame. ft_sensor_site is that flange.
# Same numbers as geodude_teleoperation.ee_viz.CHAMFER_TCP_FROM_FLANGE_*.
TCP_OFFSET_M = np.asarray(CHAMFER_TCP_FROM_FLANGE_XYZ, dtype=np.float64)
FAIL_M = 0.002


def _skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(9))
    return q


def rtde_in_vention() -> np.ndarray:
    return np.asarray(json.loads(VENTION.read_text())["matrix"], dtype=np.float64)


def fixture_in_world() -> np.ndarray:
    """Vention/world from fixture. No extra base rotation."""
    return rtde_in_vention() @ T_RTDE_FROM_FIXTURE


def _model(mold_pose):
    xml = GEODUDE.read_text().replace(
        'meshdir="assets"',
        f'meshdir="{GEODUDE.parent / "assets"}"',
    )
    xml = xml.replace(
        '<body name="right_arm_vention_base" pos="0.33 -0.848 1.12"',
        f'<body name="right_arm_vention_base" pos="{RIGHT_MOUNT}"',
    )
    xml = xml.replace(
        '<body name="left_arm_vention_base" pos="-0.33 -0.848 1.12"',
        f'<body name="left_arm_vention_base" pos="{LEFT_MOUNT}"',
    )
    xml = xml.replace(
        "</asset>",
        f'<mesh name="layup_mold" file="{MOLD}"/>\n</asset>',
        1,
    )
    pos = " ".join(f"{v:.6f}" for v in mold_pose[:3, 3])
    quat = " ".join(f"{v:.6f}" for v in _quat(mold_pose[:3, :3]))
    body = (
        f'<body name="layup_mold" pos="{pos}" quat="{quat}">'
        '<geom type="mesh" mesh="layup_mold" rgba="0.4 0.45 0.5 0.8" contype="0" conaffinity="0"/>'
        "</body>\n</worldbody>"
    )
    xml = xml.replace("</worldbody>", body, 1)
    return mujoco.MjModel.from_xml_string(xml)


def seat(model, data):
    """Ready arm pose, vention rails at the calibrated zero (not the keyframe's 0.25 m)."""
    mujoco.mj_resetDataKeyframe(model, data, model.key("ready").id)
    for name in ("right_arm_linear_vention", "left_arm_linear_vention"):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = 0.0
    mujoco.mj_forward(model, data)


def rtde_pose(model, data) -> np.ndarray:
    """RTDE base in world. The mjcf base body is that frame times Rz(180); that 180 is the URDF, not an extra twist."""
    bid = model.body("right_ur5e/base").id
    T = np.eye(4)
    T[:3, :3] = data.xmat[bid].reshape(3, 3) @ np.diag([-1.0, -1.0, 1.0])
    T[:3, 3] = data.xpos[bid]
    return T


def arm_and_table(model):
    table = model.geom("geom_1").id
    arm = []
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if model.geom_contype[i] and name.startswith("right_ur5e/") and "gripper" not in name:
            arm.append(i)
    return np.asarray(arm, dtype=np.int32), table


def table_clearance(model, data, arm, table) -> float:
    fromto = np.zeros(6)
    gap = 2.0
    for gid in arm:
        gap = min(gap, float(mujoco.mj_geomDistance(model, data, int(gid), int(table), 2.0, fromto)))
    return gap


def _ik(model, data, adr, lo, hi, target):
    site = model.site("right_ur5e/ft_sensor_site").id
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    err = 1.0
    clipped = False
    for _ in range(60):
        mujoco.mj_forward(model, data)
        R = data.site_xmat[site].reshape(3, 3)
        off = R @ TCP_OFFSET_M
        tip = data.site_xpos[site] + off
        err_v = target - tip
        err = float(np.linalg.norm(err_v))
        if err < FAIL_M:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, site)
        J = (jacp - _skew(off) @ jacr)[:, adr]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), err_v)
        q = data.qpos[adr] + dq
        clipped = clipped or bool(np.any(q < lo) or np.any(q > hi))
        data.qpos[adr] = np.clip(q, lo, hi)
    mujoco.mj_forward(model, data)
    return err, clipped


def _stick(model, data):
    return np.stack([data.xpos[model.body(name).id] for name in LINKS])


def _html(stem, path, sticks, world_tcp):
    step = max(1, len(sticks) // 25)
    traces = []
    for stick in sticks[::step]:
        traces.append(
            "{type:'scatter3d',mode:'lines',line:{width:6,color:'#335'},"
            f"x:{stick[:,0].round(4).tolist()},y:{stick[:,1].round(4).tolist()},z:{stick[:,2].round(4).tolist()}"
            "}"
        )
    traces.append(
        "{type:'scatter3d',mode:'lines',line:{width:3,color:'#c40'},name:'tcp',"
        f"x:{world_tcp[::step,0].round(4).tolist()},y:{world_tcp[::step,1].round(4).tolist()},z:{world_tcp[::step,2].round(4).tolist()}"
        "}"
    )
    page = (
        "<!doctype html><meta charset=utf-8><script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>"
        f"<h3>{stem}</h3><video src='{stem}.mp4' controls loop muted autoplay></video>"
        f"<div id='v'></div><script>Plotly.newPlot('v',[{','.join(traces)}],"
        "{scene:{aspectmode:'data'},margin:{l:0,r:0,t:0,b:0}});</script>"
    )
    path.write_text(page)


def render_one(model, data, adr, lo, hi, arm, table, T_world_fix, npz_path):
    blob = np.load(npz_path)
    pos = np.asarray(blob["position"], dtype=np.float64)
    idx = np.linspace(0, len(pos) - 1, min(200, len(pos))).astype(int)
    pos = pos[idx]
    world = (T_world_fix[:3, :3] @ pos.T).T + T_world_fix[:3, 3]
    seat(model, data)
    fails = 0
    limits = 0
    gap = 2.0
    errs = []
    qs = []
    sticks = []
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 1.4, 150, -25
    renderer = mujoco.Renderer(model, 480, 640)
    mp4 = OUT / f"robot_{npz_path.stem}.mp4"
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480",
            "-r", "20", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for target in world:
        err, clipped = _ik(model, data, adr, lo, hi, target)
        gap = min(gap, table_clearance(model, data, arm, table))
        qs.append(data.qpos.copy())
        errs.append(err)
        fails += int(err > FAIL_M)
        limits += int(clipped)
        sticks.append(_stick(model, data))
        cam.lookat[:] = target
        renderer.update_scene(data, camera=cam)
        proc.stdin.write(renderer.render().tobytes())
    proc.stdin.close()
    proc.wait()
    renderer.close()
    np.savez_compressed(OUT / f"robot_q_{npz_path.stem}.npz", qpos=np.stack(qs), world=world)
    _html(mp4.stem, OUT / f"{mp4.stem}.html", sticks, world)
    return {
        "file": str(mp4),
        "html": f"http://127.0.0.1:8810/{mp4.stem}.html",
        "frames": len(world),
        "ik_failures": fails,
        "joint_limit_hits": limits,
        "median_err_mm": round(float(np.median(errs) * 1e3), 2),
        "max_err_mm": round(float(np.max(errs) * 1e3), 2),
        "min_arm_table_mm": round(gap * 1e3, 1),
        "table_collision": bool(gap <= 0.0),
    }


def main():
    T = fixture_in_world()
    model = _model(T)
    data = mujoco.MjData(model)
    seat(model, data)
    got = rtde_pose(model, data)
    err_t = float(np.linalg.norm(got[:3, 3] - rtde_in_vention()[:3, 3]))
    err_R = float(np.linalg.norm(got[:3, :3] - rtde_in_vention()[:3, :3]))
    print(f"rtde origin err m {err_t:.4f} R {err_R:.4f}")
    if err_t > 0.002 or err_R > 0.01:
        raise SystemExit("right base is not on T_right_base_vention")
    adr = np.array([model.jnt_qposadr[model.joint(n).id] for n in JOINTS])
    rng = np.stack([model.jnt_range[model.joint(n).id] for n in JOINTS])
    arm, table = arm_and_table(model)
    reports = []
    paths = sorted(p for p in OUT.glob("*.npz") if p.name != "demo_ep40.npz" and not p.name.startswith("robot_"))
    for path in paths:
        print("render", path.name)
        reports.append({"src": path.name, **render_one(model, data, adr, rng[:, 0], rng[:, 1], arm, table, T, path)})
        print(reports[-1])
    summary = {
        "frame": "T_right_base_vention @ T_right_base_from_fixture",
        "tcp": "chamfer flange xyz [-0.0049, -0.00029, 0.14422] rotvec [1.20695, 1.20049, 1.20171]",
        "tcp_offset_m": TCP_OFFSET_M.tolist(),
        "runs": reports,
    }
    (OUT / "robot_ik.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
