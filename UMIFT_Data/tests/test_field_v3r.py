"""Field v3r: position matches field_v2. Rotation is R_tcp(t).T @ R_tcp(t+k)."""

import sys
from pathlib import Path

import numpy as np
import zarr

SCRIPTS = Path(__file__).resolve().parents[1] / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))

from diffusion_policy.common.sampler_umift import (  # noqa: E402
    sparse_action_indices_from_rgb_timestamps,
)
from field_v3r_dataset import action_v3r_to_action_sample  # noqa: E402
from fixb_labels import pose7_to_Rp, so3_to_rot6d_rows  # noqa: E402
from fixb_tcp_track import project_tcp_velocity  # noqa: E402
from heading_tilt import geodesic_deg  # noqa: E402
from r4vt_horizon import control_dts, decode_horizon  # noqa: E402

DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
V2 = DATA / "r4vt_field_v2_k3000/acp_replay_buffer.zarr"
V3 = DATA / "r4vt_field_v3r/acp_replay_buffer.zarr"
THIN = Path(
    "/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/"
    "UMIFT_Data/surface_frames_scripts/eval/thin_heldout"
)
STEPS = (1, 6, 16, 64)
H = 64
SAME = ("fixb_v_par_0", "fixb_delta_n_0", "stiffness_0", "ts_pose_command_0")


def _tangent(tool_p, frames, index):
    p = np.asarray(tool_p[int(index[0])], dtype=np.float64).copy()
    out = np.empty((len(index), 3))
    for k, i in enumerate(index):
        out[k] = p
        if k + 1 == len(index):
            break
        delta = tool_p[int(index[k + 1])] - tool_p[int(i)]
        normal = frames[k, :, 2]
        p = p + delta - normal * np.dot(delta, normal)
    return out


def _action(demo, index):
    absolute = so3_to_rot6d_rows(demo["tool_R"][index]).astype(np.float32)
    action = np.column_stack([
        demo["v"][index],
        demo["dn"][index],
        demo["stiff"][index],
        absolute,
    ]).astype(np.float32)
    base = np.eye(4)
    base[:3, :3] = demo["tool_R"][int(index[0])]
    return action_v3r_to_action_sample(action, np.zeros((0,)), [0], [base])["sparse"]


def _errors(demo, index):
    action = _action(demo, index)
    if not np.array_equal(action[:, :4], np.column_stack([
        demo["v"][index], demo["dn"][index], demo["stiff"][index],
    ]).astype(np.float32)):
        raise AssertionError("position columns changed")
    frames = demo["frame"][index]
    dts = control_dts(demo["robot_t"], demo["rgb_t"], index)
    xyz, got_R = decode_horizon(
        action, demo["tool_p"][int(index[0])], frames, dts, demo["tool_R"][int(index[0])]
    )
    ref = _tangent(demo["tool_p"], frames, index)
    pos = np.linalg.norm(xyz - ref, axis=1) * 1e3
    rot = geodesic_deg(got_R, demo["tool_R"][index])
    return pos, rot


def _assert_steps(pos_rows, rot_rows, where):
    pos = np.stack(pos_rows)
    rot = np.stack(rot_rows)
    for step in STEPS:
        pos_med = float(np.median(pos[:, step - 1]))
        rot_med = float(np.median(rot[:, step - 1]))
        assert pos_med < 1.0, f"{where} step {step} pos median {pos_med:.2f} mm"
        assert rot_med < 1.0, f"{where} step {step} rot median {rot_med:.2f} deg"


def test_position_matches_field_v2():
    src = zarr.open(str(V2), "r")["data"]
    dst = zarr.open(str(V3), "r")["data"]
    for ep_id in range(50):
        a = src[f"episode_{ep_id}"]
        b = dst[f"episode_{ep_id}"]
        for key in SAME:
            assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), f"ep {ep_id} {key}"


def test_train_roundtrip():
    root = zarr.open(str(V3), "r")["data"]
    pos_rows, rot_rows = [], []
    for episode in (1, 4, 9, 18, 27, 33, 41, 48):
        group = root[f"episode_{episode}"]
        robot_t = np.asarray(group["robot_time_stamps_0"]).reshape(-1)
        rgb_t = np.asarray(group["rgb_time_stamps_0"]).reshape(-1)
        tool_R, tool_p = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
        demo = {
            "robot_t": robot_t, "rgb_t": rgb_t, "tool_R": tool_R, "tool_p": tool_p,
            "frame": np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3),
            "v": np.asarray(group["fixb_v_par_0"], dtype=np.float32),
            "dn": np.asarray(group["fixb_delta_n_0"], dtype=np.float32).reshape(-1),
            "stiff": np.asarray(group["stiffness_0"], dtype=np.float32).reshape(-1),
        }
        for rgb in np.linspace(8, len(rgb_t) - H - 2, 4).astype(int):
            index, _times = sparse_action_indices_from_rgb_timestamps(robot_t, rgb_t, int(rgb), H, 1)
            index = np.asarray(index, dtype=np.int64)[:H]
            if len(index) < H:
                continue
            pos, rot = _errors(demo, index)
            pos_rows.append(pos)
            rot_rows.append(rot)
    assert len(pos_rows) >= 16
    _assert_steps(pos_rows, rot_rows, "train")


def test_thin_roundtrip():
    pos_rows, rot_rows = [], []
    for letter, folder in (("C", "C_first"), ("A", "A_full")):
        ep = zarr.open(THIN / folder / "replay.zarr", "r")["data"]["episode_0"]
        robot_t = np.asarray(ep["robot_time_stamps_0"]).reshape(-1)
        rgb_t = np.asarray(ep["rgb_time_stamps_0"]).reshape(-1)
        tool_R, tool_p = pose7_to_Rp(np.asarray(ep["ts_pose_fb_0"], dtype=np.float64))
        frame = np.asarray(ep["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        velocity, _dt = project_tcp_velocity(frame, tool_p, robot_t, rgb_t)
        demo = {
            "robot_t": robot_t, "rgb_t": rgb_t, "tool_R": tool_R, "tool_p": tool_p,
            "frame": frame, "v": velocity.astype(np.float32),
            "dn": np.zeros(len(robot_t), dtype=np.float32),
            "stiff": np.full(len(robot_t), 3000.0, dtype=np.float32),
        }
        blob = np.load(THIN / "export/causal_e25" / ("episode_0.npz" if letter == "C" else "episode_1.npz"))
        windows = np.asarray(blob["action_robot_idx"], dtype=np.int64)
        take = windows[np.linspace(0, len(windows) - 1, 8).astype(int)]
        for index in take:
            index = np.asarray(index, dtype=np.int64)[:H]
            if len(index) < H or int(index[-1]) >= len(robot_t):
                continue
            pos, rot = _errors(demo, index)
            pos_rows.append(pos)
            rot_rows.append(rot)
    assert pos_rows, "no thin windows"
    _assert_steps(pos_rows, rot_rows, "thin")
