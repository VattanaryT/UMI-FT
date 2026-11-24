"""GT r4vt labels through horizon mode. Median at steps 1/6/16/64."""

import sys
from pathlib import Path

import numpy as np
import zarr

SCRIPTS = Path(__file__).resolve().parents[1] / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "eval"))

import score_sign_flip as scorer  # noqa: E402
from fixb_labels import pose7_to_Rp  # noqa: E402
from heading_tilt import geodesic_deg  # noqa: E402
from r4vt_horizon import arrival_tcp  # noqa: E402
from diffusion_policy.common.sampler_umift import (  # noqa: E402
    sparse_action_indices_from_rgb_timestamps,
)
from replay_gt import R1REL_ZARR, _action_rows, _open_episode  # noqa: E402
import PyriteUtility.spatial_math.spatial_utilities as su  # noqa: E402
from PyriteConfig.tasks.umift.umift_type_conversions import (  # noqa: E402
    action21_postprocess,
    action21_to_action_sample,
)

ZARR = scorer.DATA / "r4vt_field_v2_k3000" / "acp_replay_buffer.zarr"
STEPS = (1, 6, 16, 64)
H = 64


def _tangent_demo(demo, frames, index):
    """Tangent path after each action. The normal part of the TCP step is removed."""
    tool_p = np.asarray(demo["tool_p"], dtype=np.float64)
    arrived = arrival_tcp(tool_p, demo["robot_t"], demo["rgb_t"], index)
    p = np.asarray(tool_p[int(index[0])], dtype=np.float64).copy()
    out = np.empty((len(index), 3))
    for k in range(len(index)):
        delta = arrived[k] - tool_p[int(index[k])]
        normal = frames[k, :, 2]
        p = p + delta - normal * np.dot(delta, normal)
        out[k] = p
    return out


def _windows():
    root = zarr.open(str(ZARR), "r")["data"]
    found = []
    for episode in (1, 4, 9, 18, 27, 33, 41, 48):
        group = root[f"episode_{episode}"]
        robot_t = np.asarray(group["robot_time_stamps_0"]).reshape(-1)
        rgb_t = np.asarray(group["rgb_time_stamps_0"]).reshape(-1)
        tool_R, tool_p = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
        demo = {
            "robot_t": robot_t,
            "rgb_t": rgb_t,
            "tool_p": tool_p,
            "tool_R": tool_R,
            "foot": np.asarray(group["mold_x_s_0"], dtype=np.float64).reshape(-1, 3),
            "frame": np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3),
        }
        v = np.asarray(group["fixb_v_par_0"], dtype=np.float64)
        dn = np.asarray(group["fixb_delta_n_0"], dtype=np.float64).reshape(-1)
        stiff = np.asarray(group["stiffness_0"], dtype=np.float64).reshape(-1)
        dR = np.asarray(group["fixb_delta_R_rot6d_0"], dtype=np.float64)
        rgb_ids = np.linspace(8, len(rgb_t) - H - 2, 4).astype(int)
        for rgb in rgb_ids:
            index, _times = sparse_action_indices_from_rgb_timestamps(robot_t, rgb_t, int(rgb), H, 1)
            index = np.asarray(index, dtype=np.int64)
            if len(index) < H:
                continue
            index = index[:H]
            action = np.column_stack([
                v[index],
                dn[index],
                stiff[index],
                dR[index],
            ])
            found.append((demo, index, action))
    if len(found) < 16:
        raise RuntimeError(f"only {len(found)} train windows")
    return found


def test_gt_horizon_roundtrip():
    pos_rows, rot_rows = [], []
    step1_tang, step1_from_start = [], []
    for demo, index, action in _windows():
        xyz, got_R = scorer.horizon_pose("causal", action, demo, index)
        frames = demo["frame"][index]
        ref = _tangent_demo(demo, frames, index)
        pos_rows.append(np.linalg.norm(xyz - ref, axis=1) * 1e3)
        rot_rows.append(geodesic_deg(got_R, demo["tool_R"][index]))
        arrived = arrival_tcp(demo["tool_p"], demo["robot_t"], demo["rgb_t"], index)
        err = xyz[0] - arrived[0]
        normal = frames[0, :, 2]
        along = float(np.dot(err, normal))
        step1_tang.append(np.linalg.norm(err - along * normal) * 1e3)
        step1_from_start.append(np.linalg.norm(xyz[0] - demo["tool_p"][int(index[0])]) * 1e3)
    pos = np.stack(pos_rows)
    rot = np.stack(rot_rows)
    for step in STEPS:
        pos_med = float(np.median(pos[:, step - 1]))
        rot_med = float(np.median(rot[:, step - 1]))
        assert pos_med < 1.0, f"step {step} pos median {pos_med:.2f} mm"
        assert rot_med < 1.0, f"step {step} rot median {rot_med:.2f} deg"
    # Step 1 is the pose after one action, so it has left the query TCP.
    # The miss vs the demo TCP at t+1 is the normal component.
    assert float(np.median(step1_from_start)) > 1.0
    assert float(np.median(step1_tang)) < 1.0, f"step 1 tangent {np.median(step1_tang):.2f} mm"


def test_r1rel_gt_next_action_is_tcp_at_t_plus_1():
    """r1rel action 0 is the query TCP. Action 1 is the demo TCP at t+1."""
    at_t, at_next = [], []
    for episode, rgb in ((0, 40), (5, 80), (9, 120), (18, 60)):
        group = _open_episode(R1REL_ZARR, episode)
        robot_t = np.asarray(group["robot_time_stamps_0"]).reshape(-1)
        rgb_t = np.asarray(group["rgb_time_stamps_0"]).reshape(-1)
        fb = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
        index, _times = sparse_action_indices_from_rgb_timestamps(robot_t, rgb_t, rgb, 4, 1)
        index = np.asarray(index[:4], dtype=np.int64)
        absolute = _action_rows(group, index, with_delta_r=False)
        base = su.pose7_to_SE3(fb[int(index[0])].reshape(1, 7))[0]
        relative = action21_to_action_sample(absolute, np.zeros((0,)), [0], [base])["sparse"]
        nominal, _vt, _stiff, _eoat = action21_postprocess(relative, [base], [0])
        xyz = su.SE3_to_pose7(nominal[0])[:, :3]
        at_t.append(np.linalg.norm(xyz[0] - fb[int(index[0]), :3]) * 1e3)
        at_next.append(np.linalg.norm(xyz[1] - fb[int(index[1]), :3]) * 1e3)
    assert float(np.median(at_t)) < 1.0
    assert float(np.median(at_next)) < 1.0
