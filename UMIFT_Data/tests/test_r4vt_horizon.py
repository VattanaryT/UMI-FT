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
from diffusion_policy.common.sampler_umift import (  # noqa: E402
    sparse_action_indices_from_rgb_timestamps,
)

ZARR = scorer.DATA / "r4vt_field_v2_k3000" / "acp_replay_buffer.zarr"
STEPS = (1, 6, 16, 64)
H = 64


def _tangent_demo(tool_p, frames, index):
    """TCP path with each control step's normal component removed."""
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
    for demo, index, action in _windows():
        xyz, got_R = scorer.horizon_pose("causal", action, demo, index)
        frames = demo["frame"][index]
        ref = _tangent_demo(demo["tool_p"], frames, index)
        pos_rows.append(np.linalg.norm(xyz - ref, axis=1) * 1e3)
        rot_rows.append(geodesic_deg(got_R, demo["tool_R"][index]))
    pos = np.stack(pos_rows)
    rot = np.stack(rot_rows)
    for step in STEPS:
        pos_med = float(np.median(pos[:, step - 1]))
        rot_med = float(np.median(rot[:, step - 1]))
        assert pos_med < 1.0, f"step {step} pos median {pos_med:.2f} mm"
        assert rot_med < 1.0, f"step {step} rot median {rot_med:.2f} deg"
