"""score_sign_flip decodes a label back to the TCP, and a real prediction is not that label."""

import sys
from pathlib import Path

import numpy as np
import torch
import zarr
from scipy.spatial.transform import Rotation

SCRIPTS = Path(__file__).resolve().parents[1] / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "eval"))

import score_sign_flip as scorer  # noqa: E402
from fixb_labels import pose7_to_Rp, so3_to_rot6d_rows  # noqa: E402
from heading_tilt import long_axis, pack_tilt  # noqa: E402
from mold_aware_runtime import rot6d_rows_to_R  # noqa: E402
from replay_gt import _action10_rows, _action_rows  # noqa: E402
from run_policy_open_loop_val_export import load_policy_and_cfg, predict_sparse_action  # noqa: E402
import PyriteUtility.spatial_math.spatial_utilities as su  # noqa: E402
from PyriteConfig.tasks.umift.umift_type_conversions import action21_to_action_sample  # noqa: E402

CAUSAL = scorer.DATA / "r4vt_causal_foot" / "acp_replay_buffer.zarr"
R1REL = scorer.DATA / "r1rel_world" / "acp_replay_buffer.zarr"
FIELD = scorer.DATA / "shared" / "fixb_field.npz"
K3000_E25 = scorer.TRAIN / (
    "oct-v35-h4s-r4vt-causal-foot-carry-k3000_e200_bs64_valtopk_43472139/"
    "checkpoints/epoch=0025-val_loss=0.048.ckpt"
)
K3000_E199 = scorer.TRAIN / (
    "oct-v35-h4s-r4vt-causal-foot-carry-k3000_e200_bs64_valtopk_43472139/"
    "checkpoints/e199_final.ckpt"
)
TILT_E25 = scorer.TRAIN / (
    "oct-v35-h4s-r4vt-tilt1_e200_bs64_valtopk_43467749/"
    "checkpoints/epoch=0025-val_loss=0.064.ckpt"
)
# e25 filename. Normalized residual / tilt scale is 11.4°. Twice that is the cap.
# sqrt(val_loss) / scale is 45°, and twice that (90°) still passes the 86° axis bug.
TILT_VAL_LOSS = 0.064
OFFSET_MM = 10.0
OFFSET_DEG = 10.0


def _axis():
    return long_axis(np.load(FIELD)["vertices"])


def _demo(group):
    tool_R, tool_p = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
    return {
        "robot_t": np.asarray(group["robot_time_stamps_0"]).reshape(-1),
        "rgb_t": np.asarray(group["rgb_time_stamps_0"]).reshape(-1),
        "tool_p": tool_p,
        "tool_R": tool_R,
        "frame": np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3),
    }


def _index(demo, rgb):
    from diffusion_policy.common.sampler_umift import sparse_action_indices_from_rgb_timestamps

    index, _times = sparse_action_indices_from_rgb_timestamps(
        demo["robot_t"], demo["rgb_t"], int(rgb), scorer.heldout.H, 1
    )
    index = np.asarray(index, dtype=np.int64)
    if len(index) < scorer.heldout.H:
        raise RuntimeError(f"rgb {rgb} horizon {len(index)}")
    return index[: scorer.heldout.H]


def _causal():
    group = zarr.open(str(CAUSAL), "r")["data"]["episode_9"]
    demo = _demo(group)
    index = _index(demo, len(demo["rgb_t"]) // 3)
    return demo, index, _action10_rows(group, index), _axis()


def _tilt_action(demo, index, axis, base):
    direction, _sides = scorer._episode_heading(demo, axis)
    frame = demo["frame"][index]
    tilt = pack_tilt(demo["tool_R"][index], -frame[:, :, 2], direction[index])
    action = np.zeros((len(index), 8), dtype=np.float64)
    action[:, 0:4] = base[:, 0:4]
    action[:, 4:8] = tilt
    return action


def _r1rel():
    group = zarr.open(str(R1REL), "r")["data"]["episode_9"]
    demo = {
        "robot_t": np.asarray(group["robot_time_stamps_0"]).reshape(-1),
        "rgb_t": np.asarray(group["rgb_time_stamps_0"]).reshape(-1),
        "tool_R": None,
        "tool_p": None,
    }
    tool_R, tool_p = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
    demo["tool_R"] = tool_R
    demo["tool_p"] = tool_p
    index = _index(demo, len(demo["rgb_t"]) // 3)
    absolute = _action_rows(group, index, with_delta_r=False)
    base = su.pose7_to_SE3(np.asarray(group["ts_pose_fb_0"][int(index[0])], dtype=np.float64).reshape(1, 7))[0]
    relative = action21_to_action_sample(absolute, np.zeros((0,)), [0], [base])["sparse"]
    return demo, index, np.asarray(relative, dtype=np.float64), base[:3, :3]


def _offset_surface(kind, action, dt):
    out = np.array(action, dtype=np.float64, copy=True)
    out[:, 0] += (OFFSET_MM / 1000.0) / dt
    if kind == "causal":
        turned = Rotation.from_rotvec([0.0, 0.0, -np.deg2rad(OFFSET_DEG)]).as_matrix()
        out[:, 4:10] = so3_to_rot6d_rows(np.einsum("ij,njk->nik", turned, rot6d_rows_to_R(out[:, 4:10])))
    else:
        psi = np.arctan2(out[:, 4], out[:, 5]) + np.deg2rad(OFFSET_DEG)
        out[:, 4] = np.sin(psi)
        out[:, 5] = np.cos(psi)
    return out


def _offset_r1rel(action, anchor_R, frame):
    out = np.array(action, dtype=np.float64, copy=True)
    se3 = su.pose9_to_SE3(out[:, :9])
    se3[:, :3, 3] = se3[:, :3, 3] + (anchor_R.T @ (frame[:, :, 0] * (OFFSET_MM / 1000.0)).T).T
    about_n = Rotation.from_rotvec((-frame[:, :, 2]) * np.deg2rad(OFFSET_DEG)).as_matrix()
    rel = se3[:, :3, :3]
    rel = np.einsum("ij,njk->nik", anchor_R, rel)
    rel = np.einsum("nij,njk->nik", about_n, rel)
    se3[:, :3, :3] = np.einsum("ij,njk->nik", anchor_R.T, rel)
    out[:, :9] = su.SE3_to_pose9(se3)
    return out


def test_gt_labels_are_zero():
    demo, index, action, axis = _causal()
    tilt = _tilt_action(demo, index, axis, action)
    rel_demo, rel_index, relative, _anchor_R = _r1rel()
    for kind, act, this_demo, this_index in (
        ("causal", action, demo, index),
        ("tilt", tilt, demo, index),
        ("r1rel", relative, rel_demo, rel_index),
    ):
        pos, rot, _xyz = scorer.pose_error(kind, act, this_demo, this_index, axis)
        assert pos.max() < 0.05, f"{kind} pos {pos.max()}"
        assert rot.max() < 0.05, f"{kind} rot {rot.max()}"


def test_known_offset_is_exact():
    demo, index, action, axis = _causal()
    dt, _nxt = scorer._control_step(demo, index)
    tilt = _offset_surface("tilt", _tilt_action(demo, index, axis, action), dt)
    causal = _offset_surface("causal", action, dt)
    rel_demo, rel_index, relative, anchor_R = _r1rel()
    rel_frame = demo["frame"][index]
    relative = _offset_r1rel(relative, anchor_R, rel_frame)
    for kind, act, this_demo, this_index in (
        ("causal", causal, demo, index),
        ("tilt", tilt, demo, index),
        ("r1rel", relative, rel_demo, rel_index),
    ):
        pos, rot, _xyz = scorer.pose_error(kind, act, this_demo, this_index, axis)
        assert np.allclose(pos, OFFSET_MM, atol=0.05), f"{kind} pos {pos[:4]}"
        assert np.allclose(rot, OFFSET_DEG, atol=0.05), f"{kind} rot {rot[:4]}"


def _predict(policy, dataset, idx, device):
    obs, _sample = dataset.sampler.sample_sequence(idx)
    pred = np.asarray(predict_sparse_action(policy, obs, device), dtype=np.float64)
    return pred.reshape(-1, pred.shape[-1])[: scorer.heldout.H]


def test_two_ckpts_differ_from_each_other_and_the_label():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, cfg = load_policy_and_cfg(str(K3000_E25), device)
    val, zarr_path = scorer._train_val(cfg, scorer._norm(cfg, "causal"))
    demos, grouped, _n = scorer._train_rows(val, zarr_path)
    epi, rows = next(iter(grouped.items()))
    idx, _rgb, index = rows[0]
    demo = demos[epi]
    pred_a = _predict(policy, val, idx, device)
    del policy
    torch.cuda.empty_cache()
    policy, _cfg = load_policy_and_cfg(str(K3000_E199), device)
    pred_b = _predict(policy, val, idx, device)
    del policy
    torch.cuda.empty_cache()
    label = _action10_rows(zarr.open(zarr_path, "r")["data"][f"episode_{epi}"], index)
    _pa, _ra, xyz_a = scorer.pose_error("causal", pred_a, demo, index)
    _pb, _rb, xyz_b = scorer.pose_error("causal", pred_b, demo, index)
    _pl, _rl, xyz_l = scorer.pose_error("causal", label, demo, index)
    assert not np.allclose(xyz_a[0], xyz_b[0], atol=1e-4)
    assert not np.allclose(xyz_a[0], xyz_l[0], atol=1e-4)
    assert not np.allclose(xyz_b[0], xyz_l[0], atol=1e-4)


def _tilt_scale():
    import pickle

    normalizer = pickle.load(scorer.TILT_NORM.open("rb"))
    return float(normalizer.params_dict["action"]["scale"].detach().reshape(-1)[4])


def test_tilt1_training_rotation_within_twice_val_loss():
    scale_a = _tilt_scale()
    cap = 2.0 * float(np.degrees(TILT_VAL_LOSS / scale_a))
    assert cap < 40.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, cfg = load_policy_and_cfg(str(TILT_E25), device)
    val, zarr_path = scorer._train_val(cfg, scorer.TILT_NORM)
    demos, grouped, _n = scorer._train_rows(val, zarr_path)
    axis = _axis()
    chosen = []
    for epi, rows in grouped.items():
        for row in rows:
            chosen.append((epi, row))
            if len(chosen) == 5:
                break
        if len(chosen) == 5:
            break
    assert len(chosen) == 5
    step1 = []
    for epi, (idx, _rgb, index) in chosen:
        pred = _predict(policy, val, idx, device)
        _pos, rot, _xyz = scorer.pose_error("tilt1", pred, demos[epi], index, axis)
        step1.append(float(rot[0]))
    assert float(np.mean(step1)) < cap, f"step1 {step1} cap {cap}"
