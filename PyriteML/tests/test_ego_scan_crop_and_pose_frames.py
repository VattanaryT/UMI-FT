"""Tests for the scan-crop / absolute-pose experiments (umift_ss_v2_* task configs).

Run from the UMI-FT repo root:
    PYTHONPATH=PyriteML:. pytest PyriteML/tests/test_ego_scan_crop_and_pose_frames.py
"""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import PyriteUtility.spatial_math.spatial_utilities as su
from diffusion_policy.common.drop_scan_inputs import apply_drop_scan_inputs
from diffusion_policy.common.sampler_umift import filter_rgb_ids_near_bad_poses
from diffusion_policy.model.vision.scan_heatmap_encoder import ScanHeatmapEncoder
from PyriteConfig.tasks.umift import umift_type_conversions as tc
from PyriteConfig.tasks.umift.scan_crop import crop_scan_heatmap_egocentric
from PyriteConfig.tasks.umift.workcell_frames import (
    load_workcell_extrinsics,
    pose_frames_cfg,
    scan_crop_cfg,
    shape_meta_needs_fixture_frame,
)

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "diffusion_policy" / "config")
EXT = load_workcell_extrinsics()
EXTENT = EXT.scan_extent()
CROP = {"enabled": True, "size_m": 0.16, "pixels": 64, "heading_axis": "x"}


def _shape_meta(task: str) -> dict:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="train_umift_workspace", overrides=[f"task={task}"])
    return OmegaConf.to_container(cfg.task.shape_meta, resolve=True)


def _tool_in_scanner(x, z, theta, y=0.8):
    """SE3 of a tool in scanner_base: x axis=(cos t,0,sin t), y axis=(0,1,0), z=x*y."""
    ct, st = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[:3, 0] = [ct, 0, st]
    T[:3, 1] = [0, 1, 0]
    T[:3, 2] = [-st, 0, ct]
    T[:3, 3] = [x, y, z]
    return T


def _to_fixture(T_scanner_tool):
    return np.linalg.inv(EXT.T_scanner_base_from_fixture) @ T_scanner_tool


def _bump_heatmap(x, z, sigma=0.004, H=80, W=160):
    """(1,H,W,3) heatmap with a gaussian bump at scanner (x, z)."""
    cols = EXTENT["x_min"] + (np.arange(W) + 0.5) / W * (EXTENT["x_max"] - EXTENT["x_min"])
    rows = EXTENT["z_min"] + (np.arange(H) + 0.5) / H * (EXTENT["z_max"] - EXTENT["z_min"])
    X, Z = np.meshgrid(cols, rows)
    hm = 3.0 * np.exp(-((X - x) ** 2 + (Z - z) ** 2) / (2 * sigma**2))
    return np.repeat(hm[None, :, :, None], 3, axis=-1).astype(np.float32)


def _peak(patch):
    plane = np.nan_to_num(patch[..., 0], nan=-1)
    i, j = np.unravel_index(np.argmax(plane), plane.shape)
    return i, j


# ---------------------------------------------------------------- crop geometry
def test_crop_bump_under_tool_is_at_center():
    x0, z0 = 0.55, 0.49
    hm = _bump_heatmap(x0, z0)
    out = crop_scan_heatmap_egocentric(hm, _to_fixture(_tool_in_scanner(x0, z0, 0.0)), CROP)
    assert out.shape == (1, 64, 64, 3)
    i, j = _peak(out[0])
    assert abs(i - 31.5) <= 1 and abs(j - 31.5) <= 1


def test_crop_follows_heading_axis():
    x0, z0, d = 0.55, 0.49, 0.04
    hm = _bump_heatmap(x0 + d, z0)  # bump 4 cm along scanner +x from the tool
    px_per_m = 64 / CROP["size_m"]

    # tool x axis along +x: bump is to the right (+col), same row
    i, j = _peak(
        crop_scan_heatmap_egocentric(hm, _to_fixture(_tool_in_scanner(x0, z0, 0.0)), CROP)[0]
    )
    assert abs(j - (31.5 + d * px_per_m)) <= 1.5 and abs(i - 31.5) <= 1.5

    # tool x axis along +z (yaw 90 deg): +x scanner is now -v -> up (-row), same col
    i, j = _peak(
        crop_scan_heatmap_egocentric(hm, _to_fixture(_tool_in_scanner(x0, z0, np.pi / 2)), CROP)[0]
    )
    assert abs(i - (31.5 - d * px_per_m)) <= 1.5 and abs(j - 31.5) <= 1.5

    # tool x axis along -x (yaw 180 deg): bump is behind the tool (-col)
    i, j = _peak(
        crop_scan_heatmap_egocentric(hm, _to_fixture(_tool_in_scanner(x0, z0, np.pi)), CROP)[0]
    )
    assert abs(j - (31.5 - d * px_per_m)) <= 1.5 and abs(i - 31.5) <= 1.5


def _tool_down_in_scanner(x, z, theta, y=0.8):
    """Real convention: tool z points down (+Y), tool x in the XZ plane at yaw theta,
    tool y = z cross x."""
    ct, st = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[:3, 0] = [ct, 0, st]
    T[:3, 1] = [st, 0, -ct]
    T[:3, 2] = [0, 1, 0]
    T[:3, 3] = [x, y, z]
    assert np.allclose(np.cross(T[:3, 0], T[:3, 1]), T[:3, 2])
    return T


def _ramp_heatmap(H=80, W=160):
    """ch0 = scanner x [m], ch1 = scanner z [m] at every pixel (all valid)."""
    cols = EXTENT["x_min"] + (np.arange(W) + 0.5) / W * (EXTENT["x_max"] - EXTENT["x_min"])
    rows = EXTENT["z_min"] + (np.arange(H) + 0.5) / H * (EXTENT["z_max"] - EXTENT["z_min"])
    X, Z = np.meshgrid(cols, rows)
    return np.stack([X, Z, X], axis=-1)[None].astype(np.float32)


def _patch_axes_xz(cfg, theta):
    """Scanner (x, z) unit directions of patch-right and patch-down, from ramp gradients."""
    T = _to_fixture(_tool_down_in_scanner(0.55, 0.49, theta))
    patch = crop_scan_heatmap_egocentric(_ramp_heatmap(), T, cfg)[0]  # (P, P, 3)
    P = patch.shape[0]
    mid = slice(P // 2 - 8, P // 2 + 8)
    right = np.array([np.mean(patch[mid, -1, k] - patch[mid, 0, k]) for k in (0, 1)])
    down = np.array([np.mean(patch[-1, mid, k] - patch[0, mid, k]) for k in (0, 1)])
    return right / np.linalg.norm(right), down / np.linalg.norm(down)


@pytest.mark.parametrize("theta", [0.0, 0.7, 1.9, -2.4, 3.0])
def test_crop_camera_aligned_orientation(theta):
    """heading_axis '-x' + flip_v: patch right = -tool x, patch down = -tool y, which
    matches the wrist camera image (and a proper, non-mirrored top-down view)."""
    cfg = dict(CROP, heading_axis="-x", flip_v=True)
    right, down = _patch_axes_xz(cfg, theta)
    tool = _tool_down_in_scanner(0.55, 0.49, theta)
    want_right = -tool[[0, 2], 0]
    want_down = -tool[[0, 2], 1]
    assert np.allclose(right, want_right, atol=1e-2)
    assert np.allclose(down, want_down, atol=1e-2)
    # right x down must point along the camera viewing direction (tool z = +Y, i.e.
    # into the part): with scan axes (x, z) that means cross(right, down) = -1 in (x, z)
    # because x cross z = -Y.
    assert right[0] * down[1] - right[1] * down[0] < 0


@pytest.mark.parametrize("theta", [0.0, 0.7, 1.9])
def test_crop_legacy_orientation_is_mirrored_vs_camera(theta):
    """Default (heading x, no flip) keeps the original behaviour: right = +tool x and
    rows along +e_v, which is a mirror image of the camera / top-down view."""
    right, down = _patch_axes_xz(CROP, theta)
    tool = _tool_down_in_scanner(0.55, 0.49, theta)
    assert np.allclose(right, tool[[0, 2], 0], atol=1e-2)
    assert np.allclose(down, -tool[[0, 2], 1], atol=1e-2)
    assert right[0] * down[1] - right[1] * down[0] > 0  # opposite handedness to the camera-aligned crop


def test_crop_outside_scan_is_nan_and_far_tool_is_all_nan():
    hm = np.ones((1, 80, 160, 3), dtype=np.float32)
    near_edge = _to_fixture(_tool_in_scanner(EXTENT["x_max"] - 0.01, 0.49, 0.0))
    out = crop_scan_heatmap_egocentric(hm, near_edge, CROP)
    assert np.isnan(out).any() and np.isfinite(out).any()
    far = _to_fixture(_tool_in_scanner(EXTENT["x_max"] + 0.5, 0.49, 0.0))
    assert np.isnan(crop_scan_heatmap_egocentric(hm, far, CROP)).all()


def test_crop_propagates_nan_invalid_pixels_and_values_are_unbiased():
    hm = np.full((1, 80, 160, 3), 2.5, dtype=np.float32)
    hm[0, 30:34, 70:90, :] = np.nan
    T = _to_fixture(_tool_in_scanner(0.55, 0.49, 0.3))
    out = crop_scan_heatmap_egocentric(hm, T, CROP)
    finite = out[np.isfinite(out)]
    assert finite.size > 0 and np.allclose(finite, 2.5, atol=1e-4)  # normalized bilinear
    assert np.isnan(out).any()


def test_crop_zero_heatmap_stays_zero():
    out = crop_scan_heatmap_egocentric(
        np.zeros((1, 80, 160, 3), np.float32), _to_fixture(_tool_in_scanner(0.55, 0.49, 0.0)), CROP
    )
    assert np.nanmax(np.abs(out)) == 0.0


# ------------------------------------------------------------------- pose frames
def _obs_sparse(sm, T=5, rng=None):
    rng = rng or np.random.default_rng(0)
    # a plausible fixture-frame trajectory over the part
    pos = np.stack(
        [np.linspace(0.1, 0.2, T), np.linspace(-0.05, -0.04, T), np.linspace(0.2, 0.25, T)], axis=-1
    )
    quat = np.tile([0.8, 0.1, 0.5, 0.3], (T, 1))
    pose7 = np.concatenate([pos, quat], axis=-1)
    pose9 = su.SE3_to_pose9(su.pose7_to_SE3(pose7)).astype(np.float32)
    obs = {
        "robot0_eef_pos": pose9[..., :3].copy(),
        "robot0_eef_rot_axis_angle": pose9[..., 3:].copy(),
        # abs keys use horizon 1 in umift_ss_v2_abs_input (current pose only)
        "robot0_abs_eef_pos": pose9[-1:, :3].copy(),
        "robot0_abs_eef_rot_axis_angle": pose9[-1:, 3:].copy(),
        "robot0_eef_wrench_left": rng.normal(size=(32, 6)).astype(np.float32),
        "robot0_eef_wrench_right": rng.normal(size=(32, 6)).astype(np.float32),
    }
    return obs, pose9


def _sm_lowdim_only(task):
    sm = _shape_meta(task)
    sm = copy.deepcopy(sm)
    # keep the test about low-dim/frames: drop vision/scan from sample.obs.sparse.
    for k in list(sm["sample"]["obs"]["sparse"].keys()):
        if k.startswith(("rgb", "depth", "scan_heatmap")):
            del sm["sample"]["obs"]["sparse"][k]
    # horizons: poses T=5 (abs: 1), wrench 32
    return sm


def _relative_reference(pose9):
    SE3 = su.pose9_to_SE3(pose9)
    return su.SE3_to_pose9(su.SE3_inv(SE3[-1]) @ SE3)


def test_relative_mode_is_unchanged_and_base_is_newest_pose():
    sm = _sm_lowdim_only("umift_ss_v2")
    obs, pose9 = _obs_sparse(sm)
    out, base = tc.sparse_obs_to_obs_sample(obs, sm, "reshape", [0], ignore_rgb=True)
    ref = _relative_reference(pose9)
    np.testing.assert_allclose(out["robot0_eef_pos"], ref[..., :3], atol=1e-5)
    np.testing.assert_allclose(out["robot0_eef_rot_axis_angle"], ref[..., 3:], atol=1e-5)
    np.testing.assert_allclose(base[0], su.pose9_to_SE3(pose9)[-1], atol=1e-6)
    assert out["robot0_eef_pos"].dtype == np.float32
    np.testing.assert_allclose(out["robot0_eef_pos"][-1], 0, atol=1e-6)


def test_absolute_obs_relative_action_and_world_transform():
    sm = _sm_lowdim_only("umift_ss_v2_abs_obs")
    obs, pose9 = _obs_sparse(sm)
    # training: obs already in the fixture frame
    out, base = tc.sparse_obs_to_obs_sample(obs, sm, "reshape", [0], ignore_rgb=True)
    np.testing.assert_allclose(out["robot0_eef_pos"], pose9[..., :3], atol=1e-5)
    np.testing.assert_allclose(out["robot0_eef_rot_axis_angle"], pose9[..., 3:], atol=1e-5)
    np.testing.assert_allclose(base[0], su.pose9_to_SE3(pose9)[-1], atol=1e-6)  # action: relative
    assert out["robot0_eef_pos"].dtype == np.float32

    # robot: obs come in the right-arm base frame; output is still the fixture-frame pose
    SE3_WF = EXT.T_right_base_from_fixture  # world(right_base) <- fixture
    SE3_world = SE3_WF @ su.pose9_to_SE3(pose9)
    obs_w = dict(obs)
    p9w = su.SE3_to_pose9(SE3_world).astype(np.float32)
    obs_w["robot0_eef_pos"], obs_w["robot0_eef_rot_axis_angle"] = p9w[..., :3], p9w[..., 3:]
    out_w, _ = tc.sparse_obs_to_obs_sample(
        obs_w, sm, "reshape", [0], ignore_rgb=True, SE3_fixture_from_world=EXT.T_fixture_from_right_base
    )
    np.testing.assert_allclose(out_w["robot0_eef_pos"], pose9[..., :3], atol=2e-5)
    np.testing.assert_allclose(out_w["robot0_eef_rot_axis_angle"], pose9[..., 3:], atol=2e-5)


def test_absolute_action_round_trip_to_robot_base():
    sm = _sm_lowdim_only("umift_ss_v2_abs_obs_action")
    obs, pose9 = _obs_sparse(sm)
    # label (fixture-frame action) -> relative-to-base conversion in training: base = identity
    out, base_train = tc.sparse_obs_to_obs_sample(obs, sm, "reshape", [0], ignore_rgb=True)
    np.testing.assert_allclose(base_train[0], np.eye(4), atol=1e-6)
    act_F = su.SE3_to_pose9(su.pose9_to_SE3(pose9[:3])).astype(np.float32)
    act_label = tc.action9_to_action_sample(act_F, [], [0], base_train)["sparse"]
    np.testing.assert_allclose(act_label, act_F, atol=1e-5)  # absolute label == fixture pose

    # robot: base = world_from_fixture, postprocess returns right_base poses
    SE3_W = EXT.T_right_base_from_fixture @ su.pose9_to_SE3(pose9)
    p9w = su.SE3_to_pose9(SE3_W).astype(np.float32)
    obs_w = dict(obs)
    obs_w["robot0_eef_pos"], obs_w["robot0_eef_rot_axis_angle"] = p9w[..., :3], p9w[..., 3:]
    _, base_robot = tc.sparse_obs_to_obs_sample(
        obs_w, sm, "reshape", [0], ignore_rgb=True, SE3_fixture_from_world=EXT.T_fixture_from_right_base
    )
    np.testing.assert_allclose(base_robot[0], EXT.T_right_base_from_fixture, atol=1e-5)
    world = tc.action9_postprocess(act_label, base_robot, [0])[0]
    expected = EXT.T_right_base_from_fixture @ su.pose9_to_SE3(act_F)
    np.testing.assert_allclose(world, expected, atol=1e-4)


def test_extra_abs_keys_are_fixture_frame_and_independent_of_relative_obs():
    sm = _sm_lowdim_only("umift_ss_v2_abs_input")
    obs, pose9 = _obs_sparse(sm)
    out, _ = tc.sparse_obs_to_obs_sample(obs, sm, "reshape", [0], ignore_rgb=True)
    ref = _relative_reference(pose9)
    np.testing.assert_allclose(out["robot0_eef_pos"], ref[..., :3], atol=1e-5)  # relative kept
    np.testing.assert_allclose(out["robot0_abs_eef_pos"], pose9[-1:, :3], atol=1e-5)
    np.testing.assert_allclose(out["robot0_abs_eef_rot_axis_angle"], pose9[-1:, 3:], atol=1e-5)


def test_crop_runs_through_obs_pipeline_and_matches_declared_shape():
    sm = _shape_meta("umift_ss_v2_crop")
    obs, pose9 = _obs_sparse(sm)
    obs["rgb_0"] = np.random.default_rng(1).integers(0, 255, (5, 224, 224, 3), dtype=np.uint8)
    obs["depth_0"] = np.zeros((5, 224, 224, 3), dtype=np.float16)
    # place the tool over the part: choose fixture pose = tool at scanner (0.55, 0.49)
    T_fix = _to_fixture(_tool_in_scanner(0.55, 0.49, 0.4))
    p9 = np.tile(su.SE3_to_pose9(T_fix).astype(np.float32), (5, 1))
    obs["robot0_eef_pos"], obs["robot0_eef_rot_axis_angle"] = p9[:, :3].copy(), p9[:, 3:].copy()
    obs["scan_heatmap_0"] = _bump_heatmap(0.55, 0.49)
    out, _ = tc.sparse_obs_to_obs_sample(obs, sm, "reshape", [0])
    assert out["scan_heatmap_0"].shape == (1, 3, 64, 64)
    assert out["scan_heatmap_0"].dtype == np.float32
    assert 0.0 <= out["scan_heatmap_0"].min() and out["scan_heatmap_0"].max() <= 1.0
    i, j = _peak(np.moveaxis(out["scan_heatmap_0"][0], 0, -1))
    assert abs(i - 31.5) <= 1 and abs(j - 31.5) <= 1


def test_crop_identical_for_training_frame_and_robot_base_frame():
    sm = _shape_meta("umift_ss_v2_crop")
    obs, _ = _obs_sparse(sm)
    obs["rgb_0"] = np.zeros((5, 224, 224, 3), dtype=np.uint8)
    obs["depth_0"] = np.zeros((5, 224, 224, 3), dtype=np.float16)
    obs["scan_heatmap_0"] = _bump_heatmap(0.60, 0.50)
    T_fix = _to_fixture(_tool_in_scanner(0.58, 0.49, 0.7))
    p9 = np.tile(su.SE3_to_pose9(T_fix).astype(np.float32), (5, 1))
    obs["robot0_eef_pos"], obs["robot0_eef_rot_axis_angle"] = p9[:, :3].copy(), p9[:, 3:].copy()
    train_out, _ = tc.sparse_obs_to_obs_sample(obs, sm, "reshape", [0])

    # same physical pose, reported by RTDE in the right-arm base frame
    p9w = su.SE3_to_pose9(EXT.T_right_base_from_fixture @ su.pose9_to_SE3(p9)).astype(np.float32)
    obs_w = dict(obs)
    obs_w["robot0_eef_pos"], obs_w["robot0_eef_rot_axis_angle"] = p9w[:, :3], p9w[:, 3:]
    robot_out, _ = tc.sparse_obs_to_obs_sample(
        obs_w, sm, "reshape", [0], SE3_fixture_from_world=EXT.T_fixture_from_right_base
    )
    np.testing.assert_allclose(
        robot_out["scan_heatmap_0"], train_out["scan_heatmap_0"], atol=2e-3
    )
    assert train_out["scan_heatmap_0"].max() > 0.1  # the bump is actually in view


# ------------------------------------------------------------ configs / helpers
def test_task_configs_flags():
    base = _shape_meta("umift_ss_v2")
    assert pose_frames_cfg(base) == {"obs": "relative", "action": "relative"}
    assert scan_crop_cfg(base) is None and not shape_meta_needs_fixture_frame(base)

    crop = _shape_meta("umift_ss_v2_crop")
    assert scan_crop_cfg(crop) is not None and shape_meta_needs_fixture_frame(crop)
    assert crop["obs"]["scan_heatmap_0"]["shape"] == [3, 64, 64]
    assert crop["raw"]["scan_heatmap_0"]["shape"] == [3, 80, 160]

    assert shape_meta_needs_fixture_frame(_shape_meta("umift_ss_v2_abs_input"))
    assert pose_frames_cfg(_shape_meta("umift_ss_v2_abs_obs"))["obs"] == "absolute"
    f = pose_frames_cfg(_shape_meta("umift_ss_v2_abs_obs_action"))
    assert f == {"obs": "absolute", "action": "absolute"}
    # legacy task config has no pose_frames at all
    assert pose_frames_cfg(_shape_meta("umift_zucchini")) == {"obs": "relative", "action": "relative"}


def _compose(task: str):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="train_umift_workspace", overrides=[f"task={task}"])
    apply_drop_scan_inputs(cfg)
    OmegaConf.resolve(cfg)
    apply_drop_scan_inputs(cfg)
    return cfg


def test_no_scan_task_drops_scan_inputs():
    cfg = _compose("umift_ss_v2_no_scan")
    assert cfg.task.name == "UMIFT-SS-V2-NO-SCAN"
    assert cfg.task.drop_scan_inputs is True
    for shape_meta in (
        cfg.task.shape_meta,
        cfg.policy.shape_meta,
        cfg.policy.obs_encoder.shape_meta,
        cfg.task.dataset.shape_meta,
    ):
        for section in ("raw", "obs"):
            assert not any("scan" in key for key in shape_meta[section])
        assert not any(
            "scan" in key for key in shape_meta["sample"]["obs"]["sparse"]
        )
        assert "rgb_0" in shape_meta["obs"]
        assert "depth_0" in shape_meta["sample"]["obs"]["sparse"]
        assert "robot0_eef_wrench_left" in shape_meta["obs"]

    # The baseline task is unchanged: no flag, scan input stays.
    base = _compose("umift_ss_v2")
    assert "scan_heatmap_0" in base.policy.shape_meta["obs"]
    assert "scan_heatmap_0" in base.policy.shape_meta["sample"]["obs"]["sparse"]


def test_scan_gap_intervals_copied_without_scan_obs():
    pose = np.zeros((2, 7), dtype=np.float64)
    pose[:, 3] = 1.0
    stamps = np.arange(2, dtype=np.float64).reshape(-1, 1)
    base = {
        "ts_pose_fb_0": pose,
        "rgb_time_stamps_0": stamps,
        "robot_time_stamps_0": stamps,
    }
    shape_meta = {"id_list": [0], "raw": {}, "obs": {}}
    episode = {}
    tc.raw_to_obs(
        {**base, "scan_gap_intervals_0": np.array([[0.0, 1.5]], dtype=np.float32)},
        episode,
        shape_meta,
    )
    np.testing.assert_allclose(episode["obs"]["scan_gap_intervals_0"], [[0.0, 1.5]])

    episode = {}
    tc.raw_to_obs(base, episode, shape_meta)
    assert "scan_gap_intervals_0" not in episode["obs"]


def test_filter_near_bad_poses():
    rgb_times = np.arange(0, 20, 1 / 15)
    ids = np.arange(len(rgb_times))
    bad_times = np.array([10.0, 10.005, 10.01])
    kept, n = filter_rgb_ids_near_bad_poses(ids, rgb_times, bad_times, 1.0, 2.0)
    t = rgb_times[kept]
    assert n == len(ids) - len(kept) > 0
    assert not np.any((t > 10 - 2.0 + 1e-6) & (t < 10 + 1.0 - 1e-6))  # window [t-1, t+2] hits 10.0
    assert np.all((t < 8.0) | (t > 11.0 + 1e-6))
    kept2, n2 = filter_rgb_ids_near_bad_poses(ids, rgb_times, np.empty(0), 1.0, 2.0)
    assert n2 == 0 and len(kept2) == len(ids)


def test_scan_encoder_pooling_modes():
    x_full = torch.randn(2, 3, 80, 160)
    x_crop = torch.randn(2, 3, 64, 64)
    avg = ScanHeatmapEncoder(3, 128)  # legacy default
    assert avg(x_full).shape == (2, 128) and avg(x_crop).shape == (2, 128)
    flat = ScanHeatmapEncoder(3, 128, pooling="flatten", input_hw=(64, 64))
    assert flat(x_crop).shape == (2, 128)
    # flatten keeps spatial layout: moving the content changes the feature, avg-pool is ~invariant
    a = torch.zeros(1, 3, 64, 64)
    a[:, :, 10:20, 10:20] = 1
    b = torch.roll(a, shifts=24, dims=-1)
    assert not torch.allclose(flat(a), flat(b), atol=1e-4)
    with pytest.raises(AssertionError):
        ScanHeatmapEncoder(3, 128, pooling="flatten")


# ------------------------------------------------------- spatial softmax pooling
def test_spatial_softmax_shapes_and_state_dict_compat():
    ss = ScanHeatmapEncoder(3, 128, pooling="spatial_softmax", num_keypoints=16)
    for hw in [(64, 64), (80, 160)]:
        assert ss(torch.randn(2, 3, *hw)).shape == (2, 128)
    # 16 keypoints -> 32 coords + 64 avg features
    assert ss.proj.weight.shape == (128, 32 + 64)
    no_avg = ScanHeatmapEncoder(3, 128, pooling="spatial_softmax", num_keypoints=16, include_avg=False)
    assert no_avg.proj.weight.shape == (128, 32)
    # feature map is input/8 with the default 3 stride-2 convs
    assert ss.backbone(torch.zeros(1, 3, 64, 64)).shape[-2:] == (8, 8)
    # the original modes are untouched (same backbone layout / parameter names)
    avg = ScanHeatmapEncoder(3, 128)
    assert avg.proj.weight.shape == (128, 64)
    assert avg.backbone(torch.zeros(1, 3, 64, 64)).shape[-2:] == (4, 4)
    assert "keypoint_conv.weight" not in avg.state_dict()
    assert ScanHeatmapEncoder(3, 128, pooling="avg", num_downsample=3).backbone(
        torch.zeros(1, 3, 64, 64)
    ).shape[-2:] == (8, 8)


def test_spatial_softmax_returns_expected_coordinates():
    enc = ScanHeatmapEncoder(3, 8, pooling="spatial_softmax", num_keypoints=2)
    h, w = 8, 16
    feat = torch.zeros(1, 64, h, w)
    feat[0, 0, 2, 12] = 1.0  # channel 0 peaks at row 2, col 12
    feat[0, 1, 6, 3] = 1.0  # channel 1 peaks at row 6, col 3
    with torch.no_grad():
        enc.keypoint_conv.weight.zero_()
        enc.keypoint_conv.bias.zero_()
        enc.keypoint_conv.weight[0, 0, 0, 0] = 200.0  # sharp -> ~argmax
        enc.keypoint_conv.weight[1, 1, 0, 0] = 200.0
    out = enc._spatial_softmax(feat)  # (1, 4) = [x0, x1, y0, y1]
    xs = torch.linspace(-1, 1, w)
    ys = torch.linspace(-1, 1, h)
    want = torch.stack([xs[12], xs[3], ys[2], ys[6]])
    assert torch.allclose(out[0], want, atol=1e-3)


def test_spatial_softmax_is_position_sensitive_and_trainable():
    torch.manual_seed(0)
    a = torch.zeros(1, 3, 64, 64)
    a[:, :, 10:20, 10:20] = 1
    b = torch.roll(a, shifts=24, dims=-1)
    ss = ScanHeatmapEncoder(3, 128, pooling="spatial_softmax")

    # Position information that actually reaches the pooled features: the keypoint
    # coordinates move with the blob, the avg-pooled features barely do.
    with torch.no_grad():
        kp_change = (ss._spatial_softmax(ss.backbone(a)) - ss._spatial_softmax(ss.backbone(b))).norm()
        avg_change = (ss.pool(ss.backbone(a)) - ss.pool(ss.backbone(b))).norm()  # same backbone
    assert kp_change > 5 * avg_change
    assert not torch.allclose(ss(a), ss(b), atol=1e-5)
    # gradients reach the keypoint head and the temperature
    ss(torch.randn(2, 3, 64, 64)).sum().backward()
    assert ss.temperature.grad is not None and ss.temperature.grad.abs().sum() > 0
    assert ss.keypoint_conv.weight.grad.abs().sum() > 0
    # works under bf16 autocast (softmax is done in fp32 internally)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = ss(torch.randn(2, 3, 64, 64))
    assert torch.isfinite(out.float()).all()


def test_spatial_softmax_via_policy_config():
    from hydra.utils import instantiate

    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="train_umift_workspace",
            overrides=[
                "task=umift_ss_v2_crop",
                "policy.obs_encoder.scan_heatmap_encoder_cfg.pooling=spatial_softmax",
                "policy.obs_encoder.scan_heatmap_encoder_cfg.num_keypoints=24",
            ],
        )
    enc_cfg = cfg.policy.obs_encoder.scan_heatmap_encoder_cfg
    assert enc_cfg.pooling == "spatial_softmax" and enc_cfg.num_keypoints == 24
    assert enc_cfg.num_downsample is None and enc_cfg.include_avg is None
