"""Tests for inference episode vision timeline logging and playback."""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import zarr


def _import_plot_module(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "plotly.graph_objects",
        types.SimpleNamespace(Scatter=object, Scatter3d=object, Image=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "plotly.subplots",
        types.SimpleNamespace(make_subplots=lambda *args, **kwargs: None),
    )
    return importlib.import_module("PyriteEnvSuites.scripts.plot_inference_episode")


def _write_horizon(
    root,
    idx: int,
    *,
    obs_inference: dict | None = None,
    obs_policy: dict | None = None,
    action_t0: float = 0.0,
):
    g = root.create_group(f"horizon_{idx}")
    g.create_group("meta").attrs["hardware_time_s"] = float(action_t0)
    g.create_group("action").create_dataset(
        "timestamps_s", data=np.asarray([action_t0], dtype=np.float64)
    )
    if obs_inference is not None:
        oi = g.create_group("obs_inference")
        for key, value in obs_inference.items():
            oi.create_dataset(key, data=value)
    if obs_policy is not None:
        op = g.create_group("obs_policy")
        for key, value in obs_policy.items():
            op.create_dataset(key, data=value)


def test_collect_vision_timeline_uses_raw_buffers_and_dedupes(monkeypatch, tmp_path):
    plot = _import_plot_module(monkeypatch)
    store = zarr.DirectoryStore(str(tmp_path / "episode.zarr"))
    root = zarr.group(store=store)

    rgb_a = np.stack(
        [
            np.full((4, 4, 3), 10, dtype=np.uint8),
            np.full((4, 4, 3), 20, dtype=np.uint8),
        ],
        axis=0,
    )
    rgb_b = np.stack(
        [
            np.full((4, 4, 3), 20, dtype=np.uint8),
            np.full((4, 4, 3), 30, dtype=np.uint8),
        ],
        axis=0,
    )
    depth_a = np.ones((2, 4, 4, 3), dtype=np.float16) * 0.1
    depth_b = np.ones((2, 4, 4, 3), dtype=np.float16) * 0.2

    _write_horizon(
        root,
        0,
        action_t0=100.0,
        obs_inference={
            "rgb_0": rgb_a,
            "rgb_time_stamps_0": np.asarray([100.0, 100.2], dtype=np.float64),
            "depth_0": depth_a,
            "depth_time_stamps_0": np.asarray([100.0, 100.2], dtype=np.float64),
        },
    )
    _write_horizon(
        root,
        1,
        action_t0=100.4,
        obs_inference={
            "rgb_0": rgb_b,
            "rgb_time_stamps_0": np.asarray([100.2, 100.4], dtype=np.float64),
            "depth_0": depth_b,
            "depth_time_stamps_0": np.asarray([100.2, 100.4], dtype=np.float64),
        },
    )

    rgb_images, rgb_times, depth_images, depth_times = plot.collect_vision_timeline(
        root, ["horizon_0", "horizon_1"], robot_id=0, max_horizons=2, episode_action_t0=100.0
    )

    assert len(rgb_images) == 3
    assert np.allclose(rgb_times, np.asarray([0.0, 0.2, 0.4]))
    assert rgb_images[0][0, 0, 0] == 10
    assert rgb_images[-1][0, 0, 0] == 30
    assert len(depth_images) == 3
    assert plot.vision_index_for_time(0.25, rgb_times) == 1
    assert plot.vision_index_for_time(0.4, rgb_times) == 2


def test_collect_vision_timeline_falls_back_to_obs_policy(monkeypatch, tmp_path):
    plot = _import_plot_module(monkeypatch)
    store = zarr.DirectoryStore(str(tmp_path / "episode_old.zarr"))
    root = zarr.group(store=store)

    policy_rgb = np.stack(
        [
            np.zeros((3, 2, 2), dtype=np.float32),
            np.ones((3, 2, 2), dtype=np.float32),
        ],
        axis=0,
    )
    _write_horizon(
        root,
        0,
        action_t0=5.0,
        obs_policy={"rgb_0": policy_rgb},
    )

    rgb_images, rgb_times, depth_images, depth_times = plot.collect_vision_timeline(
        root, ["horizon_0"], robot_id=0, max_horizons=1, episode_action_t0=5.0
    )

    assert len(rgb_images) == 1
    assert rgb_times.tolist() == [0.0]
    assert depth_images == []
    assert depth_times.size == 0


def test_log_raw_vision_to_obs_inference(monkeypatch):
    monkeypatch.setenv("PYRITE_CHECKPOINT_FOLDERS", "/tmp")
    monkeypatch.setenv("PYRITE_HARDWARE_CONFIG_FOLDERS", "/tmp")
    monkeypatch.setenv("PYRITE_CONTROL_LOG_FOLDERS", "/tmp")

    common = types.ModuleType("PyriteUtility.common")
    common.GracefulKiller = object
    monkeypatch.setitem(sys.modules, "PyriteUtility.common", common)

    mpc = types.ModuleType("PyriteUtility.planning_control.mpc")
    mpc.ModelPredictiveController = object
    monkeypatch.setitem(sys.modules, "PyriteUtility.planning_control.mpc", mpc)

    plotting = types.ModuleType("PyriteUtility.plotting.matplotlib_helpers")
    plotting.set_axes_equal = lambda ax: None
    monkeypatch.setitem(sys.modules, "PyriteUtility.plotting.matplotlib_helpers", plotting)

    model_io = types.ModuleType("PyriteUtility.pytorch_utils.model_io")
    model_io.load_policy = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "PyriteUtility.pytorch_utils.model_io", model_io)

    spatial_math = types.ModuleType("PyriteUtility.spatial_math")
    spatial_math.spatial_utilities = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "PyriteUtility.spatial_math", spatial_math)

    type_conversions = types.ModuleType(
        "PyriteConfig.tasks.umift.umift_type_conversions"
    )
    type_conversions.raw_to_obs_inference = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "PyriteConfig.tasks.umift.umift_type_conversions",
        type_conversions,
    )

    env_module = types.ModuleType("PyriteEnvSuites.envs.task.manip_server_layup_env")
    env_module.ManipServerLayupEnv = object
    monkeypatch.setitem(
        sys.modules,
        "PyriteEnvSuites.envs.task.manip_server_layup_env",
        env_module,
    )

    env_utils = types.ModuleType("PyriteEnvSuites.utils.env_utils")
    env_utils.pose9g1_to_traj = lambda *args, **kwargs: None
    env_utils.pose9pose9s1a2_to_traj = lambda *args, **kwargs: None
    env_utils.ts_to_js_traj = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "PyriteEnvSuites.utils.env_utils", env_utils)

    runner = importlib.import_module("PyriteEnvSuites.env_runners.umift_env_runner")

    store = zarr.DirectoryStore(str(Path("/tmp") / "test_log_raw_vision.zarr"))
    root = zarr.group(store=store)
    oi = root.create_group("obs_inference")

    obs_raw = {
        "rgb_0": np.stack(
            [np.full((2, 2, 3), 1, dtype=np.uint8), np.full((2, 2, 3), 2, dtype=np.uint8)]
        ),
        "rgb_time_stamps_0": np.asarray([1.0, 1.1], dtype=np.float64),
        "depth_0": np.ones((2, 2, 2, 3), dtype=np.float16),
        "depth_time_stamps_0": np.asarray([1.0, 1.1], dtype=np.float64),
    }
    runner.log_raw_vision_to_obs_inference(oi, obs_raw, robot_id=0)

    assert oi["rgb_0"].shape == (2, 2, 2, 3)
    assert np.allclose(oi["rgb_time_stamps_0"][:], [1.0, 1.1])
    assert oi["depth_0"].shape == (2, 2, 2, 3)
    assert np.allclose(oi["depth_time_stamps_0"][:], [1.0, 1.1])


def test_composite_rgb_depth_side_by_side_doubles_width(monkeypatch):
    plot = _import_plot_module(monkeypatch)
    rgb = np.zeros((8, 6, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200
    depth = np.zeros((4, 3, 3), dtype=np.uint8)
    depth[:, :, 1] = 100
    frame = plot.composite_rgb_depth_side_by_side(rgb, depth)
    assert frame.shape == (8, 12, 3)
    assert frame.dtype == np.uint8


def test_build_vision_video_frames_samples_at_fps(monkeypatch):
    plot = _import_plot_module(monkeypatch)
    rgb_images = [
        np.full((4, 4, 3), 10, dtype=np.uint8),
        np.full((4, 4, 3), 20, dtype=np.uint8),
        np.full((4, 4, 3), 30, dtype=np.uint8),
    ]
    rgb_times = np.asarray([0.0, 0.2, 0.4], dtype=np.float64)
    depth_images = [
        np.full((4, 4, 3), 1, dtype=np.uint8),
        np.full((4, 4, 3), 2, dtype=np.uint8),
        np.full((4, 4, 3), 3, dtype=np.uint8),
    ]
    depth_times = np.asarray([0.0, 0.2, 0.4], dtype=np.float64)

    frames = plot.build_vision_video_frames(
        rgb_images, rgb_times, depth_images, depth_times, fps=15.0
    )
    assert len(frames) == 7
    assert frames[0].shape[1] == frames[0].shape[0] * 2


def test_export_episode_vision_video_writes_mp4(monkeypatch, tmp_path):
    plot = _import_plot_module(monkeypatch)
    rgb_images = [np.full((8, 6, 3), 50, dtype=np.uint8)]
    rgb_times = np.asarray([0.0], dtype=np.float64)
    depth_images = [np.full((8, 6, 3), 25, dtype=np.uint8)]
    depth_times = np.asarray([0.0], dtype=np.float64)
    out = tmp_path / "episode_rgb_depth.mp4"
    plot.export_episode_vision_video(
        rgb_images,
        rgb_times,
        depth_images,
        depth_times,
        str(out),
        fps=15.0,
    )
    assert out.is_file()
    assert out.stat().st_size > 0
