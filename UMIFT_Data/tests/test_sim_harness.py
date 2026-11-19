"""Fake arm and control-log shape. The episode runs live in sim_harness.py."""

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))

import sim_harness as sh  # noqa: E402


def test_fake_arm_is_the_last_command_with_optional_lag():
    start = np.arange(7, dtype=np.float64)
    arm = sh.FakeArm(start, lag=0)
    assert np.allclose(arm.measure(), start)
    arm.command(start + 10)
    assert np.allclose(arm.measure(), start + 10)

    lagged = sh.FakeArm(np.zeros(7), lag=1)
    lagged.command(np.ones(7))
    assert np.allclose(lagged.measure(), 0)
    lagged.command(np.full(7, 2.0))
    assert np.allclose(lagged.measure(), 1)


def test_control_log_is_readable_by_plot_rollout_quick(tmp_path):
    import zarr
    from plot_rollout_quick import series

    store = zarr.DirectoryStore(str(tmp_path / "ep"))
    log = zarr.open(store, mode="w")
    decoded = {
        "x_d": np.zeros((2, 3)),
        "x_s": np.ones((2, 3)),
        "R_d": np.repeat(np.eye(3)[None], 2, axis=0),
        "R_field": np.repeat(np.eye(3)[None], 2, axis=0),
        "n_out": np.tile([0.0, -1.0, 0.0], (2, 1)),
        "delta_n": np.array([-0.005, -0.004]),
        "v_par": np.ones((2, 2)),
        "stiffness": np.zeros(2),
    }
    sh.write_horizon(log, 0, 1.5, 1.0 / 15.0, decoded, np.arange(6))
    data = series(zarr.open(store, mode="r"))
    assert data["t"].shape == (1,)
    assert data["dn_mm"].shape == (1,)
    assert data["v1"].shape == (1,)
    assert data["foot"].shape == (1, 3)
    assert np.allclose(data["foot"][0], 1)


def test_decode_comes_from_the_smooth_frame_tree():
    sh.boot()
    import mold_aware_runtime as runtime

    assert "UMI-FT-smooth_frame" in Path(runtime.__file__).resolve().parts
    assert sh.runner_decode is runtime.MoldAwareRuntime.decode_chunk
