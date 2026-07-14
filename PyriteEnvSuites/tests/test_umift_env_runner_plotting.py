"""Unit tests for UMIFT env runner plotting helpers."""

import importlib
import sys
import types


class DummyCanvas:
    def __init__(self):
        self.calls = []

    def draw(self):
        self.calls.append("draw")

    def flush_events(self):
        self.calls.append("flush_events")


class DummyFigure:
    def __init__(self):
        self.canvas = DummyCanvas()


def test_flush_matplotlib_figure_draws_flushes_and_pauses(monkeypatch):
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
    pauses = []
    monkeypatch.setattr(runner.plt, "pause", pauses.append)

    fig = DummyFigure()
    runner.flush_matplotlib_figure(fig, pause_s=0.25)

    assert fig.canvas.calls == ["draw", "flush_events"]
    assert pauses == [0.25]
