"""dt, action blend, and r4vt abort thresholds. No robot and no mesh."""

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))

from mold_aware_runtime import (  # noqa: E402
    FixBActionEnsembler,
    MoldAwareRuntime,
    SafetyLimits,
    fixb_decode_dt,
    r4_safety_limits,
)

RUNNER = (
    Path(__file__).resolve().parents[2]
    / "PyriteEnvSuites"
    / "env_runners"
    / "umift_env_runner.py"
)
EYE6 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def _runtime(limits, run_type="r4_world_fixb_tilt"):
    return MoldAwareRuntime(
        field=None,
        R_off=np.eye(3),
        run_type=run_type,
        limits=limits,
        dry_run=False,
    )


def test_fixb_decode_dt_is_r4vt_only():
    one = fixb_decode_dt(4, down_sample_steps=1, raw_time_step_s=1.0 / 15.0)
    two = fixb_decode_dt(2, down_sample_steps=2, raw_time_step_s=1.0 / 15.0)
    assert np.allclose(one, 1.0 / 15.0)
    assert np.allclose(two, 2.0 / 15.0)
    text = RUNNER.read_text()
    assert "enable_action_ensemble" not in text
    r4 = text.index("if r4vt:\n                    exec_now")
    other = text.index("else:\n                    dt = np.full(", r4)
    assert "fixb_decode_dt(" in text[r4:other]
    assert "slow_down_factor" not in text[r4:other]
    assert "slow_down_factor" in text[other : other + 350]
    assert "if action_ensemblers is not None and not r4vt:" in text
    assert "blend_pose_chunk(" in text
    assert "make_pose_ensembler" in text
    start = text.index("sparse_action_timesteps_s = (")
    assert "slow_down_factor" in text[start : start + 250]


def test_action_blend_averages_overlapping_velocity():
    first = np.zeros((4, 27))
    second = np.zeros((4, 27))
    first[:, 0] = 1.0
    second[:, 0] = 3.0
    first[:, 2] = -0.10
    second[:, 2] = -0.02
    first[:, 21:27] = EYE6
    second[:, 21:27] = EYE6
    ensembler = FixBActionEnsembler(decay_m=0.0)
    ensembler.push(0, first)
    ensembler.push(0, second)
    blended = ensembler.pop(1)
    assert np.isclose(blended[0, 0], 2.0)
    assert np.isclose(blended[0, 2], -0.06)


def test_r4_drops_delta_n_floor_and_aborts_on_measured_tcp():
    action = np.zeros((2, 27))
    action[:, 2] = -0.10
    action[:, 21:27] = EYE6
    r4 = _runtime(r4_safety_limits(0.020, 90.0))
    _v, delta_n, _stiff, _eoat, _raw, clamp = r4.unpack_fixb_action(action)
    assert np.allclose(delta_n, -0.10)
    assert not np.any(clamp)
    kept = _runtime(SafetyLimits(), run_type="r2_world_fixb")
    _v, floored, _stiff, _eoat, _raw, clamp = kept.unpack_fixb_action(action)
    assert np.allclose(floored, -0.005)
    assert np.all(clamp)
    r4.note_tcp_penetration(np.array([-0.019, -0.010]))
    assert "tcp_penetration" not in r4.last_hard_triggers
    r4.note_tcp_penetration(np.array([-0.021]))
    assert r4.last_hard_triggers == ["tcp_penetration"]


def test_wrench_holds_once_then_aborts():
    runtime = _runtime(r4_safety_limits())
    assert runtime.wrench_hold_update(89.0) == "ok"
    assert runtime.wrench_hold_update(90.0) == "hold"
    assert runtime.wrench_hold_update(91.0) == "abort"
    assert "wrench_abort" in runtime.last_hard_triggers
    runtime.reset()
    assert runtime.wrench_hold_update(95.0) == "hold"
    assert runtime.wrench_hold_update(10.0) == "ok"
    assert runtime.wrench_hold_update(95.0) == "hold"


def test_r1rel_replay_through_runner_stays_zero():
    import replay_gt as rg

    if not rg.R1REL_ZARR.exists():
        import pytest

        pytest.skip("r1rel zarr is not on this machine")
    for episode in (0, 9):
        report = rg.replay_episode("r1rel", episode)
        for row in report["horizons"].values():
            tcp = row["position"]["tcp_m"]
            ori = row["orientation_deg"]
            for key in (
                "decode_median",
                "decode_max",
                "ensemble_median",
                "ensemble_max",
            ):
                assert f"{tcp[key] * 1e3:.3f}" == "0.000"
                assert f"{ori[key]:.3f}" == "0.000"


def test_live_horizon_stays_6_and_both_arms_replay_it():
    from PyriteUtility.planning_control.temporal_ensemble import (
        resolve_execution_horizon,
    )
    import replay_gt as rg

    h = resolve_execution_horizon(action_horizon=rg.ACTION_HORIZON)
    assert h == 6
    assert resolve_execution_horizon(6, 64, raise_64_to_32=True) == 32
    text = RUNNER.read_text()
    assert "raise_exec_horizon_64_to_32" in text
    assert "resolve_execution_horizon(" in text
    assert 'log.attrs["ensemble_mode"] = "action_blend" if r4vt else "pose"' in text
    assert 'log.attrs["decode_dt_s"]' in text
    ensemble = (RUNNER.parents[2] / "PyriteUtility/PyriteUtility/planning_control/temporal_ensemble.py").read_text()
    body = ensemble[ensemble.index("def ensemble_pose_chunks"):]
    assert "blend_pose_chunk(" in body
    assert "blended = blend_pose_chunk(" in text
    if not rg.R1REL_ZARR.exists() or not rg.R4VT_ZARR.exists():
        import pytest

        pytest.skip("replay zarr is not on this machine")
    for arm in ("r1rel", "r4vt"):
        for episode in (0, 9):
            row = rg.replay_episode(arm, episode, exec_horizons=(h,))["horizons"][str(h)]
            assert row["execution_horizon"] == 6
            if arm == "r1rel":
                tcp = row["position"]["tcp_m"]
                ori = row["orientation_deg"]
                assert f"{tcp['ensemble_max'] * 1e3:.3f}" == "0.000"
                assert f"{ori['ensemble_max']:.3f}" == "0.000"
            else:
                depth = row["position"]["depth_m"]
                foot = row["position"]["foot_inplane_m"]
                ori = row["orientation_deg"]
                assert f"{depth['decode_max'] * 1e3:.3f}" == "0.000"
                assert foot["decode_median"] < 0.001
