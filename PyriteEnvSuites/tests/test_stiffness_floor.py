"""Press-axis stiffness floor: off by default, and the control-log trace."""

import numpy as np
import zarr

from PyriteEnvSuites.env_runners.stiffness_floor import (
    apply_press_stiffness_floor,
    read_press_stiffness_floor,
    write_stiffness_trace,
)


def test_floor_is_off_when_unset(monkeypatch):
    monkeypatch.delenv("UMIFT_STIFFNESS_FLOOR", raising=False)
    assert read_press_stiffness_floor() is None
    raw, floored = apply_press_stiffness_floor([200.0, 3000.0], None)
    assert np.array_equal(raw, [200.0, 3000.0])
    assert np.array_equal(floored, raw)
    assert raw is not floored


def test_floor_lifts_the_press_axis_and_does_not_lower_it(monkeypatch):
    monkeypatch.setenv("UMIFT_STIFFNESS_FLOOR", "3000")
    assert read_press_stiffness_floor() == 3000.0
    raw, floored = apply_press_stiffness_floor(
        [200.0, 220.0, 959.0, 3000.0, 5000.0], 3000.0
    )
    assert np.array_equal(raw, [200.0, 220.0, 959.0, 3000.0, 5000.0])
    assert np.array_equal(floored, [3000.0, 3000.0, 3000.0, 3000.0, 5000.0])


def test_floor_rejects_a_bad_value():
    try:
        read_press_stiffness_floor({"UMIFT_STIFFNESS_FLOOR": "soft"})
    except ValueError as exc:
        assert "soft" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_trace_logs_raw_and_floored_per_step():
    group = zarr.group()
    raw, floored = apply_press_stiffness_floor([200.0, 2718.0, 220.0], 3000.0)
    write_stiffness_trace(group, 0, raw, floored)
    assert np.array_equal(group["stiffness_raw_0"][:], raw)
    assert np.array_equal(group["stiffness_floored_0"][:], [3000.0, 3000.0, 3000.0])
