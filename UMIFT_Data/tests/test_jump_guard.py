"""r4vt commanded-target jump guard. No robot and no dataset."""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "surface_frames_scripts"))

from jump_guard import TargetJumpGuard


def _rots(degrees):
    return np.stack(
        [Rotation.from_euler("z", deg, degrees=True).as_matrix() for deg in degrees]
    )


def test_jump_guard_holds_three_then_aborts():
    guard = TargetJumpGuard(max_step_mm=10.0, max_step_deg=10.0, max_holds=3)
    x = np.zeros((5, 3))
    x[:, 0] = np.arange(5) * 0.001
    out_x, out_R, _field, abort = guard.apply(x, _rots([0, 1, 2, 3, 4]))
    assert not abort
    assert guard.holds == []
    assert out_x.shape[0] == 5

    guard.reset()
    x = np.zeros((2, 3))
    x[1, 0] = 0.05
    out_x, _out_R, _field, abort = guard.apply(x, _rots([0, 0]))
    assert not abort
    assert len(guard.holds) == 1
    assert guard.holds[0]["abort"] is False
    assert np.allclose(out_x[1], out_x[0])

    guard.reset()
    out_x, out_R, _field, abort = guard.apply(
        np.zeros((5, 3)), _rots([0, 90, 90, 90, 90])
    )
    assert abort
    assert [row["abort"] for row in guard.holds] == [False, False, False, True]
    assert out_x.shape[0] == 4
    assert np.allclose(out_R[1], out_R[0])
    assert np.allclose(out_R[3], out_R[0])

    guard.reset()
    x = np.zeros((4, 3))
    x[1, 0] = 0.05
    x[2, 0] = 0.001
    x[3, 0] = 0.05
    _out_x, _out_R, _field, abort = guard.apply(x, _rots([0, 0, 0, 0]))
    assert not abort
    assert [row["run"] for row in guard.holds] == [1, 1]
