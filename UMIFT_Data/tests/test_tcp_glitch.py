"""TCP glitch detection: fast steps and short reversals, not slow strokes."""

import numpy as np
from scipy.spatial.transform import Rotation

from fixb_labels import Rp_to_pose7
from fixb_tcp_glitch import SHORT_MAX_S, can_interpolate, clean_pose7, glitch_mask, glitch_spans


def _pose(points: np.ndarray) -> np.ndarray:
    R = np.repeat(np.eye(3)[None], len(points), axis=0)
    return Rp_to_pose7(R, points)


def test_fast_step_is_a_glitch_and_a_slow_stroke_is_not():
    t = np.arange(6) * 0.005
    slow = np.zeros((6, 3))
    slow[:, 0] = np.arange(6) * 0.001  # 0.2 m/s
    assert not np.any(glitch_mask(slow, t))
    fast = slow.copy()
    fast[3, 0] += 0.010  # 2 m/s into this sample
    mask = glitch_mask(fast, t)
    assert mask[3]
    assert not mask[2]


def test_fast_reversal_fills_the_samples_between_the_legs():
    t = np.arange(8) * 0.005
    p = np.zeros((8, 3))
    p[1, 0] = 0.004  # 0.8 m/s
    p[2, 0] = 0.004
    p[3, 0] = 0.000  # back within 15 ms
    mask = glitch_mask(p, t)
    assert mask[1] and mask[3]
    assert mask[2]
    spans = glitch_spans(mask, t)
    assert len(spans) == 1
    assert spans[0]["kind"] == "short"


def test_short_glitch_is_interpolated_and_a_long_one_is_left():
    t = np.arange(0.0, 0.5, 0.005)
    p = np.zeros((len(t), 3))
    p[:, 0] = np.linspace(0.0, 0.02, len(t))
    spike = p.copy()
    spike[10, 0] += 0.02
    pose = _pose(spike)
    spans = glitch_spans(glitch_mask(spike, t), t)
    cleaned = clean_pose7(pose, t, spans)
    assert abs(cleaned[10, 0] - p[10, 0]) < 1e-6
    assert np.allclose(cleaned[0], pose[0])
    long = p.copy()
    n_long = int(SHORT_MAX_S / 0.005) + 8
    long[20 : 20 + n_long, 0] += 0.02 * (np.arange(n_long) % 2)
    long_spans = glitch_spans(glitch_mask(long, t), t)
    assert any(span["kind"] == "long" for span in long_spans)
    kept = clean_pose7(_pose(long), t, long_spans)
    assert abs(kept[21, 0] - long[21, 0]) < 1e-9


def test_one_way_jump_is_not_interpolated():
    t = np.arange(8) * 0.005
    p = np.zeros((8, 3))
    p[3:, 0] = 0.02  # 4 m/s step that does not come back
    spans = glitch_spans(glitch_mask(p, t), t)
    assert spans
    assert not can_interpolate(p, t, spans[0])
    cleaned = clean_pose7(_pose(p), t, [dict(spans[0], kind="long")])
    assert abs(cleaned[4, 0] - 0.02) < 1e-9


def test_orientation_is_slerped_across_a_short_glitch():
    t = np.arange(5) * 0.01
    p = np.zeros((5, 3))
    R = np.repeat(np.eye(3)[None], 5, axis=0)
    R[2] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    pose = Rp_to_pose7(R, p)
    p[2, 0] = 0.02
    pose[2, :3] = p[2]
    spans = glitch_spans(glitch_mask(pose[:, :3], t), t)
    cleaned = clean_pose7(pose, t, spans)
    assert abs(cleaned[2, 0]) < 1e-6
    yaw = Rotation.from_quat(cleaned[2, [4, 5, 6, 3]]).as_euler("xyz")[2]
    assert abs(yaw) < 1e-6
