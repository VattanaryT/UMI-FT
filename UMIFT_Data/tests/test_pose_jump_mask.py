import numpy as np

from fixb_pose_jump import flag_from_speeds, keep_weight


def test_flag_marks_the_arrival_sample():
    speed = np.array([0.1, 2.0, 0.2])
    ang = np.array([10.0, 10.0, 10.0])
    flag = flag_from_speeds(speed, ang)
    assert flag.tolist() == [False, False, True, False]


def test_keep_weight_zeros_action_steps_after_the_jump():
    flag = np.zeros(10, dtype=bool)
    flag[4] = True
    action_idx = np.arange(6)
    keep = keep_weight(flag, action_idx, 6)
    assert keep.tolist() == [1, 1, 1, 1, 0, 0]


def test_step_zero_stays_even_if_the_chunk_starts_on_a_jump():
    flag = np.zeros(8, dtype=bool)
    flag[2] = True
    keep = keep_weight(flag, np.array([2, 3, 4, 5]), 4)
    assert keep.tolist() == [1, 0, 0, 0]
