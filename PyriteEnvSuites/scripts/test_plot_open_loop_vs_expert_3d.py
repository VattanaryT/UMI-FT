import importlib.util
import json
import pathlib
import tempfile

import numpy as np
import zarr


PLOT_SCRIPT = (
    pathlib.Path(__file__).resolve().parent / "plot_open_loop_vs_expert_3d.py"
)
SPEC = importlib.util.spec_from_file_location(
    "plot_open_loop_vs_expert_3d", PLOT_SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_resolve_episode_names():
    assert MODULE.resolve_episode_names("44") == ("episode_44", "episode_44_policy")
    assert MODULE.resolve_episode_names("episode_44") == (
        "episode_44",
        "episode_44_policy",
    )
    assert MODULE.resolve_episode_names("episode_44_policy") == (
        "episode_44",
        "episode_44_policy",
    )


def test_build_open_loop_reveal_times_spans_to_next_query():
    """Executed steps are stretched across [query_t, next_query_t)."""
    queries = [
        {
            "expert_robot_time_s": 1.0,
            "open_loop_start_idx": 0,
            "open_loop_end_idx": 2,
            "action_offsets_s": [0.0, 0.03],  # physical offsets ignored for scrub
        },
        {
            "expert_robot_time_s": 2.0,
            "open_loop_start_idx": 2,
            "open_loop_end_idx": 4,
            "action_offsets_s": [0.0, 0.03],
        },
    ]
    reveal = MODULE.build_open_loop_reveal_times(queries, n_policy=4)
    # q0: t in [1.0, 2.0) with 2 steps -> 1.0, 1.5
    # q1: last query reuses prior gap 1.0s -> [2.0, 3.0) -> 2.0, 2.5
    assert np.allclose(reveal, [1.0, 1.5, 2.0, 2.5])


def test_build_open_loop_reveal_times_single_query_uses_offsets_fallback():
    queries = [
        {
            "expert_robot_time_s": 1.0,
            "open_loop_start_idx": 0,
            "open_loop_end_idx": 2,
            "action_offsets_s": [0.0, 0.03],
        }
    ]
    reveal = MODULE.build_open_loop_reveal_times(queries, n_policy=2)
    # t1 = 1.0 + 0.03 + 0.03 = 1.06; steps at 1.0 and 1.03
    assert np.allclose(reveal, [1.0, 1.03])


def test_policy_prefix_and_visible_queries():
    reveal = np.array([1.0, 1.03, 2.0, 2.03])
    assert MODULE.policy_prefix_len(reveal, 0.5) == 0
    assert MODULE.policy_prefix_len(reveal, 1.0) == 1
    assert MODULE.policy_prefix_len(reveal, 1.03) == 2
    assert MODULE.policy_prefix_len(reveal, 2.5) == 4

    queries = [
        {"expert_robot_time_s": 1.0, "query_idx": 0},
        {"expert_robot_time_s": 2.0, "query_idx": 1},
    ]
    assert [q["query_idx"] for q in MODULE.visible_queries(queries, 1.5)] == [0]
    assert [q["query_idx"] for q in MODULE.visible_queries(queries, 2.0)] == [0, 1]


def test_action_arrow_xyz_from_expert_pose():
    queries = [
        {
            "expert_robot_time_s": 1.0,
            "expert_pose_index": 1,
            "base_pose7": [9.0, 9.0, 9.0, 1.0, 0.0, 0.0, 0.0],
            "open_loop_start_idx": 0,
            "open_loop_end_idx": 2,
            "action_offsets_s": [0.0, 0.1],
            "policy_pose_command_raw": [
                [0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
        }
    ]
    expert = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    policy = np.zeros((2, 7))  # ignored when policy_pose_key provided
    reveal = np.array([1.0, 1.5])
    xs, ys, zs = MODULE.action_arrow_xyz(
        queries,
        expert,
        policy,
        t=1.05,
        reveal_times=reveal,
        policy_pose_key="policy_pose_command_raw",
    )
    assert xs == [0.05, 0.1, None]
    assert ys == [0.0, 0.0, None]
    assert zs == [0.0, 0.0, None]


def test_policy_poses_from_query_records():
    queries = [
        {
            "open_loop_start_idx": 0,
            "open_loop_end_idx": 2,
            "policy_pose_command_raw": [
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
        },
        {
            "open_loop_start_idx": 2,
            "open_loop_end_idx": 3,
            "policy_pose_command_raw": [
                [3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
        },
    ]
    out = MODULE.policy_poses_from_query_records(
        queries, n_policy=3, key="policy_pose_command_raw"
    )
    assert out is not None
    assert np.allclose(out[:, 0], [1.0, 2.0, 3.0])


def test_load_episode_and_queries_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path = pathlib.Path(tmp) / "policy_open_loop_val.zarr"
        root = zarr.open(str(zarr_path), mode="w")
        data = root.create_group("data")
        ep = data.create_group("episode_3_policy")
        poses = np.zeros((4, 7), dtype=np.float64)
        poses[:, 3] = 1.0
        poses[:, 0] = np.arange(4) * 0.01
        ep.create_dataset("ts_pose_command_0", data=poses)
        ep.create_dataset("ts_pose_virtual_target_0", data=poses + 0.001)
        ep.create_dataset(
            "robot_time_stamps_0",
            data=np.linspace(0.0, 0.3, 4).reshape(-1, 1),
        )

        loaded = MODULE.load_episode_poses(str(zarr_path), "episode_3_policy")
        assert loaded["command"].shape == (4, 7)

        queries_path = pathlib.Path(tmp) / "policy_open_loop_val_queries.json"
        payload = {
            "episodes": {
                "episode_3_policy": [
                    {
                        "query_idx": 0,
                        "expert_robot_time_s": 0.0,
                        "open_loop_start_idx": 0,
                        "open_loop_end_idx": 2,
                        "action_offsets_s": [0.0, 0.1],
                        "base_pose7": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    }
                ]
            }
        }
        queries_path.write_text(json.dumps(payload))
        records = MODULE.load_query_records(str(queries_path), "episode_3_policy")
        assert len(records) == 1
        assert records[0]["open_loop_end_idx"] == 2
