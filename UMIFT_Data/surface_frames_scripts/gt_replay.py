"""Ground-truth label chunks for the runner, with no robot and no policy call.

The arrays are the ones ``replay_gt.pushed_action_chunks`` feeds the offline
ensembler. A checkpoint normalizer, when given, is applied as
normalize then unnormalize, the same call ``predict_action`` makes.
"""

from __future__ import annotations

import numpy as np

import replay_gt


def _cpu_numpy(value):
    """Copy a normalizer tensor to host memory before numpy sees it.

    A checkpoint normalizer lives on CUDA. ``np.asarray`` on that tensor
    raises TypeError. Call this on every tensor the replay loader reads.
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _normalizer_width(normalizer) -> int | None:
    params = getattr(normalizer, "params_dict", None)
    if params is None:
        return None
    scale = params["scale"] if "scale" in params else params["action"]["scale"]
    return int(_cpu_numpy(scale).reshape(-1).shape[0])


def apply_normalizer(normalizer, action: np.ndarray) -> np.ndarray:
    """Round-trip one chunk through the checkpoint action normalizer."""
    values = np.asarray(action, dtype=np.float32)
    out = normalizer.unnormalize(normalizer.normalize(values))
    return _cpu_numpy(out).astype(np.float64, copy=False)


def demo_tcp_disp(robot_t, rgb_t, position: np.ndarray) -> np.ndarray:
    """Displacement of the demo TCP from each robot sample to the next control step.

    Same array the sim gate stores on ``tcp_disp`` and passes as ``_tcp_step``.
    """
    from fixb_labels import interp_rows, next_control_times

    position = np.asarray(position, dtype=np.float64).reshape(-1, 3)
    arrived = next_control_times(
        np.asarray(robot_t, dtype=np.float64).reshape(-1),
        np.asarray(rgb_t, dtype=np.float64).reshape(-1),
    )
    return interp_rows(arrived, np.asarray(robot_t, dtype=np.float64).reshape(-1), position) - position


class GtReplay:
    """One episode of label chunks, in the order the runner pushes them."""

    def __init__(self, arm: str, episode: int, exec_h: int, normalizer=None):
        actions, windows = replay_gt.pushed_action_chunks(arm, int(episode), int(exec_h))
        if normalizer is not None and actions:
            width = _normalizer_width(normalizer)
            label_width = int(actions[0].shape[-1])
            if width == label_width:
                actions = [apply_normalizer(normalizer, action) for action in actions]
            else:
                print(
                    f"[gt-replay] checkpoint normalizer is {width}-d, "
                    f"labels are {label_width}-d; using the replay normalizer only"
                )
        self.arm = arm
        self.episode = int(episode)
        self.exec_h = int(exec_h)
        self.actions = actions
        self.windows = windows
        self.k = 0
        self.pos_mm: list[float] = []
        self.rot_deg: list[float] = []
        zarr_path = replay_gt.R4VT_ZARR if arm == "r4vt" else replay_gt.R1REL_ZARR
        group = replay_gt._open_episode(zarr_path, self.episode)
        self.demo = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
        self.start_pose7 = self.demo[0].copy()
        robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
        self.tcp_disp = demo_tcp_disp(robot_t, rgb_t, self.demo[:, :3])

    def reset(self) -> None:
        self.k = 0
        self.pos_mm.clear()
        self.rot_deg.clear()

    def remaining(self) -> bool:
        return self.k < len(self.actions)

    def current(self):
        return self.actions[self.k], self.windows[self.k]

    def tcp_step(self, n: int) -> np.ndarray:
        """Demo TCP displacement for the next n executed steps, in metres."""
        taken = np.asarray(self.windows[self.k][: int(n)], dtype=np.int64)
        return np.asarray(self.tcp_disp, dtype=np.float64)[taken]

    def advance(self) -> None:
        self.k += 1

    def start_pose_base(self, T_fixture_from_base: np.ndarray) -> np.ndarray:
        from PyriteEnvSuites.env_runners.dataset_start_pose import pose7_change_frame

        return pose7_change_frame(self.start_pose7, np.linalg.inv(T_fixture_from_base))

    def note_tcp(self, measured_pose7: np.ndarray) -> None:
        """Fixture-frame measured TCP against the demo TCP at this window's start."""
        head = int(self.windows[self.k][0])
        demo = self.demo[head]
        meas = np.asarray(measured_pose7, dtype=np.float64).reshape(7)
        self.pos_mm.append(float(np.linalg.norm(meas[:3] - demo[:3]) * 1e3))
        self.rot_deg.append(
            float(
                replay_gt.geodesic_deg(
                    replay_gt._pose_R(meas), replay_gt._pose_R(demo)
                ).reshape(-1)[0]
            )
        )

    def error_summary(self) -> dict:
        pos = np.asarray(self.pos_mm, dtype=np.float64)
        rot = np.asarray(self.rot_deg, dtype=np.float64)
        if pos.size == 0:
            return {"n": 0}
        return {
            "n": int(pos.size),
            "pos_median_mm": float(np.median(pos)),
            "pos_p95_mm": float(np.percentile(pos, 95)),
            "rot_median_deg": float(np.median(rot)),
            "rot_p95_deg": float(np.percentile(rot, 95)),
        }
