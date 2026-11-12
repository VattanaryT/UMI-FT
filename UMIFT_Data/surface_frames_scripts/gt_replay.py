"""Ground-truth label chunks for the runner, with no robot and no policy call.

The arrays are the ones ``replay_gt.pushed_action_chunks`` feeds the offline
ensembler. A checkpoint normalizer, when given, is applied as
normalize then unnormalize, the same call ``predict_action`` makes.
"""

from __future__ import annotations

import numpy as np

import replay_gt


def apply_normalizer(normalizer, action: np.ndarray) -> np.ndarray:
    """Round-trip one chunk through the checkpoint action normalizer."""
    values = np.asarray(action, dtype=np.float32)
    out = normalizer.unnormalize(normalizer.normalize(values))
    if hasattr(out, "detach"):
        out = out.detach().cpu().numpy()
    return np.asarray(out, dtype=np.float64)


class GtReplay:
    """One episode of label chunks, in the order the runner pushes them."""

    def __init__(self, arm: str, episode: int, exec_h: int, normalizer=None):
        actions, windows = replay_gt.pushed_action_chunks(arm, int(episode), int(exec_h))
        if normalizer is not None:
            actions = [apply_normalizer(normalizer, action) for action in actions]
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

    def reset(self) -> None:
        self.k = 0
        self.pos_mm.clear()
        self.rot_deg.clear()

    def remaining(self) -> bool:
        return self.k < len(self.actions)

    def current(self):
        return self.actions[self.k], self.windows[self.k]

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
