"""In-process stand-in for ManipServerLayupEnv.

Records every schedule_controls call. Does not open RealSense, RTDE, or the
C++ server. In command mode the position chases the schedule at 32 mm/s, the
rate measured on the 2026-11-24 ep9 bench once the error was past 20 mm
(median 31.5, p90 34, and flat from 20 mm to 140 mm). Orientation is copied
from the command, which the arm matched. follow='demo' plays the recording.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

def _press_stiffness(blocks: np.ndarray, step: int) -> float:
    """Smallest translational eigenvalue: the compliance axis."""
    col = 6 * int(step)
    if blocks.shape[1] < col + 3:
        return 3000.0
    trans = blocks[:3, col : col + 3]
    try:
        stiff = float(np.min(np.linalg.eigvalsh(0.5 * (trans + trans.T))))
    except np.linalg.LinAlgError:
        return 3000.0
    if not np.isfinite(stiff) or stiff < 0.0:
        return 0.0
    return stiff


POSE_HZ = 500.0
RGB_HZ = 15.0
WRENCH_HZ = 500.0
# Saturated chase from episode_1795569140 vs the bench replay command.
CHASE_SPEED_MPS = 0.032
SPRING_CAP_N = 25.0


class _Server:
    def is_running(self) -> bool:
        return True

    def start_listening_key_events(self) -> None:
        return None

    def start_saving_data_for_a_new_episode(self, episode_name: str = "") -> None:
        return None

    def stop_saving_data(self) -> None:
        return None

    def stop_listening_key_events(self) -> None:
        return None

    def get_episode_folder(self) -> str:
        return str(MockLayupEnv.episode_folder)


class MockLayupEnv:
    """Class attributes are the episode the driver loaded before main()."""

    initial_pose = np.array([0.3, -0.2, 0.45, 1.0, 0.0, 0.0, 0.0])
    follow = "command"  # or "demo"
    chase = True
    project_pose: Optional[Callable] = None
    episode_folder = "/tmp/umift_mock_episode"
    demo_pose = None  # (N, 7) right_base
    demo_pose_t = None  # (N,) demo seconds
    demo_rgb = None
    demo_rgb_t = None
    demo_wrench = None  # (M, 6)
    demo_wrench_t = None
    demo_origin_t = 0.0
    slow_down = 2.0
    schedules: list = []
    active: Optional["MockLayupEnv"] = None

    def __init__(self, camera_res_hw=(224, 224), query_sizes=None, **_kwargs):
        height, width = int(camera_res_hw[0]), int(camera_res_hw[1])
        self.image_hw = (height, width)
        sparse = (query_sizes or {}).get("sparse", {})
        self.n_rgb = max(int(sparse.get("rgb") or 8), 2)
        self.n_depth = max(int(sparse.get("depth") or 2), 2)
        self.n_pose = max(int(sparse.get("ts_pose_fb") or 200), 100)
        self.n_wrench = max(int(sparse.get("wrench") or 200), 100)
        self.id_list = [0]
        self.server = _Server()
        self.t = 0.0
        self.pose = np.asarray(type(self).initial_pose, dtype=np.float64).reshape(7).copy()
        self.wrench_now = np.zeros(6, dtype=np.float64)
        self.episode_armed = False
        self.episode_t0 = None
        self._hist_t = np.zeros(0)
        self._hist_pose = np.zeros((0, 7))
        self._rgb_t = np.zeros(0)
        self._wrench_t = np.zeros(0)
        self._fill_until(max(2.0, self.n_pose / POSE_HZ, self.n_wrench / WRENCH_HZ))
        self.schedules = type(self).schedules
        type(self).active = self
        print(
            f"[mock-env] pose buffer {self.n_pose} rgb {self.n_rgb} "
            f"follow={type(self).follow} no robot"
        )

    def reset(self) -> None:
        return None

    def cleanup(self) -> None:
        return None

    def calibrate_robot_wrench(self, NSamples: int = 100) -> None:
        print(f"[mock-env] calibrate_robot_wrench skipped ({NSamples} samples)")

    def start_saving_data_for_a_new_episode(self, episode_name: str = "") -> None:
        return None

    def stop_saving_data(self) -> None:
        return None

    def get_episode_folder(self) -> str:
        return str(type(self).episode_folder)

    def set_wrist_raw_rgb_sink(self, sink) -> None:
        return None

    def pause_rtde_control(self) -> bool:
        print("[mock-env] pause_rtde_control not sent")
        return True

    def resume_rtde_control(self) -> bool:
        print("[mock-env] resume_rtde_control not sent")
        return True

    @property
    def current_hardware_time_s(self) -> float:
        return float(self.t)

    def advance(self, dt: float) -> None:
        if dt <= 0.0:
            return
        self._fill_until(self.t + float(dt))

    def schedule_controls(
        self,
        pose7_cmd,
        eoat_cmd=None,
        timestamps=None,
        stiffness_matrices_6x6=None,
    ) -> bool:
        cmd = np.asarray(pose7_cmd, dtype=np.float64)
        if cmd.ndim == 1:
            cmd = cmd.reshape(7, 1)
        if cmd.shape[0] != 7 and cmd.shape[-1] == 7:
            cmd = cmd.T
        stamps = None if timestamps is None else np.asarray(timestamps, dtype=np.float64).reshape(-1) / 1000.0
        commanded = cmd.T.copy()
        force = np.zeros(len(commanded), dtype=np.float64)
        if (
            type(self).chase
            and type(self).follow == "command"
            and stamps is not None
            and stiffness_matrices_6x6 is not None
        ):
            measured, force = self._chase(commanded, stamps, stiffness_matrices_6x6)
        else:
            measured = commanded
            if type(self).project_pose is not None and stiffness_matrices_6x6 is not None:
                measured = np.stack([type(self).project_pose(row) for row in commanded])
        self.schedules.append(
            {
                "t0": None if stamps is None else float(stamps[0]),
                "t1": None if stamps is None else float(stamps[-1]),
                "pose": commanded.copy(),
                "measured": measured.copy(),
                "force_n": force.copy(),
                "stiffness": None
                if stiffness_matrices_6x6 is None
                else np.asarray(stiffness_matrices_6x6, dtype=np.float64).copy(),
                "episode": bool(self.episode_armed),
            }
        )
        if stiffness_matrices_6x6 is not None and not self.episode_armed:
            self.episode_armed = True
            self.episode_t0 = float(stamps[0]) if stamps is not None else float(self.t)
        self.pose = measured[-1].copy()
        if stamps is not None and len(stamps):
            if type(self).follow == "demo" and self.episode_armed:
                self._fill_until(float(stamps[-1]))
            else:
                self._fill_until(float(stamps[-1]), measured, stamps)
        return True

    def _chase(self, commanded, stamps, stiffness):
        """Move toward each command at CHASE_SPEED_MPS. Keep the orientation."""
        pose = self.pose.copy()
        measured = np.empty_like(commanded)
        force = np.zeros(len(commanded), dtype=np.float64)
        blocks = np.asarray(stiffness, dtype=np.float64).reshape(6, -1)
        for i in range(len(commanded)):
            # Use the schedule's own spacing. The mock clock is often already
            # past the stamp, which would give dt=0 and freeze the arm.
            if i == 0:
                dt = float(stamps[min(1, len(stamps) - 1)] - stamps[0])
                if dt <= 1e-4:
                    dt = 0.133
            else:
                dt = max(0.0, float(stamps[i] - stamps[i - 1]))
            target = commanded[i]
            delta = target[:3] - pose[:3]
            dist = float(np.linalg.norm(delta))
            allow = CHASE_SPEED_MPS * dt
            if dist > 1e-12:
                pose[:3] = pose[:3] + delta * (min(dist, allow) / dist)
            pose[3:] = target[3:]
            chased = pose.copy()
            if type(self).project_pose is not None:
                pose = np.asarray(type(self).project_pose(pose), dtype=np.float64).reshape(7)
            depth = float(np.linalg.norm(pose[:3] - chased[:3]))
            k = _press_stiffness(blocks, i)
            force[i] = min(SPRING_CAP_N, k * depth)
            measured[i] = pose
        if len(force):
            # Log uses the newest wrench sample.
            direction = pose[:3] - chased[:3]
            nrm = float(np.linalg.norm(direction))
            wrench = np.zeros(6, dtype=np.float64)
            if nrm > 1e-9:
                wrench[:3] = direction / nrm * force[-1]
            self.wrench_now = wrench
        return measured, force

    def _pose_on_schedule(self, hardware_t, measured, stamps):
        if hardware_t <= float(stamps[0]):
            return measured[0]
        if hardware_t >= float(stamps[-1]):
            return measured[-1]
        i = int(np.searchsorted(stamps, hardware_t) - 1)
        i = min(max(i, 0), len(stamps) - 2)
        span = float(stamps[i + 1] - stamps[i])
        alpha = 0.0 if span <= 1e-9 else (hardware_t - float(stamps[i])) / span
        return (1.0 - alpha) * measured[i] + alpha * measured[i + 1]

    def _fill_until(self, t_end: float, measured=None, stamps=None) -> None:
        """Append samples at exact k/500 s. Later windows must repeat those stamps."""
        t_end = float(t_end)
        step = 1.0 / POSE_HZ
        if self._hist_t.size == 0:
            last = -1
        else:
            last = int(round(float(self._hist_t[-1]) / step))
        end_index = int(np.floor(t_end / step + 1e-9))
        if end_index > last:
            idx = np.arange(last + 1, end_index + 1)
            times = idx * step
            rows = np.zeros((len(idx), 7), dtype=np.float64)
            for i, hardware_t in enumerate(times):
                if measured is not None and stamps is not None:
                    rows[i] = self._pose_on_schedule(float(hardware_t), measured, stamps)
                else:
                    rows[i] = self._pose_at(float(hardware_t))
            if self._hist_t.size == 0:
                self._hist_t = times
                self._hist_pose = rows
            else:
                self._hist_t = np.concatenate([self._hist_t, times])
                self._hist_pose = np.concatenate([self._hist_pose, rows])
        self.t = max(self.t, t_end)

    def _pose_at(self, hardware_t: float) -> np.ndarray:
        if (
            type(self).follow != "demo"
            or not self.episode_armed
            or self.episode_t0 is None
            or hardware_t < self.episode_t0
            or type(self).demo_pose is None
        ):
            return self.pose
        demo_t = float(type(self).demo_origin_t) + (hardware_t - self.episode_t0) / float(
            type(self).slow_down
        )
        times = type(self).demo_pose_t
        idx = int(np.clip(np.searchsorted(times, demo_t), 0, len(times) - 1))
        prev = max(idx - 1, 0)
        if abs(float(times[prev]) - demo_t) < abs(float(times[idx]) - demo_t):
            idx = prev
        return np.asarray(type(self).demo_pose[idx], dtype=np.float64)

    def _demo_time(self) -> float:
        if self.episode_t0 is None:
            return float(type(self).demo_origin_t)
        return float(type(self).demo_origin_t) + (self.t - self.episode_t0) / float(
            type(self).slow_down
        )

    def _tail(self, values, times, n, hz, t_end):
        if values is None:
            return None, None
        duration = (n - 1) / float(hz)
        grid = t_end - duration + np.arange(n) / float(hz)
        idx = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
        prev = np.clip(idx - 1, 0, len(times) - 1)
        use_prev = np.abs(times[prev] - grid) < np.abs(times[idx] - grid)
        idx = np.where(use_prev, prev, idx)
        return np.asarray(values)[idx], grid

    def _exact_tail(self, n: int):
        if self._hist_t.size < n:
            self._fill_until(self.t + (n - self._hist_t.size) / POSE_HZ)
        pose = self._hist_pose[-n:]
        pose_t = self._hist_t[-n:]
        return pose, pose_t

    def get_observation_from_buffer(self):
        t_end = float(self._hist_t[-1]) if self._hist_t.size else 0.0
        pose, pose_t = self._exact_tail(self.n_pose)
        if type(self).demo_rgb is not None:
            demo_t = self._demo_time() if self.episode_armed else float(type(self).demo_origin_t)
            rgb_duration = (self.n_rgb - 1) / RGB_HZ
            rgb_demo = demo_t - rgb_duration / type(self).slow_down + (
                np.arange(self.n_rgb) / RGB_HZ
            ) / type(self).slow_down
            src_t = type(self).demo_rgb_t
            ridx = np.clip(np.searchsorted(src_t, rgb_demo), 0, len(src_t) - 1)
            rgb = np.asarray(type(self).demo_rgb)[ridx]
        else:
            rgb = np.zeros((self.n_rgb, self.image_hw[0], self.image_hw[1], 3), np.uint8)
        # Same hardware clock as the pose tail, so image alignment stays under 50 ms.
        rgb_t = pose_t[-1] - (self.n_rgb - 1) / RGB_HZ + np.arange(self.n_rgb) / RGB_HZ
        if type(self).demo_wrench is not None and type(self).follow == "demo" and self.episode_armed:
            demo_t = self._demo_time()
            duration = (self.n_wrench - 1) / WRENCH_HZ
            grid = demo_t - duration / type(self).slow_down + (
                np.arange(self.n_wrench) / WRENCH_HZ
            ) / type(self).slow_down
            src_t = type(self).demo_wrench_t
            widx = np.clip(np.searchsorted(src_t, grid), 0, len(src_t) - 1)
            wrench = np.asarray(type(self).demo_wrench)[widx]
        else:
            wrench = np.repeat(self.wrench_now.reshape(1, 6), self.n_wrench, axis=0)
        wrench_t = pose_t[-self.n_wrench :]
        if len(wrench_t) < self.n_wrench:
            wrench_t = pose_t[-1] - (self.n_wrench - 1) / WRENCH_HZ + np.arange(self.n_wrench) / WRENCH_HZ
        depth = np.zeros((self.n_depth, self.image_hw[0], self.image_hw[1], 3), np.float16)
        depth_t = pose_t[-1] - (self.n_depth - 1) / RGB_HZ + np.arange(self.n_depth) / RGB_HZ
        gripper = np.zeros((self.n_pose, 1), dtype=np.float64)
        return {
            "rgb_0": rgb,
            "rgb_time_stamps_0": rgb_t.astype(np.float64),
            "depth_0": depth,
            "depth_time_stamps_0": depth_t.astype(np.float64),
            "ts_pose_fb_0": pose.astype(np.float64),
            "robot_time_stamps_0": pose_t.astype(np.float64),
            "wrench_left_0": wrench.astype(np.float64),
            "wrench_right_0": np.zeros_like(wrench),
            "wrench_time_stamps_0": wrench_t.astype(np.float64),
            "gripper_0": gripper,
            "gripper_time_stamps_0": pose_t.astype(np.float64),
        }


def save_schedules(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in MockLayupEnv.schedules:
        pose = item["pose"]
        n = len(pose)
        rows.append(pose)
        item["_n"] = n
    if not rows:
        np.savez_compressed(path, pose=np.zeros((0, 7)))
        return
    pose = np.concatenate(rows, axis=0)
    measured = np.concatenate(
        [item.get("measured", item["pose"]) for item in MockLayupEnv.schedules],
        axis=0,
    )
    force = np.concatenate(
        [
            item.get("force_n", np.zeros(len(item["pose"])))
            for item in MockLayupEnv.schedules
        ]
    )
    episode = np.concatenate(
        [
            np.full(len(item["pose"]), item["episode"], dtype=bool)
            for item in MockLayupEnv.schedules
        ]
    )
    np.savez_compressed(path, pose=pose, measured=measured, episode=episode, force_n=force)
