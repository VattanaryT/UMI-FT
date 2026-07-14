#!/usr/bin/env python3
"""
Replay recorded virtual-target trajectories through the same stiffness scheduling
path used during UMIFT inference, but with actions coming from dataset episodes.
"""

import argparse
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import zarr

from PyriteUtility.spatial_math import spatial_utilities as su

REQUIRED_KEYS = (
    "ts_pose_command_0",
    "ts_pose_virtual_target_0",
    "stiffness_0",
    "robot_time_stamps_0",
)


def validate_episode_keys(episode_data: Dict[str, np.ndarray]) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in episode_data]
    if missing:
        raise ValueError(f"Episode is missing required keys: {missing}")


def compute_replay_timestamps_s(
    count: int, raw_time_step_s: float, slow_down_factor: float
) -> np.ndarray:
    return np.arange(count, dtype=np.float64) * raw_time_step_s * slow_down_factor


def _estimate_dt_from_robot_time_stamps(robot_time_stamps: np.ndarray) -> float:
    ts = np.asarray(robot_time_stamps, dtype=np.float64).reshape(-1)
    if ts.shape[0] < 2:
        return 0.0167
    dts = np.diff(ts)
    dts = dts[dts > 1e-6]
    if dts.shape[0] == 0:
        return 0.0167
    return float(np.median(dts))


def compute_stiffness_matrices(
    current_pose7: np.ndarray,
    ts_pose_command: np.ndarray,
    ts_pose_virtual_target: np.ndarray,
    ts_stiffness: np.ndarray,
    default_stiffness: float = 5000.0,
    default_stiffness_rot: float = 100.0,
    nominal_target_stiffness_override: float = -1.0,
) -> np.ndarray:
    """
    Reconstruct per-step 6x6 stiffness using the same compliance-direction
    logic as UMIFT inference runner.
    """
    if not (
        ts_pose_command.shape == ts_pose_virtual_target.shape
        and ts_pose_command.shape[1] == 7
    ):
        raise ValueError("Pose arrays must be (N, 7) and have matching shape.")
    if ts_stiffness.shape[0] != ts_pose_command.shape[0]:
        raise ValueError("stiffness length must match pose trajectory length.")

    n = ts_pose_command.shape[0]
    ts_stiffnesses = np.zeros((6, 6 * n), dtype=np.float64)
    se3_tw = su.SE3_inv(su.pose7_to_SE3(current_pose7))

    for i in range(n):
        se3_target = su.pose7_to_SE3(ts_pose_command[i])
        se3_virtual_target = su.pose7_to_SE3(ts_pose_virtual_target[i])

        se3_t_vt = se3_tw @ se3_virtual_target
        se3_t_target = se3_tw @ se3_target
        compliance_direction_tool = (se3_t_vt[:3, 3] - se3_t_target[:3, 3]).reshape(3)

        if np.linalg.norm(compliance_direction_tool) < 0.001:
            compliance_direction_tool = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        compliance_direction_tool = (
            compliance_direction_tool / np.linalg.norm(compliance_direction_tool)
        )

        x_axis = compliance_direction_tool
        y_axis = np.cross(x_axis, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        if np.linalg.norm(y_axis) < 1e-6:
            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)

        target_stiffness = float(ts_stiffness[i])
        m = np.diag([target_stiffness, default_stiffness, default_stiffness])
        s = np.array([x_axis, y_axis, z_axis]).T
        translational_stiffness = s @ m @ np.linalg.inv(s)

        stiffness_matrix_full = np.eye(6, dtype=np.float64) * default_stiffness_rot
        stiffness_matrix_full[:3, :3] = translational_stiffness
        if nominal_target_stiffness_override > 0:
            stiffness_matrix_full[:3, :3] = (
                np.eye(3, dtype=np.float64) * nominal_target_stiffness_override
            )

        ts_stiffnesses[:, 6 * i : 6 * i + 6] = stiffness_matrix_full

    return ts_stiffnesses


def _load_episode(dataset_path: str, episode: str) -> Dict[str, np.ndarray]:
    root = zarr.open(dataset_path, mode="r")
    if "data" not in root or episode not in root["data"]:
        raise ValueError(f"Episode {episode} not found in dataset {dataset_path}")
    ep = root["data"][episode]
    ep_data = {k: np.asarray(ep[k]) for k in ep.keys()}
    validate_episode_keys(ep_data)
    return ep_data


def _build_eoat_cmd_if_available(
    episode_data: Dict[str, np.ndarray], start: int, end: int, gripper_action_scale: float
):
    if "gripper_0" not in episode_data:
        return None
    gp = np.asarray(episode_data["gripper_0"][start:end]).reshape(-1, 1) * gripper_action_scale
    gp_force = np.zeros_like(gp)
    return np.concatenate([gp, gp_force], axis=1).T


def select_target_trajectory(
    ts_pose_command: np.ndarray,
    ts_pose_virtual_target: np.ndarray,
    test_nominal_target: bool,
) -> np.ndarray:
    return ts_pose_command if test_nominal_target else ts_pose_virtual_target


def align_trajectories_to_current_pose(
    ts_pose_command: np.ndarray,
    ts_pose_virtual_target: np.ndarray,
    current_pose7: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align dataset trajectories into the live robot frame by anchoring the first
    command pose to current robot pose, then applying the same rigid transform
    to command and virtual-target trajectories.
    """
    if ts_pose_command.shape[0] == 0:
        return ts_pose_command, ts_pose_virtual_target
    se3_current = su.pose7_to_SE3(current_pose7)
    se3_dataset_cmd0 = su.pose7_to_SE3(ts_pose_command[0])
    se3_world_dataset = se3_current @ su.SE3_inv(se3_dataset_cmd0)

    se3_cmd = su.pose7_to_SE3(ts_pose_command)
    se3_vt = su.pose7_to_SE3(ts_pose_virtual_target)
    aligned_cmd = su.SE3_to_pose7(se3_world_dataset @ se3_cmd)
    aligned_vt = su.SE3_to_pose7(se3_world_dataset @ se3_vt)
    return aligned_cmd, aligned_vt


class RobotPlaybackHardware:
    """Minimal manip_server wrapper for trajectory replay (no camera/iPhone)."""

    def __init__(
        self,
        hardware_config_path: str,
        compliant_dimensionality: int = 3,
        ready_timeout_s: float = 30.0,
    ) -> None:
        from hardware_interfaces.workcell.table_top_manip.python import (
            manip_server_pybind as ms,
        )

        self.server = ms.ManipServer()
        print(f"[replay] Loading hardware config: {hardware_config_path}")
        if not self.server.initialize(hardware_config_path):
            raise RuntimeError(
                f"Failed to initialize hardware server: {hardware_config_path}"
            )
        self.server.set_high_level_maintain_position()

        if self.server.is_bimanual():
            self.id_list = [0, 1]
        else:
            self.id_list = [0]

        Tr = np.eye(6)
        default_stiffness = np.diag([5000.0, 5000.0, 5000.0, 100.0, 100.0, 100.0])
        for robot_id in self.id_list:
            self.server.set_force_controlled_axis(Tr, compliant_dimensionality, robot_id)
            self.server.set_stiffness_matrix(default_stiffness, robot_id)

        self._wait_for_pose_feedback(ready_timeout_s)

    def _wait_for_pose_feedback(self, timeout_s: float) -> None:
        print("[replay] Waiting for hardware buffers to fill...")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.server.is_ready():
                print("[replay] Hardware buffers ready.")
                return
            time.sleep(0.05)
        raise RuntimeError(
            "Timed out waiting for hardware buffers to fill. "
            "At 500 Hz RTDE, robot_buffer_size=600 needs ~1.2 s to fill. "
            "Check: UR in remote mode, no other RTDE client connected, "
            "and no stale playback processes (kill % jobs / pkill -f run_virtual_target)."
        )

    @property
    def current_hardware_time_s(self) -> float:
        return self.server.get_timestamp_now_ms() / 1000.0

    def get_current_pose7(self, robot_id: int = 0) -> np.ndarray:
        return np.asarray(self.server.get_pose(1, robot_id), dtype=np.float64).reshape(7)

    def get_current_wrench6(self, robot_id: int = 0) -> np.ndarray:
        for getter_name in ("get_wrench", "get_robot_wrench"):
            try:
                getter = getattr(self.server, getter_name)
                wrench = np.asarray(getter(1, robot_id), dtype=np.float64).reshape(-1)
                if wrench.size >= 6 and np.isfinite(wrench[:6]).all():
                    return wrench[:6]
            except Exception:
                continue
        return np.full(6, np.nan, dtype=np.float64)

    def schedule_controls(
        self,
        pose7_cmd: np.ndarray,
        eoat_cmd: np.ndarray,
        stiffness_matrices_6x6: np.ndarray,
        timestamps: np.ndarray,
    ) -> bool:
        if len(self.id_list) == 1:
            assert pose7_cmd.shape[0] == 7
        else:
            assert pose7_cmd.shape[0] == 14
        assert timestamps.shape[0] == pose7_cmd.shape[1]

        if not self.server.is_ready():
            return False

        for robot_id in self.id_list:
            self.server.schedule_waypoints(
                pose7_cmd[robot_id * 7 : (robot_id + 1) * 7, :], timestamps, robot_id
            )
            if eoat_cmd is not None and self.server.has_eoat():
                self.server.schedule_eoat_waypoints(
                    eoat_cmd[robot_id * 2 : (robot_id + 1) * 2, :],
                    timestamps,
                    robot_id,
                )

        if stiffness_matrices_6x6 is not None:
            if len(self.id_list) == 1:
                assert stiffness_matrices_6x6.shape[0] == 6
            else:
                assert stiffness_matrices_6x6.shape[0] == 12
            assert stiffness_matrices_6x6.shape[1] == timestamps.shape[0] * 6
            for robot_id in self.id_list:
                self.server.schedule_stiffness(
                    stiffness_matrices_6x6[6 * robot_id : 6 * (robot_id + 1), :],
                    timestamps,
                    robot_id,
                )
        return True

    def cleanup(self) -> None:
        pass


class ReplayLog:
    def __init__(self) -> None:
        self.time_s: List[float] = []
        self.pose_cmd: List[np.ndarray] = []
        self.pose_fb: List[np.ndarray] = []
        self.wrench: List[np.ndarray] = []

    def append(
        self,
        time_s: float,
        pose_cmd: np.ndarray,
        pose_fb: np.ndarray,
        wrench: np.ndarray,
    ) -> None:
        self.time_s.append(float(time_s))
        self.pose_cmd.append(np.asarray(pose_cmd, dtype=np.float64).reshape(7))
        self.pose_fb.append(np.asarray(pose_fb, dtype=np.float64).reshape(7))
        self.wrench.append(np.asarray(wrench, dtype=np.float64).reshape(6))

    def to_arrays(self) -> Dict[str, np.ndarray]:
        if not self.time_s:
            return {}
        return {
            "time_s": np.asarray(self.time_s, dtype=np.float64),
            "pose_cmd": np.stack(self.pose_cmd, axis=0),
            "pose_fb": np.stack(self.pose_fb, axis=0),
            "wrench": np.stack(self.wrench, axis=0),
        }


def _sample_during_chunk(
    env: RobotPlaybackHardware,
    replay_log: ReplayLog,
    replay_t0_s: float,
    target_chunk: np.ndarray,
    step_dt_s: float,
) -> None:
    n_steps = target_chunk.shape[0]
    for step_i in range(n_steps):
        if step_i > 0:
            time.sleep(step_dt_s)
        replay_log.append(
            time_s=env.current_hardware_time_s - replay_t0_s,
            pose_cmd=target_chunk[step_i],
            pose_fb=env.get_current_pose7(),
            wrench=env.get_current_wrench6(),
        )


def save_replay_log_plot(
    replay_log: ReplayLog,
    output_path: str,
    episode: str,
) -> None:
    import matplotlib.pyplot as plt

    data = replay_log.to_arrays()
    if not data:
        print("[replay] No logged samples; skipping plot.")
        return

    time_s = data["time_s"]
    pose_cmd = data["pose_cmd"]
    pose_fb = data["pose_fb"]
    wrench = data["wrench"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 1.0, 1.2])
    ax_pos = fig.add_subplot(gs[0])
    ax_force = fig.add_subplot(gs[1], sharex=ax_pos)
    ax_torque = fig.add_subplot(gs[2], sharex=ax_pos)
    ax3d = fig.add_subplot(gs[3], projection="3d")
    fig.suptitle(f"Replay log: {episode}", fontsize=14)

    pos_labels = ("x", "y", "z")
    for i, label in enumerate(pos_labels):
        ax_pos.plot(time_s, pose_cmd[:, i], "--", linewidth=1.2, label=f"cmd {label}")
        ax_pos.plot(time_s, pose_fb[:, i], "-", linewidth=1.2, label=f"fb {label}")
    ax_pos.set_ylabel("position (m)")
    ax_pos.legend(loc="upper right", ncol=2, fontsize=8)
    ax_pos.grid(True, alpha=0.3)

    force_labels = ("fx", "fy", "fz")
    for i, label in enumerate(force_labels):
        ax_force.plot(time_s, wrench[:, i], "-", linewidth=1.2, label=label)
    ax_force.set_ylabel("force (N)")
    ax_force.legend(loc="upper right", fontsize=8)
    ax_force.grid(True, alpha=0.3)

    torque_labels = ("mx", "my", "mz")
    for i, label in enumerate(torque_labels):
        ax_torque.plot(time_s, wrench[:, i + 3], "-", linewidth=1.2, label=label)
    ax_torque.set_ylabel("torque (Nm)")
    ax_torque.set_xlabel("time (s)")
    ax_torque.legend(loc="upper right", fontsize=8)
    ax_torque.grid(True, alpha=0.3)

    ax3d.plot(
        pose_cmd[:, 0],
        pose_cmd[:, 1],
        pose_cmd[:, 2],
        "--",
        color="tab:orange",
        linewidth=1.5,
        label="command",
    )
    ax3d.plot(
        pose_fb[:, 0],
        pose_fb[:, 1],
        pose_fb[:, 2],
        "-",
        color="tab:blue",
        linewidth=1.5,
        label="feedback",
    )
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.legend(loc="upper right", fontsize=8)
    ax3d.set_title("TCP trajectory")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[replay] Saved plot: {output_path}")


def plot_trajectory_preview(
    ts_pose_command: np.ndarray,
    ts_pose_virtual_target: np.ndarray,
    selected_target: np.ndarray,
    episode: str,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        ts_pose_command[:, 0],
        ts_pose_command[:, 1],
        ts_pose_command[:, 2],
        color="red",
        linewidth=1.5,
        label="command",
    )
    ax.plot(
        ts_pose_virtual_target[:, 0],
        ts_pose_virtual_target[:, 1],
        ts_pose_virtual_target[:, 2],
        color="blue",
        linewidth=1.5,
        label="virtual_target",
    )
    ax.plot(
        selected_target[:, 0],
        selected_target[:, 1],
        selected_target[:, 2],
        color="green",
        linewidth=2.0,
        label="replay_target",
    )
    ax.set_title(f"Replay trajectory preview: {episode}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay dataset virtual target/stiffness trajectories through UMIFT scheduling."
    )
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--episode", default="episode_0", type=str)
    parser.add_argument("--raw_time_step_s", default=-1.0, type=float)
    parser.add_argument("--slow_down_factor", default=1.0, type=float)
    parser.add_argument("--execution_horizon", default=12, type=int)
    parser.add_argument("--default_stiffness", default=5000.0, type=float)
    parser.add_argument("--default_stiffness_rot", default=100.0, type=float)
    parser.add_argument("--test_nominal_target", action="store_true")
    parser.add_argument("--test_nominal_target_stiffness", default=-1.0, type=float)
    parser.add_argument("--gripper_action_scale", default=1000.0, type=float)
    parser.add_argument("--pausing_mode", action="store_true")
    parser.add_argument("--plot_trajectory", action="store_true")
    parser.add_argument(
        "--disable_frame_alignment",
        action="store_true",
        help="Disable anchoring trajectory frame to current robot pose.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--hardware_config_path", default="", type=str)
    parser.add_argument("--control_start_delay_s", default=0.15, type=float)
    parser.add_argument(
        "--log_output_dir",
        default="",
        type=str,
        help="Directory for post-replay pose/force plot (default: PYRITE_CONTROL_LOG_FOLDERS or ./replay_logs).",
    )
    parser.add_argument(
        "--disable_replay_log",
        action="store_true",
        help="Skip per-step logging and post-replay plot.",
    )
    args = parser.parse_args()

    episode_data = _load_episode(args.dataset_path, args.episode)

    ts_pose_command = np.asarray(episode_data["ts_pose_command_0"], dtype=np.float64)
    ts_pose_virtual_target = np.asarray(
        episode_data["ts_pose_virtual_target_0"], dtype=np.float64
    )
    ts_stiffness = np.asarray(episode_data["stiffness_0"], dtype=np.float64).reshape(-1)
    robot_time_stamps = np.asarray(
        episode_data["robot_time_stamps_0"], dtype=np.float64
    ).reshape(-1)
    n = ts_pose_command.shape[0]

    if n == 0:
        raise ValueError("Selected episode has zero trajectory length.")
    if ts_pose_virtual_target.shape[0] != n or ts_stiffness.shape[0] != n:
        raise ValueError("Trajectory arrays have inconsistent lengths.")

    replay_dt = (
        args.raw_time_step_s
        if args.raw_time_step_s > 0
        else _estimate_dt_from_robot_time_stamps(robot_time_stamps)
    )
    print(f"[replay] Using replay dt: {replay_dt:.6f}s")
    if args.dry_run:
        selected_target_full = select_target_trajectory(
            ts_pose_command=ts_pose_command,
            ts_pose_virtual_target=ts_pose_virtual_target,
            test_nominal_target=args.test_nominal_target,
        )
        if args.plot_trajectory:
            plot_trajectory_preview(
                ts_pose_command=ts_pose_command,
                ts_pose_virtual_target=ts_pose_virtual_target,
                selected_target=selected_target_full,
                episode=args.episode,
            )
        current_pose7 = ts_pose_command[0]
        stiffness_mats = compute_stiffness_matrices(
            current_pose7=current_pose7,
            ts_pose_command=ts_pose_command,
            ts_pose_virtual_target=ts_pose_virtual_target,
            ts_stiffness=ts_stiffness,
            default_stiffness=args.default_stiffness,
            default_stiffness_rot=args.default_stiffness_rot,
            nominal_target_stiffness_override=args.test_nominal_target_stiffness,
        )
        replay_ts = compute_replay_timestamps_s(
            count=n, raw_time_step_s=replay_dt, slow_down_factor=args.slow_down_factor
        )
        print(f"[dry_run] Episode: {args.episode}")
        print(f"[dry_run] N: {n}")
        print(f"[dry_run] stiffness matrix shape: {stiffness_mats.shape}")
        print(
            f"[dry_run] first replay timestamps (s): {replay_ts[:min(5, len(replay_ts))]}"
        )
        print(f"[dry_run] finite check: {np.isfinite(stiffness_mats).all()}")
        return 0

    if not args.hardware_config_path:
        raise ValueError("--hardware_config_path is required unless --dry_run is used.")
    if "PYRITE_HARDWARE_CONFIG_FOLDERS" not in os.environ:
        raise ValueError("Please set PYRITE_HARDWARE_CONFIG_FOLDERS for hardware replay.")

    env = RobotPlaybackHardware(
        hardware_config_path=args.hardware_config_path,
        compliant_dimensionality=3,
    )

    # Frame alignment: map dataset frame to current robot pose frame using the
    # first command pose as anchor, so replay does not depend on T265 startup frame.
    if not args.disable_frame_alignment:
        current_pose7 = env.get_current_pose7()
        ts_pose_command, ts_pose_virtual_target = align_trajectories_to_current_pose(
            ts_pose_command=ts_pose_command,
            ts_pose_virtual_target=ts_pose_virtual_target,
            current_pose7=current_pose7,
        )
    selected_target_full = select_target_trajectory(
        ts_pose_command=ts_pose_command,
        ts_pose_virtual_target=ts_pose_virtual_target,
        test_nominal_target=args.test_nominal_target,
    )
    if args.plot_trajectory:
        plot_trajectory_preview(
            ts_pose_command=ts_pose_command,
            ts_pose_virtual_target=ts_pose_virtual_target,
            selected_target=selected_target_full,
            episode=args.episode,
        )

    replay_log = ReplayLog()
    replay_t0_s = env.current_hardware_time_s
    step_dt_s = replay_dt * args.slow_down_factor

    for start in range(0, n, args.execution_horizon):
        end = min(start + args.execution_horizon, n)

        current_pose7 = env.get_current_pose7()
        cmd_chunk = ts_pose_command[start:end]
        vt_chunk = ts_pose_virtual_target[start:end]
        k_chunk = ts_stiffness[start:end]

        stiffness_mats = compute_stiffness_matrices(
            current_pose7=current_pose7,
            ts_pose_command=cmd_chunk,
            ts_pose_virtual_target=vt_chunk,
            ts_stiffness=k_chunk,
            default_stiffness=args.default_stiffness,
            default_stiffness_rot=args.default_stiffness_rot,
            nominal_target_stiffness_override=args.test_nominal_target_stiffness,
        )

        target_chunk = select_target_trajectory(
            ts_pose_command=cmd_chunk,
            ts_pose_virtual_target=vt_chunk,
            test_nominal_target=args.test_nominal_target,
        )
        pose7_cmd = target_chunk.T
        eoat_cmd = _build_eoat_cmd_if_available(
            episode_data, start, end, args.gripper_action_scale
        )
        rel_ts = compute_replay_timestamps_s(
            count=end - start,
            raw_time_step_s=replay_dt,
            slow_down_factor=args.slow_down_factor,
        )
        start_time_s = env.current_hardware_time_s + args.control_start_delay_s
        ts_ms = (rel_ts + start_time_s) * 1000.0

        if args.pausing_mode:
            input(f"Press Enter to execute chunk [{start}:{end}]...")
        ok = env.schedule_controls(
            pose7_cmd=pose7_cmd,
            eoat_cmd=eoat_cmd,
            stiffness_matrices_6x6=stiffness_mats,
            timestamps=ts_ms,
        )
        if not ok:
            raise RuntimeError("Hardware server not ready; schedule_controls returned False.")

        if not args.disable_replay_log:
            _sample_during_chunk(
                env=env,
                replay_log=replay_log,
                replay_t0_s=replay_t0_s,
                target_chunk=target_chunk,
                step_dt_s=step_dt_s,
            )
        elif not args.pausing_mode:
            time.sleep(max(0.0, rel_ts[-1] if len(rel_ts) > 0 else 0.0))

    env.cleanup()
    print("[replay] Completed playback.")

    if not args.disable_replay_log:
        log_dir = args.log_output_dir
        if not log_dir:
            log_dir = os.environ.get("PYRITE_CONTROL_LOG_FOLDERS", "./replay_logs")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(log_dir, f"{args.episode}_replay_{stamp}.png")
        save_replay_log_plot(replay_log, plot_path, args.episode)
        npz_path = os.path.splitext(plot_path)[0] + ".npz"
        arrays = replay_log.to_arrays()
        if arrays:
            np.savez(npz_path, **arrays)
            print(f"[replay] Saved log data: {npz_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
