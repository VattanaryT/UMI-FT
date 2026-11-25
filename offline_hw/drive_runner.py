#!/usr/bin/env python3
"""Run umift_env_runner.main() with the robot class replaced by MockLayupEnv.

No RealSense, no RTDE, no motion. Stdout is the runner's own log.
"""

from __future__ import annotations

import argparse
import builtins
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "UMIFT_Data" / "surface_frames_scripts"
OUT = ROOT / "offline_hw" / "out"
TRAIN = Path("/mnt/data/umidata/ft/training_outputs")
ZARR_CAUSAL = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_causal_foot/acp_replay_buffer.zarr"
)
ZARR_TRACK = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_smooth_tcp_track_clean/acp_replay_buffer.zarr"
)
FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
LIFTOS_ZARR = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_field_v2n_v3r_liftos/acp_replay_buffer.zarr"
)
LOGGED_ROLLOUT = Path("/mnt/data/umidata/ft/vtevy/rollouts/2026-11-24_liftos_hw")
LOGGED_ROBOT = Path("/home/boeing/umi_ft_ws/UMI-FT/runs/episode_1795573164/robot_data_0.json")
EXTRINSICS = (
    ROOT / "PyriteConfig/tasks/umift/layup_workcell_extrinsics_20261116_orange.yaml"
)
CKPT = {
    "track": (
        "/oct_v35_h4s_r4vt_track_clean/"
        "oct-v35-h4s-r4vt-track-clean_e100_bs64_valtopk_43351238/checkpoints/"
        "epoch=0055-val_loss=0.030.ckpt"
    ),
    "liftos": str(TRAIN / "oct_v35_h4s_r4vt_field_v2n_v3r_liftos/latest.ckpt"),
    "v3r": str(TRAIN / "oct_v35_h4s_r4vt_field_v3r/latest_final.ckpt"),
}
CAUSAL_NORM = (
    SCRIPTS / "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_causal_foot.pkl"
)
CAUSAL_SHA = "14308b72d55600b4359dfa67d10aedab378e42ad26f2bbcac988eed4d97a1ac2"

WRIST_CALLS = []


def _paths() -> None:
    for entry in (
        ROOT,
        ROOT / "PyriteUtility",
        ROOT / "PyriteML" / "multimodal_representation" / "multimodal",
        ROOT / "PyriteML",
        SCRIPTS,
        ROOT / "offline_hw",
    ):
        sys.path.insert(0, str(entry))
    main = Path("/home/boeing/umi_ft_ws/UMI-FT")
    kept = []
    for entry in sys.path:
        try:
            resolved = Path(entry).resolve()
        except OSError:
            kept.append(entry)
            continue
        if resolved == main or main in resolved.parents:
            continue
        kept.append(entry)
    sys.path[:] = kept


def _prepare_env(args) -> None:
    os.environ["PYRITE_CHECKPOINT_FOLDERS"] = str(TRAIN)
    if args.hwconfig_dir:
        os.environ["PYRITE_HARDWARE_CONFIG_FOLDERS"] = str(args.hwconfig_dir)
    elif args.slow_down is not None and float(args.slow_down) == 6.0:
        os.environ["PYRITE_HARDWARE_CONFIG_FOLDERS"] = (
            "/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/offline_hw/hwconfig_spring25_sd6"
        )
    else:
        os.environ["PYRITE_HARDWARE_CONFIG_FOLDERS"] = (
            "/home/boeing/umi_ft_ws/hardware_interfaces/workcell/table_top_manip/config"
        )
    os.environ["PYRITE_CONTROL_LOG_FOLDERS"] = str(OUT / args.name)
    os.environ["UMIFT_WORKCELL_EXTRINSICS"] = str(EXTRINSICS)
    os.environ["UMIFT_FOOT_WALK"] = "geodesic"
    os.environ["UMIFT_ABORT_ON_MEASURED_MESH_DISTANCE"] = "1"
    os.environ["UMIFT_ALIGN_OBS_TO_IMAGE"] = "1"
    os.environ.pop("UMIFT_ENABLE_DELTA_N_FLOOR", None)
    os.environ.pop("UMIFT_BENCH_LIFT_M", None)
    os.environ.pop("UMIFT_LEAD_IN_S", None)
    os.environ.pop("UMIFT_STIFFNESS_FLOOR", None)
    os.environ.pop("UMIFT_RUN_DIR", None)
    if args.mode == "label":
        os.environ["UMIFT_REPLAY_EPISODE"] = "9"
        os.environ.pop("UMIFT_POLICY_START_EPISODE", None)
        os.environ.pop("UMIFT_POLICY_START_INDEX", None)
        os.environ["UMIFT_CKPT_PATH"] = CKPT["track"]
        os.environ["UMIFT_R4VT_ZARR"] = str(ZARR_CAUSAL)
        os.environ["UMIFT_R4VT_NORMALIZER"] = str(CAUSAL_NORM)
        os.environ["UMIFT_R4VT_NORMALIZER_SHA"] = CAUSAL_SHA
        os.environ["UMIFT_ACTION_NORMALIZER"] = str(CAUSAL_NORM)
        os.environ["UMIFT_ACTION_NORMALIZER_SHA"] = CAUSAL_SHA
        if args.bench:
            os.environ["UMIFT_ENABLE_DELTA_N_FLOOR"] = "1"
        os.environ.pop("UMIFT_MAX_DURATION_S", None)
    else:
        os.environ.pop("UMIFT_REPLAY_EPISODE", None)
        os.environ["UMIFT_POLICY_START_EPISODE"] = "9"
        os.environ["UMIFT_POLICY_START_INDEX"] = "1220"
        os.environ["UMIFT_CKPT_PATH"] = CKPT[args.which]
        os.environ["UMIFT_R4VT_ZARR"] = str(ZARR_TRACK)
        os.environ.pop("UMIFT_ACTION_NORMALIZER", None)
        os.environ.pop("UMIFT_ACTION_NORMALIZER_SHA", None)
        os.environ.pop("UMIFT_R4VT_NORMALIZER", None)
        os.environ.pop("UMIFT_R4VT_NORMALIZER_SHA", None)
    if args.logged:
        args.which = "liftos"
        args.mold_local = True
        args.newest_chunk = True
        os.environ["UMIFT_CKPT_PATH"] = CKPT["liftos"]
        os.environ["UMIFT_MAX_DURATION_S"] = "22"
    if args.mold_local:
        os.environ["UMIFT_MOLD_LOCAL_POS"] = "1"
        os.environ["UMIFT_MOLD_LOCAL_ZARR"] = str(LIFTOS_ZARR)
    else:
        os.environ.pop("UMIFT_MOLD_LOCAL_POS", None)
    if args.newest_chunk and args.pose_ensemble:
        raise SystemExit("--pose-ensemble and --newest-chunk cannot be combined")
    if args.newest_chunk:
        os.environ["UMIFT_TCP_ROT_NEWEST"] = "1"
    else:
        os.environ.pop("UMIFT_TCP_ROT_NEWEST", None)
    if args.pose_ensemble:
        os.environ["UMIFT_POSE_ENSEMBLE"] = "1"
    else:
        os.environ.pop("UMIFT_POSE_ENSEMBLE", None)
    if args.tcp_offset_mm is not None:
        os.environ["UMIFT_TCP_OFFSET_MM"] = str(float(args.tcp_offset_mm))
    else:
        os.environ.pop("UMIFT_TCP_OFFSET_MM", None)
    if args.dn_bias_mm is not None:
        os.environ["UMIFT_DN_BIAS_MM"] = str(float(args.dn_bias_mm))
    else:
        os.environ.pop("UMIFT_DN_BIAS_MM", None)
    if args.exec_horizon is not None:
        if int(args.exec_horizon) < 1:
            raise SystemExit("--exec-horizon needs an integer >= 1")
        os.environ["UMIFT_EXEC_HORIZON"] = str(int(args.exec_horizon))
    else:
        os.environ.pop("UMIFT_EXEC_HORIZON", None)
    if args.slow_down is not None:
        if not float(args.slow_down) > 0.0:
            raise SystemExit("--slow-down needs a number > 0")
        os.environ["UMIFT_SLOW_DOWN_FACTOR"] = str(float(args.slow_down))
    else:
        os.environ.pop("UMIFT_SLOW_DOWN_FACTOR", None)


def _load_episode(zarr_path: Path, index: int):
    import zarr
    from PyriteConfig.tasks.umift.workcell_frames import load_workcell_extrinsics
    from PyriteEnvSuites.env_runners.dataset_start_pose import pose7_change_frame as change_frame

    group = zarr.open_group(str(zarr_path), mode="r")["data"]["episode_9"]
    pose = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    pose_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rgb = group["rgb_0"]
    rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
    wrench_t = np.asarray(group["wrench_time_stamps_0"], dtype=np.float64).reshape(-1)
    T = load_workcell_extrinsics().T_fixture_from_right_base
    base = change_frame(pose, np.linalg.inv(T))
    return {
        "pose_base": base,
        "pose_t": pose_t,
        "rgb": rgb,
        "rgb_t": rgb_t,
        "wrench": wrench,
        "wrench_t": wrench_t,
        "T": T,
        "index_t": float(pose_t[index]),
        "end_t": float(pose_t[-1]),
        "pose_fix": pose,
    }


def _logged_base_pose() -> np.ndarray:
    """Right-base pose at the first command of the 2026-11-24 liftos run."""
    import json
    import re

    import zarr

    text = re.sub(r",(\s*[}\]])", r"\1", LOGGED_ROBOT.read_text())
    rows = json.loads(text)
    rt = np.array([row["robot_time_stamps"] for row in rows], dtype=np.float64) / 1000.0
    pose = np.array([row["ts_pose_fb"] for row in rows], dtype=np.float64)
    log = zarr.open_group(str(LOGGED_ROLLOUT), mode="r")
    stamp = float(np.asarray(log["horizon_0"]["action"]["timestamps_s"]).reshape(-1)[0])
    idx = int(np.clip(np.searchsorted(rt, stamp), 1, len(rt) - 1))
    if abs(rt[idx - 1] - stamp) < abs(rt[idx] - stamp):
        idx -= 1
    return pose[idx]


def _install_logged_obs() -> None:
    """First policy window is the hardware obs. Later windows are the mock."""
    import zarr
    from PyriteUtility.planning_control.mpc import ModelPredictiveController

    log = zarr.open_group(str(LOGGED_ROLLOUT), mode="r")
    window = {
        key: np.asarray(log["horizon_0"]["obs_policy"][key])
        for key in log["horizon_0"]["obs_policy"].keys()
    }
    state = {"used": False}
    real = ModelPredictiveController.preprocess_policy_obs

    def wrapped(self, obs_sampled, ignore_rgb=False):
        obs, se3 = real(self, obs_sampled, ignore_rgb=ignore_rgb)
        if not state["used"]:
            state["used"] = True
            from mold_aware_runtime import apply_mold_local_pos

            obs = apply_mold_local_pos({key: np.array(value) for key, value in window.items()})
            print("[drive] horizon 0 uses the logged obs, mold-local applied")
        return obs, se3

    ModelPredictiveController.preprocess_policy_obs = wrapped


def _report_logged(out_dir: Path) -> None:
    import sys

    import zarr
    from scipy.spatial.transform import Rotation
    from fixb_labels import pose7_to_Rp

    logs = sorted(path for path in out_dir.iterdir() if (path / "horizon_0").exists())
    if not logs:
        print("[drive] no horizon log")
        sys.exit(1)
    log = zarr.open_group(str(logs[-1]), mode="r")
    horizons = sorted(
        (name for name in log if str(name).startswith("horizon_")),
        key=lambda name: int(str(name).split("_")[1]),
    )
    dn = np.concatenate(
        [np.asarray(log[name]["action"]["delta_n"]).reshape(-1) for name in horizons]
    )
    nom = np.concatenate(
        [np.asarray(log[name]["action"]["ts_nominal_targets_0"]) for name in horizons]
    )
    rotation, _ = pose7_to_Rp(nom)
    steps = np.array(
        [
            float(
                np.degrees(
                    np.linalg.norm(
                        Rotation.from_matrix(rotation[i].T @ rotation[i + 1]).as_rotvec()
                    )
                )
            )
            for i in range(len(rotation) - 1)
        ]
    )
    pinned = int(np.sum(np.abs(steps - 8.0) < 0.08))
    print(
        f"[drive] horizons {len(horizons)} dn0 {dn[0] * 1e3:+.2f} mm "
        f"dn med {np.median(dn) * 1e3:+.2f} mm "
        f"ori step med {np.median(steps):.2f} deg pinned {pinned}/{len(steps)}"
    )
    if not (dn[0] < 0.0 and pinned < len(steps) / 4):
        sys.exit(1)


def _raised_start(pose_fix_row, T, lift_m: float):
    from PyriteEnvSuites.env_runners.dataset_start_pose import pose7_change_frame

    lifted = np.asarray(pose_fix_row, dtype=np.float64).copy()
    lifted[1] -= float(lift_m)  # fixture +Y is down
    return pose7_change_frame(lifted, np.linalg.inv(T))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("label", "policy"))
    parser.add_argument("--which", choices=("liftos", "v3r"), default="liftos")
    parser.add_argument("--follow", choices=("command", "demo"), default="command")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--mold-local", action="store_true")
    parser.add_argument("--newest-chunk", action="store_true")
    parser.add_argument("--exec-horizon", type=int, default=None)
    parser.add_argument("--slow-down", type=float, default=None)
    parser.add_argument("--pose-ensemble", action="store_true")
    parser.add_argument("--tcp-offset-mm", type=float, default=None)
    parser.add_argument("--dn-bias-mm", type=float, default=None)
    parser.add_argument("--hwconfig-dir", default=None)
    parser.add_argument("--logged", action="store_true")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.logged:
        args.mode = "policy"
        args.which = "liftos"
        args.mold_local = True
        args.newest_chunk = True
    _paths()
    _prepare_env(args)
    OUT.joinpath(args.name).mkdir(parents=True, exist_ok=True)

    from offline_hw.mock_env import MockLayupEnv, save_schedules
    from fixb_smooth_field import SmoothFrameField
    from PyriteEnvSuites.env_runners.dataset_start_pose import pose7_change_frame

    zarr_path = ZARR_CAUSAL if args.mode == "label" else ZARR_TRACK
    episode = _load_episode(zarr_path, 1220)
    field = SmoothFrameField(FIELD, FIELD.with_name("fixb_smooth_frames.npz"))
    T = episode["T"]
    if args.mode == "policy":
        start_fix = episode["pose_fix"][1220]
        if args.logged:
            MockLayupEnv.initial_pose = _logged_base_pose()
            os.environ["UMIFT_MAX_DURATION_S"] = (
                "12" if args.exec_horizon is not None else "22"
            )
        else:
            MockLayupEnv.initial_pose = _raised_start(start_fix, T, 0.12)
            if args.exec_horizon is not None:
                # 8 s hold, then a few replans. Enough to read the step count.
                os.environ["UMIFT_MAX_DURATION_S"] = "12"
            else:
                remain = (episode["end_t"] - episode["index_t"]) * 2.0 + 5.0
                os.environ["UMIFT_MAX_DURATION_S"] = f"{remain:.1f}"
    else:
        MockLayupEnv.initial_pose = episode["pose_base"][0]
    MockLayupEnv.follow = args.follow
    MockLayupEnv.demo_pose = episode["pose_base"]
    MockLayupEnv.demo_pose_t = episode["pose_t"]
    MockLayupEnv.demo_rgb = episode["rgb"]
    MockLayupEnv.demo_rgb_t = episode["rgb_t"]
    MockLayupEnv.demo_wrench = episode["wrench"]
    MockLayupEnv.demo_wrench_t = episode["wrench_t"]
    MockLayupEnv.demo_origin_t = episode["index_t"] if args.mode == "policy" else float(episode["pose_t"][0])
    MockLayupEnv.episode_folder = str(OUT / args.name / "robot_dump")
    MockLayupEnv.schedules = []

    outside = {"xyz": None}

    def project(pose_base):
        """Stop the measured TCP at the surface. The scheduled VT is left unchanged."""
        fix = pose7_change_frame(np.asarray(pose_base, dtype=np.float64).reshape(7), T)
        start = outside["xyz"]
        if start is None:
            start = fix[:3].copy()
        alphas = np.linspace(0.0, 1.0, 17)
        samples = (1.0 - alphas)[:, None] * start + alphas[:, None] * fix[:3]
        sd = np.asarray(field.query(samples)["sd"], dtype=np.float64).reshape(-1)
        inside = np.flatnonzero(sd < -0.001)
        out = fix.copy()
        if len(inside) == 0:
            outside["xyz"] = fix[:3].copy()
            return np.asarray(pose_base, dtype=np.float64).reshape(7)
        i = int(inside[0])
        if i == 0:
            queried = field.query(start.reshape(1, 3))
            rotation = np.asarray(queried["R"], dtype=np.float64).reshape(3, 3)
            out[:3] = np.asarray(queried["x_s"], dtype=np.float64).reshape(3) - 0.001 * rotation[:, 2]
        else:
            span = float(sd[i - 1] - sd[i])
            frac = 0.0 if span <= 1e-9 else float(sd[i - 1] / span)
            frac = min(max(frac, 0.0), 1.0)
            out[:3] = (1.0 - frac) * samples[i - 1] + frac * samples[i]
        outside["xyz"] = out[:3].copy()
        return pose7_change_frame(out, np.linalg.inv(T))

    MockLayupEnv.project_pose = project if args.follow == "command" else None

    import cv2
    import PyriteEnvSuites.env_runners.umift_env_runner as runner
    import PyriteEnvSuites.env_runners.policy_start_pose as start_pose

    runner.ManipServerLayupEnv = MockLayupEnv
    import PyriteEnvSuites.env_runners.rollout_cam_recorder as cams

    class _NoCamera:
        def __init__(self, out_dir, **_kwargs):
            self.out_dir = out_dir

        def start(self):
            raise RuntimeError("offline mock does not open cameras")

        def stop(self):
            return {"wrist_frames": 0, "orbbec_frames": 0, "wrist_rate": 0, "center_rate": 0}

        def push_wrist_rgb(self, *_a, **_k):
            return None

    cams.RolloutCamRecorder = _NoCamera
    runner.RolloutCamRecorder = _NoCamera
    cv2.imshow = lambda *a, **k: None
    cv2.waitKey = lambda delay=0: ord("q")
    cv2.destroyAllWindows = lambda: None

    def refuse_joints(*_a, **_k):
        print("[mock] read_robot_joints_via_rtde blocked; returning a stub that passes the wrist-3 room check")
        return [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

    def refuse_ik(*_a, **_k):
        raise RuntimeError("offline mock refuses RTDE inverse kinematics")

    runner.read_robot_joints_via_rtde = refuse_joints
    start_pose.query_wrist_ik = refuse_ik
    start_pose.movej_path = refuse_ik

    def wrist(env, target_pose7, *, python, ip):
        pose = np.asarray(target_pose7, dtype=np.float64).reshape(7).copy()
        WRIST_CALLS.append({"pose_base": pose.tolist(), "ip": str(ip)})
        print(
            "[mock] apply_policy_wrist_winding would pause RTDE and moveJ wrist 3. "
            "Not called. Target " + " ".join(f"{v:.5f}" for v in pose)
        )
        return 0.0

    runner.apply_policy_wrist_winding = wrist
    if args.logged:
        _install_logged_obs()

    real_sleep = time.sleep

    def fake_sleep(dt):
        env = MockLayupEnv.active
        if env is not None and dt:
            env.advance(float(dt))
        if dt and float(dt) >= 1.0:
            real_sleep(0.001)

    time.sleep = fake_sleep

    def fake_input(prompt=""):
        text = str(prompt)
        print(f"[mock-input] {text}")
        if "select an option" in text:
            return "q"
        return ""

    builtins.input = fake_input

    print(
        f"[drive] mode={args.mode} which={args.which} follow={args.follow} "
        f"bench={args.bench} duration_cap={os.environ.get('UMIFT_MAX_DURATION_S', '480')}"
    )
    try:
        runner.main()
    finally:
        time.sleep = real_sleep
        save_schedules(OUT / args.name / "schedules.npz")
        (OUT / args.name / "wrist_calls.json").write_text(
            json.dumps(WRIST_CALLS, indent=2) + "\n"
        )
        print(f"[drive] schedules {len(MockLayupEnv.schedules)} wrist_calls {len(WRIST_CALLS)}")
        if args.logged:
            _report_logged(OUT / args.name)


if __name__ == "__main__":
    main()
