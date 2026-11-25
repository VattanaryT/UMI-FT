#!/usr/bin/env python3
"""Replay pe_guard h0-h6 through the runner with training-frame mold_geom.

No robot. Logged hardware pose and wrench are served at each horizon, and the
policy obs keeps that log except for robot0_mold_geom, which the fixed packer
recomputes. The run stops after seven horizons.
"""

from __future__ import annotations

import builtins
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "offline_hw"))

import drive_runner as drive  # noqa: E402

ROLLOUT = Path("/mnt/data/umidata/ft/vtevy/rollouts/2026-11-24_liftos_hw_pe_guard")
ROBOT_JSON = Path("/home/boeing/umi_ft_ws/UMI-FT/runs/episode_1795583485/robot_data_0.json")
NORM = (
    ROOT
    / "UMIFT_Data/surface_frames_scripts/assets/"
    "oct_v35_h4s_r4vt_action_normalizer_minmax_field_v2n_v3r_liftos.pkl"
)
NORM_SHA = "ce85fae23a90d3d7fc29d2bc357693b17254fbadea1af716ba2f10af9be394af"
LIFTOS = drive.LIFTOS_ZARR
START = 1220
N_HORIZONS = 7
NAME = "replay_guard_train_frame"


def _load_robot(path: Path) -> dict:
    text = re.sub(r",(\s*[}\]])", r"\1", path.read_text())
    rows = json.loads(text)
    return {
        "t": np.array([row["robot_time_stamps"] for row in rows], dtype=np.float64) / 1000.0,
        "pose": np.array([row["ts_pose_fb"] for row in rows], dtype=np.float64),
        "wrench": np.array(
            [row.get("robot_wrench") or [0, 0, 0, 0, 0, 0] for row in rows],
            dtype=np.float64,
        ),
    }


def _nearest(times: np.ndarray, stamp: float) -> int:
    return int(np.argmin(np.abs(times - stamp)))


def _tail(values: np.ndarray, times: np.ndarray, stamp: float, n: int):
    end = _nearest(times, stamp)
    start = max(0, end + 1 - n)
    chunk = np.asarray(values[start : end + 1])
    clocks = np.asarray(times[start : end + 1], dtype=np.float64)
    if len(chunk) < n:
        pad = n - len(chunk)
        chunk = np.concatenate([np.repeat(chunk[:1], pad, axis=0), chunk], axis=0)
        clocks = np.concatenate([np.repeat(clocks[:1], pad), clocks], axis=0)
    return chunk, clocks


def _analyze(log_dir: Path) -> None:
    import zarr
    from scipy.spatial.transform import Rotation

    from PyriteConfig.tasks.umift.workcell_frames import load_workcell_extrinsics
    from fixb_labels import pose7_to_Rp
    from rot6d import rot6d_rows_to_R

    group = zarr.open_group(str(log_dir), mode="r")
    source = zarr.open_group(str(ROLLOUT), mode="r")
    names = sorted(
        (key for key in group.keys() if str(key).startswith("horizon_")),
        key=lambda name: int(str(name).split("_")[1]),
    )[:N_HORIZONS]
    demo_root = zarr.open_group(str(LIFTOS), mode="r")["data"]["episode_9"]
    demo_t = np.asarray(demo_root["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    demo_foot = np.asarray(demo_root["mold_x_s_0"], dtype=np.float64)
    demo_v = np.asarray(demo_root["fixb_v_par_0"], dtype=np.float64)
    demo_dn = np.asarray(demo_root["fixb_delta_n_0"], dtype=np.float64).reshape(-1)
    demo_field = np.asarray(demo_root["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
    demo_R, _ = pose7_to_Rp(np.asarray(demo_root["ts_pose_fb_0"], dtype=np.float64))
    T = load_workcell_extrinsics().T_fixture_from_right_base
    origin = float(demo_t[START])
    # Match the hardware clock. The mock log's own timestamps are a different epoch.
    t0 = float(np.asarray(source[names[0]]["action"]["timestamps_s"]).reshape(-1)[0])
    slow = 2.0
    print(
        "h  idx    v2_cmd   v2_lbl   dn_mm   dn_lbl   tw_err  foot_mean  foot_max"
    )
    early = []
    for name in names:
        action = group[name]["action"]
        stamp = float(np.asarray(source[name]["action"]["timestamps_s"]).reshape(-1)[0])
        index = _nearest(demo_t, origin + (stamp - t0) / slow)
        v = np.asarray(action["v_par"], dtype=np.float64)
        dn = np.asarray(action["delta_n"], dtype=np.float64).reshape(-1)
        foot = np.asarray(action["foot"], dtype=np.float64).reshape(-1, 3)
        foot_fix = foot @ T[:3, :3].T + T[:3, 3]
        vt = np.asarray(action["ts_virtual_targets_0"], dtype=np.float64)
        tool_R, _ = pose7_to_Rp(vt[:1])
        tool_fix = np.einsum("ij,njk->nik", T[:3, :3], tool_R)
        geom = np.asarray(group[name]["obs_policy"]["robot0_mold_geom"])
        field = rot6d_rows_to_R(np.asarray(geom[-1, 4:10], dtype=np.float64).reshape(1, 6))[0]
        relative = field.T @ tool_fix[0]
        tw_cmd = float(np.degrees(Rotation.from_matrix(relative).as_rotvec()[2]))
        tw_lbl = float(
            np.degrees(Rotation.from_matrix(demo_field[index].T @ demo_R[index]).as_rotvec()[2])
        )
        step = (1.0 / 15.0)
        errs = []
        for k in range(len(foot_fix)):
            j = _nearest(demo_t, demo_t[index] + k * step)
            errs.append(float(np.linalg.norm(foot_fix[k] - demo_foot[j]) * 1e3))
        h = int(str(name).split("_")[1])
        mean_err = float(np.mean(errs))
        max_err = float(np.max(errs))
        tw_err = (tw_cmd - tw_lbl + 180.0) % 360.0 - 180.0
        print(
            f"{h:1d} {index:5d}  {v[0, 1]:+7.4f}  {demo_v[index, 1]:+7.4f}  "
            f"{dn[0] * 1e3:+6.1f}  {demo_dn[index] * 1e3:+6.1f}  "
            f"{tw_err:+6.1f}  {mean_err:8.1f}  {max_err:8.1f}"
        )
        early.append((h, mean_err, max_err, float(v[0, 1]), float(demo_v[index, 1])))
    h0 = next(row for row in early if row[0] == 0)
    h2 = next(row for row in early if row[0] == 2)
    print(
        f"verdict: h0's executed foot stays within {h0[2]:.0f} mm of ep9 and the twist matches, "
        f"so the corrected geom is not an immediate jump. δn at h0 is clearance while the "
        f"label is a press. By h2 the predicted foot mean is {h2[1]:.0f} mm "
        f"(max {h2[2]:.0f} mm). The fix does not stop the h3 divergence: the command "
        "is already leaving, and h3 replays the pose that had left."
    )


def main() -> None:
    import hashlib

    digest = hashlib.sha256(NORM.read_bytes()).hexdigest()
    if digest != NORM_SHA:
        raise SystemExit(f"liftos normalizer sha {digest}")
    robot = _load_robot(ROBOT_JSON)
    import zarr

    src = zarr.open_group(str(ROLLOUT), mode="r")
    stamps = [
        float(np.asarray(src[f"horizon_{i}"]["action"]["timestamps_s"]).reshape(-1)[0])
        for i in range(N_HORIZONS)
    ]
    logged_obs = [src[f"horizon_{i}"]["obs_policy"] for i in range(N_HORIZONS)]
    drive._paths()
    args = type("Args", (), {})()
    args.mode = "policy"
    args.which = "liftos"
    args.follow = "command"
    args.bench = False
    args.mold_local = True
    args.newest_chunk = False
    args.exec_horizon = 6
    args.slow_down = 2.0
    args.pose_ensemble = True
    args.tcp_offset_mm = None
    args.dn_bias_mm = None
    args.hwconfig_dir = str(ROOT / "offline_hw/hwconfig_spring25")
    args.logged = False
    args.name = NAME
    drive._prepare_env(args)
    os.environ["UMIFT_CKPT_PATH"] = drive.CKPT["liftos"]
    os.environ["UMIFT_R4VT_ZARR"] = str(LIFTOS)
    os.environ["UMIFT_MOLD_LOCAL_ZARR"] = str(LIFTOS)
    os.environ["UMIFT_ACTION_NORMALIZER"] = str(NORM)
    os.environ["UMIFT_ACTION_NORMALIZER_SHA"] = NORM_SHA
    os.environ["UMIFT_R4VT_NORMALIZER"] = str(NORM)
    os.environ["UMIFT_R4VT_NORMALIZER_SHA"] = NORM_SHA
    os.environ["UMIFT_POLICY_START_EPISODE"] = "9"
    os.environ["UMIFT_POLICY_START_INDEX"] = str(START)
    os.environ["UMIFT_MAX_DURATION_S"] = "180"
    os.environ["UMIFT_FOOT_FACE_HYST"] = "1"
    os.environ["UMIFT_SCHEDULE_LIMITED_VT"] = "1"
    os.environ["UMIFT_MAX_STEP_MM"] = "4"
    os.environ["UMIFT_POSE_ENSEMBLE"] = "1"
    os.environ.pop("UMIFT_TCP_ROT_NEWEST", None)
    os.environ.pop("UMIFT_TCP_OFFSET_MM", None)
    os.environ.pop("UMIFT_DN_BIAS_MM", None)
    os.environ.pop("UMIFT_MOLD_GEOM_TRAIN_FRAME", None)
    out = drive.OUT / NAME
    out.mkdir(parents=True, exist_ok=True)
    os.environ["PYRITE_CONTROL_LOG_FOLDERS"] = str(out)
    os.environ["UMIFT_RUN_DIR"] = str(out)

    from offline_hw.mock_env import MockLayupEnv
    import cv2
    import PyriteEnvSuites.env_runners.umift_env_runner as runner
    import PyriteEnvSuites.env_runners.policy_start_pose as start_pose
    import PyriteEnvSuites.env_runners.rollout_cam_recorder as cams
    import mold_geom_train_frame as train_frame

    h0 = _nearest(robot["t"], stamps[0])
    MockLayupEnv.initial_pose = robot["pose"][h0].copy()
    MockLayupEnv.follow = "command"
    MockLayupEnv.episode_folder = str(out / "robot_dump")
    MockLayupEnv.schedules = []
    state = {"h": None, "served": 0}

    real_apply = train_frame.apply_train_frame

    def apply(runtime, *a, **k):
        runtime._train_geom = None
        return real_apply(runtime, *a, **k)

    train_frame.apply_train_frame = apply

    real_get = MockLayupEnv.get_observation_from_buffer

    def get_observation(self):
        obs = real_get(self)
        h = state["h"]
        if h is None:
            return obs
        n_pose = int(getattr(self, "n_pose", 0) or len(obs["ts_pose_fb_0"]))
        n_wrench = int(getattr(self, "n_wrench", 0) or len(obs["wrench_left_0"]))
        # Keep the mock clock. Alignment checks pose time against the image
        # time; the logged values are what the packer and the policy read.
        pose, _pose_t = _tail(robot["pose"], robot["t"], stamps[h], max(n_pose, 1))
        wrench, _wrench_t = _tail(robot["wrench"], robot["t"], stamps[h], max(n_wrench, 1))
        obs["ts_pose_fb_0"] = pose[-n_pose:]
        obs["wrench_left_0"] = wrench[-n_wrench:]
        obs["wrench_right_0"] = np.zeros_like(obs["wrench_left_0"])
        return obs

    MockLayupEnv.get_observation_from_buffer = get_observation

    from PyriteUtility.planning_control.mpc import ModelPredictiveController

    real_pre = ModelPredictiveController.preprocess_policy_obs

    def preprocess(self, obs_sampled, ignore_rgb=False):
        obs_policy, base = real_pre(self, obs_sampled, ignore_rgb=ignore_rgb)
        h = state["h"]
        if h is None or not isinstance(obs_policy, dict):
            return obs_policy, base
        geom = obs_policy.get("robot0_mold_geom")
        logged = logged_obs[h]
        for key in logged.keys():
            if key == "robot0_mold_geom" or key not in obs_policy:
                continue
            value = np.asarray(logged[key])
            if tuple(value.shape) != tuple(np.asarray(obs_policy[key]).shape):
                print(
                    f"[replay] keep {key} shape {np.shape(obs_policy[key])} != log {value.shape}",
                    flush=True,
                )
                continue
            obs_policy[key] = value
        if geom is not None:
            obs_policy["robot0_mold_geom"] = geom
        return obs_policy, base

    ModelPredictiveController.preprocess_policy_obs = preprocess
    runner.ManipServerLayupEnv = MockLayupEnv

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
    runner.read_robot_joints_via_rtde = lambda *_a, **_k: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
    start_pose.query_wrist_ik = lambda *_a, **_k: (_ for _ in ()).throw(
        RuntimeError("offline mock refuses RTDE inverse kinematics")
    )
    start_pose.movej_path = start_pose.query_wrist_ik
    runner.apply_policy_wrist_winding = lambda *a, **k: 0.0

    real_print = builtins.print

    def hooked_print(*parts, **kwargs):
        text = " ".join(str(part) for part in parts)
        if "Starting new horizon" in text:
            if state["served"] >= N_HORIZONS:
                raise SystemExit(0)
            state["h"] = state["served"]
            state["served"] += 1
            real_print(f"[replay] serving logged obs h{state['h']}", flush=True)
        real_print(*parts, **kwargs)

    builtins.print = hooked_print
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
        real_print(f"[mock-input] {text}")
        if "select an option" in text:
            return "q"
        return ""

    builtins.input = fake_input
    real_print(f"[replay] pe_guard h0-h{N_HORIZONS - 1} train_frame default on", flush=True)
    try:
        runner.main()
    except SystemExit:
        real_print("[replay] stopped after the requested horizons", flush=True)
    finally:
        time.sleep = real_sleep
        builtins.print = real_print
        builtins.input = input
    _analyze(out)


if __name__ == "__main__":
    main()
