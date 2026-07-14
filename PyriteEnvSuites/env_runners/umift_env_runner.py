import os
import time
from typing import Optional, Tuple

import cv2
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

# from PyriteConfig.tasks.common.common_type_conversions import raw_to_obs
from PyriteConfig.tasks.umift.umift_type_conversions import raw_to_obs_inference
from PyriteUtility.common import GracefulKiller
from PyriteUtility.planning_control.mpc import ModelPredictiveController
from PyriteUtility.plotting.matplotlib_helpers import set_axes_equal
from PyriteUtility.pytorch_utils.model_io import load_policy
from PyriteUtility.spatial_math import spatial_utilities as su

from PyriteEnvSuites.envs.task.manip_server_layup_env import ManipServerLayupEnv
# from PyriteEnvSuites.envs.task.manip_server_umift_env import ManipServerUMIFTEnv

from PyriteEnvSuites.utils.env_utils import (
    pose9g1_to_traj,
    pose9pose9s1a2_to_traj,
    ts_to_js_traj,
)


def flush_matplotlib_figure(fig, pause_s: float = 0.1) -> None:
    """Force a Matplotlib GUI backend to repaint before blocking on stdin."""
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(pause_s)


if "PYRITE_CHECKPOINT_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_CHECKPOINT_FOLDERS")
if "PYRITE_HARDWARE_CONFIG_FOLDERS" not in os.environ:
    raise ValueError(
        "Please set the environment variable PYRITE_HARDWARE_CONFIG_FOLDERS"
    )

if "PYRITE_CONTROL_LOG_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_CONTROL_LOG_FOLDERS")

checkpoint_folder_path = os.environ.get("PYRITE_CHECKPOINT_FOLDERS")
hardware_config_folder_path = os.environ.get("PYRITE_HARDWARE_CONFIG_FOLDERS")
control_log_folder_path = os.environ.get("PYRITE_CONTROL_LOG_FOLDERS")


def resolve_realsense_serial(hardware_config_path: str) -> Optional[str]:
    """Resolve D405 serial for pyrealsense2 (env var overrides yaml)."""
    serial = os.environ.get("UMIFT_REALSENSE_SERIAL")
    if serial:
        return serial
    with open(hardware_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("realsense_serial")


def log_horizon(
    log,
    horizon_count: int,
    id_list: list,
    exe_horizon: int,
    outputs_ts_targets,
    outputs_ts_nominal_targets,
    outputs_ts_stiffnesses,
    timestamps,
    action_start_time_s: float,
    obs_raw: dict,
    env,
    controller: ModelPredictiveController,
    obs_task: dict,
    shape_meta: dict,
    log_policy_vision: bool,
) -> None:
    """Write one MPC horizon to the episode control log (executed slice only)."""
    horizon_log = log.create_group(f"horizon_{horizon_count}")
    exe = min(exe_horizon, outputs_ts_targets[0].shape[0])

    meta = horizon_log.create_group("meta")
    meta.attrs["hardware_time_s"] = float(env.current_hardware_time_s)
    meta.attrs["horizon_index"] = int(horizon_count)
    meta.attrs["exe_horizon"] = int(exe)

    action_log = horizon_log.create_group("action")
    for robot_id in id_list:
        action_log.create_dataset(
            f"ts_virtual_targets_{robot_id}",
            data=outputs_ts_targets[robot_id][:exe],
        )
        action_log.create_dataset(
            f"ts_nominal_targets_{robot_id}",
            data=outputs_ts_nominal_targets[robot_id][:exe],
        )
        action_log.create_dataset(
            f"ts_stiffnesses_{robot_id}",
            data=outputs_ts_stiffnesses[robot_id][:, : 6 * exe],
        )
    action_log.create_dataset(
        "timestamps_s",
        data=timestamps[:exe] + action_start_time_s,
    )

    obs_policy_group = horizon_log.create_group("obs_policy")
    obs_sampled = controller.sample_sparse_obs(obs_task["obs"])
    obs_policy, _ = controller.preprocess_policy_obs(
        obs_sampled, ignore_rgb=not log_policy_vision
    )
    vision_keys = {"rgb_0", "ultrawide_0", "depth_0"}
    for key in shape_meta["sample"]["obs"]["sparse"]:
        if key not in obs_policy:
            continue
        if not log_policy_vision and key in vision_keys:
            continue
        obs_policy_group.create_dataset(key, data=obs_policy[key])

    obs_inference = horizon_log.create_group("obs_inference")
    obs_inference.attrs["hardware_time_s"] = float(env.current_hardware_time_s)
    for robot_id in id_list:
        obs_inference.create_dataset(
            f"ts_pose_fb_{robot_id}",
            data=obs_raw[f"ts_pose_fb_{robot_id}"][-1],
        )


def main():
    control_para = {
        # D405 @ 15 Hz for layup stack; iPhone/GoPro legacy path uses 0.0167 (~60 Hz).
        "raw_time_step_s": 1.0 / 15.0,
        "slow_down_factor": 5,  # slow down policy execution by this factor
        "sparse_execution_horizon": 4, # 
        "max_duration_s": 3500,
        "gripper_action_scale": 1000,
        "use_audio": False,
        "gripper_lookahead_steps": 0,  # how many steps to look ahead for gripper action
        "set_wrench_fb_to_zero": False,
        "test_nominal_target": False,  # False to follow virtual target
        "test_nominal_target_stiffness": -1,  # -1, for a 'no force case'. 1500, for a quick test without force.
        "fix_orientation": False,
        "enable_control_log": True,
        "log_policy_vision": True,
        "pausing_mode": False,  # matplotlib debug only; independent of logging
        "device": "cuda",
    }
    pipeline_para = {
        # Layup: D405 rgb (manip_server) + depth (pyrealsense2) + UR5e wrist FT.
        "use_layup_env": True,
        "iphone_params": {
            "socket_ip": "192.168.2.18",
            "ports": [5555],
        },
        "ckpt_path": "/2026.07.11_16.32.44_UMIFT-ZUCCHINI-AH3-DS8_UMIFT-ZUCCHINI/checkpoints/epoch=0270-train_loss=0.019.ckpt",
        "hardware_config_path": hardware_config_folder_path
        + "/ur5e_wrist_ft.yaml",
        "audio_device_id": [5],  # only needed for audio input
        # D405 depth @ 15 Hz (no iPhone 60 Hz / 30 Hz correction).
        "depth_downsample_correction": 1,
        # UR RTDE ~500 Hz at inference vs wild collection ~200 Hz pose / ~28 Hz wrench.
        "robot_downsample_correction": 500.0 / 200.0,
        "wrench_downsample_correction": 500.0 / 28.0,
    }
    if not pipeline_para["use_layup_env"]:
        control_para["raw_time_step_s"] = 0.0167
        pipeline_para["hardware_config_path"] = (
            hardware_config_folder_path + "/right_arm_coinft.yaml"
        )
        pipeline_para["depth_downsample_correction"] = 2
        pipeline_para["robot_downsample_correction"] = 1
        pipeline_para["wrench_downsample_correction"] = 1
    verbose = 1
    episode_id = 0

    def get_real_obs_resolution(shape_meta: dict) -> Tuple[int, int]:
        out_res = None
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            shape = attr.get("shape")
            if type == "rgb":
                co, ho, wo = shape
                if out_res is None:
                    out_res = (wo, ho)
                assert out_res == (wo, ho)
        return out_res

    def printOrNot(verbose, *args):
        if verbose >= 0:
            print(f"[Episode {episode_id}] ", *args)

    vbs_h1 = verbose + 1  # verbosity for header
    vbs_h2 = verbose  # verbosity for sub-header
    # vbs_p = verbose - 1  # verbosity for paragraph

    # load policy
    print("Loading policy: ", checkpoint_folder_path + pipeline_para["ckpt_path"])
    device = torch.device(control_para["device"])
    policy, shape_meta = load_policy(
        checkpoint_folder_path + pipeline_para["ckpt_path"], device
    )

    # image size
    (image_width, image_height) = get_real_obs_resolution(shape_meta)

    vision_key = []
    has_rgb = "rgb_0" in shape_meta["sample"]["obs"]["sparse"]
    has_ultrawide_rgb = "ultrawide_0" in shape_meta["sample"]["obs"]["sparse"]
    has_depth = "depth_0" in shape_meta["sample"]["obs"]["sparse"]

    if has_rgb:
        vision_key.append("rgb_0")
    if has_ultrawide_rgb:
        vision_key.append("ultrawide_0")
    if has_depth:
        vision_key.append("depth_0")

    # Validate vision modality constraints
    if has_rgb == 0 and has_ultrawide_rgb == 0:
        raise ValueError(
            "Either 'rgb_0' or 'ultrawide_0' must be present in shape_meta."
        )
    if len(vision_key) > 2:
        raise ValueError("More than two vision modalities present in shape_meta.")

    print("====================Remove this later==========================")
    from omegaconf import OmegaConf

    # Temporarily disable struct mode to allow deletions
    OmegaConf.set_struct(shape_meta["raw"], False)

    # ... perform deletions ...
    valid_keys = set(vision_key)
    raw_vision_keys = [
        k for k, v in shape_meta["raw"].items() if v.get("type") == "rgb"
    ]
    for key in raw_vision_keys:
        if key not in valid_keys:
            print(f"[DEBUG] Removing invalid vision key from raw: {key}")
            del shape_meta["raw"][key]

    # Optionally re-enable struct mode to protect against future accidental changes
    OmegaConf.set_struct(shape_meta["raw"], True)
    print("====================Remove this later==========================")

    print(f"Vision Keys for id:0 in this shape meta: {vision_key}")

    # Both rgb and ultrawide stream at 60Hz, so we'll just use either one. rgb, if it exists.
    rgb_key = next(k for k in vision_key if k in ["rgb_0", "ultrawide_0"])

    # ultrawide also streams at 60Hz so no mapping is needed during inference. Just use the same downsampling and horizon.
    rgb_query_size = (
        shape_meta["sample"]["obs"]["sparse"][rgb_key]["horizon"] - 1
    ) * shape_meta["sample"]["obs"]["sparse"][rgb_key]["down_sample_steps"] + 1

    # Depth physically streams at 30Hz. The shape_meta parameters are all defined based on a (effectively) 60Hz stream.
    # Correction for depth downsampling
    if "depth_0" in shape_meta["sample"]["obs"]["sparse"]:
        shape_meta["sample"]["obs"]["sparse"]["depth_0"]["down_sample_steps"] //= (
            pipeline_para["depth_downsample_correction"]
        )

        depth_query_size = (
            shape_meta["sample"]["obs"]["sparse"]["depth_0"]["horizon"] - 1
        ) * shape_meta["sample"]["obs"]["sparse"]["depth_0"]["down_sample_steps"] + 1

        depth_clip = shape_meta["obs"]["depth_0"]["clip"]
    else:
        depth_query_size = None

    # Layup: scale down_sample_steps so MPC striding on ~500 Hz RTDE buffers
    # matches wild-collection effective rates (~200 Hz pose, ~28 Hz wrench).
    if pipeline_para["use_layup_env"]:
        OmegaConf.set_struct(shape_meta["sample"]["obs"]["sparse"], False)
        robot_corr = pipeline_para["robot_downsample_correction"]
        for robot_key in ("robot0_eef_pos", "robot0_eef_rot_axis_angle"):
            if robot_key in shape_meta["sample"]["obs"]["sparse"]:
                ds = int(
                    shape_meta["sample"]["obs"]["sparse"][robot_key][
                        "down_sample_steps"
                    ]
                )
                new_ds = max(1, round(ds * robot_corr))
                shape_meta["sample"]["obs"]["sparse"][robot_key][
                    "down_sample_steps"
                ] = new_ds
                print(
                    f"[layup] {robot_key} down_sample_steps: {ds} -> {new_ds} "
                    f"(x{robot_corr:.2f})"
                )
        wrench_corr = pipeline_para["wrench_downsample_correction"]
        for wrench_key in ("robot0_eef_wrench_left", "robot0_eef_wrench_right"):
            if wrench_key in shape_meta["sample"]["obs"]["sparse"]:
                ds = int(
                    shape_meta["sample"]["obs"]["sparse"][wrench_key][
                        "down_sample_steps"
                    ]
                )
                new_ds = max(1, round(ds * wrench_corr))
                shape_meta["sample"]["obs"]["sparse"][wrench_key][
                    "down_sample_steps"
                ] = new_ds
                print(
                    f"[layup] {wrench_key} down_sample_steps: {ds} -> {new_ds} "
                    f"(x{wrench_corr:.2f})"
                )

    ts_pose_query_size = (
        shape_meta["sample"]["obs"]["sparse"]["robot0_eef_pos"]["horizon"] - 1
    ) * shape_meta["sample"]["obs"]["sparse"]["robot0_eef_pos"]["down_sample_steps"] + 1

    if "robot0_eef_wrench_left" in shape_meta["sample"]["obs"]["sparse"]:
        wrench_query_size = (
            shape_meta["sample"]["obs"]["sparse"]["robot0_eef_wrench_left"]["horizon"]
            - 1
        ) * shape_meta["sample"]["obs"]["sparse"]["robot0_eef_wrench_left"][
            "down_sample_steps"
        ] + 1
    else:
        wrench_query_size = 0

    if "robot0_gripper" in shape_meta["sample"]["obs"]["sparse"]:
        eoat_query_size = (
            shape_meta["sample"]["obs"]["sparse"]["robot0_gripper"]["horizon"] - 1
        ) * shape_meta["sample"]["obs"]["sparse"]["robot0_gripper"][
            "down_sample_steps"
        ] + 1
    else:
        eoat_query_size = 0

    query_sizes_sparse = {
        "rgb": rgb_query_size,
        "depth": depth_query_size,
        "ts_pose_fb": ts_pose_query_size,
        "wrench": wrench_query_size,
        "eoat": eoat_query_size,
    }

    if "mic_0" in shape_meta["sample"]["obs"]["sparse"]:
        mic_query_size = (
            shape_meta["sample"]["obs"]["sparse"]["mic_0"]["horizon"] - 1
        ) * shape_meta["sample"]["obs"]["sparse"]["mic_0"]["down_sample_steps"] + 1
        query_sizes_sparse["mic"] = mic_query_size

    query_sizes = {
        "sparse": query_sizes_sparse,
    }

    if (
        control_para["test_nominal_target"]
        and control_para["test_nominal_target_stiffness"] < 0
    ):
        n_af = 0
    else:
        n_af = 3

    audio_params = None
    if control_para["use_audio"]:
        audio_params = {
            "audio_device_id": pipeline_para["audio_device_id"],
        }

    if pipeline_para["use_layup_env"]:
        realsense_serial = resolve_realsense_serial(
            pipeline_para["hardware_config_path"]
        )
        print(
            f"[layup] pyrealsense2 D405 serial: "
            f"{realsense_serial or '(first connected device)'}"
        )
        env = ManipServerLayupEnv(
            camera_res_hw=(image_height, image_width),
            hardware_config_path=pipeline_para["hardware_config_path"],
            query_sizes=query_sizes,
            compliant_dimensionality=n_af,
            depth_clip=depth_clip if depth_query_size is not None else 0.5,
            vision_key=vision_key,
            realsense_serial=realsense_serial,
        )
    else:
        env = ManipServerUMIFTEnv(
            camera_res_hw=(image_height, image_width),
            hardware_config_path=pipeline_para["hardware_config_path"],
            query_sizes=query_sizes,
            audio_params=audio_params,
            compliant_dimensionality=n_af,
            iphone_params=pipeline_para["iphone_params"],
            vision_key=vision_key,
        )

    env.reset()

    p_timestep_s = control_para["raw_time_step_s"]

    sparse_action_down_sample_steps = shape_meta["sample"]["action"]["sparse"][
        "down_sample_steps"
    ]
    sparse_action_horizon = shape_meta["sample"]["action"]["sparse"]["horizon"]
    # sparse_action_horizon_s = (
    #     sparse_action_horizon * sparse_action_down_sample_steps * p_timestep_s
    # )
    sparse_execution_horizon = (
        sparse_action_down_sample_steps * control_para["sparse_execution_horizon"]
    )
    sparse_action_timesteps_s = (
        np.arange(0, sparse_action_horizon)
        * sparse_action_down_sample_steps
        * p_timestep_s
        * control_para["slow_down_factor"]
    )

    action_type = None
    id_list = [0]
    if shape_meta["action"]["shape"][0] == 9:
        action_type = "pose9"
        id_list = [0]
        action_to_trajectory = ts_to_js_traj
    elif shape_meta["action"]["shape"][0] == 10:
        action_type = "pose9g1"
        id_list = [0]
        action_to_trajectory = pose9g1_to_traj
    elif shape_meta["action"]["shape"][0] == 21:
        action_type = "pose9pose9s1a2"
        id_list = [0]
        action_to_trajectory = pose9pose9s1a2_to_traj
    elif shape_meta["action"]["shape"][0] == 42:
        action_type = "pose9pose9s1a2"
        id_list = [0, 1]
        action_to_trajectory = pose9pose9s1a2_to_traj
    else:
        raise RuntimeError("unsupported")

    printOrNot(vbs_h2, "Creating MPC.")
    controller = ModelPredictiveController(
        shape_meta=shape_meta,
        id_list=id_list,
        policy=policy,
        action_to_trajectory=action_to_trajectory,
        execution_horizon=sparse_execution_horizon,
        fix_orientation=control_para["fix_orientation"],
    )
    controller.set_time_offset(env.current_hardware_time_s)

    # timestep_idx = 0
    # stiffness = None
    episode_initial_time_s = env.current_hardware_time_s
    execution_duration_s = (
        sparse_execution_horizon * p_timestep_s * control_para["slow_down_factor"]
    )
    printOrNot(vbs_h2, "Starting main loop.")

    printOrNot(vbs_h2, "Calibrating robot wrench.")
    env.calibrate_robot_wrench(NSamples=100)
    time.sleep(2)
    obs = env.get_observation_from_buffer()
    w = np.mean(obs["wrench_left_0"][-100:], axis=0)  # or wrench_left_0 in layup
    print("wrench: ", w)

    if control_para["pausing_mode"]:
        plt.ion()  # to run GUI event loop
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        # x = np.linspace(-0.02, 0.2, 20)
        # y = np.linspace(-0.1, 0.1, 20)
        # z = np.linspace(-0.1, 0.1, 20)
        ax.set_title("Target and virtual target")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
        flush_matplotlib_figure(fig)

    horizon_count = 0
    log = None

    # plot the rgb/depth stream, confirm it is correct
    time.sleep(2)

    while True:
        obs_raw = env.get_observation_from_buffer()

        images_to_stack = []
        for key in vision_key:
            im = obs_raw[key][-1]

            if key == "depth_0":
                # Normalize float16 depth to uint8 for display
                depth_raw = im.squeeze()  # (H, W) if needed
                depth_norm = depth_raw / depth_clip  # Normalize to [0, 1]
                im = (depth_norm * 255).astype(np.uint8)
                # im = np.repeat(depth_uint8[:, :, np.newaxis], 3, axis=2)
            else:
                # Convert RGB to BGR for OpenCV
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

            images_to_stack.append(im)

        stacked = np.hstack(images_to_stack)
        cv2.imshow("image", stacked)

        key = cv2.waitKey(10)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    ts_pose_initial = []
    for id in id_list:
        ts_pose_initial.append(obs_raw[f"ts_pose_fb_{id}"][-1])
    #########################################
    # main loop starts
    #########################################
    while True:
        printOrNot(vbs_h1, "New episode. Episode ID: ", episode_id)
        input("Press Enter to start the episode.")

        # env.start_saving_data_for_a_new_episode(pipeline_para["episode_data_save_path"])
        env.start_saving_data_for_a_new_episode()

        log = None
        horizon_count = 0
        if control_para["enable_control_log"]:
            episode_log_path = os.path.join(
                control_log_folder_path, f"episode_{int(time.time())}"
            )
            log_store = zarr.DirectoryStore(path=episode_log_path)
            log = zarr.open(store=log_store, mode="w")
            log.attrs["sparse_execution_horizon"] = int(
                control_para["sparse_execution_horizon"]
            )
            log.attrs["episode_id"] = int(episode_id)
            log.attrs["episode_data_folder"] = env.get_episode_folder()
            log.attrs["ckpt_path"] = pipeline_para["ckpt_path"]
            log.attrs["hardware_config_path"] = pipeline_para["hardware_config_path"]
            log.attrs["log_policy_vision"] = bool(control_para["log_policy_vision"])
            printOrNot(vbs_h1, "Control log:", episode_log_path)

        episode_initial_time_s = env.current_hardware_time_s
        killer = GracefulKiller()
        while not killer.kill_now:
            horizon_initial_time_s = env.current_hardware_time_s
            printOrNot(vbs_h1, "Starting new horizon at ", horizon_initial_time_s)

            obs_raw = env.get_observation_from_buffer()

            # # plot the rgb image
            # if len(id_list) == 1:
            #     rgb = obs_raw["rgb_0"][-1]
            # else:
            #     rgb = np.vstack([obs_raw[f"rgb_{i}"][-1] for i in id_list])
            # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # cv2.imshow("image", bgr)
            # cv2.waitKey(10)

            # visualize all vision modalities
            images_to_stack = []
            for key in vision_key:
                im = obs_raw[key][-1]

                if key == "depth_0":
                    # Normalize float16 depth to uint8 for display
                    depth_raw = im.squeeze()  # (H, W) if needed
                    depth_norm = depth_raw / depth_clip  # Normalize to [0, 1]
                    im = (depth_norm * 255).astype(np.uint8)
                else:
                    # Convert RGB to BGR for OpenCV
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

                images_to_stack.append(im)

            stacked = np.hstack(images_to_stack)
            cv2.imshow("image", stacked)
            cv2.waitKey(10)

            obs_task = dict()

            # hack for UMIFT
            for id in id_list:
                if f"rgb_{id}" not in obs_raw:
                    # sampling done based on rgb_time_stamps during training. During inference, rgb time and ultrawide time are the same as both sample at 60Hz
                    obs_raw[f"rgb_time_stamps_{id}"] = obs_raw[
                        f"ultrawide_time_stamps_{id}"
                    ]
            # import pdb; pdb.set_trace()

            raw_to_obs_inference(obs_raw, obs_task, shape_meta)

            # hacks for UMIFT
            for id in id_list:
                # Since training data and policy use meter, but real gripper hardware uses mm
                if f"robot{id}_gripper" in obs_task["obs"]:
                    obs_task["obs"][f"robot{id}_gripper"] = (
                        obs_task["obs"][f"robot{id}_gripper"]
                        / control_para["gripper_action_scale"]
                    )
                # Training real data seems to have a mismatch in rgb channels
                # obs_task["obs"][f"rgb_{id}"] = obs_task["obs"][f"rgb_{id}"][:,[2,1,0],:,:]

                if control_para["set_wrench_fb_to_zero"]:
                    if f"robot{id}_eef_wrench_left" in obs_task["obs"]:
                        obs_task["obs"][f"robot{id}_eef_wrench_left"] = np.zeros_like(
                            obs_task["obs"][f"robot{id}_eef_wrench_left"]
                        )
                        obs_task["obs"][f"robot{id}_eef_wrench_right"] = np.zeros_like(
                            obs_task["obs"][f"robot{id}_eef_wrench_right"]
                        )

            assert (
                action_type == "pose9pose9s1a2"
                or action_type == "pose9"
                or action_type == "pose9g1"
            ), f"Unsupported action type: {action_type}"

            # Run inference
            if action_type == "pose9":
                action_sparse_target_mats = controller.compute_one_horizon_action(
                    obs_task["obs"], device
                )
                action_sparse_vt_mats = action_sparse_target_mats.copy()
                action_stiffnesses = None
                action_eoat = None
            elif action_type == "pose9g1":
                (
                    action_sparse_target_mats,
                    action_eoat_pos,
                ) = controller.compute_one_horizon_action(obs_task["obs"], device)
                action_sparse_vt_mats = action_sparse_target_mats.copy()
                action_stiffnesses = None
                action_eoat = [np.array] * len(id_list)
                for id in id_list:
                    action_eoat[id] = np.zeros([sparse_action_horizon, 2])
                    action_eoat[id][:, 0] = np.squeeze(action_eoat_pos[id])
            elif action_type == "pose9pose9s1a2":
                (
                    action_sparse_target_mats,
                    action_sparse_vt_mats,
                    action_stiffnesses,
                    action_eoat,
                ) = controller.compute_one_horizon_action(obs_task["obs"], device)

            outputs_ts_nominal_targets = [np.array] * len(id_list)
            outputs_ts_targets = [np.array] * len(id_list)
            outputs_ts_stiffnesses = [np.array] * len(id_list)

            if action_type == "pose9pose9s1a2" or action_type == "pose9g1":
                for id in id_list:
                    # hack for UMIFT, since training data and policy use meter, but real gripper hardware uses mm
                    print("Gripper action before scaling: ", action_eoat[id][:, 0])
                    print("Gripper force: ", action_eoat[id][:, 1])
                    action_eoat[id][:, 0] *= control_para["gripper_action_scale"]
                    if control_para["gripper_lookahead_steps"] > 0:
                        action_eoat[id][
                            : -control_para["gripper_lookahead_steps"], :
                        ] = action_eoat[id][
                            control_para["gripper_lookahead_steps"] :, :
                        ]

            if action_stiffnesses is not None:
                print("action_stiffness: ", action_stiffnesses[0])
            # print("action_target_z: ", action_sparse_target_mats[0][:, 2, 3])
            # print("action_virtual_target_z: ", action_sparse_vt_mats[0][:, 2, 3])
            print(
                "action_delta z: ",
                action_sparse_vt_mats[0][:, 2, 3]
                - action_sparse_target_mats[0][:, 2, 3],
            )
            outputs_eoat = action_eoat
            # decode stiffness matrix
            for id in id_list:
                SE3_TW = su.SE3_inv(su.pose7_to_SE3(obs_raw[f"ts_pose_fb_{id}"][-1]))
                ts_targets_nominal = su.SE3_to_pose7(
                    action_sparse_target_mats[id].reshape([-1, 4, 4])
                )
                ts_targets_virtual = su.SE3_to_pose7(
                    action_sparse_vt_mats[id].reshape([-1, 4, 4])
                )

                ts_stiffnesses = np.zeros([6, 6 * ts_targets_virtual.shape[0]])
                for i in range(ts_targets_virtual.shape[0]):
                    SE3_target = action_sparse_target_mats[id][i].reshape([4, 4])
                    SE3_virtual_target = action_sparse_vt_mats[id][i].reshape([4, 4])

                    # stiffness: 1. convert vt to tool frame
                    SE3_TVt = SE3_TW @ SE3_virtual_target
                    SE3_Ttarget = SE3_TW @ SE3_target

                    # stiffness: 2. compute stiffness matrix in the tool frame
                    compliance_direction_tool = (
                        SE3_TVt[:3, 3] - SE3_Ttarget[:3, 3]
                    ).reshape(3)

                    if np.linalg.norm(compliance_direction_tool) < 0.001:  #
                        compliance_direction_tool = np.array([1.0, 0.0, 0.0])

                    compliance_direction_tool /= np.linalg.norm(
                        compliance_direction_tool
                    )
                    X = compliance_direction_tool
                    Y = np.cross(X, np.array([0, 0, 1]))
                    Y /= np.linalg.norm(Y)
                    Z = np.cross(X, Y)

                    default_stiffness = 5000
                    default_stiffness_rot = 100
                    stiffness_matrix_full = np.eye(6) * default_stiffness_rot

                    if action_stiffnesses is None:
                        target_stiffness = default_stiffness
                    else:
                        target_stiffness = action_stiffnesses[id][i]

                    M = np.diag(
                        [target_stiffness, default_stiffness, default_stiffness]
                    )
                    S = np.array([X, Y, Z]).T
                    stiffness_matrix = S @ M @ np.linalg.inv(S)
                    stiffness_matrix_full[:3, :3] = stiffness_matrix

                    if (
                        control_para["test_nominal_target"]
                        and control_para["test_nominal_target_stiffness"] > 0
                    ):
                        stiffness_matrix_full[:3, :3] = (
                            np.eye(3) * control_para["test_nominal_target_stiffness"]
                        )

                    ts_stiffnesses[:, 6 * i : 6 * i + 6] = stiffness_matrix_full

                outputs_ts_nominal_targets[id] = ts_targets_nominal
                if control_para["test_nominal_target"]:
                    outputs_ts_targets[id] = ts_targets_nominal
                else:
                    outputs_ts_targets[id] = ts_targets_virtual
                outputs_ts_stiffnesses[id] = ts_stiffnesses

            # the "now" when the observation is taken
            action_start_time_s = obs_raw["robot_time_stamps_0"][-1]
            timestamps = sparse_action_timesteps_s

            if control_para["pausing_mode"]:
                # plot the actions for this horizon using matplotlib
                ax.cla()
                for i, id in enumerate(id_list):
                    # Label only the first robot to avoid duplicate legend entries.
                    label = lambda name: name if i == 0 else None
                    ax.plot3D(
                        action_sparse_target_mats[id][..., 0, 3],
                        action_sparse_target_mats[id][..., 1, 3],
                        action_sparse_target_mats[id][..., 2, 3],
                        color="red",
                        marker="o",
                        markersize=3,
                        label=label("Nominal target"),
                    )

                    ax.plot3D(
                        action_sparse_vt_mats[id][..., 0, 3],
                        action_sparse_vt_mats[id][..., 1, 3],
                        action_sparse_vt_mats[id][..., 2, 3],
                        color="blue",
                        marker="o",
                        markersize=3,
                        label=label("Virtual target"),
                    )

                    ax.plot3D(
                        action_sparse_target_mats[id][
                            : control_para["sparse_execution_horizon"], 0, 3
                        ],
                        action_sparse_target_mats[id][
                            : control_para["sparse_execution_horizon"], 1, 3
                        ],
                        action_sparse_target_mats[id][
                            : control_para["sparse_execution_horizon"], 2, 3
                        ],
                        color="yellow",
                        linewidth=3,
                        marker="o",
                        markersize=4,
                        label=label("Executed nominal"),
                    )

                    ax.plot3D(
                        action_sparse_vt_mats[id][
                            : control_para["sparse_execution_horizon"], 0, 3
                        ],
                        action_sparse_vt_mats[id][
                            : control_para["sparse_execution_horizon"], 1, 3
                        ],
                        action_sparse_vt_mats[id][
                            : control_para["sparse_execution_horizon"], 2, 3
                        ],
                        color="green",
                        linewidth=3,
                        marker="o",
                        markersize=4,
                        label=label("Executed virtual"),
                    )

                    ax.plot3D(
                        obs_raw[f"ts_pose_fb_{id}"][-1][0],
                        obs_raw[f"ts_pose_fb_{id}"][-1][1],
                        obs_raw[f"ts_pose_fb_{id}"][-1][2],
                        color="black",
                        marker="o",
                        markersize=8,
                        label=label("Current TCP"),
                    )

                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.legend(loc="best")

                set_axes_equal(ax)

                # Process the GUI event loop before input() blocks the main thread.
                flush_matplotlib_figure(fig)

                input("Press Enter to start executing the plotted actions.")

            if control_para["enable_control_log"] and log is not None:
                log_horizon(
                    log=log,
                    horizon_count=horizon_count,
                    id_list=id_list,
                    exe_horizon=control_para["sparse_execution_horizon"],
                    outputs_ts_targets=outputs_ts_targets,
                    outputs_ts_nominal_targets=outputs_ts_nominal_targets,
                    outputs_ts_stiffnesses=outputs_ts_stiffnesses,
                    timestamps=timestamps,
                    action_start_time_s=action_start_time_s,
                    obs_raw=obs_raw,
                    env=env,
                    controller=controller,
                    obs_task=obs_task,
                    shape_meta=shape_meta,
                    log_policy_vision=control_para["log_policy_vision"],
                )

            # cut the action and only keep the execution horizon
            for id in id_list:
                outputs_ts_targets[id] = outputs_ts_targets[id][
                    : control_para["sparse_execution_horizon"], :
                ]
                outputs_ts_stiffnesses[id] = outputs_ts_stiffnesses[id][
                    :, : 6 * control_para["sparse_execution_horizon"]
                ]
                if outputs_eoat is not None:
                    outputs_eoat[id] = outputs_eoat[id][
                        : control_para["sparse_execution_horizon"], :
                    ]
            timestamps = timestamps[: control_para["sparse_execution_horizon"]]
            action_start_time_s = env.current_hardware_time_s

            if len(id_list) == 1:
                outputs_ts_targets = outputs_ts_targets[0].T  # N x 7 to 7 x N
                outputs_ts_stiffnesses = outputs_ts_stiffnesses[0]
                if outputs_eoat is not None:
                    outputs_eoat = outputs_eoat[0].T  # N x 2 to 2 x N
            else:
                outputs_ts_targets = np.hstack(
                    outputs_ts_targets
                ).T  # 2 x N x 7 to 14 x N
                if outputs_eoat is not None:
                    outputs_eoat = np.hstack(outputs_eoat).T  # 2 x N x 2 to 4 x N
                outputs_ts_stiffnesses = np.vstack(
                    outputs_ts_stiffnesses
                )  # 6 x (6xN) to 12 x (6xN)

            env.schedule_controls(
                pose7_cmd=outputs_ts_targets,
                eoat_cmd=outputs_eoat,
                stiffness_matrices_6x6=outputs_ts_stiffnesses,
                timestamps=(timestamps + action_start_time_s) * 1000,
            )

            # # log the truncated action
            # horizon_log = log.create_group(f"horizon_{horizon_count}")
            # horizon_log.create_dataset("ts_targets", data=outputs_ts_targets)
            # horizon_log.create_dataset("timestamps", data=timestamps * 1000)
            # horizon_log.create_dataset("ts_stiffnesses", data=outputs_ts_stiffnesses)
            # horizon_log.create_dataset(
            #     "timestamps_s", data=timestamps + action_start_time_s
            # )
            horizon_count += 1

            if control_para["pausing_mode"]:
                c = input("Press Enter to start the next horizon. q to quit.")
                if c == "q":
                    break

            time_s = env.current_hardware_time_s
            sleep_duration_s = horizon_initial_time_s + execution_duration_s - time_s

            printOrNot(vbs_h1, "sleep_duration_s", sleep_duration_s)
            time.sleep(max(0, sleep_duration_s))

            if not control_para["pausing_mode"]:
                # only check duration when not in pausing mode
                if time_s - episode_initial_time_s > control_para["max_duration_s"]:
                    break

        printOrNot(vbs_h1, "End of episode.")
        episode_id += 1

        env.stop_saving_data()

        print("Options:")
        print("     c: continue to next episode.")
        print("     r: reset to default pose, then continue.")
        print("     b: reset to default pose, then quit the program.")
        print("     others: quit the program.")
        c = input("Please select an option: ")
        if c == "r" or c == "b":
            print("Resetting to default pose.")
            obs_raw = env.get_observation_from_buffer()
            N = 100
            duration_s = 5
            sample_indices = np.linspace(0, 1, N)
            timestamps = sample_indices * duration_s
            homing_ts_targets = np.zeros([7 * len(id_list), N])
            for id in id_list:
                ts_pose_fb = obs_raw[f"ts_pose_fb_{id}"][-1]

                pose7_waypoints = su.pose7_interp(
                    ts_pose_fb, ts_pose_initial[id], sample_indices
                )
                homing_ts_targets[0 + id * 7 : 7 + id * 7, :] = pose7_waypoints.T

            time_now_s = env.current_hardware_time_s
            env.schedule_controls(
                pose7_cmd=homing_ts_targets,
                timestamps=(timestamps + time_now_s) * 1000,
            )
        elif c == "c":
            pass
        else:
            print("Quitting the program.")
            break

        if c == "r":
            input("Press Enter to continue.")

        if c == "b":
            input("Press Enter to quit program.")
            break

        print("Continuing to execution.")
    # env.save_whole_episode()
    # env.save_video()
    env.cleanup()
    # end of episode
    # # plot


if __name__ == "__main__":
    main()
