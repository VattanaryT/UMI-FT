# This script does the following:
# 1. Compute the virtual target pose and stiffness based on the force/torque sensor data, add them to the zarr file
# 2. Optionally plot the virtual target and the target in 3D space
# 3. If ts_pose_command is not available, the script will populate it with ts_pose_fb.
# 4. offset time stamps to start from zero

import concurrent.futures
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import zarr
from tqdm import tqdm

from PyriteUtility.planning_control import compliance_helpers as ch
from PyriteUtility.plotting.matplotlib_helpers import set_axes_equal
from PyriteUtility.spatial_math import spatial_utilities as su
from PyriteUtility.data_pipeline.wrench_baseline_utils import (
    check_virtual_target_push_direction,
    subtract_wrench_baseline,
    time_based_moving_average,
)

if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")
dataset_folder_path = os.environ.get("PYRITE_DATASET_FOLDERS")

# Config for umift (single robot)
dataset_path = os.environ.get(
    "UMIFT_DATASET_PATH",
    "/home/boeing/umi_ft_ws/straight_convex_data/acp_replay_buffer.zarr",
)
id_list = [0]

wrench_moving_average_window_duration_s = 1.0  # target temporal window (seconds)

# Baseline removal: "first_n" for demos starting in free space; "session_tare" for contact-at-start
baseline_mode = "first_n"  # "first_n" | "session_tare"
baseline_duration_s = 0.3
session_offset_path = None  # required when baseline_mode == "session_tare"
# Set False (or UMIFT_SUBTRACT_WRENCH_BASELINE=0) to keep collection-time tare only
flag_subtract_wrench_baseline = os.environ.get(
    "UMIFT_SUBTRACT_WRENCH_BASELINE", "1"
).lower() not in ("0", "false", "no", "off")

# Default +1 keeps stringer / FT300-S demos unchanged. Teleop LeRobot FT needs -1
# (inverted push_alignment); override via UMIFT_WRENCH_SIGN_MULTIPLIER=-1.
wrench_sign_multiplier = float(os.environ.get("UMIFT_WRENCH_SIGN_MULTIPLIER", "1.0"))

# ‘r’ means read only (must exist); ‘r+’ means read/write (must exist); ‘a’ means read/write (create if doesn’t exist); ‘w’ means create (overwrite if exists); ‘w-’ means create (fail if exists).
buffer = zarr.open(dataset_path, mode="r+")

print(
    "wrench baseline subtraction:",
    "enabled" if flag_subtract_wrench_baseline else "disabled",
)
print(f"wrench_sign_multiplier: {wrench_sign_multiplier}")


num_of_process = 3  # 5
flag_plot = False
fin_every_n = 25  # 50

depth_clip = 0.5

apply_mask = False
clip_depth = True
zero_depth = True  # fully zero depth (ablation; overrides clip_depth)
zero_ultrawide = True  # converters write zeros; rewrite existing aliased ultrawide on reprocess
convert_BGR_to_RGB = False
flag_mute_gripper = False
flag_mute_wrench = False


print("Applying mask to the dataset")

mask_path = os.path.join(os.path.dirname(__file__), "umift_mask.json")
with open(mask_path, "r") as f:
    mask_data = json.load(f)

mask_pts = np.array(mask_data["umift_uw_mask"], dtype=np.int32)  # Extract mask points
mask_resolution = tuple(mask_data["mask_uw_resolution"])  # (W, H)

print(f"Mask resolution: {mask_resolution}")

if flag_mute_gripper:
    # there is no point of having wrench if the gripper is muted.
    flag_mute_wrench = True

print(
    f"[Modality Ablation] Is the gripper width information being muted: {flag_mute_gripper}"
)
print(f"[Modality Ablation] Is the wrench information being muted: {flag_mute_wrench}")


stiffness_estimation_para = {
    # penetration estimator
    "k_max": 3000,  # 1cm 50N
    "k_min": 200,  # 1cm 2.5N
    "f_low": 2.0,  # 0.5,
    # "f_low": 1.5,
    "f_high": 8,  # 5,
    "dim": 3,
    "characteristic_length": 0.02,
    "vel_tol": 999.002,  # vel larger than this will trigger stiffness adjustment
}

flag_real = True
if "real" in dataset_path:
    flag_real = True
    print("flag_real set!")

if flag_plot:
    assert num_of_process == 1, "Plotting is not supported for multi-process"


def process_episode(ep, ep_data, id_list):
    def _write_corrected_wrench_to_episode(
        ep_data,
        robot_id,
        wrench_corrected,
        wrench_left_original,
        wrench_right_original,
    ):
        """
        Persist corrected wrench into episode keys used by policy observations.

        The baseline/sign correction is computed on the summed wrench. For the
        common single-FT setup (right wrench is zero), write the corrected sum to
        left directly. If both sides are non-zero, split the correction delta
        evenly to preserve left+right consistency without changing total wrench.
        """
        wrench_original_sum = wrench_left_original + wrench_right_original
        correction_delta = wrench_corrected - wrench_original_sum

        if np.allclose(wrench_right_original, 0.0):
            wrench_left_corrected = wrench_corrected
            wrench_right_corrected = wrench_right_original
        else:
            wrench_left_corrected = wrench_left_original + 0.5 * correction_delta
            wrench_right_corrected = wrench_right_original + 0.5 * correction_delta

        wrench_concat_corrected = np.hstack((wrench_left_corrected, wrench_right_corrected))

        ep_data[f"wrench_left_{robot_id}"] = wrench_left_corrected
        ep_data[f"wrench_right_{robot_id}"] = wrench_right_corrected
        ep_data[f"wrench_concat_{robot_id}"] = wrench_concat_corrected

        if f"wrench_left_coinft_{robot_id}" in ep_data:
            ep_data[f"wrench_left_coinft_{robot_id}"] = wrench_left_corrected
        if f"wrench_right_coinft_{robot_id}" in ep_data:
            ep_data[f"wrench_right_coinft_{robot_id}"] = wrench_right_corrected
        if f"wrench_concat_coinft_{robot_id}" in ep_data:
            ep_data[f"wrench_concat_coinft_{robot_id}"] = wrench_concat_corrected

    for key in ep_data.keys():
        print(key)

    for id in id_list:
        print(f"Processing episode {ep}, id {id}: ")

        if flag_mute_gripper:
            ep_data[f"gripper_{id}"][:] = 0

        if flag_mute_wrench:
            ep_data[f"wrench_concat_{id}"][:] = 0
            ep_data[f"wrench_concat_coinft_{id}"][:] = 0
            ep_data[f"wrench_left_{id}"][:] = 0
            ep_data[f"wrench_left_coinft_{id}"][:] = 0
            ep_data[f"wrench_right_{id}"][:] = 0
            ep_data[f"wrench_right_coinft_{id}"][:] = 0

        # if RGB processing has not been done on this episode, do it
        if convert_BGR_to_RGB:
            rgb_conversion_flag = f"rgb_converted_{id}"
            if rgb_conversion_flag in ep_data.keys():
                print(
                    f"Skipping RGB conversion for episode {ep}, id {id} (already processed)"
                )
            else:
                print(f"Converting BGR to RGB for episode {ep}, id {id}")
                rgb_images = np.array(ep_data["rgb_0"])
                rgb_images = np.array(
                    [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in rgb_images]
                )
                ep_data["rgb_0"] = rgb_images
                ep_data[rgb_conversion_flag] = np.array([1])

        if zero_ultrawide:
            uw_images = np.array(ep_data["ultrawide_0"])
            ep_data["ultrawide_0"] = np.zeros_like(uw_images)
        elif apply_mask:
            images = np.array(ep_data["ultrawide_0"])
            # create a mask where the masked area is 0 and others are all 255 (0xFF)
            mask = np.ones_like(images[0], dtype=np.uint8) * 255
            cv2.fillPoly(mask, [mask_pts], (0, 0, 0))

            # apply the mask to the images
            images = np.array([cv2.bitwise_and(img, mask) for img in images])

            ep_data["ultrawide_0"] = images

        if zero_depth:
            depth_images = np.array(ep_data["depth_0"])
            ep_data["depth_0"] = np.zeros_like(depth_images)
        elif clip_depth:
            depth_images = np.array(ep_data["depth_0"])
            depth_images_clipped = np.clip(depth_images, a_min=0.0, a_max=depth_clip)
            ep_data["depth_0"] = depth_images_clipped

        ts_pose_fb = ep_data[f"ts_pose_fb_{id}"]
        wrench_left = np.array(ep_data[f"wrench_left_{id}"])
        wrench_right = np.array(ep_data[f"wrench_right_{id}"])

        wrench = wrench_left + wrench_right

        # make time stamps start from zero

        time_offsets = []
        for id in id_list:
            time_offsets.append(ep_data[f"rgb_time_stamps_{id}"][0])
            time_offsets.append(ep_data[f"ultrawide_time_stamps_{id}"][0])
            time_offsets.append(ep_data[f"depth_time_stamps_{id}"][0])
            time_offsets.append(ep_data[f"robot_time_stamps_{id}"][0])
            time_offsets.append(ep_data[f"wrench_time_stamps_left_{id}"][0])
            time_offsets.append(ep_data[f"wrench_time_stamps_right_{id}"][0])
        print("Time offsets: ", time_offsets)
        time_offset = np.min(time_offsets)
        for id in id_list:
            ep_data[f"wrench_time_stamps_left_{id}"] -= time_offset
            ep_data[f"wrench_time_stamps_right_{id}"] -= time_offset
            ep_data[f"robot_time_stamps_{id}"] -= time_offset
            ep_data[f"rgb_time_stamps_{id}"] -= time_offset
            ep_data[f"ultrawide_time_stamps_{id}"] -= time_offset
            ep_data[f"depth_time_stamps_{id}"] -= time_offset
            if f"gripper_time_stamps_{id}" in ep_data.keys():
                ep_data[f"gripper_time_stamps_{id}"] -= time_offset

        wrench_time_stamps = (
            np.array(ep_data[f"wrench_time_stamps_left_{id}"])
            + np.array(ep_data[f"wrench_time_stamps_right_{id}"])
        ) / 2
        robot_time_stamps = ep_data[f"robot_time_stamps_{id}"]
        # rgb_time_stamps = ep_data[f"rgb_time_stamps_{id}"]
        wrench_moving_average = np.zeros_like(wrench)

        if flag_subtract_wrench_baseline:
            wrench, wrench_offset = subtract_wrench_baseline(
                wrench,
                wrench_time_stamps,
                baseline_mode=baseline_mode,
                baseline_duration_s=baseline_duration_s,
                session_offset_path=session_offset_path,
            )
            print("wrench baseline (subtracted): ", wrench_offset)
        else:
            wrench_offset = np.zeros(6, dtype=np.float64)
            print("wrench baseline subtraction disabled; using collection-time tare only")

        if wrench_sign_multiplier != 1.0:
            wrench = wrench * wrench_sign_multiplier

        _write_corrected_wrench_to_episode(
            ep_data=ep_data,
            robot_id=id,
            wrench_corrected=wrench,
            wrench_left_original=wrench_left,
            wrench_right_original=wrench_right,
        )
        print("wrench data overwritten with corrected baseline/sign values")

        print(
            f"Computing moving average (window={wrench_moving_average_window_duration_s}s)"
        )
        wrench_moving_average = time_based_moving_average(
            wrench,
            wrench_time_stamps,
            window_duration_s=wrench_moving_average_window_duration_s,
        )

        if not flag_real:  # for simulation data
            _ft_sensor_pose_fb = ep_data["ft_sensor_pose_fb"]

        num_robot_time_steps = len(robot_time_stamps)

        print("creating virtual target estimator")

        pe = ch.VirtualTargetEstimator(
            stiffness_estimation_para["k_max"],
            stiffness_estimation_para["k_min"],
            stiffness_estimation_para["f_low"],
            stiffness_estimation_para["f_high"],
            stiffness_estimation_para["dim"],
            stiffness_estimation_para["characteristic_length"],
        )

        ts_pose_virtual_target = np.zeros((num_robot_time_steps, 7))
        stiffness = np.zeros(num_robot_time_steps)

        print("Running virtual target estimator")
        pose7_WT = ts_pose_fb
        SE3_WT = su.pose7_to_SE3(pose7_WT)

        # # find the id in wrench_time_stamps where the time is closest to robot_time_stamps
        # print("wrench_time_stamps: ", wrench_time_stamps.shape)
        # print("robot_time_stamps: ", robot_time_stamps.shape)
        wrench_id = np.searchsorted(
            np.squeeze(wrench_time_stamps), np.squeeze(robot_time_stamps)
        )
        wrench_id = np.minimum(wrench_id, len(wrench_time_stamps) - 1)
        if flag_real:
            wrench_T = wrench_moving_average[wrench_id]
        else:
            assert False, "Not implemented. Need to parallelize this part"

        # compute stiffness
        if stiffness_estimation_para["dim"] == 6:
            k, SE3_TC = pe.batch_update(wrench_T)
        else:
            k, pos_TC = pe.batch_update(wrench_T)
            SE3_TC = np.zeros((num_robot_time_steps, 4, 4))
            SE3_TC[:, :4, :4] = np.eye(4)
            SE3_TC[:, :3, 3] = pos_TC
        SE3_WC = SE3_WT @ SE3_TC

        ts_pose_virtual_target = su.SE3_to_pose7(SE3_WC)
        stiffness = np.squeeze(k)

        ep_data[f"ts_pose_virtual_target_{id}"] = ts_pose_virtual_target
        ep_data[f"stiffness_{id}"] = stiffness
        ep_data[f"wrench_time_stamps_{id}"] = wrench_time_stamps

        push_check = check_virtual_target_push_direction(
            ts_pose_fb, ts_pose_virtual_target, wrench_T
        )
        print(f"Virtual target push-direction check: {push_check}")

        if f"ts_pose_command_{id}" not in ep_data:
            ep_data[f"ts_pose_command_{id}"] = ts_pose_fb

        print("Done")

        if flag_plot:
            print("Plotting...")
            plt.ion()  # to run GUI event loop
            _fig = plt.figure()
            ax = plt.axes(projection="3d")
            x = np.linspace(-0.02, 0.2, 20)
            y = np.linspace(-0.1, 0.1, 20)
            z = np.linspace(-0.1, 0.1, 20)
            ax.plot3D(x, y, z, color="blue", marker="o", markersize=3)
            ax.plot3D(x, y, z, color="red", marker="o", markersize=3)
            ax.set_title("Target and virtual target")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.show()

            ax.cla()
            ax.plot3D(
                ts_pose_fb[..., 0],
                ts_pose_fb[..., 1],
                ts_pose_fb[..., 2],
                color="red",
                marker="o",
                markersize=2,
            )

            ax.plot3D(
                ts_pose_virtual_target[..., 0],
                ts_pose_virtual_target[..., 1],
                ts_pose_virtual_target[..., 2],
                color="blue",
                marker="o",
                markersize=2,
            )
            # starting point
            ax.plot3D(
                ts_pose_fb[0][0],
                ts_pose_fb[0][1],
                ts_pose_fb[0][2],
                color="black",
                marker="o",
                markersize=8,
            )

            # fin
            for i in np.arange(0, num_robot_time_steps, fin_every_n):
                ax.plot3D(
                    [ts_pose_fb[i][0], ts_pose_virtual_target[i][0]],
                    [ts_pose_fb[i][1], ts_pose_virtual_target[i][1]],
                    [ts_pose_fb[i][2], ts_pose_virtual_target[i][2]],
                    color="black",
                    marker="o",
                    markersize=2,
                )

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            set_axes_equal(ax)

            plt.draw()
            plt.savefig(f"plot_{ep}.png")

            input("Press Enter to continue...")
        return True


# selected_data = {}
# count = 0
# loop_count = 0
# for key, value in buffer["data"].items():
#     loop_count += 1
#     if loop_count % 7 == 0:
#         selected_data[key] = value
#         print("Selected data: ", key)
#         count += 1
#         if count == 25:
#             break

print("num_of_process: ", num_of_process)
if num_of_process == 1:
    # for ep, ep_data in tqdm(selected_data.items(), desc="Episodes"):
    for ep, ep_data in tqdm(buffer["data"].items(), desc="Episodes"):
        print("ep name: ", ep)
        # pdb.set_trace()
        process_episode(ep, ep_data, id_list)
else:
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_of_process) as executor:
        futures = [
            executor.submit(
                process_episode,
                ep,
                ep_data,
                id_list,
            )
            # for ep, ep_data in tqdm(selected_data.items(), desc="Episodes") ]
            for ep, ep_data in tqdm(buffer["data"].items(), desc="Episodes")
        ]
        for future in concurrent.futures.as_completed(futures):
            if not future.result():
                raise RuntimeError("Multi-processing failed!")


print("All Done!")
