import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
UMI_FT_ROOT = SCRIPT_DIR.parents[1]
PYRITE_UTILITY_ROOT = UMI_FT_ROOT / "PyriteUtility"
if str(PYRITE_UTILITY_ROOT) not in sys.path:
    sys.path.insert(0, str(PYRITE_UTILITY_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.spatial.transform import Rotation
from PyriteUtility.spatial_math import spatial_utilities as su


parser = argparse.ArgumentParser(
    description="Plot UMIFT dataset with ultrawide and depth."
)
parser.add_argument(
    "--absolute-pose",
    action="store_true",
    help="Plot pose in the global frame instead of relative to the initial frame.",
)
args = parser.parse_args()
use_absolute_pose = args.absolute_pose


if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")

dataset_folder_path = os.environ.get("PYRITE_DATASET_FOLDERS")
dataset_path = os.environ.get(
    "UMIFT_DATASET_PATH",
    "/mnt/data/umidata/ft/straight_convex_data_v3/acp_replay_buffer.zarr",
)

print("Loading dataset from: ", dataset_path)
# load the zarr dataset from the path
# ‘r’ means read only (must exist); ‘r+’ means read/write (must exist); ‘a’ means read/write (create if doesn’t exist); ‘w’ means create (overwrite if exists); ‘w-’ means create (fail if exists).
buffer = zarr.open(dataset_path, mode="r+")

# select which episode to visualize    
ep = "episode_58"
print("Loading episode: ", ep)
print(buffer.tree())

# preprocess pose data into relative pose
pose_IG = np.array(buffer["data"][ep]["ts_pose_fb_0"]) # This is the pose of iPhone w.r.t to the global frame
pose_IG0 = pose_IG[0] # This is the initial pose of iPhone w.r.t the global frame
SE3_IG0 = su.pose7_to_SE3(pose_IG0) # T matrix of initial pose
SE3_G0I = su.SE3_inv(SE3_IG0)
SE3_IG = np.zeros((pose_IG.shape[0], 4, 4))
SE3_G0G = np.zeros((pose_IG.shape[0], 4, 4))
for i in range(pose_IG.shape[0]):
    SE3_IGi = su.pose7_to_SE3(pose_IG[i])
    SE3_IG[i] = SE3_IGi
    SE3_G0G[i] = SE3_G0I @ SE3_IGi


def pose_arrays_from_se3(se3_array):
    position = np.array(se3_array[..., :3, 3])
    rotation_deg = Rotation.from_matrix(se3_array[..., :3, :3]).as_euler(
        "xyz", degrees=True
    )
    return position, rotation_deg


def as_1d_time(time_arr):
    return np.asarray(time_arr, dtype=np.float64).reshape(-1)


def nearest_indices(source_time, query_time):
    """For each query timestamp, index of the closest sample in source_time."""
    source_time = as_1d_time(source_time)
    query_time = as_1d_time(query_time)
    idx = np.searchsorted(source_time, query_time, side="left")
    idx = np.clip(idx, 0, len(source_time) - 1)
    idx_left = np.clip(idx - 1, 0, len(source_time) - 1)
    pick_left = np.abs(source_time[idx_left] - query_time) <= np.abs(
        source_time[idx] - query_time
    )
    return np.where(pick_left, idx_left, idx)


data1_relative, data1_rot_relative = pose_arrays_from_se3(SE3_G0G)
data1_absolute, data1_rot_absolute = pose_arrays_from_se3(SE3_IG)


def get_pose_data():
    if use_absolute_pose:
        return (
            data1_absolute,
            data1_rot_absolute,
            "Pose (Position, global)",
            "Pose (Rotation, global, deg)",
        )
    return (
        data1_relative,
        data1_rot_relative,
        "Pose (Position, rel. frame 0)",
        "Pose (Rotation, rel. frame 0, deg)",
    )


print("finished processing pose data")

# compute net wrench
wrench_left = np.array(buffer["data"][ep]["wrench_left_0"][..., :3])
wrench_right = np.array(buffer["data"][ep]["wrench_right_0"][..., :3])
wrench_TCP = wrench_left + wrench_right
grasp_force = (-wrench_left[:, 0] + wrench_right[:, 0]) /2

print("finished processing wrench data")

images = np.array(buffer["data"][ep]["rgb_0"])
ultrawide = np.array(buffer["data"][ep]["ultrawide_0"])
depth = np.array(buffer["data"][ep]["depth_0"])
print("got images")

data2 = np.array(buffer["data"][ep]["gripper_0"])
print("got data2")

print("got data3")
data4 = np.array(buffer["data"][ep]["stiffness_0"])
# change data4 from (N,) to (N, 1)
data4 = data4.reshape(-1, 1)

print("finish distributing data for plotting")

time_images = as_1d_time(buffer["data"][ep]["rgb_time_stamps_0"])
time_ultrawide = buffer["data"][ep]["ultrawide_time_stamps_0"]
time_depth = buffer["data"][ep]["depth_time_stamps_0"]

robot_time = as_1d_time(buffer["data"][ep]["robot_time_stamps_0"])
wrench_time = as_1d_time(buffer["data"][ep]["wrench_time_stamps_left_0"])

# Plot all scalar streams on the RGB timeline so the cursor matches the video frame.
plot_time = time_images
robot_at_rgb = nearest_indices(robot_time, plot_time)
wrench_at_rgb = nearest_indices(wrench_time, plot_time)

time_data1 = plot_time
time_data2 = plot_time
time_data3 = plot_time
time_data4 = plot_time

def get_pose_data_at_rgb():
    data1_full, data1_rot_full, title_pos, title_rot = get_pose_data()
    return (
        data1_full[robot_at_rgb],
        data1_rot_full[robot_at_rgb],
        title_pos,
        title_rot,
    )


data1, data1_rot, title_pos, title_rot = get_pose_data_at_rgb()
data2 = data2[robot_at_rgb]
data4 = data4[robot_at_rgb]
data3 = wrench_TCP[wrench_at_rgb, 0:3]
print("got data1")

map_to_uw_idx = np.array(buffer["data"][ep]["map_to_uw_idx_0"])
map_to_d_idx = np.array(buffer["data"][ep]["map_to_d_idx_0"])

titles = [
    title_pos,
    title_rot,
    "Gripper Width",
    "Tool Center Point Force",
    "Stiffness",
]
print("Finished loading data")


# Initialize figure and axes
fig = plt.figure(figsize=(12, 20))
gs = fig.add_gridspec(6, 3, height_ratios=[3, 1, 1, 1, 1, 1], hspace=0.8, wspace=0.3)

ax_rgb = fig.add_subplot(gs[0, 0])
ax_uw = fig.add_subplot(gs[0, 1])
ax_depth = fig.add_subplot(gs[0, 2])
ax1 = fig.add_subplot(gs[1, :])
ax1_rot = fig.add_subplot(gs[2, :])
ax2 = fig.add_subplot(gs[3, :])
ax3 = fig.add_subplot(gs[4, :])
ax4 = fig.add_subplot(gs[5, :])


# ax_rgb, ax_uw, ax_depth, ax1, ax2, ax3, ax4 = axes
ax_rgb.axis("off")
ax_uw.axis("off")
ax_depth.axis("off")


# Set plot title
fig.suptitle(
    f"Plotting for {ep}. "
    f"Pose: {'global' if use_absolute_pose else 'rel. frame 0'} "
    f"(left/right: rgb frame, a: toggle pose frame)"
)

# Plot data
plot_xlim = (float(plot_time[0]), float(plot_time[-1]))
pos_lines = ax1.plot(time_data1, data1, '.-')
rot_lines = ax1_rot.plot(time_data1, data1_rot, '.-')
ax2.plot(time_data2, data2, '.-')
ax3.plot(time_data3, data3, '.-')
ax4.plot(time_data4, data4, '.-')
time_axes = [ax1, ax1_rot, ax2, ax3, ax4]
for ax in time_axes:
    ax.set_xlim(plot_xlim)

# Set titles
ax1.set_title(titles[0])
ax1_rot.set_title(titles[1])
ax2.set_title(titles[2])
ax3.set_title(titles[3])
ax4.set_title(titles[4])

# set legends for plot 1 and 3
ax1.legend(["x", "y", "z"])
ax1_rot.legend(["roll (x)", "pitch (y)", "yaw (z)"])
ax3.legend(["x", "y", "z"])

# Shared vertical cursor (indexed to RGB frames)
cursor_frame_idx = 0
cursor_time = float(plot_time[cursor_frame_idx])
cursor_line = [ax1.axvline(cursor_time, color='r', linestyle='--'),
               ax1_rot.axvline(cursor_time, color='r', linestyle='--'),
               ax2.axvline(cursor_time, color='r', linestyle='--'),
               ax3.axvline(cursor_time, color='r', linestyle='--'),
               ax4.axvline(cursor_time, color='r', linestyle='--')]

def refresh_pose_plots():
    data1, data1_rot, title_pos, title_rot = get_pose_data_at_rgb()
    for i, line in enumerate(pos_lines):
        line.set_ydata(data1[:, i])
    for i, line in enumerate(rot_lines):
        line.set_ydata(data1_rot[:, i])
    ax1.set_title(title_pos)
    ax1_rot.set_title(title_rot)
    ax1.relim()
    ax1.autoscale_view()
    ax1_rot.relim()
    ax1_rot.autoscale_view()
    for ax in time_axes:
        ax.set_xlim(plot_xlim)
    fig.suptitle(
        f"Plotting for {ep}. "
        f"Pose: {'global' if use_absolute_pose else 'rel. frame 0'} "
        f"(left/right: rgb frame, a: toggle pose frame)"
    )

# Function to update the displayed image
def update_image(frame_idx):
    frame_idx = int(np.clip(frame_idx, 0, len(plot_time) - 1))
    cursor_time = float(plot_time[frame_idx])
    for line in cursor_line:
        line.set_xdata([cursor_time])

    print("Checking timestamps")
    print("frame:", frame_idx, "time_images:", time_images[frame_idx])
    print('time ultrawide: ', time_ultrawide[map_to_uw_idx[frame_idx]])
    print('time depth: ', time_depth[map_to_d_idx[frame_idx]])
    print("Grasp Force: ", data3[frame_idx])
    
    # RGB
    ax_rgb.clear()
    ax_rgb.imshow(images[frame_idx])
    ax_rgb.set_title("RGB Image")
    ax_rgb.axis("off")

    # Ultrawide
    ax_uw.clear()
    uw_idx = map_to_uw_idx[frame_idx]
    uw_frame = ultrawide[uw_idx]
    if uw_frame.ndim == 4 and uw_frame.shape[0] == 1:
        uw_frame = uw_frame.squeeze(0)
    ax_uw.imshow(uw_frame)
    ax_uw.set_title("Ultrawide Image")
    ax_uw.axis("off")

    # Depth
    ax_depth.clear()
    d_idx = map_to_d_idx[frame_idx]
    d_frame = depth[d_idx]
    if d_frame.ndim == 4 and d_frame.shape[-1] == 3:
        d_frame = d_frame[..., 0]  # Take one channel for grayscale
        d_frame = d_frame.squeeze(0)
    ax_depth.imshow(d_frame, cmap='gray', vmin=0.0, vmax=0.5)
    ax_depth.set_title("Depth Image")
    ax_depth.axis("off")

    fig.canvas.draw_idle()



update_image(cursor_frame_idx)

# Keyboard event handler
def on_key(event):
    global cursor_frame_idx, use_absolute_pose
    if event.key == "right":
        cursor_frame_idx += 1
    elif event.key == "left":
        cursor_frame_idx -= 1
    elif event.key == "a":
        use_absolute_pose = not use_absolute_pose
        refresh_pose_plots()
        fig.canvas.draw_idle()
        return
    else:
        return

    cursor_frame_idx = int(np.clip(cursor_frame_idx, 0, len(plot_time) - 1))
    print("Showing frame:", cursor_frame_idx, "time:", float(plot_time[cursor_frame_idx]))
    print("Image shape:", images.shape)
    update_image(cursor_frame_idx)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
