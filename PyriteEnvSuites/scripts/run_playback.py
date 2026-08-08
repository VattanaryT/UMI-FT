import os
import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import PyriteUtility.spatial_math.spatial_utilities as su
import zarr
from einops import rearrange
from hardware_interfaces.workcell.table_top_manip.python import (
    manip_server_pybind as ms,
)
from PyriteUtility.common import GracefulKiller
from PyriteUtility.computer_vision.computer_vision_utility import get_image_transform
from PyriteUtility.computer_vision.imagecodecs_numcodecs import register_codecs
from PyriteUtility.plotting.matplotlib_helpers import set_axes_equal
from PyriteUtility.umi_utils.usb_util import reset_all_elgato_devices

register_codecs()

if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")


### PLOTTING FUNCTIONS ###
# TODO (Chuer): reorg them in PyriteUtility
def flatten_timestamps(timestamps):
    """Return a 1D float array from zarr timestamps (often shape N or Nx1)."""
    return np.asarray(timestamps, dtype=float).reshape(-1)


def plot_trajectory(ax, pose_SE3_WT, **plot_kwargs):
    """Plot EE path as a single 3D line through waypoint positions."""
    positions = pose_SE3_WT[:, :3, 3]
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        **plot_kwargs,
    )


def plot_trajectory_colored(
    ax,
    pose_SE3_WT,
    end_idx,
    timestamps=None,
    cmap_name="viridis",
    linewidth=2,
    alpha=1.0,
):
    """Plot trajectory segments color-coded by time, visible up to end_idx."""
    positions = pose_SE3_WT[:end_idx, :3, 3]
    if len(positions) < 2:
        return None

    segments = np.stack([positions[:-1], positions[1:]], axis=1)
    if timestamps is not None:
        t = flatten_timestamps(timestamps)[:end_idx]
        t_vals = 0.5 * (t[:-1] + t[1:])
        t_norm = plt.Normalize(t[0], t[-1])
    else:
        t_vals = np.arange(len(segments))
        t_norm = plt.Normalize(0, len(pose_SE3_WT) - 1)

    colors = cm.get_cmap(cmap_name)(t_norm(t_vals))
    colors[:, 3] = alpha
    lc = Line3DCollection(segments, colors=colors, linewidths=linewidth)
    ax.add_collection3d(lc)
    return t_norm


def plot_camera_frames_sparse(
    ax, pose_SE3_WT, end_idx, step=25, axis_length=0.015, linewidth=1.5
):
    """Draw RGB axis triads at a fixed stride within the visible prefix."""
    if end_idx <= 0:
        return
    frame_indices = list(range(0, end_idx, step))
    if frame_indices[-1] != end_idx - 1:
        frame_indices.append(end_idx - 1)
    for idx in frame_indices:
        plot_camera_frame(
            ax, pose_SE3_WT[idx], axis_length=axis_length, linewidth=linewidth
        )


def show_interactive_trajectory(
    pose_SE3_WT,
    timestamps=None,
    title="Playback EE Trajectory",
    axis_frame_step=25,
    initial_fraction=0.02,
    reveal_fraction_step=0.02,
):
    """
    Interactive 3D trajectory viewer.

    Right/left arrow keys reveal or hide the trajectory over time.
    """
    n_points = len(pose_SE3_WT)
    reveal_end_idx = max(2, int(n_points * initial_fraction))
    reveal_step = max(1, int(n_points * reveal_fraction_step))

    if timestamps is not None:
        timestamps = flatten_timestamps(timestamps)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colorbar = None

    if timestamps is not None:
        t_norm = plt.Normalize(timestamps[0], timestamps[-1])
        colorbar_label = "Time (s)"
    else:
        t_norm = plt.Normalize(0, n_points - 1)
        colorbar_label = "Waypoint index"

    def format_title():
        if timestamps is not None and reveal_end_idx > 0:
            t_end = timestamps[min(reveal_end_idx - 1, n_points - 1)]
            t_start = timestamps[0]
            progress = f"t={t_end - t_start:.2f}s / {timestamps[-1] - timestamps[0]:.2f}s"
        else:
            progress = f"{reveal_end_idx}/{n_points} waypoints"
        return (
            f"{title}\n{progress} "
            "(→ reveal, ← hide, close window or Enter in terminal to continue)"
        )

    def redraw():
        nonlocal colorbar
        ax.cla()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(format_title())

        plot_trajectory_colored(
            ax,
            pose_SE3_WT,
            reveal_end_idx,
            timestamps=timestamps,
            linewidth=2,
        )
        plot_camera_frames_sparse(
            ax,
            pose_SE3_WT,
            reveal_end_idx,
            step=axis_frame_step,
            axis_length=0.015,
        )
        set_axes_equal(ax)

        if colorbar is None:
            sm = cm.ScalarMappable(cmap="viridis", norm=t_norm)
            sm.set_array([])
            colorbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
            colorbar.set_label(colorbar_label)
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal reveal_end_idx
        if event.key == "right":
            reveal_end_idx = min(n_points, reveal_end_idx + reveal_step)
        elif event.key == "left":
            reveal_end_idx = max(2, reveal_end_idx - reveal_step)
        else:
            return
        redraw()

    redraw()
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=False)
    input("Press Enter to continue...")
    plt.close(fig)


def plot_camera_frame(ax, pose, axis_length=0.01, linewidth=2):
    # Extract translation
    translation = pose[0:3, 3]

    # Define the local axes in the camera frame
    axes = np.array(
        [
            [1, 0, 0],  # X-axis
            [0, 1, 0],  # Y-axis
            [0, 0, 1],
        ]
    )  # Z-axis

    # Draw each axis
    for i, color in zip(range(3), ["r", "g", "b"]):
        # Calculate the end point of the axis
        end_point = translation + axis_length * (pose[0:3, 0:3] @ axes[i])
        # Plot the axis line
        ax.plot(
            [translation[0], end_point[0]],
            [translation[1], end_point[1]],
            [translation[2], end_point[2]],
            color=color,
            linewidth=linewidth,
        )


def plot_images(img_A, img_B, opacity_A=0.6):
    """
    Plots three panels: img_A, img_B, and an overlay of both with
    img_A at 70% opacity and img_B at 30% opacity.

    Args:
        img_A: (1, H, W, 3) NumPy array (observation)
        img_B: (1, H, W, 3) NumPy array (data)
        opacity_A: float, opacity for image A
    """
    # Remove the batch dimension (1, H, W, 3) -> (H, W, 3)
    img_A = img_A.squeeze(0).astype(np.uint8)
    img_B = img_B.squeeze(0)

    H, W = img_A.shape[:2]
    wi, hi = img_B.shape[:2]

    # img tranform data img to obs img shape and crop
    tf = get_image_transform(input_res=(hi, wi), output_res=(W, H), bgr_to_rgb=True)
    img_B = tf(img_B)

    # Ensure images are the same shape
    assert (
        img_A.shape == img_B.shape
    ), f"Images must have the same dimensions, img_A.shape: {img_A.shape}, img_B.shape: {img_B.shape}"

    overlay = opacity_A * img_A + (1 - opacity_A) * img_B
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Plot the images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_A)
    axes[0].set_title("Obs")
    axes[0].axis("off")

    axes[1].imshow(img_B)
    axes[1].set_title("Data")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay Initial Pose for Obs with Data")
    axes[2].axis("off")

    plt.show()


dataset_folder_path = os.environ.get("PYRITE_DATASET_FOLDERS")
# (CHUER): TBCHANGED
dataset_path = (
    dataset_folder_path + "/stringer_data_v2_full/acp_replay_buffer.zarr"
)
id_list = [0]
print("dataset_path:", dataset_path)
buffer = zarr.open(dataset_path, mode="r")

input_res = (224, 224)
if "PYRITE_HARDWARE_CONFIG_FOLDERS" not in os.environ:
    raise ValueError(
        "Please set the environment variable PYRITE_HARDWARE_CONFIG_FOLDERS"
    )
config_folders = os.environ.get("PYRITE_HARDWARE_CONFIG_FOLDERS")

reset_all_elgato_devices()
hardware_config_path = config_folders + "/ur5e_wrist_ft.yaml"

Tr = np.eye(6)
n_af = 0
default_stiffness = [5000, 5000, 5000, 100, 100, 100]

# manipulation server
server = ms.ManipServer()

if not server.initialize(hardware_config_path):
    raise RuntimeError("Failed to initialize hardware server.")
server.set_high_level_maintain_position()
if server.is_bimanual():
    id_list = [0, 1]
else:
    id_list = [0]

default_stiffness = np.diag(default_stiffness)

rgb_row_combined_buffer = []
rgb_buffer = []
output_rgb_buffer = []
for id in id_list:
    server.set_force_controlled_axis(Tr, n_af, id)
    server.set_stiffness_matrix(default_stiffness, id)

print("[Main] Waiting for hardware server to be ready...")
while not server.is_ready():
    time.sleep(0.1)
print("[Main] Hardware server is ready. Now reading data...")

# read data
ep = "episode_34"
data = buffer["data"][ep]

# get initial rgb frame and align it with initial frame of data episode being played back
# query_size_sparse_rgb = 11
# obs_rgb_raw = server.get_camera_rgb(query_size_sparse_rgb, 0)
# obs_rgb_reshape = rearrange(
#     obs_rgb_raw, "(c h) (n w)->n h w c", c=3, n=query_size_sparse_rgb
# )
# plot_images(img_A=obs_rgb_reshape[0:1], img_B=data["rgb_0"][0:1])
input("Press ENTER to continue ... ")

for key in data.keys():
    print("data keys: ", key)

print(data["ts_pose_fb_0"].shape)

# get current pose
pose_W0 = np.squeeze(server.get_pose(1, 0))
SE3_W0 = su.pose7_to_SE3(pose_W0)

# read the first pose wp
pose_X0 = data["ts_pose_fb_0"][0]
SE3_0X = su.SE3_inv(su.pose7_to_SE3(pose_X0))

# read and convert all data to world
N = data["ts_pose_fb_0"].shape[0]
pose_WT = np.zeros((7, N))
pose_SE3_WT = np.zeros((N, 4, 4))

for i in range(N):
    pose_XT = data["ts_pose_fb_0"][i]
    SE3_XT = su.pose7_to_SE3(pose_XT)
    SE3_0T = SE3_0X @ SE3_XT
    SE3_WT = SE3_W0 @ SE3_0T
    pose_WT[:, i] = su.SE3_to_pose7(SE3_WT)
    pose_SE3_WT[i] = SE3_WT

# read all Gripper data
Ng = data["gripper_0"].shape[0]
assert Ng == N
gripper = data["gripper_0"]
print("[Main] Finished reading data. Now playing back...")

# PLOTTING poses to play back
plt.ion()
robot_timestamps = flatten_timestamps(data["robot_time_stamps_0"])
axis_frame_step = max(1, N // 30)
show_interactive_trajectory(
    pose_SE3_WT,
    timestamps=robot_timestamps,
    title="Playback EE Trajectory",
    axis_frame_step=axis_frame_step,
    initial_fraction=0.02,
    reveal_fraction_step=0.02,
)

# plot gripper
plt.ion()
plt.plot(gripper, marker="o", linestyle="-")
plt.show()
plt.title("Gripper")
input("Press Enter to continue...")
plt.close()

# Create Plot for Short Horizon Play Back
plt.ion()
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.set_title("Playback EE Trajectory: Current Horizon")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# start playing back
dN = 200  # number of waypoints to send at a time
i = 0

while i < N:
    killer = GracefulKiller()
    end_id = min(i + dN, N)
    pose_cmd = pose_WT[:, i:end_id]  # 7 x dN
    gp_pos_cmd = gripper[i:end_id] * 1000
    gp_f_cmd = np.zeros((len(gp_pos_cmd), 1))
    gp_cmd = np.concatenate((gp_pos_cmd, gp_f_cmd), axis=1).transpose()  # 2 x dN

    # Plotting current window trajectory
    ax.cla()
    ax.set_title("Playback EE Trajectory: Current Horizon")
    plot_trajectory(
        ax, pose_SE3_WT, color="lightgray", linewidth=0.5, alpha=0.4, label="full"
    )
    plot_trajectory(
        ax, pose_SE3_WT[i:end_id], color="b", linewidth=2, label="horizon"
    )
    plot_camera_frame(ax, pose_SE3_WT[i], axis_length=0.02)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_axes_equal(ax)
    plt.pause(0.001)

    i += dN
    pose_fb = np.squeeze(server.get_pose(1, 0))
    print("gp_cmd:", gp_cmd)

    input("Press Enter to execute...")
    timenow_ms = server.get_timestamp_now_ms()
    time_cmd = timenow_ms + 100 + np.arange(dN) * 60
    server.schedule_waypoints(pose_cmd, time_cmd, 0)
    if server.has_eoat():
        server.schedule_eoat_waypoints(gp_cmd, time_cmd, 0)
    time.sleep(0.1)
