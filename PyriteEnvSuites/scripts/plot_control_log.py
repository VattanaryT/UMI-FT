"""Plot inference control-log horizons: nominal vs virtual targets + observed wrench.

Usage:
  export PYRITE_CONTROL_LOG_FOLDERS=$HOME/data/control_log
  python plot_control_log.py
  python plot_control_log.py --log_subdir temp --robot_id 0
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import zarr


def set_axes_equal(ax):
    """Make 3D axes equal scale so trajectories are not distorted."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range, 1e-3])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def sorted_horizon_keys(log):
    keys = [k for k in log.keys() if k.startswith("horizon_")]
    return sorted(keys, key=lambda k: int(k.split("_", 1)[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log_subdir",
        default="temp",
        help="Subfolder under PYRITE_CONTROL_LOG_FOLDERS (default: temp)",
    )
    parser.add_argument("--robot_id", type=int, default=0)
    parser.add_argument(
        "--exe_horizon",
        type=int,
        default=None,
        help="Executed steps to highlight (default: all steps in the logged action)",
    )
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument(
        "--save_dir",
        default=None,
        help="If set, save each horizon figure here instead of only showing",
    )
    args = parser.parse_args()

    if "PYRITE_CONTROL_LOG_FOLDERS" not in os.environ:
        raise ValueError(
            "Please set the environment variable PYRITE_CONTROL_LOG_FOLDERS"
        )

    control_log_path = os.path.join(
        os.environ["PYRITE_CONTROL_LOG_FOLDERS"], args.log_subdir
    )
    robot_id = args.robot_id
    downsample = max(1, args.downsample)

    print(f"Loading control log from {control_log_path}")
    log = zarr.open(store=zarr.DirectoryStore(path=control_log_path), mode="r")
    horizon_keys = sorted_horizon_keys(log)
    if not horizon_keys:
        raise RuntimeError(f"No horizon_* groups found in {control_log_path}")

    plt.ion()
    fig = plt.figure(figsize=(12, 5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_f = fig.add_subplot(1, 2, 2)
    fig.suptitle("Control log: virtual targets + wrench")

    for horizon_key in horizon_keys:
        horizon_log = log[horizon_key]
        # Current umift_env_runner layout: action/ + obs/
        if "action" in horizon_log:
            action_log = horizon_log["action"]
            obs_log = horizon_log["obs"] if "obs" in horizon_log else {}
        else:
            action_log = horizon_log
            obs_log = horizon_log

        ts_virtual = action_log[f"ts_virtual_targets_{robot_id}"][:]
        ts_nominal = action_log[f"ts_nominal_targets_{robot_id}"][:]
        horizon = ts_virtual.shape[0]
        exe_horizon = args.exe_horizon if args.exe_horizon is not None else horizon
        exe_horizon = min(exe_horizon, horizon)

        print(
            f"{horizon_key}: action_len={horizon}, exe={exe_horizon}, "
            f"vt-nom mean |dx|={np.linalg.norm(ts_virtual[:, :3] - ts_nominal[:, :3], axis=1).mean():.4f} m"
        )

        # --- 3D nominal / virtual targets ---
        ax3d.cla()
        ax3d.plot3D(
            ts_nominal[::downsample, 0],
            ts_nominal[::downsample, 1],
            ts_nominal[::downsample, 2],
            color="gold",
            marker="o",
            markersize=5,
            linewidth=2,
            label="nominal",
        )
        ax3d.plot3D(
            ts_virtual[::downsample, 0],
            ts_virtual[::downsample, 1],
            ts_virtual[::downsample, 2],
            color="orangered",
            marker="o",
            markersize=5,
            linewidth=2,
            label="virtual",
        )
        for i in range(0, exe_horizon, downsample):
            ax3d.plot3D(
                [ts_nominal[i, 0], ts_virtual[i, 0]],
                [ts_nominal[i, 1], ts_virtual[i, 1]],
                [ts_nominal[i, 2], ts_virtual[i, 2]],
                color="0.5",
                linewidth=1,
            )
        # current eef from obs, if present
        eef_key = f"robot{robot_id}_eef_pos"
        if eef_key in obs_log:
            eef = np.asarray(obs_log[eef_key][-1]).reshape(-1)
            ax3d.plot3D(
                [eef[0]], [eef[1]], [eef[2]],
                color="black",
                marker="o",
                markersize=8,
                label="eef",
            )

        ax3d.set_title(f"{horizon_key}")
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")
        ax3d.legend(loc="upper left", fontsize=8)
        set_axes_equal(ax3d)

        # --- wrench / force ---
        ax_f.cla()
        wrench_key = f"robot{robot_id}_eef_wrench_left"
        if wrench_key in obs_log:
            wrench = np.asarray(obs_log[wrench_key][:])
            ts_key = f"wrench_time_stamps_{robot_id}"
            if ts_key in obs_log:
                t = np.asarray(obs_log[ts_key][:]).reshape(-1)
                t = t - t[0]
                xlabel = "time (s)"
            else:
                t = np.arange(wrench.shape[0])
                xlabel = "sample"
            for i, label in enumerate(("fx", "fy", "fz")):
                ax_f.plot(t, wrench[:, i], linewidth=1.2, label=label)
            ax_f.set_ylabel("force (N)")
            ax_f.set_xlabel(xlabel)
            ax_f.legend(loc="upper right", fontsize=8)
            ax_f.grid(True, alpha=0.3)
            ax_f.set_title(f"{wrench_key}")
        else:
            ax_f.text(0.5, 0.5, f"no {wrench_key}", ha="center", va="center")
            ax_f.set_axis_off()

        fig.canvas.draw()
        plt.pause(0.001)

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            out = os.path.join(args.save_dir, f"{horizon_key}_r{robot_id}.png")
            fig.savefig(out, dpi=120, bbox_inches="tight")
            print(f"  saved {out}")

        c = input("Enter=next horizon, q=quit: ").strip().lower()
        if c == "q":
            break

    print("Done.")


if __name__ == "__main__":
    main()
