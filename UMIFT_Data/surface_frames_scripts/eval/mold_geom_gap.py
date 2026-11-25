"""Compare live mold_geom to the liftos zarr on the 14 roll pairs.

Also runs ep9's own poses through MoldAwareRuntime.fill_obs_raw.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT / "UMIFT_Data/surface_frames_scripts/eval"),
    str(ROOT / "UMIFT_Data/surface_frames_scripts"),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT),
]
os.environ.setdefault(
    "UMIFT_WORKCELL_EXTRINSICS",
    str(ROOT / "PyriteConfig/tasks/umift/layup_workcell_extrinsics_20261116_orange.yaml"),
)

import view_rollout as vr  # noqa: E402
from causal_foot import CausalFoot  # noqa: E402
from field_v2 import carry_frames, reorient  # noqa: E402
from fixb_labels import pose7_to_Rp, so3_to_rot6d_rows  # noqa: E402
from mold_aware_runtime import MoldAwareRuntime  # noqa: E402
from rot6d import rot6d_rows_to_R  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from swap_climb import RUNS, _demo_table, _horizons, _match, _predict, _twist_obs  # noqa: E402

NAMES = (
    "sd_m",
    "n_x",
    "n_y",
    "n_z",
    "r6_0",
    "r6_1",
    "r6_2",
    "r6_3",
    "r6_4",
    "r6_5",
)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    rel = a.T @ b
    return float(np.degrees(Rotation.from_matrix(rel).magnitude()))


def _window(demo: dict, index: int) -> np.ndarray:
    n = len(demo["sd"])
    pose_index = np.clip(index - np.arange(50, -1, -1) * 16, 0, n - 1)
    return np.concatenate(
        [
            demo["sd"][pose_index, None],
            demo["normal"][pose_index],
            demo["rot6d"][pose_index],
        ],
        axis=1,
    ).astype(np.float32)


def _pca(frames: np.ndarray, axes: np.ndarray, tool: np.ndarray | None = None) -> np.ndarray:
    out = reorient(frames, axes)
    if tool is not None:
        out = carry_frames(out, tool)
    return out


def _pack(sd: np.ndarray, frames: np.ndarray) -> np.ndarray:
    outward = -frames[:, :, 2]
    rot6d = so3_to_rot6d_rows(frames).astype(np.float32)
    sd = np.asarray(sd, dtype=np.float32).reshape(-1, 1)
    return np.concatenate([sd, outward.astype(np.float32), rot6d], axis=1)


def main() -> None:
    demo = vr._demo_path()
    table = _demo_table(demo)
    spec = zarr.open_group(vr.DEMO_ZARR, mode="r").attrs["field_v2_axes"]
    axes = np.asarray(spec["axes"], dtype=np.float64)
    pairs = []
    for name in RUNS:
        for row in _horizons(name):
            hw = row["obs"]
            mold_x = float(np.asarray(hw["robot0_eef_pos"])[-1, 0])
            index, d_twist, d_x = _match(table, _twist_obs(hw), mold_x)
            zarr_g = _window(demo, index)
            live = np.asarray(hw["robot0_mold_geom"], dtype=np.float32)
            pairs.append((name, row["h"], index, live, zarr_g, hw))
    print(f"pairs {len(pairs)}")
    print("name                         hw            demo          diff")
    abs_diff = []
    angles = []
    angles_pca = []
    for name, horizon, index, live, zarr_g, _hw in pairs:
        d = live[-1] - zarr_g[-1]
        abs_diff.append(np.abs(d))
        r_live = rot6d_rows_to_R(live[-1, 4:10]).reshape(3, 3)
        r_z = rot6d_rows_to_R(zarr_g[-1, 4:10]).reshape(3, 3)
        angles.append(_angle(r_live, r_z))
        fixed = _pca(r_live.reshape(1, 3, 3), axes)[0]
        angles_pca.append(_angle(fixed, r_z))
        short = name.replace("2026-11-24_liftos_hw_", "")
        print(f"\n{short} h{horizon}  demo {index}")
        for i, label in enumerate(NAMES):
            print(f"  {label:6s} {live[-1, i]:+9.4f}  {zarr_g[-1, i]:+9.4f}  {d[i]:+9.4f}")
    abs_diff = np.stack(abs_diff)
    print("\nmedian |diff|")
    for i, label in enumerate(NAMES):
        print(f"  {label:6s} {np.median(abs_diff[:, i]):.4f}")
    print(
        f"rot angle live vs zarr med {np.median(angles):.1f}°  "
        f"after PCA reorient {np.median(angles_pca):.1f}°"
    )
    print(f"normal dot med {np.median([float(np.dot(p[3][-1, 1:4], p[4][-1, 1:4])) for p in pairs]):.3f}")

    print("\n[ep9] runner fill_obs_raw vs zarr")
    runtime = MoldAwareRuntime.from_assets("r4_world_fixb_tilt", dry_run=False)
    runtime.reset()
    filled = runtime.fill_obs_raw(
        {
            "ts_pose_fb_0": demo["pose"],
            "robot_time_stamps_0": demo["time"],
        }
    )
    r_run = np.asarray(filled["R"])
    sd_run = np.asarray(filled["sd"]).reshape(-1)
    idxs = sorted({p[2] for p in pairs})
    raw_ang = []
    pca_ang = []
    sd_err = []
    for index in idxs:
        r_z = rot6d_rows_to_R(demo["rot6d"][index]).reshape(3, 3)
        raw_ang.append(_angle(r_run[index], r_z))
        tool = pose7_to_Rp(demo["pose"][index : index + 1])[0][0]
        fixed = _pca(r_run[index : index + 1], axes, tool.reshape(1, 3, 3))[0]
        pca_ang.append(_angle(fixed, r_z))
        z_sd = float(demo["sd"][index])
        sd_err.append(abs(float(sd_run[index]) - z_sd))
    print(
        f"  single-sample rot raw med {np.median(raw_ang):.1f}°  "
        f"pca med {np.median(pca_ang):.1f}°  |sd| med {np.median(sd_err)*1e3:.2f} mm"
    )
    # Full-rate carry on the runner frames, then compare the same indices.
    tool_all = pose7_to_Rp(demo["pose"])[0]
    carried = _pca(r_run, axes, tool_all)
    carried_ang = [_angle(carried[i], rot6d_rows_to_R(demo["rot6d"][i]).reshape(3, 3)) for i in idxs]
    n_run = -r_run[:, :, 2]
    n_dot = [float(np.dot(n_run[i], demo["normal"][i])) for i in idxs]
    print(f"  full-episode carry+pca med {np.median(carried_ang):.2f}°  max {np.max(carried_ang):.2f}°")
    print(f"  runner vs zarr normal dot med {np.median(n_dot):.4f}")

    print("\n[ep9] CausalFoot.track + reorient + carry vs zarr")
    field_npz = Path(
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
    )
    frames_npz = field_npz.with_name("fixb_smooth_frames.npz")
    foot, frame = CausalFoot(field_npz, frames_npz).track(pose7_to_Rp(demo["pose"])[1])
    trained = _pca(frame, axes, tool_all)
    train_ang = [_angle(trained[i], rot6d_rows_to_R(demo["rot6d"][i]).reshape(3, 3)) for i in idxs]
    train_n = -trained[:, :, 2]
    train_dot = [float(np.dot(train_n[i], demo["normal"][i])) for i in idxs]
    train_sd = np.abs(np.sum((pose7_to_Rp(demo["pose"])[1] - foot) * train_n, axis=1) - demo["sd"])
    print(
        f"  rot med {np.median(train_ang):.2f}° max {np.max(train_ang):.2f}°  "
        f"normal dot med {np.median(train_dot):.4f}  |sd| med {np.median(train_sd[idxs])*1e3:.2f} mm"
    )

    print("\n[policy] hardware geom with PCA frames")
    policy, device = vr._policy()
    rows = []
    for name, horizon, index, live, zarr_g, hw in pairs:
        frames = rot6d_rows_to_R(live[:, 4:10])
        tool = rot6d_rows_to_R(np.asarray(hw["robot0_eef_rot_axis_angle"]))
        fixed = _pca(frames, axes, tool)
        # signed distance stays the runner's; normal is -column2, unchanged by reorient
        geom = _pack(live[:, 0], fixed)
        obs = {key: np.asarray(value) for key, value in hw.items()}
        obs["robot0_mold_geom"] = geom
        demo_obs = vr._demo_obs(demo, index)
        pred_fix = _predict(policy, device, obs)
        pred_demo = _predict(policy, device, demo_obs)
        pred_hw = _predict(policy, device, hw)
        rec = {
            "fix": (float(np.median(pred_fix[:6, 1])), float(np.median(pred_fix[:6, 2]) * 1e3)),
            "demo": (float(np.median(pred_demo[:6, 1])), float(np.median(pred_demo[:6, 2]) * 1e3)),
            "hw": (float(np.median(pred_hw[:6, 1])), float(np.median(pred_hw[:6, 2]) * 1e3)),
            "geom_deg": float(np.median([_angle(fixed[t], rot6d_rows_to_R(zarr_g[t, 4:10]).reshape(3, 3)) for t in range(len(fixed))])),
        }
        rows.append(rec)
        print(
            f"  {name[-12:]} h{horizon} v2 hw {rec['hw'][0]:+.3f} fix {rec['fix'][0]:+.3f} "
            f"demo {rec['demo'][0]:+.3f}  window rot {rec['geom_deg']:.1f}°",
            flush=True,
        )
    for key in ("hw", "fix", "demo"):
        v2 = np.array([r[key][0] for r in rows])
        dn = np.array([r[key][1] for r in rows])
        print(f"{key:5s} v2 {np.median(v2):+.3f}  dn {np.median(dn):+.1f} mm")


if __name__ == "__main__":
    main()
