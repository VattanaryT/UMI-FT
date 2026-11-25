"""Recompute mold_geom with the training packer and re-run the 14 horizons.

Does not edit the runner, the zarr, or the hardware logs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
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
from field_v2_dataset import mold_local  # noqa: E402
from fixb_labels import pose7_to_Rp  # noqa: E402
from heading_tilt import geodesic_deg  # noqa: E402
from mold_geom_train_frame import enabled, pack_episode  # noqa: E402
from rot6d import rot6d_rows_to_R  # noqa: E402
from swap_climb import RUNS, _demo_table, _horizons, _match, _predict, _twist_obs  # noqa: E402

# Live robot json is 500 Hz. The logged policy window is every 160 ms, 80 samples.
DS = 80
HORIZON = 51


def _ang(a: np.ndarray, b: np.ndarray) -> float:
    return float(geodesic_deg(a.reshape(1, 3, 3), b.reshape(1, 3, 3))[0])


def _window_index(end: int) -> np.ndarray:
    return end - np.arange(HORIZON - 1, -1, -1) * DS


def main() -> None:
    print(f"flag default {enabled()}")
    demo = vr._demo_path()
    spec = zarr.open_group(vr.DEMO_ZARR, mode="r").attrs["field_v2_axes"]
    origin = np.asarray(spec["origin"], dtype=np.float64)
    axes = np.asarray(spec["axes"], dtype=np.float64)
    tool, position = pose7_to_Rp(demo["pose"])
    packed = pack_episode(position, tool, axes)
    zarr_R = rot6d_rows_to_R(demo["rot6d"])
    ang = geodesic_deg(packed["frame"], zarr_R)
    sd_err = np.abs(packed["sd"] - demo["sd"]) * 1e3
    n_dot = np.sum(packed["normal"] * demo["normal"], axis=1)
    band = np.arange(6008, 6317)
    print(
        f"ep9 batch vs zarr  rot med {np.median(ang):.2f}° p90 {np.percentile(ang, 90):.2f}° "
        f"max {ang.max():.2f}° at {int(np.argmax(ang))}  "
        f"|sd| med {np.median(sd_err):.2f} mm  normal dot med {np.median(n_dot):.4f}"
    )
    print(
        f"ep9 idx 6008-6316  rot med {np.median(ang[band]):.2f}° max {ang[band].max():.2f}°  "
        f"|sd| med {np.median(sd_err[band]):.2f} mm  normal dot med {np.median(n_dot[band]):.4f}"
    )

    table = _demo_table(demo)
    from PyriteConfig.tasks.umift.workcell_frames import load_workcell_extrinsics

    T = load_workcell_extrinsics().T_fixture_from_right_base
    policy, device = vr._policy()
    rows = []
    for name in RUNS:
        group = zarr.open_group(str(vr._log_root(vr.ROOT / name)), mode="r")
        robot = vr._load_robot(str(group.attrs.get("episode_data_folder", "")))
        if robot is None:
            raise SystemExit(f"no robot json for {name}")
        r_pos, r_tool = vr._pose_bundle(robot["pose"], T)
        print(f"{name} robot samples {len(r_pos)}", flush=True)
        fixed = pack_episode(r_pos, r_tool, axes)
        for row in _horizons(name):
            hw = row["obs"]
            eef = np.asarray(hw["robot0_eef_pos"])
            tip = origin + eef[-1] @ axes.T
            end = int(np.argmin(np.linalg.norm(r_pos - tip, axis=1)))
            tip_err = float(np.linalg.norm(r_pos[end] - tip) * 1e3)
            index = _window_index(end)
            if index[0] < 0:
                raise SystemExit(f"{name} h{row['h']} window starts before the log")
            local = mold_local(r_pos[index], origin, axes)
            win_err = float(np.median(np.linalg.norm(local - eef, axis=1)) * 1e3)
            geom = fixed["geom"][index]
            demo_i, d_twist, d_x = _match(table, _twist_obs(hw), float(eef[-1, 0]))
            demo_idx = np.clip(demo_i - np.arange(HORIZON - 1, -1, -1) * DS, 0, len(demo["sd"]) - 1)
            zarr_g = np.concatenate(
                [demo["sd"][demo_idx, None], demo["normal"][demo_idx], demo["rot6d"][demo_idx]],
                axis=1,
            ).astype(np.float32)
            logged_R = rot6d_rows_to_R(np.asarray(hw["robot0_mold_geom"])[-1, 4:10]).reshape(3, 3)
            vs_log = _ang(fixed["frame"][end], logged_R)
            last_ang = _ang(fixed["frame"][end], zarr_R[demo_i])
            last_sd = abs(float(fixed["sd"][end] - demo["sd"][demo_i])) * 1e3
            tcp_mm = float(np.linalg.norm(r_pos[end] - position[demo_i]) * 1e3)
            obs = {key: np.asarray(value) for key, value in hw.items()}
            obs["robot0_mold_geom"] = geom
            pred = _predict(policy, device, obs)
            demo_pred = _predict(policy, device, vr._demo_obs(demo, demo_i))
            rec = {
                "h": row["h"],
                "demo": demo_i,
                "tip_mm": tip_err,
                "win_mm": win_err,
                "tcp_mm": tcp_mm,
                "rot": last_ang,
                "vs_log": vs_log,
                "sd_mm": last_sd,
                "v2": float(np.median(pred[:6, 1])),
                "dn": float(np.median(pred[:6, 2]) * 1e3),
                "v2_demo": float(np.median(demo_pred[:6, 1])),
            }
            rows.append(rec)
            print(
                f"  h{row['h']} demo {demo_i} tip {tip_err:.1f} mm win {win_err:.1f} mm "
                f"tcp {tcp_mm:.1f} mm vs_log {vs_log:.1f}° vs_zarr {last_ang:.1f}° sd {last_sd:.1f} mm "
                f"v2 {rec['v2']:+.3f} dn {rec['dn']:+.1f} demo {rec['v2_demo']:+.3f}",
                flush=True,
            )
    v2 = np.array([r["v2"] for r in rows])
    dn = np.array([r["dn"] for r in rows])
    print(
        f"fixed vs logged rot med {np.median([r['vs_log'] for r in rows]):.1f}°  "
        f"fixed geom vs matched zarr  rot med {np.median([r['rot'] for r in rows]):.1f}°  "
        f"|sd| med {np.median([r['sd_mm'] for r in rows]):.1f} mm  "
        f"tcp med {np.median([r['tcp_mm'] for r in rows]):.1f} mm"
    )
    print(f"policy v2 med {np.median(v2):+.3f}  dn med {np.median(dn):+.1f} mm")
    print(f"window align tip med {np.median([r['tip_mm'] for r in rows]):.2f} mm  "
          f"history med {np.median([r['win_mm'] for r in rows]):.2f} mm")


if __name__ == "__main__":
    main()
