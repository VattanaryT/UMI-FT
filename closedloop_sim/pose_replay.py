"""Drive the plant with recorded TCP poses. No label decode."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "UMIFT_Data" / "surface_frames_scripts"))

from closedloop_sim.plant import (  # noqa: E402
    CONTACT_N_PER_M,
    DAMPING_NS_PER_M,
    INTEGRATE_DT_S,
    MASS_KG,
    CartesianSpringPlant,
    controller_gains,
)
from closedloop_sim.segments import label_segments  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import pose7_to_Rp  # noqa: E402

FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
ZARR = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_causal_foot/acp_replay_buffer.zarr"
)
OUT = Path(__file__).resolve().parent / "out"


def _nearest(src_t, dst_t):
    j = np.clip(np.searchsorted(src_t, dst_t), 0, len(src_t) - 1)
    prev = np.clip(j - 1, 0, len(src_t) - 1)
    use_prev = np.abs(src_t[prev] - dst_t) < np.abs(src_t[j] - dst_t)
    return np.where(use_prev, prev, j)


def replay(episode: int = 40) -> dict:
    group = zarr.open_group(str(ZARR), mode="r")["data"][f"episode_{episode}"]
    pose = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    times = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    stiff = np.asarray(group["stiffness_0"], dtype=np.float64).reshape(-1)
    rotation, position = pose7_to_Rp(pose)
    dt = np.diff(times, append=times[-1] + float(np.median(np.diff(times))))
    field = FixBMoldField(FIELD)
    plant = CartesianSpringPlant(field, position[0], rotation[0], tangent_damping=0.0)
    gains = controller_gains()
    rows_x, rows_f = [], []
    for i in range(len(position)):
        k = float(stiff[i]) if i < len(stiff) else 300.0
        gain = np.diag([k, k, k, 2.0, 2.0, 2.0])
        state = plant.step(position[i], rotation[i], gain, float(dt[i]))
        rows_x.append(plant.x.copy())
        rows_f.append(np.asarray(state["contact_world"], dtype=np.float64))
    sim_x = np.asarray(rows_x)
    sim_f = np.linalg.norm(np.asarray(rows_f), axis=1)
    query = field.query(position)
    sd = np.asarray(query["sd"], dtype=np.float64).reshape(-1)
    clearance = np.maximum(sd, 0.0)
    static_f = CONTACT_N_PER_M * np.maximum(-sd, 0.0)
    wrench_t = np.asarray(group["wrench_time_stamps_0"], dtype=np.float64).reshape(-1)
    wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
    demo_f = np.linalg.norm(wrench[_nearest(wrench_t, times), :3], axis=1)
    labels = label_segments(demo_f, clearance)
    path = np.linalg.norm(sim_x - position, axis=1) * 1e3
    force_err = np.abs(sim_f - demo_f)
    static_err = np.abs(static_f - demo_f)
    report = {
        "episode": episode,
        "command": "recorded ts_pose_fb_0",
        "stiffness": "zarr stiffness_0, isotropic translation",
        "dt_s_median": float(np.median(np.diff(times))),
        "plant_dt_s": INTEGRATE_DT_S,
        "mass_kg": MASS_KG,
        "damping_ns_per_m": DAMPING_NS_PER_M,
        "contact_n_per_m": CONTACT_N_PER_M,
        "yaml": {k: (str(v) if k == "config" else v) for k, v in gains.items()},
        "stiffness_0_median": float(np.median(stiff)),
        "target_path_mm": 5.0,
        "target_force_n": 5.0,
    }
    for name in ("contact", "air-travel"):
        mask = labels == name
        report[name] = {
            "n": int(mask.sum()),
            "path_mm": float(np.median(path[mask])),
            "force_error_n": float(np.median(force_err[mask])),
            "static_force_error_n": float(np.median(static_err[mask])),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m1_pose_ep40.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(OUT / "m1_pose_ep40.npz", position=sim_x, force=np.asarray(rows_f), t_s=times - times[0])
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    replay(40)
