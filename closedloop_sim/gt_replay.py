#!/usr/bin/env python3
"""Replay one fixb demo's virtual-target trajectory on the SDF spring. No policy.

The stored ts_pose_virtual_target_0 column is zeros. The command is the label
decoded by MoldAwareRuntime.decode_chunk and decoded_to_action_mats, then
compliance_command. Contact is the plant, not MuJoCo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "UMIFT_Data" / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

from closedloop_sim.api import (  # noqa: E402
    ModelPredictiveController,
    MoldAwareRuntime,
    compliance_command,
    load_policy,
)
from closedloop_sim.plant import CartesianSpringPlant  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import Rp_to_pose7, pose7_to_Rp  # noqa: E402

# Imported so the policy rollouts call these instead of a second copy.
POLICY_API = (load_policy, ModelPredictiveController)

FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
ZARR = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_causal_foot/acp_replay_buffer.zarr"
)
META = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb/shared/fixb_meta.json"
)
OUT = Path(__file__).resolve().parent / "out"

# Stop the replay. A settled press is not one of these.
SPEED_RUNAWAY_MPS = 20.0
PENETRATION_ABORT_M = 0.15


def _label_action(group) -> tuple[np.ndarray, np.ndarray]:
    n = int(group["fixb_v_par_0"].shape[0])
    action = np.zeros((n, 10), dtype=np.float64)
    action[:, 0:2] = np.asarray(group["fixb_v_par_0"], dtype=np.float64).reshape(n, 2)
    action[:, 2] = np.asarray(group["fixb_delta_n_0"], dtype=np.float64).reshape(n)
    action[:, 3] = np.asarray(group["stiffness_0"], dtype=np.float64).reshape(n)
    action[:, 4:10] = np.asarray(group["fixb_delta_R_rot6d_0"], dtype=np.float64).reshape(n, 6)
    times = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(n)
    return action, times


def _unstable(row: dict, fast_s: float) -> tuple[str | None, float]:
    if not np.isfinite(row["speed_mps"]) or not np.isfinite(row["penetration_m"]):
        return "non_finite", fast_s
    if row["penetration_m"] > PENETRATION_ABORT_M:
        return "penetration", fast_s
    # A single sample over 1 m/s is the tool catching a deeper command.
    # Runaway is that speed held, or an immediate blow-up.
    if row["speed_mps"] > SPEED_RUNAWAY_MPS:
        return "speed_runaway", fast_s
    if row["t_s"] > 0.5 and row["speed_mps"] > 1.0:
        fast_s += 0.005
        if fast_s > 0.15:
            return "speed_runaway", fast_s
    else:
        fast_s = 0.0
    return None, fast_s


def replay(episode: int = 40, seconds: float | None = None) -> dict:
    import zarr

    group = zarr.open_group(str(ZARR), mode="r")["data"][f"episode_{episode}"]
    action, times = _label_action(group)
    pose = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    rotation, position = pose7_to_Rp(pose)
    dt = np.diff(times, append=times[-1] + float(np.median(np.diff(times))))
    if seconds is not None:
        keep = times - times[0] <= float(seconds)
        action, times, dt = action[keep], times[keep], dt[keep]
        rotation, position = rotation[keep], position[keep]

    runtime = MoldAwareRuntime.from_assets(
        "r4_world_fixb_tilt",
        field_path=FIELD,
        meta_path=META,
        dry_run=False,
    )
    # One label step from the demo pose. A single decode of the whole episode
    # walks the foot off the demonstration.
    runtime.release_command_foot()
    first = runtime.decode_chunk(
        action[:1], position[0], rotation[0], dt[:1], apply_safety=False
    )
    _nominal0, virtual0 = runtime.decoded_to_action_mats(first)
    contact_field = FixBMoldField(FIELD)
    plant = CartesianSpringPlant(contact_field, virtual0[0, :3, 3], virtual0[0, :3, :3])

    rows = []
    stop = None
    fast_s = 0.0
    for i in range(action.shape[0]):
        runtime.release_command_foot()
        decoded = runtime.decode_chunk(
            action[i : i + 1],
            position[i],
            rotation[i],
            dt[i : i + 1],
            apply_safety=False,
        )
        nominal, virtual = runtime.decoded_to_action_mats(decoded)
        measured = Rp_to_pose7(plant.R.reshape(1, 3, 3), plant.x.reshape(1, 3))[0]
        _nominal_pose, scheduled, stiffness, _raw, _floored = compliance_command(
            nominal,
            virtual,
            np.asarray(decoded["stiffness"]).reshape(-1),
            measured,
            stiffness_floor=None,
            test_nominal_target=False,
            test_nominal_target_stiffness=-1.0,
        )
        state = plant.step(
            scheduled[0, :3],
            virtual[0, :3, :3],
            stiffness[:, :6],
            float(dt[i]),
        )
        state["t_s"] = float(times[i] - times[0])
        state["delta_n_m"] = float(np.asarray(decoded["delta_n"]).reshape(-1)[0])
        state["k_npm"] = float(np.asarray(decoded["stiffness"]).reshape(-1)[0])
        state["k_delta_n_n"] = state["k_npm"] * state["delta_n_m"]
        rows.append(state)
        stop, fast_s = _unstable(state, fast_s)
        if stop is not None:
            break

    trace = {key: np.asarray([row[key] for row in rows]) for key in rows[0] if key != "normal"}
    commanded_inside = trace["delta_n_m"] < -0.001
    lost = commanded_inside & (trace["penetration_m"] <= 0.0)
    contact = trace["penetration_m"] > 0.0
    pressing = contact & (trace["delta_n_m"] < 0.0)
    report = {
        "episode": int(episode),
        "mold": str(FIELD),
        "n_steps": int(trace["t_s"].size),
        "duration_s": float(trace["t_s"][-1] - trace["t_s"][0]) if trace["t_s"].size else 0.0,
        "stopped": stop,
        "penetration_mm": _stats(trace["penetration_m"] * 1e3),
        "contact_loss_pct": float(100.0 * np.mean(lost)) if lost.size else 0.0,
        "contact_pct": float(100.0 * np.mean(contact)) if contact.size else 0.0,
        "contact_force_n": _stats(trace["contact_n"][contact]) if np.any(contact) else None,
        "k_delta_n_n": _stats(trace["k_delta_n_n"][pressing]) if np.any(pressing) else None,
        "force_over_minus_k_dn": _ratio(trace, pressing),
        "speed_mps": _stats(trace["speed_mps"]),
        "after_start_pad": _window(trace, trace["t_s"] > 5.0),
        "n_speed_over_1": int(np.sum(trace["speed_mps"] > 1.0)),
        "plant": {
            "mass_kg": 0.2,
            "damping_ns_per_m": 15.0,
            "contact_n_per_m": 3000.0,
            "tangent_damping_ns_per_m": 5.0,
            "spring_force_limit_n": plant.spring_force_limit_n,
            "spring_force_config": str(plant.spring_force_config),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"gt_fixb_ep{episode}_springclip"
    np.savez_compressed(OUT / f"{tag}.npz", **trace)
    (OUT / f"{tag}.json").write_text(json.dumps(report, indent=2))
    return report


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _window(trace: dict, mask: np.ndarray) -> dict:
    if not np.any(mask):
        return {"n": 0}
    part = {key: np.asarray(value)[mask] for key, value in trace.items()}
    contact = part["penetration_m"] > 0.0
    commanded_inside = part["delta_n_m"] < -0.001
    lost = commanded_inside & (part["penetration_m"] <= 0.0)
    pressing = contact & (part["delta_n_m"] < 0.0)
    return {
        "n": int(mask.sum()),
        "duration_s": float(part["t_s"][-1] - part["t_s"][0]),
        "penetration_mm": _stats(part["penetration_m"] * 1e3),
        "contact_loss_pct": float(100.0 * np.mean(lost)),
        "contact_force_n": _stats(part["contact_n"][contact]) if np.any(contact) else None,
        "k_delta_n_n": _stats(part["k_delta_n_n"][pressing]) if np.any(pressing) else None,
        "force_over_minus_k_dn": _ratio(part, pressing),
        "speed_mps": _stats(part["speed_mps"]),
        "n_speed_over_1": int(np.sum(part["speed_mps"] > 1.0)),
    }


def _ratio(trace: dict, pressing: np.ndarray) -> dict | None:
    if not np.any(pressing):
        return None
    force = trace["contact_n"][pressing]
    scale = -trace["k_delta_n_n"][pressing]
    ok = scale > 1.0
    if not np.any(ok):
        return None
    ratio = force[ok] / scale[ok]
    return _stats(ratio)


def main() -> None:
    seconds = None
    episode = 40
    args = sys.argv[1:]
    if "--seconds" in args:
        seconds = float(args[args.index("--seconds") + 1])
    if "--episode" in args:
        episode = int(args[args.index("--episode") + 1])
    report = replay(episode, seconds)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
