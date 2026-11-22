#!/usr/bin/env python3
"""Fixed-input mold outputs for one checkout. Reads UMIFT_ROOT."""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ["UMIFT_ROOT"]).resolve()
SCRIPTS = ROOT / "UMIFT_Data" / "surface_frames_scripts"
for entry in (SCRIPTS, ROOT, ROOT / "PyriteML", ROOT / "PyriteUtility"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

ZARR = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_world_fixb_tilt/acp_replay_buffer.zarr"
)
FIELD = ZARR.parents[1] / "shared" / "fixb_field.npz"
META = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb/shared/fixb_meta.json"
)
EPISODES = (0, 9, 31)
CHUNK_AT = 1024
CHUNK = 4
QUERY_N = 32


def _pose(group, index):
    from fixb_labels import pose7_to_Rp

    pose = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    rotation, position = pose7_to_Rp(pose)
    i = min(index, len(position) - CHUNK)
    return position[i : i + CHUNK], rotation[i], position[i]


def _action(group, index):
    n = int(np.asarray(group["fixb_v_par_0"]).shape[0])
    i = min(index, n - CHUNK)
    action = np.zeros((CHUNK, 27), dtype=np.float64)
    action[:, 0:2] = np.asarray(group["fixb_v_par_0"], dtype=np.float64)[i : i + CHUNK]
    action[:, 2] = np.asarray(group["fixb_delta_n_0"], dtype=np.float64).reshape(-1)[i : i + CHUNK]
    if "fixb_delta_R_rot6d_0" in group:
        action[:, 21:27] = np.asarray(group["fixb_delta_R_rot6d_0"], dtype=np.float64)[i : i + CHUNK]
    if "stiffness_0" in group:
        action[:, 18] = np.asarray(group["stiffness_0"], dtype=np.float64).reshape(-1)[i : i + CHUNK]
    return action


def _decode(out, episode, group):
    from fixb_field import FixBMoldField
    from mold_aware_runtime import MoldAwareRuntime

    runtime = MoldAwareRuntime.from_assets(
        "r4_world_fixb_tilt", field_path=FIELD, meta_path=META, dry_run=False
    )
    position, rotation, p0 = _pose(group, CHUNK_AT)
    action = _action(group, CHUNK_AT)
    dt = np.full(CHUNK, 1.0 / 15.0)
    for mode, safety in (("raw", False), ("safe", True)):
        if hasattr(runtime, "release_command_foot"):
            runtime.release_command_foot()
        decoded = runtime.decode_chunk(action, p0, rotation, dt, apply_safety=safety)
        prefix = f"dec_ep{episode}_{mode}"
        out[f"{prefix}_xs"] = np.asarray(decoded["x_s"], dtype=np.float64)
        out[f"{prefix}_xd"] = np.asarray(decoded["x_d"], dtype=np.float64)
        out[f"{prefix}_Rd"] = np.asarray(decoded["R_d"], dtype=np.float64).reshape(CHUNK, 9)
        triggers = getattr(runtime, "last_triggers", [])
        out[f"{prefix}_triggers"] = np.asarray(",".join(triggers))


def _query(out, episode, position):
    from fixb_field import FixBMoldField

    field = FixBMoldField(FIELD)
    pick = np.linspace(0, len(position) - 1, QUERY_N).astype(int)
    points = position[pick]
    queried = field.query(points)
    out[f"query_ep{episode}_xs"] = np.asarray(queried["x_s"], dtype=np.float64)
    out[f"query_ep{episode}_R"] = np.asarray(queried["R"], dtype=np.float64).reshape(QUERY_N, 9)


def _foot(out, episode, group):
    import mold_aware_runtime as runtime_mod

    integrate = getattr(runtime_mod, "integrate_tcp_foot", None)
    if integrate is None:
        return
    position, rotation, p0 = _pose(group, CHUNK_AT)
    action = _action(group, CHUNK_AT)
    dt = np.full(CHUNK, 1.0 / 15.0)
    args = {
        "field": __import__("fixb_field", fromlist=["FixBMoldField"]).FixBMoldField(FIELD),
        "p0": p0,
        "v_par": action[:, 0:2],
        "delta_n": action[:, 2],
        "dt": dt,
        "prev_x": None,
        "prev_R": None,
        "hold_face": True,
        "cap_origin": p0,
        "R_tool": rotation,
    }
    keep = {name: args[name] for name in inspect.signature(integrate).parameters if name in args}
    decoded = integrate(**keep)
    out[f"foot_ep{episode}_xs"] = np.asarray(decoded["x_s"], dtype=np.float64)
    out[f"foot_ep{episode}_R"] = np.asarray(decoded["R"], dtype=np.float64).reshape(CHUNK, 9)


def _batches(out):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    import hydra

    train_cfg = ROOT / "PyriteML" / "diffusion_policy" / "config"
    scripts_cfg = SCRIPTS / "config"
    names = sorted(path.stem for path in (scripts_cfg / "task").glob("umift_oct_fixb_r4vt*.yaml"))
    for name in names:
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=str(train_cfg)):
            cfg = compose(
                config_name="train_umift_workspace",
                overrides=[
                    f"hydra.searchpath=[file://{scripts_cfg}]",
                    f"task={name}",
                    "task.dataset.exclusion_stats_path=null",
                    "hydra.job.chdir=false",
                ],
            )
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        action = dataset[0]["action"]["sparse"]
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        out[f"batch_{name}"] = np.asarray(action, dtype=np.float32)


def _gt(out):
    """Same loop as closedloop_sim.gt_replay.replay. Also keeps the TCP."""
    import zarr

    from closedloop_sim.gt_replay import _label_action
    from closedloop_sim.plant import CartesianSpringPlant
    from fixb_field import FixBMoldField
    from fixb_labels import Rp_to_pose7, pose7_to_Rp
    from mold_aware_runtime import MoldAwareRuntime, compliance_command

    zarr_path = Path(
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
        "r4vt_causal_foot/acp_replay_buffer.zarr"
    )
    group = zarr.open_group(str(zarr_path), mode="r")["data"]["episode_40"]
    action, times = _label_action(group)
    rotation, position = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
    dt = np.diff(times, append=times[-1] + float(np.median(np.diff(times))))
    runtime = MoldAwareRuntime.from_assets(
        "r4_world_fixb_tilt", field_path=FIELD, meta_path=META, dry_run=False
    )
    runtime.release_command_foot()
    first = runtime.decode_chunk(action[:1], position[0], rotation[0], dt[:1], apply_safety=False)
    _nominal0, virtual0 = runtime.decoded_to_action_mats(first)
    plant = CartesianSpringPlant(FixBMoldField(FIELD), virtual0[0, :3, 3], virtual0[0, :3, :3])
    xs, force, clock = [], [], []
    for i in range(action.shape[0]):
        runtime.release_command_foot()
        decoded = runtime.decode_chunk(
            action[i : i + 1], position[i], rotation[i], dt[i : i + 1], apply_safety=False
        )
        nominal, virtual = runtime.decoded_to_action_mats(decoded)
        measured = Rp_to_pose7(plant.R.reshape(1, 3, 3), plant.x.reshape(1, 3))[0]
        _nominal_pose, scheduled, stiffness, _raw, _floored = compliance_command(
            nominal, virtual, np.asarray(decoded["stiffness"]).reshape(-1), measured
        )
        state = plant.step(scheduled[0, :3], virtual[0, :3, :3], stiffness[:, :6], float(dt[i]))
        xs.append(plant.x.copy())
        force.append(state["contact_n"])
        clock.append(float(times[i] - times[0]))
    step = max(1, len(xs) // 200)
    out["gt_t"] = np.asarray(clock[::step], dtype=np.float64)
    out["gt_x"] = np.asarray(xs[::step], dtype=np.float64)
    out["gt_force"] = np.asarray(force[::step], dtype=np.float64)


def capture() -> dict:
    import zarr
    from fixb_labels import pose7_to_Rp

    out = {}
    root = zarr.open(str(ZARR), mode="r")["data"]
    for episode in EPISODES:
        group = root[f"episode_{episode}"]
        position = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))[1]
        _query(out, episode, position)
        try:
            _decode(out, episode, group)
        except Exception as exc:
            out[f"dec_ep{episode}_error"] = np.asarray(f"{type(exc).__name__}: {exc}")
        try:
            _foot(out, episode, group)
        except Exception as exc:
            out[f"foot_ep{episode}_error"] = np.asarray(f"{type(exc).__name__}: {exc}")
    try:
        _batches(out)
    except Exception as exc:
        out["batch_error"] = np.asarray(f"{type(exc).__name__}: {exc}")
    try:
        _gt(out)
    except Exception as exc:
        out["gt_error"] = np.asarray(f"{type(exc).__name__}: {exc}")
    return out


def main() -> None:
    destination = Path(sys.argv[1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = capture()
    np.savez_compressed(destination, **arrays)
    print(f"wrote {destination} keys={len(arrays)} bytes={destination.stat().st_size}")


if __name__ == "__main__":
    main()
