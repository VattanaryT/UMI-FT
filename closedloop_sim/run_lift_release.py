#!/usr/bin/env python3
"""v3r episode 9 with the hold/travel/timeout retract exit. No robot."""

from __future__ import annotations

import json

import numpy as np
import torch

import run_lift_deploy as drv
import closedloop_sim.policy_rollout as pr
from closedloop_sim.api import ModelPredictiveController
from fixb_field import FixBMoldField
from fixb_labels import pose7_to_Rp
from mold_aware_runtime import MoldAwareRuntime, fixb_decode_dt, r4_safety_limits
from PyriteConfig.tasks.umift.umift_type_conversions import raw_to_obs_inference

OUT = drv.OUT
PLOT = drv.PLOT
B_CKPT = drv.TRAIN / "oct_v35_h4s_r4vt_field_v2n_norgb/latest.ckpt"
OLD = {
    "A": "/home/boeing/umi_ft_ws/worktrees/UMI-FT-field_v3r/closedloop_sim/out_ckpt_abc/closed/policy_A_v2n_sanity_full_fixb_ep9.npz",
    "B": "/home/boeing/umi_ft_ws/worktrees/UMI-FT-field_v3r/closedloop_sim/out_ckpt_abc/closed/policy_B_v2n_norgb_full_fixb_ep9.npz",
}
GRID = ((0.5, 0.02), (0.5, 0.04), (1.0, 0.02), (1.0, 0.04))


def _limits():
    return r4_safety_limits(
        max_tcp_penetration_m=0.020,
        wrench_hold_n=90.0,
        abort_on_measured_mesh_distance=False,
        enable_delta_n_floor=False,
        max_press_depth_m=0.005,
    )


def _predict_at(controller, runtime, shape, pose, wrench, times, t_end, device):
    obs_raw = pr._obs_raw(shape, pose, wrench, times, len(pose), float(t_end))
    runtime.reset()
    runtime.fill_obs_raw(obs_raw, robot_id=0)
    episode = {}
    raw_to_obs_inference(obs_raw, episode, shape)
    obs = episode["obs"]
    if "depth_0" in obs:
        obs["depth_0"] = np.zeros_like(obs["depth_0"])
    sparse = shape["sample"]["obs"]["sparse"]
    if "elapsed_time_0" in sparse and "elapsed_time_0" not in obs:
        horizon = int(sparse["elapsed_time_0"]["horizon"])
        obs["elapsed_time_0"] = np.full((horizon, 1), float(t_end) / pr.SLOW_DOWN, np.float32)
    torch.manual_seed(0)
    controller.compute_one_horizon_action(obs, device)
    raw = np.asarray(controller.last_raw_action, dtype=np.float64)
    R0, p0 = pose7_to_Rp(pose[-1].reshape(1, 7))
    action_ds = int(shape["sample"]["action"]["sparse"]["down_sample_steps"])
    dt = fixb_decode_dt(raw.shape[0], action_ds, pr.RAW_DT_S)
    decoded = runtime.decode_chunk(raw, p0[0], R0[0], dt)
    return {
        "t_s": round(float(t_end), 1),
        "raw_dn_mm": round(float(raw[0, 2]) * 1e3, 1),
        "raw_dn_med_mm": round(float(np.median(raw[:, 2])) * 1e3, 1),
        "decoded_dn_mm": round(float(np.asarray(decoded["delta_n"]).reshape(-1)[0]) * 1e3, 1),
    }


def _window(pose, wrench, times, t_end, need):
    t_end = min(float(t_end), float(times[-1]))
    idx = int(np.searchsorted(times, t_end))
    idx = min(max(idx, 0), len(times) - 1)
    lo = max(0, idx - need + 1)
    pose_w = pose[lo : idx + 1]
    wrench_w = wrench[lo : idx + 1]
    time_w = times[lo : idx + 1]
    if len(pose_w) < need:
        pad = need - len(pose_w)
        pose_w = np.concatenate([np.repeat(pose_w[:1], pad, axis=0), pose_w])
        wrench_w = np.concatenate([np.repeat(wrench_w[:1], pad, axis=0), wrench_w])
        step = 1.0 / pr.POSE_HZ
        extra = time_w[0] - step * np.arange(pad, 0, -1)
        time_w = np.concatenate([extra, time_w])
    return pose_w, wrench_w, time_w, t_end


def _probe_source(name, ckpt, controller, runtime, shape, device, pose, wrench, times, logged, tag):
    sparse = shape["sample"]["obs"]["sparse"]
    need = 1
    for key in sparse:
        if str(key).startswith("robot0_"):
            need = max(need, pr._need(sparse, key))
    rows = []
    for t_end in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0):
        if t_end > float(times[-1]):
            continue
        pose_w, wrench_w, time_w, used = _window(pose, wrench, times, t_end, need)
        row = _predict_at(controller, runtime, shape, pose_w, wrench_w, time_w, used, device)
        if logged is not None:
            j = int(np.argmin(np.abs(logged[0] - used)))
            row["logged_dn_mm"] = round(float(logged[1][j]) * 1e3, 1)
        row["where"] = tag
        rows.append(row)
        print(name, tag, row, flush=True)
    return rows


def _drop_rgb_if_absent(shape):
    """B has an rgb raw key and no rgb obs. Same strip the ABC eval used."""
    if "rgb_0" in shape["obs"]:
        return shape
    from omegaconf import open_dict

    with open_dict(shape):
        if "rgb_0" in shape.sample.obs.sparse:
            del shape.sample.obs.sparse["rgb_0"]
        if "rgb_0" in shape.raw:
            del shape.raw["rgb_0"]
    return shape


_load_policy = pr.load_policy


def _load_policy_stripped(path, device):
    policy, shape = _load_policy(path, device)
    return policy, _drop_rgb_if_absent(shape)


pr.load_policy = _load_policy_stripped


def _load(ckpt, device):
    policy, shape = pr.load_policy(str(ckpt), device)
    pr._apply_runner_windows(shape)
    runtime = MoldAwareRuntime.from_assets(
        run_type="r4_world_fixb_tilt", field_path=pr.FIELD, dry_run=False, limits=_limits()
    )
    controller = ModelPredictiveController(
        shape_meta=shape,
        id_list=[0],
        policy=policy,
        action_to_trajectory=lambda _action: None,
        execution_horizon=1,
        fix_orientation=False,
    )
    return policy, shape, runtime, controller


def _demo_arrays():
    import zarr

    group = zarr.open_group(str(pr.ZARR), mode="r")["data"]["episode_9"]
    pose = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    times = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    times = times - times[0]
    wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
    wt = np.asarray(group["wrench_time_stamps_left_0"], dtype=np.float64).reshape(-1)
    wt = wt - wt[0]
    j = np.clip(np.searchsorted(wt, times), 1, len(wt) - 1)
    left = np.abs(times - wt[j - 1]) <= np.abs(times - wt[j])
    take = np.where(left, j - 1, j)
    return pose, wrench[take], times


def _trace_arrays(path):
    blob = np.load(path)
    pose = np.asarray(blob["pose7"], dtype=np.float64)
    times = np.asarray(blob["t_s"], dtype=np.float64).reshape(-1)
    force = np.asarray(blob["force"], dtype=np.float64)
    rotation, _p = pose7_to_Rp(pose)
    wrench = np.zeros((len(pose), 6), dtype=np.float64)
    wrench[:, :3] = np.einsum("nji,ni->nj", rotation, force)
    logged = (times, np.asarray(blob["delta_n_m"], dtype=np.float64))
    return pose, wrench, times, logged


def probe() -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    demo = _demo_arrays()
    out = {}
    ckpts = {"A": drv.CKPT["A"], "B": B_CKPT}
    for name, ckpt in ckpts.items():
        print(f"probe {name} {ckpt}", flush=True)
        policy, shape, runtime, controller = _load(ckpt, device)
        pose, wrench, times, logged = _trace_arrays(OLD[name])
        out[name] = {
            "closed_loop_trace": _probe_source(
                name, ckpt, controller, runtime, shape, device, pose, wrench, times, logged, "old-closed-loop"
            ),
            "demo_pose_zero_rgb": _probe_source(
                name, ckpt, controller, runtime, shape, device, demo[0], demo[1], demo[2], None, "demo"
            ),
        }
        del policy, controller, runtime
        torch.cuda.empty_cache()
    (OUT / "dn_probe.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def _after(row) -> dict:
    blob = np.load(OUT / f"policy_{row['name']}_fixb_ep9.npz")
    t = np.asarray(blob["t_s"])
    contact = np.asarray(blob["penetration_m"]) > 0
    dn = np.asarray(blob["delta_n_m"])
    for i, item in enumerate(row["retracts"]):
        t0 = float(item.get("t_release_s", item["t_s"]))
        t1 = float(row["retracts"][i + 1]["t_s"]) if i + 1 < len(row["retracts"]) else float(t[-1])
        span = (t >= t0) & (t < t1)
        item["returned_to_contact"] = bool(np.any(contact[span])) if span.any() else False
        item["contact_after_s"] = round(float(np.sum(contact[span]) * np.median(np.diff(t))), 2) if span.any() else 0.0
    held = round(sum(float(item.get("owned_s") or 0.0) for item in row["retracts"]), 2)
    row["held_s"] = held
    row["returned"] = sum(1 for item in row["retracts"] if item.get("returned_to_contact"))
    row["final"] = {
        "clearance_mm": row["final_clearance_mm"],
        "delta_n_mm": round(float(dn[-1]) * 1e3, 2),
        "in_contact": bool(contact[-1]),
        "retracting": bool(row["retracts"] and "t_release_s" not in row["retracts"][-1]),
    }
    return row


def _rank(rows):
    return sorted(rows, key=lambda r: (r["returned"], -r["spurious_mid_stroke"], len(r["retracts"]), -r["held_s"]), reverse=True)


def _plot(best):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blob = np.load(OUT / f"policy_{best['name']}_fixb_ep9.npz")
    t = np.asarray(blob["t_s"])
    field = FixBMoldField(drv.FIELD)
    clearance = np.asarray(field.query(blob["position"])["sd"], dtype=np.float64).reshape(-1) * 1e3
    dn = np.asarray(blob["delta_n_m"]) * 1e3
    force = np.asarray(blob["contact_n"])
    step = max(1, len(t) // 4000)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    axes[0].plot(t[::step], dn[::step], color="C0")
    axes[1].plot(t[::step], clearance[::step], color="C1")
    axes[2].plot(t[::step], force[::step], color="C3")
    axes[0].set_ylabel("δn mm")
    axes[1].set_ylabel("clearance mm")
    axes[2].set_ylabel("|F·n| N")
    axes[2].set_xlabel("t s")
    for item in best["retracts"]:
        t1 = item.get("t_release_s", item["t_s"] + item.get("owned_s", 0.0))
        for ax in axes:
            ax.axvspan(item["t_s"], t1, color="C0", alpha=0.15)
    fig.suptitle(best["name"])
    PLOT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOT / "v3r_release_best.png", dpi=120)
    np.savez(
        OUT / "best_trace.npz",
        t_s=t[::step],
        delta_n_mm=dn[::step],
        clearance_mm=clearance[::step],
        force_n=force[::step],
        name=np.array(best["name"]),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    probe()
    axes = drv._axes()
    ends = drv._ends(axes)
    field = FixBMoldField(drv.FIELD)
    starts = drv._demo_lifts(field)
    demo = pr._demo_episode(9)
    hook = drv.Hook(axes, ends)
    drv._install(hook)
    drv.CKPT["B"] = B_CKPT
    rows = []
    flags = {"ensemble": False, "force": False, "stroke": False, "air": False}
    rows.append(_after(drv._run(hook, "B", "B_none", flags, demo, starts, field)))
    for t_hold, travel in GRID:
        flags = {
            "ensemble": False,
            "force": True,
            "stroke": False,
            "air": False,
            "t_hold_s": t_hold,
            "travel_m": travel,
            "timeout_s": 2.0,
        }
        name = f"v3r_h{t_hold:.1f}_d{travel * 100:.0f}"
        rows.append(_after(drv._run(hook, "v3r", name, flags, demo, starts, field)))
        (OUT / "release_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    v3r = [row for row in rows if row["name"].startswith("v3r_")]
    best = _rank(v3r)[0]
    _plot(best)
    (OUT / "release_summary.json").write_text(json.dumps({"best": best["name"], "rows": rows}, indent=2) + "\n")
    print("best", best["name"], flush=True)


if __name__ == "__main__":
    main()
