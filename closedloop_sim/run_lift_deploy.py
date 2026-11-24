#!/usr/bin/env python3
"""Compare deploy-time lift strategies. Flags default off. New files only.

Closed loop is episode 9. Open loop, when the dataset loads, is episodes
9, 12, 28, 37, 41. Does not write checkpoints, zarrs, or normalizers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "UMIFT_Data/surface_frames_scripts"),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
]

import closedloop_sim.policy_rollout as pr  # noqa: E402
from closedloop_sim.lift_strategies import LiftStrategy, _check  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import pose7_to_Rp  # noqa: E402

OUT = ROOT / "closedloop_sim" / "out_lift_deploy"
PLOT = ROOT / "UMIFT_Data/surface_frames_scripts/eval/lift_deploy"
TRAIN = Path("/mnt/data/umidata/ft/training_outputs")
EPISODES = [9, 12, 28, 37, 41]
CKPT = {
    "A": TRAIN / "oct_v35_h4s_r4vt_field_v2n/latest.ckpt",
    "v3r": TRAIN / "oct_v35_h4s_r4vt_field_v3r/latest_final.ckpt",
}
FIELD = pr.FIELD
ZARR = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_field_v2_k3000/acp_replay_buffer.zarr"
)


def _axes():
    import zarr

    raw = zarr.open_group(str(ZARR), mode="r").attrs["field_v2_axes"]["axes"]
    return np.asarray(raw, dtype=np.float64).reshape(3, 3)


def _ends(axes) -> tuple[float, float]:
    data = np.load(FIELD)
    vertices = np.asarray(data["vertices"], dtype=np.float64)
    apron = np.asarray(data["is_apron"]).astype(bool) if "is_apron" in data.files else np.zeros(len(vertices), bool)
    body = vertices[~apron] if np.any(~apron) else vertices
    proj = body @ axes[:, 0]
    return float(proj.min()), float(proj.max())


def _demo_lifts(field):
    import zarr

    group = zarr.open_group(str(pr.ZARR), mode="r")["data"]["episode_9"]
    pose = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    times = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    _r, position = pose7_to_Rp(pose)
    wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
    wt = np.asarray(group["wrench_time_stamps_left_0"], dtype=np.float64).reshape(-1)
    j = np.clip(np.searchsorted(wt, times), 1, len(wt) - 1)
    left = np.abs(times - wt[j - 1]) <= np.abs(times - wt[j])
    force = np.linalg.norm(wrench[np.where(left, j - 1, j), :3], axis=1)
    clearance = np.asarray(field.query(position)["sd"], dtype=np.float64).reshape(-1)
    t = times - times[0]
    phase = "approach"
    labels = []
    for f, c in zip(force, clearance):
        if f > 5.0:
            phase = "contact"
        elif f < 2.0 and c > 0.010:
            phase = "air-travel"
        elif phase == "contact":
            phase = "lift-off"
        elif phase == "air-travel":
            phase = "approach"
        labels.append(phase)
    labels = np.asarray(labels)
    on = labels == "lift-off"
    prev = np.concatenate([[False], on[:-1]])
    return t[on & ~prev]


class Hook:
    def __init__(self, axes, ends):
        self.axes = axes
        self.ends = ends
        self.strategy: LiftStrategy | None = None
        self.only_dn = False
        self.along = 0.0
        self.v_tan = np.zeros(3)
        self._plant = None

    def observe(self, plant, state) -> None:
        self._plant = plant
        if self.strategy is not None:
            self.strategy.observe(float(state["contact_n"]))

    def note(self, decoded) -> None:
        frame = np.asarray(decoded["R_field"][0], dtype=np.float64)
        v = np.asarray(decoded["v_par"][0], dtype=np.float64).reshape(2)
        self.v_tan = frame[:, :2] @ v
        self.along = float(np.dot(self.v_tan, self.axes[:, 0]))

    def rewrite(self, wp, t):
        if self.strategy is None or self._plant is None:
            return wp
        plant = self._plant
        query = plant.field.query(plant.x.reshape(1, 3))
        normal = np.asarray(query["R"][0, :, 2], dtype=np.float64)
        normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
        foot = np.asarray(query["x_s"][0], dtype=np.float64)
        clearance = float(query["sd"][0])
        s = float(np.dot(foot, self.axes[:, 0]))
        s_end = min(abs(s - self.ends[0]), abs(s - self.ends[1]))
        away = float(np.dot(np.asarray(wp.position) - plant.x, normal)) > 0.0
        self.strategy.v_tan = self.v_tan
        cmd = self.strategy.step(t, float(wp.delta_n_m), away, self.along, s_end, clearance)
        if cmd is None:
            return wp
        return pr.Waypoint(foot + cmd * normal, wp.rotation, wp.stiffness, wp.k_npm, cmd)


def _install(hook: Hook) -> None:
    import mold_aware_runtime as runtime

    step = pr.CartesianSpringPlant.step
    sample = pr.Schedule.sample
    decode = pr.MoldAwareRuntime.decode_chunk
    orig_blend = runtime.blend_action_rows

    def blend(actions, starts, steps, decay_m=0.01):
        mixed = orig_blend(actions, starts, steps, decay_m=decay_m)
        if not hook.only_dn:
            return mixed
        expanded = [runtime.expand_r4_action(np.asarray(a, dtype=np.float64)) for a in actions]
        for row, abs_step in enumerate(steps):
            newest = None
            newest_start = -10**9
            for action, start in zip(expanded, starts):
                index = int(abs_step) - int(start)
                if 0 <= index < len(action) and int(start) >= newest_start:
                    newest_start = int(start)
                    newest = action[index]
            mixed[row, 0:2] = newest[0:2]
            if mixed.shape[1] >= 27:
                mixed[row, 21:27] = newest[21:27]
        return mixed

    def wrapped_step(self, x_cmd, rotation, stiffness, dt):
        state = step(self, x_cmd, rotation, stiffness, dt)
        hook.observe(self, state)
        return state

    def wrapped_sample(self, t):
        return hook.rewrite(sample(self, t), t)

    def wrapped_decode(self, raw_action, p0, R0, dt, apply_safety=True, frame_seed=None, label_R=None):
        decoded = decode(
            self, raw_action, p0, R0, dt,
            apply_safety=apply_safety, frame_seed=frame_seed, label_R=label_R,
        )
        hook.note(decoded)
        return decoded

    runtime.blend_action_rows = blend
    pr.CartesianSpringPlant.step = wrapped_step
    pr.Schedule.sample = wrapped_sample
    pr.MoldAwareRuntime.decode_chunk = wrapped_decode


def _tilt(pose7) -> dict:
    rotation, _p = pose7_to_Rp(np.asarray(pose7, dtype=np.float64))
    if len(rotation) < 2:
        return {"n": 0, "max_deg": 0.0, "n_gt_10": 0}
    rel = np.einsum("nji,njk->nik", rotation[:-1], rotation[1:])
    c = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    deg = np.degrees(np.arccos(c))
    return {"n": int(deg.size), "max_deg": round(float(deg.max()), 2), "n_gt_10": int(np.sum(deg > 10.0))}


def _score(starts, t, clearance, contact) -> dict:
    reached = aligned = scored = 0
    for t0 in starts:
        if len(t) == 0 or t0 + 0.5 > float(t[-1]) + 1e-6:
            continue
        scored += 1
        window = (t >= t0) & (t <= t0 + 0.5)
        before = (t >= t0 - 1.0) & (t < t0)
        hit = bool(window.any() and float(np.max(clearance[window])) >= 0.010)
        if hit:
            reached += 1
        had = bool(before.any() and np.any(contact[before]))
        already = bool(before.any() and float(np.min(clearance[before])) >= 0.010)
        if hit and had and not already:
            aligned += 1
    return {"reached_10mm": reached, "aligned": aligned, "scored": scored}


def _run(hook, which, name, flags, demo, starts, field) -> dict:
    hook.strategy = None if not any(flags.get(k) for k in ("force", "stroke", "air")) else LiftStrategy(**{
        k: v for k, v in flags.items() if k != "ensemble"
    })
    hook.only_dn = bool(flags.get("ensemble"))
    pr.CHECKPOINTS[name] = str(CKPT[which])
    pr.OUT = OUT
    print(f"rollout {name} {flags}", flush=True)
    report = pr.rollout(name, demo, episode=9)
    if hook.strategy is not None:
        hook.strategy.finish()
    blob = np.load(OUT / f"policy_{name}_fixb_ep9.npz")
    query = field.query(blob["position"])
    clearance = np.asarray(query["sd"], dtype=np.float64).reshape(-1)
    t = np.asarray(blob["t_s"])
    contact = np.asarray(blob["penetration_m"]) > 0
    lifts = _score(starts, t, clearance, contact)
    demo_p = np.asarray(demo["position"])
    demo_t = np.asarray(demo["t_s"])
    idx = np.clip(np.searchsorted(demo_t, t), 0, len(demo_t) - 1)
    path = np.linalg.norm(blob["position"] - demo_p[idx], axis=1)
    retracts = [] if hook.strategy is None else hook.strategy.retracts
    spurious = 0
    for row in retracts:
        if starts.size == 0 or float(np.min(np.abs(starts - row["t_s"]))) > 0.5:
            row["mid_stroke"] = True
            spurious += 1
        else:
            row["mid_stroke"] = False
    edges = np.flatnonzero(contact[:-1] & ~contact[1:]) if contact.size else []
    mid = 0
    for i in edges:
        later = np.flatnonzero(contact[i + 1:])
        if later.size and float(t[i + 1 + later[0]] - t[i]) <= 0.5:
            mid += 1
    row = {
        "policy": which,
        "name": name,
        "flags": flags,
        "rgb": "zeros",
        "stopped": report.get("stopped"),
        "duration_s": round(float(report.get("duration_s") or 0.0), 2),
        "lift_10mm": lifts,
        "contact_losses": {"falling_edges": int(np.size(edges)), "mid_stroke": int(mid)},
        "retracts": retracts,
        "spurious_mid_stroke": int(spurious),
        "approach_s": round(float(np.sum((~contact) & (clearance < 0.010)) * np.median(np.diff(t))) if len(t) > 1 else 0.0, 2),
        "contact_s": round(float(np.sum(contact) * np.median(np.diff(t))) if len(t) > 1 else 0.0, 2),
        "path_mm_median": round(float(np.median(path) * 1e3), 1),
        "path_mm_contact": round(float(np.median(path[contact]) * 1e3), 1) if np.any(contact) else None,
        "tilt_step_gt_10deg": _tilt(blob["pose7"]),
        "final_clearance_mm": round(float(clearance[-1]) * 1e3, 2),
    }
    (OUT / f"{name}.json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps({k: row[k] for k in ("name", "lift_10mm", "contact_losses", "spurious_mid_stroke", "stopped")}), flush=True)
    return row


def _frame_note() -> dict:
    import zarr

    group = zarr.open_group(str(ZARR), mode="r")["data"]["episode_9"]
    _r, position = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
    stored = np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
    query = np.asarray(FixBMoldField(FIELD).query(position[::50])["R"], dtype=np.float64)
    rel = np.einsum("nji,njk->nik", query, stored[::50][: len(query)])
    c = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    deg = np.degrees(np.arccos(c))
    return {
        "query_vs_mold_R_field_0_median_deg": round(float(np.median(deg)), 2),
        "executor": (
            "Committed decode uses the smooth causal frame, not FixBMoldField.query. "
            "On A that commanded R_d is 1.8° from mold_R_field_0 @ ΔR. "
            "Replacing it with the label frame raised tilt snaps from 0 to 9 (max 33.6°). Not applied. "
            "v3r R_tcp @ ΔR is 175° off that command and the jump guard aborts at 1.3 s."
        ),
    }


def _plots(rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r["name"] for r in rows]
    reached = [r["lift_10mm"]["reached_10mm"] / max(r["lift_10mm"]["scored"], 1) for r in rows]
    aligned = [r["lift_10mm"]["aligned"] / max(r["lift_10mm"]["scored"], 1) for r in rows]
    x = np.arange(len(names))
    ax.bar(x - 0.15, reached, 0.3, label="window ≥10 mm")
    ax.bar(x + 0.15, aligned, 0.3, label="aligned to contact end")
    ax.axhline(0.9, color="k", ls="--", lw=0.8)
    ax.set_xticks(x, names, rotation=70, ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of 26 demo lifts")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT / "lift_fraction.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for row in rows:
        if row["policy"] != "v3r":
            continue
        blob = np.load(OUT / f"policy_{row['name']}_fixb_ep9.npz")
        c = np.asarray(FixBMoldField(FIELD).query(blob["position"])["sd"]) * 1e3
        ax.plot(blob["t_s"], c, lw=0.8, label=row["name"])
    ax.axhline(10, color="k", ls="--", lw=0.6)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("clearance mm")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOT / "v3r_clearance.png", dpi=120)
    plt.close(fig)


def _open_loop(rows_out: list) -> None:
    """Predicted δn on the held-out episodes, strategies applied to recorded force."""
    import hydra
    import torch
    import zarr
    from omegaconf import OmegaConf
    from PyriteEnvSuites.scripts.run_policy_open_loop_val_export import (
        load_policy_and_cfg,
        predict_sparse_action,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.path[:0] = [
        "/home/boeing/umi_ft_ws/worktrees/UMI-FT-field_v3r/UMIFT_Data/surface_frames_scripts",
        "/home/boeing/umi_ft_ws/worktrees/UMI-FT-field_v2n/UMIFT_Data/surface_frames_scripts",
    ]
    field = FixBMoldField(FIELD)
    axes = _axes()
    ends = _ends(axes)
    for which, ckpt in CKPT.items():
        policy, cfg = load_policy_and_cfg(str(ckpt), device)
        OmegaConf.set_struct(cfg.task.dataset, False)
        local = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
        optional = {"filter_stats_path", "pose_jump_flag_path", "r2_action_normalizer_path"}
        for key, value in list(cfg.task.dataset.items()):
            if not isinstance(value, str) or not value.startswith("/"):
                continue
            path = Path(value)
            if path.exists():
                continue
            if key == "dataset_path":
                cfg.task.dataset[key] = str(local / path.parent.name / "acp_replay_buffer.zarr")
            elif key == "action_normalizer_path":
                found = list(Path("/home/boeing/umi_ft_ws/worktrees").glob(f"**/assets/{path.name}"))
                if not found:
                    raise SystemExit(f"normalizer missing {path.name}")
                cfg.task.dataset[key] = str(found[0])
            elif key == "long_dt_mask_path":
                cfg.task.dataset[key] = str(local / "shared/long_dt_gt_100ms.npz")
            elif key in optional or key.endswith("_path"):
                cfg.task.dataset[key] = None
        if "exclusion_stats_path" in cfg.task.dataset:
            cfg.task.dataset.exclusion_stats_path = str(OUT / f"{which}_exclusion.json")
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        val = dataset.get_validation_dataset()
        root = zarr.open_group(str(cfg.task.dataset.dataset_path), mode="r")["data"]
        rgb_cache = {}
        keep = []
        last = {}
        for idx, (ep, _rgb) in enumerate(val.sampler.indices):
            ep = int(ep)
            if ep not in EPISODES:
                continue
            if ep not in rgb_cache:
                rgb_cache[ep] = np.asarray(root[f"episode_{ep}"]["rgb_time_stamps_0"]).reshape(-1)
            rgb_t = rgb_cache[ep]
            t = float(rgb_t[int(_rgb)] - rgb_t[0])
            if t - last.get(ep, -1e9) < 0.5:
                continue
            last[ep] = t
            keep.append((idx, ep, t))
        print(f"[open] {which} queries {len(keep)}", flush=True)
        pred = []
        for n, (idx, ep, t) in enumerate(keep):
            sample = val[idx]
            obs = {
                key: value.numpy()
                for key, value in sample["obs"]["sparse"].items()
                if not str(key).startswith("rgb_") or key in policy.obs_encoder.rgb_keys
            }
            action = np.asarray(predict_sparse_action(policy, {"sparse": obs}, device), dtype=np.float64)
            pred.append({"ep": ep, "t": t, "dn": float(action[0, 2]), "v": float(action[0, 0])})
            if n and n % 40 == 0:
                print(f"[open] {which} {n}", flush=True)
        del policy, dataset, val
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        by_ep = {}
        for row in pred:
            by_ep.setdefault(row["ep"], []).append(row)
        for ep, queries in by_ep.items():
            queries.sort(key=lambda item: item["t"])
            group = root[f"episode_{ep}"]
            times = np.asarray(group["robot_time_stamps_0"]).reshape(-1)
            t = times - times[0]
            pose = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
            _r, position = pose7_to_Rp(pose)
            wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
            wt = np.asarray(group["wrench_time_stamps_left_0"]).reshape(-1)
            j = np.clip(np.searchsorted(wt, times), 1, len(wt) - 1)
            take = np.where(np.abs(times - wt[j - 1]) <= np.abs(times - wt[j]), j - 1, j)
            query = field.query(position)
            normal = np.asarray(query["R"][:, :, 2], dtype=np.float64)
            force_w = np.einsum("nij,nj->ni", _r, wrench[take, :3])
            fn = np.abs(np.einsum("ij,ij->i", force_w, normal))
            clearance = np.asarray(query["sd"], dtype=np.float64).reshape(-1)
            foot = np.asarray(query["x_s"])
            s_end = np.minimum(
                np.abs(foot @ axes[:, 0] - ends[0]),
                np.abs(foot @ axes[:, 0] - ends[1]),
            )
            step = max(1, int(round(0.01 / np.median(np.diff(t)))))
            for flags_name, flags in _catalog():
                sim = LiftStrategy(**{k: v for k, v in flags.items() if k != "ensemble"}, dt=float(np.median(np.diff(t[::step]))))
                q = 0
                dn = queries[0]["dn"]
                along = queries[0]["v"]
                commanded = np.zeros(len(t[::step]))
                tt = t[::step]
                for i, ti in enumerate(tt):
                    while q + 1 < len(queries) and queries[q + 1]["t"] <= ti:
                        q += 1
                        dn = queries[q]["dn"]
                        along = queries[q]["v"]
                    k = min(i * step, len(t) - 1)
                    sim.observe(float(fn[k]))
                    cmd = sim.step(float(ti), dn, False, along, float(s_end[k]), float(clearance[k]))
                    commanded[i] = dn if cmd is None else cmd
                sim.finish()
                # lift starts from this episode's recorded force, same rule as ep 9
                phase = "approach"
                labels = []
                for f, c in zip(fn, np.maximum(clearance, 0)):
                    if f > 5:
                        phase = "contact"
                    elif f < 2 and c > 0.010:
                        phase = "air-travel"
                    elif phase == "contact":
                        phase = "lift-off"
                    elif phase == "air-travel":
                        phase = "approach"
                    labels.append(phase)
                labels = np.asarray(labels)
                on = labels == "lift-off"
                prev = np.concatenate([[False], on[:-1]])
                starts = t[on & ~prev]
                hit = 0
                for t0 in starts:
                    w = commanded[(tt >= t0) & (tt <= t0 + 0.5)]
                    if w.size and float(np.max(w)) >= 0.010:
                        hit += 1
                rows_out.append({
                    "policy": which,
                    "episode": ep,
                    "strategy": flags_name,
                    "lift_pred_ge_10mm": hit,
                    "lift_events": int(starts.size),
                    "retracts": len(sim.retracts),
                })
        print(f"[open] {which} done", flush=True)


def _catalog():
    return [
        ("none", {"ensemble": False, "force": False, "stroke": False, "air": False}),
        ("dn_ensemble", {"ensemble": True, "force": False, "stroke": False, "air": False}),
        ("force", {"ensemble": False, "force": True, "stroke": False, "air": False}),
        ("stroke", {"ensemble": False, "force": False, "stroke": True, "air": False}),
        ("air", {"ensemble": False, "force": False, "stroke": False, "air": True}),
        ("force_air", {"ensemble": False, "force": True, "stroke": False, "air": True}),
    ]


def _meets(row) -> bool:
    lift = row["lift_10mm"]
    return (
        lift["scored"] > 0
        and lift["aligned"] / lift["scored"] >= 0.9
        and row["spurious_mid_stroke"] == 0
        and row.get("stopped") in (None, "null")
    )


def main() -> None:
    _check()
    OUT.mkdir(parents=True, exist_ok=True)
    frame = _frame_note()
    (OUT / "frame.json").write_text(json.dumps(frame, indent=2) + "\n")
    print("frame", json.dumps(frame), flush=True)
    axes = _axes()
    ends = _ends(axes)
    field = FixBMoldField(FIELD)
    starts = _demo_lifts(field)
    print(f"demo lifts {len(starts)}", flush=True)
    demo = pr._demo_episode(9)
    hook = Hook(axes, ends)
    _install(hook)
    rows = []
    for which in ("A", "v3r"):
        for name, flags in _catalog():
            rows.append(_run(hook, which, f"{which}_{name}", flags, demo, starts, field))
        (OUT / "summary.json").write_text(json.dumps({"frame": frame, "rows": rows}, indent=2) + "\n")
    helped = []
    for which in ("A", "v3r"):
        base = next(r for r in rows if r["name"] == f"{which}_none")
        ens = next(r for r in rows if r["name"] == f"{which}_dn_ensemble")
        if ens["lift_10mm"]["aligned"] > base["lift_10mm"]["aligned"]:
            helped.append(which)
    for which in helped:
        flags = {"ensemble": True, "force": True, "stroke": False, "air": True}
        rows.append(_run(hook, which, f"{which}_ens_force_air", flags, demo, starts, field))
    if not any(_meets(row) for row in rows):
        tuned = {
            "ensemble": False, "force": True, "stroke": False, "air": True,
            "off_n": 4.5, "press_dn_m": -0.004, "hold_s": 0.03,
            "retract_m": 0.015, "release_on_press": True,
        }
        rows.append(_run(hook, "v3r", "v3r_tuned", tuned, demo, starts, field))
    (OUT / "summary.json").write_text(json.dumps({"frame": frame, "rows": rows, "liftos": "no local checkpoint"}, indent=2) + "\n")
    _plots(rows)
    open_rows = []
    try:
        _open_loop(open_rows)
    except Exception as exc:  # dataset or normalizer gaps should not drop the closed-loop table
        open_rows.append({"error": str(exc)})
        print("open-loop failed", exc, flush=True)
    (OUT / "open_loop.json").write_text(json.dumps(open_rows, indent=2) + "\n")
    print("wrote", PLOT, flush=True)


if __name__ == "__main__":
    main()
