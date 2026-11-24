#!/usr/bin/env python3
"""Episode 9 closed loop. Lift override off unless --lift. New files only.

compose=native is the executor as committed (smooth-field R @ ΔR).
compose=label uses mold_R_field_0's frame (field_v2 / checkpoint A).
compose=tcp uses R_tcp @ ΔR (field_v3r).
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
from closedloop_sim.label_frame import LabelFrame, match_episode  # noqa: E402
from closedloop_sim.lift_retract import LiftRetract, _check  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import pose7_to_Rp  # noqa: E402
from mold_aware_runtime import rot6d_rows_to_R  # noqa: E402

OUT = ROOT / "closedloop_sim" / "out_lift_ep9"
TRAIN = Path("/mnt/data/umidata/ft/training_outputs")
EPISODE = 9
CKPT = {
    "A": TRAIN / "oct_v35_h4s_r4vt_field_v2n/latest.ckpt",
    "v3r": TRAIN / "oct_v35_h4s_r4vt_field_v3r/latest_final.ckpt",
}


class Hook:
    def __init__(self):
        self.compose = "native"
        self.lift: LiftRetract | None = None
        self.labels = LabelFrame()
        self.angles: list[float] = []
        self._plant = None

    def reset(self, compose: str, lift: bool) -> None:
        self.compose = compose
        self.lift = LiftRetract() if lift else None
        self.labels.reset()
        self.angles = []
        self._plant = None

    def observe(self, plant, state) -> None:
        self._plant = plant
        if self.lift is not None:
            self.lift.observe(float(state["contact_n"]))

    def recompose(self, dec, p0, R0):
        if self.compose == "native":
            return dec
        raw = np.asarray(dec["raw"], dtype=np.float64)
        delta = rot6d_rows_to_R(raw[:, 21:27])
        if self.compose == "tcp":
            base = np.asarray(R0, dtype=np.float64).reshape(3, 3)
        else:
            base = self.labels.step(np.asarray(p0, dtype=np.float64).reshape(3), R0)
        native = np.asarray(dec["R_d"][0], dtype=np.float64)
        new = np.einsum("ij,njk->nik", base, delta)
        self.angles.append(_deg(native, new[0]))
        dec["R_field"] = np.repeat(base.reshape(1, 3, 3), len(delta), axis=0)
        dec["R_d"] = new
        return dec

    def rewrite(self, wp, t_s: float):
        if self.lift is None or self._plant is None:
            return wp
        plant = self._plant
        query = plant.field.query(plant.x.reshape(1, 3))
        normal = np.asarray(query["R"][0, :, 2], dtype=np.float64)
        normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
        clearance = float(query["sd"][0])
        away = float(np.dot(np.asarray(wp.position) - plant.x, normal)) > 0.0
        cmd = self.lift.command(float(wp.delta_n_m), away, clearance, t_s)
        if cmd is None:
            return wp
        foot = np.asarray(query["x_s"][0], dtype=np.float64)
        return pr.Waypoint(foot + cmd * normal, wp.rotation, wp.stiffness, wp.k_npm, cmd)


def _deg(a, b) -> float:
    c = float(np.clip((np.trace(np.asarray(a).T @ np.asarray(b)) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _install(hook: Hook) -> None:
    step = pr.CartesianSpringPlant.step
    sample = pr.Schedule.sample
    decode = pr.MoldAwareRuntime.decode_chunk

    def wrapped_step(self, x_cmd, rotation, stiffness, dt):
        state = step(self, x_cmd, rotation, stiffness, dt)
        hook.observe(self, state)
        return state

    def wrapped_sample(self, t):
        return hook.rewrite(sample(self, t), t)

    def wrapped_decode(self, raw_action, p0, R0, dt, apply_safety=True, frame_seed=None, label_R=None):
        dec = decode(
            self, raw_action, p0, R0, dt,
            apply_safety=apply_safety, frame_seed=frame_seed, label_R=label_R,
        )
        return hook.recompose(dec, p0, R0)

    pr.CartesianSpringPlant.step = wrapped_step
    pr.Schedule.sample = wrapped_sample
    pr.MoldAwareRuntime.decode_chunk = wrapped_decode


def _tilt(pose7) -> dict:
    R, _p = pose7_to_Rp(np.asarray(pose7, dtype=np.float64))
    if len(R) < 2:
        return {"n": 0, "max_deg": 0.0, "n_gt_10": 0}
    rel = np.einsum("nji,njk->nik", R[:-1], R[1:])
    c = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    deg = np.degrees(np.arccos(c))
    return {"n": int(deg.size), "max_deg": round(float(deg.max()), 2), "n_gt_10": int(np.sum(deg > 10.0))}


def _demo_lifts(field) -> tuple[np.ndarray, np.ndarray]:
    import zarr

    group = zarr.open_group(str(pr.ZARR), mode="r")["data"][f"episode_{EPISODE}"]
    pose = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    times = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    _R, position = pose7_to_Rp(pose)
    wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
    wt = np.asarray(group["wrench_time_stamps_left_0"], dtype=np.float64).reshape(-1)
    j = np.clip(np.searchsorted(wt, times), 1, len(wt) - 1)
    left = np.abs(times - wt[j - 1]) <= np.abs(times - wt[j])
    force = np.linalg.norm(wrench[np.where(left, j - 1, j), :3], axis=1)
    query = field.query(position)
    clearance = np.asarray(query["sd"], dtype=np.float64).reshape(-1)
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
    return t[on & ~prev], labels


def _lift_score(starts, sim_t, sim_c) -> dict:
    covered = float(sim_t[-1]) if len(sim_t) else -1.0
    ok = 0
    scored = 0
    for t0 in starts:
        if t0 < float(sim_t[0]) - 1e-6 or t0 + 1.0 > covered + 1e-6:
            continue
        scored += 1
        window = sim_c[(sim_t >= t0) & (sim_t <= t0 + 1.0)]
        if window.size and float(np.max(window)) > 0.010:
            ok += 1
    return {"reached_10mm": ok, "scored": scored}


def _losses(pen, t) -> dict:
    contact = np.asarray(pen) > 0.0
    if contact.size < 2:
        return {"falling_edges": 0, "mid_stroke": 0}
    edges = np.flatnonzero(contact[:-1] & ~contact[1:])
    mid = 0
    for i in edges:
        later = np.flatnonzero(contact[i + 1:])
        if later.size and float(t[i + 1 + later[0]] - t[i]) <= 0.5:
            mid += 1
    return {"falling_edges": int(edges.size), "mid_stroke": int(mid)}


def _run(hook: Hook, which: str, compose: str, lift: bool, demo, starts) -> dict:
    name = f"{which}_{compose}_lift{int(lift)}"
    hook.reset(compose, lift)
    pr.CHECKPOINTS[name] = str(CKPT[which])
    pr.OUT = OUT
    print(f"rollout {name}", flush=True)
    report = pr.rollout(name, demo, episode=EPISODE)
    if hook.lift is not None:
        hook.lift.finish()
    blob = np.load(OUT / f"policy_{name}_fixb_ep{EPISODE}.npz")
    field = FixBMoldField(str(pr.FIELD))
    query = field.query(blob["position"])
    clearance = np.asarray(query["sd"], dtype=np.float64).reshape(-1)
    losses = _losses(blob["penetration_m"], blob["t_s"])
    lifts = _lift_score(starts, blob["t_s"], clearance)
    retracts = [] if hook.lift is None else hook.lift.retracts
    demo_contact = None
    row = {
        "policy": which,
        "checkpoint": str(CKPT[which]),
        "compose": compose,
        "lift_override": lift,
        "rgb": "zeros",
        "stopped": report.get("stopped"),
        "duration_s": report.get("duration_s"),
        "lift_10mm": lifts,
        "contact_losses": losses,
        "retracts": retracts,
        "spurious_retracts": int(sum(1 for item in retracts if item.get("spurious"))),
        "tilt_step_gt_10deg": _tilt(blob["pose7"]),
        "compose_delta_deg_median": (
            None if not hook.angles else round(float(np.median(hook.angles)), 2)
        ),
        "final_clearance_mm": round(float(clearance[-1]) * 1e3, 2),
        "force_note": "sim contact force is the plant spring, not a wrist FT",
    }
    del demo_contact
    (OUT / f"{name}.json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)
    return row


def main() -> None:
    _check()
    OUT.mkdir(parents=True, exist_ok=True)
    frame = match_episode(EPISODE)
    (OUT / "frame_match.json").write_text(json.dumps(frame, indent=2) + "\n")
    print("frame", json.dumps(frame), flush=True)
    if frame["label_vs_mold_R_field_0_median_deg"] > 5.0:
        raise SystemExit("label frame does not match mold_R_field_0")
    if frame["online_vs_mold_R_field_0_median_deg"] > 5.0:
        raise SystemExit("online label frame does not match mold_R_field_0")
    hook = Hook()
    _install(hook)
    field = FixBMoldField(str(pr.FIELD))
    starts, _labels = _demo_lifts(field)
    print(f"demo lift starts {len(starts)}", flush=True)
    demo = pr._demo_episode(EPISODE)
    rows = []
    for which, compose, lift in (
        ("A", "native", False),
        ("A", "label", False),
        ("A", "label", True),
        ("v3r", "native", False),
        ("v3r", "tcp", False),
        ("v3r", "tcp", True),
    ):
        rows.append(_run(hook, which, compose, lift, demo, starts))
    (OUT / "summary.json").write_text(json.dumps({"frame": frame, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
