#!/usr/bin/env python3
"""Closed-loop sim for B (fair, no RGB) and A (sanity, sim RGB is zeros).

Writes only under closedloop_sim/out_ckpt_abc/closed. Does not touch training code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-field_v3r")
for entry in (ROOT, ROOT / "UMIFT_Data/surface_frames_scripts", ROOT / "PyriteML", ROOT / "PyriteConfig", ROOT / "PyriteUtility"):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)

import closedloop_sim.policy_rollout as pr  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import pose7_to_Rp  # noqa: E402

OUT = Path(__file__).resolve().parent / "out_ckpt_abc" / "closed"
TRAIN = Path("/mnt/data/umidata/ft/training_outputs")
EPISODE = 9


def _tilt(pose7) -> dict:
    R, _p = pose7_to_Rp(np.asarray(pose7, dtype=np.float64))
    if len(R) < 2:
        return {"n": 0, "max_deg": 0.0, "n_gt_10": 0}
    rel = np.einsum("nji,njk->nik", R[:-1], R[1:])
    c = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    deg = np.degrees(np.arccos(c))
    return {
        "n": int(deg.size),
        "max_deg": round(float(deg.max()), 2),
        "n_gt_10": int(np.sum(deg > 10.0)),
    }


def _clearance(field, position) -> np.ndarray:
    query = field.query(np.asarray(position, dtype=np.float64))
    normal = query["R"][:, :, 2]
    return np.sum((position - query["x_s"]) * -normal, axis=1)


def _summary(name, report, trace_pose) -> dict:
    segs = report.get("segments") or {}
    contact = report.get("contact_force_n") or {}
    return {
        "policy": name,
        "episode": EPISODE,
        "note": report.get("rgb_note"),
        "stopped": report.get("stopped"),
        "duration_s": report.get("duration_s"),
        "peak_normal_force_n": None if not contact else contact.get("max"),
        "force_note": "sim contact force is not trustworthy (registration drift)",
        "contact_loss_pct": report.get("contact_loss_pct"),
        "segments": segs,
        "path_vs_demo_mm": report.get("path_vs_demo"),
        "tilt_step_gt_10deg": _tilt(trace_pose),
        "final_clearance_mm": None,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pr.OUT = OUT
    pr.publish = lambda *args, **kwargs: {"port": None}
    load_policy = pr.load_policy

    def _load_policy(path, device):
        policy, shape = load_policy(path, device)
        if "rgb_0" not in shape["obs"]:
            from omegaconf import open_dict

            with open_dict(shape):
                if "rgb_0" in shape.sample.obs.sparse:
                    del shape.sample.obs.sparse["rgb_0"]
                if "rgb_0" in shape.raw:
                    del shape.raw["rgb_0"]
        return policy, shape

    pr.load_policy = _load_policy
    pr.CHECKPOINTS = {
        "B_v2n_norgb": str(TRAIN / "oct_v35_h4s_r4vt_field_v2n_norgb/latest.ckpt"),
        "A_v2n_sanity": str(TRAIN / "oct_v35_h4s_r4vt_field_v2n/latest.ckpt"),
    }
    demo = pr._demo_episode(EPISODE)
    rows = []
    for name, note in (
        ("B_v2n_norgb", "fair: policy has no RGB input"),
        ("A_v2n_sanity", "sanity only: sim feeds zero RGB to a policy trained with wrist images"),
    ):
        print(f"rollout {name}", flush=True)
        report = pr.rollout(name, demo, episode=EPISODE, mode="full")
        report["rgb_note"] = note
        tag = OUT / f"policy_{name}_full_fixb_ep{EPISODE}.json"
        tag.write_text(json.dumps(report, indent=2) + "\n")
        blob = np.load(OUT / f"policy_{name}_full_fixb_ep{EPISODE}.npz")
        row = _summary(name, report, blob["pose7"])
        field = FixBMoldField(str(pr.FIELD))
        clearance = _clearance(field, blob["position"])
        contact = np.asarray(blob["penetration_m"]) > 0
        losses = int(np.sum(contact[:-1] & ~contact[1:])) if contact.size else 0
        normal = np.sum(blob["force"] * blob["normal"], axis=1)
        row["contact_loss_count"] = losses
        row["peak_normal_force_n"] = round(float(np.max(normal)), 2) if normal.size else None
        row["final_clearance_mm"] = round(float(clearance[-1]) * 1e3, 2)
        row["peak_clearance_mm"] = round(float(np.max(clearance)) * 1e3, 2)
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
