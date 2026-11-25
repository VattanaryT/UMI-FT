"""Swap one observation group between the in-place roll and demo ep9.

Reads liftos latest.ckpt and two hardware logs. Does not write them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
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
from rot6d import rot6d_rows_to_R  # noqa: E402

OUT = Path(__file__).resolve().parent / "climb_labels/swap_obs_pair.png"
RUNS = (
    "2026-11-24_liftos_hw_pe_guard",
    "2026-11-24_liftos_hw_pe_dnbias_i5600",
)
LO, HI = 6008, 6316
GROUPS = {
    "wrench": ("robot0_eef_wrench_left", "robot0_eef_wrench_right"),
    "rgb": ("rgb_0",),
    "eef": ("robot0_eef_pos", "robot0_eef_rot_axis_angle"),
    "geom": ("robot0_mold_geom",),
}


def _copy(obs: dict) -> dict:
    return {key: np.asarray(value) for key, value in obs.items()}


def _mix(base: dict, donor: dict, keys: tuple[str, ...]) -> dict:
    out = _copy(base)
    for key in keys:
        out[key] = np.asarray(donor[key])
    return out


def _twist_obs(obs: dict) -> float:
    tool = rot6d_rows_to_R(np.asarray(obs["robot0_eef_rot_axis_angle"])[-1]).reshape(3, 3)
    field = rot6d_rows_to_R(np.asarray(obs["robot0_mold_geom"])[-1, 4:10]).reshape(3, 3)
    return vr._twist(field, tool)


def _fn(obs: dict) -> float:
    wrench = np.asarray(obs["robot0_eef_wrench_left"])[-1, :3]
    tool = rot6d_rows_to_R(np.asarray(obs["robot0_eef_rot_axis_angle"])[-1]).reshape(3, 3)
    normal = np.asarray(obs["robot0_mold_geom"])[-1, 1:4]
    return abs(float(np.dot(tool @ wrench, normal)))


def _predict(policy, device, obs: dict) -> np.ndarray:
    batched = {
        "sparse": {
            key: torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device).unsqueeze(0)
            for key, value in obs.items()
        }
    }
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    with torch.no_grad():
        return policy.predict_action(batched)["sparse"][0].detach().cpu().numpy()


def _demo_table(demo: dict) -> dict:
    from field_v2_dataset import mold_local

    spec = zarr.open_group(vr.DEMO_ZARR, mode="r").attrs["field_v2_axes"]
    origin = np.asarray(spec["origin"])
    axes = np.asarray(spec["axes"])
    from fixb_labels import pose7_to_Rp

    index = np.arange(LO, HI + 1)
    rotation, position = pose7_to_Rp(demo["pose"][index])
    local = mold_local(position, origin, axes)
    twist = np.array(
        [vr._twist(demo["field"][i], rotation[k]) for k, i in enumerate(index)],
        dtype=np.float64,
    )
    return {"index": index, "mold_x": local[:, 0], "twist": twist}


def _match(table: dict, twist: float, mold_x: float) -> tuple[int, float, float]:
    d_twist = np.abs((table["twist"] - twist + 180.0) % 360.0 - 180.0)
    d_x = np.abs(table["mold_x"] - mold_x)
    score = (d_twist / 20.0) ** 2 + (d_x / 0.015) ** 2
    pick = int(np.argmin(score))
    return int(table["index"][pick]), float(d_twist[pick]), float(d_x[pick])


def _horizons(name: str) -> list[dict]:
    from PyriteConfig.tasks.umift.workcell_frames import load_workcell_extrinsics

    path = vr.ROOT / name
    group = zarr.open_group(str(vr._log_root(path)), mode="r")
    T = load_workcell_extrinsics().T_fixture_from_right_base
    rows = []
    for hname in vr._horizon_names(group):
        action = group[hname]["action"]
        geom = np.asarray(group[hname]["obs_policy"]["robot0_mold_geom"])
        field = rot6d_rows_to_R(geom[-1, 4:10]).reshape(3, 3)
        _, cmd_R = vr._pose_bundle(np.asarray(action["ts_virtual_targets_0"]), T)
        twist = abs(vr._twist(field, cmd_R[0]))
        if 30.0 <= twist < 150.0:
            obs = {key: np.asarray(group[hname]["obs_policy"][key]) for key in group[hname]["obs_policy"].keys()}
            rows.append({"h": int(hname.split("_")[1]), "cmd_twist": twist, "obs": obs})
    return rows


def main() -> None:
    demo = vr._demo_path()
    table = _demo_table(demo)
    policy, device = vr._policy()
    print(f"device {device} demo window mold_x {table['mold_x'].min():+.3f}..{table['mold_x'].max():+.3f}", flush=True)
    records = []
    images = []
    for name in RUNS:
        for row in _horizons(name):
            hw = row["obs"]
            obs_twist = _twist_obs(hw)
            mold_x = float(np.asarray(hw["robot0_eef_pos"])[-1, 0])
            index, d_twist, d_x = _match(table, obs_twist, mold_x)
            demo_obs = vr._demo_obs(demo, index)
            packs = {"hw": hw, "demo": demo_obs}
            for group, keys in GROUPS.items():
                packs[f"hw+demo {group}"] = _mix(hw, demo_obs, keys)
                packs[f"demo+hw {group}"] = _mix(demo_obs, hw, keys)
            stats = {}
            for label, obs in packs.items():
                action = _predict(policy, device, obs)
                stats[label] = (
                    float(np.median(action[:6, 1])),
                    float(np.median(action[:6, 2]) * 1e3),
                )
            rgb_hw = np.asarray(hw["rgb_0"], dtype=np.float32)
            rgb_demo = np.asarray(demo_obs["rgb_0"], dtype=np.float32)
            diff = float(np.mean(np.abs(rgb_hw - rgb_demo)))
            diff_now = float(np.mean(np.abs(rgb_hw[-1] - rgb_demo[-1])))
            records.append(
                {
                    "run": name,
                    "h": row["h"],
                    "cmd_twist": row["cmd_twist"],
                    "obs_twist": obs_twist,
                    "demo_idx": index,
                    "d_twist": d_twist,
                    "d_x_mm": d_x * 1e3,
                    "fn_hw": _fn(hw),
                    "fn_demo": _fn(demo_obs),
                    "rgb_diff": diff,
                    "rgb_diff_now": diff_now,
                    "v": stats,
                }
            )
            images.append((rgb_demo[-1], rgb_hw[-1], name, row["h"], index, d_twist, d_x * 1e3))
            v2 = stats["hw"][0]
            print(
                f"{name} h{row['h']} demo {index} dtw {d_twist:.1f}° dx {d_x*1e3:.1f} mm "
                f"v2 hw {v2:+.3f} demo {stats['demo'][0]:+.3f}",
                flush=True,
            )
    labels = ["hw", "demo"]
    labels += [f"hw+demo {name}" for name in GROUPS]
    labels += [f"demo+hw {name}" for name in GROUPS]
    print("\ncondition                v2 med    dn med mm    v2 p10    v2 p90")
    summary = {}
    for label in labels:
        v2 = np.array([row["v"][label][0] for row in records])
        dn = np.array([row["v"][label][1] for row in records])
        summary[label] = (float(np.median(v2)), float(np.median(dn)))
        print(
            f"{label:22s} {np.median(v2):+8.3f} {np.median(dn):+10.1f} "
            f"{np.percentile(v2, 10):+8.3f} {np.percentile(v2, 90):+8.3f}"
        )
    print(
        f"\nmatched n {len(records)} |F·n| hw med {np.median([r['fn_hw'] for r in records]):.2f} N "
        f"demo med {np.median([r['fn_demo'] for r in records]):.2f} N"
    )
    print(
        f"rgb mean abs  history {np.median([r['rgb_diff'] for r in records]):.3f}  "
        f"last frame {np.median([r['rgb_diff_now'] for r in records]):.3f}"
    )
    print(
        f"match twist med {np.median([r['d_twist'] for r in records]):.1f}°  "
        f"mold x med {np.median([r['d_x_mm'] for r in records]):.1f} mm"
    )
    best = int(np.argmin([row["d_twist"] + row["d_x_mm"] / 5.0 for row in records]))
    demo_img, hw_img, name, horizon, index, d_twist, d_x = images[best]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, image, title in (
        (axes[0], demo_img, f"ep9 idx {index}"),
        (axes[1], hw_img, f"{name.split('_')[-1]} h{horizon}"),
    ):
        ax.imshow(np.clip(np.moveaxis(image, 0, -1), 0, 1))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"matched pair  twist err {d_twist:.0f}°  mold-x err {d_x:.0f} mm")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
