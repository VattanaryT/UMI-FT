"""Open-loop force and lift check with the training-frame mold_geom.

ep9 uses the liftos zarr, whose mold_geom is that frame. The thin stringer
reorients its wall frames onto the thin mold PCA axes and carries them.
Does not write checkpoints, zarrs, or the held-out score.json.
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
from field_v2 import carry_frames, mold_frame, reorient  # noqa: E402
from field_v2_dataset import mold_local  # noqa: E402
from fixb_labels import pose7_to_Rp, so3_to_rot6d_rows  # noqa: E402
from PyriteUtility.data_pipeline.wrench_baseline_utils import time_based_moving_average  # noqa: E402

K = 3000.0
CAP_N = 25.0
CONTACT_N = 5.0
LIFTS = (1527, 2350, 4366, 4773, 5201, 6008)
WINDOW_S = 0.3
OUT = Path(__file__).resolve().parent / "climb_labels" / "force_openloop.png"
THIN = Path(__file__).resolve().parent / "thin_heldout" / "C_first" / "replay.zarr"
WALL = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/"
    "wall_fix/fixb_field_wall_vertical.npz"
)


def _predict(policy, device, obs: dict) -> np.ndarray:
    batched = {
        "sparse": {
            key: torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device).unsqueeze(0)
            for key, value in obs.items()
        }
    }
    torch.manual_seed(0)
    with torch.no_grad():
        return policy.predict_action(batched)["sparse"][0].detach().cpu().numpy()


def _load(zarr_path: str, episode: str) -> dict:
    group = zarr.open_group(zarr_path, mode="r")["data"][episode]
    pose = np.asarray(group["ts_pose_fb_0"])
    tool, position = pose7_to_Rp(pose)
    robot_t = np.asarray(group["robot_time_stamps_0"]).reshape(-1)
    wrench_t = np.asarray(group["wrench_time_stamps_0"]).reshape(-1)
    wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
    smooth = time_based_moving_average(wrench, wrench_t, 1.0)
    return {
        "pose": pose,
        "tool": tool,
        "position": position,
        "time": robot_t,
        "rgb_t": np.asarray(group["rgb_time_stamps_0"]).reshape(-1),
        "rgb": group["rgb_0"],
        "wrench": wrench,
        "wrench_t": wrench_t,
        "smooth": smooth,
        "sd": np.asarray(group["mold_signed_distance_0"]).reshape(-1),
        "normal": np.asarray(group["mold_normal_0"]),
        "rot6d": np.asarray(group["mold_R_field_rot6d_0"]),
        "frame": np.asarray(group["mold_R_field_0"]).reshape(-1, 3, 3),
        "dn": (
            np.asarray(group["fixb_delta_n_0"]).reshape(-1)
            if "fixb_delta_n_0" in group
            else None
        ),
    }


def _fn(demo: dict, index: np.ndarray, normal: np.ndarray) -> np.ndarray:
    j = np.clip(np.searchsorted(demo["wrench_t"], demo["time"][index]), 1, len(demo["wrench_t"]) - 1)
    left = np.abs(demo["time"][index] - demo["wrench_t"][j - 1]) <= np.abs(demo["time"][index] - demo["wrench_t"][j])
    j = np.where(left, j - 1, j)
    force = np.einsum("nij,nj->ni", demo["tool"][index], demo["smooth"][j, :3])
    return np.einsum("ni,ni->n", force, normal[index])


def _obs(demo: dict, index: int, origin, axes, geom: np.ndarray) -> dict:
    n = len(demo["time"])
    pose_index = np.clip(index - np.arange(50, -1, -1) * 16, 0, n - 1)
    rgb_index = int(np.clip(np.searchsorted(demo["rgb_t"], demo["time"][index]), 0, len(demo["rgb_t"]) - 1))
    rgb_ids = np.clip(rgb_index - np.arange(15, -1, -1) * 4, 0, len(demo["rgb"]) - 1)
    wrench_index = int(np.clip(np.searchsorted(demo["wrench_t"], demo["time"][index]), 0, len(demo["wrench"]) - 1))
    wrench_ids = np.clip(wrench_index - np.arange(117, -1, -1), 0, len(demo["wrench"]) - 1)
    images = np.asarray(demo["rgb"][rgb_ids])
    if images.dtype == np.uint8:
        images = np.moveaxis(images, -1, 1).astype(np.float32) / 255.0
    else:
        images = np.moveaxis(images, -1, 1).astype(np.float32)
        if float(np.nanmax(images)) > 1.5:
            images = images / 255.0
    rotation = demo["tool"][pose_index]
    return {
        "rgb_0": images,
        "robot0_eef_pos": mold_local(demo["position"][pose_index], origin, axes).astype(np.float32),
        "robot0_eef_rot_axis_angle": so3_to_rot6d_rows(rotation).astype(np.float32),
        "robot0_eef_wrench_left": demo["wrench"][wrench_ids, :6].astype(np.float32),
        "robot0_eef_wrench_right": np.zeros((118, 6), np.float32),
        "robot0_mold_geom": geom[pose_index].astype(np.float32),
    }


def _geom_from(demo: dict) -> np.ndarray:
    return np.concatenate(
        [demo["sd"][:, None], demo["normal"], demo["rot6d"]], axis=1
    ).astype(np.float32)


def _thin_geom(demo: dict) -> np.ndarray:
    blob = np.load(WALL)
    vertices = np.asarray(blob["vertices"], dtype=np.float64)
    apron = np.asarray(blob["is_apron"]).astype(bool) if "is_apron" in blob.files else np.zeros(len(vertices), bool)
    _origin, axes = mold_frame(vertices, apron)
    frame = carry_frames(reorient(demo["frame"], axes), demo["tool"])
    outward = -frame[:, :, 2]
    # reorient keeps column 2, so the stored signed distance stays valid.
    sd = demo["sd"]
    rot6d = so3_to_rot6d_rows(frame).astype(np.float32)
    return np.concatenate([sd.astype(np.float32)[:, None], outward.astype(np.float32), rot6d], axis=1)


def _contact(policy, device, demo, origin, axes, geom) -> np.ndarray:
    normal = geom[:, 1:4]
    fn = _fn(demo, np.arange(len(demo["time"])), normal)
    errors = []
    rgb_at = np.clip(np.searchsorted(demo["time"], demo["rgb_t"]), 0, len(demo["time"]) - 1)
    chosen = [int(i) for i in rgb_at if abs(float(fn[i])) >= CONTACT_N and i >= 50 * 16]
    print(f"  queries {len(chosen)}", flush=True)
    for n_done, index in enumerate(chosen):
        pred = _predict(policy, device, _obs(demo, int(index), origin, axes, geom))
        force = min(abs(float(pred[0, 2])) * K, CAP_N)
        errors.append(abs(force - abs(float(fn[index]))))
        if n_done and n_done % 40 == 0:
            print(f"  {n_done}/{len(chosen)}", flush=True)
    return np.asarray(errors, dtype=np.float64)


def _lifts(policy, device, demo, origin, axes, geom) -> list[dict]:
    rows = []
    normal = geom[:, 1:4]
    for index in LIFTS:
        t0 = float(demo["time"][index])
        label = demo["dn"][np.abs(demo["time"] - t0) <= WINDOW_S]
        queries = np.where(np.abs(demo["rgb_t"] - t0) <= WINDOW_S)[0]
        pred_dn = []
        pred_t = []
        for rgb in queries:
            robot = int(np.clip(np.searchsorted(demo["time"], demo["rgb_t"][rgb]), 0, len(demo["time"]) - 1))
            if robot < 50 * 16:
                continue
            pred = _predict(policy, device, _obs(demo, robot, origin, axes, geom))
            for step in range(len(pred)):
                stamp = float(demo["rgb_t"][rgb]) + step / 15.0
                if abs(stamp - t0) <= WINDOW_S:
                    pred_dn.append(float(pred[step, 2]))
                    pred_t.append(stamp - t0)
        pred_dn = np.asarray(pred_dn, dtype=np.float64)
        rows.append(
            {
                "index": index,
                "positive": bool(pred_dn.size and np.any(pred_dn > 0)),
                "peak_pred": float(pred_dn.max()) if pred_dn.size else float("nan"),
                "peak_label": float(np.max(label)) if label.size else float("nan"),
                "label_at": float(demo["dn"][index]),
                "t": np.asarray(pred_t),
                "dn": pred_dn,
                "label_t": demo["time"][np.abs(demo["time"] - t0) <= WINDOW_S] - t0,
                "label_dn": label,
                "fn": float(_fn(demo, np.array([index]), normal)[0]),
            }
        )
    return rows


def _stat(values: np.ndarray) -> str:
    if values.size == 0:
        return "n=0"
    return f"n={values.size}  median {np.median(values):.2f} N  p90 {np.percentile(values, 90):.2f} N"


def main() -> None:
    spec = zarr.open_group(vr.DEMO_ZARR, mode="r").attrs["field_v2_axes"]
    origin = np.asarray(spec["origin"])
    axes = np.asarray(spec["axes"])
    policy, device = vr._policy()
    print(f"device {device}", flush=True)
    ep9 = _load(vr.DEMO_ZARR, "episode_9")
    thin = _load(str(THIN), "episode_0")
    ep9_geom = _geom_from(ep9)
    thin_geom = _thin_geom(thin)
    print("[force] ep9 contact", flush=True)
    ep9_err = _contact(policy, device, ep9, origin, axes, ep9_geom)
    print(f"ep9 contact |cap(|δn|·{K:.0f}, {CAP_N:.0f}) − |F·n||  {_stat(ep9_err)}", flush=True)
    print("[force] thin contact", flush=True)
    thin_err = _contact(policy, device, thin, origin, axes, thin_geom)
    print(f"thin contact |cap(|δn|·{K:.0f}, {CAP_N:.0f}) − |F·n||  {_stat(thin_err)}", flush=True)
    lifts = _lifts(policy, device, ep9, origin, axes, ep9_geom)
    print("idx     positive  peak_pred_mm  peak_label_mm  label_at_mm")
    for row in lifts:
        print(
            f"{row['index']:6d}  {str(row['positive']):5s}     "
            f"{row['peak_pred']*1e3:8.1f}      {row['peak_label']*1e3:8.1f}       {row['label_at']*1e3:8.1f}"
        )
    fig, axes_plot = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
    for ax, row in zip(axes_plot.ravel(), lifts):
        ax.plot(row["label_t"], row["label_dn"] * 1e3, color="0.4", label="label")
        ax.scatter(row["t"], row["dn"] * 1e3, s=12, c="C0", label="pred")
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_title(f"idx {row['index']}  {'up' if row['positive'] else 'no'}")
        ax.set_xlabel("s from lift")
    axes_plot[0, 0].set_ylabel("δn mm")
    axes_plot[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"contact |F| err N   ep9 {_stat(ep9_err)}    thin {_stat(thin_err)}"
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
