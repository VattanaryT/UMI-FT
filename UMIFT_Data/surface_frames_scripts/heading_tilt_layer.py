"""Heading-tilt and causal-foot frames for the open-loop viewer.

Loaded only when view_openloop.py is started with --heading-tilt.
The 8-d action is sin ψ, cos ψ, a, b. Causal e199 is the 10-d ΔR model.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
sys.path[:0] = [
    str(SCRIPTS),
    str(SCRIPTS / "eval"),
    str(ROOT / "PyriteEnvSuites/scripts"),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT),
]

import numpy as np
import torch

from fixb_field import FixBMoldField
from heading_tilt import e_long_sequence, geodesic_deg, long_axis, pack_tilt
from heading_tilt_43448962_open_loop import FIELD, decode_pose
from heading_tilt_vs_causal_foot import (
    CAUSAL_DIR,
    CAUSAL_NORM,
    CAUSAL_ZARR,
    HEAD_NORM,
    HEAD_ZARR,
    build_dataset,
    rotation_of_rot6d,
)
from mold_aware_runtime import rot6d_rows_to_R
from run_policy_open_loop_val_export import load_policy_and_cfg, predict_sparse_action

import zarr

HEADING_CKPT = (
    "/mnt/data/umidata/ft/training_outputs/"
    "oct-v35-h4s-r4vt-heading-tilt_sc_e200_bs64_valtopk_43448962/checkpoints/latest.ckpt"
)
CAUSAL_CKPT = f"{CAUSAL_DIR}/latest.ckpt"
RGB_STRIDE = 4
STEPS = ((0, "1"), (5, "6"), (63, "64"))


def chart_normal(tool_R, n_out):
    """Normal passed to pack_tilt so its z matches the tool z.

    frame_zero points z at −normal. The bare mold field stores z along n_out,
    and the causal tool is that frame times ΔR, so passing n_out unflipped
    puts the chart on the opposite pole.
    """
    z = np.asarray(tool_R, dtype=np.float64).reshape(-1, 3, 3)[:, :, 2]
    n = np.asarray(n_out, dtype=np.float64).reshape(-1, 3)
    sign = np.sign(np.sum(z * n, axis=1))
    sign[sign == 0.0] = 1.0
    return -sign[:, None] * n


def _queries(dataset, episode: int) -> list[tuple[int, float]]:
    group = None
    for zarr_path in (HEAD_ZARR, CAUSAL_ZARR):
        root = zarr.open(zarr_path, mode="r")["data"]
        name = f"episode_{episode}"
        if name in root:
            group = root[name]
            break
    if group is None:
        return []
    rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    t0 = float(rgb_t[0])
    chosen = []
    last = None
    for idx, (epi, rgb) in enumerate(dataset.sampler.indices):
        if int(epi) != int(episode):
            continue
        rgb = int(rgb)
        if last is not None and rgb - last < RGB_STRIDE:
            continue
        if rgb >= len(rgb_t):
            continue
        chosen.append((idx, float(rgb_t[rgb] - t0)))
        last = rgb
    return chosen


def _find_dataset(built, val, episode: int):
    for dataset in (built, val):
        if dataset is None:
            continue
        for _idx, (epi, _rgb) in enumerate(dataset.sampler.indices):
            if int(epi) == int(episode):
                return dataset
    return None


def _r(matrix) -> list:
    return np.round(np.asarray(matrix, dtype=np.float64), 4).tolist()


def _heading_steps(field, axis, action, p0, r0):
    xyz, rotation = decode_pose(field, axis, action, p0, r0)
    out = {}
    for index, name in STEPS:
        if index >= len(action):
            continue
        out[name] = {
            "origin": _r(xyz[index]),
            "R": _r(rotation[index]),
            "psi": float((np.degrees(np.arctan2(action[index, 4], action[index, 5])) + 180.0) % 360.0 - 180.0),
            "tilt": float(np.degrees(np.hypot(action[index, 6], action[index, 7]))),
        }
    return out


def _causal_steps(field, axis, action, p0, r0, label_pack_normal=None, label_e=None):
    xyz, rotation, n_out = rotation_of_rot6d(field, action, p0, r0)
    normal = chart_normal(rotation, n_out) if label_pack_normal is None else label_pack_normal
    e_long = e_long_sequence(normal, axis)[0] if label_e is None else label_e
    packed = pack_tilt(rotation, normal, e_long)
    out = {"normal": normal, "e_long": e_long, "steps": {}}
    for index, name in STEPS:
        if index >= len(action):
            continue
        out["steps"][name] = {
            "origin": _r(xyz[index]),
            "R": _r(rotation[index]),
            "psi": float((np.degrees(np.arctan2(packed[index, 0], packed[index, 1])) + 180.0) % 360.0 - 180.0),
            "tilt": float(np.degrees(np.hypot(packed[index, 2], packed[index, 3]))),
            "rotation": rotation[index],
        }
    return out


class HeadingLayer:
    def __init__(self, heading_ckpt: str = HEADING_CKPT, causal_ckpt: str = CAUSAL_CKPT):
        self.heading_ckpt = str(heading_ckpt)
        self.causal_ckpt = str(causal_ckpt)
        self.cache: dict[int, dict] = {}
        self._ready = False

    def _load(self):
        if self._ready:
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.field = FixBMoldField(Path(FIELD))
        self.axis = long_axis(self.field.vertices)
        policy, cfg = load_policy_and_cfg(self.heading_ckpt, device)
        self.h_policy = policy
        self.h_data = build_dataset(cfg, HEAD_ZARR, HEAD_NORM, "e93b8756", 8)
        self.h_val = self.h_data.get_validation_dataset()
        policy, cfg = load_policy_and_cfg(self.causal_ckpt, device)
        self.c_policy = policy
        self.c_data = build_dataset(cfg, CAUSAL_ZARR, CAUSAL_NORM, "14308b72", 10)
        self.c_val = self.c_data.get_validation_dataset()
        self._ready = True
        print("[heading] checkpoints loaded", flush=True)

    def episode(self, episode: int) -> dict:
        episode = int(episode)
        if episode not in self.cache:
            self.cache[episode] = self._episode(episode)
        return self.cache[episode]

    def _predict(self, policy, dataset, idx, width):
        obs, action = dataset.sampler.sample_sequence(idx)
        pred = np.asarray(predict_sparse_action(policy, obs, self.device), dtype=np.float64).reshape(-1, width)
        gt = np.asarray(action["sparse"], dtype=np.float64).reshape(-1, width)
        n = min(len(pred), len(gt))
        sparse = obs["sparse"]
        p0 = np.asarray(sparse["robot0_eef_pos"][-1], dtype=np.float64).reshape(3)
        r0 = rot6d_rows_to_R(np.asarray(sparse["robot0_eef_rot_axis_angle"][-1], dtype=np.float64))
        return pred[:n], gt[:n], p0, r0

    def _episode(self, episode: int) -> dict:
        self._load()
        h_set = _find_dataset(self.h_data, self.h_val, episode)
        if h_set is None:
            return {"t": [], "error": f"episode {episode} is not in the heading-tilt dataset"}
        rows = _queries(h_set, episode)
        steps = {name: _columns() for _index, name in STEPS}
        print(f"[heading] episode {episode} queries {len(rows)}", flush=True)
        c_set = _find_dataset(self.c_data, self.c_val, episode)
        c_rows = _queries(c_set, episode) if c_set is not None else []
        c_by_t = []
        for n, (idx, t) in enumerate(c_rows, start=1):
            pred, gt, p0, r0 = self._predict(self.c_policy, c_set, idx, 10)
            if len(pred) < 64:
                continue
            label = _causal_steps(self.field, self.axis, gt, p0, r0)
            got = _causal_steps(
                self.field, self.axis, pred, p0, r0, label["normal"], label["e_long"]
            )
            c_by_t.append((t, label["steps"], got["steps"]))
            if n % 20 == 0:
                print(f"[heading] causal {n}/{len(c_rows)}", flush=True)
        for n, (idx, t) in enumerate(rows, start=1):
            pred, gt, p0, r0 = self._predict(self.h_policy, h_set, idx, 8)
            if len(pred) < 64:
                continue
            hp = _heading_steps(self.field, self.axis, pred, p0, r0)
            hl = _heading_steps(self.field, self.axis, gt, p0, r0)
            j = None
            if c_by_t:
                j = int(np.argmin([abs(item[0] - t) for item in c_by_t]))
                if abs(c_by_t[j][0] - t) > 0.2:
                    j = None
            for _index, name in STEPS:
                if name not in hp or name not in hl:
                    continue
                slot = steps[name]
                slot["t"].append(round(t, 3))
                slot["h_origin_p"].append(hp[name]["origin"])
                slot["h_R_p"].append(hp[name]["R"])
                slot["h_origin_l"].append(hl[name]["origin"])
                slot["h_R_l"].append(hl[name]["R"])
                slot["psi_h_p"].append(round(hp[name]["psi"], 2))
                slot["psi_h_l"].append(round(hl[name]["psi"], 2))
                slot["tilt_h_p"].append(round(hp[name]["tilt"], 2))
                slot["tilt_h_l"].append(round(hl[name]["tilt"], 2))
                slot["geo_h"].append(round(float(geodesic_deg(np.asarray(hp[name]["R"])[None], np.asarray(hl[name]["R"])[None])[0]), 2))
                if j is None or name not in c_by_t[j][1] or name not in c_by_t[j][2]:
                    for key in ("c_origin_p", "c_R_p", "c_origin_l", "c_R_l", "psi_c_p", "psi_c_l", "tilt_c_p", "tilt_c_l", "geo_c"):
                        slot[key].append(None)
                    continue
                cp, cl = c_by_t[j][2][name], c_by_t[j][1][name]
                slot["c_origin_p"].append(cp["origin"])
                slot["c_R_p"].append(cp["R"])
                slot["c_origin_l"].append(cl["origin"])
                slot["c_R_l"].append(cl["R"])
                slot["psi_c_p"].append(round(cp["psi"], 2))
                slot["psi_c_l"].append(round(cl["psi"], 2))
                slot["tilt_c_p"].append(round(cp["tilt"], 2))
                slot["tilt_c_l"].append(round(cl["tilt"], 2))
                slot["geo_c"].append(round(float(geodesic_deg(cp["rotation"][None], cl["rotation"][None])[0]), 2))
            if n % 20 == 0:
                print(f"[heading] heading {n}/{len(rows)}", flush=True)
        t = steps["1"]["t"]
        return {
            "t": t,
            "steps": {name: {key: value for key, value in slot.items() if key != "t"} for name, slot in steps.items()},
            "heading_ckpt": self.heading_ckpt,
            "causal_ckpt": self.causal_ckpt,
        }


def _columns():
    keys = (
        "t",
        "h_origin_p", "h_R_p", "h_origin_l", "h_R_l",
        "c_origin_p", "c_R_p", "c_origin_l", "c_R_l",
        "psi_h_p", "psi_h_l", "psi_c_p", "psi_c_l",
        "tilt_h_p", "tilt_h_l", "tilt_c_p", "tilt_c_l",
        "geo_h", "geo_c",
    )
    return {key: [] for key in keys}
