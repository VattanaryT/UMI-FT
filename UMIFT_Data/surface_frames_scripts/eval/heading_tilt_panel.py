#!/usr/bin/env python3
"""Heading/tilt page for the open-loop viewer. Writes HTML; does not touch view_openloop.py."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [
    str(SCRIPTS),
    str(SCRIPTS / "eval"),
    str(ROOT / "PyriteEnvSuites/scripts"),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT),
]

from fixb_field import FixBMoldField
from heading_tilt import long_axis
from heading_tilt_43448962_open_loop import FIELD, build_dataset, decode_pose
from run_policy_open_loop_val_export import load_policy_and_cfg, predict_sparse_action

# build_dataset lives in the open-loop script and already points at the heading zarr.
CKPT = (
    "/mnt/data/umidata/ft/training_outputs/"
    "oct-v35-h4s-r4vt-heading-tilt_sc_e200_bs64_valtopk_43448962/checkpoints/latest.ckpt"
)
ZARR = (
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_heading_tilt/acp_replay_buffer.zarr"
)
PARENT = SCRIPTS / "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_heading_tilt_swing_sc.pkl"
OUT = SCRIPTS / "eval/heading_tilt_43448962_e100_ep9_ep28.html"
EPISODES = (9, 28)
RGB_STRIDE = 4
AXIS_M = 0.03


def _vec(params, key):
    return params[key].detach().float().cpu().reshape(-1).numpy()


def parent_ab():
    with PARENT.open("rb") as handle:
        saved = pickle.load(handle)
    src = saved.params_dict["action"]
    scale = _vec(src, "scale")
    offset = _vec(src, "offset")
    return scale[6:8], offset[6:8]


def psi_deg(action):
    return np.degrees(np.arctan2(action[:, 4], action[:, 5]))


def mag_deg(action):
    return np.degrees(np.linalg.norm(action[:, 6:8], axis=1))


def queries(dataset, episode: int) -> list[int]:
    chosen = []
    last = None
    for idx, (epi, rgb) in enumerate(dataset.sampler.indices):
        if int(epi) != episode:
            continue
        rgb = int(rgb)
        if last is None or rgb - last >= RGB_STRIDE:
            chosen.append(idx)
            last = rgb
    if not chosen:
        raise SystemExit(f"no queries for episode {episode}")
    return chosen


def label_trace(scale, offset, episode: int) -> dict:
    group = zarr.open(ZARR, mode="r")["data"][f"episode_{episode}"]
    tilt = np.asarray(group["fixb_heading_tilt_0"], dtype=np.float64)
    times = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    n = min(len(tilt), len(times))
    tilt, times = tilt[:n:2], times[:n:2]
    t = times - times[0]
    a = np.degrees(tilt[:, 2])
    b = np.degrees(tilt[:, 3])
    return {
        "t": t.tolist(),
        "psi": np.degrees(np.unwrap(np.arctan2(tilt[:, 0], tilt[:, 1]))).tolist(),
        "a": a.tolist(),
        "b": b.tolist(),
        "mag": np.degrees(np.linalg.norm(tilt[:, 2:4], axis=1)).tolist(),
        "a_n": (tilt[:, 2] * scale[0] + offset[0]).tolist(),
        "b_n": (tilt[:, 3] * scale[1] + offset[1]).tolist(),
    }


def step_pack(action, xyz, rotation):
    return {
        "psi": float(psi_deg(action[None])[0]),
        "a": float(np.degrees(action[6])),
        "b": float(np.degrees(action[7])),
        "mag": float(mag_deg(action[None])[0]),
        "xyz": xyz.tolist(),
        "z": rotation[:, 2].tolist(),
    }


# 1-based steps of a 64-step chunk. Four RGB frames is 0.27 s.
HORIZONS = ((0, "first"), (5, "step6"), (63, "step64"))


def predict_episode(dataset, policy, field, axis, device, episode: int) -> dict:
    group = zarr.open(ZARR, mode="r")["data"][f"episode_{episode}"]
    rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    t0 = float(rgb_t[0])
    series = {name: [] for _, name in HORIZONS}
    for idx in queries(dataset, episode):
        obs, action = dataset.sampler.sample_sequence(idx)
        pred = np.asarray(predict_sparse_action(policy, obs, device), dtype=np.float64).reshape(-1, 8)
        gt = np.asarray(action["sparse"], dtype=np.float64).reshape(-1, 8)
        n = min(len(pred), len(gt))
        if n < 64:
            raise SystemExit(f"episode {episode} chunk length {n}, need 64")
        pred, gt = pred[:n], gt[:n]
        sparse = obs["sparse"]
        p0 = np.asarray(sparse["robot0_eef_pos"][-1], dtype=np.float64).reshape(3)
        from mold_aware_runtime import rot6d_rows_to_R

        r0 = rot6d_rows_to_R(np.asarray(sparse["robot0_eef_rot_axis_angle"][-1], dtype=np.float64))
        xp, rp = decode_pose(field, axis, pred, p0, r0)
        xg, rg = decode_pose(field, axis, gt, p0, r0)
        epi, rgb = dataset.sampler.indices[idx]
        t_query = float(rgb_t[int(rgb)] - t0)
        for index, name in HORIZONS:
            t = t_query + index / 15.0
            series[name].append({
                "t": t,
                "label": step_pack(gt[index], xg[index], rg[index]),
                "pred": step_pack(pred[index], xp[index], rp[index]),
            })
        print(f"ep {episode} t={t_query:.1f}s  {len(series['first'])}", flush=True)
    return series


def _corr_lag(label, pred, limit=6):
    label = np.asarray(label, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    label = label - label.mean()
    pred = pred - pred.mean()
    best = (0, -2.0)
    for lag in range(-limit, limit + 1):
        if lag < 0:
            a, b = label[-lag:], pred[:lag]
        elif lag > 0:
            a, b = label[:-lag], pred[lag:]
        else:
            a, b = label, pred
        if len(a) < 8 or np.std(a) < 1e-6 or np.std(b) < 1e-6:
            continue
        corr = float(np.corrcoef(a, b)[0, 1])
        if corr > best[1]:
            best = (lag, corr)
    return best


def _sign_disagree(label, pred, dead=2.0):
    keep = (np.abs(label) > dead) & (np.abs(pred) > dead)
    if not np.any(keep):
        return 0.0
    return float(np.mean(np.sign(label[keep]) != np.sign(pred[keep])))


def notes(pack) -> dict:
    out = {}
    for episode, blob in pack["episodes"].items():
        rows = blob["pred"]["first"]
        for key in ("a", "b", "psi"):
            label = np.array([row["label"][key] for row in rows])
            pred = np.array([row["pred"][key] for row in rows])
            lag, corr = _corr_lag(label, pred)
            err = pred - label
            flips = float(np.mean(np.sign(err[1:]) * np.sign(err[:-1]) < 0)) if len(err) > 2 else 0.0
            out[f"{episode}_{key}"] = {
                "lag_steps": lag,
                "corr": round(corr, 3),
                "corr0": round(float(np.corrcoef(label - label.mean(), pred - pred.mean())[0, 1]), 3),
                "mae": round(float(np.mean(np.abs(err))), 2),
                "sign_disagree": round(_sign_disagree(label, pred), 3),
                "error_sign_flips": round(flips, 3),
            }
    return out


def _axis_trace(rows, who, name, color):
    x, y, z = [], [], []
    for row in rows:
        origin = row[who]["xyz"]
        direction = row[who]["z"]
        x += [origin[0], origin[0] + AXIS_M * direction[0], None]
        y += [origin[1], origin[1] + AXIS_M * direction[1], None]
        z += [origin[2], origin[2] + AXIS_M * direction[2], None]
    return {"type": "scatter3d", "mode": "lines", "name": name, "x": x, "y": y, "z": z, "line": {"color": color, "width": 4}}


def page(pack) -> str:
    payload = json.dumps(pack)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>heading tilt 43448962 e100</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body {{ margin: 0; background: #161616; color: #ddd; font: 14px sans-serif; }}
h2 {{ margin: 16px; }} #note {{ margin: 0 16px 16px; color: #aaa; }}</style>
</head><body>
<h2>43448962 epoch 100 · episodes 9 and 28</h2>
<p id="note">Queries every 0.27 s (4 RGB frames). Label is the line. Markers are the prediction at the first step, step 6, and step 64.
Dotted traces are the e93b8756 normalized label, on the right axis. Tool z is the green (label) and yellow (predicted) ticks at the first step.</p>
<div id="root"></div>
<script>
const DATA = {payload};
const COLORS = {{label: "#d0d4dc", first: "#ff9f1a", end: "#ff4fd8", norm: "#7eb6ff", zL: "#3ddc97", zP: "#ffd23f"}};
function col(rows, who, key) {{ return rows.map(r => r[who][key]); }}
function timePlots(ep) {{
  const blob = DATA.episodes[ep];
  const lab = blob.label;
  const horizons = [
    ["first", "pred step 1", COLORS.first, "circle"],
    ["step6", "pred step 6", "#4cc9f0", "diamond"],
    ["step64", "pred step 64", COLORS.end, "square"]
  ];
  const traces = [];
  ["psi", "a", "b", "mag"].forEach((key, i) => {{
    traces.push({{x: lab.t, y: lab[key], name: "label", mode: "lines", line: {{color: COLORS.label}}, xaxis: "x"+(i+1), yaxis: "y"+(i+1), legendgroup: "label", showlegend: i===0}});
    horizons.forEach(([name, title, color, symbol]) => {{
      const rows = blob.pred[name];
      traces.push({{x: rows.map(r => r.t), y: col(rows, "pred", key), name: title, mode: "markers", marker: {{color, size: 7, symbol}}, xaxis: "x"+(i+1), yaxis: "y"+(i+1), legendgroup: name, showlegend: i===0}});
    }});
    if (key === "a" || key === "b") {{
      traces.push({{x: lab.t, y: lab[key + "_n"], name: "label normalized", mode: "lines", line: {{color: COLORS.norm, dash: "dot"}}, xaxis: "x"+(i+1), yaxis: key === "a" ? "y5" : "y6", legendgroup: "norm", showlegend: key==="a"}});
    }}
  }});
  const domains = [[0.78, 1], [0.52, 0.74], [0.26, 0.48], [0.0, 0.22]];
  const layout = {{
    height: 900, paper_bgcolor: "#161616", plot_bgcolor: "#1e1e1e",
    font: {{color: "#ddd"}}, margin: {{l: 60, r: 70, t: 24, b: 40}},
    legend: {{orientation: "h"}}
  }};
  const titles = ["ψ deg", "a deg", "b deg", "|tilt| deg"];
  for (let i = 1; i <= 4; i++) {{
    layout["xaxis"+i] = {{title: i===4 ? "time s" : "", gridcolor: "#333", domain: [0, 1], anchor: "y"+i}};
    layout["yaxis"+i] = {{title: titles[i-1], gridcolor: "#333", domain: domains[i-1], anchor: "x"+i}};
  }}
  layout.yaxis5 = {{title: "norm", overlaying: "y2", side: "right", anchor: "x2", range: [-1.3, 1.3], showgrid: false}};
  layout.yaxis6 = {{title: "norm", overlaying: "y3", side: "right", anchor: "x3", range: [-1.3, 1.3], showgrid: false}};
  return {{traces, layout}};
}}
function view3d(ep) {{
  const rows = DATA.episodes[ep].pred.first;
  const traces = [
    _path(rows, "label", COLORS.label),
    _path(rows, "pred", COLORS.first),
    axis(rows, "label", "label z", COLORS.zL),
    axis(rows, "pred", "pred z", COLORS.zP)
  ];
  return {{traces, layout: {{
    height: 640, paper_bgcolor: "#161616",
    scene: {{aspectmode: "data", bgcolor: "#1e1e1e",
      xaxis: {{title: "x", color: "#aaa"}}, yaxis: {{title: "y", color: "#aaa"}}, zaxis: {{title: "z", color: "#aaa"}}}},
    margin: {{l: 0, r: 0, t: 30}}, font: {{color: "#ddd"}}, title: "tool z, first step"
  }}}};
}}
function _path(rows, who, color) {{
  return {{type: "scatter3d", mode: "lines", name: who + " path",
    x: rows.map(r => r[who].xyz[0]), y: rows.map(r => r[who].xyz[1]), z: rows.map(r => r[who].xyz[2]),
    line: {{color, width: 4}}}};
}}
function axis(rows, who, name, color) {{
  const x = [], y = [], z = [];
  rows.forEach(r => {{
    const p = r[who].xyz, d = r[who].z;
    x.push(p[0], p[0] + {AXIS_M} * d[0], null);
    y.push(p[1], p[1] + {AXIS_M} * d[1], null);
    z.push(p[2], p[2] + {AXIS_M} * d[2], null);
  }});
  return {{type: "scatter3d", mode: "lines", name, x, y, z, line: {{color, width: 4}}}};
}}
const root = document.getElementById("root");
Object.keys(DATA.episodes).forEach(ep => {{
  const h = document.createElement("h2");
  h.textContent = "episode " + ep;
  root.appendChild(h);
  const plots = document.createElement("div");
  const view = document.createElement("div");
  root.appendChild(plots); root.appendChild(view);
  const p = timePlots(ep), v = view3d(ep);
  Plotly.newPlot(plots, p.traces, p.layout, {{responsive: true}});
  Plotly.newPlot(view, v.traces, v.layout, {{responsive: true}});
}});
</script></body></html>
"""


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    scale, offset = parent_ab()
    policy, cfg = load_policy_and_cfg(CKPT, device)
    dataset = build_dataset(cfg).get_validation_dataset()
    field = FixBMoldField(Path(FIELD))
    axis = long_axis(field.vertices)
    episodes = {}
    for episode in EPISODES:
        episodes[str(episode)] = {
            "label": label_trace(scale, offset, episode),
            "pred": predict_episode(dataset, policy, field, axis, device, episode),
        }
    pack = {"ckpt": CKPT, "episodes": episodes}
    text = page(pack)
    OUT.write_text(text)
    print(json.dumps(notes(pack), indent=2), flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
