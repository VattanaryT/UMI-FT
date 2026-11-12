#!/usr/bin/env python3
"""Open-loop viewer. Reads export_openloop.py directories. No policy.

  python UMIFT_Data/surface_frames_scripts/view_openloop.py \\
    --export /mnt/data/umidata/ft/open_loop_exports/r4vt_<sha> \\
    --mesh /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz

Mesh loading, fixture-frame rounding, and bad-span shading are imported from
view_oct. The camera, time slider, and plot shading in this page are the same
JavaScript functions. They are copied here because they live inside view_oct's
HTML string and cannot be imported without changing that page.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import openloop_lib as ol
from view_oct import decimate_mesh, mesh_payload, rounded, shade_spans, stride_idx

PATH_CAP = 700
SERIES_CAP = 1200
COLORS = ("#e74c3c", "#2ecc71", "#f4d03f", "#3498db", "#e67e22")


def mask_spans(times: np.ndarray, mask: np.ndarray) -> list:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    spans = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = float(times[i])
        elif not flag and start is not None:
            spans.append([start, float(times[i - 1])])
            start = None
    if start is not None:
        spans.append([start, float(times[-1])])
    return spans


def checkpoint_label(settings: dict, fallback: str) -> str:
    """r4vt ep185 (val 0.079) from the export settings."""
    role = settings.get("role") or fallback
    epoch = settings.get("checkpoint_epoch")
    text = str(role) if epoch is None else f"{role} ep{int(epoch)}"
    val = settings.get("val_loss")
    if val is None:
        return text
    return f"{text} (val {float(val):.3f})"


def _time_cap(times: np.ndarray, floor: float) -> float:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if len(times) < 2:
        return floor
    return max(floor, 5.0 * float(np.median(np.diff(times))))


def edge_gaps(index, bad=None, long=None, times=None, dt_cap=None) -> np.ndarray:
    """True on edges that jump a sample, a long-dt flag, or an excluded span."""
    index = np.asarray(index)
    n_edge = max(0, len(index) - 1)
    gap = np.zeros(n_edge, dtype=bool)
    if n_edge == 0:
        return gap
    step = np.diff(np.asarray(index, dtype=np.float64))
    gap |= step > float(np.median(step)) + 0.5
    if times is not None and dt_cap is not None:
        gap |= np.diff(np.asarray(times, dtype=np.float64)) > float(dt_cap)
    if bad is not None:
        bad = np.asarray(bad, dtype=bool).reshape(-1)
        gap |= bad[:-1] | bad[1:]
    if long is not None:
        long = np.asarray(long, dtype=bool).reshape(-1)
        gap |= long[:-1] | long[1:]
    return gap


def breaks_for_keep(edge_gap: np.ndarray, keep: np.ndarray) -> list:
    """Break a downsampled segment when any original edge inside it is a gap."""
    keep = np.asarray(keep, dtype=int)
    return [bool(np.any(edge_gap[int(a) : int(b)])) for a, b in zip(keep[:-1], keep[1:])]


def spans_from_edges(times, edge_gap, t0: float) -> list:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    spans = []
    i = 0
    n = len(edge_gap)
    while i < n:
        if not edge_gap[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and edge_gap[j + 1]:
            j += 1
        spans.append([round(float(times[i] - t0), 3), round(float(times[j + 1] - t0), 3)])
        i = j + 1
    return spans


def load_export(path: Path) -> dict:
    settings = json.loads((path / "settings.json").read_text())
    report = {}
    report_path = path / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
    episodes = {}
    for npz in sorted(path.glob("episode_*.npz")):
        episodes[int(npz.stem.split("_")[-1])] = np.load(npz)
    if not episodes:
        raise SystemExit(f"no episode npz in {path}")
    label = checkpoint_label(settings, path.name)
    return {"dir": path, "settings": settings, "report": report, "episodes": episodes, "label": label}


def query_step0(blob, key: str) -> np.ndarray:
    return np.asarray(blob[key])[:, 0]


def episode_payload(exports: list, episode: int, mesh_t0_ok: bool = True) -> dict:
    first = None
    for exp in exports:
        if episode in exp["episodes"]:
            first = exp["episodes"][episode]
            break
    if first is None:
        raise KeyError(episode)
    robot_t = np.asarray(first["robot_time_s"], dtype=np.float64)
    rgb_t = np.asarray(first["rgb_time_s"], dtype=np.float64)
    t0 = float(rgb_t[0])
    ip = stride_idx(len(robot_t), PATH_CAP)
    xyz = np.asarray(first["tcp_xyz"], dtype=np.float64)
    R = np.asarray(first["tcp_R"], dtype=np.float64)
    force_t = np.asarray(first["force_series_t"], dtype=np.float64)
    force = np.asarray(first["force_series"], dtype=np.float64)
    iw = stride_idx(len(force_t), SERIES_CAP)
    bad = np.asarray(first["bad_robot"], dtype=bool)
    shade = shade_spans({str(episode): {"zarr_s": mask_spans(robot_t, bad)}}, episode, t0)
    robot_gap = edge_gaps(np.arange(len(robot_t)), bad=bad, times=robot_t, dt_cap=_time_cap(robot_t, 0.05))
    path_break = breaks_for_keep(robot_gap, ip)
    force_gap = edge_gaps(np.arange(len(force_t)), times=force_t, dt_cap=_time_cap(force_t, 0.2))
    force_break = breaks_for_keep(force_gap, iw)
    runs = []
    for exp in exports:
        blob = exp["episodes"].get(episode)
        if blob is None:
            continue
        q_t = np.asarray(blob["query_time_s"], dtype=np.float64) - t0
        pred = np.asarray(blob["pred_action"], dtype=np.float64)
        label = np.asarray(blob["label_action"], dtype=np.float64)
        iser = stride_idx(len(q_t), SERIES_CAP)
        q_gap = edge_gaps(
            np.asarray(blob["query_rgb_idx"]),
            bad=np.asarray(blob["bad_span"])[:, 0],
            long=np.asarray(blob["long_dt"])[:, 0],
        )
        q_break = breaks_for_keep(q_gap, iser)
        gaps = spans_from_edges(np.asarray(blob["query_time_s"], dtype=np.float64), q_gap, t0)
        step_pred = pred[:, 0]
        step_label = label[:, 0]
        ang = ol.delta_r_deg(step_pred, step_label)
        pos = np.asarray(blob["pred_pos"], dtype=np.float64)[:, 0]
        runs.append(
            {
                "label": exp["label"],
                "split": str(blob["split"]),
                "q_i": [int(i) for i in iser],
                "q_t": rounded(q_t[iser], 3),
                "q_break": q_break,
                "gaps": gaps,
                "path": rounded(pos[iser], 4),
                "v1_p": rounded(step_pred[iser, 0], 4),
                "v1_l": rounded(step_label[iser, 0], 4),
                "v2_p": rounded(step_pred[iser, 1], 4),
                "v2_l": rounded(step_label[iser, 1], 4),
                "n_p": rounded(step_pred[iser, 2], 4),
                "n_l": rounded(step_label[iser, 2], 4),
                "stiff_p": rounded(step_pred[iser, 18], 4),
                "stiff_l": rounded(step_label[iser, 18], 4),
                "ang": rounded(ang[iser], 3),
                "error": ol.episode_error_table(
                    *flatten_for_table(blob)
                ),
            }
        )
    # Full per-query chunks stay out of this summary. The cursor endpoint loads one.
    return {
        "episode": episode,
        "split": runs[0]["split"] if runs else "",
        "path_t": rounded(robot_t[ip] - t0, 3),
        "path": rounded(xyz[ip], 4),
        "path_break": path_break,
        "tool_R": rounded(R[ip], 4),
        "force_t": rounded(force_t[iw] - t0, 3),
        "force": rounded(force[iw], 2),
        "force_break": force_break,
        "shade": shade,
        "q_index": [int(i) for i in range(len(np.asarray(first["query_time_s"])))],
        "runs": runs,
    }


def flatten_for_table(blob):
    pred = np.asarray(blob["pred_action"])
    label = np.asarray(blob["label_action"])
    H = pred.shape[1]
    valid = (np.arange(H)[None, :] < np.asarray(blob["n_valid"])[:, None]).reshape(-1)
    return (
        pred.reshape(-1, pred.shape[-1]),
        label.reshape(-1, label.shape[-1]),
        np.asarray(blob["force_n"]).reshape(-1),
        np.asarray(blob["bad_span"]).reshape(-1),
        np.asarray(blob["long_dt"]).reshape(-1),
        valid,
    )


def cursor_payload(exports: list, episode: int, query: int) -> dict:
    out = []
    for exp in exports:
        blob = exp["episodes"].get(episode)
        if blob is None or query < 0 or query >= len(blob["query_time_s"]):
            continue
        n_valid = int(blob["n_valid"][query])
        t0 = float(np.asarray(blob["rgb_time_s"])[0])
        out.append(
            {
                "label": exp["label"],
                "future": rounded(np.asarray(blob["pred_pos"][query, :n_valid]), 4),
                "press": rounded(np.asarray(blob["pred_pos"][query, 0]), 4),
                "R": rounded(np.asarray(blob["pred_R"][query, 0]), 4),
                "t": round(float(blob["query_time_s"][query]) - t0, 3),
            }
        )
    return {"runs": out}


PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>open loop</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body { margin:0; font-family:sans-serif; background:#111; color:#ddd; }
.bar { display:flex; gap:12px; align-items:center; padding:8px 12px; flex-wrap:wrap; }
select, button { background:#222; color:#ddd; border:1px solid #444; padding:4px 8px; }
button.on { background:#1a5276; }
input[type=range] { width:220px; }
#view3d { height:42vh; }
#plots { height:52vh; }
table { border-collapse:collapse; margin:8px 12px 16px; font-size:13px; }
td, th { border:1px solid #333; padding:3px 8px; }
</style></head>
<body>
<div class="bar">
  <span>episode</span><select id="ep"></select>
  <span id="split"></span>
  <span>checkpoint</span><select id="ckpt"></select>
  <span id="toggles"></span>
  <span>time</span><input id="time" type="range" min="0" max="0" value="0">
  <span id="readout"></span>
</div>
<div id="view3d"></div>
<div id="plots"></div>
<table id="summary"></table>
<script>
const VIEWS = {
  side: {eye: {x: 0.15, y: -1.7, z: 1.7}, center: {x: 0, y: 0, z: 0}, up: {x: 0, y: -1, z: 0}},
  top: {eye: {x: 0, y: -2.6, z: 0.02}, center: {x: 0, y: 0, z: 0}, up: {x: 0, y: 0, z: -1}},
  end: {eye: {x: 2.6, y: -0.45, z: 0.35}, center: {x: 0, y: 0, z: 0}, up: {x: 0, y: -1, z: 0}}
};
let viewName = "side";
let sceneBox = null;
let mesh = null;
let data = null;
let fi = 0;
let shown = {};

function nearest(times, t) {
  if (!times || !times.length) return 0;
  let lo = 0, hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < t) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(times[lo - 1] - t) < Math.abs(times[lo] - t)) lo--;
  return lo;
}
function span(values) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return [lo, hi];
}
function sceneBoxFrom(extra) {
  const xs = mesh.x.slice(), ys = mesh.y.slice(), zs = mesh.z.slice();
  function add(pts) { (pts || []).forEach(p => { xs.push(p[0]); ys.push(p[1]); zs.push(p[2]); }); }
  add(data.path);
  (extra || []).forEach(add);
  const pad = 0.05;
  function rng(a) { const s = span(a); return [s[0] - pad, s[1] + pad]; }
  return {x: rng(xs), y: rng(ys), z: rng(zs)};
}
function axisLock(title, range) { return {title, range, autorange: false}; }
function sceneLayout() {
  const box = sceneBox || {x: [-1, 1], y: [-1, 1], z: [-1, 1]};
  return {
    paper_bgcolor: "#111", margin: {l: 0, r: 0, t: 0, b: 0}, showlegend: true,
    legend: {x: 0, y: 1, font: {color: "#ddd"}, bgcolor: "rgba(0,0,0,0)"},
    uirevision: "keep",
    scene: {
      uirevision: "keep", aspectmode: "data", bgcolor: "#1c1c1c",
      camera: VIEWS[viewName],
      xaxis: axisLock("x", box.x), yaxis: axisLock("y", box.y), zaxis: axisLock("z", box.z)
    }
  };
}
function glScene() {
  const gd = document.getElementById("view3d");
  return gd && gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene._scene;
}
function liveCamera() {
  const scene = glScene();
  if (scene && scene.getCamera) return scene.getCamera();
  return VIEWS[viewName];
}
function applyCamera(cam) {
  const scene = glScene();
  if (!scene || !cam || !scene.glplot) return;
  const full = {
    eye: cam.eye, center: cam.center, up: cam.up,
    projection: (cam.projection && cam.projection.type) ? cam.projection : {type: "perspective"}
  };
  scene.camera.lookAt.apply(scene, [
    [full.eye.x, full.eye.y, full.eye.z],
    [full.center.x, full.center.y, full.center.z],
    [full.up.x, full.up.y, full.up.z]
  ]);
  scene.glplot.redraw();
  const gd = document.getElementById("view3d");
  if (gd.layout && gd.layout.scene) gd.layout.scene.camera = full;
}
const DIM = {
  v1: {pred: "#4f7399", label: "#d5e6f5"},
  v2: {pred: "#3f7a58", label: "#d7f3e3"},
  n: {pred: "#7a5c8f", label: "#eadcf6"},
  ang: {pred: "#a8883d", label: "#f8efd0"},
  stiff: {pred: "#3d6f8c", label: "#d4ebf6"}
};
function withBreaks(xs, ys, cut) {
  const xo = [], yo = [];
  for (let i = 0; i < xs.length; i++) {
    xo.push(xs[i]);
    yo.push(ys[i]);
    if (cut && cut[i]) { xo.push(null); yo.push(null); }
  }
  return [xo, yo];
}
function line(x, y, name, color, axis, dash, cut) {
  const broken = withBreaks(x, y, cut);
  return {
    x: broken[0], y: broken[1], name,
    xaxis: axis, yaxis: axis.replace("x", "y"),
    line: {color, width: dash ? 1.3 : 2.0, dash: dash || "solid"}
  };
}
function shadeRect(n, t0, t1) {
  return {
    type: "rect", xref: "x" + n, yref: "y" + n + " domain",
    x0: t0, x1: t1, y0: 0, y1: 1,
    fillcolor: "rgba(245,176,65,0.22)", line: {width: 0}, layer: "below"
  };
}
function gapRect(n, t0, t1) {
  return {
    type: "rect", xref: "x" + n, yref: "y" + n + " domain",
    x0: t0, x1: t1, y0: 0, y1: 1,
    fillcolor: "rgba(255,255,255,0.10)", line: {width: 0}, layer: "below"
  };
}
function axesAt(origin, R, color, name) {
  const scale = 0.04;
  const x = [origin[0]], y = [origin[1]], z = [origin[2]];
  for (let a = 0; a < 3; a++) {
    x.push(origin[0] + scale * R[0][a], null);
    y.push(origin[1] + scale * R[1][a], null);
    z.push(origin[2] + scale * R[2][a], null);
  }
  return {type: "scatter3d", x, y, z, mode: "lines", name, line: {color, width: 4}, showlegend: false};
}
function runStyle(i, n) {
  if (n <= 1) return {path: "#ff9f1a", chunk: "#ff4fd8", press: "#ffe14a"};
  const hue = [25, 140, 200, 280, 0, 170, 320, 50][i % 8];
  return {path: "hsl(" + hue + ",85%,52%)", chunk: "hsl(" + hue + ",95%,64%)", press: "#ffe14a"};
}
function scatter(pts, name, color, width, cut) {
  const x = [], y = [], z = [];
  for (let i = 0; i < pts.length; i++) {
    x.push(pts[i][0]); y.push(pts[i][1]); z.push(pts[i][2]);
    if (cut && cut[i]) { x.push(null); y.push(null); z.push(null); }
  }
  return {type: "scatter3d", x, y, z, mode: "lines", name, line: {color, width: width || 4}};
}
function paintTable(run) {
  const dims = ["v1", "v2", "delta_n", "delta_R_deg", "stiffness"];
  const cols = ["contact", "free", "clean", "bad_span"];
  let html = "<tr><th>dim</th>" + cols.map(c => "<th>" + c + "</th>").join("") + "</tr>";
  dims.forEach(dim => {
    html += "<tr><td>" + dim + "</td>";
    cols.forEach(col => {
      const v = run.error.mae[dim][col];
      html += "<td>" + (v === null || v === undefined ? "—" : Number(v).toExponential(3)) + "</td>";
    });
    html += "</tr>";
  });
  document.getElementById("summary").innerHTML = html;
}
async function paintCursor() {
  const t = data.path_t[fi];
  const qTimes = data.runs[0] ? data.runs[0].q_t : data.path_t;
  const jLocal = nearest(qTimes, t);
  const j = data.runs[0] && data.runs[0].q_i ? data.runs[0].q_i[jLocal] : jLocal;
  const cam = liveCamera();
  const cursor = await (await fetch("/api/cursor?ep=" + data.episode + "&q=" + j)).json();
  const traces = [scatter(data.path, "demo TCP", "#3ec1ff", 5, data.path_break)];
  const p = data.path[nearest(data.path_t, t)];
  const demoR = data.tool_R[nearest(data.path_t, t)];
  traces.push(axesAt(p, demoR, "#ffffff", "demo axes"));
  data.runs.forEach((run, i) => {
    if (shown[run.label] === false) return;
    const style = runStyle(i, data.runs.length);
    traces.push(scatter(run.path, run.label, style.path, 5, run.q_break));
    const item = (cursor.runs || []).find(r => r.label === run.label);
    if (!item) return;
    traces.push(scatter(item.future, run.label + " chunk", style.chunk, 9));
    traces.push(axesAt(item.future[0], item.R, style.chunk, run.label + " axes"));
    traces.push({
      type: "scatter3d", x: [item.press[0]], y: [item.press[1]], z: [item.press[2]],
      mode: "markers", name: run.label + " press",
      marker: {size: 9, color: style.press, symbol: "diamond"}
    });
  });
  traces.unshift({
    type: "mesh3d", x: mesh.x, y: mesh.y, z: mesh.z, i: mesh.i, j: mesh.j, k: mesh.k,
    color: "#9aabb6", opacity: 0.45, name: "mold", hoverinfo: "skip", showlegend: false
  });
  sceneBox = sceneBoxFrom([data.path]);
  await Plotly.react("view3d", traces, sceneLayout(), {responsive: true});
  applyCamera(cam);
  const run = data.runs[Math.max(0, data.runs.findIndex(r => r.label === document.getElementById("ckpt").value))];
  const shapes = [];
  ["", "2", "3", "4", "5"].forEach(n => {
    (data.shade || []).forEach(s => shapes.push(shadeRect(n, s[0], s[1])));
    (run.gaps || []).forEach(s => shapes.push(gapRect(n, s[0], s[1])));
    shapes.push({type: "line", xref: "x" + n, yref: "y" + n + " domain", x0: t, x1: t, y0: 0, y1: 1, line: {color: "#fff", width: 1}});
  });
  const x = {domain: [0, 1], automargin: false, title: ""};
  const br = run.q_break;
  const plots = [
    line(run.q_t, run.v1_p, "v1 pred", DIM.v1.pred, "x", "dash", br),
    line(run.q_t, run.v2_p, "v2 pred", DIM.v2.pred, "x", "dash", br),
    line(run.q_t, run.v1_l, "v1 label", DIM.v1.label, "x", null, br),
    line(run.q_t, run.v2_l, "v2 label", DIM.v2.label, "x", null, br),
    line(data.force_t, data.force, "||F||", "#e67e22", "x2", null, data.force_break),
    line(run.q_t, run.n_p, "δn pred", DIM.n.pred, "x3", "dash", br),
    line(run.q_t, run.n_l, "δn label", DIM.n.label, "x3", null, br),
    line(run.q_t, run.ang, "ΔR deg", DIM.ang.pred, "x4", "dash", br),
    line(run.q_t, run.stiff_p, "stiff pred", DIM.stiff.pred, "x5", "dash", br),
    line(run.q_t, run.stiff_l, "stiff label", DIM.stiff.label, "x5", null, br)
  ];
  await Plotly.react("plots", plots, {
    paper_bgcolor: "#111", plot_bgcolor: "#1c1c1c", font: {color: "#ddd"},
    margin: {l: 64, r: 16, t: 28, b: 28}, legend: {orientation: "h", y: 1.12},
    shapes,
    xaxis: Object.assign({anchor: "y"}, x),
    yaxis: {title: "v m/s", domain: [0.80, 1], anchor: "x"},
    xaxis2: Object.assign({anchor: "y2", matches: "x"}, x),
    yaxis2: {title: "N", domain: [0.60, 0.78], anchor: "x2"},
    xaxis3: Object.assign({anchor: "y3", matches: "x"}, x),
    yaxis3: {title: "δn m", domain: [0.40, 0.58], anchor: "x3"},
    xaxis4: Object.assign({anchor: "y4", matches: "x"}, x),
    yaxis4: {title: "ΔR deg", domain: [0.20, 0.38], anchor: "x4"},
    xaxis5: Object.assign({anchor: "y5", matches: "x", title: "s"}, x),
    yaxis5: {title: "stiff", domain: [0, 0.18], anchor: "x5"}
  }, {responsive: true});
  paintTable(run);
  document.getElementById("readout").textContent = t.toFixed(2) + " s";
  document.getElementById("time").value = fi;
}
async function show() {
  const episode = document.getElementById("ep").value;
  data = await (await fetch("/api/episode?ep=" + episode)).json();
  document.getElementById("split").textContent = data.split;
  document.getElementById("time").max = Math.max(0, data.path_t.length - 1);
  fi = 0;
  const ckpt = document.getElementById("ckpt");
  ckpt.innerHTML = data.runs.map(r => "<option>" + r.label + "</option>").join("");
  const toggles = document.getElementById("toggles");
  toggles.innerHTML = "";
  data.runs.forEach(run => {
    if (!(run.label in shown)) shown[run.label] = true;
    const box = document.createElement("label");
    box.style.color = "#ddd";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = shown[run.label];
    input.onchange = () => { shown[run.label] = input.checked; paintCursor(); };
    box.appendChild(input);
    box.appendChild(document.createTextNode(" " + run.label));
    toggles.appendChild(box);
  });
  await paintCursor();
}
document.getElementById("time").oninput = () => { fi = +document.getElementById("time").value; paintCursor(); };
document.getElementById("ep").onchange = () => show();
document.getElementById("ckpt").onchange = () => paintCursor();
(async function init() {
  const meta = await (await fetch("/api/meta")).json();
  mesh = meta.mesh;
  const ep = document.getElementById("ep");
  ep.innerHTML = meta.episodes.map(e => "<option value='" + e + "'>" + e + "</option>").join("");
  show();
})();
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    exports: list = []
    mesh = None

    def log_message(self, fmt, *args):
        return

    def _send(self, code, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        args = dict(part.split("=", 1) for part in query.split("&") if "=" in part)
        if path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            return
        if path == "/api/meta":
            episodes = sorted({ep for exp in self.exports for ep in exp["episodes"]})
            body = json.dumps({"episodes": episodes, "mesh": self.mesh}).encode()
            self._send(200, body, "application/json")
            return
        try:
            episode = int(args.get("ep", "0"))
        except ValueError:
            self._send(400, b"{}", "application/json")
            return
        if path == "/api/episode":
            body = json.dumps(episode_payload(self.exports, episode)).encode()
            self._send(200, body, "application/json")
            return
        if path == "/api/cursor":
            body = json.dumps(cursor_payload(self.exports, episode, int(args.get("q", "0")))).encode()
            self._send(200, body, "application/json")
            return
        self._send(404, b"not found", "text/plain")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, action="append", required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    field = np.load(args.mesh)
    vertices, faces = decimate_mesh(field["vertices"], field["faces"], field["is_apron"])
    Handler.mesh = mesh_payload(vertices, faces)
    Handler.exports = [load_export(path) for path in args.export]
    print(f"[view_openloop] http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
