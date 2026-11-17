#!/usr/bin/env python3
"""Open-loop viewer. Reads export_openloop.py directories. No policy.

  python UMIFT_Data/surface_frames_scripts/view_openloop.py \\
    --export /mnt/data/umidata/ft/open_loop_exports/r4vt_<sha> \\
    --mesh /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz

  # optional: old frames faded, new frames solid, old jumps marked
  python UMIFT_Data/surface_frames_scripts/view_openloop.py \\
    --export /mnt/data/umidata/ft/open_loop_exports/r4vt_<sha> \\
    --mesh /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz \\
    --frames_npz /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_frame_continuity_causal/train_episode_frames.npz

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


def _step_deg(R: np.ndarray) -> np.ndarray:
    """Per-sample geodesic step of a frame series, degrees. Sample 0 is 0."""
    R = np.asarray(R, dtype=np.float64)
    out = np.zeros(len(R), dtype=np.float64)
    if len(R) > 1:
        rel = np.einsum("nji,njk->nik", R[:-1], R[1:])
        c = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
        out[1:] = np.degrees(np.arccos(c))
    return out


def _event_rows(blob, prefix: str, with_xyz: bool) -> list:
    if f"{prefix}_episode" not in blob.files:
        return []
    rows = []
    n = len(blob[f"{prefix}_episode"])
    xyz = np.asarray(blob[f"{prefix}_xyz"]) if with_xyz and f"{prefix}_xyz" in blob.files else None
    for i in range(n):
        row = {
            "ep": int(blob[f"{prefix}_episode"][i]),
            "i": int(blob[f"{prefix}_index"][i]),
            "time_s": float(blob[f"{prefix}_time_s"][i]),
            "frame": round(float(blob[f"{prefix}_frame_deg"][i]), 3),
            "tool": round(float(blob[f"{prefix}_tool_deg"][i]), 3),
        }
        if xyz is not None:
            row["x"] = [round(float(v), 4) for v in xyz[i]]
        rows.append(row)
    return rows


def load_frames_npz(path: Path) -> dict:
    """epN -> old/new feet and frames. Optional event tables stay under "events"."""
    blob = np.load(path)
    out = {}
    for name in blob.files:
        if not (name.startswith("ep") and name.endswith("_R_old")):
            continue
        ep = int(name[2:].split("_", 1)[0])
        prefix = f"ep{ep}_"
        pack = {
            "x_old": np.asarray(blob[f"{prefix}x_old"]),
            "x_new": np.asarray(blob[f"{prefix}x_new"]),
            "R_old": np.asarray(blob[f"{prefix}R_old"]),
            "R_new": np.asarray(blob[f"{prefix}R_new"]),
            "old_jump": np.asarray(blob[f"{prefix}old_jump"]).astype(bool),
        }
        if f"{prefix}robot_time_s" in blob.files:
            pack["robot_time_s"] = np.asarray(blob[f"{prefix}robot_time_s"], dtype=np.float64)
            pack["t0"] = float(np.asarray(blob[f"{prefix}t0"]).reshape(-1)[0])
            pack["tcp_xyz"] = np.asarray(blob[f"{prefix}tcp_xyz"])
            pack["tcp_R"] = np.asarray(blob[f"{prefix}tcp_R"])
            pack["split"] = str(np.asarray(blob[f"{prefix}split"]).reshape(-1)[0])
        out[ep] = pack
    if not out:
        raise SystemExit(f"no ep*_R_old arrays in {path}")
    switches = _event_rows(blob, "switch", True)
    nearcap = _event_rows(blob, "nearcap", False)
    for row in switches + nearcap:
        pack = out.get(row["ep"])
        if pack and "t0" in pack:
            row["t"] = round(float(row["time_s"]) - float(pack["t0"]), 3)
    if switches or nearcap:
        out["events"] = {"switches": switches, "nearcap": nearcap}
    return out


def frame_episode_ids(frames: dict | None) -> list:
    if not frames:
        return []
    return sorted(ep for ep in frames if isinstance(ep, int))


def _switches_for(frames: dict | None, episode: int, t0: float) -> list:
    events = (frames or {}).get("events") or {}
    rows = []
    for row in events.get("switches") or []:
        if int(row["ep"]) != int(episode):
            continue
        item = {
            "t": round(float(row["time_s"]) - t0, 3),
            "frame": row["frame"],
            "tool": row["tool"],
            "i": row["i"],
        }
        if "x" in row:
            item["x"] = row["x"]
        rows.append(item)
    return rows


def _nearest_index(times: np.ndarray, t: float) -> int:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    i = int(np.searchsorted(times, t))
    if i <= 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    if abs(float(times[i - 1]) - t) < abs(float(times[i]) - t):
        return i - 1
    return i


def frame_overlay(pack: dict, robot_t: np.ndarray, t0: float) -> dict | None:
    """Downsampled |ΔR| and the old jump feet. None when the series length differs."""
    robot_t = np.asarray(robot_t, dtype=np.float64).reshape(-1)
    if len(pack["R_old"]) != len(robot_t) or len(pack["R_new"]) != len(robot_t):
        return None
    d_old = _step_deg(pack["R_old"])
    d_new = _step_deg(pack["R_new"])
    ip = stride_idx(len(robot_t), SERIES_CAP)
    gap = edge_gaps(np.arange(len(robot_t)), times=robot_t, dt_cap=_time_cap(robot_t, 0.05))
    jump = np.asarray(pack["old_jump"], dtype=bool).reshape(-1)
    return {
        "t": rounded(robot_t[ip] - t0, 3),
        "dR_old": rounded(d_old[ip], 3),
        "dR_new": rounded(d_new[ip], 3),
        "break": breaks_for_keep(gap, ip),
        "jump_t": rounded(robot_t[jump] - t0, 3),
        "jump_y": rounded(d_old[jump], 3),
        "jumps": rounded(np.asarray(pack["x_old"])[jump], 4),
    }


def frame_at(pack: dict, robot_t: np.ndarray, t_abs: float) -> dict:
    i = _nearest_index(robot_t, t_abs)
    return {
        "x_old": rounded(np.asarray(pack["x_old"])[i], 4),
        "R_old": rounded(np.asarray(pack["R_old"])[i], 4),
        "x_new": rounded(np.asarray(pack["x_new"])[i], 4),
        "R_new": rounded(np.asarray(pack["R_new"])[i], 4),
        "jump": bool(np.asarray(pack["old_jump"])[i]),
    }


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


def episode_payload(exports: list, episode: int, mesh_t0_ok: bool = True, frames: dict | None = None) -> dict:
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
        if pred.shape[-1] >= 27:
            ang = ol.delta_r_deg(step_pred, step_label)
        else:
            ang = ol.relative_command_rot_deg(step_pred, step_label)
        if "exec_pos" in blob.files:
            exec_pos = np.asarray(blob["exec_pos"], dtype=np.float64)
            exec_edge = np.asarray(blob["exec_break"], dtype=bool)
            ipath = stride_idx(len(exec_pos), PATH_CAP)
            path = rounded(exec_pos[ipath], 4)
            path_break = breaks_for_keep(exec_edge, ipath)
        else:
            pos = np.asarray(blob["pred_pos"], dtype=np.float64)[:, 0]
            path = rounded(pos[iser], 4)
            path_break = q_break
        runs.append(
            {
                "label": exp["label"],
                "role": exp["settings"].get("role", "r4vt"),
                "split": str(blob["split"]),
                "q_i": [int(i) for i in iser],
                "q_t": rounded(q_t[iser], 3),
                "q_break": q_break,
                "gaps": gaps,
                "path": path,
                "path_break": path_break,
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
    payload = {
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
    if frames and episode in frames:
        overlay = frame_overlay(frames[episode], robot_t, t0)
        if overlay is not None:
            payload["frames"] = overlay
        switches = _switches_for(frames, episode, t0)
        if switches:
            payload["switches"] = switches
        if "t0" in frames[episode] or switches:
            payload["t0"] = t0
    return payload


def frames_only_payload(frames: dict, episode: int) -> dict:
    """Foot path and frame step when this episode has no open-loop export."""
    pack = frames.get(episode)
    if not pack or "robot_time_s" not in pack:
        raise KeyError(episode)
    robot_t = np.asarray(pack["robot_time_s"], dtype=np.float64)
    t0 = float(pack["t0"])
    ip = stride_idx(len(robot_t), PATH_CAP)
    gap = edge_gaps(np.arange(len(robot_t)), times=robot_t, dt_cap=_time_cap(robot_t, 0.05))
    overlay = frame_overlay(pack, robot_t, t0)
    payload = {
        "episode": episode,
        "split": pack.get("split", ""),
        "policy": False,
        "path_t": rounded(robot_t[ip] - t0, 3),
        "path": rounded(np.asarray(pack["tcp_xyz"])[ip], 4),
        "path_break": breaks_for_keep(gap, ip),
        "tool_R": rounded(np.asarray(pack["tcp_R"])[ip], 4),
        "force_t": [],
        "force": [],
        "force_break": [],
        "shade": [],
        "q_index": [],
        "runs": [],
        "path_i": [int(i) for i in ip],
        "t0": t0,
        "frames": overlay,
    }
    switches = _switches_for(frames, episode, t0)
    if switches:
        payload["switches"] = switches
    return payload


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


def cursor_payload(exports: list, episode: int, query: int, frames: dict | None = None) -> dict:
    out = []
    frame = None
    for exp in exports:
        blob = exp["episodes"].get(episode)
        if blob is None or query < 0 or query >= len(blob["query_time_s"]):
            continue
        t0 = float(np.asarray(blob["rgb_time_s"])[0])
        if frame is None and frames and episode in frames and "robot_time_s" in blob.files:
            robot_t = np.asarray(blob["robot_time_s"], dtype=np.float64)
            if len(frames[episode]["R_old"]) == len(robot_t):
                t_abs = float(np.asarray(blob["query_time_s"])[query])
                frame = frame_at(frames[episode], robot_t, t_abs)
        if "chunk_pos" in blob.files:
            n_chunk = int(blob["chunk_n"][query])
            future = np.asarray(blob["chunk_pos"][query, :n_chunk])
            press = np.asarray(blob["chunk_pos"][query, 0])
            rot = np.asarray(blob["chunk_R"][query, 0])
        else:
            n_valid = int(blob["n_valid"][query])
            future = np.asarray(blob["pred_pos"][query, :n_valid])
            press = np.asarray(blob["pred_pos"][query, 0])
            rot = np.asarray(blob["pred_R"][query, 0])
        out.append(
            {
                "label": exp["label"],
                "future": rounded(future, 4),
                "press": rounded(press, 4),
                "R": rounded(rot, 4),
                "t": round(float(blob["query_time_s"][query]) - t0, 3),
            }
        )
    if frame is None and frames and episode in frames and not out:
        pack = frames[episode]
        if "robot_time_s" in pack:
            robot_t = np.asarray(pack["robot_time_s"], dtype=np.float64)
            i = int(np.clip(query, 0, len(robot_t) - 1))
            frame = frame_at(pack, robot_t, float(robot_t[i]))
    payload = {"runs": out}
    if frame is not None:
        payload["frame"] = frame
    return payload


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
#events { display:none; margin:8px 12px 24px; font-size:13px; max-height:46vh; overflow:auto; }
#events table { margin:4px 0 12px; }
#events tr[data-kind] { cursor:pointer; }
#events tr[data-kind]:hover { background:#222; }
#events h3 { margin:12px 0 4px; font-size:14px; }
table { border-collapse:collapse; margin:8px 12px 16px; font-size:13px; }
td, th { border:1px solid #333; padding:3px 8px; }
</style></head>
<body>
<div class="bar">
  <span>episode</span><select id="ep"></select>
  <span id="split"></span>
  <span>checkpoint</span><select id="ckpt"></select>
  <span id="toggles"></span>
  <span id="frame_toggles"></span>
  <span>time</span><input id="time" type="range" min="0" max="0" value="0">
  <span id="readout"></span>
</div>
<div id="view3d"></div>
<div id="plots"></div>
<table id="summary"></table>
<div id="events"></div>
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
let shownFrame = {old: true, new: true};

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
function axesAt(origin, R, color, name, width) {
  const scale = 0.04;
  const x = [origin[0]], y = [origin[1]], z = [origin[2]];
  for (let a = 0; a < 3; a++) {
    x.push(origin[0] + scale * R[0][a], null);
    y.push(origin[1] + scale * R[1][a], null);
    z.push(origin[2] + scale * R[2][a], null);
  }
  return {type: "scatter3d", x, y, z, mode: "lines", name, line: {color, width: width || 4}, showlegend: false};
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
  const dims = Object.keys(run.error.mae);
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
async function paintFrameSeries(t) {
  const shapes = [{
    type: "line", xref: "x", yref: "y domain", x0: t, x1: t, y0: 0, y1: 1,
    line: {color: "#fff", width: 1}
  }];
  (data.switches || []).forEach(s => shapes.push({
    type: "line", xref: "x", yref: "y domain", x0: s.t, x1: s.t, y0: 0, y1: 1,
    line: {color: "#5dade2", width: 1, dash: "dot"}
  }));
  const plots = [
    line(data.frames.t, data.frames.dR_old, "old |ΔR|", "rgba(255,255,255,0.55)", "x", "dash", data.frames.break),
    line(data.frames.t, data.frames.dR_new, "new |ΔR|", "#ff9f1a", "x", null, data.frames.break),
    {
      x: data.frames.jump_t, y: data.frames.jump_y, name: "old jump",
      type: "scatter", mode: "markers",
      marker: {color: "#e74c3c", size: 7, symbol: "diamond"}
    }
  ];
  if (data.switches && data.switches.length) {
    plots.push({
      x: data.switches.map(s => s.t), y: data.switches.map(s => s.frame),
      name: "crown-wall", type: "scatter", mode: "markers",
      marker: {color: "#5dade2", size: 9, symbol: "triangle-up"}
    });
  }
  await Plotly.react("plots", plots, {
    paper_bgcolor: "#111", plot_bgcolor: "#1c1c1c", font: {color: "#ddd"},
    margin: {l: 64, r: 16, t: 28, b: 28}, legend: {orientation: "h", y: 1.12},
    shapes,
    xaxis: {title: "s"},
    yaxis: {title: "frame deg"}
  }, {responsive: true});
}
function buildEvents(meta) {
  const root = document.getElementById("events");
  const sw = meta.switches || [];
  const cap = meta.nearcap || [];
  let html = "<h3>Crown-to-wall (" + sw.length + ")</h3>";
  html += "<table><tr><th>ep</th><th>sample</th><th>time</th><th>frame step</th><th>tool</th></tr>";
  sw.forEach((row, n) => {
    html += "<tr data-kind='switch' data-n='" + n + "'><td>" + row.ep + "</td><td>" + row.i
      + "</td><td>" + Number(row.t).toFixed(2) + " s</td><td>" + Number(row.frame).toFixed(2)
      + "°</td><td>" + Number(row.tool).toFixed(2) + "°</td></tr>";
  });
  html += "</table><h3>Near-cap steps (" + cap.length + ")</h3>";
  const groups = {};
  cap.forEach((row, n) => { (groups[row.ep] = groups[row.ep] || []).push(n); });
  Object.keys(groups).map(Number).sort((a, b) => a - b).forEach(ep => {
    html += "<details><summary>ep " + ep + " · " + groups[ep].length + "</summary><table>";
    groups[ep].forEach(n => {
      const row = cap[n];
      html += "<tr data-kind='near' data-n='" + n + "'><td>" + row.i + "</td><td>"
        + Number(row.t).toFixed(2) + " s</td><td>" + Number(row.frame).toFixed(2)
        + "°</td><td>tool " + Number(row.tool).toFixed(2) + "°</td></tr>";
    });
    html += "</table></details>";
  });
  root.innerHTML = html;
  root.style.display = "block";
  root.onclick = async (ev) => {
    const tr = ev.target.closest("tr");
    if (!tr || !tr.dataset.kind) return;
    const row = (tr.dataset.kind === "switch" ? sw : cap)[+tr.dataset.n];
    await jumpTo(row.ep, row.t);
  };
}
async function jumpTo(ep, tRel) {
  const sel = document.getElementById("ep");
  if (![...sel.options].some(opt => opt.value === String(ep))) return;
  if (String(sel.value) !== String(ep)) {
    sel.value = String(ep);
    await show();
  }
  fi = nearest(data.path_t, tRel);
  document.getElementById("time").value = fi;
  await paintCursor();
}
async function paintCursor() {
  const t = data.path_t[fi];
  const qTimes = data.runs[0] ? data.runs[0].q_t : data.path_t;
  const jLocal = nearest(qTimes, t);
  let j = jLocal;
  if (data.path_i) j = data.path_i[Math.min(fi, data.path_i.length - 1)];
  else if (data.runs[0] && data.runs[0].q_i) j = data.runs[0].q_i[jLocal];
  const cam = liveCamera();
  const cursor = await (await fetch("/api/cursor?ep=" + data.episode + "&q=" + j)).json();
  const traces = [scatter(data.path, "demo TCP", "#3ec1ff", 5, data.path_break)];
  const p = data.path[nearest(data.path_t, t)];
  const demoR = data.tool_R[nearest(data.path_t, t)];
  traces.push(axesAt(p, demoR, "#ffffff", "demo axes"));
  if (data.frames && cursor.frame && shownFrame.old) {
    traces.push(axesAt(cursor.frame.x_old, cursor.frame.R_old, "rgba(255,255,255,0.28)", "old frame", 3));
  }
  if (data.frames && cursor.frame && shownFrame.new) {
    traces.push(axesAt(cursor.frame.x_new, cursor.frame.R_new, "#ff9f1a", "new frame", 6));
  }
  if (data.switches && data.switches.length) {
    const sx = [], sy = [], sz = [];
    data.switches.forEach(s => {
      if (!s.x) return;
      sx.push(s.x[0]); sy.push(s.x[1]); sz.push(s.x[2]);
    });
    if (sx.length) traces.push({
      type: "scatter3d", x: sx, y: sy, z: sz, mode: "markers", name: "crown-wall",
      marker: {size: 5, color: "#5dade2", symbol: "diamond"}
    });
  }
  if (data.frames && shownFrame.old && data.frames.jumps && data.frames.jumps.length) {
    const jx = [], jy = [], jz = [];
    data.frames.jumps.forEach(pt => { jx.push(pt[0]); jy.push(pt[1]); jz.push(pt[2]); });
    traces.push({
      type: "scatter3d", x: jx, y: jy, z: jz, mode: "markers", name: "old jump",
      marker: {size: 3.5, color: "#e74c3c", symbol: "diamond"}
    });
  }
  const visible = data.runs.filter(run => shown[run.label] !== false);
  data.runs.forEach(run => {
    if (shown[run.label] === false) return;
    const style = runStyle(visible.indexOf(run), visible.length);
    traces.push(scatter(run.path, run.label, style.path, 5, run.path_break));
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
  if (data.policy === false) {
    await paintFrameSeries(t);
    document.getElementById("summary").innerHTML = "";
    document.getElementById("readout").textContent = t.toFixed(2) + " s";
    document.getElementById("time").value = fi;
    return;
  }
  const run = data.runs[Math.max(0, data.runs.findIndex(r => r.label === document.getElementById("ckpt").value))];
  const hasFrames = !!(data.frames && data.frames.t);
  const shapes = [];
  (hasFrames ? ["", "2", "3", "4", "5", "6"] : ["", "2", "3", "4", "5"]).forEach(n => {
    (data.shade || []).forEach(s => shapes.push(shadeRect(n, s[0], s[1])));
    (run.gaps || []).forEach(s => shapes.push(gapRect(n, s[0], s[1])));
    shapes.push({type: "line", xref: "x" + n, yref: "y" + n + " domain", x0: t, x1: t, y0: 0, y1: 1, line: {color: "#fff", width: 1}});
    (data.switches || []).forEach(s => shapes.push({
      type: "line", xref: "x" + n, yref: "y" + n + " domain",
      x0: s.t, x1: s.t, y0: 0, y1: 1, line: {color: "#5dade2", width: 1, dash: "dot"}
    }));
  });
  const x = {domain: [0, 1], automargin: false, title: ""};
  const br = run.q_break;
  const rel = run.role === "r1rel";
  const a = rel ? "x" : "v1";
  const b = rel ? "y" : "v2";
  const c = rel ? "z" : "δn";
  const d = rel ? "rot deg" : "ΔR deg";
  const plots = [
    line(run.q_t, run.v1_p, a + " pred", DIM.v1.pred, "x", "dash", br),
    line(run.q_t, run.v2_p, b + " pred", DIM.v2.pred, "x", "dash", br),
    line(run.q_t, run.v1_l, a + " label", DIM.v1.label, "x", null, br),
    line(run.q_t, run.v2_l, b + " label", DIM.v2.label, "x", null, br),
    line(data.force_t, data.force, "||F||", "#e67e22", "x2", null, data.force_break),
    line(run.q_t, run.n_p, c + " pred", DIM.n.pred, "x3", "dash", br),
    line(run.q_t, run.n_l, c + " label", DIM.n.label, "x3", null, br),
    line(run.q_t, run.ang, d, DIM.ang.pred, "x4", "dash", br),
    line(run.q_t, run.stiff_p, "stiff pred", DIM.stiff.pred, "x5", "dash", br),
    line(run.q_t, run.stiff_l, "stiff label", DIM.stiff.label, "x5", null, br)
  ];
  const layout = {
    paper_bgcolor: "#111", plot_bgcolor: "#1c1c1c", font: {color: "#ddd"},
    margin: {l: 64, r: 16, t: 28, b: 28}, legend: {orientation: "h", y: 1.12},
    shapes,
    xaxis: Object.assign({anchor: "y"}, x),
    yaxis: {title: rel ? "x m" : "v m/s", domain: hasFrames ? [0.86, 1] : [0.80, 1], anchor: "x"},
    xaxis2: Object.assign({anchor: "y2", matches: "x"}, x),
    yaxis2: {title: "N", domain: hasFrames ? [0.70, 0.84] : [0.60, 0.78], anchor: "x2"},
    xaxis3: Object.assign({anchor: "y3", matches: "x"}, x),
    yaxis3: {title: rel ? "z m" : "δn m", domain: hasFrames ? [0.54, 0.68] : [0.40, 0.58], anchor: "x3"},
    xaxis4: Object.assign({anchor: "y4", matches: "x"}, x),
    yaxis4: {title: rel ? "rot deg" : "ΔR deg", domain: hasFrames ? [0.38, 0.52] : [0.20, 0.38], anchor: "x4"},
    xaxis5: Object.assign({anchor: "y5", matches: "x", title: hasFrames ? "" : "s"}, x),
    yaxis5: {title: "stiff", domain: hasFrames ? [0.22, 0.36] : [0, 0.18], anchor: "x5"}
  };
  if (hasFrames) {
    plots.push(line(data.frames.t, data.frames.dR_old, "old |ΔR|", "rgba(255,255,255,0.55)", "x6", "dash", data.frames.break));
    plots.push(line(data.frames.t, data.frames.dR_new, "new |ΔR|", "#ff9f1a", "x6", null, data.frames.break));
    plots.push({
      x: data.frames.jump_t, y: data.frames.jump_y, name: "old jump",
      type: "scatter", mode: "markers", xaxis: "x6", yaxis: "y6",
      marker: {color: "#e74c3c", size: 7, symbol: "diamond"}
    });
    if (data.switches && data.switches.length) {
      plots.push({
        x: data.switches.map(s => s.t), y: data.switches.map(s => s.frame),
        name: "crown-wall", type: "scatter", mode: "markers",
        xaxis: "x6", yaxis: "y6",
        marker: {color: "#5dade2", size: 9, symbol: "triangle-up"}
      });
    }
    layout.xaxis6 = Object.assign({anchor: "y6", matches: "x", title: "s"}, x);
    layout.yaxis6 = {title: "frame deg", domain: [0, 0.18], anchor: "x6"};
  }
  await Plotly.react("plots", plots, layout, {responsive: true});
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
  const frameToggles = document.getElementById("frame_toggles");
  frameToggles.innerHTML = "";
  if (data.frames) {
    [["old", "old frames"], ["new", "new frames"]].forEach(pair => {
      const box = document.createElement("label");
      box.style.color = "#ddd";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = shownFrame[pair[0]];
      input.onchange = () => { shownFrame[pair[0]] = input.checked; paintCursor(); };
      box.appendChild(input);
      box.appendChild(document.createTextNode(" " + pair[1] + " "));
      frameToggles.appendChild(box);
    });
  }
  await paintCursor();
}
document.getElementById("time").oninput = () => { fi = +document.getElementById("time").value; paintCursor(); };
document.getElementById("ep").onchange = () => show();
document.getElementById("ckpt").onchange = () => paintCursor();
(async function init() {
  const meta = await (await fetch("/api/meta")).json();
  mesh = meta.mesh;
  const ep = document.getElementById("ep");
  ep.innerHTML = meta.episodes.map(e => {
    const split = meta.splits && meta.splits[String(e)];
    const label = split ? (e + " " + split) : String(e);
    return "<option value='" + e + "'>" + label + "</option>";
  }).join("");
  if (meta.switches || meta.nearcap) buildEvents(meta);
  show();
})();
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    exports: list = []
    mesh = None
    frames = None

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
            body = {"episodes": episodes, "mesh": self.mesh}
            extra = frame_episode_ids(self.frames)
            if extra:
                body["episodes"] = sorted(set(episodes) | set(extra))
                splits = {
                    str(ep): self.frames[ep]["split"]
                    for ep in extra
                    if self.frames[ep].get("split")
                }
                if splits:
                    body["splits"] = splits
                events = self.frames.get("events") or {}
                if events.get("switches"):
                    body["switches"] = events["switches"]
                if events.get("nearcap"):
                    body["nearcap"] = events["nearcap"]
            self._send(200, json.dumps(body).encode(), "application/json")
            return
        try:
            episode = int(args.get("ep", "0"))
        except ValueError:
            self._send(400, b"{}", "application/json")
            return
        if path == "/api/episode":
            try:
                payload = episode_payload(self.exports, episode, frames=self.frames)
            except KeyError:
                if not self.frames or episode not in self.frames:
                    self._send(404, b"{}", "application/json")
                    return
                payload = frames_only_payload(self.frames, episode)
            self._send(200, json.dumps(payload).encode(), "application/json")
            return
        if path == "/api/cursor":
            body = json.dumps(cursor_payload(self.exports, episode, int(args.get("q", "0")), frames=self.frames)).encode()
            self._send(200, body, "application/json")
            return
        self._send(404, b"not found", "text/plain")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, action="append", required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--frames_npz", type=Path, default=None)
    args = parser.parse_args()
    field = np.load(args.mesh)
    vertices, faces = decimate_mesh(field["vertices"], field["faces"], field["is_apron"])
    Handler.mesh = mesh_payload(vertices, faces)
    Handler.exports = [load_export(path) for path in args.export]
    Handler.frames = load_frames_npz(args.frames_npz) if args.frames_npz else None
    if Handler.frames is not None:
        print(f"[view_openloop] frame overlay on episodes {frame_episode_ids(Handler.frames)}", flush=True)
    print(f"[view_openloop] http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
