#!/usr/bin/env python3
"""October stringer_v3.5 viewer.

  python UMIFT_Data/surface_frames_scripts/view_oct.py

http://127.0.0.1:8765
  r1rel  source zarr, one-step motion in the current tool frame (standard BC)
  r4     mold copy with v1, v2, n and tilt
Left/right arrows step the wrist frame and the marker on the 3D path.
Up/down arrows change episode. Drag the 3D view to rotate.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import numpy as np
import open3d as o3d
import zarr
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

ORIG = "/mnt/data/umidata/ft/datasets/stringer_v3.5/acp_replay_buffer.zarr"
R4 = (
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb/"
    "r4_world_fixb_tilt/acp_replay_buffer.zarr"
)
FIELD = (
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb/shared/fixb_field.npz"
)
BAG_DIR = Path("/mnt/data/umidata/ft/datasets/stringer_v3.5/bags")
TAG_DIR = Path("/mnt/data/umidata/ft/datasets/stringer_v3.5/tag_traj")
UNIFIED_DIR = Path("/mnt/data/umidata/ft/datasets/stringer_v3.5/zarr")
CORR_DIR = Path("/mnt/data/umidata/ft/datasets/stringer_v3.5/tag_traj_odom_corr")
# One-sample odom teleport. At 200 Hz this is 30 m/s, not a hand motion.
ODOM_STEP_M = 0.15
# Zarr RGB clock starts at the 5 s hold. Bag frame 0 is the first live frame.
PAD_S = 5.0
MESH_TRI_CAP = 8000
PATH_CAP = 700
SERIES_CAP = 1200
GAP_EPISODES = (13, 25, 29, 42)
AXIS_M = 0.04


def episode_ids(root) -> list[int]:
    keys = [k for k in root["data"].keys() if k.startswith("episode_")]
    return sorted(int(k.split("_")[-1]) for k in keys)


def recording_names() -> list[str]:
    files = sorted(TAG_DIR.glob("*_ee_master.npz"))
    return [path.name.replace("_ee_master.npz", "") for path in files]


def decimate_mesh(vertices: np.ndarray, faces: np.ndarray, apron: np.ndarray):
    faces = np.asarray(faces, dtype=np.int32)
    body = ~np.asarray(apron, dtype=bool)[faces].any(axis=1)
    faces = faces[body]
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    vertices = np.asarray(vertices, dtype=np.float64)[used]
    faces = remap[faces]
    if len(faces) > MESH_TRI_CAP:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=MESH_TRI_CAP)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.int32)
    return vertices, faces


def increasing(t: np.ndarray, rows: np.ndarray):
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    rows = np.asarray(rows, dtype=np.float64)
    order = np.argsort(t, kind="mergesort")
    t = t[order]
    rows = rows[order]
    keep = np.ones(len(t), dtype=bool)
    if len(t) > 1:
        keep[1:] = np.diff(t) > 1e-9
    return t[keep], rows[keep]


def interp_to(t_dst, t_src, rows):
    t_src, rows = increasing(t_src, rows)
    if rows.ndim == 1:
        return np.interp(t_dst, t_src, rows)
    return np.column_stack(
        [np.interp(t_dst, t_src, rows[:, i]) for i in range(rows.shape[1])]
    )


def stride_idx(n: int, cap: int) -> np.ndarray:
    step = max(1, int(np.ceil(n / cap)))
    return np.arange(0, n, step)


def rounded(a, ndigits: int):
    return np.round(np.asarray(a, dtype=np.float64), ndigits).tolist()


def tool_rotation(pose7: np.ndarray) -> np.ndarray:
    q_xyzw = np.asarray(pose7, dtype=np.float64)[:, [4, 5, 6, 3]]
    return Rotation.from_quat(q_xyzw).as_matrix()


def virtual_target_xyz(ep) -> np.ndarray:
    """Fixture-frame xyz of ts_pose_virtual_target_0.

    That array is SE3_WC from the wrench labeller. Relative training does
    inv(TCP) @ SE3_WC, and inference multiplies the TCP back on, so this xyz
    is the point the relative label decodes to.
    """
    return np.asarray(ep["ts_pose_virtual_target_0"], dtype=np.float64)[:, :3]


def attach_virtual_target(out, xyz, vt_xyz, ip, R_field=None, iser=None) -> None:
    m = min(len(xyz), len(vt_xyz))
    ip = ip[ip < m]
    out["vt"] = rounded(vt_xyz[:m][ip], 4)
    if R_field is None or iser is None:
        return
    offset = vt_xyz[:m] - xyz[:m]
    # mold_R_field columns are t1, t2, n, so R^T @ offset is the surface-frame split.
    comp = np.einsum("nji,nj->ni", R_field[:m], offset)
    sl = iser[iser < m]
    out["vt_t1"] = rounded(comp[sl, 0], 4)
    out["vt_t2"] = rounded(comp[sl, 1], 4)
    out["vt_n"] = rounded(comp[sl, 2], 4)


def episode_payload(root, episode: int, r4: bool, frame_t: np.ndarray, vt_xyz=None) -> dict:
    ep = root["data"][f"episode_{episode}"]
    robot_t = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rgb_t = np.asarray(ep["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    t0 = float(rgb_t[0])
    pose = np.asarray(ep["ts_pose_fb_0"], dtype=np.float64)
    xyz = pose[:, :3]
    R = tool_rotation(pose)
    wrench = np.asarray(ep["wrench_left_0"], dtype=np.float64)
    wrench_t = np.asarray(ep["wrench_time_stamps_0"], dtype=np.float64).reshape(-1)
    force = np.linalg.norm(wrench[:, :3], axis=1)
    n = len(robot_t)
    ip = stride_idx(n, PATH_CAP)
    iw = stride_idx(len(wrench_t), SERIES_CAP)
    iser = stride_idx(n, SERIES_CAP)
    out = {
        "episode": episode,
        "r4": r4,
        "frame_t": rounded(frame_t, 3),
        "path_t": rounded(robot_t[ip] - t0, 3),
        "path": rounded(xyz[ip], 4),
        "tool_z": rounded(R[ip, :, 2], 4),
        "force_t": rounded(wrench_t[iw] - t0, 3),
        "force": rounded(force[iw], 2),
        "gap": episode in GAP_EPISODES,
    }
    if not r4:
        idx = np.clip(np.searchsorted(robot_t, rgb_t), 0, n - 1)
        i0, i1 = idx[:-1], idx[1:]
        rel = np.einsum("nji,nj->ni", R[i0], xyz[i1] - xyz[i0])
        rel_R = np.einsum("nji,njk->nik", R[i0], R[i1])
        ang = np.degrees(np.linalg.norm(Rotation.from_matrix(rel_R).as_rotvec(), axis=1))
        step = max(1, len(i0) // SERIES_CAP)
        sl = slice(None, None, step)
        out["sig_t"] = rounded(rgb_t[:-1][sl] - t0, 3)
        out["dx"] = rounded(rel[sl, 0], 4)
        out["dy"] = rounded(rel[sl, 1], 4)
        out["dz"] = rounded(rel[sl, 2], 4)
        out["dang"] = rounded(ang[sl], 2)
        if vt_xyz is None:
            vt_xyz = virtual_target_xyz(ep)
        attach_virtual_target(out, xyz, vt_xyz, ip)
        return out

    count = min(n, len(pose), len(ep["mold_normal_0"]), len(ep["fixb_v_par_0"]))
    normal = np.asarray(ep["mold_normal_0"], dtype=np.float64)[:count]
    v = np.asarray(ep["fixb_v_par_0"], dtype=np.float64)[:count]
    dn = np.asarray(ep["fixb_delta_n_0"], dtype=np.float64).reshape(-1)[:count]
    foot = np.asarray(ep["mold_x_s_0"], dtype=np.float64)[:count]
    ip = ip[ip < count]
    iser = iser[iser < count]
    z_tool = R[:count, :, 2]
    cos = np.clip(np.sum(z_tool * (-normal), axis=1), -1.0, 1.0)
    tilt = np.degrees(np.arccos(cos))
    f_tcp = interp_to(robot_t[:count], wrench_t, wrench[:, :3])
    fn = np.sum(np.einsum("nij,nj->ni", R[:count], f_tcp) * normal, axis=1)
    out["path_t"] = rounded(robot_t[ip] - t0, 3)
    out["path"] = rounded(xyz[ip], 4)
    out["tool_z"] = rounded(z_tool[ip], 4)
    out["tilt"] = rounded(tilt[ip], 1)
    out["foot"] = rounded(foot[ip], 4)
    out["sig_t"] = rounded(robot_t[iser] - t0, 3)
    out["v1"] = rounded(v[iser, 0], 4)
    out["v2"] = rounded(v[iser, 1], 4)
    out["n"] = rounded(dn[iser], 4)
    out["fn_t"] = out["sig_t"]
    out["fn"] = rounded(fn[iser], 2)
    if vt_xyz is None:
        vt_xyz = virtual_target_xyz(ep)
    R_field = np.asarray(ep["mold_R_field_0"], dtype=np.float64)[:count].reshape(-1, 3, 3)
    attach_virtual_target(out, xyz[:count], vt_xyz, ip, R_field, iser)
    return out


def attach_raw_odom(payload: dict, name: str) -> None:
    """Raw t265_odom positions in the fixture frame.

    The pipeline's T265-only curve (dead_reckon_from_first_anchor) is the
    solved TCP at the first tag fix, times inv(T_odom) at that instant, times
    each later odom pose. T_d405_odom equals T_d405_tcp in the stringer config,
    so the camera conjugation cancels and this is that same rigid map applied
    to the raw samples. Jumps in the odom stream stay jumps.
    """
    zpath = UNIFIED_DIR / f"{name}_unified.zarr"
    side_path = TAG_DIR / f"{name}_ee_master.npz"
    if not zpath.exists() or not side_path.exists():
        return
    side = np.load(side_path, allow_pickle=True)
    root = zarr.open(str(zpath), "r")
    odom = root["sensors"]["t265_odom"]

    def col(key: str) -> np.ndarray:
        return np.asarray(odom[key], dtype=np.float64).reshape(-1)

    ot = col("timestamp_sec_unified")
    pos = np.column_stack([col("pos_x"), col("pos_y"), col("pos_z")])
    quat = np.column_stack(
        [col("orient_x"), col("orient_y"), col("orient_z"), col("orient_w")]
    )
    st = np.asarray(side["timestamp_sec_unified"], dtype=np.float64).reshape(-1)
    n_vis = np.asarray(side["n_tags_visible"], dtype=np.int32)
    pose = np.asarray(side["pose7"], dtype=np.float64)
    hits = np.flatnonzero(n_vis > 0)
    anchor = int(hits[0]) if len(hits) else 0
    ta = float(st[anchor])
    i = int(np.clip(np.searchsorted(ot, ta), 1, len(ot) - 1))
    t0o, t1o = float(ot[i - 1]), float(ot[i])
    span = t1o - t0o
    alpha = 0.0 if span <= 1e-9 else float(np.clip((ta - t0o) / span, 0.0, 1.0))
    po = (1.0 - alpha) * pos[i - 1] + alpha * pos[i]
    rots = Rotation.from_quat(quat[[i - 1, i]])
    if span <= 1e-9:
        Ro = rots[0].as_matrix()
    else:
        Ro = Slerp([0.0, 1.0], rots)([alpha]).as_matrix()[0]
    solved = pose[anchor]
    Rs = Rotation.from_quat(solved[[4, 5, 6, 3]]).as_matrix()
    ps = solved[:3]
    R_fix_odom = Rs @ Ro.T
    mapped = pos @ R_fix_odom.T + (ps - R_fix_odom @ po)
    t_view = (ot - float(st[0])) + PAD_S
    step = np.linalg.norm(np.diff(mapped, axis=0), axis=1)
    dt = np.diff(ot)
    land = np.flatnonzero((step > ODOM_STEP_M) & (dt < 0.05)) + 1
    ip = stride_idx(len(mapped), PATH_CAP)
    payload["odom_t"] = rounded(t_view[ip], 3)
    payload["odom"] = rounded(mapped[ip], 4)
    payload["odom_jump_t"] = rounded(t_view[land], 3)
    payload["odom_jump"] = rounded(mapped[land], 4)


def attach_corrected_pose(payload: dict, name: str) -> None:
    """Optional trace from the odom-teleport sidecar. Off until the button is on."""
    path = CORR_DIR / f"{name}_odom_corr.npz"
    if not path.exists():
        return
    side = np.load(path, allow_pickle=True)
    st = np.asarray(side["timestamp_sec_unified"], dtype=np.float64).reshape(-1)
    pose = np.asarray(side["pose7"], dtype=np.float64)
    ip = stride_idx(len(pose), PATH_CAP)
    payload["corr_t"] = rounded((st[ip] - float(st[0])) + PAD_S, 3)
    payload["corr"] = rounded(pose[ip, :3], 4)
    span = np.asarray(side["bad_span_s"], dtype=np.float64).reshape(-1, 2)
    if len(span):
        payload["bad_span"] = [round(float(span[0, 0]) + PAD_S, 1), round(float(span[0, 1]) + PAD_S, 1)]


def mesh_payload(vertices: np.ndarray, faces: np.ndarray) -> dict:
    return {
        "x": rounded(vertices[:, 0], 4),
        "y": rounded(vertices[:, 1], 4),
        "z": rounded(vertices[:, 2], 4),
        "i": faces[:, 0].astype(int).tolist(),
        "j": faces[:, 1].astype(int).tolist(),
        "k": faces[:, 2].astype(int).tolist(),
    }


PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>October stringer_v3.5</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body { margin:0; font-family:sans-serif; background:#111; color:#ddd; }
.bar { display:flex; gap:12px; align-items:center; padding:8px 12px; }
button { background:#222; color:#ddd; border:1px solid #444; padding:4px 10px; }
button.on { background:#1a5276; }
input[type=range] { width:180px; }
#note { color:#f5b041; }
#hint { color:#888; font-size:13px; }
.top { display:grid; grid-template-columns: 340px minmax(0,1fr); height:48vh; }
#wrist { width:100%; height:100%; object-fit:contain; background:#000; }
#view3d, #plots { height:48vh; }
</style></head>
<body>
<div class="bar">
  <button id="rel">r1rel</button>
  <button id="r4" class="on">r4</button>
  <span>episode</span>
  <input id="ep" type="range" min="0" max="49" value="0">
  <span id="label">episode 0</span>
  <span>time</span>
  <input id="time" type="range" min="0" max="1" value="0">
  <span id="readout"></span>
  <span id="note"></span>
  <span>view</span>
  <button id="viewSide" class="on">side</button>
  <button id="viewTop">top</button>
  <button id="viewEnd">end</button>
  <button id="vt">virtual target</button>
  <button id="odom">t265</button>
  <button id="corr">t265 fixed</button>
  <span id="hint">drag to orbit · ← → frame · ↑ ↓ episode</span>
</div>
<div class="top">
  <img id="wrist" alt="wrist">
  <div id="view3d"></div>
</div>
<div id="plots"></div>
<script>
const relBtn = document.getElementById("rel");
const r4Btn = document.getElementById("r4");
const ep = document.getElementById("ep");
const time = document.getElementById("time");
const label = document.getElementById("label");
const note = document.getElementById("note");
const readout = document.getElementById("readout");
const wrist = document.getElementById("wrist");
let mode = "r4";
let showVt = false;
let showOdom = false;
let showCorr = false;
let counts = {r1rel: 50, r4: 50};
let mesh = null;
let data = null;
let fi = 0;

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

function cursorAt(j) {
  const p = data.path[j];
  const z = data.tool_z[j];
  const tip = [p[0] + 0.04 * z[0], p[1] + 0.04 * z[1], p[2] + 0.04 * z[2]];
  return {p, tip};
}

function sceneTraces(d) {
  const path = {
    type: "scatter3d",
    x: d.path.map(p => p[0]), y: d.path.map(p => p[1]), z: d.path.map(p => p[2]),
    mode: d.tilt ? "markers" : "lines",
    name: "tcp",
    showlegend: false,
    line: {color: "#5dade2", width: 4},
    marker: d.tilt ? {
      size: 2.2, color: d.tilt, colorscale: "Turbo", cmin: 0, cmax: 90,
      colorbar: {title: "tilt deg", len: 0.45, thickness: 12}
    } : {size: 2, color: "#5dade2"}
  };
  const c0 = cursorAt(0);
  const cursor = {
    type: "scatter3d", x: [c0.p[0]], y: [c0.p[1]], z: [c0.p[2]],
    mode: "markers", name: "measured TCP",
    marker: {size: 11, color: "#ffffff", line: {color: "#111111", width: 3}}
  };
  const axis = {
    type: "scatter3d",
    x: [c0.p[0], c0.tip[0]], y: [c0.p[1], c0.tip[1]], z: [c0.p[2], c0.tip[2]],
    mode: "lines", name: "tool z", showlegend: false,
    line: {color: "#e74c3c", width: 6}
  };
  const v0 = (d.vt && d.vt.length) ? d.vt[0] : c0.p;
  const vt = {
    type: "scatter3d",
    x: (d.vt || []).map(p => p[0]), y: (d.vt || []).map(p => p[1]), z: (d.vt || []).map(p => p[2]),
    mode: "lines", name: "virtual target", visible: showVt,
    line: {color: "#f5b7ff", width: 3}
  };
  const link = {
    type: "scatter3d",
    x: [c0.p[0], v0[0]], y: [c0.p[1], v0[1]], z: [c0.p[2], v0[2]],
    mode: "lines", name: "tcp to vt", visible: showVt, showlegend: false,
    line: {color: "#f5b7ff", width: 5}
  };
  const odom = odomTraces(d).concat(corrTrace(d));
  if (!d.r4) return [path, cursor, vt, link].concat(odom);
  const mold = {
    type: "mesh3d", x: mesh.x, y: mesh.y, z: mesh.z, i: mesh.i, j: mesh.j, k: mesh.k,
    color: "#9aabb6", opacity: 0.45, name: "mold", hoverinfo: "skip", showlegend: false
  };
  const foot = {
    type: "scatter3d",
    x: d.foot.map(p => p[0]), y: d.foot.map(p => p[1]), z: d.foot.map(p => p[2]),
    mode: "lines", name: "foot", showlegend: false, line: {color: "#f4d03f", width: 3}
  };
  return [mold, path, foot, cursor, axis, vt, link].concat(odom);
}

function odomTraces(d) {
  const pts = d.odom || [];
  const jumps = d.odom_jump || [];
  return [
    {
      type: "scatter3d",
      x: pts.map(p => p[0]), y: pts.map(p => p[1]), z: pts.map(p => p[2]),
      mode: "lines", name: "t265", visible: showOdom,
      line: {color: "#ff9f1a", width: 3}
    },
    {
      type: "scatter3d",
      x: jumps.map(p => p[0]), y: jumps.map(p => p[1]), z: jumps.map(p => p[2]),
      mode: "markers", name: "odom step", visible: showOdom,
      marker: {size: 6, color: "#e74c3c", symbol: "diamond"}
    }
  ];
}

function corrTrace(d) {
  const pts = d.corr || [];
  return {
    type: "scatter3d",
    x: pts.map(p => p[0]), y: pts.map(p => p[1]), z: pts.map(p => p[2]),
    mode: "lines", name: "t265 fixed", visible: showCorr,
    line: {color: "#2ecc71", width: 3}
  };
}

const VIEWS = {
  side: {eye: {x: 0.15, y: -1.7, z: 1.7}, center: {x: 0, y: 0, z: 0}, up: {x: 0, y: -1, z: 0}},
  top: {eye: {x: 0, y: -2.6, z: 0.02}, center: {x: 0, y: 0, z: 0}, up: {x: 0, y: 0, z: -1}},
  end: {eye: {x: 2.6, y: -0.45, z: 0.35}, center: {x: 0, y: 0, z: 0}, up: {x: 0, y: -1, z: 0}}
};
let viewName = "side";
let sceneBox = null;

function span(values) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return [lo, hi];
}

function sceneBoxFrom(d) {
  const xs = mesh.x.slice();
  const ys = mesh.y.slice();
  const zs = mesh.z.slice();
  d.path.forEach(p => { xs.push(p[0]); ys.push(p[1]); zs.push(p[2]); });
  if (d.foot) d.foot.forEach(p => { xs.push(p[0]); ys.push(p[1]); zs.push(p[2]); });
  if (showVt && d.vt) d.vt.forEach(p => { xs.push(p[0]); ys.push(p[1]); zs.push(p[2]); });
  if (showOdom && d.odom) d.odom.forEach(p => { xs.push(p[0]); ys.push(p[1]); zs.push(p[2]); });
  if (showOdom && d.odom_jump) d.odom_jump.forEach(p => { xs.push(p[0]); ys.push(p[1]); zs.push(p[2]); });
  if (showCorr && d.corr) d.corr.forEach(p => { xs.push(p[0]); ys.push(p[1]); zs.push(p[2]); });
  const pad = 0.05;
  function rng(a) {
    const s = span(a);
    return [s[0] - pad, s[1] + pad];
  }
  return {x: rng(xs), y: rng(ys), z: rng(zs)};
}

function axisLock(title, range) {
  return {title, range, autorange: false};
}

function sceneLayout() {
  const box = sceneBox || {x: [-1, 1], y: [-1, 1], z: [-1, 1]};
  return {
    paper_bgcolor: "#111", margin: {l: 0, r: 0, t: 0, b: 0},
    showlegend: true,
    legend: {x: 0, y: 1, font: {color: "#ddd"}, bgcolor: "rgba(0,0,0,0)"},
    uirevision: "keep",
    scene: {
      uirevision: "keep",
      aspectmode: "data",
      bgcolor: "#1c1c1c",
      camera: VIEWS[viewName],
      xaxis: axisLock("x", box.x),
      yaxis: axisLock("y", box.y),
      zaxis: axisLock("z", box.z)
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

function line(x, y, name, color, axis) {
  return {x, y, name, xaxis: axis, yaxis: axis.replace("x", "y"), line: {color, width: 1.4}};
}

function hideUnlessVt(trace) {
  trace.visible = showVt;
  return trace;
}

function plotTraces(d) {
  const force = line(d.force_t, d.force, "||F||", "#e67e22", "x2");
  if (!d.r4) {
    return [
      line(d.sig_t, d.dx, "dx", "#e74c3c", "x"),
      line(d.sig_t, d.dy, "dy", "#2ecc71", "x"),
      line(d.sig_t, d.dz, "dz", "#3498db", "x"),
      force,
      line(d.sig_t, d.dang, "dR deg", "#f4d03f", "x3")
    ];
  }
  return [
    line(d.sig_t, d.v1, "v1", "#e74c3c", "x"),
    line(d.sig_t, d.v2, "v2", "#2ecc71", "x"),
    force,
    d.fn ? line(d.fn_t, d.fn, "F·n", "#af7ac5", "x2") : force,
    line(d.sig_t, d.n, "n", "#9b59b6", "x3"),
    hideUnlessVt(line(d.sig_t, d.vt_t1 || [], "vt·t1", "#f5b7ff", "x3")),
    hideUnlessVt(line(d.sig_t, d.vt_t2 || [], "vt·t2", "#d7bde2", "x3")),
    hideUnlessVt(line(d.sig_t, d.vt_n || [], "vt·n", "#f5b041", "x3"))
  ];
}

function vline(n) {
  return {
    type: "line", xref: "x" + n, yref: "y" + n + " domain",
    x0: 0, x1: 0, y0: 0, y1: 1, line: {color: "#ffffff", width: 1}
  };
}

function jumpLine(n, t) {
  return {
    type: "line", xref: "x" + n, yref: "y" + n + " domain",
    x0: t, x1: t, y0: 0, y1: 1, line: {color: "#e74c3c", width: 1}
  };
}

function plotShapes(d) {
  const shapes = [vline(""), vline("2"), vline("3")];
  const t = d.frame_t && d.frame_t.length ? d.frame_t[Math.max(0, Math.min(fi, d.frame_t.length - 1))] : 0;
  shapes.forEach(s => { s.x0 = t; s.x1 = t; });
  if (showOdom && d.odom_jump_t) {
    d.odom_jump_t.forEach(jt => {
      shapes.push(jumpLine("", jt));
      shapes.push(jumpLine("2", jt));
      shapes.push(jumpLine("3", jt));
    });
  }
  return shapes;
}

function timeEnd(d) {
  let t1 = 0;
  [d.sig_t, d.force_t, d.frame_t].forEach(a => {
    if (a && a.length) t1 = Math.max(t1, a[a.length - 1]);
  });
  return t1;
}

function plotLayout(d) {
  const top = d.r4 ? "v1, v2 (m/s)" : "step in current TCP (m)";
  const bot = d.r4 ? "n (m)" : "step rotation (deg)";
  const x = {
    domain: [0, 1], automargin: false, tick0: 0, dtick: 10, title: ""
  };
  return {
    paper_bgcolor: "#111", plot_bgcolor: "#1c1c1c", font: {color: "#ddd"},
    margin: {l: 72, r: 16, t: 28, b: 32},
    legend: {orientation: "h", y: 1.12},
    shapes: plotShapes(d),
    xaxis: Object.assign({anchor: "y", range: [0, timeEnd(d)]}, x),
    yaxis: {title: top, domain: [0.70, 1], anchor: "x", automargin: false},
    xaxis2: Object.assign({anchor: "y2", matches: "x"}, x),
    yaxis2: {title: "N", domain: [0.36, 0.64], anchor: "x2", automargin: false},
    xaxis3: Object.assign({anchor: "y3", matches: "x", title: "s"}, x),
    yaxis3: {title: bot, domain: [0, 0.30], anchor: "x3", automargin: false}
  };
}

function paintCursor() {
  const t = data.frame_t[fi];
  const j = nearest(data.path_t, t);
  const c = cursorAt(j);
  const cursorIdx = data.r4 ? 3 : 1;
  const axisIdx = 4;
  const cam = liveCamera();
  const jobs = [
    Plotly.restyle("view3d", {x: [[c.p[0]]], y: [[c.p[1]]], z: [[c.p[2]]]}, [cursorIdx])
  ];
  if (data.r4) {
    jobs.push(Plotly.restyle("view3d", {
      x: [[c.p[0], c.tip[0]]], y: [[c.p[1], c.tip[1]]], z: [[c.p[2], c.tip[2]]]
    }, [axisIdx]));
  }
  if (data.vt && data.vt.length) {
    const q = data.vt[Math.min(j, data.vt.length - 1)];
    const linkIdx = data.r4 ? 6 : 3;
    jobs.push(Plotly.restyle("view3d", {
      x: [[c.p[0], q[0]]], y: [[c.p[1], q[1]]], z: [[c.p[2], q[2]]]
    }, [linkIdx]));
  }
  Promise.all(jobs).then(() => {
    applyCamera(cam);
    requestAnimationFrame(() => applyCamera(cam));
  });
  const upd = {};
  ["", "2", "3"].forEach((n, k) => {
    upd["shapes[" + k + "].x0"] = t;
    upd["shapes[" + k + "].x1"] = t;
  });
  Plotly.relayout("plots", upd);
  readout.textContent = t.toFixed(2) + " s";
  wrist.src = "/api/frame?ep=" + data.episode + "&i=" + fi;
  time.value = fi;
}

async function show(keepFrame) {
  const i = +ep.value;
  const d = await (await fetch("/api/episode?ep=" + i + "&mode=" + mode)).json();
  data = d;
  label.textContent = "episode " + d.episode + " / " + (counts[mode] - 1);
  note.textContent = noteText(d);
  time.max = Math.max(0, d.frame_t.length - 1);
  if (!keepFrame) fi = 0;
  fi = Math.max(0, Math.min(fi, d.frame_t.length - 1));
  time.value = fi;
  sceneBox = sceneBoxFrom(d);
  await Plotly.react("view3d", sceneTraces(d), sceneLayout(), {responsive: true});
  await Plotly.react("plots", plotTraces(d), plotLayout(d), {responsive: true});
  paintCursor();
}

function setMode(next) {
  mode = next;
  relBtn.classList.toggle("on", mode === "r1rel");
  r4Btn.classList.toggle("on", mode === "r4");
  ep.max = counts[mode] - 1;
  if (+ep.value > +ep.max) ep.value = ep.max;
  show(false);
}

function setView(name) {
  viewName = name;
  ["viewSide", "viewTop", "viewEnd"].forEach(id => {
    document.getElementById(id).classList.toggle("on", id === "view" + name[0].toUpperCase() + name.slice(1));
  });
  applyCamera(VIEWS[name]);
}

relBtn.onclick = () => setMode("r1rel");
r4Btn.onclick = () => setMode("r4");
document.getElementById("viewSide").onclick = () => setView("side");
document.getElementById("viewTop").onclick = () => setView("top");
document.getElementById("viewEnd").onclick = () => setView("end");
document.getElementById("vt").onclick = () => {
  showVt = !showVt;
  document.getElementById("vt").classList.toggle("on", showVt);
  if (!data) return;
  sceneBox = sceneBoxFrom(data);
  const layout = sceneLayout();
  const cam = liveCamera();
  const vtIdx = data.r4 ? 5 : 2;
  const jobs = [
    Plotly.restyle("view3d", {visible: showVt}, [vtIdx, vtIdx + 1]),
    Plotly.relayout("view3d", {
      "scene.xaxis.range": layout.scene.xaxis.range,
      "scene.yaxis.range": layout.scene.yaxis.range,
      "scene.zaxis.range": layout.scene.zaxis.range
    })
  ];
  if (data.vt_n) {
    const n = document.getElementById("plots").data.length;
    jobs.push(Plotly.restyle("plots", {visible: showVt}, [n - 3, n - 2, n - 1]));
  }
  Promise.all(jobs).then(() => applyCamera(cam));
};
document.getElementById("odom").onclick = () => {
  showOdom = !showOdom;
  document.getElementById("odom").classList.toggle("on", showOdom);
  if (!data) return;
  sceneBox = sceneBoxFrom(data);
  const layout = sceneLayout();
  const cam = liveCamera();
  const odomIdx = data.r4 ? 7 : 4;
  const jobs = [
    Plotly.restyle("view3d", {visible: showOdom}, [odomIdx, odomIdx + 1]),
    Plotly.relayout("view3d", {
      "scene.xaxis.range": layout.scene.xaxis.range,
      "scene.yaxis.range": layout.scene.yaxis.range,
      "scene.zaxis.range": layout.scene.zaxis.range
    }),
    Plotly.relayout("plots", {shapes: plotShapes(data)})
  ];
  Promise.all(jobs).then(() => applyCamera(cam));
};
document.getElementById("corr").onclick = () => {
  showCorr = !showCorr;
  document.getElementById("corr").classList.toggle("on", showCorr);
  if (!data) return;
  note.textContent = noteText(data);
  sceneBox = sceneBoxFrom(data);
  const layout = sceneLayout();
  const cam = liveCamera();
  const corrIdx = data.r4 ? 9 : 6;
  const jobs = [
    Plotly.restyle("view3d", {visible: showCorr}, [corrIdx]),
    Plotly.relayout("view3d", {
      "scene.xaxis.range": layout.scene.xaxis.range,
      "scene.yaxis.range": layout.scene.yaxis.range,
      "scene.zaxis.range": layout.scene.zaxis.range
    })
  ];
  Promise.all(jobs).then(() => applyCamera(cam));
};
function noteText(d) {
  const parts = [];
  if (d.gap) parts.push("RGB/depth gap > 0.5 s");
  if (showCorr && d.bad_span) parts.push("flagged " + d.bad_span[0] + "–" + d.bad_span[1] + " s");
  return parts.join(" · ");
}
ep.oninput = () => show(false);
time.oninput = () => { fi = +time.value; paintCursor(); };

window.addEventListener("keydown", (ev) => {
  if (ev.key === "ArrowLeft") { fi = Math.max(0, fi - 1); ev.preventDefault(); paintCursor(); }
  else if (ev.key === "ArrowRight") {
    fi = Math.min(data.frame_t.length - 1, fi + 1); ev.preventDefault(); paintCursor();
  } else if (ev.key === "ArrowUp") {
    ep.value = Math.max(0, +ep.value - 1); ev.preventDefault(); show(false);
  } else if (ev.key === "ArrowDown") {
    ep.value = Math.min(+ep.max, +ep.value + 1); ev.preventDefault(); show(false);
  }
});

(async function init() {
  const meta = await (await fetch("/api/meta")).json();
  counts = meta.counts;
  mesh = meta.mesh;
  ep.max = counts.r4 - 1;
  show(false);
})();
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    root_orig = None
    root_r4 = None
    ids_orig: list[int] = []
    ids_r4: list[int] = []
    names: list[str] = []
    mesh = None
    manifests: dict = {}
    raws: dict = {}

    def log_message(self, fmt, *args):
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _args(self) -> dict:
        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        return dict(part.split("=", 1) for part in query.split("&") if "=" in part)

    def _bag(self, episode: int):
        name = self.names[episode]
        if name not in self.manifests:
            path = BAG_DIR / f"{name}_camera"
            self.manifests[name] = json.loads((path / "manifest.json").read_text())
            self.raws[name] = np.memmap(path / "color.raw", dtype=np.uint8, mode="r")
        return name, self.manifests[name], self.raws[name]

    def _frame_times(self, episode: int) -> np.ndarray:
        _, manifest, _ = self._bag(episode)
        frames = manifest["streams"]["color"]["frames"]
        t0 = float(frames[0]["timestamp_sec"])
        rel = np.array([float(frame["timestamp_sec"]) - t0 for frame in frames])
        # Live frames sit after the 5 s hold that the zarr prepends.
        return rel + PAD_S

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            return
        if path == "/api/meta":
            body = json.dumps(
                {
                    "counts": {"r1rel": len(self.ids_orig), "r4": len(self.ids_r4)},
                    "mesh": self.mesh,
                }
            ).encode()
            self._send(200, body, "application/json")
            return
        if path not in ("/api/episode", "/api/frame"):
            self._send(404, b"not found", "text/plain")
            return
        args = self._args()
        r4 = args.get("mode", "r4") != "r1rel"
        ids = self.ids_r4 if r4 else self.ids_orig
        try:
            episode = int(args.get("ep", "0"))
        except ValueError:
            self._send(400, b'{"error":"bad episode"}', "application/json")
            return
        if episode not in ids or episode >= len(self.names):
            self._send(404, b'{"error":"missing episode"}', "application/json")
            return
        if path == "/api/frame":
            name, manifest, raw = self._bag(episode)
            color = manifest["streams"]["color"]
            frames = color["frames"]
            try:
                index = int(args.get("i", "0"))
            except ValueError:
                index = 0
            # frame_t is pad + bag time, so index 0 of the slider is the hold.
            # Bag index 0 is the first live frame, which is slider time PAD_S.
            times = self._frame_times(episode)
            if times.size == 0:
                self._send(404, b"no frames", "text/plain")
                return
            index = int(np.clip(index, 0, len(frames) - 1))
            frame = frames[index]
            blob = np.asarray(raw[frame["offset"] : frame["offset"] + frame["length"]])
            rgb = blob.reshape(int(color["height"]), int(color["width"]), 3)
            buf = BytesIO()
            Image.fromarray(rgb).save(buf, format="JPEG", quality=70)
            self._send(200, buf.getvalue(), "image/jpeg")
            return
        root = self.root_r4 if r4 else self.root_orig
        # r4's own ts_pose_virtual_target_0 is the identity placeholder.
        # The wrench label r1rel trains on still lives on the source zarr.
        vt_xyz = None
        if r4:
            vt_xyz = virtual_target_xyz(self.root_orig["data"][f"episode_{episode}"])
        payload = episode_payload(root, episode, r4, self._frame_times(episode), vt_xyz)
        attach_raw_odom(payload, self.names[episode])
        attach_corrected_pose(payload, self.names[episode])
        self._send(200, json.dumps(payload).encode(), "application/json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    field = np.load(FIELD)
    vertices, faces = decimate_mesh(field["vertices"], field["faces"], field["is_apron"])
    Handler.mesh = mesh_payload(vertices, faces)
    Handler.root_orig = zarr.open(ORIG, "r")
    Handler.root_r4 = zarr.open(R4, "r")
    Handler.ids_orig = episode_ids(Handler.root_orig)
    Handler.ids_r4 = episode_ids(Handler.root_r4)
    Handler.names = recording_names()
    print(
        f"[view_oct] r1rel {len(Handler.ids_orig)} episodes, "
        f"r4 {len(Handler.ids_r4)}, mesh {len(faces)} tris",
        flush=True,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[view_oct] http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
