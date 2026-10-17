#!/usr/bin/env python3
"""Enhanced Fix B scrubber: scanner_v2 (in-memory labels) + current Fix B data.

Does not write datasets. Reuses fixb_labels.episode_arrays on
stringer_scanner_v2 and the already-built r1/r2/r3 zarrs for comparison.

  python fixb_pretrain_scrubber.py
  python fixb_pretrain_scrubber.py --scanner_episodes 0 --fixb_episodes 9 --skip_rgb

Open eval/pretrain_viz/6_scrubber.html in Firefox (or the localhost:8766 server).
Default view is right_base, same frame as replay_wrist_cam.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
import zarr
from plotly.offline import get_plotlyjs
from scipy.spatial.transform import Rotation

from fixb_field import FixBMoldField
from fixb_labels import (
    Rp_to_pose7,
    episode_arrays,
    fixb_pose7_command,
    identity_pose7,
    pose7_to_Rp,
)
from fixb_pretrain_viz import (
    BRANCH_MARGIN_DEG,
    H,
    NEAR_M,
    OFF_EDGE_M,
    QUERY_STRIDE,
    RUNS,
    SCRUB_FLAGS,
    ScanOnlySD,
    display_mesh,
    process_episode,
    rnd,
    scrub_episode as scrub_core,
)

FIXB_ROOT = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_data_v2_full_fixb")
SCANNER_ZARR = Path("/mnt/data/umidata/ft/datasets/stringer_scanner_v2/acp_replay_buffer.zarr")
EXTRINSICS = Path("/home/boeing/umi_ft_ws/UMI-FT/PyriteConfig/tasks/umift/layup_workcell_extrinsics.yaml")
R_Y90 = Rotation.from_euler("y", 90.0, degrees=True).as_matrix()


def pearson(a, b) -> float:
    a, b = np.asarray(a, float).reshape(-1), np.asarray(b, float).reshape(-1)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def load_T_rb_f() -> np.ndarray:
    cfg = yaml.safe_load(EXTRINSICS.read_text())
    return np.asarray(cfg["T_right_base_from_fixture"], dtype=np.float64)


def wrap_scanner_episode(ep, field) -> dict:
    """In-memory r1/r2/r3 groups. Same packing as fixb_build_datasets, nothing written."""
    mold = episode_arrays(field, ep, smooth_window_s=0.1, v_par_mode="foot", delta_n_source="fb")
    keys = (
        "robot_time_stamps_0", "rgb_time_stamps_0", "wrench_time_stamps_0",
        "ts_pose_fb_0", "ts_pose_command_0", "ts_pose_virtual_target_0",
        "stiffness_0", "gripper_0", "wrench_left_0", "wrench_right_0", "wrench_concat_0",
    )
    base = {k: np.asarray(ep[k]) for k in keys}
    r1 = {**base, **mold}
    r2 = dict(r1)
    r2["ts_pose_command_0"] = fixb_pose7_command(mold["fixb_v_par_0"], mold["fixb_delta_n_0"])
    r2["ts_pose_virtual_target_0"] = identity_pose7(len(mold["fixb_v_par_0"]))
    r3 = dict(r2)
    R_tool, p = pose7_to_Rp(r1["ts_pose_fb_0"])
    R_f = mold["mold_R_field_0"].astype(np.float64).reshape(-1, 3, 3)
    p_loc = np.einsum("nji,nj->ni", R_f, p - mold["mold_x_s_0"].astype(np.float64))
    R_loc = np.einsum("nji,njk->nik", R_f, R_tool)
    r3["ts_pose_fb_0"] = Rp_to_pose7(R_loc, p_loc)
    return {"r1": r1, "r2": r2, "r3": r3}


def process_wrapped(name, groups, field, scan_sd, R_off, source, label):
    data = {k: {name: groups[k]} for k in groups}
    checks = {k: True if k == "timestamps_equal" else 0.0
              for k in ("timestamps_equal", "labels_r2_vs_r3_max_abs", "mold_obs_max_abs_vs_r2",
                        "fb_r1_vs_r2_max_abs", "r1_cmd_vs_fb_max_abs")}
    e = process_episode(name, data, field, scan_sd, R_off, checks)
    eid = e["id"]
    e["source"] = source
    e["key"] = ("s" if source == "scanner_v2" else "f") + str(eid)
    e["label"] = label
    e["split"] = "scanner_v2" if source == "scanner_v2" else e["split"]
    e["checks"] = checks
    return e


def pick_strokes(e, n=2, min_len=20):
    ir, v = e["ir"], e["rgb_view"]["v"]
    mask = (e["dist"][ir] < NEAR_M) & (np.linalg.norm(v, axis=1) > 0.03)
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    segs = [(int(s), int(t)) for s, t in zip(starts, ends) if t - s >= min_len]
    if not segs:
        return []
    if len(segs) == 1:
        return segs[:1]
    xs = e["x_s"][ir]
    means = np.array([xs[a:b].mean(0) for a, b in segs])
    spd = np.linalg.norm(v, axis=1)
    best, pair = -1.0, (segs[0], segs[1])
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            ratio = min(np.median(spd[segs[i][0]:segs[i][1]]), np.median(spd[segs[j][0]:segs[j][1]])) / max(
                np.median(spd[segs[i][0]:segs[i][1]]), np.median(spd[segs[j][0]:segs[j][1]]), 1e-6
            )
            score = float(np.linalg.norm(means[i] - means[j])) * ratio
            if score > best:
                best, pair = score, (segs[i], segs[j])
    return list(pair)[:n]


def pack_stroke(e, lo, hi, n=24):
    idx = np.linspace(lo, max(lo, hi - 1), n).astype(int)
    q0 = int(np.searchsorted(e["q"], lo))
    q0 = min(q0, len(e["q"]) - 1)
    return {
        "lo": int(lo), "hi": int(hi), "q0": q0,
        "t": rnd((np.arange(hi - lo) * float(np.median(np.diff(e["rgb"])))), 3) if hi > lo else [0],
        "tn": rnd(np.linspace(0, 1, n), 3),
        "P": rnd(e["p_fb"][e["ir"][idx]] * 1e3, 1),
        "v": rnd(e["rgb_view"]["v"][idx] * 1e3, 1),
        "dn": rnd(e["rgb_view"]["dn"][idx] * 1e3, 1),
        "n": rnd(e["rgb_view"]["Rf"][idx, :, 2], 3),
        "dp": rnd(np.diff(e["p_fb"][e["ir"][idx]], axis=0, append=e["p_fb"][e["ir"][idx][-1:]]) * 1e3, 2),
    }


def write_jpegs(rgb, dst: Path, quality=50) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = len(rgb)
    have = list(dst.glob("*.jpg"))
    if len(have) == n:
        return n
    for p in have:
        p.unlink()
    for i, im in enumerate(np.asarray(rgb)):
        a = np.clip(im, 0, 255).astype(np.uint8) if im.dtype != np.uint8 else im
        bgr = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(dst / f"{i:05d}.jpg"), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return n


def source_stats(eps):
    """Orientation correlation + rotate-the-mold invariance on near-surface RGB steps."""
    if not eps:
        return {}
    ir_list, P, v, dn, n = [], [], [], [], []
    for e in eps:
        ir, near = e["ir"], e["near_rgb"]
        if not near.any():
            continue
        P.append(e["p_fb"][ir][near])
        v.append(e["rgb_view"]["v"][near])
        dn.append(e["rgb_view"]["dn"][near])
        n.append(e["rgb_view"]["Rf"][near, :, 2])
        ir_list.append(near)
    if not P:
        return {}
    P, v, dn, n = map(np.concatenate, (P, v, dn, n))
    dp = np.diff(P, axis=0, append=P[-1:])
    axes = "xyz"
    corr_r1 = {f"dp_{a}_vs_n_{b}": round(pearson(dp[:, i], n[:, j]), 3)
               for i, a in enumerate(axes) for j, b in enumerate(axes)}
    corr_r2 = {f"{lab}_vs_n_{b}": round(pearson(src, n[:, j]), 3)
               for lab, src in (("v_t1", v[:, 0]), ("v_t2", v[:, 1]), ("delta_n", dn))
               for j, b in enumerate(axes)}
    dp_r, n_r = dp @ R_Y90.T, n @ R_Y90.T
    Rf_t1t2 = np.concatenate(
        [e["rgb_view"]["Rf"][e["near_rgb"], :, :2] for e in eps if e["near_rgb"].any()], axis=0
    )[: len(dp)]
    v_from_dp = np.einsum("nia,ni->na", Rf_t1t2, dp)
    Rfull = np.dstack([Rf_t1t2[:, :, 0], Rf_t1t2[:, :, 1], n])
    Rfull_r = np.einsum("ij,njk->nik", R_Y90, Rfull)
    v_rot = np.einsum("nia,ni->na", Rfull_r[:, :, :2], dp_r)
    inv = {
        "v_par_max_abs_change_mm_s": round(float(np.max(np.abs(v_rot - v_from_dp)) * 1e3), 4),
        "r1_world_dp_change_p50_mm": round(float(np.median(np.linalg.norm(dp_r - dp, axis=1)) * 1e3), 2),
        "r1_dpz_vs_nz_before": round(pearson(dp[:, 2], n[:, 2]), 3),
        "r1_dpz_vs_nz_after_rotate": round(pearson(dp_r[:, 2], n_r[:, 2]), 3),
        "r2_vt1_vs_nz_before": round(pearson(v[:, 0], n[:, 2]), 3),
        "r2_vt1_vs_nz_after_rotate": round(pearson(v_rot[:, 0], n_r[:, 2]), 3),
    }
    rng = np.random.default_rng(0)
    take = rng.choice(len(dp), size=min(800, len(dp)), replace=False)
    scatter = {
        "nz": rnd(n[take, 2], 3),
        "r1_dpz_mm": rnd(dp[take, 2] * 1e3, 2),
        "r1_dpx_mm": rnd(dp[take, 0] * 1e3, 2),
        "r2_vt1_mm_s": rnd(v[take, 0] * 1e3, 1),
        "r2_vt2_mm_s": rnd(v[take, 1] * 1e3, 1),
        "r2_dn_mm": rnd(dn[take] * 1e3, 1),
    }
    return {"n": int(len(dp)), "corr_r1": corr_r1, "corr_r2": corr_r2, "rotate_y90": inv, "scatter": scatter}


def pack_episode(e, rgb_rel):
    d = scrub_core(e)
    d["source"] = e["source"]
    d["label"] = e["label"]
    d["rgb_dir"] = rgb_rel
    d["n_rgb"] = int(len(e["rgb"]))
    d["strokes"] = [pack_stroke(e, a, b) for a, b in pick_strokes(e)]
    d["k23_p50_mm"] = round(float(np.median(e["err"][:, -1])), 2)
    d["near_frac"] = round(float(e["near"].mean()), 3)
    d["sd_peak_p50_mm"] = round(float(np.median(e["sd_peak"]) * 1e3), 1) if len(e["sd_peak"]) else None
    return d


def load_fixb_episode(root: Path, eid: int, field, scan_sd, R_off):
    name = f"episode_{eid}"
    data = {k: zarr.open(str(root / v / "acp_replay_buffer.zarr"), "r")["data"] for k, v in RUNS.items()}
    if name not in data["r1"]:
        raise SystemExit(f"fixb missing {name}")
    e = process_episode(name, data, field, scan_sd, R_off, {
        "timestamps_equal": True, "labels_r2_vs_r3_max_abs": 0.0, "mold_obs_max_abs_vs_r2": 0.0,
        "fb_r1_vs_r2_max_abs": 0.0, "r1_cmd_vs_fb_max_abs": 0.0,
    })
    e["source"] = "fixb"
    e["key"] = f"f{eid}"
    e["label"] = f"fixb ep {eid} ({e['split']})"
    return e, np.asarray(data["r1"][name]["rgb_0"])


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Fix B scrubber</title>
<script>/*PLOTLY*/</script>
<style>
body{margin:0;font:13px system-ui,sans-serif;color:#222;background:#fff}
#note{background:#fff8d8;border-bottom:1px solid #e6d98c;padding:6px 10px;font-size:12px;line-height:1.35}
#bar,#bar2{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:5px 10px;border-bottom:1px solid #ddd}
#slider{flex:1;min-width:240px}
#wrap{display:grid;grid-template-columns:minmax(0,1fr) 340px;grid-template-rows:52vh 42vh;gap:4px;padding:4px}
#scene{grid-column:1;grid-row:1;min-width:0}
#side{grid-column:2;grid-row:1/3;overflow:auto;border-left:1px solid #eee;padding:0 8px}
#wrist{display:block;width:224px;height:224px;background:#111;margin:6px 0;object-fit:contain}
#bottom{grid-column:1;grid-row:2;display:flex;flex-wrap:wrap;gap:4px;min-width:0;align-content:stretch}
.cell{flex:1 1 280px;display:flex;flex-direction:column;min-width:0;min-height:150px}
.cell.stack .plot{flex:1;min-height:70px}
.cell.stack .cap{padding-top:4px}
.cap{font-size:11px;line-height:1.3;padding:2px 4px;color:#444}
.plot{flex:1;min-height:0}
table{border-collapse:collapse;width:100%}
td{padding:2px 4px;border-bottom:1px solid #f2f2f2;vertical-align:top}
td:first-child{color:#555;white-space:nowrap}
.on{color:#b00020;font-weight:600}
.off{color:#999}
h3{margin:8px 0 4px;font-size:13px}
.note{color:#666;font-size:11px;line-height:1.35}
.sw{display:inline-block;width:10px;height:10px;margin-right:3px;vertical-align:middle}
</style></head><body>
<div id="note"></div>
<div id="bar">
  <label>episode <select id="ep"></select></label>
  <button id="prev">&lt;</button>
  <input type="range" id="slider" min="0" max="0" value="0">
  <button id="next">&gt;</button>
  <button id="play">play</button>
  <label>speed <select id="speed"><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option></select></label>
  <label>jump <select id="flag"></select></label>
  <button id="fprev">prev</button><button id="fnext">next</button>
</div>
<div id="bar2">
  <label>frame <select id="frame"><option value="right_base" selected>right_base (viser)</option><option value="fixture">fixture (dataset)</option></select></label>
  <label><input type="checkbox" id="rotate"> rotate mold+demos 90° about +Y</label>
  <label><input type="checkbox" id="follow" checked> follow, box</label>
  <select id="box"><option value="75">&plusmn;75 mm</option><option value="150" selected>&plusmn;150 mm</option><option value="300">&plusmn;300 mm</option><option value="600">&plusmn;600 mm</option></select>
  <label><span class="sw" style="background:#1f77b4"></span><input type="checkbox" id="show_r1" checked> r1</label>
  <label><span class="sw" style="background:#ff7f0e"></span><input type="checkbox" id="show_r2" checked> r2</label>
  <label><span class="sw" style="background:#2ca02c"></span><input type="checkbox" id="show_r3" checked> r3</label>
  <label><input type="checkbox" id="arrows" checked> step arrows</label>
  <label><input type="checkbox" id="legend" checked> legend</label>
  <button id="sa">stroke A</button><button id="sb">stroke B</button>
</div>
<div id="wrap">
  <div id="scene"></div>
  <div id="side">
    <img id="wrist" alt="wrist rgb">
    <div id="readout"></div>
  </div>
  <div id="bottom">
    <div class="cell" id="cell_r1"><div class="cap"><b>r1 path</b> command minus k=0 [mm]
      (<span style="color:#d62728">x</span> <span style="color:#2ca02c">y</span> <span style="color:#1f77b4">z</span>);
      purple ▲ = foot hop in this chunk</div>
      <div class="plot" id="lab_r1"></div></div>
    <div class="cell stack" id="cell_fixb">
      <div class="cap"><b>r2 path</b> decoded x_d minus k=0 [mm] — same units as r1
        (<span style="color:#d62728">x</span> <span style="color:#2ca02c">y</span> <span style="color:#1f77b4">z</span>); purple ▲ = hop</div>
      <div class="plot" id="lab_fixb_path"></div>
      <div class="cap"><b>r2 rates</b> what the policy sees:
        <span style="color:#d62728">v_t1</span> <span style="color:#2ca02c">v_t2</span> [mm/s];
        <span style="color:#1f77b4">delta_n</span> [mm]. Spikes at hops are expected.</div>
      <div class="plot" id="lab_fixb_vel"></div>
    </div>
    <div class="cell" id="cell_tl"><div class="cap"><b>timeline</b> |p−x_s| / sd [mm]; twist ψ; click to jump</div>
      <div class="plot" id="timeline"></div></div>
    <div class="cell" id="cell_st"><div class="cap"><b>two strokes</b> r2 (v_t1, v_t2) vs normalized time — should match if the local frame works</div>
      <div class="plot" id="strokes"></div></div>
    <div class="cell" id="cell_co"><div class="cap"><b>label vs n_z</b> world Δz tracks the surface; v∥ should not</div>
      <div class="plot" id="corr"></div></div>
  </div>
</div>
<script>
const D = /*DATA*/;
const H = D.H, RC = {r1:"#1f77b4", r2:"#ff7f0e", r3:"#2ca02c"};
const AXC = ["#d62728","#2ca02c","#1f77b4"];
const ERR_CS = [[0,"#fff5eb"],[0.1,"#fdd0a2"],[0.3,"#fd8d3c"],[0.6,"#d94801"],[1,"#7f2704"]];
const CFG = {displaylogo:false, responsive:true};
const $ = id => document.getElementById(id);
const scene = $("scene"), labR1 = $("lab_r1"), labPath = $("lab_fixb_path"), labVel = $("lab_fixb_vel"), tl = $("timeline");
let E = null, ci = 0, timer = null, tlBound = false, NSTATIC = 0, PATH_IDX = 0, MESHES = [], clipKey = "", lastVis = "";
const show = {r1:true, r2:true, r3:true};

const p3 = (a,i) => [a[3*i], a[3*i+1], a[3*i+2]];
const c3 = (R,i,c) => [R[9*i+c], R[9*i+3+c], R[9*i+6+c]];
const sub = (a,b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const dot = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const deg = r => r*180/Math.PI;
const f1 = x => (x===null || Number.isNaN(x)) ? "nan" : (+x).toFixed(1);
const pts = l => ({x:l.map(p=>p[0]), y:l.map(p=>p[1]), z:l.map(p=>p[2])});
const seg = (p,d,s) => ({x:[p[0], p[0]+s*d[0]], y:[p[1], p[1]+s*d[1]], z:[p[2], p[2]+s*d[2]]});

function xfP(p) {
  let x=p[0], y=p[1], z=p[2];
  if ($("rotate").checked) { const t=x; x=z; z=-t; }
  if ($("frame").value === "right_base") {
    const R=D.R_rb, t=D.t_rb_mm;
    return [R[0][0]*x+R[0][1]*y+R[0][2]*z+t[0], R[1][0]*x+R[1][1]*y+R[1][2]*z+t[1], R[2][0]*x+R[2][1]*y+R[2][2]*z+t[2]];
  }
  return [x,y,z];
}
function xfV(v) {
  let x=v[0], y=v[1], z=v[2];
  if ($("rotate").checked) { const t=x; x=z; z=-t; }
  if ($("frame").value === "right_base") {
    const R=D.R_rb;
    return [R[0][0]*x+R[0][1]*y+R[0][2]*z, R[1][0]*x+R[1][1]*y+R[1][2]*z, R[2][0]*x+R[2][1]*y+R[2][2]*z];
  }
  return [x,y,z];
}
function xfList(a) {
  const o=[];
  for (let i=0;i<a.length;i+=3) { const q=xfP([a[i],a[i+1],a[i+2]]); o.push(q[0],q[1],q[2]); }
  return o;
}
function xfMesh(m) {
  const x=[], y=[], z=[];
  for (let i=0;i<m.x.length;i++) { const q=xfP([m.x[i], m.y[i], m.z[i]]); x.push(q[0]); y.push(q[1]); z.push(q[2]); }
  return {x,y,z,i:m.i,j:m.j,k:m.k};
}

function line3(name, color, width, extra) {
  return Object.assign({type:"scatter3d", mode:"lines", x:[], y:[], z:[], name, hoverinfo:"skip", line:{color, width}}, extra||{});
}
function meshTrace(m, name, color, opacity) {
  const t = xfMesh(m);
  return {type:"mesh3d", x:t.x, y:t.y, z:t.z, i:t.i, j:t.j, k:t.k, name, color, opacity,
          showlegend:true, hoverinfo:"skip", flatshading:true,
          lighting:{ambient:0.55, diffuse:0.75, specular:0.05, roughness:0.9}};
}

const DYN = [
  ["pts",null],["near",null],["ntxt",null],
  ["tx",null],["ty",null],["tz",null],["ft1",null],["ft2",null],["fn",null],
  ["r1","r1"],["r1vt","r1"],["r1x","r1"],["r1y","r1"],["r1z","r1"],
  ["r2","r2"],["r3","r3"],["dx","fixb"],["dy","fixb"],["dz","fixb"],
  ["ar1","r1"],["ar2","r2"],["adn","r2"],["stA",null],["stB",null],
];
const MC = {pts:["#000000","#d62ad6"], r1:RC.r1, r1vt:RC.r1, r3:RC.r3, ar1:"#1f77b4", ar2:"#ff7f0e", adn:"#9467bd", stA:"#e377c2", stB:"#17becf"};

function dynTraces() {
  const dash = {dash:"dash"};
  return [
    {type:"scatter3d", mode:"markers", x:[], y:[], z:[], name:"tool p / nearest x_s",
     marker:{size:5, color:["#000","#d62ad6"]}, hoverinfo:"skip"},
    line3("nearest-point line", "#000", 6),
    {type:"scatter3d", mode:"text", x:[], y:[], z:[], text:[], name:"distance", showlegend:false,
     textposition:"middle right", textfont:{size:11}, hoverinfo:"skip"},
    line3("tool x", AXC[0], 8, {legendgroup:"tool", showlegend:false}),
    line3("tool y", AXC[1], 8, {legendgroup:"tool", showlegend:false}),
    line3("tool axes (solid)", AXC[2], 8, {legendgroup:"tool"}),
    line3("t1", AXC[0], 4, {line:Object.assign({color:AXC[0], width:4}, dash), legendgroup:"field", showlegend:false}),
    line3("t2", AXC[1], 4, {line:Object.assign({color:AXC[1], width:4}, dash), legendgroup:"field", showlegend:false}),
    line3("R(x_s) t1/t2/n (dashed)", AXC[2], 4, {line:Object.assign({color:AXC[2], width:4}, dash), legendgroup:"field"}),
    {type:"scatter3d", mode:"lines+markers", x:[], y:[], z:[], name:"r1 24 cmd positions",
     line:{color:RC.r1, width:5}, marker:{size:3, color:RC.r1}, hoverinfo:"skip"},
    {type:"scatter3d", mode:"lines+markers", x:[], y:[], z:[], name:"r1 virtual target",
     line:{color:RC.r1, width:2, dash:"dot"}, marker:{size:2, color:RC.r1}, opacity:0.6, hoverinfo:"skip"},
    line3("r1x", AXC[0], 3, {legendgroup:"r1ax", showlegend:false}),
    line3("r1y", AXC[1], 3, {legendgroup:"r1ax", showlegend:false}),
    line3("r1 cmd axes k=23", AXC[2], 3, {legendgroup:"r1ax"}),
    {type:"scatter3d", mode:"lines+markers", x:[], y:[], z:[], name:"r2 decoded from p(k=0)",
     line:{color:RC.r2, width:5}, hoverinfo:"skip",
     marker:{size:4, color:[], cmin:0, cmax:30, colorscale:ERR_CS,
             colorbar:{title:{text:"decode err [mm]"}, len:0.32, y:0.16, thickness:10}}},
    {type:"scatter3d", mode:"lines+markers", x:[], y:[], z:[], name:"r3 decoded (same labels)",
     line:{color:RC.r3, width:2, dash:"dot"}, marker:{size:7, symbol:"circle-open", color:RC.r3}, hoverinfo:"skip"},
    line3("dec x", AXC[0], 3, {legendgroup:"decax", showlegend:false, line:{color:AXC[0], width:3, dash:"dot"}}),
    line3("dec y", AXC[1], 3, {legendgroup:"decax", showlegend:false, line:{color:AXC[1], width:3, dash:"dot"}}),
    line3("decoded heading k=23", AXC[2], 3, {legendgroup:"decax", line:{color:AXC[2], width:3, dash:"dot"}}),
    line3("r1 world step arrows", RC.r1, 4),
    line3("r2 (v∥ dt) in world", RC.r2, 4),
    line3("r2 δn along n", "#9467bd", 3),
    line3("stroke A", "#e377c2", 6),
    line3("stroke B", "#17becf", 6),
  ];
}

function sceneLayout() {
  const cam = D.cam[$("frame").value] || D.cam.fixture;
  return {
    margin:{l:0,r:0,t:0,b:0}, uirevision:"keep", showlegend:$("legend").checked,
    legend:{x:0, y:1, font:{size:10}, bgcolor:"rgba(255,255,255,0.7)"},
    scene:{
      aspectmode:"data",
      xaxis:{title:{text:"x [mm]"}}, yaxis:{title:{text:"y [mm]"}}, zaxis:{title:{text:"z [mm]"}},
      camera:{up:{x:cam.up[0], y:cam.up[1], z:cam.up[2]},
              eye:{x:cam.eye[0], y:cam.eye[1], z:cam.eye[2]},
              center:{x:0, y:0, z:0}},
    },
  };
}

function staticTraces() {
  const t = [];
  MESHES = [];
  if (D.mesh.scan) { MESHES.push([D.mesh.scan, t.length]); t.push(meshTrace(D.mesh.scan, "scan mesh", "#b8b8b8", 0.55)); }
  if (D.mesh.apron) { MESHES.push([D.mesh.apron, t.length]); t.push(meshTrace(D.mesh.apron, "flange apron", "#e3dccb", 0.3)); }
  PATH_IDX = t.length;
  t.push({type:"scatter3d", mode:"lines", x:[], y:[], z:[], name:"demo path", line:{color:"#8c8c8c", width:2}, hoverinfo:"skip"});
  t.push({type:"scatter3d", mode:"markers", x:[], y:[], z:[], name:"normal flips", marker:{color:"#d62728", size:3, symbol:"x"}, hoverinfo:"skip"});
  NSTATIC = t.length; clipKey = ""; lastVis = "";
  return t;
}

function axisLen() { return $("follow").checked ? +$("box").value / 5 : 60; }

function sceneRanges(p) {
  if (!$("follow").checked)
    return {"scene.xaxis.autorange":true, "scene.yaxis.autorange":true, "scene.zaxis.autorange":true, "scene.aspectmode":"data"};
  const b = +$("box").value;
  return {"scene.xaxis.range":[p[0]-b, p[0]+b], "scene.yaxis.range":[p[1]-b, p[1]+b],
          "scene.zaxis.range":[p[2]-b, p[2]+b], "scene.aspectmode":"cube"};
}

function nullSegs(pairs) {
  const x=[], y=[], z=[];
  for (const [a,b] of pairs) {
    x.push(a[0], b[0], null); y.push(a[1], b[1], null); z.push(a[2], b[2], null);
  }
  return {x,y,z};
}

function nearInfo(q) {
  const p = xfP(p3(E.P,q)), xs = xfP(p3(E.X,q)), n = xfV(c3(E.Rf,q,2)), d = sub(p, xs);
  const dist = Math.hypot(d[0], d[1], d[2]), sd = E.sd[q];
  const lineAng = dist > 1e-6 ? deg(Math.acos(Math.min(1, Math.abs(dot(d,n))/dist))) : 0;
  const tzn = deg(Math.acos(Math.max(-1, Math.min(1, dot(xfV(c3(E.Rt,q,2)), n)))));
  return {p, xs, n, dist, sd, lineAng, tzn, inpl: Math.sqrt(Math.max(dist*dist - sd*sd, 0))};
}

function strokePts(s) {
  const out=[];
  for (let i=0;i<s.P.length/3;i++) out.push(xfP(p3(s.P,i)));
  return out;
}

function chunkGeometry() {
  const q = E.q[ci], N = E.t.length, L = axisLen();
  const ni = nearInfo(q), p = ni.p, xs = ni.xs, g = {};
  g.pts = pts([p, xs]); g.near = pts([p, xs]);
  g.ntxt = Object.assign(pts([[(p[0]+xs[0])/2,(p[1]+xs[1])/2,(p[2]+xs[2])/2]]), {
    text:[`${f1(ni.dist)} mm (sd ${f1(ni.sd)})\nline vs n ${f1(ni.lineAng)}°, tilt ${f1(E.tilt[q])}°`]});
  ["tx","ty","tz"].forEach((k,c) => g[k] = seg(p, xfV(c3(E.Rt,q,c)), L));
  ["ft1","ft2","fn"].forEach((k,c) => g[k] = seg(xs, xfV(c3(E.Rf,q,c)), L));
  const r1=[], vt=[], dec=[];
  for (let k=0;k<H;k++) {
    const i = Math.min(q+k, N-1);
    r1.push(xfP(p3(E.P,i))); vt.push(xfP(p3(E.vt,i))); dec.push(xfP(p3(E.xd, ci*H+k)));
  }
  g.r1 = pts(r1); g.r1vt = pts(vt); g.r2 = pts(dec); g.r3 = pts(dec);
  const ie = Math.min(q+H-1, N-1);
  ["r1x","r1y","r1z"].forEach((k,c) => g[k] = seg(r1[H-1], xfV(c3(E.Rt,ie,c)), L));
  ["dx","dy","dz"].forEach((k,c) => g[k] = seg(dec[H-1], xfV(c3(E.Rd,ci,c)), L));
  const ar1=[], ar2=[], adn=[];
  if ($("arrows").checked) {
    for (let k=0;k<H-1;k++) {
      const i = Math.min(q+k, N-2), j = Math.min(i+1, N-1);
      const a = xfP(p3(E.P,i)), b = xfP(p3(E.P,j));
      ar1.push([a,b]);
      const dt = Math.max(1e-3, E.t[j]-E.t[i]);
      const t1 = xfV(c3(E.Rf,i,0)), t2 = xfV(c3(E.Rf,i,1)), nn = xfV(c3(E.Rf,i,2));
      const v1 = E.v[2*i], v2 = E.v[2*i+1], dn = E.dn[i];
      ar2.push([a, [a[0]+t1[0]*v1*dt, a[1]+t1[1]*v1*dt, a[2]+t1[2]*v1*dt]]);
      const xsi = xfP(p3(E.X,i));
      adn.push([xsi, [xsi[0]+nn[0]*dn, xsi[1]+nn[1]*dn, xsi[2]+nn[2]*dn]]);
    }
  }
  g.ar1 = nullSegs(ar1); g.ar2 = nullSegs(ar2); g.adn = nullSegs(adn);
  g.stA = E.strokes[0] ? pts(strokePts(E.strokes[0])) : {x:[],y:[],z:[]};
  g.stB = E.strokes[1] ? pts(strokePts(E.strokes[1])) : {x:[],y:[],z:[]};
  return {g, p};
}

function visible(group) {
  if (group===null) return true;
  if (group==="fixb") return show.r2 || show.r3;
  return show[group];
}

function clipStatic(p) {
  const follow = $("follow").checked, b = +$("box").value;
  const qx = follow ? Math.round(p[0] / Math.max(b/2, 1)) : 0;
  const qy = follow ? Math.round(p[1] / Math.max(b/2, 1)) : 0;
  const qz = follow ? Math.round(p[2] / Math.max(b/2, 1)) : 0;
  const key = [$("ep").value, follow, b, $("frame").value, +$("rotate").checked, qx, qy, qz].join("|");
  if (key === clipKey) return;
  clipKey = key;
  const B = 1.6 * b;
  const ins = q => !follow || (Math.abs(q[0]-p[0])<=B && Math.abs(q[1]-p[1])<=B && Math.abs(q[2]-p[2])<=B);
  if (MESHES.length) {
    const xs=[], ys=[], zs=[], ii=[], jj=[], kk=[];
    for (const [m] of MESHES) {
      const xm = xfMesh(m);
      xs.push(xm.x); ys.push(xm.y); zs.push(xm.z);
      if (!follow) { ii.push(m.i); jj.push(m.j); kk.push(m.k); continue; }
      const inV = xm.x.map((x,v) => ins([x, xm.y[v], xm.z[v]]));
      const i=[], j=[], k=[];
      for (let f=0; f<m.i.length; f++) if (inV[m.i[f]]||inV[m.j[f]]||inV[m.k[f]]) { i.push(m.i[f]); j.push(m.j[f]); k.push(m.k[f]); }
      ii.push(i); jj.push(j); kk.push(k);
    }
    Plotly.restyle(scene, {x:xs,y:ys,z:zs,i:ii,j:jj,k:kk}, MESHES.map(([,ti])=>ti));
  }
  const path=[], fl=[];
  for (let i=0;i<E.t.length;i++) {
    const q=xfP(p3(E.P,i));
    path.push(ins(q)?q:[null,null,null]);
  }
  E.flips.forEach(i => { const q=xfP(p3(E.P,i)); if (ins(q)) fl.push(q); });
  const a=pts(path), f=pts(fl);
  Plotly.restyle(scene, {x:[a.x,f.x], y:[a.y,f.y], z:[a.z,f.z]}, [PATH_IDX, PATH_IDX+1]);
}

function drawScene() {
  const {g,p} = chunkGeometry();
  clipStatic(p);
  const idx = DYN.map((_,j) => NSTATIC+j);
  const data = {
    x: DYN.map(([k]) => g[k].x), y: DYN.map(([k]) => g[k].y), z: DYN.map(([k]) => g[k].z),
    text: DYN.map(([k]) => g[k].text || []),
    "marker.color": DYN.map(([k]) => k==="r2" ? E.err.slice(ci*H, ci*H+H) : (MC[k] || "#000")),
  };
  const vis = DYN.map(([,grp]) => visible(grp));
  const vkey = vis.join()+"|"+$("arrows").checked;
  if (vkey !== lastVis) { data.visible = vis; lastVis = vkey; }
  Plotly.update(scene, data, sceneRanges(p), idx);
}

function hopSteps(q) {
  const out = [];
  (E.hops || []).forEach(i => { const k = i - q; if (k >= 0 && k < H) out.push(k); });
  return out;
}

function hopMarkers(hops, ys) {
  if (!hops.length) return [];
  return [{
    x: hops, y: ys, mode: "markers", name: "foot hop",
    marker: {symbol: "triangle-up", size: 9, color: "#9467bd", line: {width: 1, color: "#4a2a6b"}},
    hovertemplate: "hop at k=%{x}<extra></extra>",
  }];
}

function hopShapes(hops) {
  return hops.map(k => ({
    type: "line", xref: "x", yref: "paper", x0: k, x1: k, y0: 0, y1: 1,
    line: {color: "#9467bd", width: 1, dash: "dot"},
  }));
}

function drawLabels() {
  const q = E.q[ci], N = E.t.length, ks = [...Array(H).keys()], p0 = p3(E.P, q);
  const r1 = [[], [], []], r2p = [[], [], []], v1 = [], v2 = [], dn = [];
  const d0 = p3(E.xd, ci * H);
  for (const k of ks) {
    const i = Math.min(q + k, N - 1), p = p3(E.P, i), d = p3(E.xd, ci * H + k);
    for (let c = 0; c < 3; c++) { r1[c].push(p[c] - p0[c]); r2p[c].push(d[c] - d0[c]); }
    v1.push(E.v[2 * i]); v2.push(E.v[2 * i + 1]); dn.push(E.dn[i]);
  }
  const hops = hopSteps(q);
  const base = {margin: {l: 40, r: 36, t: 4, b: 28}, font: {size: 10}, showlegend: false,
                xaxis: {title: {text: "k"}, range: [-0.5, H - 0.5]}, shapes: hopShapes(hops)};
  if (show.r1) {
    const t = ["x", "y", "z"].map((a, c) => ({x: ks, y: r1[c], mode: "lines+markers",
      marker: {size: 3}, line: {color: AXC[c]}, name: a}));
    t.push(...hopMarkers(hops, hops.map(k => Math.max(r1[0][k], r1[1][k], r1[2][k]))));
    Plotly.react(labR1, t, Object.assign({}, base, {yaxis: {title: {text: "mm"}}}), CFG);
  }
  if (show.r2 || show.r3) {
    const pathT = ["x", "y", "z"].map((a, c) => ({x: ks, y: r2p[c], mode: "lines+markers",
      marker: {size: 3}, line: {color: AXC[c]}, name: a}));
    pathT.push(...hopMarkers(hops, hops.map(k => Math.max(r2p[0][k], r2p[1][k], r2p[2][k]))));
    Plotly.react(labPath, pathT, Object.assign({}, base, {yaxis: {title: {text: "mm"}}}), CFG);

    const velT = [
      {x: ks, y: v1, mode: "lines+markers", marker: {size: 3}, line: {color: AXC[0]}, name: "v_t1"},
      {x: ks, y: v2, mode: "lines+markers", marker: {size: 3}, line: {color: AXC[1]}, name: "v_t2"},
      {x: ks, y: dn, mode: "lines+markers", marker: {size: 3}, yaxis: "y2", line: {color: AXC[2]}, name: "delta_n"},
    ];
    velT.push(...hopMarkers(hops, hops.map(k => v2[k])));
    Plotly.react(labVel, velT, Object.assign({}, base, {
      yaxis: {title: {text: "mm/s"}},
      yaxis2: {title: {text: "mm"}, overlaying: "y", side: "right", showgrid: false},
    }), CFG);
  }
}

function buildTimeline() {
  const fl=E.flips, hp=E.hops;
  Plotly.react(tl, [
    {x:E.t, y:E.dist, name:"|p-x_s|", mode:"lines", line:{color:"#000", width:1}},
    {x:E.t, y:E.sd, name:"sd", mode:"lines", line:{color:"#999", width:1}},
    {x:fl.map(i=>E.t[i]), y:fl.map(i=>E.dist[i]), name:"flip", mode:"markers", marker:{color:"#d62728", symbol:"x", size:6}},
    {x:hp.map(i=>E.t[i]), y:hp.map(i=>E.dist[i]), name:"hop", mode:"markers", marker:{color:"#9467bd", symbol:"triangle-up", size:5}},
    {x:E.t, y:E.psi, name:"psi", mode:"markers", yaxis:"y2", marker:{size:2, color:E.flip.map(b=>b?"#d62728":"#1f77b4")}},
  ], {
    margin:{l:44,r:8,t:6,b:32}, font:{size:10}, showlegend:false,
    xaxis:{title:{text:"t [s]"}, anchor:"y2"},
    yaxis:{title:{text:"mm"}, domain:[0.42,1], range:[-60,200]},
    yaxis2:{title:{text:"psi"}, domain:[0,0.34], range:[-185,185], tickvals:[-180,-90,0,90,180]},
    shapes:[
      {type:"rect", xref:"x", yref:"paper", x0:0, x1:0, y0:0, y1:1, fillcolor:"#ff7f0e", opacity:0.18, line:{width:0}},
      {type:"line", xref:"x", yref:"paper", x0:0, x1:0, y0:0, y1:1, line:{color:"#ff7f0e", width:2}},
    ],
  }, CFG);
  if (!tlBound) {
    tl.on("plotly_click", ev => {
      const tc = ev.points[0].x; let best=0;
      E.q.forEach((q,j)=>{ if (Math.abs(E.t[q]-tc) < Math.abs(E.t[E.q[best]]-tc)) best=j; });
      setChunk(best);
    });
    tlBound = true;
  }
}

function drawStrokes() {
  const t = [];
  const col = [["#e377c2","#c49"],["#17becf","#1a8"]];
  E.strokes.forEach((s,i) => {
    const tn = s.tn, n = tn.length;
    const v1=[], v2=[];
    for (let k=0;k<n;k++) { v1.push(s.v[2*k]); v2.push(s.v[2*k+1]); }
    t.push({x:tn, y:v1, name:`${i?"B":"A"} v_t1`, mode:"lines", line:{color:col[i][0]}});
    t.push({x:tn, y:v2, name:`${i?"B":"A"} v_t2`, mode:"lines", line:{color:col[i][1], dash:"dot"}});
  });
  Plotly.react($("strokes"), t, {
    margin:{l:40,r:8,t:6,b:32}, font:{size:10}, showlegend:true, legend:{orientation:"h", y:-0.28, font:{size:9}},
    xaxis:{title:{text:"normalized stroke time"}, range:[0,1]}, yaxis:{title:{text:"mm/s"}},
  }, CFG);
}

function drawCorr() {
  const S = D.stats[E.source];
  if (!S || !S.scatter) { Plotly.purge($("corr")); return; }
  const sc = S.scatter;
  Plotly.react($("corr"), [
    {x:sc.nz, y:sc.r1_dpz_mm, name:`r1 Δz (r=${S.corr_r1.dp_z_vs_n_z})`, mode:"markers", marker:{size:4, color:RC.r1, opacity:0.45}},
    {x:sc.nz, y:sc.r2_vt1_mm_s, name:`r2 v_t1 (r=${S.corr_r2.v_t1_vs_n_z})`, mode:"markers", marker:{size:4, color:RC.r2, opacity:0.45}, yaxis:"y2"},
  ], {
    margin:{l:40,r:36,t:6,b:32}, font:{size:10}, showlegend:true, legend:{orientation:"h", y:-0.28, font:{size:9}},
    xaxis:{title:{text:"n_z"}}, yaxis:{title:{text:"r1 Δz [mm]"}},
    yaxis2:{title:{text:"r2 v_t1 [mm/s]"}, overlaying:"y", side:"right", showgrid:false},
  }, CFG);
}

function drawCursor() {
  const q=E.q[ci], t0=E.t[q], t1=E.t[Math.min(q+H-1, E.t.length-1)];
  Plotly.relayout(tl, {"shapes[0].x0":t0,"shapes[0].x1":t1,"shapes[1].x0":t0,"shapes[1].x1":t0});
}

function drawWrist() {
  if (!E.rgb_dir) { $("wrist").style.display="none"; return; }
  $("wrist").style.display="";
  $("wrist").src = E.rgb_dir + String(E.q[ci]).padStart(5,"0") + ".jpg";
}

function readout() {
  const q=E.q[ci], f=E.flags[ci], ni=nearInfo(q);
  const loc=p3(E.loc,q), vtOff=Math.hypot(...sub(p3(E.vt,q), p3(E.P,q)));
  const e0=E.err[ci*H], e23=E.err[ci*H+H-1];
  const row=(k,v,hot)=>`<tr><td>${k}</td><td class="${hot?"on":""}">${v}</td></tr>`;
  const S=D.stats[E.source]||{};
  let h=`<h3>${E.label}, t=${E.t[q].toFixed(2)} s</h3>`;
  h+=`<div class="note">chunk ${ci+1}/${E.q.length} · RGB ${q} · k23 p50 ${E.k23_p50_mm} mm · near ${E.near_frac} · force-peak sd p50 ${E.sd_peak_p50_mm} mm</div>`;
  h+=`<h3>nearest point</h3><table>`;
  h+=row("|p − x_s|", `${f1(ni.dist)} mm`, ni.dist>=D.near_mm);
  h+=row("signed distance", `${f1(ni.sd)} mm`);
  h+=row("in-plane", `${f1(ni.inpl)} mm`, ni.inpl>D.off_edge_mm);
  h+=row("line vs n", `${f1(ni.lineAng)} deg`, ni.lineAng>10);
  h+=row("tool z vs n", `${f1(ni.tzn)} deg`);
  h+=row("tilt / twist ψ", `${f1(E.tilt[q])} / ${f1(E.psi[q])} deg`, Math.abs(Math.abs(E.psi[q])-90)<D.branch_margin_deg);
  h+=row("heading B0", E.flip[q] ? "180° branch" : "R_off");
  h+=`</table><h3>flags</h3><table>`;
  D.flags.forEach((k,b)=>{ h+=`<tr><td title="${D.flag_text[k]}">${k}</td><td class="${(f>>b)&1?"on":"off"}">${(f>>b)&1?"yes":"no"}</td></tr>`; });
  h+=`</table>`;
  if (show.r1) {
    h+=`<h3>r1</h3><table>`+row("virtual target |offset| k=0", `${f1(vtOff)} mm`)+`</table>`;
  }
  if (show.r2||show.r3) {
    h+=`<h3>r2/r3 Fix B</h3><table>`;
    h+=row("v_t1, v_t2", `${f1(E.v[2*q])}, ${f1(E.v[2*q+1])} mm/s`);
    h+=row("delta_n", `${f1(E.dn[q])} mm`);
    h+=row("decode k0 / k23", `${f1(e0)} / ${f1(e23)} mm`, e23>20);
    h+=row("heading err k0 / k23", `${f1(E.he[2*ci])} / ${f1(E.he[2*ci+1])} deg`);
    h+=`</table>`;
  }
  if (show.r3) {
    h+=`<h3>r3 obs</h3><table>`+row("Rᵀ(p−x_s)", `${f1(loc[0])}, ${f1(loc[1])}, ${f1(loc[2])} mm`)+
       row("k1, k2", `${E.curv[2*q].toFixed(2)}, ${E.curv[2*q+1].toFixed(2)} 1/m`)+`</table>`;
  }
  if (S.corr_r1) {
    const R=S.rotate_y90;
    h+=`<h3>orientation corr (near RGB, this source)</h3><table>`;
    h+=row("r1 Δz vs n_z", S.corr_r1.dp_z_vs_n_z);
    h+=row("r1 Δx vs n_x", S.corr_r1.dp_x_vs_n_x);
    h+=row("r2 v_t1 vs n_z", S.corr_r2.v_t1_vs_n_z);
    h+=row("r2 v_t2 vs n_z", S.corr_r2.v_t2_vs_n_z);
    h+=row("r2 δn vs n_z", S.corr_r2.delta_n_vs_n_z);
    h+=`</table><h3>rotate-the-mold (90° about +Y)</h3><table>`;
    h+=row("r2 |Δv∥| after rotating mold+demo", `${R.v_par_max_abs_change_mm_s} mm/s (want 0)`);
    h+=row("r1 |Δ(world step)| p50", `${R.r1_world_dp_change_p50_mm} mm (changes)`);
    h+=row("r1 Δz vs n_z after rotate", R.r1_dpz_vs_nz_after_rotate);
    h+=row("r2 v_t1 vs n_z after rotate", R.r2_vt1_vs_nz_after_rotate);
    h+=`</table>`;
  }
  h+=`<h3>notes</h3><div class="note">${D.blurb}</div>`;
  $("readout").innerHTML = h;
}

function layoutCells() {
  $("cell_r1").style.display = show.r1 ? "" : "none";
  $("cell_fixb").style.display = (show.r2||show.r3) ? "" : "none";
}

function setChunk(j) {
  ci = Math.max(0, Math.min(E.q.length-1, j));
  $("slider").value = ci;
  layoutCells();
  drawScene(); drawLabels(); drawCursor(); drawWrist(); readout();
  history.replaceState(null,"", `#ep=${$("ep").value}&chunk=${ci}&frame=${$("frame").value}&rot=${+$("rotate").checked}`);
}

function setEpisode(id, chunk) {
  E = D.episodes[id];
  $("ep").value = id;
  $("slider").max = E.q.length-1;
  $("sa").disabled = !E.strokes[0];
  $("sb").disabled = !E.strokes[1];
  Plotly.react(scene, staticTraces().concat(dynTraces()), sceneLayout(), CFG);
  buildTimeline(); drawStrokes(); drawCorr();
  setChunk(chunk||0);
}

function stop(){ if(timer){ clearInterval(timer); timer=null; $("play").textContent="play"; } }
function play(){
  if (timer) return stop();
  $("play").textContent="pause";
  const dt = 1000*(E.t[E.q[Math.min(1,E.q.length-1)]]-E.t[E.q[0]] || 0.133);
  timer=setInterval(()=>{ if(ci>=E.q.length-1) return stop(); setChunk(ci+1); }, dt/+$("speed").value);
}
function jumpFlag(dir){
  const b=D.flags.indexOf($("flag").value);
  for (let j=ci+dir; j>=0 && j<E.q.length; j+=dir) if ((E.flags[j]>>b)&1) return setChunk(j);
}

$("note").textContent = D.banner;
D.order.forEach(id => {
  const o=document.createElement("option"); o.value=id; o.textContent=D.episodes[id].label; $("ep").appendChild(o);
});
D.flags.forEach(k => { const o=document.createElement("option"); o.value=k; o.textContent=k; $("flag").appendChild(o); });
$("ep").onchange=()=>{ stop(); setEpisode($("ep").value); };
$("slider").oninput=()=>setChunk(+$("slider").value);
$("prev").onclick=()=>setChunk(ci-1); $("next").onclick=()=>setChunk(ci+1);
$("play").onclick=play; $("speed").onchange=()=>{ if(timer){ stop(); play(); } };
$("fprev").onclick=()=>jumpFlag(-1); $("fnext").onclick=()=>jumpFlag(1);
$("follow").onchange=()=>setChunk(ci); $("box").onchange=()=>setChunk(ci);
$("arrows").onchange=()=>setChunk(ci);
$("legend").onchange=()=>Plotly.relayout(scene,{showlegend:$("legend").checked});
["r1","r2","r3"].forEach(r => $(`show_${r}`).onchange=()=>{ show[r]=$(`show_${r}`).checked; setChunk(ci); });
$("frame").onchange=()=>{ stop(); setEpisode($("ep").value, ci); };
$("rotate").onchange=()=>{ stop(); setEpisode($("ep").value, ci); };
$("sa").onclick=()=>{ if(E.strokes[0]) setChunk(E.strokes[0].q0); };
$("sb").onclick=()=>{ if(E.strokes[1]) setChunk(E.strokes[1].q0); };
document.addEventListener("keydown", ev => {
  if (ev.target.tagName==="SELECT" || (ev.target.tagName==="INPUT" && ev.target.type!=="range")) return;
  if (ev.key==="ArrowRight") { ev.preventDefault(); setChunk(ci+1); }
  if (ev.key==="ArrowLeft") { ev.preventDefault(); setChunk(ci-1); }
  if (ev.key===" ") { ev.preventDefault(); play(); }
});
const hash = Object.fromEntries(location.hash.slice(1).split("&").filter(Boolean).map(s=>s.split("=")));
if (hash.frame) $("frame").value = hash.frame;
if (hash.rot==="1") $("rotate").checked = true;
setEpisode(D.episodes[hash.ep] ? hash.ep : D.order[0], +hash.chunk || 0);
</script></body></html>
"""


def sitting_cam(verts_mm, frame: str):
    pts = np.asarray(verts_mm, dtype=float) * 1e-3
    c = pts.mean(0)
    _, ev = np.linalg.eigh(np.cov((pts - c).T))
    up = ev[:, 0]
    up = up / (np.linalg.norm(up) + 1e-30)
    if abs(up[1]) >= abs(up[2]) and abs(up[1]) >= abs(up[0]):
        if up[1] > 0:
            up = -up
    elif up[2] < 0:
        up = -up
    span = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    dist = max(0.28, 1.35 * span)
    long_ax = ev[:, 2]
    horiz = long_ax - up * float(long_ax @ up)
    if np.linalg.norm(horiz) < 1e-8:
        horiz = np.array([1.0, 0.0, 0.0]) - up * float(up[0])
    horiz = horiz / (np.linalg.norm(horiz) + 1e-30)
    if horiz[0] < 0:
        horiz = -horiz
    view = np.cross(up, horiz)
    view = view / (np.linalg.norm(view) + 1e-30)
    eye = c + dist * (0.50 * view + 0.72 * up + 0.22 * horiz)
    # plotly eye is relative to center, in scene units (mm) roughly as a direction
    rel = (eye - c)
    rel = rel / (np.linalg.norm(rel) + 1e-30) * 1.8
    return {"up": rel_round(up), "eye": rel_round(rel), "frame": frame}


def rel_round(x):
    return np.round(np.asarray(x, float), 4).tolist()


def xform_pts(P, R, t):
    return P @ R.T + t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixb_root", type=Path, default=FIXB_ROOT)
    ap.add_argument("--scanner_zarr", type=Path, default=SCANNER_ZARR)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "eval" / "pretrain_viz")
    ap.add_argument("--scanner_episodes", type=int, nargs="*", default=[0, 10, 40])
    ap.add_argument("--fixb_episodes", type=int, nargs="*", default=[9, 20])
    ap.add_argument("--skip_rgb", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    field = FixBMoldField(args.fixb_root / "shared" / "fixb_field.npz")
    npz = np.load(args.fixb_root / "shared" / "fixb_field.npz")
    field.is_apron = npz["is_apron"].astype(bool) if "is_apron" in npz.files else None
    meta = json.loads((args.fixb_root / "shared" / "fixb_meta.json").read_text())
    R_off = np.asarray(meta["R_off"], dtype=np.float64)
    scan_sd = ScanOnlySD(field)
    T_rb = load_T_rb_f()
    R_rb, t_rb = T_rb[:3, :3], T_rb[:3, 3]

    eps = []
    rgb_of = {}
    scan_root = zarr.open(str(args.scanner_zarr), "r")["data"]
    for eid in args.scanner_episodes:
        name = f"episode_{eid}"
        if name not in scan_root:
            raise SystemExit(f"scanner_v2 missing {name}")
        print(f"[scrub] label scanner_v2 {name} in memory")
        groups = wrap_scanner_episode(scan_root[name], field)
        extra = " (viser)" if eid == 0 else ""
        e = process_wrapped(name, groups, field, scan_sd, R_off, "scanner_v2", f"scanner_v2 ep {eid}{extra}")
        eps.append(e)
        rgb_of[e["key"]] = np.asarray(scan_root[name]["rgb_0"])

    for eid in args.fixb_episodes:
        print(f"[scrub] load fixb episode_{eid}")
        e, rgb = load_fixb_episode(args.fixb_root, eid, field, scan_sd, R_off)
        eps.append(e)
        rgb_of[e["key"]] = rgb

    rgb_dir = args.out / "scrub_rgb"
    if args.skip_rgb:
        rel = {e["key"]: "" for e in eps}
    else:
        rel = {}
        for e in eps:
            dest = rgb_dir / e["key"]
            print(f"[scrub] jpeg {e['key']} ({len(rgb_of[e['key']])} frames)")
            write_jpegs(rgb_of[e["key"]], dest)
            rel[e["key"]] = f"scrub_rgb/{e['key']}/"

    packed = {e["key"]: pack_episode(e, rel[e["key"]]) for e in eps}
    stats = {
        "scanner_v2": source_stats([e for e in eps if e["source"] == "scanner_v2"]),
        "fixb": source_stats([e for e in eps if e["source"] == "fixb"]),
    }

    V = field.vertices[~field.is_apron] * 1e3 if field.is_apron is not None else field.vertices * 1e3
    Vrb = xform_pts(V, R_rb, t_rb * 1e3)
    payload = {
        "H": H, "chunk_stride": QUERY_STRIDE, "near_mm": NEAR_M * 1e3, "off_edge_mm": OFF_EDGE_M * 1e3,
        "branch_margin_deg": BRANCH_MARGIN_DEG, "flags": list(SCRUB_FLAGS), "flag_text": SCRUB_FLAGS,
        "mesh": {"scan": display_mesh(field, False, 30000), "apron": display_mesh(field, True, 4000)},
        "R_rb": np.round(R_rb, 6).tolist(), "t_rb_mm": np.round(t_rb * 1e3, 2).tolist(),
        "cam": {"fixture": sitting_cam(V, "fixture"), "right_base": sitting_cam(Vrb, "right_base")},
        "order": [e["key"] for e in eps],
        "episodes": packed,
        "stats": stats,
        "banner": (
            "scanner_v2 (September, viser ep 0) = in-memory Fix B labels, not written. "
            "fixb (July) = trained r1/r2/r3 zarrs; same mesh, TCP 20–70 mm off it. "
            "Frame default: right_base, same as replay_wrist_cam.py. "
            "Rotate 90° about +Y: world labels move, v∥ does not."
        ),
        "blurb": (
            "Orange arrows = r1 world step p[k+1]−p[k]. Blue arrows = r2 v∥ Δt mapped through R(x_s). "
            "Purple = δn along n from x_s. Magenta/cyan = two contact strokes. "
            "r2 and r3 labels are identical; r3 only changes the obs frame."
        ),
    }
    html = HTML.replace("/*DATA*/", json.dumps(payload, separators=(",", ":"))).replace("/*PLOTLY*/", get_plotlyjs())
    path = args.out / "6_scrubber.html"
    path.write_text(html)
    print(json.dumps({
        "file": str(path), "mb": round(path.stat().st_size / 1e6, 1),
        "episodes": [e["label"] for e in eps],
        "stats": {k: {kk: vv for kk, vv in (v or {}).items() if kk != "scatter"} for k, v in stats.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
