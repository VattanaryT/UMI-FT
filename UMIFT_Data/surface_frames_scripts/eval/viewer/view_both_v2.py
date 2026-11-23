#!/usr/bin/env python3
"""One-screen layout of the combined open-loop page, on port 8803.

Does not edit view_both.py. Ports 8802 and 8766 are left running.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import view_both as both  # noqa: E402
import view_openloop as viewer  # noqa: E402
from http.server import ThreadingHTTPServer  # noqa: E402

both.PORT = 8803

_CSS = r"""
html, body { height:100%; overflow:hidden; }
body { display:flex; flex-direction:column; }
#topbar { flex:0 0 40px; flex-wrap:nowrap !important; overflow-x:auto; overflow-y:hidden;
  height:40px; box-sizing:border-box; gap:8px; padding:4px 8px; align-items:center; }
#topbar > * { flex:0 0 auto; }
#toggles { display:inline-flex; flex-wrap:nowrap; gap:6px; align-items:center; }
#topbar button, #topbar select, #topbar label { white-space:nowrap; }
#topbar select, #topbar button { padding:2px 6px; }
#folds { flex:0 0 auto; display:flex; gap:14px; padding:0 8px; position:relative; z-index:4; }
#folds summary { cursor:pointer; color:#9ad; font-size:12px; }
#folds details[open] { position:absolute; top:100%; left:0; background:#111; border:1px solid #333;
  max-height:46vh; overflow:auto; padding:4px; z-index:5; }
#stage { flex:1 1 auto; min-height:0; display:grid; grid-template-columns:60% 40%; }
#viewWrap { position:relative; min-height:0; height:100%; }
#view3d { height:100% !important; min-height:0; }
#camFloat { display:none; position:absolute; left:8px; top:8px; width:240px; z-index:6;
  background:rgba(17,17,17,0.94); border:1px solid #555; }
#camFloat.on { display:block; }
#camHandle { cursor:grab; font-size:12px; padding:2px 6px; color:#9ad; user-select:none; }
#camFloat #wrist { display:block; padding:0; }
#camFloat #wristImg { width:224px; height:auto; }
#camFloat #strip { max-width:224px; }
#camFloat #strip img { height:28px; }
#faceMode { background:#222; color:#ddd; border:1px solid #444; padding:1px 6px; }
#side { min-height:0; display:flex; flex-direction:column; }
#side .slot { flex:1 1 0; min-height:0; }
#pFace, #pScore { flex:0 0 28px; min-height:0; }
#pAct { flex:1.7 1 0; min-height:0; }
#legend { flex:0 0 auto; display:flex; gap:8px; align-items:center; padding:0 6px;
  font-size:11px; color:#bbb; white-space:nowrap; overflow:hidden; }
#legend i { display:inline-block; width:9px; height:9px; margin-right:3px; vertical-align:-1px; }
#timebar { flex:0 0 auto; display:flex; align-items:center; gap:8px; padding:2px 8px 4px; }
#timebar input[type=range] { flex:1 1 auto; width:auto; }
"""

_JS = r"""
function layoutV2() {
  const top = document.querySelector("body > .bar");
  top.id = "topbar";
  const timebar = document.createElement("div");
  timebar.id = "timebar";
  const time = document.getElementById("time");
  const label = time.previousElementSibling;
  if (label && label.textContent === "time") timebar.appendChild(label);
  timebar.appendChild(time);
  timebar.appendChild(document.getElementById("readout"));
  const variant = document.getElementById("variant");
  if (variant) {
    const tag = document.createElement("span");
    tag.textContent = "variant";
    document.getElementById("ep").after(tag, variant);
    const bar = document.getElementById("alignBar");
    [...bar.childNodes].forEach(n => {
      if (n.nodeType === 1 && n.tagName === "SPAN" && n.textContent === "variant") n.remove();
    });
  }
  function fold(id, title, ids) {
    const box = document.createElement("details");
    box.id = id;
    box.innerHTML = "<summary>" + title + "</summary>";
    ids.forEach(name => {
      const el = document.getElementById(name);
      if (el) box.appendChild(el);
    });
    return box;
  }
  const folds = document.createElement("div");
  folds.id = "folds";
  const camBtn = document.createElement("button");
  camBtn.id = "camBtn";
  camBtn.textContent = "camera";
  camBtn.onclick = () => document.getElementById("camFloat").classList.toggle("on");
  folds.appendChild(camBtn);
  folds.appendChild(fold("runsFold", "runs", ["toggles", "frame_toggles"]));
  folds.appendChild(fold("alignFold", "align", ["alignBar"]));
  folds.appendChild(fold("moreFold", "more", ["tilt","gapPlot","inRot","inFrame","inFoot","wallFoot","oodPlot","plots","headingBar","headingPlots","summary","events","ckptpath"]));
  const side = document.createElement("div");
  side.id = "side";
  side.innerHTML = '<div id="legend"><i style="background:#4aa3ff"></i>wall <i style="background:#e06c75"></i>cap <i style="background:#c792ea"></i>back <i style="background:#ffb020"></i>apron <i style="background:#fff"></i>TCP <i style="background:#ffe14a"></i>foot <i style="background:#7dd3fc"></i>frame z <span>solid label · dash pred</span> <button id="faceMode" type="button">foot closest</button> <i style="background:#3ddc97"></i>scored</div>'
    + '<div id="pDist" class="slot"></div><div id="pForce" class="slot"></div><div id="pFace"></div><div id="pScore"></div><div id="pRot" class="slot"></div><div id="pAct"></div>';
  const wrap = document.createElement("div");
  wrap.id = "viewWrap";
  const cam = document.createElement("div");
  cam.id = "camFloat";
  cam.innerHTML = '<div id="camHandle">camera</div>';
  cam.appendChild(document.getElementById("wrist"));
  wrap.append(document.getElementById("view3d"), cam);
  const stage = document.createElement("div");
  stage.id = "stage";
  stage.append(wrap, side);
  document.body.append(folds, stage, timebar);
  const handle = document.getElementById("camHandle");
  let drag = null;
  handle.addEventListener("pointerdown", (ev) => {
    drag = {x: ev.clientX, y: ev.clientY, l: cam.offsetLeft, t: cam.offsetTop};
    handle.setPointerCapture(ev.pointerId);
  });
  handle.addEventListener("pointermove", (ev) => {
    if (!drag) return;
    cam.style.left = (drag.l + ev.clientX - drag.x) + "px";
    cam.style.top = (drag.t + ev.clientY - drag.y) + "px";
  });
  handle.addEventListener("pointerup", () => { drag = null; });
  document.getElementById("faceMode").onclick = () => {
    faceMode = faceMode === "aligned" ? "closest" : "aligned";
    document.getElementById("faceMode").textContent = faceMode === "aligned" ? "foot aligned" : "foot closest";
    if (data) paintCursor();
  };
}
let faceMode = "closest";
layoutV2();
function v2Low() {
  const spans = [];
  const t = data.force_t || [], f = data.force || [];
  let a = null;
  for (let i = 0; i < t.length; i++) {
    if (!(f[i] >= 8)) { if (a === null) a = t[i]; }
    else if (a !== null) { spans.push([a, t[i]]); a = null; }
  }
  if (a !== null && t.length) spans.push([a, t[t.length - 1]]);
  return spans;
}
function v2Shapes(t, xref) {
  const ref = xref || "x";
  const shapes = [{type:"line", xref:ref, yref:"paper", x0:t, x1:t, y0:0, y1:1, line:{color:"#fff", width:1}}];
  v2Low().forEach(s => shapes.push({
    type:"rect", xref:ref, yref:"paper", x0:s[0], x1:s[1], y0:0, y1:1,
    fillcolor:"rgba(160,160,160,0.35)", line:{width:0}, layer:"below"
  }));
  return shapes;
}
function v2Lay(title, t0, t1, extra) {
  return Object.assign({
    paper_bgcolor:"#111", plot_bgcolor:"#111", font:{color:"#bbb", size:10},
    margin:{l:40, r:4, t:14, b:14}, showlegend:false, dragmode:false,
    title:{text:title, font:{size:11}, x:0, xanchor:"left", y:1, yanchor:"top"},
    xaxis:{range:[t0, t1], gridcolor:"#333", fixedrange:true},
    yaxis:{gridcolor:"#333", fixedrange:true},
    shapes:v2Shapes(window._v2t)
  }, extra || {});
}
function v2Time(gd, ev) {
  const xa = gd._fullLayout && gd._fullLayout.xaxis;
  if (!xa || !data || !data.path_t) return;
  const left = gd.getBoundingClientRect().left + (xa._offset || 0);
  let t = null;
  if (typeof xa.p2l === "function") t = xa.p2l(ev.clientX - left);
  if (typeof t !== "number" || !isFinite(t)) {
    const span = xa.range || [0, 1];
    const w = xa._length || 1;
    t = span[0] + ((ev.clientX - left) / w) * (span[1] - span[0]);
  }
  if (typeof t !== "number" || !isFinite(t)) return;
  fi = nearest(data.path_t, t);
  document.getElementById("time").value = fi;
  paintCursor();
}
function v2Bind(id) {
  const gd = document.getElementById(id);
  if (!gd || gd._v2) return;
  gd._v2 = true;
  gd.addEventListener("pointerdown", (ev) => v2Time(gd, ev));
}
function paintV2(t, run) {
  window._v2t = t;
  const times = data.path_t || [];
  const t0 = times.length ? times[0] : 0;
  const t1 = times.length ? times[times.length - 1] : 1;
  const opt = {displayModeBar:false, responsive:true};
  const dx = (data.gap_t && data.gap_t.length) ? data.gap_t : data.in_t;
  const dy = (data.gap_t && data.gap_t.length) ? data.gap_mm : data.in_foot;
  Plotly.react("pDist", (dx && dx.length) ? [{x:dx, y:dy, mode:"lines", line:{color:"#4aa3ff", width:1.4}}] : [], v2Lay("distance mm", t0, t1), opt);
  Plotly.react("pForce", [{x:data.force_t, y:data.force, mode:"lines", line:{color:"#e67e22", width:1.4}}], v2Lay("|F| N", t0, t1), opt);
  const face = (faceMode === "aligned" && data.foot_face_aligned && data.foot_face_aligned.length)
    ? data.foot_face_aligned : (data.foot_face || []);
  const nFace = Math.min(face.length, times.length);
  Plotly.react("pFace", nFace ? [{
    x: times.slice(0, nFace), y: Array(nFace).fill(0), mode:"markers",
    marker:{color: face.slice(0, nFace).map(f => ({wall:"#4aa3ff", cap:"#e06c75", back:"#c792ea", apron:"#ffb020"})[f] || "#888"), symbol:"square", size:7},
    text: face.slice(0, nFace), hovertemplate:"%{text} %{x}s<extra></extra>"
  }] : [], v2Lay("foot " + faceMode, t0, t1, {yaxis:{visible:false, range:[-1, 1], fixedrange:true}, margin:{l:40, r:4, t:12, b:0}}), opt);
  const rot = [];
  if (run && run.q_t) rot.push(line(run.q_t, run.ang, "ΔR", DIM.ang.pred, "x", "dash", run.q_break));
  if (data.in_t && data.in_ident) {
    rot.push({x:data.in_t, y:data.in_ident, mode:"lines", line:{color:"#ffb020", width:1.2}});
    rot.push({x:data.in_t, y:data.in_rzero, mode:"lines", line:{color:"#e06c75", width:1.2}});
  }
  const st = data.scored_t || [], sk = data.scored_keep || [];
  Plotly.react("pScore", st.length ? [{
    x: st, y: st.map(() => 0), mode: "markers",
    marker: {color: sk.map(k => k ? "#3ddc97" : "#444"), symbol: "square", size: 7},
    text: sk.map(k => k ? "kept" : "dropped"), hovertemplate: "%{text} %{x}s<extra></extra>"
  }] : [], v2Lay("scored", t0, t1, {yaxis:{visible:false, range:[-1, 1], fixedrange:true}, margin:{l:40, r:4, t:12, b:0}}), opt);
  Plotly.react("pRot", rot, v2Lay("rotation deg", t0, t1), opt);
  const act = [];
  if (run && run.q_t) {
    const br = run.q_break;
    act.push(line(run.q_t, run.v1_p, "v1", DIM.v1.pred, "x", "dash", br));
    act.push(line(run.q_t, run.v2_p, "v2", DIM.v2.pred, "x", "dash", br));
    act.push(line(run.q_t, run.v1_l, "v1l", DIM.v1.label, "x", null, br));
    act.push(line(run.q_t, run.v2_l, "v2l", DIM.v2.label, "x", null, br));
    act.push(line(run.q_t, run.n_p, "dn", DIM.n.pred, "x2", "dash", br));
    act.push(line(run.q_t, run.n_l, "dnl", DIM.n.label, "x2", null, br));
    act.push(line(run.q_t, run.ang, "dR", DIM.ang.pred, "x3", "dash", br));
    act.push(line(run.q_t, run.stiff_p, "k", DIM.stiff.pred, "x4", "dash", br));
    act.push(line(run.q_t, run.stiff_l, "kl", DIM.stiff.label, "x4", null, br));
  }
  const rel = run && run.role === "r1rel";
  const row = (y0, y1, title) => ({domain:[y0, y1], title:{text:title, font:{size:10}}, gridcolor:"#333", fixedrange:true});
  Plotly.react("pAct", act, v2Lay(rel ? "x y z rot stiff" : "v1 v2 · δn · ΔR · stiff", t0, t1, {
    xaxis:{range:[t0, t1], domain:[0, 1], gridcolor:"#333", fixedrange:true, anchor:"y"},
    yaxis: row(0.78, 1, rel ? "xy" : "v"),
    xaxis2:{range:[t0, t1], matches:"x", anchor:"y2"},
    yaxis2: row(0.52, 0.74, rel ? "z" : "δn"),
    xaxis3:{range:[t0, t1], matches:"x", anchor:"y3"},
    yaxis3: row(0.26, 0.48, "ΔR"),
    xaxis4:{range:[t0, t1], matches:"x", anchor:"y4"},
    yaxis4: row(0, 0.22, "stiff"),
    shapes:v2Shapes(t).concat(v2Shapes(t, "x2"), v2Shapes(t, "x3"), v2Shapes(t, "x4"))
  }), opt);
  ["pDist","pForce","pFace","pScore","pRot","pAct"].forEach(v2Bind);
}
"""


def _layout(page: str) -> str:
    needle = 'await Plotly.react("plots", plots, layout, {responsive: true});'
    scene = 'paper_bgcolor: "#111", margin: {l: 0, r: 0, t: 0, b: 0}, showlegend: true,'
    faces = 'traces.push({type: "scatter3d", mode: "markers", name: name,\n'
    frame = 'traces.push({type: "scatter3d", mode: "lines", name: "frame z",\n'
    align = 'if (bar) bar.style.display = mold === "thin" ? "flex" : "none";'
    for label, token in (
        ("plot hook", needle),
        ("scene legend", scene),
        ("foot markers", faces),
        ("frame z", frame),
        ("align bar", align),
        ("script start", "<script>\nconst VIEWS"),
        ("style end", "</style>"),
    ):
        if token not in page:
            raise SystemExit(f"v2 {label} missed")
    mesh = (
        'traces.unshift({\n'
        '    type: "mesh3d", x: mesh.x, y: mesh.y, z: mesh.z, i: mesh.i, j: mesh.j, k: mesh.k,\n'
        '    color: "#9aabb6", opacity: 0.45, name: "mold", hoverinfo: "skip", showlegend: false\n'
        '  });'
    )
    if mesh not in page:
        raise SystemExit("v2 mesh trace missed")
    page = page.replace(mesh, """traces.unshift(Object.assign({
    type: "mesh3d", x: mesh.x, y: mesh.y, z: mesh.z, i: mesh.i, j: mesh.j, k: mesh.k,
    name: "frame jump", showlegend: false, hovertemplate: "%{intensity:.1f}°<extra>frame jump</extra>"
  }, (mesh.jump && mesh.jump.length) ? {
    intensity: mesh.jump, intensitymode: "cell", colorscale: "Hot", cmin: 0, cmax: 60, showscale: true,
    colorbar: {title: "deg", thickness: 10, len: 0.45, x: 1}, opacity: 0.9
  } : {color: "#9aabb6", opacity: 0.45, hoverinfo: "skip"}));""", 1)
    page = page.replace(needle, "paintV2(t, run);", 1)
    page = page.replace('name: "TCP–foot", showlegend: false', 'name: "TCP–foot", showlegend: true', 1)
    page = page.replace('marker: {size: 8, color: "#ffffff"}', 'marker: {size: 16, color: "#ffffff"}', 1)
    page = page.replace('marker: {size: 8, color: "#ffe14a"}', 'marker: {size: 16, color: "#ffe14a"}', 1)
    page = page.replace('name: "TCP now", showlegend: false', 'name: "TCP now"', 1)
    page = page.replace('name: "foot now", showlegend: false', 'name: "foot now"', 1)
    page = page.replace(
        align,
        align + '\n  const alignFold = document.getElementById("alignFold");\n'
        '  if (alignFold) alignFold.style.display = mold === "thin" ? "" : "none";',
        1,
    )
    for label, token in (
        ("tcp dot", 'marker: {size: 16, color: "#ffffff"}'),
        ("foot dot", 'marker: {size: 16, color: "#ffe14a"}'),
        ("tcp-foot legend", 'name: "TCP–foot", showlegend: true'),
    ):
        if token not in page:
            raise SystemExit(f"v2 {label} missed")
    page = page.replace("</style>", _CSS + "\n</style>", 1)
    page = page.replace("<script>\nconst VIEWS", "<script>\n" + _JS + "\nconst VIEWS", 1)
    if "function layoutV2" not in page or "function paintV2" not in page:
        raise SystemExit("v2 layout missed")
    return page


_orig_init = ThreadingHTTPServer.__init__


_TOOLS = None
_DEMO = {}
_KIND = None


def _tools():
    global _TOOLS
    if _TOOLS is not None:
        return _TOOLS
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "thin_heldout"))
    import heldout
    from aligned_foot import AlignedFoot
    from fixb_field import FixBMoldField

    src = FixBMoldField(str(heldout.WALL))
    _TOOLS = (heldout, AlignedFoot(src, foot_mode="aligned"), src)
    return _TOOLS


def _aligned_on(out):
    if getattr(viewer.Handler, "mold", "training") != "thin":
        return None
    try:
        episode = int(out.get("episode", -1))
    except (TypeError, ValueError):
        return None
    if episode not in (0, 1) or not out.get("path") or not out.get("tool_R"):
        return None
    import numpy as np

    heldout, selector, src = _tools()
    global _KIND
    if _KIND is None:
        from aligned_foot import thin_kind
        _KIND = thin_kind(src, heldout.WALL)
    if episode not in _DEMO:
        name = "C_first" if episode == 0 else "A_full"
        _DEMO[episode] = heldout._demo(heldout.HERE / name / "replay.zarr", None)
    demo = _DEMO[episode]
    times = np.asarray(out["path_t"], dtype=np.float64) + float(demo["rgb_t"][0])
    at = heldout._nearest(demo["robot_t"], times)
    j = heldout._nearest(demo["wrench_t"], demo["robot_t"][at])
    n = min(len(out["path"]), len(out["tool_R"]), len(at))
    pts = np.asarray(out["path"], dtype=np.float64).reshape(-1, 3)[:n]
    rot = np.asarray(out["tool_R"], dtype=np.float64).reshape(-1, 3, 3)[:n]
    smooth = demo["smooth"][j][:n]
    force = np.einsum("nij,nj->ni", rot, smooth)
    mag = np.linalg.norm(smooth, axis=1)
    direction = np.where((mag >= 8.0)[:, None], -force, rot[:, :, 2])
    hit, _vid, _w, tri, _fb = selector.choose(pts, direction)
    face = _KIND[tri].astype(object).copy()
    behind = (face == "wall") & (np.sum((pts - hit) * selector._geo[tri], axis=1) < -0.002)
    face[behind] = "back"
    return {"foot_face_aligned": face.tolist()}


def _scored(episode):
    path = Path(__file__).resolve().parents[1] / "thin_heldout" / "contact_wall" / "kept.json"
    if not path.is_file():
        return None
    try:
        row = json.loads(path.read_text()).get(str(int(episode)))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not row:
        return None
    return {"scored_t": row.get("t") or [], "scored_keep": row.get("keep") or []}


def _arm():
    if getattr(viewer.episode_payload, "_v2faces", False):
        return
    original = viewer.episode_payload

    def episode_payload(*args, **kwargs):
        out = original(*args, **kwargs)
        try:
            extra = _aligned_on(out)
        except Exception as exc:
            out["foot_align_error"] = str(exc)
            return out
        if extra:
            out.update(extra)
        scored = _scored(out.get("episode"))
        if scored:
            out.update(scored)
        return out

    episode_payload._v2faces = True
    viewer.episode_payload = episode_payload


def _init(self, server_address, RequestHandlerClass, bind_and_activate=True):
    if "function layoutV2" not in viewer.PAGE:
        viewer.PAGE = _layout(viewer.PAGE)
    _arm()
    _orig_init(self, server_address, RequestHandlerClass, bind_and_activate)


ThreadingHTTPServer.__init__ = _init

import frame_jump

_orig_mesh = both._mesh


def _mesh_jump(path):
    payload = _orig_mesh(path)
    payload["jump"] = frame_jump.for_display(path, payload)
    return payload


both._mesh = _mesh_jump


if __name__ == "__main__":
    both.main()
