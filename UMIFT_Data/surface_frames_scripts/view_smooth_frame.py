#!/usr/bin/env python3
"""October viewer plus the smooth-frame r4vt zarr and the two velocity options.

view_oct.py is unchanged. This page adds smooth, track, and dof. It opens on smooth.

  python UMIFT_Data/surface_frames_scripts/view_smooth_frame.py

http://127.0.0.1:8767
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

import view_oct
from view_oct import (
    R4VT_EXCLUDED,
    Handler as OctHandler,
    attach_corrected_pose,
    attach_raw_odom,
    decimate_mesh,
    episode_ids,
    episode_payload,
    mesh_payload,
    recording_names,
    shade_spans,
)

# Gravity is +Y, so straight up is -Y. Tool z angle from vertical is the angle
# between the tool z axis and that up direction.
_UP = np.array([0.0, -1.0, 0.0])


def world_tool_series(pose7: np.ndarray) -> dict:
    """World tool x/z, z-from-vertical degrees, and unwrapped yaw of tool x."""
    R = view_oct.tool_rotation(np.asarray(pose7, dtype=np.float64))
    tool_x = R[:, :, 0]
    tool_z = R[:, :, 2]
    z_up = np.degrees(np.arccos(np.clip(tool_z @ _UP, -1.0, 1.0)))
    horizontal = tool_x - np.outer(tool_x @ _UP, _UP)
    yaw = np.degrees(np.unwrap(np.arctan2(horizontal[:, 2], horizontal[:, 0])))
    return {"tool_x": tool_x, "tool_z": tool_z, "z_up": z_up, "yaw": yaw}


def _geodesic_deg(R: np.ndarray) -> np.ndarray:
    trace = np.clip((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def attach_world_orientation(payload: dict, root, episode: int) -> None:
    """Tool triads on the path, plus z-from-vertical, yaw, and ΔR over time."""
    ep = root["data"][f"episode_{episode}"]
    pose = np.asarray(ep["ts_pose_fb_0"], dtype=np.float64)
    robot_t = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rgb_t = np.asarray(ep["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    n = min(len(pose), len(robot_t))
    series = world_tool_series(pose[:n])
    t0 = float(rgb_t[0])
    iser = view_oct.stride_idx(n, view_oct.SERIES_CAP)
    ip = view_oct.stride_idx(n, view_oct.PATH_CAP)
    payload["ori_t"] = view_oct.rounded(robot_t[iser] - t0, 3)
    payload["z_up"] = view_oct.rounded(series["z_up"][iser], 2)
    payload["yaw"] = [
        None if not np.isfinite(v) else round(float(v), 2) for v in series["yaw"][iser]
    ]
    if "fixb_delta_R_rot6d_0" in ep:
        from mold_aware_runtime import rot6d_rows_to_R

        r6 = np.asarray(ep["fixb_delta_R_rot6d_0"], dtype=np.float64).reshape(-1, 6)[:n]
        payload["dR"] = view_oct.rounded(_geodesic_deg(rot6d_rows_to_R(r6))[iser], 2)
    else:
        payload["dR"] = []
    every = max(1, len(ip) // 28)
    xyz = pose[:n, :3]
    triads = []
    for i in ip[::every]:
        triads.append(
            np.round(
                np.concatenate([xyz[i], series["tool_x"][i], series["tool_z"][i]]), 4
            ).tolist()
        )
    payload["triads"] = triads


def attach_foot_frames(payload: dict, root, episode: int) -> None:
    """Frame axes at the same samples as the foot polyline. Columns are t1, t2, n."""
    ep = root["data"][f"episode_{episode}"]
    if "mold_R_field_0" not in ep or "mold_x_s_0" not in ep:
        payload["foot_R"] = []
        return
    robot_t = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    pose = np.asarray(ep["ts_pose_fb_0"], dtype=np.float64)
    n = len(robot_t)
    count = min(n, len(pose), len(ep["mold_normal_0"]), len(ep["fixb_v_par_0"]), len(ep["mold_R_field_0"]))
    ip = view_oct.stride_idx(n, view_oct.PATH_CAP)
    ip = ip[ip < count]
    frames = np.asarray(ep["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)[ip]
    axes = np.concatenate([frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]], axis=1)
    payload["foot_R"] = np.round(axes, 4).tolist()


def _zarr(name: str):
    return view_oct.ODOM_ROOT / name / "acp_replay_buffer.zarr"


SMOOTH = _zarr("r4vt_smooth_frame")
TRACK = _zarr("r4vt_smooth_tcp_track")
DOF = _zarr("r4vt_smooth_dof")
CLEAN = _zarr("r4vt_smooth_tcp_track_clean")
EXTRA = ("smooth", "track", "dof", "trackclean")


def _page() -> str:
    page = view_oct.PAGE
    swaps = (
        (
            '<button id="r4vt">r4vt</button>',
            '<button id="r4vt">r4vt</button>\n  <button id="smooth">smooth</button>\n'
            '  <button id="track">track</button>\n  <button id="dof">dof</button>\n'
            '  <button id="trackclean">track clean</button>',
        ),
        (
            'const r4vtBtn = document.getElementById("r4vt");',
            'const r4vtBtn = document.getElementById("r4vt");\n'
            'const smoothBtn = document.getElementById("smooth");\n'
            'const trackBtn = document.getElementById("track");\n'
            'const dofBtn = document.getElementById("dof");\n'
            'const trackcleanBtn = document.getElementById("trackclean");',
        ),
        (
            "let counts = {r1rel: 50, r4: 50, r4vt: 50};",
            "let counts = {r1rel: 50, r4: 50, r4vt: 50, smooth: 50, track: 50, dof: 50, trackclean: 50};",
        ),
        ('let mode = "r4";', 'let mode = "smooth";'),
        (
            'flag.textContent = d.excluded ? "EXCLUDED from r4vt" : (d.r4vt ? "r4vt excludes 35 and 44" : "");',
            'flag.textContent = d.excluded ? "EXCLUDED from training" : '
            '(d.trackclean ? "cleaned TCP, glitches in red, excludes 35 and 44" : '
            '(d.dof ? "diffused field, excludes 35 and 44" : '
            '(d.track ? "tracked TCP velocity, excludes 35 and 44" : '
            '(d.smooth ? "smooth frame, excludes 35 and 44" : '
            '(d.r4vt ? "r4vt excludes 35 and 44" : "")))));',
        ),
        (
            'r4vtBtn.classList.toggle("on", mode === "r4vt");',
            'r4vtBtn.classList.toggle("on", mode === "r4vt");\n'
            '  smoothBtn.classList.toggle("on", mode === "smooth");\n'
            '  trackBtn.classList.toggle("on", mode === "track");\n'
            '  dofBtn.classList.toggle("on", mode === "dof");\n'
            '  trackcleanBtn.classList.toggle("on", mode === "trackclean");',
        ),
        (
            'r4vtBtn.onclick = () => setMode("r4vt");',
            'r4vtBtn.onclick = () => setMode("r4vt");\n'
            'smoothBtn.onclick = () => setMode("smooth");\n'
            'trackBtn.onclick = () => setMode("track");\n'
            'dofBtn.onclick = () => setMode("dof");\n'
            'trackcleanBtn.onclick = () => setMode("trackclean");',
        ),
        ("ep.max = counts.r4 - 1;", "ep.max = counts[mode] - 1;"),
        (
            "function shadeRect(n, t0, t1) {",
            "function glitchRect(n, t0, t1) {\n"
            "  return {\n"
            '    type: "rect", xref: "x" + n, yref: "y" + n + " domain",\n'
            "    x0: t0, x1: t1, y0: 0, y1: 1,\n"
            '    fillcolor: "rgba(231,76,60,0.28)", line: {width: 0}, layer: "below"\n'
            "  };\n"
            "}\n\n"
            "function shadeRect(n, t0, t1) {",
        ),
        (
            '  (d.shade || []).forEach(span => {\n'
            '    ["", "2", "3"].forEach(n => shapes.push(shadeRect(n, span[0], span[1])));\n'
            "  });",
            '  (d.shade || []).forEach(span => {\n'
            '    ["", "2", "3"].forEach(n => shapes.push(shadeRect(n, span[0], span[1])));\n'
            "  });\n"
            "  (d.glitch || []).forEach(span => {\n"
            '    ["", "2", "3", "4"].forEach(n => shapes.push(glitchRect(n, span[0], span[1])));\n'
            "  });",
        ),
        (
            '    ["", "2", "3"].forEach(n => shapes.push(shadeRect(n, span[0], span[1])));',
            '    ["", "2", "3", "4"].forEach(n => shapes.push(shadeRect(n, span[0], span[1])));',
        ),
        (
            'const shapes = [vline(""), vline("2"), vline("3")];',
            'const shapes = [vline(""), vline("2"), vline("3"), vline("4")];',
        ),
        (
            '["", "2", "3"].forEach((n, k) => {',
            '["", "2", "3", "4"].forEach((n, k) => {',
        ),
        (
            'yaxis: {title: top, domain: [0.70, 1], anchor: "x", automargin: false},',
            'yaxis: {title: top, domain: [0.78, 1], anchor: "x", automargin: false},',
        ),
        (
            'yaxis2: {title: "N", domain: [0.36, 0.64], anchor: "x2", automargin: false},',
            'yaxis2: {title: "N", domain: [0.52, 0.74], anchor: "x2", automargin: false},',
        ),
        (
            'yaxis3: {title: bot, domain: [0, 0.30], anchor: "x3", automargin: false}',
            'yaxis3: {title: bot, domain: [0.26, 0.48], anchor: "x3", automargin: false},\n'
            '    xaxis4: Object.assign({anchor: "y4", matches: "x", title: "s"}, x),\n'
            '    yaxis4: {title: "tool deg", domain: [0, 0.22], anchor: "x4", automargin: false}',
        ),
        ("#view3d, #plots { height:48vh; }", "#view3d { height:40vh; } #plots { height:56vh; }"),
        (
            "function plotTraces(d) {",
            "function oriTraces(d) {\n"
            "  return [\n"
            '    line(d.ori_t || [], d.z_up || [], "z from up", "#e74c3c", "x4"),\n'
            '    line(d.ori_t || [], d.yaw || [], "yaw", "#5dade2", "x4"),\n'
            '    line(d.ori_t || [], d.dR || [], "ΔR", "#f4d03f", "x4")\n'
            "  ];\n"
            "}\n\n"
            "function plotTraces(d) {",
        ),
        (
            '      line(d.sig_t, d.dang, "dR deg", "#f4d03f", "x3")\n'
            "    ];",
            '      line(d.sig_t, d.dang, "dR deg", "#f4d03f", "x3")\n'
            "    ].concat(oriTraces(d));",
        ),
        (
            '    hideUnlessVt(line(d.sig_t, d.vt_n || [], "vt·n", "#f5b041", "x3"))\n'
            "  ];",
            '    hideUnlessVt(line(d.sig_t, d.vt_n || [], "vt·n", "#f5b041", "x3"))\n'
            "  ].concat(oriTraces(d));",
        ),
        (
            "function sceneTraces(d) {",
            "function glitchTrace(d) {\n"
            "  const pts = d.glitch_xyz || [];\n"
            "  return {\n"
            '    type: "scatter3d",\n'
            "    x: pts.map(p => p[0]), y: pts.map(p => p[1]), z: pts.map(p => p[2]),\n"
            '    mode: "markers", name: "glitch",\n'
            '    marker: {size: 4, color: "#e74c3c", symbol: "diamond"}\n'
            "  };\n"
            "}\n\n"
            "function triadTraces(d) {\n"
            "  const segs = d.triads || [];\n"
            "  const xz = {x: [], y: [], z: []};\n"
            "  const zz = {x: [], y: [], z: []};\n"
            "  const L = 0.016;\n"
            "  segs.forEach(s => {\n"
            "    xz.x.push(s[0], s[0] + L * s[3], null);\n"
            "    xz.y.push(s[1], s[1] + L * s[4], null);\n"
            "    xz.z.push(s[2], s[2] + L * s[5], null);\n"
            "    zz.x.push(s[0], s[0] + L * s[6], null);\n"
            "    zz.y.push(s[1], s[1] + L * s[7], null);\n"
            "    zz.z.push(s[2], s[2] + L * s[8], null);\n"
            "  });\n"
            "  function arm(c, color) {\n"
            "    return {\n"
            '      type: "scatter3d", x: c.x, y: c.y, z: c.z, mode: "lines",\n'
            '      name: "tool axis", showlegend: false, line: {color, width: 3}\n'
            "    };\n"
            "  }\n"
            '  return [arm(xz, "#e74c3c"), arm(zz, "#5dade2")];\n'
            "}\n\n"
            "function sceneTraces(d) {",
        ),
        (
            "  return [mold, path, foot, cursor, axis, vt, link].concat(odom);",
            "  return [mold, path, foot, cursor, axis, vt, link].concat(odom).concat([glitchTrace(d)]).concat(triadTraces(d));",
        ),
        (
            "  if (!d.r4) return [path, cursor, vt, link].concat(odom);",
            "  if (!d.r4) return [path, cursor, vt, link].concat(odom).concat(triadTraces(d));",
        ),
        (
            '<button id="corr">t265 fixed</button>',
            '<button id="corr">t265 fixed</button>\n  <button id="foot">foot frame</button>',
        ),
        ("let showCorr = false;", "let showCorr = false;\nlet showFoot = false;"),
        (
            "function sceneTraces(d) {",
            "function footFrameTraces(d) {\n"
            "  const path = d.path || [];\n"
            "  const foot = d.foot || [];\n"
            "  const frames = d.foot_R || [];\n"
            "  const n = Math.min(path.length, foot.length);\n"
            "  const link = {x: [], y: [], z: []};\n"
            "  const pts = {x: [], y: [], z: []};\n"
            "  const arms = [0, 1, 2].map(() => ({x: [], y: [], z: []}));\n"
            "  const L = 0.025;\n"
            "  const every = Math.max(1, Math.floor(n / 80));\n"
            "  for (let i = 0; i < n; i++) {\n"
            "    const a = path[i], b = foot[i];\n"
            "    link.x.push(a[0], b[0], null);\n"
            "    link.y.push(a[1], b[1], null);\n"
            "    link.z.push(a[2], b[2], null);\n"
            "    if (i % every) continue;\n"
            "    pts.x.push(b[0]); pts.y.push(b[1]); pts.z.push(b[2]);\n"
            "    const R = frames[i];\n"
            "    if (!R || R.length < 9) continue;\n"
            "    for (let c = 0; c < 3; c++) {\n"
            "      const ax = R.slice(3 * c, 3 * c + 3);\n"
            "      arms[c].x.push(b[0], b[0] + L * ax[0], null);\n"
            "      arms[c].y.push(b[1], b[1] + L * ax[1], null);\n"
            "      arms[c].z.push(b[2], b[2] + L * ax[2], null);\n"
            "    }\n"
            "  }\n"
            "  function seg(c, name, color, legend) {\n"
            "    return {\n"
            '      type: "scatter3d", x: c.x, y: c.y, z: c.z, mode: "lines",\n'
            "      name, showlegend: legend, visible: showFoot, legendgroup: \"footframe\",\n"
            "      line: {color, width: 4}\n"
            "    };\n"
            "  }\n"
            "  return [\n"
            "    {\n"
            '      type: "scatter3d", x: link.x, y: link.y, z: link.z, mode: "lines",\n'
            '      name: "tcp to foot", showlegend: true, visible: showFoot, legendgroup: "footframe",\n'
            '      line: {color: "#f4d03f", width: 2}\n'
            "    },\n"
            "    {\n"
            '      type: "scatter3d", x: pts.x, y: pts.y, z: pts.z, mode: "markers",\n'
            '      name: "foot point", showlegend: false, visible: showFoot, legendgroup: "footframe",\n'
            '      marker: {size: 3, color: "#f4d03f"}\n'
            "    },\n"
            '    seg(arms[0], "frame t1", "#2ecc71", false),\n'
            '    seg(arms[1], "frame t2", "#5dade2", false),\n'
            '    seg(arms[2], "frame n", "#f5b041", true)\n'
            "  ];\n"
            "}\n\n"
            "function sceneTraces(d) {",
        ),
        (
            "  return [mold, path, foot, cursor, axis, vt, link].concat(odom).concat([glitchTrace(d)]).concat(triadTraces(d));",
            "  return [mold, path, foot, cursor, axis, vt, link].concat(odom).concat([glitchTrace(d)]).concat(triadTraces(d)).concat(footFrameTraces(d));",
        ),
        (
            "  if (!d.r4) return [path, cursor, vt, link].concat(odom).concat(triadTraces(d));",
            "  if (!d.r4) return [path, cursor, vt, link].concat(odom).concat(triadTraces(d)).concat(footFrameTraces(d));",
        ),
        (
            "function noteText(d) {",
            "document.getElementById(\"foot\").onclick = () => {\n"
            "  showFoot = !showFoot;\n"
            "  document.getElementById(\"foot\").classList.toggle(\"on\", showFoot);\n"
            "  if (!data) return;\n"
            "  const gd = document.getElementById(\"view3d\");\n"
            "  const idx = [];\n"
            "  (gd.data || []).forEach((tr, i) => { if (tr.legendgroup === \"footframe\") idx.push(i); });\n"
            "  const cam = liveCamera();\n"
            "  if (idx.length) Plotly.restyle(\"view3d\", {visible: showFoot}, idx).then(() => applyCamera(cam));\n"
            "};\n"
            "function noteText(d) {",
        ),
    )
    for old, new in swaps:
        if old not in page:
            raise RuntimeError(f"view_oct page has no {old!r}")
        page = page.replace(old, new, 1)
    return page


PAGE = _page()


class Handler(OctHandler):
    roots = {}
    ids = {}
    glitches = {}

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            return
        if path == "/api/meta":
            counts = {
                "r1rel": len(self.ids_orig),
                "r4": len(self.ids_r4),
                "r4vt": len(self.ids_r4vt),
            }
            counts.update({name: len(self.ids[name]) for name in EXTRA})
            body = json.dumps({"counts": counts, "mesh": self.mesh}).encode()
            self._send(200, body, "application/json")
            return
        args = self._args()
        mode = args.get("mode")
        if path == "/api/episode" and mode in EXTRA:
            self._extra_episode(args, mode)
            return
        if path == "/api/episode":
            self._base_episode(args)
            return
        super().do_GET()

    def _extra_episode(self, args: dict, mode: str) -> None:
        try:
            episode = int(args.get("ep", "0"))
        except ValueError:
            self._send(400, b'{"error":"bad episode"}', "application/json")
            return
        if episode not in self.ids[mode] or episode >= len(self.names):
            self._send(404, b'{"error":"missing episode"}', "application/json")
            return
        root = self.roots[mode]
        vt_xyz = view_oct.virtual_target_xyz(self.root_r1rel["data"][f"episode_{episode}"])
        payload = episode_payload(root, episode, True, self._frame_times(episode), vt_xyz)
        t0 = float(np.asarray(root["data"][f"episode_{episode}"]["rgb_time_stamps_0"]).reshape(-1)[0])
        payload["shade"] = shade_spans(self.bad_spans, episode, t0)
        payload["r4vt"] = False
        payload["smooth"] = mode == "smooth"
        payload["track"] = mode == "track"
        payload["dof"] = mode == "dof"
        payload["trackclean"] = mode == "trackclean"
        payload["excluded"] = episode in R4VT_EXCLUDED
        if mode == "trackclean":
            spans = self.glitches.get(str(episode), [])
            payload["glitch"] = [[s["t0"], s["t1"]] for s in spans]
            raw = np.asarray(self.roots["track"]["data"][f"episode_{episode}"]["ts_pose_fb_0"])
            xyz = []
            for span in spans:
                sl = raw[int(span["i0"]) : int(span["i1"]) + 1, :3]
                if len(sl) > 30:
                    sl = sl[:: max(1, len(sl) // 30)]
                xyz.extend(np.round(sl, 5).tolist())
            payload["glitch_xyz"] = xyz
        attach_raw_odom(payload, self.names[episode])
        attach_corrected_pose(payload, self.names[episode])
        attach_world_orientation(payload, root, episode)
        attach_foot_frames(payload, root, episode)
        self._send(200, json.dumps(payload).encode(), "application/json")

    def _base_episode(self, args: dict) -> None:
        mode = args.get("mode", "r4")
        r4 = mode != "r1rel"
        if mode == "r4vt":
            ids, root, vt_src = self.ids_r4vt, self.root_r4vt, self.root_r1rel
        elif r4:
            ids, root, vt_src = self.ids_r4, self.root_r4, self.root_orig
        else:
            ids, root, vt_src = self.ids_orig, self.root_orig, None
        try:
            episode = int(args.get("ep", "0"))
        except ValueError:
            self._send(400, b'{"error":"bad episode"}', "application/json")
            return
        if episode not in ids or episode >= len(self.names):
            self._send(404, b'{"error":"missing episode"}', "application/json")
            return
        vt_xyz = None if vt_src is None else view_oct.virtual_target_xyz(
            vt_src["data"][f"episode_{episode}"]
        )
        payload = episode_payload(root, episode, r4, self._frame_times(episode), vt_xyz)
        t0 = float(np.asarray(root["data"][f"episode_{episode}"]["rgb_time_stamps_0"]).reshape(-1)[0])
        payload["shade"] = shade_spans(self.bad_spans, episode, t0)
        payload["r4vt"] = mode == "r4vt"
        payload["excluded"] = mode == "r4vt" and episode in R4VT_EXCLUDED
        attach_raw_odom(payload, self.names[episode])
        attach_corrected_pose(payload, self.names[episode])
        attach_world_orientation(payload, root, episode)
        attach_foot_frames(payload, root, episode)
        self._send(200, json.dumps(payload).encode(), "application/json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--bags-dir", type=Path, default=view_oct.BAG_DIR)
    args = parser.parse_args()
    Handler.bag_dir = args.bags_dir
    field = np.load(view_oct.FIELD)
    vertices, faces = decimate_mesh(field["vertices"], field["faces"], field["is_apron"])
    Handler.mesh = mesh_payload(vertices, faces)
    Handler.root_orig = zarr.open(view_oct.ORIG, "r")
    Handler.root_r4 = zarr.open(view_oct.R4, "r")
    Handler.root_r1rel = zarr.open(str(view_oct.R1REL), "r")
    Handler.root_r4vt = zarr.open(str(view_oct.R4VT), "r")
    Handler.roots = {
        "smooth": zarr.open(str(SMOOTH), "r"),
        "track": zarr.open(str(TRACK), "r"),
        "dof": zarr.open(str(DOF), "r"),
        "trackclean": zarr.open(str(CLEAN), "r"),
    }
    glitch_path = CLEAN.parent / "glitches.json"
    Handler.glitches = json.loads(glitch_path.read_text()).get("episodes", {}) if glitch_path.exists() else {}
    Handler.bad_spans = json.loads(view_oct.LABEL_STATS.read_text()).get("bad_spans", {})
    Handler.ids_orig = episode_ids(Handler.root_orig)
    Handler.ids_r4 = episode_ids(Handler.root_r4)
    Handler.ids_r4vt = episode_ids(Handler.root_r4vt)
    Handler.ids = {name: episode_ids(root) for name, root in Handler.roots.items()}
    Handler.names = recording_names()
    print(
        "[view_smooth] "
        + ", ".join(f"{name} {len(ids)}" for name, ids in Handler.ids.items())
        + f", mesh {len(faces)} tris",
        flush=True,
    )
    server = view_oct.ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[view_smooth] http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
