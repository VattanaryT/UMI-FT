"""Plotly viewer for one closed-loop rollout. Port 8810.

MuJoCo is not installed in the pyrite env, so the video path records that
and does not invent a file.
"""

from __future__ import annotations

import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

PORT = 8810
_SERVER = None


def publish(out_dir, tag, field, sim_xyz, demo_xyz, normals, force) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html = out_dir / f"{tag}.html"
    html.write_text(_html(field, sim_xyz, demo_xyz, normals, force))
    _serve(out_dir)
    return {
        "html": str(html),
        "video": None,
        "video_note": "mujoco is not installed in the pyrite env",
        "port": PORT,
        "url": f"http://127.0.0.1:{PORT}/{html.name}",
    }


def _serve(directory: Path) -> None:
    global _SERVER
    if _SERVER is not None:
        return

    class _Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

    try:
        _SERVER = ThreadingHTTPServer(("127.0.0.1", PORT), _Handler)
    except OSError:
        _SERVER = False
        return
    threading.Thread(target=_SERVER.serve_forever, daemon=True).start()


def _html(field, sim_xyz, demo_xyz, normals, force) -> str:
    sim = np.asarray(sim_xyz, dtype=np.float64).reshape(-1, 3)
    demo = np.asarray(demo_xyz, dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    force = np.linalg.norm(np.asarray(force, dtype=np.float64).reshape(-1, 3), axis=1)
    step = max(1, len(sim) // 40)
    arrow = sim[::step]
    direction = normals[::step] * 0.02
    faces = np.asarray(field.faces)
    if len(faces) > 8000:
        faces = faces[:: max(1, len(faces) // 8000)]
    payload = {
        "x": field.vertices[:, 0].tolist(),
        "y": field.vertices[:, 1].tolist(),
        "z": field.vertices[:, 2].tolist(),
        "i": faces[:, 0].tolist(),
        "j": faces[:, 1].tolist(),
        "k": faces[:, 2].tolist(),
        "sx": sim[:, 0].tolist(),
        "sy": sim[:, 1].tolist(),
        "sz": sim[:, 2].tolist(),
        "f": force.tolist(),
        "dx": demo[:, 0].tolist(),
        "dy": demo[:, 1].tolist(),
        "dz": demo[:, 2].tolist(),
        "ax": arrow[:, 0].tolist(),
        "ay": arrow[:, 1].tolist(),
        "az": arrow[:, 2].tolist(),
        "ux": direction[:, 0].tolist(),
        "uy": direction[:, 1].tolist(),
        "uz": direction[:, 2].tolist(),
    }
    return _PAGE.replace("__DATA__", json.dumps(payload))


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
<body><div id="c" style="height:100vh"></div><script>
const d = __DATA__;
Plotly.newPlot("c", [
  {type:"mesh3d", x:d.x, y:d.y, z:d.z, i:d.i, j:d.j, k:d.k, opacity:0.35, name:"mold"},
  {type:"scatter3d", mode:"lines", x:d.dx, y:d.dy, z:d.dz, name:"demo", line:{color:"#444", width:4}},
  {type:"scatter3d", mode:"lines+markers", x:d.sx, y:d.sy, z:d.sz, name:"sim",
   marker:{size:3, color:d.f, colorscale:"Viridis", colorbar:{title:"|F| N"}}, line:{color:"#888", width:2}},
  {type:"cone", x:d.ax, y:d.ay, z:d.az, u:d.ux, v:d.uy, w:d.uz, sizemode:"absolute", sizeref:0.02,
   showscale:false, name:"normal", anchor:"tail"}
], {scene:{aspectmode:"data"}});
</script></body></html>
"""
