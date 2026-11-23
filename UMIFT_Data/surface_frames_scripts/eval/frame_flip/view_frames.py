#!/usr/bin/env python3
"""Side-by-side field axes for the fixb mold and the thin stringer.

Serves one Plotly page on 127.0.0.1. Does not import or edit the other viewers.

    python UMIFT_Data/surface_frames_scripts/eval/frame_flip/view_frames.py
"""

from __future__ import annotations

import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import zarr
from plotly.subplots import make_subplots

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS), str(ROOT / "PyriteEnvSuites/scripts")]

from fixb_field import FixBMoldField
from fixb_labels import pose7_to_Rp
from heading_tilt import e_long_sequence, long_axis
from view_oct import decimate_mesh

FIXB_FIELD = (
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
THIN_FIELD = "/mnt/data/umidata/ft/held_out_mold/thin_stringer/field/fixb_field.npz"
FIXB_ZARR = (
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_causal_foot/acp_replay_buffer.zarr"
)
THIN_ZARR = (
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/"
    "remap_to_training/acp_replay_buffer.zarr"
)
PORT = 8799
ARROW = 0.03


def _self_check() -> None:
    if ARROW <= 0 or PORT < 1024:
        raise RuntimeError("viewer settings")


def arrows(fig, col, origin, direction, color, name, show):
    tip = origin + direction * ARROW
    for a, b in zip(origin, tip):
        fig.add_trace(
            go.Scatter3d(
                x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]],
                mode="lines",
                line=dict(color=color, width=6),
                name=name,
                showlegend=show,
                legendgroup=name,
            ),
            row=1,
            col=1 if col == "scene" else 2,
        )
        show = False


def panel(field_path, zarr_path, episode, title):
    blob = np.load(field_path)
    verts, faces = decimate_mesh(blob["vertices"], blob["faces"], blob["is_apron"])
    field = FixBMoldField(field_path)
    axis = long_axis(field.vertices)
    group = zarr.open(zarr_path, "r")["data"][episode]
    pose = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    tool_R, tool_p = pose7_to_Rp(pose[::8])
    geom = field.query(tool_p)
    foot = np.asarray(geom["x_s"], dtype=np.float64).reshape(-1, 3)
    normal = np.asarray(geom["R"], dtype=np.float64).reshape(-1, 3, 3)[:, :, 2]
    dist = np.linalg.norm(tool_p - foot, axis=1)
    near = np.flatnonzero(dist < 0.04)
    if len(near) < 5:
        near = np.argsort(dist)[:20]
    take = near[np.linspace(0, len(near) - 1, min(20, len(near))).astype(int)]
    e_long, _ = e_long_sequence(normal[take], axis)
    mesh = go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="#c5d0d6", opacity=0.55, name=title, showlegend=False,
    )
    path = go.Scatter3d(
        x=tool_p[:, 0], y=tool_p[:, 1], z=tool_p[:, 2],
        mode="lines", line=dict(color="#222", width=3), name="demo path",
    )
    return mesh, path, foot[take], e_long, normal[take], tool_R[take, :, 2]


def page() -> bytes:
    _self_check()
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("fixb training mold", "thin stringer"),
    )
    packs = (
        panel(FIXB_FIELD, FIXB_ZARR, "episode_9", "fixb"),
        panel(THIN_FIELD, THIN_ZARR, "episode_0", "thin"),
    )
    for col, (mesh, path, origin, e_long, normal, tool_z) in zip(("scene", "scene2"), packs):
        fig.add_trace(mesh, row=1, col=1 if col == "scene" else 2)
        fig.add_trace(path, row=1, col=1 if col == "scene" else 2)
        arrows(fig, col, origin, e_long, "#d4532b", "e_long", True)
        arrows(fig, col, origin, normal, "#1d4e89", "n_out", True)
        arrows(fig, col, origin, tool_z, "#2a9d8f", "tool z", True)
    fig.update_layout(
        title="e_long red, n_out blue, tool z green",
        height=720,
        legend=dict(orientation="h"),
        scene=dict(aspectmode="data", yaxis_title="y down"),
        scene2=dict(aspectmode="data", yaxis_title="y down"),
    )
    return fig.to_html(include_plotlyjs=True, full_html=True).encode()


class Handler(BaseHTTPRequestHandler):
    body = b""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, fmt, *args):
        print(f"[frame-flip] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    Handler.body = page()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"http://127.0.0.1:{PORT}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
