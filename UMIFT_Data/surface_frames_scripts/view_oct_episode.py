#!/usr/bin/env python3
"""Wrist-video viewer for an October stringer_v3.5 demo.

Same ingredients as plot_inference_episode.py: a time slider, the wrist
image, tool xyz, and a 3D path. Handheld bags have no center/Orbbec
recording, so that panel says so. Tag-visible frames come from
n_tags_visible in the ee_tag_traj sidecar; the rest are T265 fills.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw
from plotly.subplots import make_subplots

ZARR = (
    "/mnt/data/umidata/ft/mold_aware_datasets/"
    "stringer_v3.5_fixb/r1_world_world/acp_replay_buffer.zarr"
)
TAG_DIR = Path("/mnt/data/umidata/ft/datasets/stringer_v3.5/tag_traj")
BAG_DIR = Path("/mnt/data/umidata/ft/datasets/stringer_v3.5/bags")
START_PAD_S = 5.0
OUT_DIR = Path("/tmp")


def _recording_name(episode: int) -> str:
    files = sorted(TAG_DIR.glob("*_ee_master.npz"))
    return files[episode].name.replace("_ee_master.npz", "")


def _jpeg_uri(rgb: np.ndarray, size=(320, 240)) -> str:
    image = Image.fromarray(rgb).resize(size, Image.BILINEAR)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=38)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _placeholder(text: str) -> str:
    image = Image.new("RGB", (320, 240), (28, 28, 32))
    draw = ImageDraw.Draw(image)
    draw.multiline_text((16, 90), text, fill=(220, 220, 220), spacing=6)
    return _jpeg_uri(np.asarray(image), size=(320, 240))


def _wrist_frames(name: str, step: int) -> tuple[np.ndarray, list[str]]:
    manifest = json.loads((BAG_DIR / f"{name}_camera" / "manifest.json").read_text())
    frames = manifest["streams"]["color"]["frames"]
    raw = np.memmap(
        BAG_DIR / f"{name}_camera" / "color.raw", dtype=np.uint8, mode="r"
    )
    times = []
    uris = []
    for frame in frames[::step]:
        blob = np.asarray(raw[frame["offset"] : frame["offset"] + frame["length"]])
        rgb = blob.reshape(480, 640, 3)
        uris.append(_jpeg_uri(rgb))
        times.append(float(frame["timestamp_sec"]))
    t0 = float(frames[0]["timestamp_sec"])
    rel = np.asarray(times) - t0 + START_PAD_S
    return rel, uris


def _step24_window(rgb_t: np.ndarray, xyz_rgb: np.ndarray) -> tuple[float, float]:
    span = 23
    delta = np.linalg.norm(xyz_rgb[span:] - xyz_rgb[:-span], axis=1)
    i = int(np.argmax(delta))
    return float(rgb_t[i] - 1.0), float(rgb_t[i + span] + 1.0)


def build(episode: int, highlight: tuple[float, float] | None) -> Path:
    import zarr

    name = _recording_name(episode)
    root = zarr.open(ZARR, "r")
    ep = root["data"][f"episode_{episode}"]
    robot_t = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    pose = np.asarray(ep["ts_pose_fb_0"], dtype=np.float64)
    xyz = pose[:, :3]
    dt = np.maximum(np.diff(robot_t), 0.004)
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / dt
    speed_t = robot_t[1:]
    dist = np.asarray(ep["mold_signed_distance_0"], dtype=np.float64).reshape(-1)
    wrench = np.asarray(ep["wrench_left_0"], dtype=np.float64)
    wrench_t = np.asarray(ep["wrench_time_stamps_0"], dtype=np.float64).reshape(-1)
    force = np.linalg.norm(wrench[:, :3], axis=1)
    rgb_t = np.asarray(ep["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    xyz_rgb = np.vstack(
        [np.interp(rgb_t, robot_t, xyz[:, k]) for k in range(3)]
    ).T
    if highlight is None:
        highlight = _step24_window(rgb_t, xyz_rgb)

    tag = np.load(TAG_DIR / f"{name}_ee_master.npz")
    tag_t = np.asarray(tag["timestamp_sec_unified"], dtype=np.float64) + START_PAD_S
    tag_on = np.asarray(tag["n_tags_visible"]) > 0

    print(f"[view] episode {episode} {name} encoding wrist", flush=True)
    frame_t, frame_uri = _wrist_frames(name, step=3)
    center_uri = _placeholder("No center camera.\nHandheld bag is the wrist D405 only.")

    # Downsample the path so the HTML stays movable.
    stride = max(1, len(robot_t) // 800)
    path = xyz[::stride]
    path_speed = np.concatenate([[speed[0]], speed])[::stride]

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "scene"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "tool xyz (m)",
            "path colored by speed",
            "speed (m/s)",
            "signed distance (m)",
            "wrist force norm (N)",
            "tag visible (1) vs T265 fill (0)",
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )
    for k, color, label in (
        (0, "#e74c3c", "x"),
        (1, "#2ecc71", "y"),
        (2, "#3498db", "z"),
    ):
        fig.add_trace(
            go.Scatter(x=robot_t, y=xyz[:, k], name=label, line=dict(color=color, width=1.5)),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter3d(
            x=path[:, 0],
            y=path[:, 1],
            z=path[:, 2],
            mode="markers",
            name="speed",
            marker=dict(
                size=3,
                color=path_speed,
                colorscale="Turbo",
                colorbar=dict(title="m/s", len=0.4, y=0.82),
            ),
        ),
        row=1,
        col=2,
    )
    tag_xyz = np.vstack(
        [np.interp(tag_t, robot_t, xyz[:, k]) for k in range(3)]
    ).T
    fig.add_trace(
        go.Scatter3d(
            x=tag_xyz[tag_on, 0],
            y=tag_xyz[tag_on, 1],
            z=tag_xyz[tag_on, 2],
            mode="markers",
            name="tag visible",
            marker=dict(size=4, color="#f4f4f4", symbol="diamond"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=speed_t, y=speed, name="speed", line=dict(color="#f1c40f", width=1.2)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=robot_t, y=dist, name="signed distance", line=dict(color="#9b59b6", width=1.2)),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=wrench_t, y=force, name="force", line=dict(color="#e67e22", width=1.2)),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=tag_t[tag_on],
            y=np.ones(int(tag_on.sum())),
            name="tag",
            mode="markers",
            marker=dict(size=7, color="#2ecc71", symbol="line-ns"),
        ),
        row=3,
        col=2,
    )
    fill_t = tag_t[~tag_on][::4]
    fig.add_trace(
        go.Scatter(
            x=fill_t,
            y=np.zeros(len(fill_t)),
            name="T265 fill",
            mode="markers",
            marker=dict(size=4, color="#7f8c8d"),
        ),
        row=3,
        col=2,
    )
    lo, hi = highlight
    # xy axes skip the 3D scene: xyz, speed, distance, force, tag visibility.
    shapes = []
    for n in ("", "2", "3", "4", "5"):
        shapes.append(
            dict(
                type="rect",
                xref="x" + n,
                yref="y" + n + " domain",
                x0=lo,
                x1=hi,
                y0=0,
                y1=1,
                fillcolor="rgba(231, 76, 60, 0.18)",
                line=dict(width=0),
                layer="below",
            )
        )
        shapes.append(
            dict(
                type="line",
                xref="x" + n,
                yref="y" + n + " domain",
                x0=float(frame_t[0]),
                x1=float(frame_t[0]),
                y0=0,
                y1=1,
                line=dict(color="#ffffff", width=1),
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=860,
        title=f"stringer_v3.5 episode {episode} ({name})  highlight {lo:.2f}–{hi:.2f} s",
        legend=dict(orientation="h", y=1.08),
        shapes=shapes,
        margin=dict(l=40, r=20, t=80, b=40),
    )
    fig.update_scenes(aspectmode="data", camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)))
    plot = fig.to_html(full_html=False, include_plotlyjs="cdn")
    frames_json = json.dumps({"t": frame_t.tolist(), "src": frame_uri})
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>episode {episode}</title>
<style>
body {{ margin:0; background:#111; color:#ddd; font-family:sans-serif; }}
.cams {{ display:flex; gap:12px; padding:12px 12px 0; }}
.cams figure {{ margin:0; }}
.cams img {{ width:480px; height:360px; object-fit:contain; background:#000; }}
.cams figcaption {{ font-size:13px; margin-bottom:4px; }}
input[type=range] {{ width:100%; }}
.bar {{ padding:8px 16px 0; }}
</style></head><body>
<div class="cams">
<figure><figcaption>wrist D405</figcaption><img id="wrist"></figure>
<figure><figcaption>center</figcaption><img id="center" src="{center_uri}"></figure>
</div>
<div class="bar">
<input id="slider" type="range" min="0" max="{len(frame_uri) - 1}" value="0" step="1">
<div id="readout"></div>
</div>
{plot}
<script>
const FRAMES = {frames_json};
const slider = document.getElementById("slider");
const wrist = document.getElementById("wrist");
const gd = document.querySelector(".plotly-graph-div");
function show(i) {{
  i = Math.max(0, Math.min(FRAMES.t.length - 1, i));
  wrist.src = FRAMES.src[i];
  const t = FRAMES.t[i];
  document.getElementById("readout").textContent = t.toFixed(2) + " s";
  const upd = {{}};
  for (const key of ["shapes[1].x0", "shapes[1].x1", "shapes[3].x0", "shapes[3].x1",
                     "shapes[5].x0", "shapes[5].x1", "shapes[7].x0", "shapes[7].x1",
                     "shapes[9].x0", "shapes[9].x1"]) upd[key] = t;
  Plotly.relayout(gd, upd);
}}
slider.addEventListener("input", () => show(+slider.value));
show(0);
</script></body></html>
"""
    # The first shape of each pair is the highlight rect (even index) and the
    # line is the playhead (odd). 5 axes -> lines at 1,3,5,7,9.
    out = OUT_DIR / f"oct_ep{episode}.html"
    out.write_text(page)
    print(f"[view] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return out


def main() -> None:
    build(44, (30.0, 37.0))
    build(42, None)
    build(47, None)


if __name__ == "__main__":
    main()
