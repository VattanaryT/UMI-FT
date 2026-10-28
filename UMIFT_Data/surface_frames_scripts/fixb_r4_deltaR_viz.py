#!/usr/bin/env python3
"""r4 ΔR check: mesh, demo tool vs field frame, tilt and twist.

Picks three val episodes (flat, fillet, off-edge) and writes PNGs plus one
rotatable HTML under the r4 checks/viz directory.

  python fixb_r4_deltaR_viz.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fixb_field import FixBMoldField
from fixb_labels import pose7_to_Rp
from fixb_r4_chunk_filter import FIELD_JUMP_DEG, episode_masks_from_zarr
from fixb_r4_pretrain_report import twist_about_n_deg
from mold_aware_runtime import rot6d_rows_to_R

import zarr

ROOT = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb")
R4 = ROOT / "r4_world_fixb_tilt" / "acp_replay_buffer.zarr"
FIELD = ROOT / "shared" / "fixb_field.npz"
OUT = ROOT / "r4_world_fixb_tilt" / "checks" / "viz"
VAL_EPISODES = (5, 8, 14, 37, 56, 57, 65, 73)
ACTION_H = 24
TRIAD_M = 0.015
MESH_TRI_CAP = 20000


def tilt_from_normal_deg(R_tool: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Angle (deg) between tool +z and −n. Same quantity the 45° clamp uses."""
    z = R_tool[:, :, 2]
    target = -n / np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9, None)
    cos = np.clip(np.sum(z * target, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def masked_after_jump(jump: np.ndarray, horizon: int = ACTION_H) -> np.ndarray:
    """True where a chunk of `horizon` steps would zero ΔR at this sample.

    A step t is masked when some chunk start s in [t-horizon+1, t) contains a
    field jump in (s, t]. That is any jump in (t-horizon, t].
    """
    jump = np.asarray(jump, dtype=bool)
    masked = np.zeros(len(jump), dtype=bool)
    idx = np.flatnonzero(jump)
    for t in idx:
        masked[t : min(len(jump), t + horizon)] = True
    return masked


def decimated_mesh(vertices: np.ndarray, faces: np.ndarray):
    if len(faces) <= MESH_TRI_CAP:
        return vertices, faces
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=MESH_TRI_CAP)
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    return v, f


def episode_arrays(ep):
    R_tool_gt, p = pose7_to_Rp(ep["ts_pose_fb_0"])
    R_f = np.asarray(ep["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
    dR = rot6d_rows_to_R(np.asarray(ep["fixb_delta_R_rot6d_0"], dtype=np.float64).reshape(-1, 6))
    n = min(len(p), len(R_f), len(dR), len(R_tool_gt))
    R_f, dR, p = R_f[:n], dR[:n], p[:n]
    R_tool = np.einsum("tij,tjk->tik", R_f, dR)
    t = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)[:n]
    t = t - t[0]
    masks = episode_masks_from_zarr(ep)
    jump = np.asarray(masks["field_jump"])[:n]
    return {
        "p": p,
        "R_f": R_f,
        "R_tool": R_tool,
        "t": t,
        "tilt": tilt_from_normal_deg(R_tool, R_f[:, :, 2]),
        "twist": twist_about_n_deg(dR),
        "jump": jump,
        "masked": masked_after_jump(jump),
        "off_edge": np.asarray(masks["off_edge"])[:n],
    }


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    best = cur = start = 0
    run_start = 0
    for i, flag in enumerate(np.asarray(mask, dtype=bool)):
        if flag:
            if cur == 0:
                run_start = i
            cur += 1
            if cur > best:
                best = cur
                start = run_start
        else:
            cur = 0
    return best, start


def select_episodes(root, field: FixBMoldField) -> dict:
    """Pick by contiguous stretches, not episode-mean curvature.

    Flat: longest on-mesh run with |k1| < 8 /m. Fillet: longest on-mesh run
    with |k1| > 30 /m (about a 33 mm radius; the field's upper tail).
    Off-edge: longest in-plane > 5 mm run.
    """
    rows = []
    step = 10
    for epi in VAL_EPISODES:
        ep = root[f"episode_{epi}"]
        p = np.asarray(ep["ts_pose_fb_0"])[:, :3]
        k1 = np.abs(field.query(p[::step], with_curvature=True)["curv"][:, 0])
        off = episode_masks_from_zarr(ep)["off_edge"][::step][: len(k1)]
        on = ~off
        flat_n, flat_i = _longest_run((k1 < 8.0) & on)
        fil_n, fil_i = _longest_run((k1 > 30.0) & on)
        off_n, off_i = _longest_run(off)
        rows.append(
            {
                "episode": int(epi),
                "off_edge_frac": float(np.mean(off)),
                "flat_frac": float(np.mean((k1 < 8.0) & on)),
                "fillet_frac": float(np.mean((k1 > 30.0) & on)),
                "flat_run": int(flat_n * step),
                "flat_start": int(flat_i * step),
                "fillet_run": int(fil_n * step),
                "fillet_start": int(fil_i * step),
                "off_run": int(off_n * step),
                "off_start": int(off_i * step),
                "n": int(len(p)),
            }
        )
    flat = max(rows, key=lambda r: r["flat_run"])
    off = max(rows, key=lambda r: (r["episode"] != flat["episode"], r["off_run"]))
    fillet = max(
        rows,
        key=lambda r: (
            r["episode"] not in (flat["episode"], off["episode"]),
            r["fillet_run"],
        ),
    )
    chosen = {"flat": flat, "fillet": fillet, "off_edge": off}
    run_of = {"flat": "flat", "fillet": "fillet", "off_edge": "off"}
    for role, row in chosen.items():
        prefix = run_of[role]
        center = int(row[f"{prefix}_start"] + row[f"{prefix}_run"] // 2)
        row["zoom_start"] = max(0, center - 50)
    return {**chosen, "candidates": rows}


def _triad_segments(origin, R, length):
    segs = []
    for k in range(3):
        tip = origin + R[:, k] * length
        segs.append((origin, tip))
    return segs


def _style_3d(ax, pts: np.ndarray, pad: float = 0.02) -> None:
    lo = np.asarray(pts, dtype=np.float64).min(0) - pad
    hi = np.asarray(pts, dtype=np.float64).max(0) + pad
    span = np.maximum(hi - lo, 0.02)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(float(s) for s in span))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=22, azim=-58)


def _draw_triads(ax, p, R_f, R_tool, ids) -> None:
    for i in ids:
        for (a, b), color in zip(
            _triad_segments(p[i], R_f[i], TRIAD_M), ("#d62728", "#2ca02c", "#1f77b4")
        ):
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, lw=1.2)
        for (a, b), color in zip(
            _triad_segments(p[i], R_tool[i], TRIAD_M * 0.75),
            ("#ff7f0e", "#9467bd", "#111111"),
        ):
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, lw=1.2)


def save_png(role: str, epi: int, data: dict, vertices, zoom_start: int, path: Path) -> None:
    p, t = data["p"], data["t"]
    # Full path, lightly subsampled. Triads only on a ~0.5 s window, every 10 steps.
    path_step = max(1, len(p) // 2500)
    zoom0 = int(np.clip(zoom_start, 0, max(len(p) - 120, 0)))
    zoom1 = min(len(p), zoom0 + 100)
    triad_ids = np.arange(zoom0, zoom1, 10)
    fig = plt.figure(figsize=(12, 9))
    mesh_ids = np.arange(0, len(vertices), 12)
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.scatter(
        vertices[mesh_ids, 0],
        vertices[mesh_ids, 1],
        vertices[mesh_ids, 2],
        c="0.55",
        s=1.0,
        depthshade=False,
    )
    sc = ax.scatter(
        p[::path_step, 0],
        p[::path_step, 1],
        p[::path_step, 2],
        c=data["tilt"][::path_step],
        cmap="turbo",
        s=3,
        vmin=0,
        vmax=45,
        depthshade=False,
    )
    jm = np.flatnonzero(data["jump"])
    if len(jm):
        ax.scatter(p[jm, 0], p[jm, 1], p[jm, 2], c="red", s=12, marker="x", depthshade=False)
    _style_3d(ax, vertices, pad=0.03)
    ax.set_title("full path, color = tilt vs −n")
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.12, label="deg")

    axz = fig.add_subplot(2, 2, 2, projection="3d")
    axz.scatter(
        vertices[mesh_ids, 0],
        vertices[mesh_ids, 1],
        vertices[mesh_ids, 2],
        c="0.8",
        s=0.3,
        depthshade=False,
    )
    sl = slice(zoom0, zoom1)
    axz.scatter(
        p[sl, 0], p[sl, 1], p[sl, 2], c=data["tilt"][sl], cmap="turbo", s=8, vmin=0, vmax=45, depthshade=False
    )
    _draw_triads(axz, p, data["R_f"], data["R_tool"], triad_ids)
    if np.any(data["jump"][sl]):
        jj = np.flatnonzero(data["jump"][sl]) + zoom0
        axz.scatter(p[jj, 0], p[jj, 1], p[jj, 2], c="red", s=30, marker="x", depthshade=False)
    _style_3d(axz, p[sl], pad=0.02)
    axz.set_title(
        f"zoom {t[zoom0]:.1f}–{t[zoom1-1]:.1f}s, triads every 10 steps\n"
        "field t1/t2/n = red/green/blue; tool = orange/purple/black"
    )

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(t, data["tilt"], color="C0", lw=0.6, label="tilt vs −n (deg)")
    ax2.plot(t, data["twist"], color="C1", lw=0.6, label="twist about n (deg)")
    if len(jm):
        ax2.scatter(t[jm], data["tilt"][jm], c="red", s=10, zorder=3, label="field jump")
    ax2.axvspan(t[zoom0], t[min(zoom1, len(t) - 1)], color="0.85", label="triad window")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("deg")
    ax2.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"ep {epi} ({role})  red x = field jump >{FIELD_JUMP_DEG:.0f}°  "
        f"masked ΔR {data['masked'].mean():.0%} of steps",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _mesh_trace(v, f):
    return go.Mesh3d(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=f[:, 0],
        j=f[:, 1],
        k=f[:, 2],
        color="lightgray",
        opacity=0.35,
        name="mold",
        hoverinfo="skip",
        showlegend=False,
    )


def _triad_trace(p, R, ids, length, color, name):
    xs, ys, zs = [], [], []
    for i in ids:
        for k in range(3):
            a = p[i]
            b = p[i] + R[i, :, k] * length
            xs += [a[0], b[0], None]
            ys += [a[1], b[1], None]
            zs += [a[2], b[2], None]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color=color, width=3),
        name=name,
        hoverinfo="skip",
    )


def save_html(episodes: list[tuple[str, int, dict, int]], v, f, path: Path) -> None:
    titles = []
    for role, epi, _, _ in episodes:
        titles += [f"ep {epi} {role}", f"ep {epi} tilt / twist"]
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}] for _ in range(3)],
        subplot_titles=titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.06,
    )
    for row, (role, epi, data, zoom_start) in enumerate(episodes, start=1):
        p, t = data["p"], data["t"]
        zoom0 = int(np.clip(zoom_start, 0, max(len(p) - 120, 0)))
        zoom1 = min(len(p), zoom0 + 100)
        ids = np.arange(zoom0, zoom1, 10)
        path_ids = np.arange(0, len(p), max(1, len(p) // 2500))
        fig.add_trace(_mesh_trace(v, f), row=row, col=1)
        fig.add_trace(
            go.Scatter3d(
                x=p[path_ids, 0],
                y=p[path_ids, 1],
                z=p[path_ids, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=data["tilt"][path_ids],
                    colorscale="Turbo",
                    cmin=0,
                    cmax=45,
                    colorbar=dict(title="tilt deg", len=0.25, y=1.0 - (row - 1) * 0.33)
                    if row == 1
                    else None,
                    showscale=(row == 1),
                ),
                name=f"ep{epi} path",
                text=[f"t={tt:.2f}s tilt={ang:.1f}" for tt, ang in zip(t[path_ids], data["tilt"][path_ids])],
                hoverinfo="text",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            _triad_trace(p, data["R_f"], ids, TRIAD_M, "#1f77b4", f"ep{epi} field"),
            row=row,
            col=1,
        )
        fig.add_trace(
            _triad_trace(p, data["R_tool"], ids, TRIAD_M * 0.85, "#111111", f"ep{epi} tool"),
            row=row,
            col=1,
        )
        jm = np.flatnonzero(data["jump"])
        if len(jm):
            fig.add_trace(
                go.Scatter3d(
                    x=p[jm, 0],
                    y=p[jm, 1],
                    z=p[jm, 2],
                    mode="markers",
                    marker=dict(size=4, color="red", symbol="x"),
                    name=f"ep{epi} field jump",
                ),
                row=row,
                col=1,
            )
        mk = np.flatnonzero(data["masked"])
        mk = mk[:: max(1, len(mk) // 300)] if len(mk) else mk
        if len(mk):
            fig.add_trace(
                go.Scatter3d(
                    x=p[mk, 0],
                    y=p[mk, 1],
                    z=p[mk, 2],
                    mode="markers",
                    marker=dict(size=2, color="red"),
                    name=f"ep{epi} masked ΔR",
                    opacity=0.5,
                ),
                row=row,
                col=1,
            )
        fig.add_trace(
            go.Scatter(x=t, y=data["tilt"], mode="lines", name=f"ep{epi} tilt", line=dict(width=1)),
            row=row,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=t, y=data["twist"], mode="lines", name=f"ep{epi} twist", line=dict(width=1)),
            row=row,
            col=2,
        )
        if len(jm):
            fig.add_trace(
                go.Scatter(
                    x=t[jm],
                    y=data["tilt"][jm],
                    mode="markers",
                    marker=dict(color="red", size=6, symbol="x"),
                    name=f"ep{epi} jump",
                ),
                row=row,
                col=2,
            )
        fig.update_scenes(aspectmode="data", row=row, col=1)
        fig.update_xaxes(title_text="time (s)", row=row, col=2)
        fig.update_yaxes(title_text="deg", row=row, col=2)
    fig.update_layout(
        height=1600,
        width=1400,
        title="r4 val ΔR. Path colored by tilt. Triads every 10 steps in a 0.5 s window: field blue, tool black. Red = field jump.",
        legend=dict(font=dict(size=9)),
    )
    fig.write_html(str(path), include_plotlyjs="cdn")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    field = FixBMoldField(FIELD)
    v, f = decimated_mesh(field.vertices, field.faces)
    root = zarr.open(str(R4), "r")["data"]
    choice = select_episodes(root, field)
    (OUT / "selection.json").write_text(json.dumps(choice, indent=2))
    packed = []
    for role in ("flat", "fillet", "off_edge"):
        epi = int(choice[role]["episode"])
        data = episode_arrays(root[f"episode_{epi}"])
        png = OUT / f"episode_{epi}_{role}.png"
        save_png(role, epi, data, v, int(choice[role]["zoom_start"]), png)
        packed.append((role, epi, data, int(choice[role]["zoom_start"])))
        print(
            f"{role}: episode_{epi} off_edge={choice[role]['off_edge_frac']:.3f} "
            f"flat_run={choice[role]['flat_run']} fillet_run={choice[role]['fillet_run']} "
            f"off_run={choice[role]['off_run']} -> {png}"
        )
    html = OUT / "deltaR_val.html"
    save_html(packed, v, f, html)
    print("html", html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
