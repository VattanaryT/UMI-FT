#!/usr/bin/env python3
"""Build the fixture-frame mold mesh, frame field and curvature for Fix B.

  1. scan (scanner base) -> fixture frame, voxel downsample, normals
  2. screened Poisson mesh, clip to the scan (drops the Poisson underside),
     keep the largest component, light Taubin smoothing
  3. orient outward: flip so demo tool positions in free space have sd > 0
  4. t1 = heat-gradient tangent (geodude_teleoperation.stringer_field
     length_tangents, the field the teleop controller uses), n = vertex normal
  5. per-vertex shape operator from a weighted quadric fit, smoothed by
     Gaussian averaging of the 3x3 tangent tensor

Writes <out_dir>/fixb_field.npz, fixb_field.json, fixb_mold_fixture.ply and a
preview PNG.

Usage:
  python fixb_build_mold_field.py \
    --out_dir /mnt/data/umidata/ft/mold_aware_datasets/stringer_data_v2_full_fixb/shared
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pymeshlab
import trimesh
import zarr
from scipy.spatial import cKDTree

from fixb_field import FixBMoldField, curvature_features, orthonormal_frames

STRINGER_FIELD_PY = Path(
    "/home/boeing/ros2_ws/src/geodude-teleoperation/src/geodude_teleoperation/stringer_field.py"
)
MOLD_ID = "stringer_scan_20261008_170729"


def load_stringer_field():
    spec = importlib.util.spec_from_file_location("stringer_field", STRINGER_FIELD_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scan_to_fixture(scan_ply: Path, T: np.ndarray, voxel: float) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(scan_ply))
    pcd.transform(T)
    pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=6 * voxel, max_nn=40))
    pcd.orient_normals_consistent_tangent_plane(30)
    return pcd


def poisson_mesh(pcd: o3d.geometry.PointCloud, depth: int, clip_scale: float, taubin_iters: int):
    pts = np.asarray(pcd.points)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=pts, v_normals_matrix=np.asarray(pcd.normals)))
    ms.generate_surface_reconstruction_screened_poisson(depth=depth, preclean=True)
    m = ms.current_mesh()
    mesh = trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), process=True)

    tree = cKDTree(pts)
    spacing = float(np.median(tree.query(pts, k=2)[0][:, 1]))
    dist, _ = tree.query(mesh.vertices)
    keep_v = dist <= clip_scale * spacing
    mesh.update_faces(keep_v[mesh.faces].all(axis=1))
    mesh.remove_unreferenced_vertices()
    parts = mesh.split(only_watertight=False)
    mesh = max(parts, key=lambda c: len(c.faces))
    trimesh.smoothing.filter_taubin(mesh, iterations=taubin_iters)
    mesh.fix_normals()
    return mesh, spacing


def demo_tool_positions(source_zarr: Path, stride: int) -> np.ndarray:
    data = zarr.open(str(source_zarr), "r")["data"]
    pts = [np.asarray(data[ep]["ts_pose_fb_0"][::stride, :3]) for ep in data.keys()]
    return np.concatenate(pts, axis=0)


def orient_outward(mesh: trimesh.Trimesh, tool_xyz: np.ndarray):
    """Flip so tool positions 10-100 mm off the surface are mostly on the +n side."""
    closest, dist, tri = trimesh.proximity.closest_point(mesh, tool_xyz)
    sd = np.sum((tool_xyz - closest) * mesh.face_normals[tri], axis=1)
    band = (dist > 0.01) & (dist < 0.10)
    frac_pos = float(np.mean(sd[band] > 0))
    if frac_pos < 0.5:
        mesh.invert()
        frac_pos = 1.0 - frac_pos
    return frac_pos, int(band.sum())


def flange_apron(mesh: trimesh.Trimesh, length_axis: np.ndarray, t1_scan: np.ndarray, extent: float, spacing: float):
    """Flat grid in the flange plane beside the scan (sides only, not past the ends).

    Returns vertices, faces, normals, t1 for a mesh piece that is not connected
    to the scan mesh; cells whose centre is within 0.75*spacing of the scan
    footprint (x-z) are left out.
    """
    V = np.asarray(mesh.vertices)
    n = np.asarray(mesh.vertex_normals)
    flat = np.abs(n[:, 1]) > 0.98
    y_top = V[flat, 1].max()
    flange = flat & (V[:, 1] > y_top - 0.015)
    y0 = float(np.median(V[flange, 1]))
    n_out = np.array([0.0, np.sign(n[flange, 1].mean()), 0.0])

    lo, hi = V.min(axis=0), V.max(axis=0)
    xs = np.arange(lo[0], hi[0] + 0.5 * spacing, spacing)
    zs = np.arange(lo[2] - extent, hi[2] + extent + 0.5 * spacing, spacing)
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    verts = np.stack([gx.ravel(), np.full(gx.size, y0), gz.ravel()], axis=1)
    nx, nz = len(xs), len(zs)
    ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(nz - 1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    centres = np.stack([xs[ii] + 0.5 * spacing, zs[jj] + 0.5 * spacing], axis=1)
    keep = cKDTree(V[:, [0, 2]]).query(centres)[0] > 0.75 * spacing
    ii, jj = ii[keep], jj[keep]
    v00, v10, v01, v11 = ii * nz + jj, (ii + 1) * nz + jj, ii * nz + jj + 1, (ii + 1) * nz + jj + 1
    faces = np.concatenate([np.stack([v00, v10, v11], 1), np.stack([v00, v11, v01], 1)])
    tri_n = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    if np.mean(tri_n @ n_out) < 0:
        faces = faces[:, [0, 2, 1]]
    used = np.unique(faces)
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    verts, faces = verts[used], remap[faces]

    axis = np.asarray(length_axis, dtype=np.float64)
    if np.mean(t1_scan[flange] @ axis) < 0:
        axis = -axis
    normals = np.tile(n_out, (len(verts), 1))
    t1 = np.tile(axis, (len(verts), 1))
    return verts, faces, normals, t1, y0


def shape_operators(mesh, frames, fit_radius: float, smooth_radius: float) -> np.ndarray:
    V = np.asarray(mesh.vertices)
    tree = cKDTree(V)
    S3 = np.zeros((len(V), 3, 3))
    for i, nbrs in enumerate(tree.query_ball_point(V, fit_radius)):
        nbrs = np.asarray(nbrs)
        local = (V[nbrs] - V[i]) @ frames[i]
        x, y, z = local[:, 0], local[:, 1], local[:, 2]
        w = np.exp(-0.5 * (x**2 + y**2) / (0.5 * fit_radius) ** 2)
        A = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], axis=1)
        if len(nbrs) < 8:
            continue
        coef, *_ = np.linalg.lstsq(A * w[:, None], z * w, rcond=None)
        a, b, c = coef[:3]
        # z(x, y) along +n; the surface bends away from the tool (convex) when z < 0.
        S2 = -np.array([[2 * a, b], [b, 2 * c]])
        B = frames[i][:, :2]
        S3[i] = B @ S2 @ B.T
    if smooth_radius > 0:
        out = np.zeros_like(S3)
        for i, nbrs in enumerate(tree.query_ball_point(V, smooth_radius)):
            nbrs = np.asarray(nbrs)
            d2 = np.sum((V[nbrs] - V[i]) ** 2, axis=1)
            w = np.exp(-0.5 * d2 / (0.5 * smooth_radius) ** 2)
            M = np.tensordot(w, S3[nbrs], axes=1) / w.sum()
            B = frames[i][:, :2]
            out[i] = B @ (B.T @ M @ B) @ B.T
        S3 = out
    return S3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan_ply", type=Path, default=Path("/mnt/data/boeing/scans/scan_20261008_170729/merged.ply"))
    ap.add_argument(
        "--T_fixture_from_scanner",
        type=Path,
        default=Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_data_v2_full/shared/T_fixture_from_scanner_base.npy"),
    )
    ap.add_argument(
        "--source_zarr",
        type=Path,
        default=Path("/mnt/data/umidata/ft/datasets/stringer_data_v2_full/acp_replay_buffer.zarr"),
    )
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--voxel", type=float, default=0.002)
    ap.add_argument("--poisson_depth", type=int, default=9)
    ap.add_argument("--clip_scale", type=float, default=3.0)
    ap.add_argument("--taubin_iters", type=int, default=10)
    ap.add_argument("--end_fraction", type=float, default=0.06)
    ap.add_argument("--diffusion_scalar", type=float, default=5000.0)
    ap.add_argument("--curv_fit_radius", type=float, default=0.012)
    ap.add_argument("--curv_smooth_radius", type=float, default=0.010)
    ap.add_argument("--apron_m", type=float, default=0.25, help="flange-plane extension beside the scan; 0 = off")
    ap.add_argument("--apron_spacing", type=float, default=0.005)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    T = np.load(args.T_fixture_from_scanner)
    pcd = scan_to_fixture(args.scan_ply, T, args.voxel)
    print(f"[field] scan -> {len(pcd.points)} fixture points at {args.voxel * 1e3:.1f} mm")
    mesh, spacing = poisson_mesh(pcd, args.poisson_depth, args.clip_scale, args.taubin_iters)
    print(f"[field] mesh {len(mesh.vertices)} verts / {len(mesh.faces)} faces, spacing {spacing * 1e3:.2f} mm")
    frac_pos, n_band = orient_outward(mesh, demo_tool_positions(args.source_zarr, stride=40))
    print(f"[field] outward check: {frac_pos:.3f} of {n_band} free-space tool samples have sd > 0")

    sf = load_stringer_field()
    tangent, sink, source, axis = sf.length_tangents(
        mesh, fraction=args.end_fraction, flow="length", diffusion_scalar=args.diffusion_scalar
    )
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    frames = orthonormal_frames(tangent, normals)
    S3 = shape_operators(mesh, frames, args.curv_fit_radius, args.curv_smooth_radius)

    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    is_apron = np.zeros(len(vertices), dtype=bool)
    flange_y = None
    if args.apron_m > 0:
        av, af, an, at1, flange_y = flange_apron(mesh, axis, frames[..., 0], args.apron_m, args.apron_spacing)
        faces = np.concatenate([faces, af + len(vertices)])
        vertices = np.concatenate([vertices, av])
        normals = np.concatenate([normals, an])
        frames = np.concatenate([frames, orthonormal_frames(at1, an)])
        S3 = np.concatenate([S3, np.zeros((len(av), 3, 3))])
        is_apron = np.concatenate([is_apron, np.ones(len(av), dtype=bool)])
        print(f"[field] apron: {len(av)} verts in flange plane y={flange_y * 1e3:.2f} mm, +-{args.apron_m:g} m in z")

    npz = args.out_dir / "fixb_field.npz"
    np.savez_compressed(
        npz,
        vertices=vertices,
        faces=faces,
        normals=normals,
        frames=frames,
        curv_tensor=S3,
        is_apron=is_apron,
    )
    trimesh.Trimesh(vertices, faces, process=False).export(args.out_dir / "fixb_mold_fixture.ply")
    meta = {
        "mold_id": MOLD_ID,
        "mesh_version": (
            f"scan_20261008_170729/merged.ply->fixture voxel{args.voxel * 1e3:g}mm "
            f"poisson_d{args.poisson_depth} clip{args.clip_scale:g} taubin{args.taubin_iters} outward"
            + (f" +flange_apron{args.apron_m:g}m@{args.apron_spacing * 1e3:g}mm" if args.apron_m > 0 else "")
        ),
        "field_version": (
            f"stringer_field.length_tangents frac{args.end_fraction:g} ds{args.diffusion_scalar:g}; "
            f"n=vertex normal; barycentric closest-point interp; "
            f"curv quadric r{args.curv_fit_radius * 1e3:g}mm smooth{args.curv_smooth_radius * 1e3:g}mm"
        ),
        "frame_columns": ["t1", "t2", "n"],
        "normal_orientation": "outward (sd > 0 in free space)",
        "outward_frac_positive": frac_pos,
        "pca_length_axis": np.asarray(axis).tolist(),
        "curvature_features": ["k1", "k2", "a*cos(2*theta)", "a*sin(2*theta)"],
        "curvature_sign": "convex (bends away from the tool) > 0, units 1/m",
        "n_vertices": int(len(vertices)),
        "n_faces": int(len(faces)),
        "n_apron_vertices": int(is_apron.sum()),
        "flange_plane_y_m": flange_y,
        "voxel_m": args.voxel,
        "T_fixture_from_scanner": T.tolist(),
        "scan_ply": str(args.scan_ply),
    }
    npz.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    feats = curvature_features(np.einsum("via,vij,vjb->vab", frames[..., :2], S3, frames[..., :2]))
    V = np.asarray(mesh.vertices)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    idx = np.arange(0, len(V), max(1, len(V) // 1500))
    axs[0].quiver(V[idx, 0], V[idx, 2], frames[idx, 0, 0], frames[idx, 2, 0], scale=40, width=0.002)
    axs[0].set_title("t1 (heat-gradient tangent), x-z view")
    for a, k, title in ((axs[1], 0, "k1 [1/m]"), (axs[2], 1, "k2 [1/m]")):
        lim = np.percentile(np.abs(feats[:, k]), 98)
        sc = a.scatter(V[:, 0], V[:, 2], c=feats[:, k], s=1, cmap="RdBu", vmin=-lim, vmax=lim)
        plt.colorbar(sc, ax=a)
        a.set_title(title)
    for a in axs:
        a.set_aspect("equal")
        a.set_xlabel("x [m]")
        a.set_ylabel("z [m]")
    fig.tight_layout()
    fig.savefig(args.out_dir / "fixb_field_preview.png", dpi=110)
    print(f"[field] wrote {npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
