"""Closest-point field with end-cap and apron triangles removed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d

from fixb_field import FixBMoldField


def _mask(field, npz: Path) -> np.ndarray:
    n_cap = int(json.loads(npz.with_name("gap_report_vertical.json").read_text())["trim"]["field"]["cap_vertices"])
    cap = np.zeros(len(field.vertices), dtype=bool)
    cap[-n_cap:] = True
    apron = np.asarray(np.load(npz)["is_apron"]).astype(bool)
    tri = field.faces
    return ~(apron[tri].any(axis=1) | cap[tri].all(axis=1))


class WallField:
    """Same vertex normals and frames. `closest` only sees wall triangles."""

    def __init__(self, src: FixBMoldField, faces: np.ndarray):
        self.vertices = src.vertices
        self.normals = src.normals
        self.frames = src.frames
        self.faces = np.asarray(faces, dtype=np.int64)
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(self.vertices.astype(np.float32))
        mesh.triangle.indices = o3d.core.Tensor(self.faces.astype(np.int32))
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(mesh)

    def closest(self, p):
        q = o3d.core.Tensor(np.asarray(p, dtype=np.float32).reshape(-1, 3))
        ans = self.scene.compute_closest_points(q)
        tri = np.clip(ans["primitive_ids"].numpy().astype(np.int64), 0, len(self.faces) - 1)
        uv = ans["primitive_uvs"].numpy().astype(np.float64)
        w = np.stack([1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]], axis=-1)
        vid = self.faces[tri]
        hit = np.einsum("qk,qkd->qd", w, self.vertices[vid])
        return hit, vid, w

    def query(self, p, **_ignored):
        return FixBMoldField._closest_frame(self, p, with_curvature=False)


def wall_field(src: FixBMoldField, npz: Path) -> WallField:
    mask = _mask(src, npz)
    if int(mask.sum()) == 0:
        raise RuntimeError("no wall faces")
    wall = WallField(src, src.faces[mask])
    _check(src, wall, mask)
    return wall


def _check(src: FixBMoldField, wall: WallField, mask: np.ndarray) -> None:
    # ponytail: one centroid. The sheet is thinner than a 4 mm offset, so distance-to-offset is not the test.
    center = src.vertices[src.faces[mask][len(src.faces[mask]) // 2]].mean(axis=0)
    hit, _vid, _w = wall.closest(center)
    if float(np.linalg.norm(hit.reshape(3) - center)) > 1e-3:
        raise RuntimeError("wall centroid did not snap to the wall")
    cap = src.vertices[-1]
    hit_c, _vid, _w = wall.closest(cap)
    if float(np.linalg.norm(hit_c.reshape(3) - cap)) < 1e-4:
        raise RuntimeError("cap vertex is still a wall hit")
