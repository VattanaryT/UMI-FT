"""Foot on a face within 30 mm whose outward normal matches the press.

Default foot_mode is closest-point. foot_mode='aligned' changes the choice.
Press direction is -F_world at or above 8 N, and the tool z axis below that.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

RADIUS = 0.030
FORCE_N = 8.0


def direction_from(rotation, force_world, force_mag: float) -> np.ndarray:
    """World-frame direction the outward normal should match."""
    if float(force_mag) >= FORCE_N:
        return -np.asarray(force_world, dtype=np.float64).reshape(3)
    return np.asarray(rotation, dtype=np.float64).reshape(3, 3)[:, 2]


def _closest_on_triangles(point: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Closest point on each triangle. `tri` is (K, 3, 3). Ericson, vectorized."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    p = np.asarray(point, dtype=np.float64).reshape(1, 3)
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1 = np.sum(ab * ap, axis=1)
    d2 = np.sum(ac * ap, axis=1)
    d3 = np.sum(ab * bp, axis=1)
    d4 = np.sum(ac * bp, axis=1)
    d5 = np.sum(ab * cp, axis=1)
    d6 = np.sum(ac * cp, axis=1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    out = np.empty_like(a)
    region = np.zeros(len(a), dtype=np.int8)
    mask = (d1 <= 0.0) & (d2 <= 0.0)
    out[mask] = a[mask]
    region[mask] = 1
    mask = (region == 0) & (d3 >= 0.0) & (d4 <= d3)
    out[mask] = b[mask]
    region[mask] = 1
    mask = (region == 0) & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    v = (d1 / np.maximum(d1 - d3, 1e-12))[mask]
    out[mask] = a[mask] + v[:, None] * ab[mask]
    region[mask] = 1
    mask = (region == 0) & (d6 >= 0.0) & (d5 <= d6)
    out[mask] = c[mask]
    region[mask] = 1
    mask = (region == 0) & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    w = (d2 / np.maximum(d2 - d6, 1e-12))[mask]
    out[mask] = a[mask] + w[:, None] * ac[mask]
    region[mask] = 1
    mask = (region == 0) & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    w = ((d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1e-12))[mask]
    out[mask] = b[mask] + w[:, None] * (c[mask] - b[mask])
    region[mask] = 1
    mask = region == 0
    denom = np.maximum(va + vb + vc, 1e-12)
    v = (vb / denom)[mask]
    w = (vc / denom)[mask]
    out[mask] = a[mask] + v[:, None] * ab[mask] + w[:, None] * ac[mask]
    return out


def _barycentric(foot: np.ndarray, tri: np.ndarray) -> np.ndarray:
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0, v1, v2 = b - a, c - a, foot - a
    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    d20 = np.sum(v2 * v0, axis=1)
    d21 = np.sum(v2 * v1, axis=1)
    denom = np.maximum(d00 * d11 - d01 * d01, 1e-18)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    weight = np.stack([u, v, w], axis=1)
    weight = np.clip(weight, 0.0, None)
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
    return weight


class AlignedFoot:
    def __init__(self, src, foot_mode: str = "closest", radius: float = RADIUS):
        self.src = src
        self.vertices = src.vertices
        self.normals = src.normals
        self.frames = src.frames
        self.faces = src.faces
        self.foot_mode = foot_mode
        self.radius = float(radius)
        self.direction = None
        self._build()
        _check(self)

    def _build(self) -> None:
        faces = self.faces
        verts = self.vertices
        tri = verts[faces]
        edge = np.stack([
            np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
            np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
        ], axis=1)
        self._pad = float(edge.max())
        self._tri = tri
        geo = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        mean_n = self.normals[faces].mean(axis=1)
        geo *= np.where(np.sum(geo * mean_n, axis=1) < 0.0, -1.0, 1.0)[:, None]
        self._geo = geo / np.clip(np.linalg.norm(geo, axis=1, keepdims=True), 1e-12, None)
        owners = np.repeat(np.arange(len(faces)), 3)
        vids = faces.reshape(-1)
        order = np.argsort(vids, kind="stable")
        self._face_ids = owners[order]
        cuts = np.searchsorted(vids[order], np.arange(len(verts) + 1))
        self._vert_span = cuts
        self._tree = cKDTree(verts)

    def closest(self, p):
        from fixb_field import FixBMoldField

        pts = np.asarray(p, dtype=np.float64).reshape(-1, 3)
        if self.foot_mode != "aligned" or self.direction is None:
            return FixBMoldField.closest(self.src, pts)
        direction = np.asarray(self.direction, dtype=np.float64).reshape(-1, 3)
        if len(direction) == 1 and len(pts) != 1:
            direction = np.repeat(direction, len(pts), axis=0)
        return self._choose(pts, direction)[:3]

    def query(self, p, **_ignored):
        from fixb_field import FixBMoldField

        return FixBMoldField._closest_frame(self, p, with_curvature=False)

    def choose(self, points, directions):
        """Foot, vertex ids, barycentric weights, triangle id, fell-back mask."""
        return self._choose(np.asarray(points, dtype=np.float64).reshape(-1, 3),
                            np.asarray(directions, dtype=np.float64).reshape(-1, 3))

    def _choose(self, pts, directions):
        n = len(pts)
        hit = np.zeros((n, 3))
        vid = np.zeros((n, 3), dtype=np.int64)
        weight = np.zeros((n, 3))
        tri = np.zeros(n, dtype=np.int64)
        fallback = np.zeros(n, dtype=bool)
        direct = directions / np.clip(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12, None)
        balls = self._tree.query_ball_point(pts, self.radius + self._pad)
        for i, verts in enumerate(balls):
            if len(verts) == 0 or not np.isfinite(direct[i]).all():
                fallback[i] = True
                continue
            spans = [self._face_ids[self._vert_span[v]:self._vert_span[v + 1]] for v in verts]
            faces = np.unique(np.concatenate(spans)) if spans else np.zeros(0, dtype=np.int64)
            if len(faces) == 0:
                fallback[i] = True
                continue
            cand = self._tri[faces]
            near = _closest_on_triangles(pts[i], cand)
            dist = np.linalg.norm(near - pts[i], axis=1)
            keep = dist <= self.radius + 1e-8
            if not np.any(keep):
                fallback[i] = True
                continue
            dots = self._geo[faces][keep] @ direct[i]
            order = np.lexsort((dist[keep], -dots))
            pick = int(np.flatnonzero(keep)[order[0]])
            face = int(faces[pick])
            tri[i] = face
            hit[i] = near[pick]
            weight[i] = _barycentric(near[pick:pick + 1], self._tri[face:face + 1])[0]
            vid[i] = self.faces[face]
        if np.any(fallback):
            import open3d as o3d
            from fixb_field import FixBMoldField

            h, v, w = FixBMoldField.closest(self.src, pts[fallback])
            hit[fallback] = h
            vid[fallback] = v
            weight[fallback] = w
            ans = self.src.scene.compute_closest_points(
                o3d.core.Tensor(pts[fallback].astype(np.float32))
            )
            tri[fallback] = np.clip(ans["primitive_ids"].numpy().astype(np.int64), 0, len(self.faces) - 1)
        return hit, vid, weight, tri, fallback


def _check(field: AlignedFoot) -> None:
    eye = np.eye(3)
    force = np.array([0.0, 0.0, 4.0])
    if not np.allclose(direction_from(eye, force, 9.0), -force):
        raise RuntimeError("aligned direction is not -F above 8 N")
    if not np.allclose(direction_from(eye, force, 1.0), eye[:, 2]):
        raise RuntimeError("aligned direction is not tool z below 8 N")
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    got = _closest_on_triangles(np.array([0.25, 0.25, 0.0]), a[None])[0]
    if np.linalg.norm(got - np.array([0.25, 0.25, 0.0])) > 1e-6:
        raise RuntimeError("triangle closest left the face")
    above = _closest_on_triangles(np.array([0.25, 0.25, 0.4]), a[None])[0]
    if np.linalg.norm(above - np.array([0.25, 0.25, 0.0])) > 1e-6:
        raise RuntimeError("triangle closest missed the plane")
    areas = np.linalg.norm(np.cross(field._tri[:, 1] - field._tri[:, 0], field._tri[:, 2] - field._tri[:, 0]), axis=1)
    face = int(np.argmax(areas))
    center = field._tri[face].mean(axis=0)
    point = center + 0.01 * field._geo[face]
    saved_mode, saved_dir = field.foot_mode, field.direction
    field.foot_mode = "aligned"
    field.direction = field._geo[face]
    hit, _vid, _w = field.closest(point)
    field.foot_mode = saved_mode
    field.direction = saved_dir
    if float(np.linalg.norm(np.asarray(hit).reshape(3) - point)) > 0.02:
        raise RuntimeError(f"aligned snap {np.linalg.norm(np.asarray(hit).reshape(3) - point) * 1e3:.1f} mm")


def thin_kind(field_src, npz: Path) -> np.ndarray:
    n_cap = int(json.loads(Path(npz).with_name("gap_report_vertical.json").read_text())["trim"]["field"]["cap_vertices"])
    cap = np.zeros(len(field_src.vertices), dtype=bool)
    cap[-n_cap:] = True
    apron = np.asarray(np.load(npz)["is_apron"]).astype(bool)
    tri = field_src.faces
    kind = np.full(len(tri), "wall", dtype=object)
    kind[apron[tri].any(axis=1)] = "apron"
    kind[cap[tri].all(axis=1)] = "cap"
    return kind
