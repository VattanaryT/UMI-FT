"""One foot step from the previous foot, the TCP, and the mesh.

The search stays on triangles near the previous foot. A global closest point
is taken only when it is closer to the TCP by more than the switch margin.
The normal is the precomputed smoothed field. t1 is carried by the rotation
that takes the old normal to the new one (two reflections). t2 = n x t1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from fixb_field import FACE_SWITCH_MARGIN_M, orthonormal_frames
from fixb_smooth_field import SmoothFrameField

LOCAL_RADIUS_M = 0.030
SWITCH_MARGIN_M = FACE_SWITCH_MARGIN_M


def double_reflect(t1: np.ndarray, n_prev: np.ndarray, n_new: np.ndarray) -> np.ndarray:
    """Carry a tangent by the rotation that maps ``n_prev`` onto ``n_new``."""
    a = np.asarray(n_prev, dtype=np.float64).reshape(3)
    b = np.asarray(n_new, dtype=np.float64).reshape(3)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-10:
        rot = np.eye(3)
    elif c < -1.0 + 1e-10:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(axis)) < 1e-8:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        K = _skew(axis)
        rot = np.eye(3) + 2.0 * (K @ K)
    else:
        v = np.cross(a, b)
        K = _skew(v)
        rot = np.eye(3) + K + (K @ K) * (1.0 / (1.0 + c))
    return rot @ np.asarray(t1, dtype=np.float64).reshape(3)


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _closest_on_triangles(point, vertices, faces, tri_ids):
    """Closest point on ``tri_ids`` and its barycentric weights. None if empty."""
    if len(tri_ids) == 0:
        return None
    tri = faces[np.asarray(tri_ids, dtype=np.int64)]
    a = vertices[tri[:, 0]]
    b = vertices[tri[:, 1]]
    c = vertices[tri[:, 2]]
    ab = b - a
    ac = c - a
    ap = point.reshape(1, 3) - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = point.reshape(1, 3) - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = point.reshape(1, 3) - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    # Region 0 vertex A, 1 vertex B, 2 vertex C, 3 edge AB, 4 edge AC, 5 edge BC, 6 interior.
    region = np.full(len(tri), 6, dtype=np.int8)
    region[(d1 <= 0.0) & (d2 <= 0.0)] = 0
    region[(region == 6) & (d3 >= 0.0) & (d4 <= d3)] = 1
    region[(region == 6) & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)] = 3
    region[(region == 6) & (d6 >= 0.0) & (d5 <= d6)] = 2
    region[(region == 6) & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)] = 4
    region[(region == 6) & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)] = 5
    w = np.zeros((len(tri), 3), dtype=np.float64)
    v_ab = np.divide(d1, np.clip(d1 - d3, 1e-12, None))
    w_ac = np.divide(d2, np.clip(d2 - d6, 1e-12, None))
    w_bc = np.divide(d4 - d3, np.clip((d4 - d3) + (d5 - d6), 1e-12, None))
    denom = np.clip(va + vb + vc, 1e-12, None)
    interior_v = vb / denom
    interior_w = vc / denom
    for code, bary in (
        (0, (1.0, 0.0, 0.0)),
        (1, (0.0, 1.0, 0.0)),
        (2, (0.0, 0.0, 1.0)),
    ):
        m = region == code
        w[m, 0], w[m, 1], w[m, 2] = bary
    m = region == 3
    w[m, 0] = 1.0 - v_ab[m]
    w[m, 1] = v_ab[m]
    m = region == 4
    w[m, 0] = 1.0 - w_ac[m]
    w[m, 2] = w_ac[m]
    m = region == 5
    w[m, 1] = 1.0 - w_bc[m]
    w[m, 2] = w_bc[m]
    m = region == 6
    w[m, 0] = 1.0 - interior_v[m] - interior_w[m]
    w[m, 1] = interior_v[m]
    w[m, 2] = interior_w[m]
    point_on = np.einsum("ti,tij->tj", w, vertices[tri])
    dist = np.linalg.norm(point_on - point.reshape(1, 3), axis=1)
    best = int(np.argmin(dist))
    return point_on[best], w[best], tri[best]


class CausalFoot:
    """``step(tcp, prev) -> (x, R)``. ``prev`` is ``(x, R)`` or None."""

    def __init__(self, field_npz, frames_npz, radius_m: float = LOCAL_RADIUS_M, margin_m: float = SWITCH_MARGIN_M):
        self.field = SmoothFrameField(Path(field_npz), Path(frames_npz))
        self.vertices = self.field.vertices
        self.faces = self.field.faces
        self.n_in = self.field._ori[:, :, 2].copy()
        self.t1_field = self.field._ori[:, :, 0].copy()
        self.radius_m = float(radius_m)
        self.margin_m = float(margin_m)
        self._tree = cKDTree(self.vertices)
        buckets = [[] for _ in range(len(self.vertices))]
        for face_id, tri in enumerate(self.faces):
            for vid in tri:
                buckets[int(vid)].append(face_id)
        self._faces_of = [np.asarray(bucket, dtype=np.int64) for bucket in buckets]

    def track(self, tcp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tcp = np.asarray(tcp, dtype=np.float64).reshape(-1, 3)
        x = np.zeros_like(tcp)
        frame = np.zeros((len(tcp), 3, 3), dtype=np.float64)
        prev = None
        for i, point in enumerate(tcp):
            x[i], frame[i] = self.step(point, prev)
            prev = (x[i], frame[i])
        return x, frame

    def step(self, tcp: np.ndarray, prev) -> tuple[np.ndarray, np.ndarray]:
        tcp = np.asarray(tcp, dtype=np.float64).reshape(3)
        if prev is None:
            return self._from_global(tcp, carried=None)
        x_prev, r_prev = prev
        r_prev = np.asarray(r_prev, dtype=np.float64).reshape(3, 3)
        local = self._local(tcp, np.asarray(x_prev, dtype=np.float64).reshape(3))
        global_hit = self._global(tcp)
        if local is None:
            chosen = global_hit
        else:
            d_local = float(np.linalg.norm(tcp - local[0]))
            d_global = float(np.linalg.norm(tcp - global_hit[0]))
            chosen = global_hit if d_local - d_global > self.margin_m else local
        n = chosen[1]
        t1 = double_reflect(r_prev[:, 0], r_prev[:, 2], n)
        return chosen[0], orthonormal_frames(t1, n)

    def _global(self, tcp: np.ndarray):
        x, vid, w = self.field.closest(tcp.reshape(1, 3))
        n = w[0] @ self.n_in[vid[0]]
        n = n / max(float(np.linalg.norm(n)), 1e-12)
        t1 = w[0] @ self.t1_field[vid[0]]
        return x[0], n, t1

    def _from_global(self, tcp, carried):
        x, n, t1 = self._global(tcp)
        if carried is not None:
            t1 = double_reflect(carried, n, n)
        return x, orthonormal_frames(t1, n)

    def _local(self, tcp: np.ndarray, x_prev: np.ndarray):
        vids = self._tree.query_ball_point(x_prev, self.radius_m)
        groups = [self._faces_of[int(v)] for v in vids if len(self._faces_of[int(v)])]
        if not groups:
            return None
        tri_ids = np.unique(np.concatenate(groups))
        found = _closest_on_triangles(tcp, self.vertices, self.faces, tri_ids)
        if found is None:
            return None
        point, bary, verts = found
        n = bary @ self.n_in[verts]
        n = n / max(float(np.linalg.norm(n)), 1e-12)
        return point, n
