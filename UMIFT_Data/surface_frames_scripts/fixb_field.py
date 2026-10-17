"""Mold surface model for the Fix B datasets: mesh, frame field, curvature.

Saved by fixb_build_mold_field.py as ``fixb_field.npz`` (fixture frame):
  vertices (V,3), faces (F,3), normals (V,3) outward,
  frames (V,3,3) columns [t1, t2, n] from the diffused heat-gradient tangent
  (same construction as geodude_teleoperation.stringer_field),
  curv_tensor (V,3,3) smoothed tangent-plane shape operator (1/m, convex > 0).

Queries use the exact closest point on the triangle mesh and barycentric
interpolation of the vertex quantities, so R(x_s) varies continuously.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import open3d as o3d


def orthonormal_frames(t1: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Columns [t1, t2 = n x t1, n] with t1 projected into the plane of n."""
    n = n / np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12, None)
    t1 = t1 - np.sum(t1 * n, axis=-1, keepdims=True) * n
    t1 = t1 / np.clip(np.linalg.norm(t1, axis=-1, keepdims=True), 1e-12, None)
    t2 = np.cross(n, t1)
    return np.stack([t1, t2, n], axis=-1)


def curvature_features(S2: np.ndarray) -> np.ndarray:
    """2x2 shape operators in the [t1, t2] basis -> [k1, k2, a cos2th, a sin2th].

    k1 >= k2 are the principal curvatures, th the angle of the k1 direction from
    t1, and a = (k1 - k2) / 2. Weighting the doubled-angle direction by the
    anisotropy keeps it continuous where k1 ~ k2 (direction undefined).
    """
    s11, s12, s22 = S2[..., 0, 0], 0.5 * (S2[..., 0, 1] + S2[..., 1, 0]), S2[..., 1, 1]
    mean = 0.5 * (s11 + s22)
    a_cos = 0.5 * (s11 - s22)
    a_sin = s12
    a = np.sqrt(a_cos**2 + a_sin**2)
    return np.stack([mean + a, mean - a, a_cos, a_sin], axis=-1)


class FixBMoldField:
    def __init__(self, npz_path: Path):
        d = np.load(npz_path)
        self.vertices = d["vertices"].astype(np.float64)
        self.faces = d["faces"].astype(np.int64)
        self.normals = d["normals"].astype(np.float64)
        self.frames = d["frames"].astype(np.float64)
        self.curv_tensor = d["curv_tensor"].astype(np.float64)
        meta_path = Path(npz_path).with_suffix(".json")
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(self.vertices.astype(np.float32))
        mesh.triangle.indices = o3d.core.Tensor(self.faces.astype(np.int32))
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(mesh)

    def closest(self, p: np.ndarray):
        """Closest surface points and barycentric weights (Q,3), (Q,3) vertex ids."""
        q = o3d.core.Tensor(np.asarray(p, dtype=np.float32).reshape(-1, 3))
        ans = self.scene.compute_closest_points(q)
        tri = ans["primitive_ids"].numpy().astype(np.int64)
        uv = ans["primitive_uvs"].numpy().astype(np.float64)
        w = np.stack([1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]], axis=-1)
        vid = self.faces[tri]
        x_s = np.einsum("qk,qkd->qd", w, self.vertices[vid])
        return x_s, vid, w

    def query(self, p: np.ndarray, with_curvature: bool = False) -> Dict[str, np.ndarray]:
        p = np.asarray(p, dtype=np.float64)
        shape = p.shape[:-1]
        x_s, vid, w = self.closest(p)
        n = np.einsum("qk,qkd->qd", w, self.normals[vid])
        t1 = np.einsum("qk,qkd->qd", w, self.frames[vid][..., 0])
        R = orthonormal_frames(t1, n)
        sd = np.sum((p.reshape(-1, 3) - x_s) * R[..., 2], axis=-1)
        out = {
            "x_s": x_s.reshape(shape + (3,)),
            "R": R.reshape(shape + (3, 3)),
            "sd": sd.reshape(shape),
        }
        if with_curvature:
            S3 = np.einsum("qk,qkij->qij", w, self.curv_tensor[vid])
            B = R[..., :2]
            S2 = np.einsum("qia,qij,qjb->qab", B, S3, B)
            out["curv"] = curvature_features(S2).reshape(shape + (4,))
        return out

    def project(self, p: np.ndarray) -> np.ndarray:
        return self.closest(p)[0].reshape(np.asarray(p).shape)


def integrate_fixb(
    field: FixBMoldField,
    p0: np.ndarray,
    v_par: np.ndarray,
    delta_n: np.ndarray,
    dt: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Decode chunks: x_s(0) = foot of p0; x_d(k) = x_s(k) + dn(k) n(x_s(k));
    x_s(k+1) = foot of x_s(k) + R(x_s(k)) [v_par(k); 0] dt(k).

    p0 (Q,3), v_par (Q,H,2), delta_n (Q,H), dt (Q,H) (last column unused).
    Returns x_s, x_d (Q,H,3) and R (Q,H,3,3).
    """
    Q, H = delta_n.shape
    x_s = np.zeros((Q, H, 3))
    R = np.zeros((Q, H, 3, 3))
    cur = field.query(p0)
    xs, Rk = cur["x_s"], cur["R"]
    for k in range(H):
        x_s[:, k] = xs
        R[:, k] = Rk
        if k + 1 < H:
            step = np.einsum("qij,qj->qi", Rk[..., :2], v_par[:, k]) * dt[:, k, None]
            nxt = field.query(xs + step)
            xs, Rk = nxt["x_s"], nxt["R"]
    x_d = x_s + delta_n[..., None] * R[..., 2]
    return {"x_s": x_s, "x_d": x_d, "R": R}
