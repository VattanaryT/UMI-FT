"""Mold surface model for the Fix B datasets: mesh, frame field, curvature.

Saved by fixb_build_mold_field.py as ``fixb_field.npz`` (fixture frame):
  vertices (V,3), faces (F,3), normals (V,3) outward,
  frames (V,3,3) columns [t1, t2, n] from the diffused heat-gradient tangent
  (same construction as geodude_teleoperation.stringer_field),
  curv_tensor (V,3,3) smoothed tangent-plane shape operator (1/m, convex > 0).

Queries use the exact closest point on the triangle mesh and barycentric
interpolation of the vertex quantities. A time series keeps the previous face
unless the new one is closer by more than 5 mm or the tool axis has rotated
toward the new normal. A same-face tangent flip the tool did not make is held.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import open3d as o3d


# A Voronoi hop of a couple of millimetres is not a crown-to-wall move.
FACE_SWITCH_MARGIN_M = 0.005
TOOL_TOWARD_DEG = 5.0
SAME_FACE_DOT = 0.9063  # cos 25 deg
SAME_FACE_M = 0.015
FRAME_HOLD_DEG = 30.0


def _as_unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def geodesic_deg_one(a: np.ndarray, b: np.ndarray) -> float:
    rel = np.asarray(a, dtype=np.float64).T @ np.asarray(b, dtype=np.float64)
    c = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def tool_toward_deg(a_prev: np.ndarray, a_now: np.ndarray, n_new: np.ndarray) -> float:
    """Degrees by which the tool axis closed on the new face normal."""
    a_prev, a_now, n_new = _as_unit(a_prev), _as_unit(a_now), _as_unit(n_new)

    def _ang(a, n):
        return float(np.degrees(np.arccos(np.clip(float(np.dot(a, n)), -1.0, 1.0))))

    return _ang(a_prev, n_new) - _ang(a_now, n_new)


def accept_new_face(
    d_prev: float,
    d_new: float,
    a_prev: np.ndarray | None = None,
    a_now: np.ndarray | None = None,
    n_new: np.ndarray | None = None,
    margin_m: float = FACE_SWITCH_MARGIN_M,
    toward_deg: float = TOOL_TOWARD_DEG,
) -> tuple[bool, str]:
    """Keep the previous face unless the new one is clearly closer, or the tool tilted onto it."""
    if float(d_prev) - float(d_new) > float(margin_m):
        return True, "margin"
    if a_prev is None or a_now is None or n_new is None:
        return False, "hold"
    if tool_toward_deg(a_prev, a_now, n_new) >= toward_deg:
        return True, "rotation"
    return False, "hold"


def pressing_axis(R: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Tool body axis, sign included, that best matches the face normal."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    n = _as_unit(n)
    best, score = R[:, 2], -np.inf
    for i in range(3):
        for sign in (1.0, -1.0):
            axis = sign * R[:, i]
            s = float(np.dot(axis, n))
            if s > score:
                best, score = axis, s
    return best


def slew_frame(R_from: np.ndarray, R_to: np.ndarray, max_deg: float = 25.0) -> np.ndarray:
    """Move R_from toward R_to by at most max_deg. A seam crossing is then several steps, not one."""
    R_from = np.asarray(R_from, dtype=np.float64).reshape(3, 3)
    R_to = np.asarray(R_to, dtype=np.float64).reshape(3, 3)
    rel = R_from.T @ R_to
    c = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    ang = float(np.degrees(np.arccos(c)))
    if ang <= max_deg or ang < 1e-6:
        return R_to
    v = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
    axis = v / max(float(np.linalg.norm(v)), 1e-12)
    th = np.radians(max_deg)
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    dR = np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)
    return R_from @ dR


def unexplained_deg(R_prev, R_new, R_tool_prev=None, R_tool_now=None) -> float:
    """Field rotation left over after removing the tool's own rotation."""
    Rf = np.asarray(R_prev, dtype=np.float64).T @ np.asarray(R_new, dtype=np.float64)
    if R_tool_prev is None or R_tool_now is None:
        Rt = np.eye(3)
    else:
        Rt = np.asarray(R_tool_prev, dtype=np.float64).T @ np.asarray(R_tool_now, dtype=np.float64)
    return geodesic_deg_one(Rf, Rt)


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

    def _closest_frame(
        self, p: np.ndarray, with_curvature: bool = False, R_override: np.ndarray | None = None
    ) -> Dict[str, np.ndarray]:
        p = np.asarray(p, dtype=np.float64)
        shape = p.shape[:-1]
        x_s, vid, w = self.closest(p)
        n = np.einsum("qk,qkd->qd", w, self.normals[vid])
        t1 = np.einsum("qk,qkd->qd", w, self.frames[vid][..., 0])
        R = orthonormal_frames(t1, n)
        if R_override is not None:
            R = np.asarray(R_override, dtype=np.float64).reshape(R.shape)
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

    def _decide(self, p, p_prev, x_prev, R_prev, x_free, R_free, R_tool_prev, R_tool_now):
        p = np.asarray(p, dtype=np.float64).reshape(3)
        p_prev = np.asarray(p_prev, dtype=np.float64).reshape(3)
        x_prev = np.asarray(x_prev, dtype=np.float64).reshape(3)
        x_free = np.asarray(x_free, dtype=np.float64).reshape(3)
        R_prev = np.asarray(R_prev, dtype=np.float64).reshape(3, 3)
        R_free = np.asarray(R_free, dtype=np.float64).reshape(3, 3)
        n_prev, n_free = R_prev[:, 2], R_free[:, 2]
        same = float(np.dot(n_free, n_prev)) > SAME_FACE_DOT and float(np.linalg.norm(x_free - x_prev)) < SAME_FACE_M
        if same:
            if unexplained_deg(R_prev, R_free, R_tool_prev, R_tool_now) > FRAME_HOLD_DEG:
                return x_free, orthonormal_frames(R_prev[:, 0], n_free), 3
            return x_free, R_free, 0
        # Walk toward the tool on the face we already have. A hat-section hop is only
        # ~1 mm closer than that walk; the 5 mm rule holds it.
        x_stay, R_stay = self._stay_on_face(p, x_prev, R_prev, R_tool_prev, R_tool_now)
        d_prev = float(np.linalg.norm(p - x_stay))
        d_free = float(np.linalg.norm(p - x_free))
        a_prev = a_now = None
        if R_tool_prev is not None and R_tool_now is not None:
            a_prev = pressing_axis(R_tool_prev, n_prev)
            a_now = np.asarray(R_tool_now, dtype=np.float64).reshape(3, 3) @ (
                np.asarray(R_tool_prev, dtype=np.float64).reshape(3, 3).T @ a_prev
            )
        switch, why = accept_new_face(d_prev, d_free, a_prev, a_now, n_free)
        if switch:
            return x_free, R_free, 1 if why == "margin" else 2
        return x_stay, R_stay, 3

    def _stay_on_face(self, p, x_prev, R_prev, R_tool_prev, R_tool_now):
        """Closest point on the current face along the tool's tangential motion."""
        n_prev = np.asarray(R_prev, dtype=np.float64).reshape(3, 3)[:, 2]
        delta = np.asarray(p, dtype=np.float64).reshape(3) - np.asarray(x_prev, dtype=np.float64).reshape(3)
        tang = delta - np.dot(delta, n_prev) * n_prev
        dist = float(np.linalg.norm(tang))
        x = np.asarray(x_prev, dtype=np.float64).reshape(3)
        R = np.asarray(R_prev, dtype=np.float64).reshape(3, 3)
        if dist >= 1e-4:
            steps = min(40, int(np.ceil(dist / 0.002)))
            step = tang / steps
            for _ in range(steps):
                guess = x + step
                held = self._closest_frame(guess.reshape(1, 3))
                x_h, R_h = held["x_s"][0], held["R"][0]
                if float(np.dot(R_h[:, 2], R[:, 2])) <= SAME_FACE_DOT or float(np.linalg.norm(x_h - guess)) >= 0.008:
                    break
                if float(np.linalg.norm(p - x_h)) > float(np.linalg.norm(p - x)) + 1e-9:
                    break
                x, R = x_h, R_h
        if unexplained_deg(R_prev, R, R_tool_prev, R_tool_now) > FRAME_HOLD_DEG:
            R = orthonormal_frames(np.asarray(R_prev, dtype=np.float64).reshape(3, 3)[:, 0], R[:, 2])
        return x, R

    def track_step(self, p, p_prev, x_prev, R_prev):
        """One decode step for Q independent feet. No tool rotation, so only the 5 mm rule."""
        p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
        p_prev = np.asarray(p_prev, dtype=np.float64).reshape(-1, 3)
        x_prev = np.asarray(x_prev, dtype=np.float64).reshape(-1, 3)
        R_prev = np.asarray(R_prev, dtype=np.float64).reshape(-1, 3, 3)
        free = self._closest_frame(p)
        x = np.zeros_like(p)
        R = np.zeros_like(R_prev)
        for i in range(len(p)):
            x[i], R_face, _ = self._decide(
                p[i], p_prev[i], x_prev[i], R_prev[i], free["x_s"][i], free["R"][i], None, None
            )
            R[i] = slew_frame(R_prev[i], R_face)
        return x, R

    def track_foot(
        self,
        p,
        R_tool=None,
        x_seed=None,
        R_seed=None,
        R_tool_seed=None,
        p_seed=None,
        R_out_seed=None,
    ):
        """Time series. reason: 0 same face, 1 closer by >5 mm, 2 tool tilted onto the face, 3 held.

        R_seed is the face the next decision compares against. R_out_seed is the
        last published frame, so a later call slews from that frame. p_seed is
        the previous tool position. Labels call this once per episode with no seed.
        """
        p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
        n = len(p)
        free = self._closest_frame(p) if n else {"x_s": p, "R": np.zeros((0, 3, 3))}
        x = np.zeros((n, 3))
        R = np.zeros((n, 3, 3))
        reason = np.zeros(n, np.int8)
        R_tool = None if R_tool is None else np.asarray(R_tool, dtype=np.float64).reshape(n, 3, 3)
        if x_seed is None:
            if n == 0:
                return {"x_s": x, "R": R, "reason": reason, "R_face": np.eye(3)}
            x[0], R[0] = free["x_s"][0], free["R"][0]
            p_prev, x_prev, R_face, R_out = p[0], x[0], R[0], R[0]
            start = 1
        else:
            p_prev = p[0] if p_seed is None else np.asarray(p_seed, dtype=np.float64).reshape(3)
            x_prev = np.asarray(x_seed, dtype=np.float64).reshape(3)
            R_face = np.eye(3) if R_seed is None else np.asarray(R_seed, dtype=np.float64).reshape(3, 3)
            R_out = R_face if R_out_seed is None else np.asarray(R_out_seed, dtype=np.float64).reshape(3, 3)
            start = 0
        for i in range(start, n):
            if R_tool is None:
                Rt_prev = Rt_now = None
            elif i == 0:
                Rt_prev = None if R_tool_seed is None else np.asarray(R_tool_seed, dtype=np.float64).reshape(3, 3)
                Rt_now = None if Rt_prev is None else R_tool[0]
            else:
                Rt_prev, Rt_now = R_tool[i - 1], R_tool[i]
            x[i], R_face, reason[i] = self._decide(
                p[i], p_prev, x_prev, R_face, free["x_s"][i], free["R"][i], Rt_prev, Rt_now
            )
            R_out = slew_frame(R_out, R_face)
            R[i] = R_out
            p_prev, x_prev = p[i], x[i]
        return {"x_s": x, "R": R, "reason": reason, "R_face": np.asarray(R_face, dtype=np.float64).reshape(3, 3)}

    def query(
        self,
        p: np.ndarray,
        with_curvature: bool = False,
        R_tool: np.ndarray | None = None,
        prev_x: np.ndarray | None = None,
        prev_R: np.ndarray | None = None,
        prev_R_tool: np.ndarray | None = None,
        prev_p: np.ndarray | None = None,
        prev_R_out: np.ndarray | None = None,
    ) -> Dict[str, np.ndarray]:
        p = np.asarray(p, dtype=np.float64)
        flat = p.reshape(-1, 3)
        track = prev_x is not None or (R_tool is not None and len(flat) > 1)
        if not track:
            return self._closest_frame(p, with_curvature=with_curvature)
        tracked = self.track_foot(
            flat,
            R_tool=R_tool,
            x_seed=prev_x,
            R_seed=prev_R,
            R_tool_seed=prev_R_tool,
            p_seed=prev_p,
            R_out_seed=prev_R_out,
        )
        geom = self._closest_frame(tracked["x_s"], with_curvature=with_curvature, R_override=tracked["R"])
        shape = p.shape[:-1]
        nrm = tracked["R"][:, :, 2]
        sd = np.sum((flat - tracked["x_s"]) * nrm, axis=1)
        geom["x_s"] = tracked["x_s"].reshape(shape + (3,))
        geom["R"] = tracked["R"].reshape(shape + (3, 3))
        geom["sd"] = sd.reshape(shape)
        geom["reason"] = tracked["reason"]
        geom["R_face"] = tracked["R_face"]
        return geom

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
            guess = xs + step
            if hasattr(field, "track_step"):
                xs, Rk = field.track_step(guess, xs, xs, Rk)
            else:
                nxt = field.query(guess)
                xs, Rk = nxt["x_s"], nxt["R"]
    x_d = x_s + delta_n[..., None] * R[..., 2]
    return {"x_s": x_s, "x_d": x_d, "R": R}
