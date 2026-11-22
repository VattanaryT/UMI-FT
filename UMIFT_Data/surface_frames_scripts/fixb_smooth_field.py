"""Position-only mold orientation. Same frames for r4vt labels and decode.

The mesh and fixb_field.npz stay as they are. This writes a sidecar of
per-vertex frames whose z is the inward normal, so a typical press has ΔR
near the identity. δn stays on the outward normal.

Smoothing radius is a support ball, area-weighted. 25 mm is smaller than the
crown-to-wall fillet, so the ball is 40 mm unless the measured fillet is wider.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from fixb_field import FixBMoldField, curvature_features, orthonormal_frames

# Cross-section fillet is ~35 mm. A 25 mm ball leaves that corner sharp.
SMOOTH_RADIUS_M = 0.040
VOXEL_FRACTION = 8.0


def vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    b = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    area = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
    w = np.zeros(len(vertices))
    for k in range(3):
        np.add.at(w, faces[:, k], area / 3.0)
    return w


def mesh_edges(faces: np.ndarray) -> np.ndarray:
    e = np.stack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.sort(e.reshape(-1, 2), axis=1)
    return np.unique(e, axis=0)


def fillet_radius_m(vertices: np.ndarray, normals: np.ndarray, curv: np.ndarray) -> float:
    """Median crown-to-wall radius (m) from cross-section curvature."""
    k = np.sort(np.abs(np.linalg.eigvalsh(curv)), axis=1)[:, -1]
    up = np.array([0.0, -1.0, 0.0])
    axis = _length_axis(vertices)
    s = (vertices - vertices.mean(0)) @ axis
    lo, hi = np.percentile(s, [20, 80])
    radii = []
    for station in np.linspace(lo, hi, 5):
        slab = np.abs(s - station) < 0.004
        if int(slab.sum()) < 30:
            continue
        ang = np.degrees(np.arccos(np.clip(normals[slab] @ up, -1.0, 1.0)))
        band = (ang > 20.0) & (ang < 75.0) & (k[slab] > 1e-3)
        if int(band.sum()) < 8:
            continue
        radii.append(float(np.median(1.0 / k[slab][band])))
    if not radii:
        raise RuntimeError("no crown-to-wall fillet samples")
    return float(np.median(radii))


def _length_axis(vertices: np.ndarray) -> np.ndarray:
    cov = np.cov((vertices - vertices.mean(0)).T)
    _, evecs = np.linalg.eigh(cov)
    return evecs[:, -1]


def smooth_normals(vertices, normals, areas, radius: float) -> np.ndarray:
    """Area-weighted average of normals inside ``radius`` of each vertex."""
    voxel = float(radius) / VOXEL_FRACTION
    g = np.floor(vertices / voxel).astype(np.int64)
    g -= g.min(0)
    span = g.max(0) + 1
    code = g[:, 0] + span[0] * (g[:, 1] + span[1] * g[:, 2])
    uniq, inv = np.unique(code, return_inverse=True)
    n_c = len(uniq)
    acc_n = np.zeros((n_c, 3))
    acc_a = np.zeros(n_c)
    np.add.at(acc_n, inv, normals * areas[:, None])
    np.add.at(acc_a, inv, areas)
    acc_a = np.maximum(acc_a, 1e-12)
    coarse_n = acc_n / np.linalg.norm(acc_n, axis=1, keepdims=True)
    cent = np.zeros((n_c, 3))
    np.add.at(cent, inv, vertices * areas[:, None])
    cent /= acc_a[:, None]

    tree = cKDTree(cent)
    nbrs = tree.query_ball_point(cent, radius)
    blurred = np.zeros_like(coarse_n)
    for i, idx in enumerate(nbrs):
        if len(idx) == 0:
            blurred[i] = coarse_n[i]
            continue
        w = acc_a[idx]
        n = (coarse_n[idx] * w[:, None]).sum(0)
        n /= max(float(np.linalg.norm(n)), 1e-12)
        if float(np.dot(n, coarse_n[i])) < 0.0:
            n = -n
        blurred[i] = n

    k = min(8, n_c)
    dist, ix = tree.query(vertices, k=k, distance_upper_bound=radius)
    dist = np.asarray(dist, dtype=np.float64)
    ix = np.asarray(ix)
    if k == 1:
        dist = dist[:, None]
        ix = ix[:, None]
    ok = np.isfinite(dist) & (ix < n_c)
    w = np.where(ok, acc_a[np.clip(ix, 0, n_c - 1)] / np.maximum(dist, voxel * 0.25) ** 2, 0.0)
    picked = blurred[np.clip(ix, 0, n_c - 1)]
    n = np.sum(picked * w[:, :, None], axis=1)
    weak = np.linalg.norm(n, axis=1) < 1e-8
    n[weak] = normals[weak]
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    flip = np.sum(n * normals, axis=1) < 0.0
    n[flip] *= -1.0
    return n


def propagate_t1(t1: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Flip tangents so each vertex agrees with the first neighbor that reached it."""
    t1 = np.array(t1, dtype=np.float64, copy=True)
    adj: list[list[int]] = [[] for _ in range(len(t1))]
    for i, j in edges:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    seen = np.zeros(len(t1), dtype=bool)
    order = np.argsort(-np.linalg.norm(t1, axis=1))
    for seed in order:
        seed = int(seed)
        if seen[seed] or float(np.linalg.norm(t1[seed])) < 1e-8:
            continue
        seen[seed] = True
        q = deque([seed])
        while q:
            i = q.popleft()
            for j in adj[i]:
                if seen[j]:
                    continue
                if float(np.dot(t1[i], t1[j])) < 0.0:
                    t1[j] *= -1.0
                seen[j] = True
                q.append(j)
    return t1


def neighbor_t1_deg(t1: np.ndarray, edges: np.ndarray) -> float:
    a = t1[edges[:, 0]]
    b = t1[edges[:, 1]]
    c = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)).max()) if len(c) else 0.0


def project_tangent(t1: np.ndarray, normal: np.ndarray) -> np.ndarray:
    t1 = t1 - np.sum(t1 * normal, axis=1, keepdims=True) * normal
    nrm = np.linalg.norm(t1, axis=1, keepdims=True)
    bad = nrm[:, 0] < 1e-8
    t1[bad] = 0.0
    t1[~bad] /= nrm[~bad]
    return t1


def circular_mean(angle: np.ndarray) -> float:
    a = np.asarray(angle, dtype=np.float64).reshape(-1)
    return float(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))


# A closest point on the same face can sit on a tip vortex. Its normal still
# agrees, but t1 has spun. Carry the previous tangent; do not take that frame.
NORMAL_AGREE_COS = float(np.cos(np.deg2rad(25.0)))
TANGENT_SPIN_HOLD_DEG = 30.0


def tangent_spin_deg(R_prev: np.ndarray, R_new: np.ndarray) -> float:
    """Signed angle of t1 about the new normal, in degrees."""
    R_prev = np.asarray(R_prev, dtype=np.float64).reshape(3, 3)
    R_new = np.asarray(R_new, dtype=np.float64).reshape(3, 3)
    n = R_new[:, 2]
    old = R_prev[:, 0] - np.dot(R_prev[:, 0], n) * n
    new = R_new[:, 0] - np.dot(R_new[:, 0], n) * n
    old_n = float(np.linalg.norm(old))
    new_n = float(np.linalg.norm(new))
    if old_n < 1e-8 or new_n < 1e-8:
        return 0.0
    old = old / old_n
    new = new / new_n
    return float(np.degrees(np.arctan2(np.dot(n, np.cross(old, new)), np.dot(old, new))))


def tangent_on_normal(seed_R: np.ndarray, queried_R: np.ndarray) -> np.ndarray:
    """Queried normal, with t1 taken from ``seed_R`` when it has a tangent part."""
    queried_R = np.asarray(queried_R, dtype=np.float64).reshape(3, 3)
    n = queried_R[:, 2]
    t1 = np.asarray(seed_R, dtype=np.float64).reshape(3, 3)[:, 0]
    t1 = t1 - np.dot(t1, n) * n
    if float(np.linalg.norm(t1)) < 1e-8:
        return queried_R.copy()
    return orthonormal_frames(t1, n)


def hold_tangent_through_spin(R_prev: np.ndarray, R_new: np.ndarray) -> np.ndarray:
    """Keep the previous tangent when the new frame only spins about the normal."""
    R_prev = np.asarray(R_prev, dtype=np.float64).reshape(3, 3)
    R_new = np.asarray(R_new, dtype=np.float64).reshape(3, 3)
    if float(np.dot(R_prev[:, 2], R_new[:, 2])) < NORMAL_AGREE_COS:
        return R_new.copy()
    if abs(tangent_spin_deg(R_prev, R_new)) <= TANGENT_SPIN_HOLD_DEG:
        return R_new.copy()
    return tangent_on_normal(R_prev, R_new)


def rotate_t1(t1: np.ndarray, normal: np.ndarray, theta: float) -> np.ndarray:
    t2 = np.cross(normal, t1)
    out = np.cos(theta) * t1 + np.sin(theta) * t2
    return project_tangent(out, normal)


class SmoothFrameField(FixBMoldField):
    """Closest-point foot, orientation from the sidecar. z points into the mold."""

    orientation_z_inward = True

    def __init__(self, npz_path: Path, frames_path: Path):
        super().__init__(Path(npz_path))
        blob = np.load(frames_path)
        self._ori = np.asarray(blob["frames"], dtype=np.float64)
        if self._ori.shape != (len(self.vertices), 3, 3):
            raise RuntimeError(
                f"smooth frames {self._ori.shape} do not match {len(self.vertices)} vertices"
            )
        self.smooth_radius_m = float(blob["radius_m"]) if "radius_m" in blob.files else SMOOTH_RADIUS_M

    def query(
        self,
        p: np.ndarray,
        with_curvature: bool = False,
        R_tool=None,
        prev_x=None,
        prev_R=None,
        prev_R_tool=None,
        prev_p=None,
        prev_R_out=None,
    ):
        # History is ignored on purpose: labels and decode both call this.
        del R_tool, prev_x, prev_R, prev_R_tool, prev_p, prev_R_out
        p = np.asarray(p, dtype=np.float64)
        shape = p.shape[:-1]
        flat = p.reshape(-1, 3)
        x_s, vid, w = self.closest(flat)
        # Last axis is the column. [:, :, 0] is a row and scrambles z.
        t1 = np.einsum("qk,qkd->qd", w, self._ori[vid][..., 0])
        z = np.einsum("qk,qkd->qd", w, self._ori[vid][..., 2])
        R = orthonormal_frames(t1, z)
        n_out = -R[:, :, 2]
        sd = np.sum((flat - x_s) * n_out, axis=1)
        out = {
            "x_s": x_s.reshape(shape + (3,)),
            "R": R.reshape(shape + (3, 3)),
            "sd": sd.reshape(shape),
            "n_out": n_out.reshape(shape + (3,)),
            "R_face": R[-1] if len(R) else np.eye(3),
            "reason": np.zeros(len(flat), dtype=np.int8),
        }
        if with_curvature:
            S3 = np.einsum("qk,qkij->qij", w, self.curv_tensor[vid])
            B = R[..., :2]
            S2 = np.einsum("qia,qij,qjb->qab", B, S3, B)
            out["curv"] = curvature_features(S2).reshape(shape + (4,))
        return out

    def track_step(self, p, p_prev, x_prev, R_prev):
        """Closest point, but a spin about an agreeing normal keeps the tangent.

        The position hop is returned as queried. The 10 mm hop hold lives in
        the caller, not here.
        """
        del p_prev, x_prev
        q = self.query(p)
        x = np.asarray(q["x_s"], dtype=np.float64).reshape(-1, 3)
        R = np.asarray(q["R"], dtype=np.float64).reshape(-1, 3, 3)
        if R_prev is None:
            return x, R
        prev = np.asarray(R_prev, dtype=np.float64).reshape(-1, 3, 3)
        out = np.empty_like(R)
        for i in range(len(R)):
            out[i] = hold_tangent_through_spin(prev[0 if len(prev) == 1 else i], R[i])
        return x, out

    def track_foot(self, p, R_tool=None, x_seed=None, R_seed=None, R_tool_seed=None, p_seed=None, R_out_seed=None):
        del R_tool, x_seed, R_seed, R_tool_seed, p_seed, R_out_seed
        q = self.query(p)
        R = q["R"].reshape(-1, 3, 3)
        return {
            "x_s": q["x_s"].reshape(-1, 3),
            "R": R,
            "reason": np.zeros(len(R), dtype=np.int8),
            "R_face": R[-1] if len(R) else np.eye(3),
        }


def build_smooth_frames(
    field_npz: Path,
    out_npz: Path,
    *,
    tangent_samples: np.ndarray | None = None,
    tool_x: np.ndarray | None = None,
) -> dict:
    """Write the sidecar. ``tangent_samples`` are (position, tool x) rows used to aim t1.

    tool_x is (N, 3) at the same rows as tangent_samples (N, 3). Either both or neither.
    """
    if out_npz.exists():
        raise SystemExit(f"refusing to overwrite {out_npz}")
    field = FixBMoldField(field_npz)
    fillet = fillet_radius_m(field.vertices, field.normals, field.curv_tensor)
    radius = SMOOTH_RADIUS_M
    if radius <= fillet:
        radius = float(np.ceil(fillet * 1.15 / 0.005) * 0.005)
    areas = vertex_areas(field.vertices, field.faces)
    outward = smooth_normals(field.vertices, field.normals, areas, radius)
    t1 = project_tangent(field.frames[:, :, 0], outward)
    missing = np.linalg.norm(t1, axis=1) < 1e-8
    if np.any(missing):
        t1[missing] = project_tangent(field.frames[missing, :, 1], outward[missing])
    edges = mesh_edges(field.faces)
    # Keep the heat-gradient signs. Re-propagating from one seed cuts a long
    # seam across the mold. The source field's remaining flips are tip vortices.
    theta = 0.0
    strength = 1.0
    if tangent_samples is not None:
        if tool_x is None:
            raise ValueError("tool_x is required with tangent_samples")
        theta, strength = _t1_align_angle(field, t1, outward, tangent_samples, tool_x)
        t1 = rotate_t1(t1, outward, theta)
    seam = _seam_report(t1, edges, field.vertices)
    frames = orthonormal_frames(t1, -outward).astype(np.float32)
    np.savez_compressed(
        out_npz,
        frames=frames,
        radius_m=np.float64(radius),
        fillet_radius_m=np.float64(fillet),
        t1_align_rad=np.float64(theta),
        t1_neighbor_max_deg=np.float64(seam["t1_neighbor_max_deg"]),
    )
    return {
        "radius_m": radius,
        "fillet_radius_m": fillet,
        "t1_align_deg": float(np.degrees(theta)),
        "t1_axis_strength": strength,
        **seam,
        "path": str(out_npz),
    }


def _seam_report(t1: np.ndarray, edges: np.ndarray, vertices: np.ndarray) -> dict:
    c = np.clip(np.sum(t1[edges[:, 0]] * t1[edges[:, 1]], axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(c))
    x = vertices[:, 0]
    keep = (x > x.min() + 0.04) & (x < x.max() - 0.04)
    interior = keep[edges[:, 0]] & keep[edges[:, 1]]
    return {
        "t1_neighbor_max_deg": float(ang.max()) if len(ang) else 0.0,
        "t1_neighbor_p99_deg": float(np.percentile(ang, 99)) if len(ang) else 0.0,
        "t1_edges_over_90": int(np.sum(ang > 90.0)),
        "t1_neighbor_max_deg_interior": float(ang[interior].max()) if np.any(interior) else 0.0,
    }


def _t1_align_angle(field, t1, outward, positions, tool_x) -> tuple[float, float]:
    x_s, vid, w = field.closest(np.asarray(positions, dtype=np.float64))
    del x_s
    n = np.einsum("qk,qkd->qd", w, outward[vid])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    t = np.einsum("qk,qkd->qd", w, t1[vid])
    t = project_tangent(t, n)
    t2 = np.cross(n, t)
    tool = np.asarray(tool_x, dtype=np.float64)
    xp = tool - np.sum(tool * n, axis=1, keepdims=True) * n
    keep = np.linalg.norm(xp, axis=1) > 1e-6
    ang = np.arctan2(np.sum(t2[keep] * xp[keep], axis=1), np.sum(t[keep] * xp[keep], axis=1))
    if len(ang) == 0:
        raise RuntimeError("no demo samples to aim t1")
    # Demos run both ways, so 0° and 180° are the same tangent axis.
    s2, c2 = float(np.mean(np.sin(2.0 * ang))), float(np.mean(np.cos(2.0 * ang)))
    theta = 0.5 * float(np.arctan2(s2, c2))
    strength = float(np.hypot(s2, c2))
    shifted = np.arctan2(np.sin(ang - theta), np.cos(ang - theta))
    if float(np.mean(np.cos(shifted))) < 0.0:
        theta += np.pi
    return theta, strength


def _demo_grid() -> None:
    """A corner of two planes. After a wide smooth, a 5 mm shift is not a 90° flip."""
    xs = np.linspace(-0.05, 0.0, 11)
    zs = np.linspace(0.0, 0.05, 11)
    floor = np.stack([np.repeat(xs, len(zs)), np.zeros(len(xs) * len(zs)), np.tile(zs, len(xs))], axis=1)
    wall_z = np.linspace(0.0, 0.05, 11)
    wall_x = np.linspace(0.0, 0.05, 6)
    wall = np.stack(
        [np.repeat(wall_x, len(wall_z)), np.zeros(len(wall_x) * len(wall_z)), np.tile(wall_z, len(wall_x))],
        axis=1,
    )
    pts = np.concatenate([floor, wall], axis=0)
    nrm = np.concatenate(
        [np.tile([0.0, 0.0, 1.0], (len(floor), 1)), np.tile([1.0, 0.0, 0.0], (len(wall), 1))],
        axis=0,
    )
    area = np.ones(len(pts))
    sm = smooth_normals(pts, nrm, area, 0.04)
    tree = cKDTree(pts)

    def at(p):
        d, i = tree.query(p, k=8, distance_upper_bound=0.04)
        ok = np.isfinite(d)
        w = np.where(ok, 1.0 / np.maximum(d, 1e-4) ** 2, 0.0)
        n = (sm[np.clip(i, 0, len(sm) - 1)] * w[:, None]).sum(0)
        return n / np.linalg.norm(n)

    a = at(np.array([-0.005, 0.0, 0.005]))
    b = at(np.array([0.005, 0.0, 0.005]))
    turn = float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))
    assert turn < 25.0, turn
    # Planted sign flip on a line comes back under 1°.
    t1 = np.tile([1.0, 0.0, 0.0], (5, 1))
    t1[3:] *= -1.0
    edges = np.array([[i, i + 1] for i in range(4)])
    fixed = propagate_t1(t1, edges)
    assert neighbor_t1_deg(fixed, edges) < 1.0


if __name__ == "__main__":
    _demo_grid()
    print("smooth field self-check ok")
