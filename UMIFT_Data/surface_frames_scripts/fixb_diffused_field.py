"""Off-surface orientation by the Laplace equation Bilaloglu et al. solve with Walk on Spheres.

The smooth on-surface frames are the Dirichlet boundary. A grid in the box
around the mold relaxes the quaternion components to a harmonic field, which is
the same PDE. Lookup is trilinear, so a query moves continuously as the TCP
moves and does not hop from crown to wall.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def relax_quaternions(q: np.ndarray, free: np.ndarray, n_iter: int = 400, tol: float = 1e-4) -> int:
    """Component-wise Jacobi for Laplace's equation. ``free`` is not updated on the border.

    ``q`` is (nx, ny, nz, 4), xyzw. Fixed cells stay put. Returns the iteration
    that reached ``tol``, or ``n_iter`` if it did not.
    """
    q = np.asarray(q)
    free = np.asarray(free, dtype=bool)
    if free.shape != q.shape[:-1]:
        raise ValueError("free mask does not match the grid")
    if np.any(free[0]) or np.any(free[-1]) or np.any(free[:, 0]) or np.any(free[:, -1]):
        raise ValueError("the border has to stay fixed; roll wraps")
    if np.any(free[:, :, 0]) or np.any(free[:, :, -1]):
        raise ValueError("the border has to stay fixed; roll wraps")
    done = n_iter
    for it in range(n_iter):
        acc = np.zeros_like(q)
        for axis in range(3):
            acc += np.roll(q, 1, axis=axis)
            acc += np.roll(q, -1, axis=axis)
        acc *= 1.0 / 6.0
        delta = float(np.max(np.abs(acc[free] - q[free]))) if np.any(free) else 0.0
        q[free] = acc[free]
        if delta < tol:
            done = it + 1
            break
    return done


def _sample(grid: np.ndarray, origin: np.ndarray, cell: float, p: np.ndarray) -> np.ndarray:
    """Trilinear quaternion sample, signs aligned to the low corner, then normalized."""
    g = (np.asarray(p, dtype=np.float64).reshape(-1, 3) - origin) / cell
    hi = np.array(grid.shape[:3], dtype=np.int64) - 2
    i0 = np.clip(np.floor(g).astype(np.int64), 0, hi)
    f = np.clip(g - i0, 0.0, 1.0)
    acc = np.zeros((len(i0), 4), dtype=np.float64)
    base = grid[i0[:, 0], i0[:, 1], i0[:, 2]]
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                corner = grid[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
                flip = np.sum(corner * base, axis=1) < 0.0
                corner = np.where(flip[:, None], -corner, corner)
                w = (
                    np.where(dx == 0, 1.0 - f[:, 0], f[:, 0])
                    * np.where(dy == 0, 1.0 - f[:, 1], f[:, 1])
                    * np.where(dz == 0, 1.0 - f[:, 2], f[:, 2])
                )
                acc += w[:, None] * corner
    nrm = np.linalg.norm(acc, axis=1, keepdims=True)
    acc /= np.maximum(nrm, 1e-12)
    return acc


class DiffusedOrientationField:
    """Harmonic extension of a smooth frame field. Query returns R with z inward."""

    def __init__(self, grid: np.ndarray, origin: np.ndarray, cell: float):
        self.grid = np.asarray(grid, dtype=np.float32)
        self.origin = np.asarray(origin, dtype=np.float64).reshape(3)
        self.cell = float(cell)

    def quaternions(self, p: np.ndarray) -> np.ndarray:
        return _sample(self.grid, self.origin, self.cell, p)

    def rotations(self, p: np.ndarray) -> np.ndarray:
        q = self.quaternions(p)
        return Rotation.from_quat(q).as_matrix()


def build_diffused_field(field, cell: float = 0.004, pad: float = 0.04, band: float = 0.05) -> DiffusedOrientationField:
    """Dirichlet shell on the smooth frames, harmonic fill out to ``band`` meters."""
    lo = field.vertices.min(0) - pad
    hi = field.vertices.max(0) + pad
    axes = [np.arange(lo[d], hi[d] + cell * 0.5, cell) for d in range(3)]
    shape = tuple(len(a) for a in axes)
    print(f"[dof] grid {shape} cell {cell}", flush=True)
    origin = np.array([a[0] for a in axes], dtype=np.float64)
    coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    R = np.empty((len(coords), 3, 3), dtype=np.float64)
    sd = np.empty(len(coords), dtype=np.float64)
    for i in range(0, len(coords), 200000):
        sl = slice(i, i + 200000)
        q = field.query(coords[sl])
        R[sl] = q["R"].reshape(-1, 3, 3)
        sd[sl] = np.asarray(q["sd"], dtype=np.float64).reshape(-1)
        print(f"[dof] closest {min(i + 200000, len(coords))}/{len(coords)}", flush=True)
    quat = Rotation.from_matrix(R.reshape(-1, 3, 3)).as_quat().astype(np.float32)
    grid = quat.reshape(shape + (4,))
    sd = sd.reshape(shape)
    fixed = np.abs(sd) <= cell * 0.75
    surface = np.flatnonzero(fixed.reshape(-1))
    if len(surface) == 0:
        raise RuntimeError("no surface cells in the diffused grid")
    ref = grid.reshape(-1, 4)[int(surface[0])]
    dots = np.sum(grid * ref.reshape(1, 1, 1, 4), axis=-1)
    grid[dots < 0.0] *= -1.0
    free = (np.abs(sd) < band) & ~fixed
    free[0] = free[-1] = False
    free[:, 0] = free[:, -1] = False
    free[:, :, 0] = free[:, :, -1] = False
    steps = relax_quaternions(grid, free)
    print(f"[dof] relax steps {steps} fixed {int(fixed.sum())} free {int(free.sum())}", flush=True)
    return DiffusedOrientationField(grid, origin, cell)
