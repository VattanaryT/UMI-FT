"""Keep the current foot face until a new face wins for several steps.

Off unless ``UMIFT_FOOT_FACE_HYST`` is ``1``. ``UMIFT_FOOT_HYST_N`` is the
number of consecutive steps (default 1). ``UMIFT_FOOT_HYST_D_MM`` is how much
closer, in millimetres, the new face must be (default 5).

A replay can bind the chunk's stored ``mold_x_s`` rows. The first step then
starts on that point's face, and a later switch is refused when the stored
foot is still on the old face.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from causal_foot import _closest_on_triangles
from fixb_field import orthonormal_frames
from sliding_foot import face_tags

_FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
_STATE = None
_CHUNK = None
_CHUNK_R = None
_CURSOR = 0


def enabled() -> bool:
    return os.environ.get("UMIFT_FOOT_FACE_HYST", "").strip().lower() in ("1", "true", "yes")


def _n_steps() -> int:
    raw = os.environ.get("UMIFT_FOOT_HYST_N", "1").strip()
    n = int(raw)
    if n < 1:
        raise ValueError(f"UMIFT_FOOT_HYST_N={raw!r} must be >= 1")
    return n


def _d_m() -> float:
    raw = os.environ.get("UMIFT_FOOT_HYST_D_MM", "5").strip()
    d = float(raw) / 1000.0
    if d < 0.0:
        raise ValueError(f"UMIFT_FOOT_HYST_D_MM={raw!r} must be >= 0")
    return d


class _Hold:
    def __init__(self, field, n: int, d_m: float):
        self.n = int(n)
        self.d = float(d_m)
        self.field = field
        self.v = np.asarray(field.vertices, dtype=np.float64)
        self.f = np.asarray(field.faces, dtype=np.int64)
        self.n_in = np.asarray(field._ori[:, :, 2], dtype=np.float64)
        self.t1 = np.asarray(field._ori[:, :, 0], dtype=np.float64)
        self.tags = face_tags(_FIELD)
        if len(self.tags) != len(self.f):
            raise RuntimeError("face tags do not match the live mesh")
        a, b, c = self.v[self.f[:, 0]], self.v[self.f[:, 1]], self.v[self.f[:, 2]]
        cent = (a + b + c) / 3.0
        self._by_tag = {}
        for tag in range(int(self.tags.max()) + 1):
            ids = np.flatnonzero(self.tags == tag)
            if ids.size == 0:
                continue
            self._by_tag[tag] = (ids, cKDTree(cent[ids]))
        self.tag = None
        self.streak = 0
        self.streak_tag = None

    def reset(self) -> None:
        self.tag = None
        self.streak = 0
        self.streak_tag = None

    def _on(self, point: np.ndarray, tri_ids: np.ndarray):
        point = np.asarray(point, dtype=np.float64).reshape(3)
        best_d = 1e9
        best_p = point
        best_bary = np.array([1.0, 0.0, 0.0])
        best_tri = int(tri_ids[0])
        for tri in np.asarray(tri_ids, dtype=np.int64).reshape(-1):
            hit, bary, _verts = _closest_on_triangles(
                point, self.v, self.f, np.array([int(tri)], dtype=np.int64)
            )
            dist = float(np.linalg.norm(hit - point))
            if dist < best_d:
                best_d = dist
                best_p = hit
                best_bary = bary
                best_tri = int(tri)
        return best_p, best_bary, best_tri, best_d

    def tag_of(self, point: np.ndarray) -> int:
        q = self.field.scene.compute_closest_points(
            o3d.core.Tensor(np.asarray(point, dtype=np.float32).reshape(1, 3))
        )
        tri = int(q["primitive_ids"].numpy().reshape(-1)[0])
        return int(self.tags[tri])

    def _project(self, point: np.ndarray, tag: int):
        ids, tree = self._by_tag[tag]
        _d, local = tree.query(np.asarray(point, dtype=np.float64).reshape(3), k=min(12, len(ids)))
        return self._on(point, ids[np.atleast_1d(local)])

    def _frame(self, bary: np.ndarray, tri: int) -> np.ndarray:
        verts = self.f[tri]
        normal = bary @ self.n_in[verts]
        tangent = bary @ self.t1[verts]
        return orthonormal_frames(tangent, normal)

    def apply(self, tcp, foot, frame, demo_foot, demo_frame=None):
        tcp = np.asarray(tcp, dtype=np.float64).reshape(3)
        foot = np.asarray(foot, dtype=np.float64).reshape(3)
        frame = np.asarray(frame, dtype=np.float64).reshape(3, 3)
        proposed = self.tag_of(foot)
        demo_tag = None if demo_foot is None else self.tag_of(demo_foot)
        # Replay follows the stored foot. A new stored face must persist for N
        # steps. The closer-by-d test is for a policy, which has no stored foot.
        if demo_foot is not None:
            demo_foot = np.asarray(demo_foot, dtype=np.float64).reshape(3)
            demo_R = None if demo_frame is None else np.asarray(demo_frame, dtype=np.float64).reshape(3, 3)
            if self.tag is None:
                self.tag = demo_tag
                return (demo_foot, demo_R) if demo_R is not None else (demo_foot, frame)
            if demo_tag == self.tag:
                self.streak = 0
                self.streak_tag = None
                return (demo_foot, demo_R) if demo_R is not None else (demo_foot, frame)
            if demo_tag == self.streak_tag:
                self.streak += 1
            else:
                self.streak_tag = demo_tag
                self.streak = 1
            if self.streak >= self.n:
                self.tag = demo_tag
                self.streak = 0
                self.streak_tag = None
                return (demo_foot, demo_R) if demo_R is not None else (demo_foot, frame)
            held, bary, tri, _dist = self._project(tcp, self.tag)
            return held, self._frame(bary, tri)

        if self.tag is None:
            self.tag = proposed
            return foot, frame
        if proposed == self.tag:
            self.streak = 0
            self.streak_tag = None
            return foot, frame
        _held, _bary, _tri, held_d = self._project(tcp, self.tag)
        new_d = float(np.linalg.norm(tcp - foot))
        closer = held_d - new_d >= self.d
        if closer and proposed == self.streak_tag:
            self.streak += 1
        elif closer:
            self.streak_tag = proposed
            self.streak = 1
        else:
            self.streak_tag = None
            self.streak = 0
        if self.streak >= self.n:
            self.tag = proposed
            self.streak = 0
            self.streak_tag = None
            return foot, frame
        held, bary, tri, _dist = self._project(tcp, self.tag)
        return held, self._frame(bary, tri)


def _state(field) -> _Hold:
    global _STATE
    n, d = _n_steps(), _d_m()
    if _STATE is None or _STATE.n != n or _STATE.d != d or _STATE.field is not field:
        _STATE = _Hold(field, n, d)
    return _STATE


def reset_episode() -> None:
    if _STATE is not None:
        _STATE.reset()


def bind_chunk(points, frames=None) -> None:
    """Stored foot rows for this replay chunk, or None for a policy chunk.

    ``frames`` is the stored ``mold_R_field`` at those rows. Replay uses it
    while the face tag matches, so the label's field-relative ΔR is composed
    with the same frame it was built on.
    """
    global _CHUNK, _CHUNK_R, _CURSOR
    if points is None:
        _CHUNK = None
        _CHUNK_R = None
    else:
        _CHUNK = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        _CHUNK_R = None if frames is None else np.asarray(frames, dtype=np.float64).reshape(-1, 3, 3)
    _CURSOR = 0


def hold_foot(tcp, foot, frame, field):
    """Return ``(foot, frame)`` unchanged when the flag is off."""
    if not enabled():
        return foot, frame
    global _CURSOR
    demo = None
    demo_R = None
    if _CHUNK is not None and _CURSOR < len(_CHUNK):
        demo = _CHUNK[_CURSOR]
        if _CHUNK_R is not None and _CURSOR < len(_CHUNK_R):
            demo_R = _CHUNK_R[_CURSOR]
    _CURSOR += 1
    return _state(field).apply(tcp, foot, frame, demo, demo_R)
