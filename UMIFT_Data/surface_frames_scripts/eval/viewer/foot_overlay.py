"""Foot of the thin-page TCP on the full wall mesh. One scene, reused."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d

MESH = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/"
    "wall_fix/fixb_field_wall_vertical.npz"
)
_FIELD = None
_KIND = None


def _load():
    global _FIELD, _KIND
    if _FIELD is not None:
        return _FIELD, _KIND
    from fixb_field import FixBMoldField

    field = FixBMoldField(str(MESH))
    n_cap = int(json.loads(MESH.with_name("gap_report_vertical.json").read_text())["trim"]["field"]["cap_vertices"])
    cap = np.zeros(len(field.vertices), dtype=bool)
    cap[-n_cap:] = True
    apron = np.asarray(np.load(MESH)["is_apron"]).astype(bool)
    tri = field.faces
    kind = np.full(len(tri), "wall", dtype=object)
    kind[apron[tri].any(axis=1)] = "apron"
    kind[cap[tri].all(axis=1)] = "cap"
    _check(field, kind, cap)
    _FIELD, _KIND = field, kind
    return field, kind


def _check(field, kind, cap) -> None:
    wall = np.flatnonzero(kind == "wall")
    center = field.vertices[field.faces[wall[len(wall) // 2]]].mean(axis=0)
    foot, face, z = _project(field, kind, center.reshape(1, 3))
    if face[0] != "wall" or abs(float(np.linalg.norm(z[0])) - 1.0) > 1e-6:
        raise RuntimeError(f"foot overlay {face[0]} |z| {np.linalg.norm(z[0])}")
    if float(np.linalg.norm(foot[0] - center)) > 1e-3:
        raise RuntimeError("wall centroid did not snap")
    hit, face_c, _z = _project(field, kind, field.vertices[np.flatnonzero(cap)[-1]].reshape(1, 3))
    if face_c[0] == "wall" and float(np.linalg.norm(hit[0] - field.vertices[np.flatnonzero(cap)[-1]])) < 1e-4:
        raise RuntimeError("cap vertex classified as a wall point")


def _project(field, kind, points):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    ans = field.scene.compute_closest_points(o3d.core.Tensor(pts.astype(np.float32)))
    tri = np.clip(ans["primitive_ids"].numpy().astype(np.int64), 0, len(field.faces) - 1)
    uv = ans["primitive_uvs"].numpy().astype(np.float64)
    w = np.stack([1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]], axis=1)
    vid = field.faces[tri]
    foot = np.einsum("qk,qkd->qd", w, field.vertices[vid])
    normal = np.einsum("qk,qkd->qd", w, field.normals[vid])
    normal /= np.clip(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12, None)
    face = kind[tri].copy()
    # ponytail: 2 mm. Vertex normals are not the face normal, so a graze is not "back".
    behind = (face == "wall") & (np.sum((pts - foot) * normal, axis=1) < -0.002)
    face[behind] = "back"
    return foot, face, -normal


def overlay(points) -> dict:
    """Foot, face, and inward frame-z. Same length and order as `points`."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return {"foot": [], "foot_face": [], "foot_z": []}
    field, kind = _load()
    foot, face, z = _project(field, kind, pts)
    return {
        "foot": np.round(foot, 4).tolist(),
        "foot_face": face.tolist(),
        "foot_z": np.round(z, 4).tolist(),
    }
