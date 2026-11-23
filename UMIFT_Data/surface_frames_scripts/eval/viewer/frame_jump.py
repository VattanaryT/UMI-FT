#!/usr/bin/env python3
"""Frame jump between neighboring work triangles.

Work faces are wall, fillet, and crown. An edge is on the walk when the two
triangles share a side and either the region steps wall→fillet or fillet→crown,
or both are the same region and the step follows the long axis. The frame is
the diffused field at the triangle centroid. A heading flip is t1 reversing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from fixb_field import orthonormal_frames  # noqa: E402
from heading_tilt import long_axis  # noqa: E402

WALL, FILLET, CROWN = 0, 1, 2
ORDER = {WALL: 0, FILLET: 1, CROWN: 2}
UP = np.array([0.0, -1.0, 0.0])
THIN = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/"
    "wall_fix/fixb_field_wall_vertical.npz"
)
TRAIN = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)


def _tags(vertices, faces, normals, apron) -> np.ndarray:
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    geo = np.cross(b - a, c - a)
    geo /= np.clip(np.linalg.norm(geo, axis=1, keepdims=True), 1e-12, None)
    vn = normals[faces].mean(1)
    vn /= np.clip(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12, None)
    geo *= np.where((geo * vn).sum(1) < 0.0, -1.0, 1.0)[:, None]
    tilt = np.degrees(np.arccos(np.clip(geo @ UP, -1.0, 1.0)))
    tag = np.full(len(faces), WALL, np.uint8)
    tag[(tilt >= 25.0) & (tilt < 55.0)] = FILLET
    tag[tilt < 25.0] = CROWN
    if apron is not None and np.any(apron):
        apron_tri = np.asarray(apron, dtype=bool)[faces].any(1)
        near = cKDTree(vertices[np.asarray(apron, dtype=bool)]).query((a + b + c) / 3.0, workers=1)[0] < 0.015
        tag[near & ~apron_tri] = 3
        tag[apron_tri] = 6
    return tag


def _frames(vertices, faces, normals, frames):
    vid = faces
    t1 = frames[vid][..., 0].mean(1)
    n = normals[vid].mean(1)
    return orthonormal_frames(t1, n)


def _edges(faces):
    edge: dict[tuple[int, int], list[int]] = {}
    for i, tri in enumerate(faces):
        for u, w in ((0, 1), (1, 2), (2, 0)):
            edge.setdefault((int(min(tri[u], tri[w])), int(max(tri[u], tri[w]))), []).append(i)
    pairs = [ids for ids in edge.values() if len(ids) == 2]
    if not pairs:
        return np.zeros((0, 2), np.int64)
    return np.asarray(pairs, dtype=np.int64)


def _geodesic(ra, rb) -> np.ndarray:
    rel = np.einsum("nji,njk->nik", ra, rb)
    trace = rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]
    return np.degrees(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))


def measure(path: Path) -> dict:
    d = np.load(path)
    vertices = d["vertices"].astype(np.float64)
    faces = d["faces"].astype(np.int64)
    normals = d["normals"].astype(np.float64)
    frames = d["frames"].astype(np.float64)
    apron = np.asarray(d["is_apron"]).astype(bool) if "is_apron" in d.files else None
    tag = _tags(vertices, faces, normals, apron)
    orient = _frames(vertices, faces, normals, frames)
    pairs = _edges(faces)
    work = np.isin(tag, (WALL, FILLET, CROWN))
    keep = work[pairs[:, 0]] & work[pairs[:, 1]]
    pairs = pairs[keep]
    left, right = pairs[:, 0], pairs[:, 1]
    sa = np.array([ORDER[int(t)] for t in tag[left]])
    sb = np.array([ORDER[int(t)] for t in tag[right]])
    across = np.abs(sa - sb) == 1
    same = sa == sb
    axis = long_axis(vertices)
    cent = vertices[faces].mean(1)
    delta = cent[right] - cent[left]
    along = same & (np.abs(delta @ axis) >= (np.linalg.norm(delta, axis=1) * 0.707))
    take = across | along
    ang_all = _geodesic(orient[left], orient[right])
    ang = ang_all[take]
    t1 = orient[:, :, 0]
    flip_all = np.sum(t1[left] * t1[right], axis=1) < 0.0
    flip = flip_all[take]
    per = np.full(len(faces), -1.0)
    if len(ang):
        chosen = pairs[take]
        for col in (0, 1):
            np.maximum.at(per, chosen[:, col], ang)
    per[per < 0.0] = np.nan
    return {
        "n_edges": int(take.sum()),
        "n_across": int(across.sum()),
        "n_along": int(along.sum()),
        "median": round(float(np.median(ang)), 2) if len(ang) else None,
        "p99": round(float(np.percentile(ang, 99)), 2) if len(ang) else None,
        "max": round(float(np.max(ang)), 2) if len(ang) else None,
        "heading_flips": int(flip.sum()),
        "heading_flip_frac": round(float(flip.mean()), 4) if len(flip) else None,
        "across": _band(ang_all[across], flip_all[across]),
        "along": _band(ang_all[along], flip_all[along]),
        "per_face": per,
        "centroids": cent,
    }


def for_display(path: Path, mesh: dict) -> list:
    """Jump of the nearest full-mesh face, one value per displayed triangle."""
    stat = measure(Path(path))
    known = np.isfinite(stat["per_face"])
    if not np.any(known):
        return []
    tree = cKDTree(stat["centroids"][known])
    verts = np.column_stack([mesh["x"], mesh["y"], mesh["z"]])
    faces = np.column_stack([mesh["i"], mesh["j"], mesh["k"]])
    cent = verts[faces].mean(1)
    _dist, nn = tree.query(cent, workers=1)
    values = stat["per_face"][known][nn]
    return [round(float(v), 2) for v in values]


def _check() -> None:
    eye = np.eye(3)
    flipped = eye.copy()
    flipped[:, 0] *= -1
    flipped[:, 1] *= -1
    ang = _geodesic(eye.reshape(1, 3, 3), flipped.reshape(1, 3, 3))
    if abs(float(ang[0]) - 180.0) > 1e-4:
        raise RuntimeError(f"heading reversal geodesic {ang[0]}")
    if float(np.sum(eye[:, 0] * flipped[:, 0])) >= 0.0:
        raise RuntimeError("reversal was not a heading flip")
    if float(_geodesic(eye.reshape(1, 3, 3), eye.reshape(1, 3, 3))[0]) > 1e-6:
        raise RuntimeError("identical frames jumped")


def _band(ang: np.ndarray, flip: np.ndarray) -> dict:
    if len(ang) == 0:
        return {"n": 0}
    return {
        "n": int(len(ang)),
        "median": round(float(np.median(ang)), 2),
        "p99": round(float(np.percentile(ang, 99)), 2),
        "max": round(float(np.max(ang)), 2),
        "heading_flips": int(flip.sum()),
    }


def _public(stat: dict) -> dict:
    keep = (
        "n_edges", "n_across", "n_along", "median", "p99", "max",
        "heading_flips", "heading_flip_frac", "across", "along",
    )
    return {key: stat[key] for key in keep}


def main() -> None:
    _check()
    report = {"thin": _public(measure(THIN)), "train": _public(measure(TRAIN))}
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
