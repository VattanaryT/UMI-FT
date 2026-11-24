#!/usr/bin/env python3
"""Sliding work-surface foot. Probe only: does not write a zarr or a field.

Face tag, per triangle, uint8:
  0 wall, 1 crown, 2 fillet, 3 flange, 4 cap, 5 back, 6 apron.
  Work is 0..3. Excluded is 4..6. Code 5 is reserved: neither mesh has a
  second skin, so "back" is the query landing, (tcp - foot) · n < -2 mm.

The foot slides with the tool's tangential motion and may cross onto a
neighboring work face. A global closest point is tried only when contact
drops below 8 N or the foot is more than 20 mm from the TCP, and it is kept
only on a work face, on the near side, and within 30 mm of the TCP. The frame
is the barycentric diffused field at that surface point.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from causal_foot import _closest_on_triangles
from fixb_field import FixBMoldField, orthonormal_frames
from fixb_labels import pose7_to_Rp

NAMES = ("wall", "crown", "fillet", "flange", "cap", "back", "apron")
WORK = np.array([0, 1, 2, 3], dtype=np.uint8)
UP = np.array([0.0, -1.0, 0.0])
ACCEPT_M = 0.030
GAP_M = 0.020
BACK_M = 0.002
CONTACT_N = 8.0
HORIZON_S = 4.2
TRAIN_FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
TRAIN_ZARR = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_causal_foot/acp_replay_buffer.zarr"
)
THIN_FIELD = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/wall_fix/fixb_field_wall_vertical.npz"
)


def face_tags(path: Path, cap_vertices: int = 0) -> np.ndarray:
    """One uint8 per triangle. Cap wins over apron, apron over the tilt class."""
    d = np.load(path)
    vertices = d["vertices"].astype(np.float64)
    faces = d["faces"].astype(np.int64)
    normals = d["normals"].astype(np.float64)
    apron = np.asarray(d["is_apron"]).astype(bool) if "is_apron" in d.files else np.zeros(len(vertices), bool)
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    geo = np.cross(b - a, c - a)
    geo /= np.clip(np.linalg.norm(geo, axis=1, keepdims=True), 1e-12, None)
    vn = normals[faces].mean(1)
    vn /= np.clip(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12, None)
    geo *= np.where((geo * vn).sum(1) < 0.0, -1.0, 1.0)[:, None]
    tilt = np.degrees(np.arccos(np.clip(geo @ UP, -1.0, 1.0)))
    tag = np.zeros(len(faces), np.uint8)
    tag[tilt < 25.0] = 1
    tag[(tilt >= 25.0) & (tilt < 55.0)] = 2
    apron_tri = apron[faces].any(1)
    if apron.any():
        near = cKDTree(vertices[apron]).query((a + b + c) / 3.0, workers=1)[0] < 0.015
        tag[near & ~apron_tri] = 3
    tag[apron_tri] = 6
    if cap_vertices:
        cap = np.zeros(len(vertices), bool)
        cap[-int(cap_vertices):] = True
    else:
        centered = vertices - vertices.mean(0)
        axis = np.linalg.svd(centered, full_matrices=False)[2][0]
        coord = centered @ axis
        lo, hi = np.quantile(coord, [0.06, 0.94])
        cap = (coord < lo) | (coord > hi)
    tag[cap[faces].all(1)] = 4
    return tag


class SlidingFoot:
    def __init__(self, path: Path, tag: np.ndarray):
        self.field = FixBMoldField(str(path))
        self.v = self.field.vertices
        self.f = self.field.faces
        self.tag = np.asarray(tag, dtype=np.uint8)
        self.frames = self.field.frames
        self.normals = self.field.normals
        a, b, c = self.v[self.f[:, 0]], self.v[self.f[:, 1]], self.v[self.f[:, 2]]
        geo = np.cross(b - a, c - a)
        geo /= np.clip(np.linalg.norm(geo, axis=1, keepdims=True), 1e-12, None)
        vn = self.normals[self.f].mean(1)
        geo *= np.where((geo * vn).sum(1) < 0.0, -1.0, 1.0)[:, None]
        self.tri_n = geo
        self.cent = (a + b + c) / 3.0
        nbr = [[] for _ in range(len(self.f))]
        edge: dict[tuple[int, int], list[int]] = {}
        for i, tri in enumerate(self.f):
            for u, w in ((0, 1), (1, 2), (2, 0)):
                edge.setdefault((int(min(tri[u], tri[w])), int(max(tri[u], tri[w]))), []).append(i)
        for ids in edge.values():
            for a, b in zip(ids, ids[1:]):
                nbr[a].append(b)
                nbr[b].append(a)
        self.nbr = nbr

    def closest(self, p: np.ndarray):
        q = o3d_closest(self.field, np.asarray(p, dtype=np.float64).reshape(1, 3))
        return q

    def frame_at(self, point, tri: int) -> np.ndarray:
        found = _closest_on_triangles(point, self.v, self.f, np.array([tri]))
        _point, bary, verts = found
        n = bary @ self.normals[verts]
        t1 = bary @ self.frames[verts][:, :, 0]
        return orthonormal_frames(t1, n)

    def slide(self, x, tri: int, tang: np.ndarray):
        remaining = np.asarray(tang, dtype=np.float64).reshape(3).copy()
        for _ in range(30):
            dist = float(np.linalg.norm(remaining))
            if dist < 1e-4:
                break
            step = remaining if dist <= 0.002 else remaining * (0.002 / dist)
            guess = x + step
            ids = [tri, *self.nbr[tri]]
            work = np.asarray([i for i in ids if self.tag[i] in WORK], dtype=np.int64)
            if len(work) == 0:
                break
            found = _closest_on_triangles(guess, self.v, self.f, work)
            if found is None:
                break
            point, _bary, verts = found
            key = tuple(sorted(int(v) for v in verts))
            landed = tri
            for cand in work:
                if tuple(sorted(int(v) for v in self.f[cand])) == key:
                    landed = int(cand)
                    break
            x = point
            tri = landed
            if float(np.linalg.norm(guess - point)) > 5e-4:
                break
            remaining = remaining - step
        return x, tri

    def run(self, tcp: np.ndarray, force: np.ndarray):
        """tcp (N,3), |F| (N,). Returns foot (N,3), tri (N,), n_requery, n_rejected."""
        tcp = np.asarray(tcp, dtype=np.float64).reshape(-1, 3)
        force = np.asarray(force, dtype=np.float64).reshape(-1)
        free = o3d_closest(self.field, tcp)
        out = np.full_like(tcp, np.nan)
        tri_out = np.full(len(tcp), -1, np.int64)
        x = tri = None
        asked = rejected = 0
        for i, p in enumerate(tcp):
            lost = float(force[i]) < CONTACT_N
            far = x is not None and float(np.linalg.norm(p - x)) > GAP_M
            if x is None or lost or far:
                asked += 1
                gx, gtri = free[0][i], int(free[1][i])
                near = self._ok(p, gx, gtri, seed=(x is None))
                if near:
                    x, tri = gx, gtri
                else:
                    rejected += 1
            elif x is not None:
                delta = p - tcp[i - 1]
                tang = delta - self.tri_n[tri] * float(np.dot(delta, self.tri_n[tri]))
                x, tri = self.slide(x, tri, tang)
            if x is not None:
                out[i] = x
                tri_out[i] = tri
        return out, tri_out, asked, rejected

    def _ok(self, tcp, foot, tri: int, seed: bool) -> bool:
        if self.tag[tri] not in WORK:
            return False
        if float(np.dot(tcp - foot, self.tri_n[tri])) < -BACK_M:
            return False
        del seed
        return float(np.linalg.norm(tcp - foot)) <= ACCEPT_M


def o3d_closest(field, points):
    import open3d as o3d

    q = o3d.core.Tensor(np.asarray(points, dtype=np.float32).reshape(-1, 3))
    ans = field.scene.compute_closest_points(q)
    tri = ans["primitive_ids"].numpy().astype(np.int64)
    uv = ans["primitive_uvs"].numpy().astype(np.float64)
    w = np.stack([1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]], axis=1)
    vid = field.faces[tri]
    x = np.einsum("qk,qkd->qd", w, field.vertices[vid])
    return x, tri


def _check() -> None:
    cap_n = int(json.loads(THIN_FIELD.with_name("gap_report_vertical.json").read_text())["trim"]["field"]["cap_vertices"])
    tag = face_tags(THIN_FIELD, cap_n)
    if tag[tag == 4].size == 0 or tag[tag == 6].size == 0:
        raise RuntimeError("thin cap or apron tag missing")
    foot = SlidingFoot(THIN_FIELD, tag)
    work = np.flatnonzero(np.isin(tag, WORK))
    tri = int(work[len(work) // 3])
    tangent = np.cross(foot.tri_n[tri], np.array([1.0, 0.0, 0.0]))
    if float(np.linalg.norm(tangent)) < 1e-6:
        tangent = np.cross(foot.tri_n[tri], np.array([0.0, 1.0, 0.0]))
    tangent = tangent / np.linalg.norm(tangent) * 0.003
    landed, tri2 = foot.slide(foot.cent[tri], tri, tangent)
    frame = foot.frame_at(landed, tri2)
    if float(np.linalg.norm(landed - foot.cent[tri])) < 0.001 or tag[tri2] not in WORK:
        raise RuntimeError("tangent slide left the work surface")
    if frame.shape != (3, 3) or abs(float(np.linalg.det(frame)) - 1.0) > 1e-3:
        raise RuntimeError("field frame at the foot is not a rotation")
    cap_tri = int(np.flatnonzero(tag == 4)[0])
    apron_tri = int(np.flatnonzero(tag == 6)[0])
    if foot._ok(foot.cent[cap_tri], foot.cent[cap_tri], cap_tri, seed=True):
        raise RuntimeError("cap seed accepted")
    if foot._ok(foot.cent[apron_tri], foot.cent[apron_tri], apron_tri, seed=False):
        raise RuntimeError("apron re-query accepted")
    near = foot.cent[tri] + foot.tri_n[tri] * 0.020
    far = foot.cent[tri] + foot.tri_n[tri] * (ACCEPT_M + 0.005)
    if not foot._ok(near, foot.cent[tri], tri, seed=False):
        raise RuntimeError("20 mm work face was rejected")
    if foot._ok(far, foot.cent[tri], tri, seed=False):
        raise RuntimeError("beyond 30 mm was accepted")


def _counts(tag: np.ndarray) -> dict:
    return {NAMES[i]: int(np.sum(tag == i)) for i in range(len(NAMES)) if np.any(tag == i)}


def _windows(n_target: int = 200):
    import zarr
    from PyriteUtility.data_pipeline.wrench_baseline_utils import time_based_moving_average

    root = zarr.open(TRAIN_ZARR, "r")["data"]
    pairs = []
    for key in root.keys():
        if not key.startswith("episode_"):
            continue
        n = int(np.asarray(root[key]["rgb_time_stamps_0"]).reshape(-1).shape[0])
        pairs.extend((key, i) for i in range(n))
    step = max(1, len(pairs) // n_target)
    chosen = pairs[::step][:n_target]
    by_ep: dict[str, list[int]] = {}
    for key, rgb in chosen:
        by_ep.setdefault(key, []).append(rgb)
    return root, by_ep


def train_shift() -> dict:
    root, by_ep = _windows()
    tag = face_tags(TRAIN_FIELD, 0)
    foot = SlidingFoot(TRAIN_FIELD, tag)
    from PyriteUtility.data_pipeline.wrench_baseline_utils import time_based_moving_average

    moved, changed, contact_moved = [], [], []
    closest_tag, slide_tag = [], []
    asked = rejected = seeded = 0
    gaps = []
    for key, rgbs in by_ep.items():
        group = root[key]
        robot_t = np.asarray(group["robot_time_stamps_0"]).reshape(-1)
        rgb_t = np.asarray(group["rgb_time_stamps_0"]).reshape(-1)
        _rot, pos = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
        wrench_key = "wrench_time_stamps_0" if "wrench_time_stamps_0" in group else "wrench_time_stamps_left_0"
        wrench_t = np.asarray(group[wrench_key]).reshape(-1)
        smooth = time_based_moving_average(np.asarray(group["wrench_left_0"], dtype=np.float64), wrench_t, 1.0)[:, :3]
        mag = np.linalg.norm(smooth, axis=1)
        for rgb in rgbs:
            at = int(np.argmin(np.abs(robot_t - rgb_t[rgb])))
            t0 = robot_t[at] - HORIZON_S
            i0 = int(np.searchsorted(robot_t, t0))
            sl = slice(i0, at + 1)
            j = np.clip(np.searchsorted(wrench_t, robot_t[sl]), 0, len(mag) - 1)
            walked, tris, n_ask, n_rej = foot.run(pos[sl], mag[j])
            asked += n_ask
            rejected += n_rej
            free = o3d_closest(foot.field, pos[at:at + 1])
            gaps.append(float(np.linalg.norm(pos[at] - free[0][0])) * 1e3)
            if tris[-1] < 0:
                continue
            seeded += 1
            shift = float(np.linalg.norm(walked[-1] - free[0][0])) * 1e3
            moved.append(shift)
            closest_tag.append(int(tag[int(free[1][0])]))
            slide_tag.append(int(tag[int(tris[-1])]))
            changed.append(closest_tag[-1] != slide_tag[-1])
            if float(mag[j[-1]]) >= CONTACT_N:
                contact_moved.append(shift)
    moved = np.asarray(moved, dtype=np.float64)
    contact_moved = np.asarray(contact_moved, dtype=np.float64)
    return {
        "n_windows": int(sum(len(v) for v in by_ep.values())),
        "n_seeded": seeded,
        "move_mm_median": round(float(np.median(moved)), 2),
        "move_mm_p90": round(float(np.percentile(moved, 90)), 2),
        "far_frac": round(float(np.mean(moved > 10.0)), 3),
        "contact_n": int(len(contact_moved)),
        "contact_move_mm_median": round(float(np.median(contact_moved)), 2) if len(contact_moved) else None,
        "contact_move_mm_p90": round(float(np.percentile(contact_moved, 90)), 2) if len(contact_moved) else None,
        "contact_far_frac": round(float(np.mean(contact_moved > 10.0)), 3) if len(contact_moved) else None,
        "tag_changed_frac": round(float(np.mean(changed)), 3),
        "closest_gap_mm_median": round(float(np.median(gaps)), 2),
        "requery": asked,
        "requery_rejected": rejected,
        "closest_cap": int(np.sum(np.asarray(closest_tag) == 4)),
        "closest_apron": int(np.sum(np.asarray(closest_tag) == 6)),
        "sliding_cap": int(np.sum(np.asarray(slide_tag) == 4)),
        "sliding_apron": int(np.sum(np.asarray(slide_tag) == 6)),
        "closest": _counts(np.asarray(closest_tag)),
        "sliding": _counts(np.asarray(slide_tag)),
        "train_tags": _counts(tag),
    }


def main() -> None:
    _check()
    cap_n = int(json.loads(THIN_FIELD.with_name("gap_report_vertical.json").read_text())["trim"]["field"]["cap_vertices"])
    thin = _counts(face_tags(THIN_FIELD, cap_n))
    report = train_shift()
    report["thin_tags"] = thin
    report["accept_mm"] = ACCEPT_M * 1e3
    report["requery_gap_mm"] = GAP_M * 1e3
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
