#!/usr/bin/env python3
"""Aligned foot. Does not rewrite score.json or wall_only/.

Stops before the rescore if the same rule moves the training-stringer foot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import heldout  # noqa: E402
import input_check  # noqa: E402
from aligned_foot import FORCE_N, AlignedFoot, direction_from, thin_kind  # noqa: E402

import torch  # noqa: E402
import zarr  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from PyriteUtility.data_pipeline.wrench_baseline_utils import time_based_moving_average  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import pose7_to_Rp  # noqa: E402
from heading_tilt import long_axis  # noqa: E402
from run_policy_open_loop_val_export import load_policy_and_cfg  # noqa: E402

OUT = HERE / "aligned"
TRAIN_ZARR = input_check.TRAIN_ZARR
TRAIN_FIELD = input_check.TRAIN_FIELD
WANT = ("causal e199", "heading e100", "r1rel e25", "r1rel e195")
UP = np.array([0.0, -1.0, 0.0])
# Training is unchanged when the foot stays on the closest face.
STOP_MEDIAN_MM = 5.0
STOP_FAR_MM = 10.0
STOP_FAR_FRAC = 0.10


def _counts(names) -> dict:
    return {name: int(np.sum(np.asarray(names) == name)) for name in ("wall", "cap", "back", "apron")}


def _label(kind, tri, tcp, foot, geo):
    face = kind[tri].astype(object).copy()
    behind = (face == "wall") & (np.sum((tcp - foot) * geo[tri], axis=1) < -0.002)
    face[behind] = "back"
    return face


def _rgb_rows(demo):
    at = heldout._nearest(demo["robot_t"], demo["rgb_t"])
    j = heldout._nearest(demo["wrench_t"], demo["robot_t"][at])
    rot = demo["tool_R"][at]
    force = np.einsum("nij,nj->ni", rot, demo["smooth"][j])
    mag = np.linalg.norm(demo["smooth"][j], axis=1)
    direction = np.where((mag >= FORCE_N)[:, None], -force, rot[:, :, 2])
    return demo["tool_p"][at], direction


def _thin_counts(selector, kind):
    out = {}
    for letter, path in (
        ("C", HERE / "C_first" / "replay.zarr"),
        ("A", HERE / "A_full" / "replay.zarr"),
    ):
        demo = heldout._demo(path, None)
        tcp, direction = _rgb_rows(demo)
        hit, _vid, _w, tri, fallback = selector.choose(tcp, direction)
        face = _label(kind, tri, tcp, hit, selector._geo)
        out[letter] = {
            "n": int(len(tcp)),
            "faces": _counts(face),
            "fallback": int(fallback.sum()),
            "gap_mm_median": round(float(np.median(np.linalg.norm(tcp - hit, axis=1)) * 1e3), 2),
        }
        print(f"[aligned] {letter} faces {out[letter]['faces']}  fallback {out[letter]['fallback']}  "
              f"gap median {out[letter]['gap_mm_median']} mm", flush=True)
    return out


def _regions(normal, foot, apron):
    n = normal / np.clip(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12, None)
    tilt = np.degrees(np.arccos(np.clip(n @ UP, -1.0, 1.0)))
    name = np.full(len(n), "wall", dtype=object)
    name[tilt < 25.0] = "crown"
    name[(tilt >= 25.0) & (tilt < 55.0)] = "fillet"
    name[apron.query(foot, workers=1)[0] < 0.015] = "flange"
    return name


def _normal(field, vid, weight):
    n = np.einsum("qk,qkd->qd", weight, field.normals[vid])
    return n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-12, None)


def _train(src):
    root = zarr.open(TRAIN_ZARR, "r")["data"]
    keys = [key for key in root.keys() if key.startswith("episode_")]
    pairs = []
    for key in keys:
        n = int(np.asarray(root[key]["rgb_time_stamps_0"]).reshape(-1).shape[0])
        pairs.extend((key, i) for i in range(n))
    step = max(1, len(pairs) // 200)
    chosen = pairs[::step][:200]
    apron = cKDTree(src.vertices[np.asarray(np.load(TRAIN_FIELD)["is_apron"]).astype(bool)])
    selector = AlignedFoot(src, foot_mode="aligned")
    loaded = {}

    def episode(key):
        if key in loaded:
            return loaded[key]
        group = root[key]
        robot_t = np.asarray(group["robot_time_stamps_0"]).reshape(-1)
        rgb_t = np.asarray(group["rgb_time_stamps_0"]).reshape(-1)
        rot, pos = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
        wrench_key = "wrench_time_stamps_0" if "wrench_time_stamps_0" in group else "wrench_time_stamps_left_0"
        wrench_t = np.asarray(group[wrench_key]).reshape(-1)
        smooth = time_based_moving_average(np.asarray(group["wrench_left_0"], dtype=np.float64), wrench_t, 1.0)[:, :3]
        loaded[key] = (robot_t, rgb_t, rot, pos, wrench_t, smooth)
        return loaded[key]

    tcp, direction = [], []
    for key, rgb in chosen:
        robot_t, rgb_t, rot, pos, wrench_t, smooth = episode(key)
        at = int(heldout._nearest(robot_t, np.array([rgb_t[rgb]]))[0])
        j = int(heldout._nearest(wrench_t, np.array([robot_t[at]]))[0])
        tcp.append(pos[at])
        direction.append(direction_from(rot[at], rot[at] @ smooth[j], float(np.linalg.norm(smooth[j]))))
    tcp = np.stack(tcp)
    direction = np.stack(direction)
    hit, vid, weight, _tri, _fb = selector.choose(tcp, direction)
    from fixb_field import FixBMoldField

    c_hit, c_vid, c_w = FixBMoldField.closest(src, tcp)
    moved = np.linalg.norm(hit - c_hit, axis=1) * 1e3
    aligned_region = _regions(_normal(selector, vid, weight), hit, apron)
    closest_region = _regions(_normal(src, c_vid, c_w), c_hit, apron)
    changed = aligned_region != closest_region
    report = {
        "n": int(len(tcp)),
        "move_mm_median": round(float(np.median(moved)), 2),
        "move_mm_p90": round(float(np.percentile(moved, 90)), 2),
        "far_frac": round(float(np.mean(moved > STOP_FAR_MM)), 3),
        "region_changed_frac": round(float(np.mean(changed)), 3),
        "closest": _counts_any(closest_region),
        "aligned": _counts_any(aligned_region),
    }
    report["unchanged"] = bool(
        report["move_mm_median"] <= STOP_MEDIAN_MM and report["far_frac"] <= STOP_FAR_FRAC
    )
    print(f"[aligned] train n {report['n']}  move median {report['move_mm_median']} mm"
          f"  p90 {report['move_mm_p90']}  far {report['far_frac']}"
          f"  region changed {report['region_changed_frac']}", flush=True)
    print(f"[aligned] train closest {report['closest']}", flush=True)
    print(f"[aligned] train aligned {report['aligned']}", flush=True)
    return report


def _counts_any(names) -> dict:
    vals, counts = np.unique(names, return_counts=True)
    return {str(v): int(c) for v, c in zip(vals, counts)}


def _gt(selector, axis):
    out = {}
    for letter, path in (
        ("C", HERE / "C_first" / "replay.zarr"),
        ("A", HERE / "A_full" / "replay.zarr"),
    ):
        demo = input_check._load(path)
        demo["smooth"] = heldout._demo(path, None)["smooth"]
        demo["wrench_t"] = heldout._demo(path, None)["wrench_t"]
        queries, _rgb = input_check._window_queries(letter)
        pos = []
        for index in queries:
            i0 = int(index[0])
            j = int(heldout._nearest(demo["wrench_t"], np.array([demo["robot_t"][i0]]))[0])
            force = demo["tool_R"][i0] @ demo["smooth"][j]
            selector.direction = direction_from(demo["tool_R"][i0], force, float(np.linalg.norm(demo["smooth"][j])))
            foot0, frame0 = heldout._wall_anchor(selector, demo["tool_p"][i0])
            dts = heldout._dts(demo, index)
            walked, _frames = heldout._walk(
                selector, np.zeros((len(index), 10)), foot0, frame0, demo["tool_p"][i0], dts
            )
            pos.append(float(np.linalg.norm(walked[0] - demo["tool_p"][i0]) * 1e3))
        arr = np.asarray(pos)
        out[letter] = {
            "n": int(len(arr)),
            "step1_mm_mean": round(float(arr.mean()), 2),
            "step1_mm_median": round(float(np.median(arr)), 2),
        }
        print(f"[aligned] GT {letter} step-1 mean {out[letter]['step1_mm_mean']} "
              f"median {out[letter]['step1_mm_median']} mm", flush=True)
    return out


def _rescore(selector, axis):
    spec = json.loads((HERE / "checkpoints.json").read_text())
    found = []
    for entry in spec["models"]:
        if entry["name"] not in WANT:
            continue
        path = heldout._resolve(entry, Path(spec.get("train_root") or heldout.TRAIN))
        if path is None:
            raise SystemExit(f"missing {entry['name']}")
        found.append((entry, path))
    t_mid = json.loads((heldout.SCRIPTS / "eval/thin_variants/contact_half.json").read_text())["episodes"]["C"]["t_mid_s"]
    recordings = {}
    for letter, path in (("C", HERE / "C_first" / "replay.zarr"), ("A", HERE / "A_full" / "replay.zarr")):
        opened = heldout._demo(path, None)
        cutoff = None if letter == "A" else float(opened["rgb_t"][0]) + float(t_mid)
        demo = opened if cutoff is None else heldout._demo(path, cutoff)
        demo["axis"] = axis
        recordings[letter] = {"path": path, "demo": demo}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = {}
    tables = {"C": [], "A": [], "combined": []}
    report = {"foot_mode": "aligned", "radius_mm": 30, "recordings": {}}
    for entry, ckpt in found:
        policy, cfg = load_policy_and_cfg(str(ckpt), device)
        resolved = heldout._norm_of(cfg, entry.get("kind") or "auto")
        if resolved is None:
            raise SystemExit(f"unsupported {entry['name']}")
        norm, _dim, decode, sha = resolved
        scored = {}
        for letter, rec in recordings.items():
            key = (str(rec["path"]), str(norm))
            if key not in cache:
                cache[key] = heldout._dataset(cfg, rec["path"], norm)
                cache[key + ("q",)] = heldout._queries(cache[key], rec["demo"])
            rows = cache[key + ("q",)]
            print(f"[aligned] score {entry['name']} {letter} {len(rows)}", flush=True)
            block = heldout._score_one(
                decode, policy, cache[key], selector, rec["demo"], rows, device, axis,
                foot_mode="aligned" if decode != "r1rel" else None,
            )
            scored[letter] = block
        pooled = heldout._pool([scored["C"], scored["A"]])
        for letter in ("C", "A"):
            public = heldout._public(scored[letter])
            tables[letter].append((entry["name"], public))
            report["recordings"].setdefault(letter, {})[entry["name"]] = {
                **public, "ckpt": str(ckpt), "decode": decode, "sha": sha[:12],
            }
        tables["combined"].append((entry["name"], pooled))
        report["recordings"].setdefault("combined", {})[entry["name"]] = pooled
        del policy
        torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "score.json").write_text(json.dumps(report, indent=2))
    heldout._print_table("C  aligned foot", tables["C"])
    heldout._print_table("A  aligned foot", tables["A"])
    print(f"[aligned] wrote {OUT}", flush=True)


def main() -> None:
    heldout._check()
    src = FixBMoldField(str(heldout.WALL))
    kind = thin_kind(src, heldout.WALL)
    selector = AlignedFoot(src, foot_mode="aligned")
    faces = _thin_counts(selector, kind)
    train = _train(FixBMoldField(str(TRAIN_FIELD)))
    OUT.mkdir(parents=True, exist_ok=True)
    if not train["unchanged"]:
        (OUT / "stopped.json").write_text(json.dumps({"faces": faces, "train": train}, indent=2))
        print("[aligned] training foot moved. Not rescoring.", flush=True)
        return
    axis = long_axis(src.vertices)
    gt = _gt(selector, axis)
    (OUT / "check.json").write_text(json.dumps({"faces": faces, "train": train, "gt": gt}, indent=2))
    _rescore(selector, axis)


if __name__ == "__main__":
    main()
