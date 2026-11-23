#!/usr/bin/env python3
"""Wall-face foot. Does not rewrite thin_heldout/score.json.

Ground-truth labels go through the scorer with the foot snapped to wall
triangles only. Then causal e199 / heading e100 inputs are compared with
~200 training windows, and every checkpoint is rescored into wall_only/.
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
from wall_foot import wall_field  # noqa: E402

import torch  # noqa: E402
import zarr  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from diffusion_policy.common.drop_depth_inputs import apply_drop_depth_inputs  # noqa: E402
from diffusion_policy.common.drop_scan_inputs import apply_drop_scan_inputs  # noqa: E402
from heading_tilt import geodesic_deg, long_axis  # noqa: E402
from mold_aware_runtime import rot6d_rows_to_R  # noqa: E402
from run_policy_open_loop_val_export import load_policy_and_cfg, predict_sparse_action  # noqa: E402

OUT = HERE / "wall_only"
OLD = HERE / "score.json"
COMPONENTS = {
    "robot0_eef_pos": ("eef_x", "eef_y", "eef_z"),
    "robot0_eef_rot_axis_angle": ("rot_x", "rot_y", "rot_z"),
    "robot0_eef_wrench_left": ("fx", "fy", "fz", "tx", "ty", "tz"),
    "robot0_mold_geom": ("sd_m", "n_x", "n_y", "n_z", "R6_0", "R6_1", "R6_2", "R6_3", "R6_4", "R6_5"),
}
WATCH = ("causal e199", "heading e100")


def _project(wall, points):
    geom = wall.query(np.asarray(points, dtype=np.float64))
    outward = np.asarray(geom["R"], dtype=np.float64).reshape(-1, 3, 3)
    foot = np.asarray(geom["x_s"], dtype=np.float64).reshape(-1, 3)
    return foot, heldout.thin.inward_frame(outward)


def _rgb_series(full, wall, demo, kind):
    t0 = float(demo["rgb_t"][0])
    rel = demo["rgb_t"] - t0
    at = heldout._nearest(demo["robot_t"], demo["rgb_t"])
    tcp = demo["tool_p"][at]
    foot, _frame = _project(wall, tcp)
    tri = input_check.o3d_closest(full, demo["foot"][at])
    return {
        "t": [round(float(v), 3) for v in rel],
        "wall_mm": [round(float(v), 2) for v in np.linalg.norm(tcp - foot, axis=1) * 1e3],
        "full_face": kind[tri].tolist(),
    }


def _ground_truth(full, wall, axis):
    kind = input_check._face_kind(full)
    out = {"recordings": {}}
    for letter, path in (
        ("C", HERE / "C_first" / "replay.zarr"),
        ("A", HERE / "A_full" / "replay.zarr"),
    ):
        demo = input_check._load(path)
        queries, _rgb = input_check._window_queries(letter)
        stored = demo["foot"].copy()
        foot, frame = _project(wall, demo["tool_p"])
        swapped = dict(demo)
        swapped["foot"] = foot
        swapped["frame"] = frame
        gt = input_check._score_gt(wall, swapped, axis, queries)
        i0 = queries[:, 0]
        gap = np.linalg.norm(demo["tool_p"][i0] - foot[i0], axis=1) * 1e3
        moved = np.linalg.norm(foot[i0] - stored[i0], axis=1) * 1e3
        tri = input_check.o3d_closest(full, stored[i0])
        face = kind[tri]
        series = _rgb_series(full, wall, demo, kind)
        pos = gt["pos"]
        series["note"] = (
            f"wall-only foot   GT step-1 pos mean {pos['mean']['1']} mm"
            f"  median {pos['median']['1']} mm   color = unrestricted face"
        )
        out["recordings"][letter] = {
            "gt": gt,
            "query_gap_mm": {
                "mean": round(float(gap.mean()), 2),
                "median": round(float(np.median(gap)), 2),
            },
            "foot_moved_mm_median": round(float(np.median(moved)), 2),
            "query_face": {name: int(np.sum(face == name)) for name in ("wall", "cap", "apron")},
            "series": series,
        }
        print(
            f"[wall] {letter} step-1 pos mean {pos['mean']['1']} median {pos['median']['1']} mm"
            f"  gap median {np.median(gap):.1f}  moved median {np.median(moved):.1f}"
            f"  rot median {gt['rot']['median']['1']}°  faces {out['recordings'][letter]['query_face']}",
            flush=True,
        )
    return out, kind


def _feats(obs, foot):
    sparse = obs["sparse"]
    out = {}
    for key, names in COMPONENTS.items():
        if key not in sparse:
            continue
        row = np.asarray(sparse[key], dtype=np.float64)
        row = row.reshape(-1, row.shape[-1])[-1]
        for i, name in enumerate(names):
            out[name] = float(row[i])
    out["force_n"] = float(np.linalg.norm([out["fx"], out["fy"], out["fz"]]))
    out["rot_deg"] = float(np.degrees(np.linalg.norm([out["rot_x"], out["rot_y"], out["rot_z"]])))
    out["sd_mm"] = out["sd_m"] * 1e3
    frame = rot6d_rows_to_R(np.array([[out[f"R6_{i}"] for i in range(6)]], dtype=np.float64))
    out["frame_deg"] = float(geodesic_deg(frame, np.eye(3)[None])[0])
    out["foot_x"], out["foot_y"], out["foot_z"] = (float(v) for v in np.asarray(foot).reshape(3))
    return out


def _pred_angle(kind, pred):
    if kind == "causal":
        delta = rot6d_rows_to_R(pred[0:1, 4:10])
        return float(geodesic_deg(delta, np.eye(3)[None])[0])
    tilt = pred[0, 4:8]
    psi = np.arctan2(tilt[0], tilt[1])
    c, s = np.cos(psi), np.sin(psi)
    rz = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    swing = Rotation.from_rotvec([tilt[2], tilt[3], 0.0]).as_matrix()
    return float(geodesic_deg((rz @ swing)[None], np.eye(3)[None])[0])


def _stats(values):
    val = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(val)),
        "p10": round(float(np.percentile(val, 10)), 3),
        "p50": round(float(np.median(val)), 3),
        "p90": round(float(np.percentile(val, 90)), 3),
        "min": round(float(val.min()), 3),
        "max": round(float(val.max()), 3),
    }


def _compare(train, thin):
    rows = []
    for key in train[0]:
        a = np.array([row[key] for row in train], dtype=np.float64)
        b = np.array([row[key] for row in thin], dtype=np.float64)
        lo, hi = float(a.min()), float(a.max())
        outside = (b < lo - 1e-8) | (b > hi + 1e-8)
        frac = float(outside.mean()) if len(b) else 0.0
        med = float(np.median(b))
        flag = frac >= 0.10 or med < lo or med > hi
        span = hi - lo
        unit = None
        if flag and span > 1e-9:
            unit = [round(float(v), 3) for v in (b - lo) / span]
        rows.append({
            "name": key,
            "flag": flag,
            "frac": round(frac, 3),
            "train_min": round(lo, 4),
            "train_max": round(hi, 4),
            "train_p50": round(float(np.median(a)), 4),
            "thin_p50": round(med, 4),
            "thin_p10": round(float(np.percentile(b, 10)), 4),
            "thin_p90": round(float(np.percentile(b, 90)), 4),
            "unit": unit,
        })
    return rows


class _Grab:
    def __init__(self, demo, kind, full, faces):
        self.demo = demo
        self.kind = kind
        self.full = full
        self.faces = faces
        self.t, self.ang, self.feats = [], [], []
        self.wall_rot = []

    def __call__(self, pred, obs, rgb, index):
        i0 = int(index[0])
        t0 = float(self.demo["rgb_t"][0])
        self.t.append(round(float(self.demo["rgb_t"][rgb] - t0), 3))
        self.ang.append(round(_pred_angle(self.kind, pred), 3))
        self.feats.append(_feats(obs, self.demo["foot"][i0]))
        if self.kind != "causal":
            return
        frame = self.demo["frame"][i0]
        got = frame @ rot6d_rows_to_R(pred[0:1, 4:10])[0]
        err = float(geodesic_deg(got[None], self.demo["tool_R"][i0][None])[0])
        tri = int(input_check.o3d_closest(self.full, self.demo["foot"][i0].reshape(1, 3))[0])
        if self.faces[tri] == "wall":
            self.wall_rot.append(err)


HOST_DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")


def _host_path(path: str) -> str:
    """Checkpoints store container paths under /data. The host copy is the odom-corr tree."""
    text = str(path)
    if text.startswith("/data/"):
        host = HOST_DATA / text[len("/data/"):]
        if host.exists():
            return str(host)
    return text


def _host_zarr(cfg_path: str) -> str:
    host = _host_path(cfg_path)
    if not (Path(host) / "data").is_dir():
        raise SystemExit(f"no host zarr for {cfg_path} ({host})")
    return host


def _train_ds(cfg, norm):
    ds_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    apply_drop_scan_inputs(ds_cfg)
    apply_drop_depth_inputs(ds_cfg)
    zarr_path = _host_zarr(str(cfg.task.dataset.dataset_path))
    OmegaConf.update(ds_cfg, "task.dataset_path", zarr_path, merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.dataset_path", zarr_path, merge=False)
    for key in ("long_dt_mask_path", "pose_jump_flag_path"):
        if key in ds_cfg.task.dataset and ds_cfg.task.dataset[key]:
            OmegaConf.update(ds_cfg, f"task.dataset.{key}", _host_path(str(ds_cfg.task.dataset[key])), merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.action_normalizer_path", str(norm), merge=False)
    for key in ("exclusion_stats_path", "filter_stats_path"):
        if key in ds_cfg.task.dataset:
            OmegaConf.update(ds_cfg, f"task.dataset.{key}", None, merge=False)
    return heldout.fair.hydra_instantiate(ds_cfg.task.dataset)


def _train_rows(policy, dataset, root, device, kind):
    n = len(dataset.sampler.indices)
    chosen = np.unique(np.linspace(0, n - 1, 200).astype(int))
    angs, feats = [], []
    for done, idx in enumerate(chosen):
        obs, _sample = dataset.sampler.sample_sequence(int(idx))
        pred = np.asarray(predict_sparse_action(policy, obs, device), dtype=np.float64)
        epi, rgb = dataset.sampler.indices[int(idx)]
        ep = root[f"episode_{int(epi)}"]
        rgb_t = float(np.asarray(ep["rgb_time_stamps_0"]).reshape(-1)[int(rgb)])
        robot_t = np.asarray(ep["robot_time_stamps_0"]).reshape(-1)
        j = int(np.clip(np.searchsorted(robot_t, rgb_t), 1, len(robot_t) - 1))
        if abs(rgb_t - robot_t[j - 1]) <= abs(rgb_t - robot_t[j]):
            j -= 1
        foot = np.asarray(ep["mold_x_s_0"][j], dtype=np.float64)
        angs.append(_pred_angle(kind, pred))
        feats.append(_feats(obs, foot))
        if done and done % 40 == 0:
            print(f"[wall] train {kind} {done}/{len(chosen)}", flush=True)
    return angs, feats


def _print_flags(letter, kind, rows):
    flagged = [row for row in rows if row["flag"]]
    print(f"[wall] {letter} {kind}  {len(flagged)}/{len(rows)} inputs outside the training range", flush=True)
    for row in flagged:
        print(
            f"    {row['name']:10} thin p50 {row['thin_p50']:9.3f}  "
            f"train [{row['train_min']:.3f}, {row['train_max']:.3f}] p50 {row['train_p50']:.3f}  "
            f"outside {row['frac']:.2f}",
            flush=True,
        )


def _page(gt, grabs, compared):
    """compared[letter][name] = channel rows. grabs[letter][name] = _Grab."""
    recordings = {}
    for letter, rec in gt["recordings"].items():
        causal = grabs[letter].get("causal e199")
        heading = grabs[letter].get("heading e100")
        rows = compared[letter]["causal e199"]
        flagged = [row for row in rows if row["flag"] and row["unit"] is not None]
        flagged.sort(key=lambda row: -row["frac"])
        flagged = flagged[:8]
        recordings[letter] = {
            "t": rec["series"]["t"],
            "wall_mm": rec["series"]["wall_mm"],
            "full_face": rec["series"]["full_face"],
            "note": rec["series"]["note"],
            "ood_t": causal.t if causal else [],
            "ood_ht": heading.t if heading else [],
            "causal_deg": causal.ang if causal else [],
            "heading_deg": heading.ang if heading else [],
            "names": [row["name"] for row in flagged],
            "units": [row["unit"] for row in flagged],
            "ood_note": "predicted |ΔR|   "
            + (", ".join(row["name"] for row in flagged) or "no low-dim input outside training on ≥10% of windows")
            + "   right axis 0–1 = training min–max",
        }
    return {"recordings": recordings}


def main() -> None:
    heldout._check()
    full = heldout.FixBMoldField(str(heldout.WALL))
    wall = wall_field(full, heldout.WALL)
    axis = long_axis(full.vertices)
    gt, faces = _ground_truth(full, wall, axis)
    spec = json.loads((HERE / "checkpoints.json").read_text())
    root = Path(spec.get("train_root") or heldout.TRAIN)
    found = []
    for entry in spec["models"]:
        path = heldout._resolve(entry, root)
        if path is None:
            print(f"[wall] skip {entry['name']}", flush=True)
            continue
        found.append((entry, path))
    t_mid = json.loads((heldout.SCRIPTS / "eval/thin_variants/contact_half.json").read_text())["episodes"]["C"]["t_mid_s"]
    recordings = {
        "C": {"path": HERE / "C_first" / "replay.zarr"},
        "A": {"path": HERE / "A_full" / "replay.zarr"},
    }
    for letter, rec in recordings.items():
        opened = heldout._demo(rec["path"], None)
        cutoff = None if letter == "A" else float(opened["rgb_t"][0]) + float(t_mid)
        demo = opened if cutoff is None else heldout._demo(rec["path"], cutoff)
        demo["axis"] = axis
        rec["demo"] = demo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = {}
    tables = {"C": [], "A": [], "combined": []}
    report = {"foot": "wall faces only", "r1rel": "no foot in the decode; copied from score.json", "recordings": {}}
    grabs = {"C": {}, "A": {}}
    train_cache = {}
    old = json.loads(OLD.read_text())
    for entry, ckpt in found:
        name = entry["name"]
        if name.startswith("r1rel"):
            for letter in ("C", "A", "combined"):
                block = old["recordings"][letter][name]
                if letter != "combined":
                    tables[letter].append((name, block))
                report["recordings"].setdefault(letter, {})[name] = block
            tables["combined"].append((name, old["recordings"]["combined"][name]))
            print(f"[wall] {name} copied", flush=True)
            continue
        policy, cfg = load_policy_and_cfg(str(ckpt), device)
        resolved = heldout._norm_of(cfg, entry.get("kind") or "auto")
        if resolved is None:
            print(f"[wall] skip {name}", flush=True)
            del policy
            continue
        norm, _dim, decode, sha = resolved
        use_wall = decode in ("causal", "heading")
        scored = {}
        for letter, rec in recordings.items():
            key = (str(rec["path"]), str(norm))
            if key not in cache:
                cache[key] = heldout._dataset(cfg, rec["path"], norm)
                cache[key + ("q",)] = heldout._queries(cache[key], rec["demo"])
            rows = cache[key + ("q",)]
            grab = None
            if name in WATCH:
                grab = _Grab(rec["demo"], decode, full, faces)
                grabs[letter][name] = grab
            print(f"[wall] score {name} {letter} queries {len(rows)}", flush=True)
            block = heldout._score_one(
                decode, policy, cache[key], full, rec["demo"], rows, device, axis,
                foot_field=wall if use_wall else None, on_query=grab,
            )
            scored[letter] = block
        pooled = heldout._pool([scored["C"], scored["A"]])
        for letter in ("C", "A"):
            public = heldout._public(scored[letter])
            tables[letter].append((name, public))
            report["recordings"].setdefault(letter, {})[name] = {**public, "ckpt": str(ckpt), "decode": decode, "sha": sha[:12]}
        tables["combined"].append((name, pooled))
        report["recordings"].setdefault("combined", {})[name] = pooled
        if name in WATCH:
            zpath = _host_zarr(str(cfg.task.dataset.dataset_path))
            if zpath not in train_cache:
                train_cache[zpath] = (_train_ds(cfg, norm), zarr.open(zpath, "r")["data"])
            dataset, zroot = train_cache[zpath]
            print(f"[wall] train windows {name}", flush=True)
            angs, feats = _train_rows(policy, dataset, zroot, device, decode)
            train_cache[name] = {"ang": angs, "feats": feats}
        del policy
        torch.cuda.empty_cache()
    compared = {}
    ood = {"train_n": 200, "recordings": {}}
    for letter in ("C", "A"):
        compared[letter] = {}
        ood["recordings"][letter] = {}
        for name in WATCH:
            grab = grabs[letter].get(name)
            trained = train_cache.get(name)
            if grab is None or trained is None:
                continue
            rows = _compare(trained["feats"], grab.feats)
            compared[letter][name] = rows
            pred = {
                "train": _stats(trained["ang"]),
                "thin": _stats(grab.ang),
            }
            wall_rot = grab.wall_rot
            ood["recordings"][letter][name] = {
                "pred_angle_deg": pred,
                "wall_foot_step1_rot_deg": None if not wall_rot else {
                    "n": len(wall_rot),
                    "mean": round(float(np.mean(wall_rot)), 2),
                    "median": round(float(np.median(wall_rot)), 2),
                },
                "channels": [{k: v for k, v in row.items() if k != "unit"} for row in rows],
            }
            _print_flags(letter, name, rows)
            print(
                f"[wall] {letter} {name} |ΔR| thin p50 {pred['thin']['p50']}  "
                f"train p50 {pred['train']['p50']} [{pred['train']['min']}, {pred['train']['max']}]",
                flush=True,
            )
            if wall_rot:
                print(
                    f"[wall] {letter} causal step-1 rot on wall feet  n {len(wall_rot)}"
                    f"  mean {np.mean(wall_rot):.1f}°  median {np.median(wall_rot):.1f}°",
                    flush=True,
                )
    report["ground_truth"] = {
        letter: {"gt": rec["gt"], "query_gap_mm": rec["query_gap_mm"], "query_face": rec["query_face"]}
        for letter, rec in gt["recordings"].items()
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "score.json").write_text(json.dumps(report, indent=2))
    (OUT / "ood.json").write_text(json.dumps(ood, indent=2))
    (OUT / "page.json").write_text(json.dumps(_page(gt, grabs, compared)))
    heldout._print_table("C  wall-only foot", tables["C"])
    heldout._print_table("A  wall-only foot", tables["A"])
    heldout._print_table("combined  wall-only foot", tables["combined"])
    print(f"[wall] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
