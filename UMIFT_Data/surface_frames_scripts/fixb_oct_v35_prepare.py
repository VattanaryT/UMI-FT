#!/usr/bin/env python3
"""Audit stringer_v3.5 and build Fix B r1/r2 from tracked feet.

The mold package is the September field (scan_20261008_170729): same npz as
stringer_scanner_v2_fixb. Labels are fixb_v4_tracked (tracked foot, v_par from
the foot, delta_n from the smoothed feedback pose, 0.1 s smooth). R_off is fit
on the stored tracked frames inside the 3 cm band, stride 20, same
fit_heading_offset as September.

  python fixb_oct_v35_prepare.py            # audit + label cache
  python fixb_oct_v35_prepare.py --build    # also write r1 and r2 zarrs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import zarr

from fixb_build_datasets import MOLD_KEYS, link_tree, replace_array
from fixb_field import FixBMoldField
from fixb_labels import (
    episode_arrays,
    fit_heading_offset,
    fixb_pose7_command,
    identity_pose7,
    pose7_to_Rp,
)
from fixb_labels_tracked import episode_arrays_tracked

OCT = Path("/mnt/data/umidata/ft/datasets/stringer_v3.5/acp_replay_buffer.zarr")
SEP = Path("/mnt/data/umidata/ft/datasets/stringer_scanner_v2/acp_replay_buffer.zarr")
FIELD_NPZ = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_data_v2_full_fixb/shared/fixb_field.npz"
)
SCANNER_NPZ = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb/shared/fixb_field.npz"
)
OUT = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb")
TAG_DIR = Path("/mnt/data/umidata/ft/datasets/stringer_v3.5/tag_traj")
UNIFIED_DIR = Path("/mnt/data/umidata/ft/datasets/stringer_v3.5/zarr")
SMOOTH_S = 0.1
NEAR_M = 0.03
LABEL_VERSION = "fixb_v4_tracked"
PAD_S = 5.0
IDLE_SPEED_MPS = 0.005
IDLE_MIN_S = 2.0
JUMP_M = 0.02


def episode_names(group) -> list[str]:
    return sorted(
        (k for k in group.keys() if str(k).startswith("episode_")),
        key=lambda s: int(str(s).split("_")[-1]),
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stream_stats(ts: np.ndarray) -> dict:
    ts = np.asarray(ts, dtype=np.float64).reshape(-1)
    if len(ts) < 2:
        return {"n": int(len(ts)), "hz_median": None}
    dt = np.diff(ts)
    pos = dt[np.isfinite(dt) & (dt > 0)]
    med = float(np.median(pos)) if len(pos) else float("nan")
    return {
        "n": int(len(ts)),
        "span_s": float(ts[-1] - ts[0]),
        "hz_median": float(1.0 / med) if med > 0 else None,
        "dt_median_s": med,
        "n_gap_gt_3x_median": int(np.sum(pos > 3.0 * med)) if med > 0 else 0,
        "n_gap_gt_0p5s": int(np.sum(pos > 0.5)),
        "max_gap_s": float(np.max(pos)) if len(pos) else None,
        "n_nonincreasing": int(np.sum(dt <= 0)),
    }


def idle_stretches(t: np.ndarray, p: np.ndarray) -> list[dict]:
    """Contiguous low-speed runs at t >= PAD_S. Speed is |Δp|/Δt."""
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    ok = dt > 1e-6
    speed = np.zeros_like(step)
    speed[ok] = step[ok] / dt[ok]
    # A sample i (i>=1) is idle if the step that arrived there is slow and t_i >= pad.
    idle = np.zeros(len(t), dtype=bool)
    idle[1:] = (speed < IDLE_SPEED_MPS) & (t[1:] >= PAD_S) & ok
    stretches = []
    i = 1
    while i < len(t):
        if not idle[i]:
            i += 1
            continue
        j = i
        while j < len(t) and idle[j]:
            j += 1
        dur = float(t[j - 1] - t[i - 1])
        if dur >= IDLE_MIN_S:
            stretches.append(
                {
                    "t0": float(t[i - 1]),
                    "t1": float(t[j - 1]),
                    "dur_s": dur,
                }
            )
        i = j
    return stretches


def audit_zarr(path: Path, *, with_motion: bool) -> dict:
    root = zarr.open(str(path), "r")
    data = root["data"]
    names = episode_names(data)
    keys0 = set(data[names[0]].keys())
    info = json.loads(root.attrs["conversion_info"])
    spans = []
    rates = {k: [] for k in ("rgb", "depth", "robot", "wrench")}
    gaps = {k: {"n_gap_gt_3x": 0, "n_gap_gt_0p5s": 0, "max_gap_s": 0.0, "episodes": []} for k in rates}
    motion = {
        "n_pose_jump_gt_2cm": 0,
        "n_abs_pos_gt_2m": 0,
        "n_nonfinite_pose": 0,
        "idle_stretches": [],
        "idle_total_s": 0.0,
        "max_abs_pos_m": 0.0,
    }
    shapes = {}
    ep0 = data[names[0]]
    for k in sorted(keys0):
        a = ep0[k]
        shapes[k] = {"shape": list(a.shape), "dtype": str(a.dtype)}
    for name in names:
        ep = data[name]
        rt = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        spans.append(float(rt[-1] - rt[0]))
        streams = {
            "rgb": np.asarray(ep["rgb_time_stamps_0"]),
            "depth": np.asarray(ep["depth_time_stamps_0"]),
            "robot": rt,
            "wrench": np.asarray(ep["wrench_time_stamps_0"]),
        }
        for key, ts in streams.items():
            st = stream_stats(ts)
            rates[key].append(st["hz_median"])
            gaps[key]["n_gap_gt_3x"] += st["n_gap_gt_3x_median"]
            gaps[key]["n_gap_gt_0p5s"] += st["n_gap_gt_0p5s"]
            if st["max_gap_s"] and st["max_gap_s"] > gaps[key]["max_gap_s"]:
                gaps[key]["max_gap_s"] = st["max_gap_s"]
            if st["n_gap_gt_0p5s"]:
                gaps[key]["episodes"].append(name)
        if with_motion:
            p = np.asarray(ep["ts_pose_fb_0"], dtype=np.float64)[:, :3]
            motion["n_nonfinite_pose"] += int((~np.isfinite(p)).any(axis=1).sum())
            motion["max_abs_pos_m"] = max(motion["max_abs_pos_m"], float(np.nanmax(np.abs(p))))
            motion["n_abs_pos_gt_2m"] += int((np.linalg.norm(p, axis=1) > 2.0).sum())
            dt = np.diff(rt)
            step = np.linalg.norm(np.diff(p, axis=0), axis=1)
            after = rt[1:] >= PAD_S
            motion["n_pose_jump_gt_2cm"] += int(np.sum((step > JUMP_M) & after & (dt > 0) & (dt < 0.05)))
            stretches = idle_stretches(rt, p)
            if stretches:
                motion["idle_stretches"].append(
                    {"episode": name, "n": len(stretches), "total_s": float(sum(s["dur_s"] for s in stretches)),
                     "longest_s": float(max(s["dur_s"] for s in stretches))}
                )
                motion["idle_total_s"] += float(sum(s["dur_s"] for s in stretches))
    def med(xs):
        a = np.asarray([x for x in xs if x is not None], dtype=np.float64)
        return float(np.median(a)) if len(a) else None

    return {
        "path": str(path),
        "n_episodes": len(names),
        "duration_s_including_5s_pad": float(sum(spans)),
        "duration_s_per_episode_median": float(np.median(spans)),
        "duration_s_per_episode_min": float(np.min(spans)),
        "duration_s_per_episode_max": float(np.max(spans)),
        "keys": sorted(keys0),
        "shapes_episode0": shapes,
        "hz_median": {k: med(v) for k, v in rates.items()},
        "timestamp_gaps": gaps,
        "motion": motion if with_motion else None,
        "pose_source": info.get("pose_source"),
        "frame_convention": info.get("frame_convention_version"),
        "world_frame_override": info.get("world_frame_override"),
        "start_pad_seconds": root.attrs.get("start_pad_seconds"),
        "depth_source": info.get("depth_source"),
        "wrench_source": info.get("wrench_source"),
        "has_scan_heatmap": "scan_heatmap_0" in keys0,
        "has_scan_gap": "scan_gap_intervals_0" in keys0,
    }


def tracking_from_tags(names: list[str]) -> dict:
    unified = sorted(UNIFIED_DIR.glob("*_unified.zarr"))
    if len(unified) != len(names):
        return {"error": f"unified {len(unified)} != episodes {len(names)}"}
    rows = []
    n_zero = 0
    n_all = 0
    dropout_stretches = []
    for name, uni in zip(names, unified):
        tag = TAG_DIR / (uni.name.replace("_unified.zarr", "_ee_master.npz"))
        z = np.load(tag, allow_pickle=True)
        vis = np.asarray(z["n_tags_visible"]).reshape(-1)
        ts = np.asarray(z["timestamp_sec_unified"], dtype=np.float64).reshape(-1)
        n_zero += int(np.sum(vis == 0))
        n_all += int(len(vis))
        # dropout runs
        i = 0
        longest = 0.0
        n_runs = 0
        while i < len(vis):
            if vis[i] != 0:
                i += 1
                continue
            j = i
            while j < len(vis) and vis[j] == 0:
                j += 1
            dur = float(ts[j - 1] - ts[i]) if j > i + 1 else 0.0
            if j - i >= 2 and dur >= 0.2:
                n_runs += 1
                longest = max(longest, dur)
            i = j
        if n_runs:
            dropout_stretches.append(
                {"episode": name, "bag": uni.name, "n_runs": n_runs, "longest_s": longest,
                 "frac_zero": float(np.mean(vis == 0))}
            )
        rows.append(
            {
                "episode": name,
                "bag": uni.name,
                "tag_residual_rms_m": float(np.asarray(z["tag_residual_rms"]).reshape(-1)[0]),
                "frac_zero_tags": float(np.mean(vis == 0)),
                "min_tags": int(vis.min()),
                "median_tags": float(np.median(vis)),
            }
        )
    rms = np.array([r["tag_residual_rms_m"] for r in rows])
    return {
        "n_tag_samples": n_all,
        "n_zero_tag_samples": n_zero,
        "frac_zero_tag_samples": float(n_zero / max(n_all, 1)),
        "episodes_with_dropout_run_ge_0p2s": dropout_stretches,
        "tag_residual_rms_m": {
            "median": float(np.median(rms)),
            "max": float(np.max(rms)),
            "p90": float(np.percentile(rms, 90)),
        },
        "per_episode": rows,
    }


def label_pass(field: FixBMoldField, src, names: list[str], cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    label_cfg = {
        "label_version": LABEL_VERSION,
        "v_par_mode": "foot",
        "delta_n_source": "fb",
        "smooth_window_s": SMOOTH_S,
        "foot": "tracked",
    }
    L = []
    dn_abs_max = 0.0
    dn_at = None
    over = []
    per_ep = []
    t0 = time.time()
    for i, name in enumerate(names):
        cache = cache_dir / f"{name}.npz"
        ep = src["data"][name]
        # Heading offset uses the raw closest-point frames (September's first pass).
        # Stored mold arrays are the tracked-foot labels.
        raw = episode_arrays(
            field, ep, smooth_window_s=SMOOTH_S, v_par_mode="foot", delta_n_source="fb"
        )
        R_tool, _ = pose7_to_Rp(ep["ts_pose_fb_0"])
        R_f = raw["mold_R_field_0"].astype(np.float64).reshape(-1, 3, 3)
        band = np.abs(raw["mold_signed_distance_0"][:, 0]) < NEAR_M
        if band.any():
            L.append(np.einsum("nji,njk->nik", R_f[band][::20], R_tool[band][::20]))
        del raw
        if cache.exists():
            arr = {k: np.load(cache)[k] for k in np.load(cache).files if k != "label_cfg"}
        else:
            arr = episode_arrays_tracked(
                field, ep, smooth_window_s=SMOOTH_S, v_par_mode="foot", delta_n_source="fb"
            )
            np.savez_compressed(cache, label_cfg=json.dumps(label_cfg), **arr)
        dn = np.asarray(arr["fixb_delta_n_0"], dtype=np.float64).reshape(-1)
        rt = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        ip_p = np.asarray(ep["ts_pose_fb_0"], dtype=np.float64)[:, :3]
        x = np.asarray(arr["mold_x_s_0"], dtype=np.float64).reshape(-1, 3)
        n = np.asarray(arr["mold_normal_0"], dtype=np.float64).reshape(-1, 3)
        # in-plane offset of the unsmoothed tool (same definition as the r4 report)
        sd_raw = np.sum((ip_p - x) * n, axis=1)
        ip = np.sqrt(np.maximum(np.linalg.norm(ip_p - x, axis=1) ** 2 - sd_raw**2, 0.0))
        abs_dn = np.abs(dn)
        j = int(np.argmax(abs_dn))
        if abs_dn[j] > dn_abs_max:
            dn_abs_max = float(abs_dn[j])
            dn_at = {
                "episode": name,
                "t_s": float(rt[j]) if j < len(rt) else None,
                "delta_n_m": float(dn[j]),
                "in_plane_m": float(ip[j]),
                "in_start_pad": bool(rt[j] < PAD_S) if j < len(rt) else None,
            }
        big = np.flatnonzero(abs_dn > 0.05)
        if len(big):
            # keep the worst 8 per episode
            order = big[np.argsort(-abs_dn[big])][:8]
            over.append(
                {
                    "episode": name,
                    "n_steps": int(len(big)),
                    "frac": float(len(big) / len(dn)),
                    "max_abs_m": float(abs_dn[big].max()),
                    "worst": [
                        {
                            "t_s": float(rt[k]),
                            "delta_n_m": float(dn[k]),
                            "in_plane_m": float(ip[k]),
                            "in_start_pad": bool(rt[k] < PAD_S),
                        }
                        for k in order
                    ],
                }
            )
        per_ep.append(
            {
                "episode": name,
                "n": int(len(dn)),
                "dn_min": float(dn.min()),
                "dn_max": float(dn.max()),
                "dn_abs_p50": float(np.median(abs_dn)),
                "frac_abs_gt_5cm": float(np.mean(abs_dn > 0.05)),
                "frac_in_plane_gt_5mm": float(np.mean(ip > 0.005)),
                "in_plane_p50_m": float(np.median(ip)),
            }
        )
        print(
            f"[label {i + 1}/{len(names)}] {name} N={len(dn)} "
            f"|dn|max={abs_dn.max():.4f} in_plane_p50={np.median(ip)*1e3:.1f}mm "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
    R_off, flip, head_err = fit_heading_offset(np.concatenate(L))
    tilt = float(np.degrees(np.arccos(np.clip(-R_off[2, 2], -1.0, 1.0))))
    dn_all_max = max(r["dn_max"] for r in per_ep)
    dn_all_min = min(r["dn_min"] for r in per_ep)
    return {
        "elapsed_s": time.time() - t0,
        "R_off": R_off.tolist(),
        "R_off_tool_z_tilt_from_minus_n_deg": tilt,
        "heading_flipped_frac": float(flip.mean()),
        "heading_residual_deg_p50_p90": np.percentile(head_err, [50, 90]).tolist(),
        "delta_n_m": {
            "min": dn_all_min,
            "max": dn_all_max,
            "max_abs": dn_abs_max,
            "max_abs_at": dn_at,
            "episodes_with_abs_gt_5cm": over,
            "n_episodes_with_abs_gt_5cm": len(over),
        },
        "per_episode_summary": per_ep,
        "label_cfg": label_cfg,
    }


def build_r1_r2(src_path: Path, names: list[str], cache_dir: Path, meta: dict, out_root: Path) -> None:
    import shutil

    shared = out_root / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FIELD_NPZ, shared / "fixb_field.npz")
    field_json = FIELD_NPZ.with_suffix(".json")
    if field_json.exists():
        shutil.copy2(field_json, shared / "fixb_field.json")
    (shared / "fixb_meta.json").write_text(json.dumps(meta, indent=2))
    arrays = {}
    for name in names:
        loaded = np.load(cache_dir / f"{name}.npz")
        arrays[name] = {k: loaded[k] for k in MOLD_KEYS}
    for variant in ("r1_world_world", "r2_world_fixb"):
        zpath = out_root / variant / "acp_replay_buffer.zarr"
        print(f"[{variant}] linking {src_path} -> {zpath}", flush=True)
        link_tree(src_path, zpath)
        root = zarr.open(str(zpath), "a")
        root.attrs.update(
            {
                "mold_aware_variant": variant,
                "fixb": meta,
                "world_frame": "fixture",
                "label_version": LABEL_VERSION,
            }
        )
        for name in names:
            ep = root["data"][name]
            arr = arrays[name]
            for key in MOLD_KEYS:
                replace_array(ep, key, arr[key])
            ep.attrs.update(
                {
                    "pose_variant": variant,
                    "label_version": LABEL_VERSION,
                    "world_frame": "fixture",
                    "mold_id": meta["mold_id"],
                }
            )
            if variant == "r1_world_world":
                continue
            replace_array(
                ep,
                "ts_pose_command_0",
                fixb_pose7_command(arr["fixb_v_par_0"], arr["fixb_delta_n_0"]),
                ep["ts_pose_command_0"],
            )
            replace_array(
                ep,
                "ts_pose_virtual_target_0",
                identity_pose7(len(arr["fixb_v_par_0"])),
                ep["ts_pose_virtual_target_0"],
            )
        print(f"[{variant}] wrote {len(names)} episodes", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--skip-labels", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    audit_dir = OUT / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    print("[audit] october vs september streams", flush=True)
    oct_a = audit_zarr(OCT, with_motion=True)
    sep_a = audit_zarr(SEP, with_motion=True)
    src = zarr.open(str(OCT), "r")
    names = episode_names(src["data"])
    print("[audit] tag tracking", flush=True)
    tags = tracking_from_tags(names)
    field_same = sha256_file(FIELD_NPZ) == sha256_file(SCANNER_NPZ)
    field_meta = json.loads(FIELD_NPZ.with_suffix(".json").read_text())

    report = {
        "october": oct_a,
        "september": {
            "path": sep_a["path"],
            "n_episodes": sep_a["n_episodes"],
            "duration_s_including_5s_pad": sep_a["duration_s_including_5s_pad"],
            "hz_median": sep_a["hz_median"],
            "keys_only_in_september": sorted(set(sep_a["keys"]) - set(oct_a["keys"])),
            "keys_only_in_october": sorted(set(oct_a["keys"]) - set(sep_a["keys"])),
            "timestamp_gaps": sep_a["timestamp_gaps"],
            "motion": sep_a["motion"],
            "has_scan_heatmap": sep_a["has_scan_heatmap"],
        },
        "mold": {
            "mold_id": field_meta["mold_id"],
            "mesh_version": field_meta["mesh_version"],
            "field_version": field_meta["field_version"],
            "field_npz_sha256": sha256_file(FIELD_NPZ),
            "scanner_v2_field_npz_identical": field_same,
            "scan_ply": field_meta["scan_ply"],
        },
        "demonstrator": (
            "Not recorded. zarr conversion_info, episode attrs, bag metadata.yaml, "
            "and tag trajectories have timestamps and sensor topics only. "
            "No demonstrator name in the October dataset."
        ),
        "scan_markers": "50/50 *_scan_markers.json files have events: []",
        "tracking": {k: tags[k] for k in tags if k != "per_episode"},
        "tracking_per_episode": tags.get("per_episode"),
    }
    if not args.skip_labels:
        print("[labels] tracked feet + raw heading fit", flush=True)
        field = FixBMoldField(FIELD_NPZ)
        labels = label_pass(field, src, names, OUT / "shared" / "episode_cache")
        report["labels"] = labels
        meta = {
            "label_version": LABEL_VERSION,
            "v_par_mode": "foot",
            "delta_n_source": "fb",
            "smooth_window_s": SMOOTH_S,
            "foot": "tracked (independent closest point, hops repaired from previous foot + tool step)",
            "source_zarr": str(OCT),
            "n_episodes": len(names),
            "mold_id": field_meta["mold_id"],
            "mesh_version": field_meta["mesh_version"],
            "field_version": field_meta["field_version"],
            "field_npz_sha256": report["mold"]["field_npz_sha256"],
            "R_off": labels["R_off"],
            "R_off_source": "fit_heading_offset on raw closest-point frames, |sd|<0.03 m, stride 20",
            "R_off_tool_z_tilt_from_minus_n_deg": labels["R_off_tool_z_tilt_from_minus_n_deg"],
            "heading_flipped_frac": labels["heading_flipped_frac"],
            "heading_residual_deg_p50_p90": labels["heading_residual_deg_p50_p90"],
            "R_off_convention": "R_tool ~= R_field(x_s) @ B, B in {R_off, diag(-1,-1,1)@R_off} from measured tool at chunk start",
            "scan_in_policy": False,
            "scan_arrays_present": False,
            "decode": "x_s0 = tracked_foot(p_meas); x_s+ = foot(x_s + R[:, :2] v dt); x_d = x_s + delta_n n; R_d = R(x_s) B",
            "code_commit_note": "surface_frame includes twist fix 2d9f24f, jump mask efd2e04, mesh_jump_hold in mold_aware_runtime",
        }
        (OUT / "shared").mkdir(parents=True, exist_ok=True)
        (OUT / "shared" / "fixb_meta.json").write_text(json.dumps(meta, indent=2))
        if args.build:
            build_r1_r2(OCT, names, OUT / "shared" / "episode_cache", meta, OUT)
    # Drop the bulky per-episode tracking list from the short file; keep a sidecar.
    side = report.pop("tracking_per_episode", None)
    (audit_dir / "audit.json").write_text(json.dumps(report, indent=2))
    if side is not None:
        (audit_dir / "tracking_per_episode.json").write_text(json.dumps(side, indent=2))
    print(f"[audit] wrote {audit_dir / 'audit.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
