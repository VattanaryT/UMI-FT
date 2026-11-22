#!/usr/bin/env python3
"""Open-loop viewer. Reads export_openloop.py directories. No policy.

  python UMIFT_Data/surface_frames_scripts/view_openloop.py \\
    --export /mnt/data/umidata/ft/open_loop_exports/r4vt_<sha> \\
    --mesh /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz

  # optional: old frames faded, new frames solid, old jumps marked
  python UMIFT_Data/surface_frames_scripts/view_openloop.py \\
    --export /mnt/data/umidata/ft/open_loop_exports/r4vt_<sha> \\
    --mesh /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz \\
    --frames_npz /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_frame_continuity_causal/train_episode_frames.npz

--heading-tilt adds the 8-d heading-tilt checkpoint and causal-foot e199.
Existing exports stay as they are. The new frames and the ψ panel appear only
when that flag is set.

Mesh loading, fixture-frame rounding, and bad-span shading are imported from
view_oct. The camera, time slider, and plot shading in this page are the same
JavaScript functions. They are copied here because they live inside view_oct's
HTML string and cannot be imported without changing that page.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import openloop_lib as ol
from view_oct import decimate_mesh, mesh_payload, rounded, shade_spans, stride_idx

PATH_CAP = 700
SERIES_CAP = 1200
COLORS = ("#e74c3c", "#2ecc71", "#f4d03f", "#3498db", "#e67e22")
EXPORT_ROOT = Path("/mnt/data/umidata/ft/open_loop_exports")
LABEL_COLOR = "#5dade2"
PRED_COLOR = "#ff9f1a"


def exports_for_checkpoint(ckpt: Path) -> list[Path]:
    """Export directories whose settings.json names this checkpoint."""
    want = ckpt.expanduser()
    try:
        want_key = str(want.resolve())
    except OSError:
        want_key = str(want)
    found = []
    if not EXPORT_ROOT.is_dir():
        return found
    for settings_path in sorted(EXPORT_ROOT.glob("*/settings.json")):
        data = json.loads(settings_path.read_text())
        got = data.get("ckpt")
        if not got:
            continue
        try:
            got_key = str(Path(got).expanduser().resolve())
        except OSError:
            got_key = str(got)
        if got_key == want_key:
            found.append(settings_path.parent)
    return found


def prediction_source(blob):
    """(pos, R, n_valid) for the chunk the toggle draws.

    raw_pos is the full decoded chunk. Older exports stored that full chunk in
    pred_pos instead, and the 6-step files only have chunk_pos.
    """
    if "raw_pos" in blob.files:
        pos = np.asarray(blob["raw_pos"], dtype=np.float64)
        rot = _matching_rot(blob, pos, ("raw_R", "pred_R"))
        return pos, rot, _clipped_count(blob, pos, "n_valid")
    if "pred_pos" in blob.files:
        pos = np.asarray(blob["pred_pos"], dtype=np.float64)
        if pos.ndim == 3 and pos.shape[1] > 1:
            rot = _matching_rot(blob, pos, ("pred_R",))
            return pos, rot, _clipped_count(blob, pos, "n_valid")
    if "chunk_pos" in blob.files:
        pos = np.asarray(blob["chunk_pos"], dtype=np.float64)
        rot = _matching_rot(blob, pos, ("chunk_R",))
        return pos, rot, _clipped_count(blob, pos, "chunk_n")
    return None, None, None


def _clipped_count(blob, pos: np.ndarray, key: str) -> np.ndarray:
    if key in blob.files:
        count = np.asarray(blob[key])
    else:
        count = np.full(len(pos), pos.shape[1])
    return np.minimum(count, pos.shape[1])


def _matching_rot(blob, pos: np.ndarray, keys: tuple[str, ...]):
    for key in keys:
        if key not in blob.files:
            continue
        rot = np.asarray(blob[key], dtype=np.float64)
        if rot.shape[:2] == pos.shape[:2]:
            return rot
    return None


def mask_spans(times: np.ndarray, mask: np.ndarray) -> list:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    spans = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = float(times[i])
        elif not flag and start is not None:
            spans.append([start, float(times[i - 1])])
            start = None
    if start is not None:
        spans.append([start, float(times[-1])])
    return spans


def checkpoint_label(settings: dict, fallback: str) -> str:
    """r4vt ep185 (val 0.079) from the export settings.

    r1rel predicts the tool pose. r4vt predicts foot velocity, δn, and ΔR,
    and the 3D point drawn for it is the virtual target.
    """
    role = settings.get("role") or fallback
    epoch = settings.get("checkpoint_epoch")
    text = str(role) if epoch is None else f"{role} ep{int(epoch)}"
    if role == "r1rel":
        text = f"{text} tool pose"
    val = settings.get("val_loss")
    if val is None:
        return text
    return f"{text} (val {float(val):.3f})"


def _time_cap(times: np.ndarray, floor: float) -> float:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if len(times) < 2:
        return floor
    return max(floor, 5.0 * float(np.median(np.diff(times))))


def edge_gaps(index, bad=None, long=None, times=None, dt_cap=None) -> np.ndarray:
    """True on edges that jump a sample, a long-dt flag, or an excluded span."""
    index = np.asarray(index)
    n_edge = max(0, len(index) - 1)
    gap = np.zeros(n_edge, dtype=bool)
    if n_edge == 0:
        return gap
    step = np.diff(np.asarray(index, dtype=np.float64))
    gap |= step > float(np.median(step)) + 0.5
    if times is not None and dt_cap is not None:
        gap |= np.diff(np.asarray(times, dtype=np.float64)) > float(dt_cap)
    if bad is not None:
        bad = np.asarray(bad, dtype=bool).reshape(-1)
        gap |= bad[:-1] | bad[1:]
    if long is not None:
        long = np.asarray(long, dtype=bool).reshape(-1)
        gap |= long[:-1] | long[1:]
    return gap


def breaks_for_keep(edge_gap: np.ndarray, keep: np.ndarray) -> list:
    """Break a downsampled segment when any original edge inside it is a gap."""
    keep = np.asarray(keep, dtype=int)
    return [bool(np.any(edge_gap[int(a) : int(b)])) for a, b in zip(keep[:-1], keep[1:])]


def spans_from_edges(times, edge_gap, t0: float) -> list:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    spans = []
    i = 0
    n = len(edge_gap)
    while i < n:
        if not edge_gap[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and edge_gap[j + 1]:
            j += 1
        spans.append([round(float(times[i] - t0), 3), round(float(times[j + 1] - t0), 3)])
        i = j + 1
    return spans


def _step_deg(R: np.ndarray) -> np.ndarray:
    """Per-sample geodesic step of a frame series, degrees. Sample 0 is 0."""
    R = np.asarray(R, dtype=np.float64)
    out = np.zeros(len(R), dtype=np.float64)
    if len(R) > 1:
        rel = np.einsum("nji,njk->nik", R[:-1], R[1:])
        c = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
        out[1:] = np.degrees(np.arccos(c))
    return out


def _event_rows(blob, prefix: str, with_xyz: bool) -> list:
    if f"{prefix}_episode" not in blob.files:
        return []
    rows = []
    n = len(blob[f"{prefix}_episode"])
    xyz = np.asarray(blob[f"{prefix}_xyz"]) if with_xyz and f"{prefix}_xyz" in blob.files else None
    for i in range(n):
        row = {
            "ep": int(blob[f"{prefix}_episode"][i]),
            "i": int(blob[f"{prefix}_index"][i]),
            "time_s": float(blob[f"{prefix}_time_s"][i]),
            "frame": round(float(blob[f"{prefix}_frame_deg"][i]), 3),
            "tool": round(float(blob[f"{prefix}_tool_deg"][i]), 3),
        }
        if xyz is not None:
            row["x"] = [round(float(v), 4) for v in xyz[i]]
        rows.append(row)
    return rows


def load_frames_npz(path: Path) -> dict:
    """epN -> old/new feet and frames. Optional event tables stay under "events"."""
    blob = np.load(path)
    out = {}
    for name in blob.files:
        if not (name.startswith("ep") and name.endswith("_R_old")):
            continue
        ep = int(name[2:].split("_", 1)[0])
        prefix = f"ep{ep}_"
        pack = {
            "x_old": np.asarray(blob[f"{prefix}x_old"]),
            "x_new": np.asarray(blob[f"{prefix}x_new"]),
            "R_old": np.asarray(blob[f"{prefix}R_old"]),
            "R_new": np.asarray(blob[f"{prefix}R_new"]),
            "old_jump": np.asarray(blob[f"{prefix}old_jump"]).astype(bool),
        }
        if f"{prefix}robot_time_s" in blob.files:
            pack["robot_time_s"] = np.asarray(blob[f"{prefix}robot_time_s"], dtype=np.float64)
            pack["t0"] = float(np.asarray(blob[f"{prefix}t0"]).reshape(-1)[0])
            pack["tcp_xyz"] = np.asarray(blob[f"{prefix}tcp_xyz"])
            pack["tcp_R"] = np.asarray(blob[f"{prefix}tcp_R"])
            pack["split"] = str(np.asarray(blob[f"{prefix}split"]).reshape(-1)[0])
        out[ep] = pack
    if not out:
        raise SystemExit(f"no ep*_R_old arrays in {path}")
    switches = _event_rows(blob, "switch", True)
    nearcap = _event_rows(blob, "nearcap", False)
    for row in switches + nearcap:
        pack = out.get(row["ep"])
        if pack and "t0" in pack:
            row["t"] = round(float(row["time_s"]) - float(pack["t0"]), 3)
    if switches or nearcap:
        out["events"] = {"switches": switches, "nearcap": nearcap}
    return out


def frame_episode_ids(frames: dict | None) -> list:
    if not frames:
        return []
    return sorted(ep for ep in frames if isinstance(ep, int))


def _switches_for(frames: dict | None, episode: int, t0: float) -> list:
    events = (frames or {}).get("events") or {}
    rows = []
    for row in events.get("switches") or []:
        if int(row["ep"]) != int(episode):
            continue
        item = {
            "t": round(float(row["time_s"]) - t0, 3),
            "frame": row["frame"],
            "tool": row["tool"],
            "i": row["i"],
        }
        if "x" in row:
            item["x"] = row["x"]
        rows.append(item)
    return rows


def _nearest_index(times: np.ndarray, t: float) -> int:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    i = int(np.searchsorted(times, t))
    if i <= 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    if abs(float(times[i - 1]) - t) < abs(float(times[i]) - t):
        return i - 1
    return i


def frame_overlay(pack: dict, robot_t: np.ndarray, t0: float) -> dict | None:
    """Downsampled |ΔR| and the old jump feet. None when the series length differs."""
    robot_t = np.asarray(robot_t, dtype=np.float64).reshape(-1)
    if len(pack["R_old"]) != len(robot_t) or len(pack["R_new"]) != len(robot_t):
        return None
    d_old = _step_deg(pack["R_old"])
    d_new = _step_deg(pack["R_new"])
    ip = stride_idx(len(robot_t), SERIES_CAP)
    gap = edge_gaps(np.arange(len(robot_t)), times=robot_t, dt_cap=_time_cap(robot_t, 0.05))
    jump = np.asarray(pack["old_jump"], dtype=bool).reshape(-1)
    return {
        "t": rounded(robot_t[ip] - t0, 3),
        "dR_old": rounded(d_old[ip], 3),
        "dR_new": rounded(d_new[ip], 3),
        "break": breaks_for_keep(gap, ip),
        "jump_t": rounded(robot_t[jump] - t0, 3),
        "jump_y": rounded(d_old[jump], 3),
        "jumps": rounded(np.asarray(pack["x_old"])[jump], 4),
    }


def frame_at(pack: dict, robot_t: np.ndarray, t_abs: float) -> dict:
    i = _nearest_index(robot_t, t_abs)
    return {
        "x_old": rounded(np.asarray(pack["x_old"])[i], 4),
        "R_old": rounded(np.asarray(pack["R_old"])[i], 4),
        "x_new": rounded(np.asarray(pack["x_new"])[i], 4),
        "R_new": rounded(np.asarray(pack["R_new"])[i], 4),
        "jump": bool(np.asarray(pack["old_jump"])[i]),
    }


def load_export(path: Path) -> dict:
    settings = json.loads((path / "settings.json").read_text())
    report = {}
    report_path = path / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
    episodes = {}
    for npz in sorted(path.glob("episode_*.npz")):
        episodes[int(npz.stem.split("_")[-1])] = np.load(npz)
    if not episodes:
        raise SystemExit(f"no episode npz in {path}")
    label = checkpoint_label(settings, path.name)
    return {"dir": path, "settings": settings, "report": report, "episodes": episodes, "label": label}


def _replan_indices(rgb, exec_h: int) -> list:
    rgb = np.asarray(rgb, dtype=np.int64)
    chosen = []
    cursor = int(rgb[0]) if len(rgb) else 0
    for i, value in enumerate(rgb):
        value = int(value)
        if value < cursor:
            continue
        chosen.append(i)
        cursor = value + int(exec_h)
    return chosen


def _chunk_lines(pos, n_valid, rgb, horizon: int, full: bool):
    """One broken polyline of the replanned chunks. full uses every valid step."""
    pts = []
    cuts = []
    for index in _replan_indices(rgb, horizon):
        n = int(n_valid[index])
        if not full:
            n = min(n, int(horizon))
        if n <= 0:
            continue
        pts.append(np.asarray(pos[index, :n], dtype=np.float64))
        cuts.extend([False] * (n - 1) + [True])
    if not pts:
        return [], []
    if cuts:
        cuts[-1] = False
    return rounded(np.concatenate(pts, axis=0), 4), cuts


def query_step0(blob, key: str) -> np.ndarray:
    return np.asarray(blob[key])[:, 0]


def query_anchor(blob) -> np.ndarray:
    """Robot sample of each query. That sample's mold_x_s_0 is the label foot."""
    return ol.query_anchor_index(blob["query_rgb_idx"], blob["robot_time_s"], blob["rgb_time_s"])


_DATASETS: dict = {}


def _dataset_group(dataset: str, episode: int):
    if dataset not in _DATASETS:
        import zarr

        _DATASETS[dataset] = zarr.open(dataset, "r")
    return _DATASETS[dataset]["data"][f"episode_{episode}"]


def mold_normal(dataset: str, episode: int):
    group = _dataset_group(dataset, episode)
    if "mold_normal_0" not in group:
        return None
    return np.asarray(group["mold_normal_0"], dtype=np.float64)


def mold_foot_normal(dataset: str, episode: int):
    """Label foot and the normal δn was measured along. mold_normal_0 is that axis."""
    group = _dataset_group(dataset, episode)
    if "mold_x_s_0" not in group or "mold_normal_0" not in group:
        return None
    foot = np.asarray(group["mold_x_s_0"], dtype=np.float64).reshape(-1, 3)
    normal = np.asarray(group["mold_normal_0"], dtype=np.float64).reshape(-1, 3)
    if len(foot) == 0 or len(foot) != len(normal):
        return None
    return foot, normal


def label_virtual_target(foot, normal, delta_n) -> np.ndarray:
    """Foot plus δn along the stored normal. This is the same point as decoded x_d."""
    delta_n = np.asarray(delta_n, dtype=np.float64).reshape(-1)
    return np.asarray(foot, dtype=np.float64) + delta_n[:, None] * np.asarray(normal, dtype=np.float64)


def query_label_vt(blob, dataset: str | None, episode: int, query: int | None = None, n: int | None = None):
    """Label virtual target. One step-0 point per query, or the first n steps of one query."""
    if not dataset or "label_action" not in getattr(blob, "files", ()) or "action_robot_idx" not in blob.files:
        return None
    label = np.asarray(blob["label_action"], dtype=np.float64)
    if label.shape[-1] < 3:
        return None
    pair = mold_foot_normal(dataset, episode)
    if pair is None:
        return None
    foot, normal = pair
    idx = np.asarray(blob["action_robot_idx"], dtype=np.int64)
    dn = label[..., 2]
    if query is None:
        idx = idx[:, 0]
        dn = dn[:, 0]
    else:
        n = int(idx.shape[1] if n is None else n)
        idx = idx[query, :n]
        dn = dn[query, :n]
    known = (idx >= 0) & (idx < len(foot))
    out = np.full((len(idx), 3), np.nan, dtype=np.float64)
    if np.any(known):
        out[known] = label_virtual_target(foot[idx[known]], normal[idx[known]], dn[known])
    return out


def _points_or_null(xyz) -> list:
    rows = []
    for row in np.asarray(xyz, dtype=np.float64).reshape(-1, 3):
        if np.all(np.isfinite(row)):
            rows.append([round(float(v), 4) for v in row])
        else:
            rows.append(None)
    return rows


def _dn_list(values) -> list:
    out = []
    for value in np.asarray(values, dtype=np.float64).reshape(-1):
        out.append(None if not np.isfinite(value) else round(float(value), 4))
    return out


def table_keep(times, settings: dict) -> np.ndarray:
    """Queries for the comparison tables.

    An export that already kept every 4th query is used as stored. A full
    export is reduced to every 4th query in time order.
    """
    order = np.argsort(np.asarray(times, dtype=np.float64), kind="mergesort")
    stored = int(settings.get("query_stride") or 1)
    step = 1 if stored >= 4 else 4
    return order[::step]


def prediction_feet(blob, export_dir: Path, episode: int):
    if "raw_foot" in getattr(blob, "files", ()) and "raw_n" in blob.files:
        return (
            np.asarray(blob["raw_foot"], dtype=np.float64),
            np.asarray(blob["raw_n"], dtype=np.float64),
        )
    side = export_dir.parent / f"{export_dir.name}_feet" / f"episode_{episode}.npz"
    if side.is_file():
        extra = np.load(side)
        if "raw_foot" in extra.files and "raw_n" in extra.files:
            return (
                np.asarray(extra["raw_foot"], dtype=np.float64),
                np.asarray(extra["raw_n"], dtype=np.float64),
            )
    return None, None


def _label_feet(blob, dataset: str | None, episode: int, query: int | None = None, n: int | None = None):
    if not dataset or "action_robot_idx" not in getattr(blob, "files", ()):
        return None, None, None
    label = np.asarray(blob["label_action"], dtype=np.float64)
    if label.shape[-1] < 3:
        return None, None, None
    pair = mold_foot_normal(dataset, episode)
    if pair is None:
        return None, None, None
    foot, normal = pair
    idx = np.asarray(blob["action_robot_idx"], dtype=np.int64)
    dn = label[..., 2]
    if query is None:
        idx, dn = idx[:, 0], dn[:, 0]
    else:
        n = int(idx.shape[1] if n is None else n)
        idx, dn = idx[query, :n], dn[query, :n]
    known = (idx >= 0) & (idx < len(foot))
    out_f = np.full((len(idx), 3), np.nan, dtype=np.float64)
    out_n = np.full((len(idx), 3), np.nan, dtype=np.float64)
    if np.any(known):
        out_f[known] = foot[idx[known]]
        out_n[known] = normal[idx[known]]
    return out_f, out_n, dn


def foot_path_breaks(rgb, bad) -> np.ndarray:
    """True where the segment to the next query should not be drawn.

    A break is a bad_span_0 sample on either query, or a gap larger than the
    usual spacing between stored queries.
    """
    rgb = np.asarray(rgb, dtype=np.int64).reshape(-1)
    bad = np.asarray(bad, dtype=bool).reshape(-1)
    cut = np.zeros(len(rgb), dtype=bool)
    if len(rgb) < 2:
        return cut
    step = np.diff(rgb)
    positive = step[step > 0]
    typical = int(np.median(positive)) if positive.size else 1
    cut[:-1] = (step > typical) | bad[:-1] | bad[1:]
    return cut


def _attach_feet(run: dict, blob, exp: dict, episode: int) -> None:
    if exp["settings"].get("role") != "r4vt":
        return
    dataset = exp["settings"].get("dataset")
    pred_foot, pred_n = prediction_feet(blob, exp["dir"], episode)
    if pred_foot is not None:
        run["pred_foot"] = _points_or_null(pred_foot[:, 0])
        run["pred_n"] = _points_or_null(pred_n[:, 0])
        run["pred_dn"] = _dn_list(np.asarray(blob["pred_action"])[:, 0, 2])
    label_foot, label_n, label_dn = _label_feet(blob, dataset, episode)
    if label_foot is not None:
        run["label_foot"] = _points_or_null(label_foot)
        run["label_n"] = _points_or_null(label_n)
        run["label_dn"] = _dn_list(label_dn)
    if "query_rgb_idx" in blob.files and "bad_span" in blob.files:
        run["foot_break"] = [bool(v) for v in foot_path_breaks(blob["query_rgb_idx"], np.asarray(blob["bad_span"])[:, 0])]


def _attach_chunk_feet(item: dict, blob, exp: dict, episode: int, query: int, n: int) -> None:
    if exp["settings"].get("role") != "r4vt":
        return
    pred_foot, pred_n = prediction_feet(blob, exp["dir"], episode)
    if pred_foot is not None:
        item["pred_foot"] = _points_or_null(pred_foot[query, :n])
        item["pred_n"] = _points_or_null(pred_n[query, :n])
        item["pred_dn"] = _dn_list(np.asarray(blob["pred_action"][query, :n, 2]))
    label_foot, label_n, label_dn = _label_feet(blob, exp["settings"].get("dataset"), episode, query, n)
    if label_foot is not None:
        item["label_foot"] = _points_or_null(label_foot)
        item["label_n"] = _points_or_null(label_n)
        item["label_dn"] = _dn_list(label_dn)


def _yaw_about_normal(action) -> np.ndarray:
    from mold_aware_runtime import rot6d_rows_to_R

    rotation = rot6d_rows_to_R(np.asarray(action, dtype=np.float64)[..., 21:27])
    return np.degrees(np.arctan2(rotation[..., 1, 0], rotation[..., 0, 0]))


def _attach_region_yaw(run: dict, blob, exp: dict, episode: int) -> None:
    """Region δn/ΔR and the yaw-gap fraction, on the stride-4 query set."""
    if exp["settings"].get("role") != "r4vt" or "pred_action" not in blob.files:
        return
    dataset = exp["settings"].get("dataset")
    if not dataset:
        return
    pred = np.asarray(blob["pred_action"], dtype=np.float64)
    if pred.shape[-1] < 27:
        return
    keep = table_keep(blob["query_time_s"], exp["settings"])
    normal = mold_normal(dataset, episode)
    if normal is None:
        return
    anchor = np.minimum(query_anchor(blob)[keep], len(normal) - 1)
    region = ol.region_code(normal[anchor])
    label = np.asarray(blob["label_action"], dtype=np.float64)[keep]
    kept_pred = pred[keep]
    n_valid = np.asarray(blob["n_valid"])[keep]
    horizon = int(exp["settings"].get("execution_horizon") or 6)
    run["region_error"] = ol.region_action_errors(kept_pred, label, n_valid, region, horizon)
    run["table_queries"] = int(len(keep))
    yaw = _yaw_about_normal(kept_pred[:, 0])
    gap = (np.abs(yaw) >= 45.0) & (np.abs(yaw) <= 135.0)
    by_region = {}
    for code, name in enumerate(ol.REGION_NAMES):
        mask = region == code
        by_region[name] = {
            "n": int(mask.sum()),
            "gap": int(np.sum(gap & mask)),
        }
    run["yaw_gap"] = {
        "n": int(len(yaw)),
        "gap": int(gap.sum()),
        "frac": float(gap.mean()) if len(yaw) else None,
        "by_region": by_region,
    }


def query_region_names(blob, episode: int, dataset: str | None) -> list[str] | None:
    """Crown / fillet / wall from mold_normal_0 at the query's label foot."""
    if not dataset or "query_rgb_idx" not in getattr(blob, "files", ()):
        return None
    normal = mold_normal(dataset, episode)
    if normal is None or len(normal) == 0:
        return None
    idx = np.minimum(query_anchor(blob), len(normal) - 1)
    codes = ol.region_code(normal[idx])
    return [ol.REGION_NAMES[int(code)] for code in codes]


def episode_payload(exports: list, episode: int, mesh_t0_ok: bool = True, frames: dict | None = None) -> dict:
    first = None
    for exp in exports:
        if episode in exp["episodes"]:
            first = exp["episodes"][episode]
            break
    if first is None:
        raise KeyError(episode)
    robot_t = np.asarray(first["robot_time_s"], dtype=np.float64)
    rgb_t = np.asarray(first["rgb_time_s"], dtype=np.float64)
    t0 = float(rgb_t[0])
    ip = stride_idx(len(robot_t), PATH_CAP)
    xyz = np.asarray(first["tcp_xyz"], dtype=np.float64)
    R = np.asarray(first["tcp_R"], dtype=np.float64)
    force_t = np.asarray(first["force_series_t"], dtype=np.float64)
    force = np.asarray(first["force_series"], dtype=np.float64)
    iw = stride_idx(len(force_t), SERIES_CAP)
    bad = np.asarray(first["bad_robot"], dtype=bool)
    shade = shade_spans({str(episode): {"zarr_s": mask_spans(robot_t, bad)}}, episode, t0)
    robot_gap = edge_gaps(np.arange(len(robot_t)), bad=bad, times=robot_t, dt_cap=_time_cap(robot_t, 0.05))
    path_break = breaks_for_keep(robot_gap, ip)
    force_gap = edge_gaps(np.arange(len(force_t)), times=force_t, dt_cap=_time_cap(force_t, 0.2))
    force_break = breaks_for_keep(force_gap, iw)
    runs = []
    for exp in exports:
        blob = exp["episodes"].get(episode)
        if blob is None:
            continue
        q_t = np.asarray(blob["query_time_s"], dtype=np.float64) - t0
        pred = np.asarray(blob["pred_action"], dtype=np.float64)
        label = np.asarray(blob["label_action"], dtype=np.float64)
        iser = stride_idx(len(q_t), SERIES_CAP)
        q_gap = edge_gaps(
            np.asarray(blob["query_rgb_idx"]),
            bad=np.asarray(blob["bad_span"])[:, 0],
            long=np.asarray(blob["long_dt"])[:, 0],
        )
        q_break = breaks_for_keep(q_gap, iser)
        gaps = spans_from_edges(np.asarray(blob["query_time_s"], dtype=np.float64), q_gap, t0)
        step_pred = pred[:, 0]
        step_label = label[:, 0]
        if pred.shape[-1] >= 27:
            ang = ol.delta_r_deg(step_pred, step_label)
        else:
            ang = ol.relative_command_rot_deg(step_pred, step_label)
        if "exec_pos" in blob.files:
            exec_pos = np.asarray(blob["exec_pos"], dtype=np.float64)
            exec_edge = np.asarray(blob["exec_break"], dtype=bool)
            ipath = stride_idx(len(exec_pos), PATH_CAP)
            path = rounded(exec_pos[ipath], 4)
            path_break = breaks_for_keep(exec_edge, ipath)
        else:
            pos = np.asarray(blob["pred_pos"], dtype=np.float64)[:, 0]
            path = rounded(pos[iser], 4)
            path_break = q_break
        runs.append(
            {
                "label": exp["label"],
                "role": exp["settings"].get("role", "r4vt"),
                "split": str(blob["split"]),
                "q_i": [int(i) for i in iser],
                "q_t": rounded(q_t[iser], 3),
                "q_break": q_break,
                "gaps": gaps,
                "path": path,
                "path_break": path_break,
                "v1_p": rounded(step_pred[iser, 0], 4),
                "v1_l": rounded(step_label[iser, 0], 4),
                "v2_p": rounded(step_pred[iser, 1], 4),
                "v2_l": rounded(step_label[iser, 1], 4),
                "n_p": rounded(step_pred[iser, 2], 4),
                "n_l": rounded(step_label[iser, 2], 4),
                "stiff_p": rounded(step_pred[iser, 18], 4),
                "stiff_l": rounded(step_label[iser, 18], 4),
                "ang": rounded(ang[iser], 3),
                "error": ol.episode_error_table(
                    *flatten_for_table(blob)
                ),
                "exec_horizon": int(exp["settings"].get("execution_horizon") or 6),
                "horizon_error": _horizon_error(blob, exp),
                "ckpt": exp["settings"].get("ckpt", ""),
                "epoch": exp["settings"].get("checkpoint_epoch"),
                "query_stride": int(exp["settings"].get("query_stride") or 1),
            }
        )
        _attach_feet(runs[-1], blob, exp, episode)
        _attach_region_yaw(runs[-1], blob, exp, episode)
        names = query_region_names(blob, episode, exp["settings"].get("dataset"))
        if names is not None:
            anchor = np.minimum(query_anchor(blob), len(np.asarray(blob["tcp_xyz"])) - 1)
            runs[-1]["region"] = names
            runs[-1]["q_t_all"] = rounded(q_t, 3)
            runs[-1]["q_xyz"] = rounded(np.asarray(blob["tcp_xyz"], dtype=np.float64)[anchor], 4)
        pos, _rot, n_valid = prediction_source(blob)
        if pos is not None and "query_rgb_idx" in blob.files:
            horizon = runs[-1]["exec_horizon"]
            rgb = np.asarray(blob["query_rgb_idx"])
            runs[-1]["pred_exec"], runs[-1]["pred_exec_break"] = _chunk_lines(
                pos, n_valid, rgb, horizon, False
            )
            runs[-1]["pred_full"], runs[-1]["pred_full_break"] = _chunk_lines(
                pos, n_valid, rgb, horizon, True
            )
    # Full per-query chunks stay out of this summary. The cursor endpoint loads one.
    payload = {
        "episode": episode,
        "split": runs[0]["split"] if runs else "",
        "path_t": rounded(robot_t[ip] - t0, 3),
        "path": rounded(xyz[ip], 4),
        "path_break": path_break,
        "tool_R": rounded(R[ip], 4),
        "force_t": rounded(force_t[iw] - t0, 3),
        "force": rounded(force[iw], 2),
        "force_break": force_break,
        "shade": shade,
        "q_index": [int(i) for i in range(len(np.asarray(first["query_time_s"])))],
        "runs": runs,
    }
    if frames and episode in frames:
        overlay = frame_overlay(frames[episode], robot_t, t0)
        if overlay is not None:
            payload["frames"] = overlay
        switches = _switches_for(frames, episode, t0)
        if switches:
            payload["switches"] = switches
        if "t0" in frames[episode] or switches:
            payload["t0"] = t0
    return payload


def frames_only_payload(frames: dict, episode: int) -> dict:
    """Foot path and frame step when this episode has no open-loop export."""
    pack = frames.get(episode)
    if not pack or "robot_time_s" not in pack:
        raise KeyError(episode)
    robot_t = np.asarray(pack["robot_time_s"], dtype=np.float64)
    t0 = float(pack["t0"])
    ip = stride_idx(len(robot_t), PATH_CAP)
    gap = edge_gaps(np.arange(len(robot_t)), times=robot_t, dt_cap=_time_cap(robot_t, 0.05))
    overlay = frame_overlay(pack, robot_t, t0)
    payload = {
        "episode": episode,
        "split": pack.get("split", ""),
        "policy": False,
        "path_t": rounded(robot_t[ip] - t0, 3),
        "path": rounded(np.asarray(pack["tcp_xyz"])[ip], 4),
        "path_break": breaks_for_keep(gap, ip),
        "tool_R": rounded(np.asarray(pack["tcp_R"])[ip], 4),
        "force_t": [],
        "force": [],
        "force_break": [],
        "shade": [],
        "q_index": [],
        "runs": [],
        "path_i": [int(i) for i in ip],
        "t0": t0,
        "frames": overlay,
    }
    switches = _switches_for(frames, episode, t0)
    if switches:
        payload["switches"] = switches
    return payload


def _horizon_error(blob, exp) -> dict:
    horizon = int(exp["settings"].get("execution_horizon") or 6)
    pred_pos = np.asarray(blob["raw_pos"]) if "raw_pos" in blob.files else None
    label_pos = None
    if pred_pos is not None:
        label_pos = ol.indexed_positions(blob["tcp_xyz"], blob["action_robot_idx"])
    return ol.chunk_horizon_errors(
        blob["pred_action"],
        blob["label_action"],
        pred_pos,
        label_pos,
        blob["n_valid"],
        horizon,
    )


def flatten_for_table(blob):
    pred = np.asarray(blob["pred_action"])
    label = np.asarray(blob["label_action"])
    H = pred.shape[1]
    valid = (np.arange(H)[None, :] < np.asarray(blob["n_valid"])[:, None]).reshape(-1)
    return (
        pred.reshape(-1, pred.shape[-1]),
        label.reshape(-1, label.shape[-1]),
        np.asarray(blob["force_n"]).reshape(-1),
        np.asarray(blob["bad_span"]).reshape(-1),
        np.asarray(blob["long_dt"]).reshape(-1),
        valid,
    )


def cursor_payload(exports: list, episode: int, query: int, frames: dict | None = None, span: str = "exec") -> dict:
    out = []
    frame = None
    for exp in exports:
        blob = exp["episodes"].get(episode)
        if blob is None or query < 0 or query >= len(blob["query_time_s"]):
            continue
        t0 = float(np.asarray(blob["rgb_time_s"])[0])
        if frame is None and frames and episode in frames and "robot_time_s" in blob.files:
            robot_t = np.asarray(blob["robot_time_s"], dtype=np.float64)
            if len(frames[episode]["R_old"]) == len(robot_t):
                t_abs = float(np.asarray(blob["query_time_s"])[query])
                frame = frame_at(frames[episode], robot_t, t_abs)
        horizon = int(exp["settings"].get("execution_horizon") or 6)
        pos, rot, n_valid = prediction_source(blob)
        if pos is None:
            continue
        n = int(n_valid[query])
        if span != "full":
            n = min(n, horizon)
        n = int(np.clip(n, 0, pos.shape[1]))
        if n == 0:
            continue
        future = np.asarray(pos[query, :n])
        future_R = np.asarray(rot[query, :n]) if rot is not None else None
        label_xyz = None
        label_R = None
        if "action_robot_idx" in blob.files:
            idx = np.asarray(blob["action_robot_idx"][query, :n], dtype=np.int64)
            known = idx >= 0
            if "tcp_xyz" in blob.files and np.any(known):
                label_xyz = np.asarray(blob["tcp_xyz"], dtype=np.float64)[idx[known]]
            if "tcp_R" in blob.files and np.any(known):
                label_R = np.asarray(blob["tcp_R"], dtype=np.float64)[idx[known]]
        item = {
            "label": exp["label"],
            "future": rounded(future, 4),
            "press": rounded(future[0], 4),
            "R": rounded(future_R[0] if future_R is not None else np.eye(3), 4),
            "t": round(float(blob["query_time_s"][query]) - t0, 3),
        }
        if future_R is not None:
            item["future_R"] = rounded(future_R, 4)
        if label_xyz is not None:
            item["label_xyz"] = rounded(label_xyz, 4)
        if label_R is not None:
            item["label_R"] = rounded(label_R, 4)
        _attach_chunk_feet(item, blob, exp, episode, query, n)
        out.append(item)
    if frame is None and frames and episode in frames and not out:
        pack = frames[episode]
        if "robot_time_s" in pack:
            robot_t = np.asarray(pack["robot_time_s"], dtype=np.float64)
            i = int(np.clip(query, 0, len(robot_t) - 1))
            frame = frame_at(pack, robot_t, float(robot_t[i]))
    payload = {"runs": out}
    if frame is not None:
        payload["frame"] = frame
    return payload


PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>open loop</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body { margin:0; font-family:sans-serif; background:#111; color:#ddd; }
.bar { display:flex; gap:12px; align-items:center; padding:8px 12px; flex-wrap:wrap; }
select, button { background:#222; color:#ddd; border:1px solid #444; padding:4px 8px; }
button.on { background:#1a5276; }
input[type=range] { width:220px; }
#view3d { height:42vh; }
#plots { height:52vh; }
#events { display:none; margin:8px 12px 24px; font-size:13px; max-height:46vh; overflow:auto; }
#events table { margin:4px 0 12px; }
#events tr[data-kind] { cursor:pointer; }
#events tr[data-kind]:hover { background:#222; }
#events h3 { margin:12px 0 4px; font-size:14px; }
table { border-collapse:collapse; margin:8px 12px 16px; font-size:13px; }
td, th { border:1px solid #333; padding:3px 8px; }
</style></head>
<body>
<div class="bar">
  <span>episode</span><select id="ep"></select>
  <span id="split"></span>
  <span>checkpoint</span><select id="ckpt"></select>
  <span id="toggles"></span>
  <span id="frame_toggles"></span>
  <button id="spanExec" class="on">6 steps</button>
  <button id="spanFull">full chunk</button>
  <span id="spanNote" style="color:#e67e22"></span>
  <span>region</span>
  <button id="regAll" class="on">all</button>
  <button id="regCrown">crown</button>
  <button id="regFillet">fillet</button>
  <button id="regWall">wall</button>
  <button id="geomFoot" class="on">foot + δn</button>
  <button id="geomPoint">virtual target</button>
  <button id="onlySel" class="on">only selected</button>
  <span>time</span><input id="time" type="range" min="0" max="0" value="0">
  <span id="readout"></span>
</div>
<div class="bar" id="ckptpath" style="color:#9ad;font-size:12px;padding-top:0"></div>
<div id="view3d"></div>
<div id="plots"></div>
<div class="bar" id="headingBar" style="display:none">
  <span>heading step</span>
  <button id="hStep1" class="on">1</button>
  <button id="hStep6">6</button>
  <button id="hStep64">64</button>
  <label><input id="hPred" type="checkbox" checked> heading pred</label>
  <label><input id="hLabel" type="checkbox" checked> heading label</label>
  <label><input id="hCausal" type="checkbox"> causal e199</label>
  <span id="headingNote"></span>
</div>
<div id="headingPlots" style="display:none;height:32vh"></div>
<table id="summary"></table>
<div id="events"></div>
<script>
const VIEWS = {
  side: {eye: {x: 0.15, y: -1.7, z: 1.7}, center: {x: 0, y: 0, z: 0}, up: {x: 0, y: -1, z: 0}},
  top: {eye: {x: 0, y: -2.6, z: 0.02}, center: {x: 0, y: 0, z: 0}, up: {x: 0, y: 0, z: -1}},
  end: {eye: {x: 2.6, y: -0.45, z: 0.35}, center: {x: 0, y: 0, z: 0}, up: {x: 0, y: -1, z: 0}}
};
let metaHeading = false;
let viewName = "side";
let sceneBox = null;
let mesh = null;
let data = null;
let fi = 0;
let shown = {};
let shownFrame = {old: true, new: true};
let span = "exec";
let regionFilter = "all";
let geomMode = "foot";
let selectedLabel = "";
let headingStep = "1";
let showHeadingPred = true;
let showHeadingLabel = true;
let showCausal = false;
let onlySelected = true;
const REGION_COLOR = {crown: "#2ecc71", fillet: "#f4d03f", wall: "#e74c3c"};

function nearest(times, t) {
  if (!times || !times.length) return 0;
  let lo = 0, hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < t) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(times[lo - 1] - t) < Math.abs(times[lo] - t)) lo--;
  return lo;
}
function valueSpan(values) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return [lo, hi];
}
function sceneBoxFrom(extra) {
  const xs = mesh.x.slice(), ys = mesh.y.slice(), zs = mesh.z.slice();
  function add(pts) { (pts || []).forEach(p => { xs.push(p[0]); ys.push(p[1]); zs.push(p[2]); }); }
  add(data.path);
  (extra || []).forEach(add);
  const pad = 0.05;
  function rng(a) { const s = valueSpan(a); return [s[0] - pad, s[1] + pad]; }
  return {x: rng(xs), y: rng(ys), z: rng(zs)};
}
function axisLock(title, range) { return {title, range, autorange: false}; }
function sceneLayout() {
  const box = sceneBox || {x: [-1, 1], y: [-1, 1], z: [-1, 1]};
  return {
    paper_bgcolor: "#111", margin: {l: 0, r: 0, t: 0, b: 0}, showlegend: true,
    legend: {x: 0, y: 1, font: {color: "#ddd"}, bgcolor: "rgba(0,0,0,0)"},
    uirevision: "keep",
    scene: {
      uirevision: "keep", aspectmode: "data", bgcolor: "#1c1c1c",
      camera: VIEWS[viewName],
      xaxis: axisLock("x", box.x), yaxis: axisLock("y", box.y), zaxis: axisLock("z", box.z)
    }
  };
}
function glScene() {
  const gd = document.getElementById("view3d");
  return gd && gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene._scene;
}
function liveCamera() {
  const scene = glScene();
  if (scene && scene.getCamera) return scene.getCamera();
  return VIEWS[viewName];
}
function applyCamera(cam) {
  const scene = glScene();
  if (!scene || !cam || !scene.glplot) return;
  const full = {
    eye: cam.eye, center: cam.center, up: cam.up,
    projection: (cam.projection && cam.projection.type) ? cam.projection : {type: "perspective"}
  };
  scene.camera.lookAt.apply(scene, [
    [full.eye.x, full.eye.y, full.eye.z],
    [full.center.x, full.center.y, full.center.z],
    [full.up.x, full.up.y, full.up.z]
  ]);
  scene.glplot.redraw();
  const gd = document.getElementById("view3d");
  if (gd.layout && gd.layout.scene) gd.layout.scene.camera = full;
}
const DIM = {
  v1: {pred: "#4f7399", label: "#d5e6f5"},
  v2: {pred: "#3f7a58", label: "#d7f3e3"},
  n: {pred: "#7a5c8f", label: "#eadcf6"},
  ang: {pred: "#a8883d", label: "#f8efd0"},
  stiff: {pred: "#3d6f8c", label: "#d4ebf6"}
};
function withBreaks(xs, ys, cut) {
  const xo = [], yo = [];
  for (let i = 0; i < xs.length; i++) {
    xo.push(xs[i]);
    yo.push(ys[i]);
    if (cut && cut[i]) { xo.push(null); yo.push(null); }
  }
  return [xo, yo];
}
function line(x, y, name, color, axis, dash, cut) {
  const broken = withBreaks(x, y, cut);
  return {
    x: broken[0], y: broken[1], name,
    xaxis: axis, yaxis: axis.replace("x", "y"),
    line: {color, width: dash ? 1.3 : 2.0, dash: dash || "solid"}
  };
}
function shadeRect(n, t0, t1) {
  return {
    type: "rect", xref: "x" + n, yref: "y" + n + " domain",
    x0: t0, x1: t1, y0: 0, y1: 1,
    fillcolor: "rgba(245,176,65,0.22)", line: {width: 0}, layer: "below"
  };
}
function gapRect(n, t0, t1) {
  return {
    type: "rect", xref: "x" + n, yref: "y" + n + " domain",
    x0: t0, x1: t1, y0: 0, y1: 1,
    fillcolor: "rgba(255,255,255,0.10)", line: {width: 0}, layer: "below"
  };
}
function axesAt(origin, R, color, name, width) {
  const scale = 0.04;
  const x = [origin[0]], y = [origin[1]], z = [origin[2]];
  for (let a = 0; a < 3; a++) {
    x.push(origin[0] + scale * R[0][a], null);
    y.push(origin[1] + scale * R[1][a], null);
    z.push(origin[2] + scale * R[2][a], null);
  }
  return {type: "scatter3d", x, y, z, mode: "lines", name, line: {color, width: width || 4}, showlegend: false};
}
function runStyle(i, n) {
  if (n <= 1) return {path: "#ff9f1a", chunk: "#ff4fd8", press: "#ffe14a"};
  const hue = [25, 140, 200, 280, 0, 170, 320, 50][i % 8];
  return {path: "hsl(" + hue + ",85%,52%)", chunk: "hsl(" + hue + ",95%,64%)", press: "#ffe14a"};
}
function dnSegments(feet, normals, dn, name, color, width) {
  const x = [], y = [], z = [];
  const n = Math.min(feet ? feet.length : 0, normals ? normals.length : 0, dn ? dn.length : 0);
  for (let i = 0; i < n; i++) {
    const f = feet[i], v = normals[i], d = dn[i];
    if (!f || !v || d === null || d === undefined) continue;
    x.push(f[0], f[0] + d * v[0], null);
    y.push(f[1], f[1] + d * v[1], null);
    z.push(f[2], f[2] + d * v[2], null);
  }
  return {
    type: "scatter3d", x, y, z, mode: "lines", name,
    showlegend: true, legendgroup: name,
    line: {color, width: width || 3}
  };
}
function scatterNullable(pts, name, color, width, cut) {
  const x = [], y = [], z = [];
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    if (!p) { x.push(null); y.push(null); z.push(null); continue; }
    x.push(p[0]); y.push(p[1]); z.push(p[2]);
    if (cut && cut[i]) { x.push(null); y.push(null); z.push(null); }
  }
  return {
    type: "scatter3d", x, y, z, mode: "lines", name,
    showlegend: true, legendgroup: name,
    line: {color, width: width || 4}
  };
}
function scatter(pts, name, color, width, cut, times) {
  const x = [], y = [], z = [], cd = [];
  for (let i = 0; i < pts.length; i++) {
    x.push(pts[i][0]); y.push(pts[i][1]); z.push(pts[i][2]);
    if (times) cd.push(times[i]);
    if (cut && cut[i]) {
      x.push(null); y.push(null); z.push(null);
      if (times) cd.push(null);
    }
  }
  const out = {type: "scatter3d", x, y, z, mode: "lines", name, line: {color, width: width || 4}};
  if (times) out.customdata = cd;
  return out;
}
function axisSegs(origins, Rs, col, scale) {
  const x = [], y = [], z = [];
  for (let i = 0; i < origins.length; i++) {
    const p = origins[i], R = Rs[i];
    if (!p || !R) continue;
    x.push(p[0], p[0] + scale * R[0][col], null);
    y.push(p[1], p[1] + scale * R[1][col], null);
    z.push(p[2], p[2] + scale * R[2][col], null);
  }
  return {x, y, z};
}
function axisTrace(seg, name, color, legend) {
  return {
    type: "scatter3d", x: seg.x, y: seg.y, z: seg.z, mode: "lines",
    name, showlegend: !!legend, legendgroup: name,
    line: {color, width: 4}
  };
}
function pathTriads(d) {
  const origins = [], Rs = [];
  for (let i = 0; i < d.path.length; i += 20) {
    origins.push(d.path[i]);
    Rs.push(d.tool_R[i]);
  }
  return [
    axisTrace(axisSegs(origins, Rs, 0, 0.014), "label axes", "#4aa3ff", true),
    axisTrace(axisSegs(origins, Rs, 2, 0.014), "label axes", "#3ddc97", false)
  ];
}
function chunkTriads(item) {
  const every = item.future.length > 12 ? 4 : 1;
  const out = [];
  if (item.future_R) {
    const o = [], r = [];
    for (let k = 0; k < item.future.length; k += every) {
      o.push(item.future[k]);
      r.push(item.future_R[k]);
    }
    out.push(axisTrace(axisSegs(o, r, 0, 0.022), "predicted axes", "#ff4d6a", true));
    out.push(axisTrace(axisSegs(o, r, 2, 0.022), "predicted axes", "#ffd23f", false));
  }
  if (item.label_xyz && item.label_R) {
    const o = [], r = [];
    for (let k = 0; k < item.label_xyz.length; k += every) {
      o.push(item.label_xyz[k]);
      r.push(item.label_R[k]);
    }
    out.push(axisTrace(axisSegs(o, r, 0, 0.022), "label axes", "#4aa3ff", false));
    out.push(axisTrace(axisSegs(o, r, 2, 0.022), "label axes", "#3ddc97", false));
  }
  return out;
}
const FADE = [1, 0.72, 0.42, 0.18];
function fadedPrediction(pts, cut) {
  const groups = FADE.map(() => ({x: [], y: [], z: []}));
  let i = 0;
  while (i < pts.length) {
    let j = i;
    while (j < pts.length && !(cut && cut[j])) j++;
    const end = Math.min(pts.length, j + 1);
    const n = end - i;
    for (let k = 0; k < n; k++) {
      const b = Math.min(FADE.length - 1, Math.floor(k * FADE.length / n));
      groups[b].x.push(pts[i + k][0]);
      groups[b].y.push(pts[i + k][1]);
      groups[b].z.push(pts[i + k][2]);
    }
    groups.forEach(g => { g.x.push(null); g.y.push(null); g.z.push(null); });
    i = end;
  }
  return groups.map((g, b) => ({
    type: "scatter3d", x: g.x, y: g.y, z: g.z, mode: "lines",
    name: "prediction", showlegend: b === 0, legendgroup: "prediction",
    line: {color: "rgba(255,77,106," + FADE[b] + ")", width: 5}
  }));
}
function filteredQueries(run) {
  const region = (run && run.region) || [];
  const times = (run && run.q_t_all) || [];
  const xyz = (run && run.q_xyz) || [];
  const out = [];
  for (let i = 0; i < region.length; i++) {
    if (regionFilter === "all" || region[i] === regionFilter) {
      out.push({i: i, t: times[i], p: xyz[i], region: region[i]});
    }
  }
  return out;
}
function regionMarkers(run) {
  if (!run || !run.q_xyz || !run.region) return [];
  const groups = {crown: [], fillet: [], wall: []};
  run.region.forEach((name, i) => {
    if (regionFilter !== "all" && name !== regionFilter) return;
    if (groups[name] && run.q_xyz[i]) groups[name].push(run.q_xyz[i]);
  });
  return Object.keys(groups).filter(name => groups[name].length).map(name => ({
    type: "scatter3d",
    x: groups[name].map(p => p[0]),
    y: groups[name].map(p => p[1]),
    z: groups[name].map(p => p[2]),
    mode: "markers",
    name: name,
    marker: {size: regionFilter === name ? 6 : 3, color: REGION_COLOR[name]}
  }));
}
function selectedRun() {
  if (!data || !data.runs || !data.runs.length) return null;
  const name = selectedLabel || document.getElementById("ckpt").value;
  return data.runs.find(r => r.label === name) || data.runs[0];
}
function visibleRuns() {
  if (!data || !data.runs) return [];
  const picked = selectedRun();
  if (onlySelected) {
    if (!picked || shown[picked.label] === false) return [];
    return [picked];
  }
  return data.runs.filter(r => shown[r.label] !== false);
}
function showCheckpoint(run) {
  const el = document.getElementById("ckptpath");
  const note = document.getElementById("spanNote");
  if (!run) {
    if (el) el.textContent = "";
    if (note) note.textContent = "";
    return;
  }
  const epoch = run.epoch === null || run.epoch === undefined ? "?" : run.epoch;
  const kind = run.role === "r1rel" ? "    tool pose" : "";
  const stride = run.query_stride > 1 ? "    every " + run.query_stride + "th query" : "";
  el.textContent = (run.ckpt || "") + "    epoch " + epoch + kind + stride;
  const full = run.pred_full || [];
  const exec = run.pred_exec || [];
  note.textContent = full.length && full.length === exec.length ? "stored chunk is 6 steps" : "";
}
function paintTable(run) {
  const dims = Object.keys(run.error.mae);
  const cols = ["contact", "free", "clean", "bad_span"];
  let html = "<tr><th>dim</th>" + cols.map(c => "<th>" + c + "</th>").join("") + "</tr>";
  dims.forEach(dim => {
    html += "<tr><td>" + dim + "</td>";
    cols.forEach(col => {
      const v = run.error.mae[dim][col];
      html += "<td>" + (v === null || v === undefined ? "—" : Number(v).toExponential(3)) + "</td>";
    });
    html += "</tr>";
  });
  const regions = run.region_error;
  if (regions) {
    html += "<tr><th colspan='5'>stride-4 queries " + (run.table_queries || "") + "</th></tr>";
    html += "<tr><th>steps</th><th>region</th><th>n</th><th>δn mm</th><th>ΔR deg</th></tr>";
    ["exec", "full"].forEach(which => {
      ["crown", "fillet", "wall"].forEach(name => {
        const cell = regions[which][name];
        const nameSpan = which === "exec" ? "first 6" : "full chunk";
        const dn = cell.delta_n_mm === null || cell.delta_n_mm === undefined ? "—" : Number(cell.delta_n_mm).toFixed(1);
        const dR = cell.delta_R_deg === null || cell.delta_R_deg === undefined ? "—" : Number(cell.delta_R_deg).toFixed(1);
        html += "<tr><td>" + nameSpan + "</td><td>" + name + "</td><td>" + cell.n + "</td><td>" + dn + "</td><td>" + dR + "</td></tr>";
      });
    });
  }
  if (run.yaw_gap) {
    const g = run.yaw_gap;
    const pct = g.frac === null ? "—" : (100 * g.frac).toFixed(1) + "%";
    html += "<tr><td>yaw gap</td><td colspan='4'>" + g.gap + "/" + g.n + " (" + pct + ") |yaw| 45–135°</td></tr>";
  }
  const h = run.horizon_error;
  if (h) {
    const modes = ["all", "near_identity", "return_yaw"];
    html += "<tr><th>steps</th><th>dim</th>" + modes.map(m => "<th>" + m + "</th>").join("") + "</tr>";
    ["exec", "full"].forEach(which => {
      ["position_mm", "delta_n", "delta_R_deg"].forEach(dim => {
        const row = h[which][dim];
        if (!row) return;
        const name = which === "exec" ? ("first " + h.horizon) : "full chunk";
        html += "<tr><td>" + name + "</td><td>" + dim + "</td>";
        modes.forEach(m => {
          const v = row[m];
          html += "<td>" + (v === null || v === undefined ? "—" : Number(v).toExponential(3)) + "</td>";
        });
        html += "</tr>";
      });
    });
  }
  document.getElementById("summary").innerHTML = html;
}
async function paintFrameSeries(t) {
  const shapes = [{
    type: "line", xref: "x", yref: "y domain", x0: t, x1: t, y0: 0, y1: 1,
    line: {color: "#fff", width: 1}
  }];
  (data.switches || []).forEach(s => shapes.push({
    type: "line", xref: "x", yref: "y domain", x0: s.t, x1: s.t, y0: 0, y1: 1,
    line: {color: "#5dade2", width: 1, dash: "dot"}
  }));
  const plots = [
    line(data.frames.t, data.frames.dR_old, "old |ΔR|", "rgba(255,255,255,0.55)", "x", "dash", data.frames.break),
    line(data.frames.t, data.frames.dR_new, "new |ΔR|", "#ff9f1a", "x", null, data.frames.break),
    {
      x: data.frames.jump_t, y: data.frames.jump_y, name: "old jump",
      type: "scatter", mode: "markers",
      marker: {color: "#e74c3c", size: 7, symbol: "diamond"}
    }
  ];
  if (data.switches && data.switches.length) {
    plots.push({
      x: data.switches.map(s => s.t), y: data.switches.map(s => s.frame),
      name: "crown-wall", type: "scatter", mode: "markers",
      marker: {color: "#5dade2", size: 9, symbol: "triangle-up"}
    });
  }
  await Plotly.react("plots", plots, {
    paper_bgcolor: "#111", plot_bgcolor: "#1c1c1c", font: {color: "#ddd"},
    margin: {l: 64, r: 16, t: 28, b: 28}, legend: {orientation: "h", y: 1.12},
    shapes,
    xaxis: {title: "s"},
    yaxis: {title: "frame deg"}
  }, {responsive: true});
}
function buildEvents(meta) {
  const root = document.getElementById("events");
  const sw = meta.switches || [];
  const cap = meta.nearcap || [];
  let html = "<h3>Crown-to-wall (" + sw.length + ")</h3>";
  html += "<table><tr><th>ep</th><th>sample</th><th>time</th><th>frame step</th><th>tool</th></tr>";
  sw.forEach((row, n) => {
    html += "<tr data-kind='switch' data-n='" + n + "'><td>" + row.ep + "</td><td>" + row.i
      + "</td><td>" + Number(row.t).toFixed(2) + " s</td><td>" + Number(row.frame).toFixed(2)
      + "°</td><td>" + Number(row.tool).toFixed(2) + "°</td></tr>";
  });
  html += "</table><h3>Near-cap steps (" + cap.length + ")</h3>";
  const groups = {};
  cap.forEach((row, n) => { (groups[row.ep] = groups[row.ep] || []).push(n); });
  Object.keys(groups).map(Number).sort((a, b) => a - b).forEach(ep => {
    html += "<details><summary>ep " + ep + " · " + groups[ep].length + "</summary><table>";
    groups[ep].forEach(n => {
      const row = cap[n];
      html += "<tr data-kind='near' data-n='" + n + "'><td>" + row.i + "</td><td>"
        + Number(row.t).toFixed(2) + " s</td><td>" + Number(row.frame).toFixed(2)
        + "°</td><td>tool " + Number(row.tool).toFixed(2) + "°</td></tr>";
    });
    html += "</table></details>";
  });
  root.innerHTML = html;
  root.style.display = "block";
  root.onclick = async (ev) => {
    const tr = ev.target.closest("tr");
    if (!tr || !tr.dataset.kind) return;
    const row = (tr.dataset.kind === "switch" ? sw : cap)[+tr.dataset.n];
    await jumpTo(row.ep, row.t);
  };
}
async function jumpTo(ep, tRel) {
  const sel = document.getElementById("ep");
  if (![...sel.options].some(opt => opt.value === String(ep))) return;
  if (String(sel.value) !== String(ep)) {
    sel.value = String(ep);
    await show();
  }
  fi = nearest(data.path_t, tRel);
  document.getElementById("time").value = fi;
  await paintCursor();
}
function headingRow(t) {
  const pack = data && data.heading;
  if (!pack || !pack.t || !pack.t.length) return null;
  let best = 0;
  let dist = Infinity;
  pack.t.forEach((time, i) => {
    const d = Math.abs(time - t);
    if (d < dist) { dist = d; best = i; }
  });
  return {i: best, step: pack.steps[headingStep]};
}
function headingAxes(t) {
  const hit = headingRow(t);
  if (!hit || !hit.step) return [];
  const s = hit.step;
  const i = hit.i;
  const traces = [];
  if (showHeadingPred && s.h_origin_p && s.h_origin_p[i] && s.h_R_p[i]) {
    traces.push(axesAt(s.h_origin_p[i], s.h_R_p[i], "#ff4d6a", "heading pred", 6));
  }
  if (showHeadingLabel && s.h_origin_l && s.h_origin_l[i] && s.h_R_l[i]) {
    traces.push(axesAt(s.h_origin_l[i], s.h_R_l[i], "#4aa3ff", "heading label", 4));
  }
  if (showCausal && s.c_origin_p && s.c_origin_p[i] && s.c_R_p[i]) {
    traces.push(axesAt(s.c_origin_p[i], s.c_R_p[i], "#ff9f1a", "causal pred", 6));
  }
  if (showCausal && s.c_origin_l && s.c_origin_l[i] && s.c_R_l[i]) {
    traces.push(axesAt(s.c_origin_l[i], s.c_R_l[i], "#3ddc97", "causal label", 4));
  }
  return traces;
}
function headingLine(x, y, name, color, dash) {
  return {
    x, y, name, type: "scatter", mode: "lines",
    line: {color, width: 2, dash: dash || "solid"}
  };
}
function paintHeading(t) {
  const box = document.getElementById("headingPlots");
  const pack = data && data.heading;
  if (!pack || !pack.t || !pack.t.length || !pack.steps || !pack.steps[headingStep]) {
    box.style.display = "none";
    return;
  }
  box.style.display = "block";
  const s = pack.steps[headingStep];
  const plots = [
    headingLine(pack.t, s.psi_h_l, "ψ label", "#4aa3ff"),
    headingLine(pack.t, s.psi_h_p, "ψ pred", "#ff4d6a", "dash")
  ];
  const tilt = [
    headingLine(pack.t, s.tilt_h_l, "tilt label", "#4aa3ff"),
    headingLine(pack.t, s.tilt_h_p, "tilt pred", "#ff4d6a", "dash")
  ];
  const geo = [headingLine(pack.t, s.geo_h, "heading geodesic", "#ff4d6a")];
  if (showCausal) {
    plots.push(headingLine(pack.t, s.psi_c_l, "causal ψ label", "#3ddc97"));
    plots.push(headingLine(pack.t, s.psi_c_p, "causal ψ pred", "#ff9f1a", "dash"));
    tilt.push(headingLine(pack.t, s.tilt_c_l, "causal tilt label", "#3ddc97"));
    tilt.push(headingLine(pack.t, s.tilt_c_p, "causal tilt pred", "#ff9f1a", "dash"));
    geo.push(headingLine(pack.t, s.geo_c, "causal geodesic", "#ff9f1a", "dash"));
  }
  tilt.forEach(tr => { tr.xaxis = "x2"; tr.yaxis = "y2"; });
  geo.forEach(tr => { tr.xaxis = "x3"; tr.yaxis = "y3"; });
  const mark = {
    type: "line", xref: "x", yref: "y domain", x0: t, x1: t, y0: 0, y1: 1,
    line: {color: "#fff", width: 1}
  };
  Plotly.react("headingPlots", plots.concat(tilt, geo), {
    paper_bgcolor: "#111", plot_bgcolor: "#1c1c1c", font: {color: "#ddd"},
    margin: {l: 64, r: 16, t: 24, b: 28}, legend: {orientation: "h", y: 1.18},
    shapes: [mark, Object.assign({}, mark, {xref: "x2", yref: "y2 domain"}), Object.assign({}, mark, {xref: "x3", yref: "y3 domain"})],
    xaxis: {anchor: "y", domain: [0, 1]},
    yaxis: {title: "ψ deg", domain: [0.70, 1], anchor: "x"},
    xaxis2: {anchor: "y2", matches: "x", domain: [0, 1]},
    yaxis2: {title: "tilt deg", domain: [0.36, 0.66], anchor: "x2"},
    xaxis3: {anchor: "y3", matches: "x", title: "s", domain: [0, 1]},
    yaxis3: {title: "geodesic deg", domain: [0, 0.32], anchor: "x3"}
  }, {responsive: true});
}
async function paintCursor() {
  const picked0 = selectedRun();
  const slider = document.getElementById("time");
  let t, j;
  if (regionFilter !== "all" && picked0 && picked0.region) {
    const hits = filteredQueries(picked0);
    slider.max = Math.max(0, hits.length - 1);
    fi = Math.max(0, Math.min(fi, Math.max(0, hits.length - 1)));
    if (!hits.length) {
      document.getElementById("readout").textContent = "no " + regionFilter + " queries";
      return;
    }
    t = hits[fi].t;
    j = hits[fi].i;
  } else {
    slider.max = Math.max(0, data.path_t.length - 1);
    fi = Math.max(0, Math.min(fi, data.path_t.length - 1));
    t = data.path_t[fi];
    const qTimes = data.runs[0] ? data.runs[0].q_t : data.path_t;
    const jLocal = nearest(qTimes, t);
    j = jLocal;
    if (data.path_i) j = data.path_i[Math.min(fi, data.path_i.length - 1)];
    else if (data.runs[0] && data.runs[0].q_i) j = data.runs[0].q_i[jLocal];
  }
  const cam = liveCamera();
  const cursor = await (await fetch("/api/cursor?ep=" + data.episode + "&q=" + j + "&span=" + span)).json();
  const picked = selectedRun();
  showCheckpoint(picked);
  const item = (cursor.runs || []).find(r => picked && r.label === picked.label);
  const drawn = visibleRuns();
  const footMode = geomMode === "foot" && drawn.some(r => r.label_foot && r.pred_foot);
  const traces = [];
  if (footMode) {
    drawn.forEach(run => {
      if (!run.label_foot || !run.pred_foot) return;
      const selected = picked && run.label === picked.label;
      const cut = run.foot_break;
      traces.push(scatterNullable(run.label_foot, selected ? "label foot" : run.label + " label", selected ? "#4aa3ff" : "#8eb4d6", selected ? 4 : 2, cut));
      traces.push(scatterNullable(run.pred_foot, selected ? "pred foot" : run.label + " pred", selected ? "#ff4d6a" : "#e67e22", selected ? 4 : 2, cut));
      if (!selected) return;
      traces.push(dnSegments(run.label_foot, run.label_n, run.label_dn, "label δn", "#3ddc97", 3));
      traces.push(dnSegments(run.pred_foot, run.pred_n, run.pred_dn, "pred δn", "#ffd23f", 3));
    });
    if (item && item.label_foot && picked && drawn.some(r => r.label === picked.label)) {
      const lab = dnSegments(item.label_foot, item.label_n, item.label_dn, "label δn", "#d5ffe9", 6);
      lab.showlegend = false;
      traces.push(lab);
      const pred = dnSegments(item.pred_foot, item.pred_n, item.pred_dn, "pred δn", "#fff3bf", 6);
      pred.showlegend = false;
      traces.push(pred);
    }
  } else {
    const labelPath = scatter(data.path, "label", "#4aa3ff", 5, data.path_break, data.path_t);
    labelPath.showlegend = true;
    labelPath.legendgroup = "label";
    traces.push(labelPath);
  }
  const p = footMode && item && item.label_foot && item.label_foot[0]
    ? item.label_foot[0]
    : data.path[nearest(data.path_t, t)];
  traces.push({
    type: "scatter3d", x: [p[0]], y: [p[1]], z: [p[2]], mode: "markers",
    name: "time", showlegend: false, marker: {size: 7, color: "#ffffff"}
  });
  regionMarkers(picked0).forEach(tr => traces.push(tr));
  if (!footMode && data.tool_R) pathTriads(data).forEach(tr => traces.push(tr));
  if (data.frames && cursor.frame && shownFrame.old) {
    traces.push(axesAt(cursor.frame.x_old, cursor.frame.R_old, "rgba(255,255,255,0.28)", "old frame", 3));
  }
  if (data.frames && cursor.frame && shownFrame.new) {
    traces.push(axesAt(cursor.frame.x_new, cursor.frame.R_new, "#ff9f1a", "new frame", 6));
  }
  headingAxes(t).forEach(tr => traces.push(tr));
  if (data.switches && data.switches.length) {
    const sx = [], sy = [], sz = [];
    data.switches.forEach(s => {
      if (!s.x) return;
      sx.push(s.x[0]); sy.push(s.x[1]); sz.push(s.x[2]);
    });
    if (sx.length) traces.push({
      type: "scatter3d", x: sx, y: sy, z: sz, mode: "markers", name: "crown-wall",
      marker: {size: 5, color: "#5dade2", symbol: "diamond"}
    });
  }
  if (data.frames && shownFrame.old && data.frames.jumps && data.frames.jumps.length) {
    const jx = [], jy = [], jz = [];
    data.frames.jumps.forEach(pt => { jx.push(pt[0]); jy.push(pt[1]); jz.push(pt[2]); });
    traces.push({
      type: "scatter3d", x: jx, y: jy, z: jz, mode: "markers", name: "old jump",
      marker: {size: 3.5, color: "#e74c3c", symbol: "diamond"}
    });
  }
  if (!footMode && picked && shown[picked.label] !== false) {
    const pred = span === "full" && picked.pred_full ? picked.pred_full : (picked.pred_exec || picked.path);
    const predCut = span === "full" && picked.pred_full ? picked.pred_full_break : (picked.pred_exec ? picked.pred_exec_break : picked.path_break);
    fadedPrediction(pred, predCut).forEach(tr => traces.push(tr));
    if (item) {
      const chunk = scatter(item.future, "this chunk", "#ffe14a", 8);
      chunk.showlegend = true;
      traces.push(chunk);
      chunkTriads(item).forEach(tr => traces.push(tr));
    }
  }
  traces.unshift({
    type: "mesh3d", x: mesh.x, y: mesh.y, z: mesh.z, i: mesh.i, j: mesh.j, k: mesh.k,
    color: "#9aabb6", opacity: 0.45, name: "mold", hoverinfo: "skip", showlegend: false
  });
  sceneBox = sceneBoxFrom([data.path]);
  await Plotly.react("view3d", traces, sceneLayout(), {responsive: true});
  applyCamera(cam);
  if (data.policy === false) {
    await paintFrameSeries(t);
    document.getElementById("summary").innerHTML = "";
    document.getElementById("readout").textContent = t.toFixed(2) + " s";
    document.getElementById("time").value = fi;
    paintHeading(t);
    bindPicks();
    return;
  }
  const run = data.runs[Math.max(0, data.runs.findIndex(r => r.label === document.getElementById("ckpt").value))];
  const hasFrames = !!(data.frames && data.frames.t);
  const shapes = [];
  (hasFrames ? ["", "2", "3", "4", "5", "6"] : ["", "2", "3", "4", "5"]).forEach(n => {
    (data.shade || []).forEach(s => shapes.push(shadeRect(n, s[0], s[1])));
    (run.gaps || []).forEach(s => shapes.push(gapRect(n, s[0], s[1])));
    shapes.push({type: "line", xref: "x" + n, yref: "y" + n + " domain", x0: t, x1: t, y0: 0, y1: 1, line: {color: "#fff", width: 1}});
    (data.switches || []).forEach(s => shapes.push({
      type: "line", xref: "x" + n, yref: "y" + n + " domain",
      x0: s.t, x1: s.t, y0: 0, y1: 1, line: {color: "#5dade2", width: 1, dash: "dot"}
    }));
  });
  const x = {domain: [0, 1], automargin: false, title: ""};
  const br = run.q_break;
  const rel = run.role === "r1rel";
  const a = rel ? "x" : "v1";
  const b = rel ? "y" : "v2";
  const c = rel ? "z" : "δn";
  const d = rel ? "rot deg" : "ΔR deg";
  const plots = [
    line(run.q_t, run.v1_p, a + " pred", DIM.v1.pred, "x", "dash", br),
    line(run.q_t, run.v2_p, b + " pred", DIM.v2.pred, "x", "dash", br),
    line(run.q_t, run.v1_l, a + " label", DIM.v1.label, "x", null, br),
    line(run.q_t, run.v2_l, b + " label", DIM.v2.label, "x", null, br),
    line(data.force_t, data.force, "||F||", "#e67e22", "x2", null, data.force_break),
    line(run.q_t, run.n_p, c + " pred", DIM.n.pred, "x3", "dash", br),
    line(run.q_t, run.n_l, c + " label", DIM.n.label, "x3", null, br),
    line(run.q_t, run.ang, d, DIM.ang.pred, "x4", "dash", br),
    line(run.q_t, run.stiff_p, "stiff pred", DIM.stiff.pred, "x5", "dash", br),
    line(run.q_t, run.stiff_l, "stiff label", DIM.stiff.label, "x5", null, br)
  ];
  const layout = {
    paper_bgcolor: "#111", plot_bgcolor: "#1c1c1c", font: {color: "#ddd"},
    margin: {l: 64, r: 16, t: 28, b: 28}, legend: {orientation: "h", y: 1.12},
    shapes,
    xaxis: Object.assign({anchor: "y"}, x),
    yaxis: {title: rel ? "x m" : "v m/s", domain: hasFrames ? [0.86, 1] : [0.80, 1], anchor: "x"},
    xaxis2: Object.assign({anchor: "y2", matches: "x"}, x),
    yaxis2: {title: "N", domain: hasFrames ? [0.70, 0.84] : [0.60, 0.78], anchor: "x2"},
    xaxis3: Object.assign({anchor: "y3", matches: "x"}, x),
    yaxis3: {title: rel ? "z m" : "δn m", domain: hasFrames ? [0.54, 0.68] : [0.40, 0.58], anchor: "x3"},
    xaxis4: Object.assign({anchor: "y4", matches: "x"}, x),
    yaxis4: {title: rel ? "rot deg" : "ΔR deg", domain: hasFrames ? [0.38, 0.52] : [0.20, 0.38], anchor: "x4"},
    xaxis5: Object.assign({anchor: "y5", matches: "x", title: hasFrames ? "" : "s"}, x),
    yaxis5: {title: "stiff", domain: hasFrames ? [0.22, 0.36] : [0, 0.18], anchor: "x5"}
  };
  if (hasFrames) {
    plots.push(line(data.frames.t, data.frames.dR_old, "old |ΔR|", "rgba(255,255,255,0.55)", "x6", "dash", data.frames.break));
    plots.push(line(data.frames.t, data.frames.dR_new, "new |ΔR|", "#ff9f1a", "x6", null, data.frames.break));
    plots.push({
      x: data.frames.jump_t, y: data.frames.jump_y, name: "old jump",
      type: "scatter", mode: "markers", xaxis: "x6", yaxis: "y6",
      marker: {color: "#e74c3c", size: 7, symbol: "diamond"}
    });
    if (data.switches && data.switches.length) {
      plots.push({
        x: data.switches.map(s => s.t), y: data.switches.map(s => s.frame),
        name: "crown-wall", type: "scatter", mode: "markers",
        xaxis: "x6", yaxis: "y6",
        marker: {color: "#5dade2", size: 9, symbol: "triangle-up"}
      });
    }
    layout.xaxis6 = Object.assign({anchor: "y6", matches: "x", title: "s"}, x);
    layout.yaxis6 = {title: "frame deg", domain: [0, 0.18], anchor: "x6"};
  }
  await Plotly.react("plots", plots, layout, {responsive: true});
  paintTable(run);
  let readout = t.toFixed(2) + " s";
  if (regionFilter !== "all" && picked0 && picked0.region) {
    const n = filteredQueries(picked0).length;
    readout = regionFilter + " " + (fi + 1) + "/" + n + " · " + readout;
  }
  document.getElementById("readout").textContent = readout;
  document.getElementById("time").value = fi;
  paintHeading(t);
  bindPicks();
}
async function show() {
  const episode = document.getElementById("ep").value;
  data = await (await fetch("/api/episode?ep=" + episode)).json();
  const horizon = data.runs[0] ? data.runs[0].exec_horizon : 6;
  document.getElementById("spanExec").textContent = horizon + " steps";
  document.getElementById("spanExec").classList.toggle("on", span === "exec");
  document.getElementById("spanFull").classList.toggle("on", span === "full");
  document.getElementById("split").textContent = data.split;
  document.getElementById("time").max = Math.max(0, data.path_t.length - 1);
  fi = 0;
  const ckpt = document.getElementById("ckpt");
  const keep = selectedLabel || ckpt.value;
  ckpt.innerHTML = data.runs.map(r => "<option>" + r.label + "</option>").join("");
  if (data.runs.some(r => r.label === keep)) ckpt.value = keep;
  selectedLabel = ckpt.value;
  const toggles = document.getElementById("toggles");
  toggles.innerHTML = "";
  data.runs.forEach(run => {
    if (!(run.label in shown)) shown[run.label] = true;
    const box = document.createElement("label");
    box.style.color = "#ddd";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = shown[run.label];
    input.onchange = () => { shown[run.label] = input.checked; paintCursor(); };
    box.appendChild(input);
    box.appendChild(document.createTextNode(" " + run.label));
    toggles.appendChild(box);
  });
  document.getElementById("headingBar").style.display = metaHeading ? "flex" : "none";
  document.getElementById("headingNote").textContent = metaHeading ? "loading heading-tilt…" : "";
  if (!data.heading) document.getElementById("headingPlots").style.display = "none";
  const frameToggles = document.getElementById("frame_toggles");
  frameToggles.innerHTML = "";
  if (data.frames) {
    [["old", "old frames"], ["new", "new frames"]].forEach(pair => {
      const box = document.createElement("label");
      box.style.color = "#ddd";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = shownFrame[pair[0]];
      input.onchange = () => { shownFrame[pair[0]] = input.checked; paintCursor(); };
      box.appendChild(input);
      box.appendChild(document.createTextNode(" " + pair[1] + " "));
      frameToggles.appendChild(box);
    });
  }
  await paintCursor();
  if (metaHeading) loadHeading(episode);
}
async function loadHeading(episode) {
  const pack = await (await fetch("/api/heading?ep=" + episode)).json();
  if (!data || String(data.episode) !== String(episode)) return;
  data.heading = pack;
  const headingOn = !!(pack.t && pack.t.length);
  document.getElementById("headingNote").textContent = pack.error || (headingOn ? "" : "no heading queries");
  if (!headingOn) document.getElementById("headingPlots").style.display = "none";
  paintCursor();
}
document.getElementById("time").oninput = () => { fi = +document.getElementById("time").value; paintCursor(); };
document.getElementById("ep").onchange = () => show();
document.getElementById("ckpt").onchange = () => {
  selectedLabel = document.getElementById("ckpt").value;
  paintCursor();
};
function setSpan(next) {
  span = next;
  document.getElementById("spanExec").classList.toggle("on", span === "exec");
  document.getElementById("spanFull").classList.toggle("on", span === "full");
  if (data) paintCursor();
}
function setHeadingStep(step) {
  headingStep = step;
  ["1", "6", "64"].forEach(name => {
    document.getElementById("hStep" + name).classList.toggle("on", name === step);
  });
  if (data) paintCursor();
}
document.getElementById("hStep1").onclick = () => setHeadingStep("1");
document.getElementById("hStep6").onclick = () => setHeadingStep("6");
document.getElementById("hStep64").onclick = () => setHeadingStep("64");
document.getElementById("hPred").onchange = (ev) => { showHeadingPred = ev.target.checked; if (data) paintCursor(); };
document.getElementById("hLabel").onchange = (ev) => { showHeadingLabel = ev.target.checked; if (data) paintCursor(); };
document.getElementById("hCausal").onchange = (ev) => { showCausal = ev.target.checked; if (data) paintCursor(); };
document.getElementById("spanExec").onclick = () => setSpan("exec");
document.getElementById("spanFull").onclick = () => setSpan("full");
function setGeom(next) {
  geomMode = next;
  document.getElementById("geomFoot").classList.toggle("on", geomMode === "foot");
  document.getElementById("geomPoint").classList.toggle("on", geomMode === "point");
  if (data) paintCursor();
}
document.getElementById("geomFoot").onclick = () => setGeom("foot");
document.getElementById("geomPoint").onclick = () => setGeom("point");
function setOnly(next) {
  onlySelected = next;
  document.getElementById("onlySel").classList.toggle("on", onlySelected);
  if (data) paintCursor();
}
document.getElementById("onlySel").onclick = () => setOnly(!onlySelected);
function setRegion(name) {
  regionFilter = name;
  document.getElementById("regAll").classList.toggle("on", name === "all");
  document.getElementById("regCrown").classList.toggle("on", name === "crown");
  document.getElementById("regFillet").classList.toggle("on", name === "fillet");
  document.getElementById("regWall").classList.toggle("on", name === "wall");
  fi = 0;
  if (data) paintCursor();
}
document.getElementById("regAll").onclick = () => setRegion("all");
document.getElementById("regCrown").onclick = () => setRegion("crown");
document.getElementById("regFillet").onclick = () => setRegion("fillet");
document.getElementById("regWall").onclick = () => setRegion("wall");
document.addEventListener("keydown", (ev) => {
  if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
  const tag = ev.target && ev.target.tagName;
  if (tag === "INPUT" || tag === "SELECT") return;
  if (!data) return;
  const slider = document.getElementById("time");
  fi = Math.max(0, Math.min(+slider.max, fi + (ev.key === "ArrowRight" ? 1 : -1)));
  slider.value = String(fi);
  paintCursor();
});
(async function init() {
  const meta = await (await fetch("/api/meta")).json();
  mesh = meta.mesh;
  metaHeading = !!meta.heading;
  const ep = document.getElementById("ep");
  ep.innerHTML = meta.episodes.map(e => {
    const split = meta.splits && meta.splits[String(e)];
    const label = split ? (e + " " + split) : String(e);
    return "<option value='" + e + "'>" + label + "</option>";
  }).join("");
  if (meta.switches || meta.nearcap) buildEvents(meta);
  show();
})();
function bindPicks() {
  const plots = document.getElementById("plots");
  const view = document.getElementById("view3d");
  if (plots._pick || typeof plots.on !== "function" || typeof view.on !== "function") return;
  plots._pick = true;
  plots.on("plotly_click", (ev) => {
    const t = ev.points && ev.points[0] && ev.points[0].x;
    if (typeof t !== "number" || !data || !data.path_t) return;
    fi = nearest(data.path_t, t);
    document.getElementById("time").value = fi;
    paintCursor();
  });
  view.on("plotly_click", (ev) => {
    const cd = ev.points && ev.points[0] && ev.points[0].customdata;
    if (typeof cd !== "number" || !data || !data.path_t) return;
    fi = nearest(data.path_t, cd);
    document.getElementById("time").value = fi;
    paintCursor();
  });
}
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    exports: list = []
    mesh = None
    frames = None
    heading = None

    def log_message(self, fmt, *args):
        return

    def _send(self, code, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        args = dict(part.split("=", 1) for part in query.split("&") if "=" in part)
        if path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            return
        if path == "/api/meta":
            episodes = sorted({ep for exp in self.exports for ep in exp["episodes"]})
            body = {"episodes": episodes, "mesh": self.mesh}
            extra = frame_episode_ids(self.frames)
            if extra:
                body["episodes"] = sorted(set(episodes) | set(extra))
                splits = {
                    str(ep): self.frames[ep]["split"]
                    for ep in extra
                    if self.frames[ep].get("split")
                }
                if splits:
                    body["splits"] = splits
                events = self.frames.get("events") or {}
                if events.get("switches"):
                    body["switches"] = events["switches"]
                if events.get("nearcap"):
                    body["nearcap"] = events["nearcap"]
            if self.heading is not None:
                body["heading"] = True
            self._send(200, json.dumps(body).encode(), "application/json")
            return
        try:
            episode = int(args.get("ep", "0"))
        except ValueError:
            self._send(400, b"{}", "application/json")
            return
        if path == "/api/episode":
            try:
                payload = episode_payload(self.exports, episode, frames=self.frames)
            except KeyError:
                if not self.frames or episode not in self.frames:
                    self._send(404, b"{}", "application/json")
                    return
                payload = frames_only_payload(self.frames, episode)
            self._send(200, json.dumps(payload).encode(), "application/json")
            return
        if path == "/api/heading":
            if self.heading is None:
                self._send(200, b'{"t":[]}', "application/json")
                return
            try:
                pack = self.heading.episode(episode)
            except Exception as exc:
                pack = {"t": [], "error": str(exc)}
            self._send(200, json.dumps(pack).encode(), "application/json")
            return
        if path == "/api/cursor":
            body = json.dumps(cursor_payload(
                self.exports, episode, int(args.get("q", "0")), frames=self.frames, span=args.get("span", "exec")
            )).encode()
            self._send(200, body, "application/json")
            return
        self._send(404, b"not found", "text/plain")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, action="append", default=None)
    parser.add_argument("--ckpt", type=Path, action="append", default=None,
                        help="Checkpoint path. Loads every export whose settings.json names it.")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--frames_npz", type=Path, default=None)
    parser.add_argument("--heading-tilt", action="store_true",
                        help="Add heading-tilt e100 and causal-foot e199 frames.")
    parser.add_argument("--heading-ckpt", type=Path, default=None)
    parser.add_argument("--causal-ckpt", type=Path, default=None)
    args = parser.parse_args()
    export_paths = list(args.export or [])
    for ckpt in args.ckpt or []:
        matched = exports_for_checkpoint(ckpt)
        if not matched:
            raise SystemExit(f"no open-loop export for {ckpt}")
        export_paths.extend(matched)
    # Keep the first copy when --export and --ckpt name the same directory.
    seen = set()
    unique = []
    for path in export_paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    if not unique:
        raise SystemExit("pass --export or --ckpt")
    field = np.load(args.mesh)
    vertices, faces = decimate_mesh(field["vertices"], field["faces"], field["is_apron"])
    Handler.mesh = mesh_payload(vertices, faces)
    Handler.exports = [load_export(path) for path in unique]
    labels = [exp["label"] for exp in Handler.exports]
    if len(set(labels)) != len(labels):
        for exp in Handler.exports:
            exp["label"] = f"{exp['label']} [{exp['dir'].name}]"
    Handler.frames = load_frames_npz(args.frames_npz) if args.frames_npz else None
    Handler.heading = None
    if args.heading_tilt:
        from heading_tilt_layer import CAUSAL_CKPT, HEADING_CKPT, HeadingLayer
        Handler.heading = HeadingLayer(
            str(args.heading_ckpt) if args.heading_ckpt else HEADING_CKPT,
            str(args.causal_ckpt) if args.causal_ckpt else CAUSAL_CKPT,
        )
        print("[view_openloop] heading-tilt layer on", flush=True)
    if Handler.frames is not None:
        print(f"[view_openloop] frame overlay on episodes {frame_episode_ids(Handler.frames)}", flush=True)
    print(f"[view_openloop] http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
