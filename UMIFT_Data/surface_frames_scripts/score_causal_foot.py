#!/usr/bin/env python3
"""Score causal-foot open loop, and write a folder the existing viewer can open.

Labels only (no checkpoint):

  python UMIFT_Data/surface_frames_scripts/score_causal_foot.py

Point this at the epoch-40 causal-foot checkpoint. It also scores track_clean
epoch 95 on the same held-out episodes and the same metrics. Neither run is
started by importing this file.

  python UMIFT_Data/surface_frames_scripts/score_causal_foot.py \\
    --ckpt /path/to/43390127/epoch=0040-val_loss=....ckpt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import openloop_causal_foot as cf  # noqa: E402


def _require_fresh(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"refusing to write into an existing directory: {path}")
    path.mkdir(parents=True)


def _export_episode(episode: int, scored: dict, out: Path) -> None:
    walked = scored["walked"]
    queries = scored["queries"]
    if not queries:
        return
    runner = cf.load_runner()
    rgb = np.asarray([rgb_id for rgb_id, _index in queries], dtype=np.int64)
    index = np.stack([idx for _rgb, idx in queries])
    label = np.stack(
        [cf.action10(walked["v"], walked["dn"], walked["stiffness"], walked["dR"], idx) for idx in index]
    )
    label27 = runner.expand_r4_action(label)
    x_d = np.stack([walked["x_d"][idx] for idx in index])
    r_d = np.stack([walked["R_d"][idx] for idx in index])
    n_valid = np.full(len(queries), cf.EXEC_H, dtype=np.int64)
    exec_pos = x_d.reshape(-1, 3)
    exec_break = np.zeros(len(exec_pos), dtype=bool)
    exec_break[cf.EXEC_H - 1 :: cf.EXEC_H] = True
    if len(exec_break):
        exec_break[-1] = False
    bad = walked["bad"][index]
    long = np.zeros_like(bad)
    wrench_t = walked["wrench_t"]
    nearest = np.clip(np.searchsorted(wrench_t, walked["robot_t"][index], side="left"), 0, len(wrench_t) - 1)
    prev = np.clip(nearest - 1, 0, len(wrench_t) - 1)
    take_prev = np.abs(wrench_t[prev] - walked["robot_t"][index]) < np.abs(wrench_t[nearest] - walked["robot_t"][index])
    nearest = np.where(take_prev, prev, nearest)
    force_n = np.linalg.norm(walked["wrench"][nearest, :3], axis=-1)
    np.savez(
        out / f"episode_{episode}.npz",
        split="val",
        query_rgb_idx=rgb,
        query_time_s=walked["rgb_t"][rgb],
        n_valid=n_valid,
        pred_action=label27,
        label_action=label27,
        pred_pos=x_d,
        pred_R=r_d,
        raw_pos=x_d,
        raw_R=r_d,
        raw_foot=np.stack([walked["x_s"][idx] for idx in index]),
        raw_n=np.stack([walked["normal"][idx] for idx in index]),
        chunk_pos=x_d,
        chunk_R=r_d,
        chunk_n=n_valid,
        exec_pos=exec_pos,
        exec_R=r_d.reshape(-1, 3, 3),
        exec_time_s=walked["robot_t"][index].reshape(-1),
        exec_break=exec_break,
        action_time_s=walked["robot_t"][index],
        action_robot_idx=index,
        bad_span=bad,
        long_dt=long,
        force_n=force_n,
        tcp_xyz=walked["p"],
        tcp_R=walked["R"],
        robot_time_s=walked["robot_t"],
        rgb_time_s=walked["rgb_t"],
        force_series_t=walked["wrench_t"],
        force_series=np.linalg.norm(walked["wrench"][:, :3], axis=1),
        bad_robot=walked["bad"],
    )


def _write_label_export(result: dict, out: Path) -> None:
    _require_fresh(out)
    settings = {
        "role": "r4vt",
        "dataset": str(cf.CAUSAL_ZARR),
        "normalizer": str(cf.CAUSAL_NORM),
        "normalizer_sha256": cf.CAUSAL_SHA,
        "execution_horizon": cf.EXEC_H,
        "query_stride": cf.RGB_STRIDE,
        "episodes": result["episodes"],
        "decode": (
            "runner_causal_foot decode_chunk, apply_safety=False, "
            "tcp advanced by the demo's next robot sample"
        ),
        "checkpoint_epoch": None,
        "val_loss": None,
    }
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    (out / "report.json").write_text(json.dumps({"summary": result["summary"]}, indent=2))
    for episode, scored in result["per_episode"].items():
        _export_episode(int(episode), scored, out)


def _match_queries(dataset, queries_by_episode: dict[int, list[tuple[int, np.ndarray]]]):
    """Sampler index for each (episode, rgb) query. Missing queries are skipped."""
    index = {
        (int(episode), int(rgb)): i
        for i, (episode, rgb) in enumerate(dataset.sampler.indices)
    }
    matched = []
    missing = 0
    for episode, queries in queries_by_episode.items():
        for rgb_id, robot_index in queries:
            key = (int(episode), int(rgb_id))
            if key not in index:
                missing += 1
                continue
            matched.append((index[key], episode, rgb_id, robot_index))
    return matched, missing


def _load_val(ckpt: Path, zarr: Path, normalizer: Path, sha: str, reduced: bool, device_name: str):
    import export_openloop as exporter
    from omegaconf import OmegaConf

    if not Path(ckpt).is_file():
        raise SystemExit(f"missing checkpoint {ckpt}")
    if cf.file_sha256(normalizer) != sha:
        raise SystemExit(f"{normalizer} sha is not {sha}")
    policy, cfg, stock, device = exporter.load_policy(ckpt, device_name)
    scratch = Path("/tmp/causal_foot_score_exclusion")
    scratch.mkdir(parents=True, exist_ok=True)
    exporter.point_dataset(cfg, zarr, normalizer, scratch)
    OmegaConf.set_struct(cfg.task.dataset, False)
    cfg.task.dataset.action_normalizer_sha256 = sha
    cfg.task.dataset.pose_jump_flag_path = None
    cfg.task.dataset.exclude_episodes = [35, 44]
    cfg.task.dataset.long_dt_mask_path = str(cf.LONG_DT)
    if reduced:
        cfg.task.dataset.reduced_r4 = True
    _train, val = exporter.build_datasets(cfg)
    return policy, stock, device, val


def _track_source(episode: int, walked: dict) -> dict:
    """Track-clean foot and velocity. Region and the tool rotation stay on ``walked``."""
    import zarr

    group = zarr.open(str(cf.TRACK_ZARR), "r")["data"][f"episode_{episode}"]
    source = dict(walked)
    source["v"] = np.asarray(group["fixb_v_par_0"], dtype=np.float64)
    source["dn"] = np.asarray(group["fixb_delta_n_0"], dtype=np.float64).reshape(-1)
    source["dR"] = np.asarray(group["fixb_delta_R_rot6d_0"], dtype=np.float64)
    source["stiffness"] = np.asarray(group["stiffness_0"], dtype=np.float64).reshape(-1)
    source["foot"] = np.asarray(group["mold_x_s_0"], dtype=np.float64)
    source["frame"] = np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
    return source


def _sample_rgb(sampler, episode: int, rgb_id: int):
    """Observation at one rgb index, including queries the train sampler dropped."""
    saved = sampler.indices
    sampler.indices = [(int(episode), int(rgb_id))]
    try:
        return sampler.sample_sequence(0)
    finally:
        sampler.indices = saved


def _query_list(label_result: dict) -> list[tuple[int, int, np.ndarray]]:
    found = []
    for episode, scored in label_result["per_episode"].items():
        for rgb_id, robot_index in scored["queries"]:
            found.append((int(episode), int(rgb_id), robot_index))
    return found


def _score_policy(name, policy, stock, device, val, queries, walked, runner, use_track: bool):
    import export_openloop as exporter

    rows = []
    failed = []
    track_source = {}
    for done, (episode, rgb_id, robot_index) in enumerate(queries, start=1):
        try:
            obs, _action = _sample_rgb(val.sampler, episode, rgb_id)
        except Exception as exc:
            failed.append({"episode": episode, "rgb": rgb_id, "error": str(exc)})
            continue
        pred = np.asarray(
            exporter.predict_one(policy, obs, device, stock, "r4vt", rgb_id),
            dtype=np.float64,
        )[: cf.EXEC_H]
        source = None
        if use_track:
            if episode not in track_source:
                track_source[episode] = _track_source(episode, walked[episode])
            source = track_source[episode]
        rows.append(
            cf.errors_against_label(
                runner, walked[episode], robot_index, pred, pose_source=source
            )
        )
        if done % 50 == 0 or done == len(queries):
            print(f"[causal-foot] {name} {done}/{len(queries)}", flush=True)
    return rows, failed


def _load_policy(ckpt: Path, device_name: str):
    import export_openloop as exporter

    if not Path(ckpt).is_file():
        raise SystemExit(f"missing checkpoint {ckpt}")
    return exporter.load_policy(ckpt, device_name)


def score_checkpoints(label_result: dict, causal_ckpts: list[Path], device_name: str) -> dict:
    """Each causal checkpoint and track_clean epoch 95, on the label query list."""
    import gc

    import torch

    queries = _query_list(label_result)
    walked = {
        episode: scored["walked"] for episode, scored in label_result["per_episode"].items()
    }
    runner = cf.load_runner()
    causal = {}
    causal_val = None
    for i, ckpt in enumerate(causal_ckpts):
        if i == 0:
            policy, stock, device, causal_val = _load_val(
                ckpt, cf.CAUSAL_ZARR, cf.CAUSAL_NORM, cf.CAUSAL_SHA, True, device_name
            )
        else:
            policy, _cfg, stock, device = _load_policy(ckpt, device_name)
        rows, failed = _score_policy(
            ckpt.name, policy, stock, device, causal_val, queries, walked, runner, False
        )
        causal[str(ckpt)] = {"summary": cf.summarize(rows), "failed": failed, "n_scored": len(rows)}
        del policy, stock
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    track_policy, track_stock, track_device, track_val = _load_val(
        cf.TRACK_CKPT, cf.TRACK_ZARR, cf.TRACK_NORM, cf.TRACK_SHA, False, device_name
    )
    track_rows, track_failed = _score_policy(
        "track_clean_e95", track_policy, track_stock, track_device, track_val, queries, walked, runner, True
    )
    return {
        "n_queries": len(queries),
        "causal": causal,
        "track_clean_e95": {
            "ckpt": str(cf.TRACK_CKPT),
            "summary": cf.summarize(track_rows),
            "failed": track_failed,
            "n_scored": len(track_rows),
        },
    }


R1REL_ZARR = cf.DATA_ROOT / "r1rel_world" / "acp_replay_buffer.zarr"
R1REL_NORM = SCRIPTS / "assets" / "oct_v35_h4s_r1rel_odom_action_pose_normalizer_2c20fba56851.pkl"
R1REL_SHA = "2c20fba56851f59ec7d1a652d4e08a979ec9f7a8ea46368737df984475a0cf05"
R1REL_NOTE = (
    "r1rel labels still contain the T265 glitches. The glitch clean was applied "
    "to the track_clean and causal-foot zarrs only; r1rel_world was left unchanged."
)


def _fixture_pose(action, base_p, base_R) -> tuple[np.ndarray, np.ndarray]:
    """Relative action21 to a fixture-frame tool pose. Same postprocess as the runner."""
    import PyriteUtility.spatial_math.spatial_utilities as su
    from PyriteConfig.tasks.umift.umift_type_conversions import action21_postprocess

    base = np.eye(4, dtype=np.float64)
    base[:3, :3] = base_R
    base[:3, 3] = base_p
    command, _vt, _stiff, _eoat = action21_postprocess(np.asarray(action, dtype=np.float64), [base], [0])
    pose = su.SE3_to_pose7(command[0])
    se3 = su.pose7_to_SE3(pose)
    return pose[:, :3], se3[:, :3, :3]


def _sparse_action(action_sample) -> np.ndarray:
    action = action_sample
    if isinstance(action, dict):
        action = action.get("sparse", action)
    if isinstance(action, dict):
        action = action["action"] if "action" in action else next(iter(action.values()))
    return np.asarray(action, dtype=np.float64)


def _pose_only(rows: list[dict]) -> dict:
    for row in rows:
        n = len(row["position_mm"])
        for key in ("delta_n_mm", "v1_mm_s", "v2_mm_s", "stiffness"):
            row[key] = np.full(n, np.nan)
    summary = cf.summarize(rows)
    for block in summary.values():
        for key in ("delta_n_mm", "v1_mm_s", "v2_mm_s", "stiffness"):
            block.pop(key, None)
    return summary


def score_r1rel(ckpt: Path, device_name: str) -> dict:
    """r1rel tool-pose error on the causal-foot query list, in the fixture frame."""
    import export_openloop as exporter
    import zarr
    from fixb_labels import pose7_to_Rp

    listed = cf.index_queries()
    policy, stock, device, val = _load_val(
        ckpt, R1REL_ZARR, R1REL_NORM, R1REL_SHA, False, device_name
    )
    root = zarr.open(str(R1REL_ZARR), "r")["data"]
    runner = listed["runner"]
    rows = []
    failed = []
    cached = {}
    queries = []
    for episode, scored in listed["per_episode"].items():
        for rgb_id, robot_index in scored["queries"]:
            queries.append((episode, rgb_id, robot_index))
    for done, (episode, rgb_id, robot_index) in enumerate(queries, start=1):
        if episode not in cached:
            group = root[f"episode_{episode}"]
            rotation, position = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
            robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
            rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
            causal = listed["per_episode"][episode]
            if robot_t.shape != causal["robot_t"].shape or rgb_t.shape != causal["rgb_t"].shape:
                raise SystemExit(f"episode {episode} r1rel time grid does not match the causal-foot grid")
            if np.max(np.abs(robot_t - causal["robot_t"])) > 1e-6 or np.max(np.abs(rgb_t - causal["rgb_t"])) > 1e-6:
                raise SystemExit(f"episode {episode} r1rel timestamps differ from the causal-foot queries")
            cached[episode] = (position, rotation, robot_t, rgb_t)
        try:
            obs, label_action = _sample_rgb(val.sampler, episode, rgb_id)
        except Exception as exc:
            failed.append({"episode": episode, "rgb": rgb_id, "error": str(exc)})
            continue
        pred = np.asarray(
            exporter.predict_one(policy, obs, device, stock, "r1rel", rgb_id),
            dtype=np.float64,
        )[: cf.EXEC_H]
        label = _sparse_action(label_action)[: cf.EXEC_H]
        position, rotation, _robot_t, _rgb_t = cached[episode]
        anchor = int(robot_index[0])
        pred_xyz, pred_R = _fixture_pose(pred, position[anchor], rotation[anchor])
        label_xyz, label_R = _fixture_pose(label, position[anchor], rotation[anchor])
        causal = listed["per_episode"][episode]
        index = np.asarray(robot_index, dtype=np.int64)
        rows.append(
            {
                "position_mm": np.linalg.norm(pred_xyz - label_xyz, axis=1) * 1e3,
                "rotation_deg": cf.geodesic_deg(pred_R, label_R),
                "tilt_deg": cf._axis_deg(pred_R[:, :, 2], label_R[:, :, 2]),
                "region": cf.region_code(causal["normal"][index]),
                "mode": cf._mode(causal["dR"][index], runner),
            }
        )
        if done % 50 == 0 or done == len(queries):
            print(f"[causal-foot] r1rel {done}/{len(queries)}", flush=True)
    return {
        "ckpt": str(ckpt),
        "zarr": str(R1REL_ZARR),
        "normalizer_sha256": R1REL_SHA,
        "n_queries": len(queries),
        "n_scored": len(rows),
        "n_failed": len(failed),
        "failed": failed[:5],
        "note": R1REL_NOTE,
        "summary": _pose_only(rows),
    }


def _side_by_side(r1rel: dict, epoch40: dict, track: dict) -> str:
    lines = []
    for key, title in (
        ("position_mm", "position mm"),
        ("rotation_deg", "rotation °"),
        ("tilt_deg", "tilt °"),
    ):
        lines.append(title)
        lines.append(f"{'split':<16} {'n':>6} {'r1rel':>16} {'epoch 40':>16} {'track e95':>16}")
        order = ("all",) + cf.REGION_NAMES + ("near_identity", "return_yaw")
        for name in order:
            cells = []
            count = ""
            for summary in (r1rel, epoch40, track):
                pair = (summary.get(name) or {}).get(key)
                if not pair:
                    cells.append(f"{'—':>16}")
                    continue
                count = str(pair["n"])
                cells.append(f"{pair['median']:.3g} / {pair['p90']:.3g}".rjust(16))
            lines.append(f"{name:<16} {count:>6} {''.join(cells)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _write_r1rel_report(args) -> None:
    prior = json.loads(args.prior_report.read_text())
    epoch40 = None
    for path, item in prior["causal"].items():
        if "epoch=0040" in path:
            epoch40 = item["summary"]
            epoch40_path = path
    if epoch40 is None:
        raise SystemExit(f"no epoch 40 summary in {args.prior_report}")
    track = prior["track_clean_e95"]["summary"]
    out = args.out or (cf.EXPORT_ROOT / "causal_foot_r1rel_e40_track")
    scored = score_r1rel(args.r1rel_ckpt, args.device)
    _require_fresh(out)
    text = _side_by_side(scored["summary"], epoch40, track)
    print(text, flush=True)
    print("[r4vt only] δn mm", flush=True)
    print(_metric_table("delta_n_mm", epoch40, track), flush=True)
    print("[r4vt only] v1 mm/s", flush=True)
    print(_metric_table("v1_mm_s", epoch40, track), flush=True)
    print("[r4vt only] v2 mm/s", flush=True)
    print(_metric_table("v2_mm_s", epoch40, track), flush=True)
    print(scored["note"], flush=True)
    payload = {
        "episodes": [9, 12, 28, 37, 41],
        "n_queries": scored["n_queries"],
        "units": "median / p90. Position, rotation, and tilt are fixture-frame tool pose versus the label.",
        "note": scored["note"],
        "r1rel": {key: scored[key] for key in ("ckpt", "zarr", "normalizer_sha256", "n_scored", "n_failed", "summary")},
        "epoch_40": {"ckpt": epoch40_path, "summary": epoch40},
        "track_clean_e95": {
            "ckpt": prior["track_clean_e95"]["ckpt"],
            "summary": track,
        },
        "table": text,
    }
    (out / "report.json").write_text(json.dumps(payload, indent=2))
    (out / "settings.json").write_text(
        json.dumps(
            {
                "role": "r1rel+r4vt",
                "episodes": [9, 12, 28, 37, 41],
                "execution_horizon": cf.EXEC_H,
                "query_stride": cf.RGB_STRIDE,
                "r1rel_ckpt": str(args.r1rel_ckpt),
                "r1rel_zarr": str(R1REL_ZARR),
                "epoch_40": epoch40_path,
                "track_clean": prior["track_clean_e95"]["ckpt"],
                "prior_report": str(args.prior_report),
                "note": scored["note"],
            },
            indent=2,
        )
    )
    print(f"[causal-foot] report {out / 'report.json'}", flush=True)
    print(f"[causal-foot] r1rel scored {scored['n_scored']} failed {scored['n_failed']}", flush=True)


def _metric_table(key: str, epoch40: dict, track: dict) -> str:
    lines = [f"{'split':<16} {'n':>6} {'epoch 40':>16} {'track e95':>16}"]
    order = ("all",) + cf.REGION_NAMES + ("near_identity", "return_yaw")
    for name in order:
        cells = []
        count = ""
        for summary in (epoch40, track):
            pair = (summary.get(name) or {}).get(key)
            if not pair:
                cells.append(f"{'—':>16}")
                continue
            count = str(pair["n"])
            cells.append(f"{pair['median']:.3g} / {pair['p90']:.3g}".rjust(16))
        lines.append(f"{name:<16} {count:>6} {''.join(cells)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, action="append", default=None, help="causal-foot checkpoint; repeat for each")
    parser.add_argument("--r1rel-ckpt", type=Path, default=None)
    parser.add_argument(
        "--prior-report",
        type=Path,
        default=cf.EXPORT_ROOT / "causal_foot_43390127_e25_e40" / "report.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.r1rel_ckpt is not None:
        _write_r1rel_report(args)
        return
    result = cf.score_labels()
    print(cf.format_table(result["summary"]), flush=True)
    if not args.ckpt:
        out = args.out or (cf.EXPORT_ROOT / "causal_foot_gt_labels")
        if out.exists():
            print(f"[causal-foot] leaving existing export {out}", flush=True)
        else:
            _write_label_export(result, out)
            print(f"[causal-foot] viewer export {out}", flush=True)
        return
    out = args.out or (cf.EXPORT_ROOT / "causal_foot_43390127_e25_e40")
    _require_fresh(out)
    compared = score_checkpoints(result, args.ckpt, args.device)
    payload = {
        "episodes": result["episodes"],
        "n_queries": compared["n_queries"],
        "units": "median / p90; v1 and v2 are mm/s; stiffness is the label scalar",
        "labels": result["summary"],
        "causal": {
            path: {"summary": item["summary"], "n_scored": item["n_scored"], "n_failed": len(item["failed"])}
            for path, item in compared["causal"].items()
        },
        "track_clean_e95": {
            "ckpt": compared["track_clean_e95"]["ckpt"],
            "summary": compared["track_clean_e95"]["summary"],
            "n_scored": compared["track_clean_e95"]["n_scored"],
            "n_failed": len(compared["track_clean_e95"]["failed"]),
        },
    }
    (out / "report.json").write_text(json.dumps(payload, indent=2))
    (out / "settings.json").write_text(
        json.dumps(
            {
                "role": "r4vt",
                "dataset": str(cf.CAUSAL_ZARR),
                "episodes": result["episodes"],
                "execution_horizon": cf.EXEC_H,
                "query_stride": cf.RGB_STRIDE,
                "checkpoints": [str(path) for path in args.ckpt],
                "track_clean": str(cf.TRACK_CKPT),
            },
            indent=2,
        )
    )
    print(f"[causal-foot] report {out / 'report.json'}", flush=True)
    for path, item in compared["causal"].items():
        print(f"[causal-foot] {path} scored {item['n_scored']} failed {len(item['failed'])}", flush=True)
        print(cf.format_table(item["summary"]), flush=True)
    print(
        f"[track_clean e95] scored {compared['track_clean_e95']['n_scored']} "
        f"failed {len(compared['track_clean_e95']['failed'])}",
        flush=True,
    )
    print(cf.format_table(compared["track_clean_e95"]["summary"]), flush=True)


if __name__ == "__main__":
    main()
