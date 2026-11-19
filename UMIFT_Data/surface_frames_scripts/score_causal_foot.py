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


def _load_val(ckpt: Path, zarr: Path, normalizer: Path, sha: str, reduced: bool):
    import export_openloop as exporter
    from omegaconf import OmegaConf

    if not Path(ckpt).is_file():
        raise SystemExit(f"missing checkpoint {ckpt}")
    if cf.file_sha256(normalizer) != sha:
        raise SystemExit(f"{normalizer} sha is not {sha}")
    policy, cfg, stock, device = exporter.load_policy(ckpt, "cpu")
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
    source["foot"] = np.asarray(group["mold_x_s_0"], dtype=np.float64)
    source["frame"] = np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
    return source


def score_checkpoints(label_result: dict, causal_ckpt: Path) -> dict:
    """Causal checkpoint versus track_clean epoch 95, same queries and bins."""
    import export_openloop as exporter

    queries = {
        episode: scored["queries"] for episode, scored in label_result["per_episode"].items()
    }
    walked = {
        episode: scored["walked"] for episode, scored in label_result["per_episode"].items()
    }
    causal_policy, causal_stock, causal_device, causal_val = _load_val(
        causal_ckpt, cf.CAUSAL_ZARR, cf.CAUSAL_NORM, cf.CAUSAL_SHA, True
    )
    track_policy, track_stock, track_device, track_val = _load_val(
        cf.TRACK_CKPT, cf.TRACK_ZARR, cf.TRACK_NORM, cf.TRACK_SHA, False
    )
    causal_matched, causal_missing = _match_queries(causal_val, queries)
    track_matched, track_missing = _match_queries(track_val, queries)
    both = {(ep, rgb) for _i, ep, rgb, _idx in causal_matched} & {
        (ep, rgb) for _i, ep, rgb, _idx in track_matched
    }
    runner = cf.load_runner()
    track_source = {}

    def rows_for(policy, stock, device, val, matched, use_track: bool):
        rows = []
        for sampler_index, episode, rgb_id, robot_index in matched:
            if (episode, rgb_id) not in both:
                continue
            obs, _action = val.sampler.sample_sequence(sampler_index)
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
        return rows

    return {
        "shared_queries": len(both),
        "causal_missing_from_sampler": causal_missing,
        "track_missing_from_sampler": track_missing,
        "causal_foot": cf.summarize(
            rows_for(causal_policy, causal_stock, causal_device, causal_val, causal_matched, False)
        ),
        "track_clean_e95": cf.summarize(
            rows_for(track_policy, track_stock, track_device, track_val, track_matched, True)
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=None, help="causal-foot epoch 40 checkpoint, job 43390127")
    parser.add_argument(
        "--out",
        type=Path,
        default=cf.EXPORT_ROOT / "causal_foot_gt_labels",
    )
    args = parser.parse_args(argv)
    result = cf.score_labels()
    text = cf.format_table(result["summary"])
    print(text, flush=True)
    if args.out.exists():
        print(f"[causal-foot] leaving existing export {args.out}", flush=True)
    else:
        _write_label_export(result, args.out)
        print(f"[causal-foot] viewer export {args.out}", flush=True)
    if args.ckpt is None:
        return
    compared = score_checkpoints(result, args.ckpt)
    payload = {
        "labels": result["summary"],
        "checkpoints": {
            "causal_foot": compared["causal_foot"],
            "track_clean_e95": compared["track_clean_e95"],
            "shared_queries": compared["shared_queries"],
        },
    }
    (args.out / "compare.json").write_text(json.dumps(payload, indent=2))
    print("[causal-foot] checkpoint", flush=True)
    print(cf.format_table(compared["causal_foot"]), flush=True)
    print("[track_clean e95]", flush=True)
    print(cf.format_table(compared["track_clean_e95"]), flush=True)


if __name__ == "__main__":
    main()
