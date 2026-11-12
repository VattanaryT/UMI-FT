#!/usr/bin/env python3
"""Export open-loop predictions for one checkpoint. No GUI.

Writes a new directory of per-episode npz files plus settings.json and report.json.
Does not modify the checkpoint, the zarr, or any normalizer pickle.

  python UMIFT_Data/surface_frames_scripts/export_openloop.py \\
    --role r4vt \\
    --ckpt /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_odom_r4vt/checkpoints/epoch=0185-train_loss=0.011.ckpt \\
    --val-loss 0.07917
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import openloop_lib as ol  # noqa: E402

DATA_ROOT = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
EXPORT_ROOT = Path("/mnt/data/umidata/ft/open_loop_exports")
R4VT_ZARR = DATA_ROOT / "r4vt_world_fixb_tilt" / "acp_replay_buffer.zarr"
R1REL_ZARR = DATA_ROOT / "r1rel_world" / "acp_replay_buffer.zarr"
FIELD_NPZ = DATA_ROOT / "shared" / "fixb_field.npz"
LONG_DT = DATA_ROOT / "shared" / "long_dt_gt_100ms.npz"
R4VT_NORMALIZER = (
    SCRIPTS
    / "assets"
    / "oct_v35_h4s_r4vt_action_normalizer_minmax_exbad.pkl"
)

# offline_compare_driver.query_seed. Repeated here so this file does not import
# that driver (it pulls the stock exporter). The test checks the two match.
def query_seed(model_key: str, query_id: int) -> int:
    digest = hashlib.sha256(f"{model_key}:{int(query_id)}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def epoch_from_name(path: Path) -> int | None:
    match = re.search(r"epoch=(\d+)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def require_fresh(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"refusing to write into an existing directory: {path}")
    path.mkdir(parents=True)


def _scripts_on_path() -> None:
    root = SCRIPTS.parents[1]
    for entry in (root, root / "PyriteML", root / "PyriteUtility", SCRIPTS):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)


def load_stock():
    """The stock exporter is a script, not a package. Same loader as offline_compare_driver."""
    import importlib.util

    path = SCRIPTS.parents[1] / "PyriteEnvSuites" / "scripts" / "run_policy_open_loop_val_export.py"
    spec = importlib.util.spec_from_file_location("run_policy_open_loop_val_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_policy(ckpt: Path, device_name: str):
    _scripts_on_path()
    import torch

    stock = load_stock()

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda was requested but is not available")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    policy, cfg = stock.load_policy_and_cfg(str(ckpt), device)
    return policy, cfg, stock, device


def point_dataset(cfg, dataset_path: Path, normalizer: Path, out: Path):
    from omegaconf import OmegaConf

    OmegaConf.update(cfg, "task.dataset.dataset_path", str(dataset_path), merge=True)
    OmegaConf.update(cfg, "task.dataset.action_normalizer_path", str(normalizer), merge=True)
    OmegaConf.update(cfg, "task.dataset.long_dt_mask_path", str(LONG_DT), merge=True)
    OmegaConf.update(
        cfg, "task.dataset.exclusion_stats_path", str(out / "exclusion_stats.json"), merge=True
    )
    OmegaConf.update(cfg, "task.dataset.pose_jump_flag_path", None, merge=True)
    OmegaConf.update(cfg, "task.dataset.filter_stats_path", None, merge=True)


def build_datasets(cfg):
    import hydra

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    val = dataset.get_validation_dataset()
    return dataset, val


def lowest_episode(dataset) -> int:
    ids = sorted({int(epi) for epi, _rgb in dataset.sampler.indices})
    if not ids:
        raise SystemExit("split has no queries after the exclusion filter")
    return ids[0]


def long_dt_sets(path: Path) -> dict[int, set[int]]:
    blob = np.load(path)
    out: dict[int, set[int]] = {}
    for ep, idx in zip(blob["flagged_episode"], blob["flagged_index"]):
        out.setdefault(int(ep), set()).add(int(idx))
    return out


def nearest_wrench(wrench_t, wrench, times) -> np.ndarray:
    idx = np.searchsorted(wrench_t, times, side="left")
    idx = np.clip(idx, 0, len(wrench_t) - 1)
    prev = np.clip(idx - 1, 0, len(wrench_t) - 1)
    take_prev = np.abs(wrench_t[prev] - times) < np.abs(wrench_t[idx] - times)
    idx = np.where(take_prev, prev, idx)
    return np.linalg.norm(wrench[idx, :3], axis=1)


def predict_one(policy, obs, device, stock, model_key: str, rgb_id: int) -> np.ndarray:
    import torch

    torch.manual_seed(query_seed(model_key, rgb_id))
    return stock_predict(policy, obs, device, stock)


def stock_predict(policy, obs, device, stock) -> np.ndarray:
    """Same batching as offline_compare_driver.predict_recorded. Depth is not zeroed here."""
    import copy

    obs = copy.deepcopy(obs)

    def to_batch(value):
        tensor = stock.torch.as_tensor(np.asarray(value)).to(device)
        return stock.rearrange(tensor, "... -> 1 ...")

    batched = stock.dict_apply(obs, to_batch)
    with stock.torch.no_grad():
        result = policy.predict_action(batched)
    key = "sparse" if result.get("sparse") is not None else "action"
    return result[key][0].detach().to("cpu").numpy()


def episode_arrays(zarr_path: Path, episode: int) -> dict:
    import zarr

    group = zarr.open(str(zarr_path), mode="r")["data"][f"episode_{episode}"]
    from fixb_labels import pose7_to_Rp

    pose = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    R, p = pose7_to_Rp(pose)
    robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
    wrench_t = np.asarray(group["wrench_time_stamps_0"], dtype=np.float64).reshape(-1)
    bad = np.asarray(group["bad_span_0"], dtype=bool).reshape(-1)
    return {
        "R": R,
        "p": p,
        "robot_t": robot_t,
        "rgb_t": rgb_t,
        "wrench": wrench,
        "wrench_t": wrench_t,
        "bad": bad,
        "v_par": np.asarray(group["fixb_v_par_0"], dtype=np.float64),
        "delta_n": np.asarray(group["fixb_delta_n_0"], dtype=np.float64).reshape(-1),
        "dR": np.asarray(group["fixb_delta_R_rot6d_0"], dtype=np.float64).reshape(-1, 6),
        "stiffness": np.asarray(group["stiffness_0"], dtype=np.float64).reshape(-1),
        "x_s": np.asarray(group["mold_x_s_0"], dtype=np.float64).reshape(-1, 3),
        "n": np.asarray(group["mold_normal_0"], dtype=np.float64).reshape(-1, 3),
        "R_field": np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3),
    }


def export_episode(policy, dataset, stock, device, runtime, episode, split, model_key, tcp, long_idx):
    from diffusion_policy.common.sampler_umift import sparse_action_indices_from_rgb_timestamps

    horizon = int(dataset.shape_meta["sample"]["action"]["sparse"]["horizon"])
    stride = int(dataset.shape_meta["sample"]["action"]["sparse"]["down_sample_steps"])
    queries = [
        (i, int(rgb))
        for i, (epi, rgb) in enumerate(dataset.sampler.indices)
        if int(epi) == int(episode)
    ]
    if not queries:
        raise SystemExit(f"episode {episode} has no {split} queries")

    preds, labels = [], []
    pred_pos, pred_R = [], []
    label_pos, label_R = [], []
    pred_xs, pred_n = [], []
    times_rows, idx_rows = [], []
    force_rows, bad_rows, long_rows = [], [], []
    n_valids, rgb_ids, seeds, qtimes = [], [], [], []

    for done, (idx, rgb_id) in enumerate(queries, start=1):
        obs, action = dataset.sampler.sample_sequence(idx)
        label = np.asarray(action["sparse"], dtype=np.float64)
        pred = predict_one(policy, obs, device, stock, model_key, rgb_id)
        if pred.shape != label.shape:
            raise SystemExit(
                f"episode {episode} rgb {rgb_id}: pred {pred.shape} vs label {label.shape}"
            )
        action_idx, target_t = sparse_action_indices_from_rgb_timestamps(
            action_times=tcp["robot_t"],
            rgb_times=tcp["rgb_t"],
            rgb_id=rgb_id,
            action_horizon=horizon,
            rgb_frame_stride=stride,
        )
        n_valid = int(len(action_idx))
        H = pred.shape[0]
        if n_valid > H:
            raise SystemExit(f"n_valid {n_valid} exceeds horizon {H}")
        # Padded tail repeats the last real target time so integrate_fixb has H steps.
        if n_valid == 0:
            raise SystemExit(f"episode {episode} rgb {rgb_id} has no action targets")
        full_t = np.zeros(H, dtype=np.float64)
        full_idx = np.full(H, -1, dtype=np.int64)
        full_t[:n_valid] = target_t
        full_idx[:n_valid] = action_idx
        if n_valid < H:
            dt_last = target_t[-1] - target_t[-2] if n_valid > 1 else 1.0 / 15.0
            for k in range(n_valid, H):
                full_t[k] = full_t[k - 1] + dt_last
        q_robot = int(np.searchsorted(tcp["robot_t"], tcp["rgb_t"][rgb_id]))
        q_robot = min(max(q_robot, 0), len(tcp["robot_t"]) - 1)
        p0 = tcp["p"][q_robot]
        R0 = tcp["R"][q_robot]
        dec_p = ol.decode_r4vt(runtime, pred, p0, R0, full_t)
        dec_l = ol.decode_r4vt(runtime, label, p0, R0, full_t)
        valid_idx = full_idx.copy()
        valid_idx[n_valid:] = full_idx[n_valid - 1]
        force = nearest_wrench(tcp["wrench_t"], tcp["wrench"], full_t)
        bad = np.zeros(H, dtype=bool)
        long = np.zeros(H, dtype=bool)
        known = full_idx >= 0
        bad[known] = tcp["bad"][full_idx[known]]
        long[known] = np.isin(full_idx[known], list(long_idx))

        preds.append(pred)
        labels.append(label)
        pred_pos.append(dec_p["x_d"])
        pred_R.append(dec_p["R_d"])
        label_pos.append(dec_l["x_d"])
        label_R.append(dec_l["R_d"])
        pred_xs.append(dec_p["x_s"])
        pred_n.append(dec_p["n"])
        times_rows.append(full_t)
        idx_rows.append(full_idx)
        force_rows.append(force)
        bad_rows.append(bad)
        long_rows.append(long)
        n_valids.append(n_valid)
        rgb_ids.append(rgb_id)
        seeds.append(query_seed(model_key, rgb_id))
        qtimes.append(float(tcp["rgb_t"][rgb_id]))
        if done % 25 == 0 or done == len(queries):
            print(f"  episode {episode} {split} {done}/{len(queries)}", flush=True)

    return {
        "split": split,
        "query_rgb_idx": np.asarray(rgb_ids, dtype=np.int64),
        "query_time_s": np.asarray(qtimes, dtype=np.float64),
        "query_seed": np.asarray(seeds, dtype=np.int64),
        "n_valid": np.asarray(n_valids, dtype=np.int64),
        "pred_action": np.stack(preds),
        "label_action": np.stack(labels),
        "pred_pos": np.stack(pred_pos),
        "pred_R": np.stack(pred_R),
        "label_pos": np.stack(label_pos),
        "label_R": np.stack(label_R),
        "pred_x_s": np.stack(pred_xs),
        "pred_n": np.stack(pred_n),
        "action_time_s": np.stack(times_rows),
        "action_robot_idx": np.stack(idx_rows),
        "force_n": np.stack(force_rows),
        "bad_span": np.stack(bad_rows),
        "long_dt": np.stack(long_rows),
        "tcp_xyz": tcp["p"],
        "tcp_R": tcp["R"],
        "robot_time_s": tcp["robot_t"],
        "rgb_time_s": tcp["rgb_t"],
        "force_series_t": tcp["wrench_t"],
        "force_series": np.linalg.norm(tcp["wrench"][:, :3], axis=1),
        "bad_robot": tcp["bad"],
    }


def flat_valid(blob) -> dict:
    pred = blob["pred_action"].reshape(-1, blob["pred_action"].shape[-1])
    label = blob["label_action"].reshape(-1, blob["label_action"].shape[-1])
    H = blob["pred_action"].shape[1]
    valid = np.arange(H)[None, :] < blob["n_valid"][:, None]
    return {
        "pred": pred,
        "label": label,
        "force_n": blob["force_n"].reshape(-1),
        "bad_span": blob["bad_span"].reshape(-1),
        "long_dt": blob["long_dt"].reshape(-1),
        "valid": valid.reshape(-1),
    }


def vt_offset_for_episode(episode: int) -> dict:
    import zarr
    from fixb_labels import pose7_to_Rp

    r4 = zarr.open(str(R4VT_ZARR), mode="r")["data"][f"episode_{episode}"]
    rel = zarr.open(str(R1REL_ZARR), mode="r")["data"][f"episode_{episode}"]
    x_s = np.asarray(r4["mold_x_s_0"], dtype=np.float64).reshape(-1, 3)
    n = np.asarray(r4["mold_normal_0"], dtype=np.float64).reshape(-1, 3)
    _R, p_vt = pose7_to_Rp(np.asarray(rel["ts_pose_virtual_target_0"]))
    m = min(len(x_s), len(p_vt))
    return ol.tangential_vt_offset(p_vt[:m], x_s[:m], n[:m])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("r4vt", "r1rel"), required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--val-loss", type=float, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--normalizer", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.role != "r4vt":
        raise SystemExit("this run is r4vt only; r1rel export is not implemented in this pass")

    ckpt = args.ckpt.resolve()
    if not ckpt.is_file():
        raise SystemExit(f"missing checkpoint {ckpt}")
    dataset_path = args.dataset or R4VT_ZARR
    normalizer = args.normalizer or R4VT_NORMALIZER
    for path in (dataset_path, normalizer, FIELD_NPZ, LONG_DT, R1REL_ZARR):
        if not path.exists():
            raise SystemExit(f"missing {path}")

    started = time.perf_counter()
    digest = sha256_file(ckpt)
    epoch = epoch_from_name(ckpt)
    out = args.out or (EXPORT_ROOT / f"r4vt_{digest[:12]}")
    require_fresh(out)

    settings = {
        "role": "r4vt",
        "ckpt": str(ckpt),
        "ckpt_sha256": digest,
        "checkpoint_epoch": epoch,
        "val_loss": float(args.val_loss),
        "safety_clamps": False,
        "dry_run": False,
        "decode": "MoldAwareRuntime.decode_chunk apply_safety=False dry_run=False",
        "decode_note": "x_d = x_s + delta_n n; R_d = R(x_s) @ Gram-Schmidt(action[:, 21:27])",
        "seed": "sha256(f'{role}:{rgb_id}')[:8] little-endian, same as offline_compare_driver.query_seed",
        "dataset": str(dataset_path),
        "normalizer": str(normalizer),
        "normalizer_sha256": sha256_file(normalizer),
        "field": str(FIELD_NPZ),
        "long_dt": str(LONG_DT),
        "contact_n": ol.CONTACT_N,
    }
    (out / "settings.json").write_text(json.dumps(settings, indent=2))

    print(f"loading {ckpt}", flush=True)
    policy, cfg, stock, device = load_policy(ckpt, args.device)
    from diffusion_policy.common.drop_depth_inputs import apply_drop_depth_inputs

    apply_drop_depth_inputs(cfg)
    point_dataset(cfg, dataset_path, normalizer, out)
    print(f"building dataset {dataset_path}", flush=True)
    train_ds, val_ds = build_datasets(cfg)
    if args.episodes:
        chosen = [(int(ep), None) for ep in args.episodes]
    else:
        chosen = [(lowest_episode(train_ds), "train"), (lowest_episode(val_ds), "val")]

    from fixb_field import FixBMoldField

    runtime = ol.make_r4_runtime(FixBMoldField(FIELD_NPZ), dry_run=False)
    long_sets = long_dt_sets(LONG_DT)
    report_eps = []
    for episode, hinted in chosen:
        split = hinted
        if split is None:
            val_ids = {int(epi) for epi, _ in val_ds.sampler.indices}
            split = "val" if episode in val_ids else "train"
        dataset = val_ds if split == "val" else train_ds
        # An explicitly requested id must actually sit in that split.
        ids = {int(epi) for epi, _ in dataset.sampler.indices}
        if episode not in ids:
            other = "train" if split == "val" else "val"
            raise SystemExit(f"episode {episode} is not in the {split} split (it is {other} or empty)")
        print(f"episode {episode} {split}", flush=True)
        tcp = episode_arrays(dataset_path, episode)
        blob = export_episode(
            policy,
            dataset,
            stock,
            device,
            runtime,
            episode,
            split,
            "r4vt",
            tcp,
            long_sets.get(episode, set()),
        )
        flat = flat_valid(blob)
        table = ol.episode_error_table(
            flat["pred"], flat["label"], flat["force_n"], flat["bad_span"], flat["long_dt"], flat["valid"]
        )
        offset = vt_offset_for_episode(episode)
        np.savez_compressed(out / f"episode_{episode}.npz", **blob)
        report_eps.append(
            {
                "episode": episode,
                "split": split,
                "n_queries": int(len(blob["query_rgb_idx"])),
                "error": table,
                "tangential_vt_offset_m": offset,
            }
        )
        print(
            f"episode {episode} {split} queries={len(blob['query_rgb_idx'])} "
            f"vt tangential median={offset['median_m']:.6f} m max={offset['max_m']:.6f} m",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    report = {"seconds": elapsed, "episodes": report_eps}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    settings["episodes"] = [
        {"episode": row["episode"], "split": row["split"], "n_queries": row["n_queries"]}
        for row in report_eps
    ]
    settings["seconds"] = elapsed
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    print(f"wrote {out} in {elapsed:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
