"""How much does giving the policy a scan from a *different* sample hurt validation?

For a trained checkpoint, evaluates the validation split (same episodes / seed as in
training) under paired conditions that share every random draw:

  real              unmodified batch
  scan_shuffled     scan_heatmap_0 replaced by the scan of another validation sample
                    from a *different episode* (the crop, if the policy uses one, is
                    the donor's crop, i.e. "what the policy would see somewhere else")
  scan_zero         scan_heatmap_0 set to 0 (the "no scan yet / flat" input)
  shuffle:<key>     optional references (--also-shuffle rgb_0 ...): same donor swap
                    for another obs key, to put the scan number on a scale

Two metrics:

1. Denoising loss -- exactly the training ``val_loss`` objective, but with the noise,
   input perturbation and diffusion timesteps fixed per sample and shared by all
   conditions (averaged over --loss-draws draws), so differences are due to the
   observation swap only. Reported with a paired standard error.
2. Sampled actions (first --sample-batches batches) -- DDIM sampling from the same
   initial noise per condition: normalised action MSE vs ground truth (the training
   ``val_sparse_naction_mse_error``), target-position error vs ground truth in mm,
   and how far each condition moves the predicted target positions away from the
   real-scan prediction. ``real_reseeded`` (real obs, different initial noise) gives
   the sampling-noise floor: changes no bigger than that mean the scan is not used.

Usage (from UMI-FT/):
  PYTHONPATH=PyriteML:.:PyriteUtility:/home/boeing/umi_ft_ws \
    python PyriteML/scripts/eval_scan_shuffle_ablation.py \
      --ckpt /mnt/data/umidata/ft/training_outputs/stringer_scanner_v2/og/epoch=0035-train_loss=0.019.ckpt \
      --dataset /mnt/data/umidata/ft/datasets/stringer_scanner_v2/acp_replay_buffer.zarr

Writes ``<ckpt stem>_scan_shuffle_eval.json`` next to the checkpoint (or --out).
"""

import argparse
import json
import pathlib
import time
from typing import Dict, List

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

SCAN_KEY = "scan_heatmap_0"


def load_policy_and_cfg(ckpt_path: str, device: torch.device):
    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir="/tmp/_scan_shuffle_eval_ws")
    workspace.load_payload(payload, exclude_keys=["optimizer"], include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.num_inference_steps = cfg.policy.num_inference_steps
    policy.eval().to(device)
    return policy, cfg


def build_val_dataset(cfg, dataset_path: str):
    OmegaConf.set_struct(cfg, False)
    cfg.task.dataset_path = dataset_path
    cfg.task.dataset.dataset_path = dataset_path
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    return dataset.get_validation_dataset()


def pick_donors(episodes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """For each sample, a random other sample in the batch from a different episode."""
    n = len(episodes)
    donors = np.empty(n, dtype=np.int64)
    for i in range(n):
        cand = np.flatnonzero(episodes != episodes[i])
        if len(cand) == 0:  # whole batch from one episode: fall back to another sample
            cand = np.array([j for j in range(n) if j != i] or [i])
        donors[i] = rng.choice(cand)
    return donors


def swap_key(obs: Dict[str, torch.Tensor], key: str, donors: torch.Tensor):
    out = dict(obs)
    out[key] = obs[key][donors]
    return out


def zero_key(obs: Dict[str, torch.Tensor], key: str):
    out = dict(obs)
    out[key] = torch.zeros_like(obs[key])
    return out


@torch.no_grad()
def per_sample_denoise_loss(policy, obs, nact, noise, noise2, t):
    """Training objective (sparse part) with externally fixed noise / timesteps."""
    nobs = policy.sparse_normalizer.normalize(obs)
    cond = policy.obs_encoder(nobs)
    noisy = policy.noise_scheduler.add_noise(nact, noise + policy.input_pertub * noise2, t)
    pred = policy.model_sparse(noisy, t, local_cond=None, global_cond=cond)
    pred_type = policy.noise_scheduler.config.prediction_type
    target = noise if pred_type == "epsilon" else nact
    return ((pred - target) ** 2).mean(dim=tuple(range(1, pred.dim())))


@torch.no_grad()
def sample_actions(policy, obs, seed: int):
    torch.manual_seed(seed)
    return policy.predict_action({"sparse": obs})["sparse"]


def paired_stats(base: np.ndarray, other: np.ndarray) -> dict:
    d = other - base
    return {
        "mean": float(other.mean()),
        "delta": float(d.mean()),
        "delta_se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
        "delta_rel_pct": float(100.0 * d.mean() / base.mean()) if base.mean() != 0 else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True, help="acp_replay_buffer.zarr used for training")
    ap.add_argument("--max-samples", type=int, default=2048, help="validation samples to evaluate")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--loss-draws", type=int, default=4, help="noise/timestep draws per sample")
    ap.add_argument("--sample-batches", type=int, default=8, help="batches with DDIM sampling (slow)")
    ap.add_argument("--also-shuffle", nargs="*", default=[], help="reference obs keys, e.g. rgb_0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    t0 = time.time()
    policy, cfg = load_policy_and_cfg(args.ckpt, device)
    sparse_keys = list(cfg.task.shape_meta["sample"]["obs"]["sparse"].keys())
    assert SCAN_KEY in sparse_keys, f"{SCAN_KEY} is not a policy input of this checkpoint"
    for k in args.also_shuffle:
        assert k in sparse_keys, f"--also-shuffle {k}: not a policy input ({sparse_keys})"
    print(f"[eval] policy loaded in {time.time() - t0:.0f}s")

    t0 = time.time()
    val = build_val_dataset(cfg, args.dataset)
    indices = val.sampler.indices  # list of (episode_id, rgb_id)
    episodes_all = np.array([int(e) for e, _ in indices])
    print(
        f"[eval] val dataset: {len(val)} samples from {len(np.unique(episodes_all))} episodes "
        f"(built in {time.time() - t0:.0f}s)"
    )

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(val))[: min(args.max_samples, len(val))]
    loader = DataLoader(
        Subset(val, order.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    loss_conds = ["real", "scan_shuffled", "scan_zero"] + [f"shuffle:{k}" for k in args.also_shuffle]
    sample_conds = loss_conds + ["real_reseeded"]
    losses: Dict[str, List[np.ndarray]] = {c: [] for c in loss_conds}
    scan_differs: List[np.ndarray] = []
    nmse: Dict[str, List[np.ndarray]] = {c: [] for c in sample_conds}
    pos_err_mm: Dict[str, List[np.ndarray]] = {c: [] for c in sample_conds}
    pos_shift_mm: Dict[str, List[np.ndarray]] = {c: [] for c in sample_conds if c != "real"}
    normalizer = policy.sparse_normalizer

    start = 0
    for b, batch in enumerate(loader):
        tb = time.time()
        obs = {k: v.to(device) for k, v in batch["obs"]["sparse"].items()}
        act = batch["action"]["sparse"].to(device)
        B = act.shape[0]
        eps = episodes_all[order[start : start + B]]
        start += B
        donors = torch.from_numpy(pick_donors(eps, rng)).to(device)

        scan = obs[SCAN_KEY]
        differs = (scan.flatten(1) != scan[donors].flatten(1)).any(dim=1)
        scan_differs.append(differs.cpu().numpy())

        cond_obs = {
            "real": obs,
            "scan_shuffled": swap_key(obs, SCAN_KEY, donors),
            "scan_zero": zero_key(obs, SCAN_KEY),
        }
        for k in args.also_shuffle:
            cond_obs[f"shuffle:{k}"] = swap_key(obs, k, donors)

        # 1. denoising loss with shared noise / timesteps
        nact = normalizer["action"].normalize(act)
        acc = {c: torch.zeros(B, device=device) for c in loss_conds}
        for d in range(args.loss_draws):
            g = torch.Generator(device=device).manual_seed(args.seed * 100003 + b * 101 + d)
            noise = torch.randn(nact.shape, device=device, generator=g)
            noise2 = torch.randn(nact.shape, device=device, generator=g)
            t = torch.randint(
                0, policy.noise_scheduler.config.num_train_timesteps, (B,), device=device, generator=g
            ).long()
            for c in loss_conds:
                acc[c] += per_sample_denoise_loss(policy, cond_obs[c], nact, noise, noise2, t)
        for c in loss_conds:
            losses[c].append((acc[c] / args.loss_draws).cpu().numpy())

        # 2. sampled actions from shared initial noise
        if b < args.sample_batches:
            seed = 1_000_000 + args.seed * 1000 + b
            preds = {c: sample_actions(policy, cond_obs[c], seed) for c in loss_conds}
            preds["real_reseeded"] = sample_actions(policy, obs, seed + 7919)
            gt_n = normalizer["action"].normalize(act)
            for c, p in preds.items():
                p_n = normalizer["action"].normalize(p)
                nmse[c].append(((p_n - gt_n) ** 2).mean(dim=(1, 2)).cpu().numpy())
                pos_err_mm[c].append(
                    (1000.0 * (p[..., :3] - act[..., :3]).norm(dim=-1).mean(dim=1)).cpu().numpy()
                )
                if c != "real":
                    pos_shift_mm[c].append(
                        (1000.0 * (p[..., :3] - preds["real"][..., :3]).norm(dim=-1).mean(dim=1))
                        .cpu()
                        .numpy()
                    )
        print(f"[eval] batch {b + 1}/{len(loader)} ({time.time() - tb:.1f}s)")

    L = {c: np.concatenate(v) for c, v in losses.items()}
    differs = np.concatenate(scan_differs)
    report = {
        "ckpt": args.ckpt,
        "dataset": args.dataset,
        "n_samples": int(len(L["real"])),
        "n_val_episodes": int(len(np.unique(episodes_all))),
        "frac_donor_scan_differs": float(differs.mean()),
        "loss_draws": args.loss_draws,
        "denoise_loss": {"real_mean": float(L["real"].mean())},
        "denoise_loss_scan_differs_only": {},
        "sampled": {},
    }
    for c in loss_conds[1:]:
        report["denoise_loss"][c] = paired_stats(L["real"], L[c])
        if differs.any():
            report["denoise_loss_scan_differs_only"][c] = paired_stats(L["real"][differs], L[c][differs])
    if nmse["real"]:
        for c in sample_conds:
            entry = {
                "naction_mse": float(np.concatenate(nmse[c]).mean()),
                "pos_err_mm": float(np.concatenate(pos_err_mm[c]).mean()),
            }
            if c != "real":
                entry["pos_shift_vs_real_mm"] = float(np.concatenate(pos_shift_mm[c]).mean())
            report["sampled"][c] = entry
        report["sampled"]["n_samples"] = int(len(np.concatenate(nmse["real"])))

    # ------------------------------------------------------------------ print
    print("\n=== scan shuffle ablation ===")
    print(f"ckpt: {args.ckpt}")
    print(
        f"{report['n_samples']} val samples, {report['n_val_episodes']} val episodes; "
        f"donor scan differs from own scan for {100 * report['frac_donor_scan_differs']:.1f}%"
    )
    print("\nDenoising loss (training val_loss objective, shared noise/timesteps):")
    print(f"  {'real':<22s} {L['real'].mean():.5f}")
    for c in loss_conds[1:]:
        s = report["denoise_loss"][c]
        print(
            f"  {c:<22s} {s['mean']:.5f}   delta {s['delta']:+.5f} ± {s['delta_se']:.5f} (SE)"
            f"   {s['delta_rel_pct']:+.2f}%"
        )
    if report["sampled"]:
        print(f"\nSampled actions ({report['sampled']['n_samples']} samples, DDIM, shared initial noise):")
        print(f"  {'condition':<22s} {'naction_mse':>12s} {'pos_err_mm':>11s} {'shift_vs_real_mm':>17s}")
        for c in sample_conds:
            e = report["sampled"][c]
            shift = f"{e['pos_shift_vs_real_mm']:.2f}" if "pos_shift_vs_real_mm" in e else "-"
            print(f"  {c:<22s} {e['naction_mse']:12.5f} {e['pos_err_mm']:11.2f} {shift:>17s}")
        print("  (real_reseeded = real obs, different initial noise: the sampling-noise floor)")

    out = pathlib.Path(args.out) if args.out else pathlib.Path(args.ckpt).with_name(
        pathlib.Path(args.ckpt).stem + "_scan_shuffle_eval.json"
    )
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
