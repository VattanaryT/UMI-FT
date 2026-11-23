#!/usr/bin/env python3
"""Copy the carry zarr with stiffness_0 = 3000. Does not touch the carry files."""

from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS)]

from fixb_build_datasets import link_tree, replace_array

DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
SRC = DATA / "r4vt_causal_foot_carry" / "acp_replay_buffer.zarr"
OUT = DATA / "r4vt_causal_foot_carry_k3000"
OLD_NORM = SCRIPTS / "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_causal_foot_carry.pkl"
NORM = SCRIPTS / "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_causal_foot_carry_k3000.pkl"
K = np.float32(3000.0)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pin_normalizer() -> None:
    if NORM.exists():
        raise SystemExit(f"refusing to overwrite {NORM}")
    lin = pickle.load(OLD_NORM.open("rb"))
    action = lin["action"].params_dict
    scale = action["scale"].detach().clone()
    offset = action["offset"].detach().clone()
    # Same span floor as build_carry.fit: a constant dim maps to -1.
    span = np.float32(1e-7)
    scale_k = np.float32(2.0) / span
    scale[3] = torch.tensor(scale_k, dtype=scale.dtype)
    offset[3] = torch.tensor(np.float32(-1.0) - scale_k * K, dtype=offset.dtype)
    action["scale"] = torch.nn.Parameter(scale)
    action["offset"] = torch.nn.Parameter(offset)
    for name in ("min", "max", "mean"):
        values = action["input_stats"][name].detach().clone()
        values[3] = torch.tensor(K, dtype=values.dtype)
        action["input_stats"][name] = torch.nn.Parameter(values)
    std = action["input_stats"]["std"].detach().clone()
    std[3] = 0
    action["input_stats"]["std"] = torch.nn.Parameter(std)
    with NORM.open("wb") as handle:
        pickle.dump(lin, handle)


def _check(src_before: np.ndarray) -> None:
    got = np.asarray(
        zarr.open(str(OUT / "acp_replay_buffer.zarr"), "r")["data"]["episode_0"]["stiffness_0"]
    )
    src = np.asarray(
        zarr.open(str(SRC), "r")["data"]["episode_0"]["stiffness_0"]
    )
    if not np.all(got == 3000) or np.all(src == 3000) or not np.array_equal(src, src_before):
        raise RuntimeError("stiffness pin touched the source or missed the copy")
    lin = pickle.load(NORM.open("rb"))
    scale = lin["action"].params_dict["scale"].detach().numpy()
    value = float(scale[3] * 3000 + lin["action"].params_dict["offset"].detach().numpy()[3])
    # float32 cannot hold -1 - 2e7*3000, so 3000 lands on 0. Unnormalizing 0 is 3000.
    if abs(scale[3] - 2e7) > 1 or abs(value) > 1e-4:
        raise RuntimeError(f"constant stiffness did not map to 0 ({scale[3]}, {value})")


def main() -> None:
    out_zarr = OUT / "acp_replay_buffer.zarr"
    if NORM.exists():
        raise SystemExit(f"refusing to overwrite {NORM}")
    src_before = np.asarray(zarr.open(str(SRC), "r")["data"]["episode_0"]["stiffness_0"])
    if not out_zarr.exists():
        link_tree(SRC, out_zarr)
        root = zarr.open(str(out_zarr), "a")
        root.attrs["label_version"] = "causal_foot_carry_k3000_v1"
        root.attrs["stiffness"] = "3000 on every row"
        for ep_id in range(50):
            ep = root["data"][f"episode_{ep_id}"]
            stored = ep["stiffness_0"]
            replace_array(ep, "stiffness_0", np.full(stored.shape, K), stored)
            ep.attrs["label_version"] = "causal_foot_carry_k3000_v1"
    _pin_normalizer()
    _check(src_before)
    print(_sha(NORM), flush=True)


if __name__ == "__main__":
    main()
