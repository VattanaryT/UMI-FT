#!/usr/bin/env python3
"""Merge UMIFT episode zarr stores into one contiguous acp_replay_buffer.

Unlike ``zarr_utils.merge_zarr_datasets``, this:

1. Sorts source episodes **numerically** (``episode_2`` before ``episode_10``).
2. **Rebuilds** cumulative ``meta/episode_*_len`` from copied array lengths
   instead of concatenating already-cumulative arrays (which breaks offsets).
3. Preserves each array's compressor / chunks / filters.

Example:
  python UMIFT_Data/scripts/merge_umift_zarr_datasets.py \\
    --inputs \\
      /mnt/data/umidata/ft/stringer_data_v2_full/acp_replay_buffer_10hz_224.zarr \\
      /mnt/data/umidata/ft/teleop_umift_08_03_right/acp_replay_buffer.zarr \\
    --output /mnt/data/umidata/ft/stringer_teleop_merged_10hz_224/acp_replay_buffer.zarr
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import zarr
from numcodecs import Blosc

# Map meta array name -> episode dataset key used for length.
META_LEN_KEYS: Dict[str, str] = {
    "episode_rgb0_len": "rgb_0",
    "episode_ultrawide0_len": "ultrawide_0",
    "episode_depth0_len": "depth_0",
    "episode_robot0_len": "ts_pose_fb_0",
    "episode_gripper0_len": "gripper_0",
    "episode_wrench0_len": "wrench_left_0",
}

REQUIRED_EPISODE_KEYS = tuple(META_LEN_KEYS.values())


def _episode_index(name: str) -> int:
    if not name.startswith("episode_"):
        raise ValueError(f"Unexpected episode group name: {name}")
    return int(name.split("_", 1)[1])


def list_episodes_numeric(data_group: zarr.Group) -> List[str]:
    names = [n for n in data_group.group_keys() if n.startswith("episode_")]
    return sorted(names, key=_episode_index)


def copy_array(src: zarr.Array, dst_group: zarr.Group, name: str) -> zarr.Array:
    """Copy one array, preserving dtype / chunks / compressor / filters / attrs."""
    dst = dst_group.create_dataset(
        name=name,
        shape=src.shape,
        chunks=src.chunks,
        dtype=src.dtype,
        compressor=src.compressor,
        filters=src.filters,
        overwrite=True,
    )
    dst[:] = src[:]
    if src.attrs:
        dst.attrs.update(dict(src.attrs))
    return dst


def copy_episode_group(src_group: zarr.Group, dst_group: zarr.Group) -> None:
    for arr_name, src_arr in src_group.arrays():
        copy_array(src_arr, dst_group, arr_name)
    for sub_name, src_sub in src_group.groups():
        dst_sub = dst_group.require_group(sub_name)
        copy_episode_group(src_sub, dst_sub)
        if src_sub.attrs:
            dst_sub.attrs.update(dict(src_sub.attrs))


def _append_cumulative(cumulative: List[int], this_len: int) -> None:
    cumulative.append((cumulative[-1] if cumulative else 0) + int(this_len))


def validate_episode_keys(ep_group: zarr.Group, ep_name: str, source: str) -> None:
    missing = [k for k in REQUIRED_EPISODE_KEYS if k not in ep_group]
    if missing:
        raise KeyError(
            f"{source}/{ep_name} missing required keys: {missing}. "
            "Both inputs must be full UMIFT episode zarrs."
        )


def episode_lengths(ep_group: zarr.Group) -> Dict[str, int]:
    return {meta: int(ep_group[key].shape[0]) for meta, key in META_LEN_KEYS.items()}


def merge_umift_zarr_datasets(
    input_zarr_paths: Sequence[str | Path],
    output_zarr_path: str | Path,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
    drop_episodes: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """Merge UMIFT zarr stores; return a small summary dict.

    ``drop_episodes`` removes those **source** episode indices (per input, before
    renumbering) from the output. Useful for surgical edits of one store.
    """
    inputs = [Path(p).resolve() for p in input_zarr_paths]
    output = Path(output_zarr_path).resolve()
    drop_set = {int(i) for i in (drop_episodes or [])}

    if len(inputs) < 1:
        raise ValueError("Need at least one --inputs path")
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(p)

    # Preflight: count episodes and check schema.
    plan: List[Tuple[Path, List[str]]] = []
    found_drops: set[int] = set()
    for src_path in inputs:
        root = zarr.open(str(src_path), mode="r")
        if "data" not in root:
            raise KeyError(f"{src_path} has no /data group")
        eps = list_episodes_numeric(root["data"])
        if not eps:
            raise RuntimeError(f"{src_path} has no episode_* groups under /data")
        validate_episode_keys(root["data"][eps[0]], eps[0], str(src_path))
        validate_episode_keys(root["data"][eps[-1]], eps[-1], str(src_path))
        keep = [e for e in eps if _episode_index(e) not in drop_set]
        drop = [e for e in eps if _episode_index(e) in drop_set]
        found_drops.update(_episode_index(e) for e in drop)
        plan.append((src_path, keep))
        print(
            f"[merge] plan {src_path}: keep {len(keep)}/{len(eps)} "
            f"({keep[0] if keep else 'none'} .. {keep[-1] if keep else 'none'})"
        )
        if drop:
            print(f"[merge]   drop: {', '.join(drop)}")

    if drop_set:
        missing = sorted(drop_set - found_drops)
        if missing:
            raise KeyError(f"--drop-episodes not found in any input: {missing}")

    total = sum(len(eps) for _, eps in plan)
    print(f"[merge] total episodes -> {total} at {output}")
    if dry_run:
        return {
            "output": str(output),
            "n_episodes": total,
            "dropped": sorted(drop_set),
            "dry_run": True,
        }

    # Write to a temp dir when overwriting an input in-place.
    output.parent.mkdir(parents=True, exist_ok=True)
    write_output = output
    tmp_output: Optional[Path] = None
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output exists: {output}. Pass --overwrite to replace it."
            )
        if output.resolve() in {p.resolve() for p in inputs}:
            tmp_output = output.parent / f".{output.name}.tmp_merge"
            if tmp_output.exists():
                shutil.rmtree(tmp_output)
            write_output = tmp_output
            print(f"[merge] In-place edit via temp: {tmp_output}")
        else:
            print(f"[merge] Removing existing output: {output}")
            shutil.rmtree(output)

    store = zarr.DirectoryStore(str(write_output))
    out_root = zarr.group(store=store, overwrite=True)
    out_data = out_root.create_group("data")
    out_meta = out_root.create_group("meta")

    meta_cum: Dict[str, List[int]] = {k: [] for k in META_LEN_KEYS}
    source_map: List[Dict[str, object]] = []
    out_idx = 0
    numeric_compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)

    for src_path, eps in plan:
        src_root = zarr.open(str(src_path), mode="r")
        src_data = src_root["data"]
        # Preserve non-episode root attrs from single-input rewrite.
        if len(inputs) == 1 and src_root.attrs:
            out_root.attrs.update(dict(src_root.attrs))
        for src_ep_name in eps:
            src_ep = src_data[src_ep_name]
            validate_episode_keys(src_ep, src_ep_name, str(src_path))
            lengths = episode_lengths(src_ep)

            dst_name = f"episode_{out_idx}"
            dst_ep = out_data.create_group(dst_name)
            copy_episode_group(src_ep, dst_ep)
            if src_ep.attrs:
                dst_ep.attrs.update(dict(src_ep.attrs))
            dst_ep.attrs["source_zarr"] = str(src_path)
            dst_ep.attrs["source_episode"] = src_ep_name

            for meta_key, this_len in lengths.items():
                _append_cumulative(meta_cum[meta_key], this_len)

            uniq = set(lengths.values())
            if len(uniq) != 1:
                print(
                    f"[merge] WARNING {dst_name} modality lengths differ: {lengths}"
                )

            source_map.append(
                {
                    "out_episode": dst_name,
                    "out_index": out_idx,
                    "source_zarr": str(src_path),
                    "source_episode": src_ep_name,
                    "length": lengths["episode_rgb0_len"],
                }
            )
            print(
                f"[merge] {dst_name} <- {src_path.name}/{src_ep_name} "
                f"T={lengths['episode_rgb0_len']}"
            )
            out_idx += 1

    for meta_key, cum in meta_cum.items():
        arr = np.asarray(cum, dtype=np.int64)
        out_meta.create_dataset(
            name=meta_key,
            data=arr,
            shape=arr.shape,
            chunks=(arr.shape[0],),
            dtype=arr.dtype,
            compressor=numeric_compressor,
        )
        print(f"[merge] meta/{meta_key} shape={arr.shape} last={int(arr[-1])}")

    out_root.attrs["merged_from"] = [str(p) for p in inputs]
    out_root.attrs["n_episodes"] = int(out_idx)
    out_root.attrs["merge_tool"] = "merge_umift_zarr_datasets.py"
    if drop_set:
        out_root.attrs["dropped_source_episodes"] = sorted(drop_set)

    map_path = output.parent / "episode_source_map.json"
    map_path.write_text(json.dumps(source_map, indent=2) + "\n")
    print(f"[merge] Wrote source map: {map_path}")

    if tmp_output is not None:
        backup = output.parent / f".{output.name}.bak_before_drop"
        if backup.exists():
            shutil.rmtree(backup)
        print(f"[merge] Replacing {output} (backup -> {backup.name})")
        output.rename(backup)
        tmp_output.rename(output)
        shutil.rmtree(backup)
        write_output = output

    print(f"[merge] Done: {out_idx} episodes -> {output}")
    return {
        "output": str(output),
        "n_episodes": out_idx,
        "source_map": str(map_path),
        "dropped": sorted(drop_set),
        "dry_run": False,
    }


def validate_merged_zarr(zarr_path: str | Path) -> None:
    """Raise if meta cumulative lengths disagree with episode array shapes."""
    root = zarr.open(str(zarr_path), mode="r")
    eps = list_episodes_numeric(root["data"])
    meta = root["meta"]
    for meta_key, arr_key in META_LEN_KEYS.items():
        if meta_key not in meta:
            raise KeyError(f"missing meta/{meta_key}")
        cum = np.asarray(meta[meta_key][:], dtype=np.int64)
        if cum.shape[0] != len(eps):
            raise RuntimeError(
                f"{meta_key} length {cum.shape[0]} != n_episodes {len(eps)}"
            )
        for i, ep_name in enumerate(eps):
            start = 0 if i == 0 else int(cum[i - 1])
            end = int(cum[i])
            expected = end - start
            actual = int(root["data"][ep_name][arr_key].shape[0])
            if expected != actual:
                raise RuntimeError(
                    f"{ep_name} {arr_key}: meta says {expected}, array has {actual}"
                )
    print(f"[validate] OK: {len(eps)} episodes, meta matches array lengths")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        nargs="+",
        default=[],
        help="Input UMIFT acp_replay_buffer.zarr paths (order preserved).",
    )
    p.add_argument("--output", required=True, help="Output zarr path.")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output path if it already exists.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the merge plan.",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing merged zarr (uses --output) and exit.",
    )
    p.add_argument(
        "--drop-episodes",
        nargs="+",
        type=int,
        default=[],
        help=(
            "Source episode indices to exclude (e.g. 20 44). "
            "Remaining episodes are renumbered contiguously; meta is rebuilt."
        ),
    )
    args = p.parse_args(argv)
    if not args.validate_only and not args.inputs:
        p.error("--inputs is required unless --validate-only")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.validate_only:
        validate_merged_zarr(args.output)
        return 0
    merge_umift_zarr_datasets(
        args.inputs,
        args.output,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        drop_episodes=args.drop_episodes,
    )
    if not args.dry_run:
        validate_merged_zarr(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
