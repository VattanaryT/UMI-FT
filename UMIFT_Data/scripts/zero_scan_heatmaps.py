#!/usr/bin/env python3
"""Zero scan_heatmap_* arrays in a UMIFT replay buffer.

Leaves map_to_scan_idx_* and scan_gap_intervals_* untouched so the sampler
still indexes the key and still drops operator scan-pause windows.

  python UMIFT_Data/scripts/zero_scan_heatmaps.py \\
    --input  /mnt/data/umidata/ft/datasets/stringer_scanner/t265_acp_replay_buffer.zarr \\
    --output /mnt/data/umidata/ft/datasets/stringer_scanner/t265_acp_replay_buffer_zero_scan.zarr

  python UMIFT_Data/scripts/zero_scan_heatmaps.py --in-place path/to/replay_buffer.zarr
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import zarr

_HEATMAP_NAME = re.compile(r"^scan_heatmap_\d+$")


def zero_scan_heatmaps(root: zarr.Group) -> int:
    n_zeroed = 0
    data = root["data"]
    for ep_name in data.group_keys():
        ep = data[ep_name]
        for arr_name, arr in list(ep.arrays()):
            if _HEATMAP_NAME.match(arr_name):
                arr[:] = np.zeros(arr.shape, dtype=arr.dtype)
                n_zeroed += 1
    return n_zeroed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source .zarr path")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path, help="Copy here, then zero the copy")
    group.add_argument(
        "--in-place",
        action="store_true",
        help="Zero the input store (destructive)",
    )
    args = parser.parse_args()

    src = args.input.expanduser().resolve()
    if not src.exists():
        print(f"Error: {src} does not exist", file=sys.stderr)
        return 1

    if args.in_place:
        target = src
    else:
        target = args.output.expanduser().resolve()
        if target.exists():
            print(f"Error: {target} already exists", file=sys.stderr)
            return 1
        print(f"Copying {src} -> {target} (full store, may take a while) ...")
        shutil.copytree(src, target)

    root = zarr.open(str(target), mode="r+")
    n_zeroed = zero_scan_heatmaps(root)
    n_eps = len(list(root["data"].group_keys()))
    print(f"Zeroed {n_zeroed} scan_heatmap array(s) across {n_eps} episode(s) in {target}")
    if n_zeroed == 0:
        print("Warning: no scan_heatmap_* arrays found", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
