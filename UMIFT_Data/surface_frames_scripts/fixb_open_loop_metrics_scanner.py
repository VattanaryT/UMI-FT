#!/usr/bin/env python3
"""Open-loop metrics for the September Fix B datasets.

Decode is the same as fixb_open_loop_metrics.py: r2/r3 integrate (v∥, δn)
from the measured pose at each chunk start; heading R(x_s) B. This file only
changes the default dataset_root. Do not edit fixb_open_loop_metrics.py.

  python fixb_open_loop_metrics_scanner.py \
    --exports_root /mnt/data/umidata/ft/training_outputs/scanner_fixb/eval/exports
"""

from __future__ import annotations

import sys
from pathlib import Path

import fixb_open_loop_metrics as _m

_DEFAULT = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb")


def main() -> int:
    if "--dataset_root" not in sys.argv:
        sys.argv[1:1] = ["--dataset_root", str(_DEFAULT)]
    return _m.main()


if __name__ == "__main__":
    raise SystemExit(main())
