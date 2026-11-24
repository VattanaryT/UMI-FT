#!/usr/bin/env python3
"""New 10-d normalizer: δn from field_v2n, rot6d from field_v3r.

v1, v2, and stiffness scale/offset are copied from field_v2n. They match
field_v3r. Does not open either source for writing.
"""

from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "PyriteML"), str(ROOT)]

from diffusion_policy.model.common.normalizer import (  # noqa: E402
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)

V2N = Path(
    "/home/boeing/umi_ft_ws/worktrees/UMI-FT-field_v2n/UMIFT_Data/"
    "surface_frames_scripts/assets/oct_v35_h4s_r4vt_action_normalizer_minmax_field_v2n.pkl"
)
V3R = Path(
    "/home/boeing/umi_ft_ws/worktrees/UMI-FT-field_v3r/UMIFT_Data/"
    "surface_frames_scripts/assets/oct_v35_h4s_r4vt_action_normalizer_minmax_field_v3r.pkl"
)
OUT = Path(__file__).resolve().parent / (
    "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_field_v2n_v3r_liftos.pkl"
)
V2N_SHA = "4d7c4febbafb93293cc2c470960ddeec6dc02731cebd90ea4ee0b11785f73ac2"
V3R_SHA = "55a99fbffed1c47e018a4638e044208021bf833b92c10f98d22c957e5eae9f30"
PROTECTED = {
    V2N_SHA,
    V3R_SHA,
    "14308b72d55600b4359dfa67d10aedab378e42ad26f2bbcac988eed4d97a1ac2",
    "71bd4cdc7be5a066e786d0b1fb9be526e01d6a6ff1e4ad7dd9c6280ac3709f84",
    "557e4d25a584ea026c253c95134f983dec0d5624dd1dd90652b78cd679630215",
    "e93b87561e762c2d7f1549a883eb18d45e6d50d929df463c43fce4817444e2f5",
    "3f9b16b1b06e4e2c27cba06053da3152c5f12322b35cd5cb94ff744f478d1c41",
    "743c83c80bcb7d5a85bb3441d80e4e4adc8b3010070f84378de40ba3c828040f",
}
DN = 2
ROT = slice(4, 10)
OTHER = (0, 1, 3)


def _vec(action, key):
    return action[key].detach().float().cpu().numpy().reshape(-1).copy()


def _stats(action, key):
    return action["input_stats"][key].detach().float().cpu().numpy().reshape(-1).copy()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    for path, sha in ((V2N, V2N_SHA), (V3R, V3R_SHA)):
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != sha:
            raise SystemExit(f"{path} sha {got} != {sha}")
    before = {path: path.stat().st_mtime_ns for path in (V2N, V3R)}
    v2n = pickle.load(V2N.open("rb")).params_dict["action"]
    v3r = pickle.load(V3R.open("rb")).params_dict["action"]
    scale = _vec(v2n, "scale")
    offset = _vec(v2n, "offset")
    scale_v3 = _vec(v3r, "scale")
    offset_v3 = _vec(v3r, "offset")
    if scale.shape != (10,) or offset.shape != (10,):
        raise SystemExit(f"expected 10-d, got {scale.shape}")
    for i in OTHER:
        if scale[i] != scale_v3[i] or offset[i] != offset_v3[i]:
            raise SystemExit(f"dim {i} scale/offset differ between sources")
    scale[ROT] = scale_v3[ROT]
    offset[ROT] = offset_v3[ROT]
    stats = {}
    for key in ("min", "max", "mean", "std"):
        values = _stats(v2n, key)
        values[ROT] = _stats(v3r, key)[ROT]
        stats[key] = values
    if not np.array_equal(scale[DN], _vec(v2n, "scale")[DN]):
        raise SystemExit("δn scale was not kept")
    if not np.array_equal(scale[ROT], scale_v3[ROT]):
        raise SystemExit("rot6d scale was not copied")
    lin = LinearNormalizer()
    lin["action"] = SingleFieldLinearNormalizer.create_manual(scale, offset, stats)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("wb") as handle:
        pickle.dump(lin, handle)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    if digest in PROTECTED:
        OUT.unlink()
        raise SystemExit(f"new sha collided with a protected normalizer {digest}")
    for path, mtime in before.items():
        if path.stat().st_mtime_ns != mtime:
            raise SystemExit(f"source mtime changed: {path}")
    print(digest)


if __name__ == "__main__":
    main()
