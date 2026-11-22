"""Current checkout against tests/golden/*.npz. Explained keys may differ."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "golden"))
os.environ["UMIFT_ROOT"] = str(HERE.parents[1])

from record_golden import capture  # noqa: E402

GOLDEN = HERE / "golden"
EXPLAINED = GOLDEN / "explained.json"
ATOL = 1e-5


def _explained():
    if not EXPLAINED.is_file():
        return {}
    return json.loads(EXPLAINED.read_text())


def _diff(got, ref):
    if got.shape != ref.shape:
        return f"shape {got.shape} != {ref.shape}"
    if got.dtype.kind in "SU":
        if str(got) != str(ref):
            return f"{got} != {ref}"
        return None
    delta = np.nanmax(np.abs(np.asarray(got, dtype=np.float64) - np.asarray(ref, dtype=np.float64)))
    if not np.isfinite(delta) or delta > ATOL:
        return f"max abs {delta}"
    return None


def test_golden_outputs():
    saved = sorted(GOLDEN.glob("*.npz"))
    assert saved, "no golden npz; run record_golden.py first"
    current = capture()
    reasons = _explained()
    problems = []
    for path in saved:
        ref = np.load(path, allow_pickle=False)
        branch = path.stem
        allowed = reasons.get(branch, {})
        if allowed.get("_skip"):
            continue
        for key in ref.files:
            if key not in current:
                reason = allowed.get(key)
                if reason:
                    continue
                problems.append(f"{branch}:{key} missing on this checkout")
                continue
            reason = allowed.get(key)
            delta = _diff(current[key], ref[key])
            if delta and not reason:
                problems.append(f"{branch}:{key} {delta}")
    assert not problems, "\n".join(problems)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
