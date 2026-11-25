"""Counterfactual hist stiffness on the pe_guard log, horizons 0-6.

Reads the logged nominal and virtual targets. Does not start the runner or
the robot. The press gain is the logged floored scalar (h0 has no overlap;
later horizons use that horizon's own scalar, not the decay-weighted mean).
"""

from __future__ import annotations

import numpy as np
import zarr

from mold_aware_runtime import rebuild_hist_stiffness

LOG = "/mnt/data/umidata/ft/vtevy/rollouts/2026-11-24_liftos_hw_pe_guard"


def _eigs(packed, step):
    block = np.asarray(packed)[:, 6 * step : 6 * step + 6]
    return np.sort(np.linalg.eigvalsh(0.5 * (block[:3, :3] + block[:3, :3].T)))


def main() -> None:
    group = zarr.open_group(LOG, "r")
    print(
        "pe_guard decay",
        group.attrs["ensemble_decay_m"],
        "exec",
        group.attrs["sparse_execution_horizon"],
        "wrench_hold",
        group.attrs["wrench_hold_n"],
    )
    print(
        "h step  logged_eigs              gap    rebuilt_eigs            v2     dn_mm"
    )
    worst_logged = 0.0
    worst_rebuilt = 0.0
    for h in range(7):
        action = group[f"horizon_{h}"]["action"]
        logged = np.asarray(action["ts_stiffnesses_0"])
        nominal = np.asarray(action["ts_nominal_targets_0"])
        virtual = np.asarray(action["ts_virtual_targets_0"])
        floored = np.asarray(action["stiffness_floored_0"])
        rebuilt = rebuild_hist_stiffness(nominal, virtual, floored)
        v2 = np.asarray(action["v_par"])[:, 1]
        delta_n = np.asarray(action["delta_n"])
        for step in range(logged.shape[1] // 6):
            logged_eigs = _eigs(logged, step)
            rebuilt_eigs = _eigs(rebuilt, step)
            logged_gap = float(logged_eigs[2] - logged_eigs[1])
            rebuilt_gap = float(rebuilt_eigs[2] - rebuilt_eigs[1])
            worst_logged = max(worst_logged, logged_gap)
            worst_rebuilt = max(worst_rebuilt, rebuilt_gap)
            if step == 0:
                print(
                    f"{h} {step:4d}  {np.array2string(logged_eigs, precision=0, floatmode='fixed'):22s}  "
                    f"{logged_gap:6.0f}  "
                    f"{np.array2string(rebuilt_eigs, precision=0, floatmode='fixed'):22s}  "
                    f"{v2[step]:+.3f}  {delta_n[step] * 1e3:+.1f}"
                )
            assert rebuilt_gap < 1e-4
            assert np.allclose(rebuilt_eigs, 3000.0)
            assert abs(floored[step] - 3000.0) < 1e-3
    print(f"stiff-gap max logged {worst_logged:.1f} N/m  rebuilt {worst_rebuilt:.3f} N/m")
    if worst_rebuilt > 1e-4:
        raise SystemExit("rebuilt axis is not pure")
    print("HIST STIFFNESS CHECK PASS")


if __name__ == "__main__":
    main()
