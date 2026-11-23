#!/usr/bin/env python3
"""Per kept C query: field heading versus demo travel, and carry restarts.

A chunk restarts the long-axis carry when its first sample is within 20° of
the long axis, so e_long has no previous good sample and uses the horizontal
fallback. Does not change labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "viewer"))

import contact_wall_v2 as v2  # noqa: E402
import heldout  # noqa: E402
import input_check  # noqa: E402
from heading_tilt import _tangent, geodesic_deg, long_axis, rotation_from_tilt  # noqa: E402

EXPORT = HERE / "export" / "causal_e25" / "episode_0.npz"


def _angle(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return float("nan")
    return round(float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))), 1)


def main() -> None:
    path = HERE / "C_first" / "replay.zarr"
    demo = heldout._demo(path, None)
    labels = input_check._load(path)
    demo["dR"] = labels["dR"]
    demo["tilt"] = labels["tilt"]
    field = v2.FixBMoldField(str(heldout.WALL))
    axis = long_axis(field.vertices)
    idx = np.load(EXPORT)["action_robot_idx"]
    rows = [(0, 0, index) for index in idx]
    _kept, detail, counts = v2.cw._select(demo, rows)
    kept_t = json.loads((HERE / "contact_wall_v2" / "kept.json").read_text())["0"]
    got = [item["t"] for item in detail if item["keep"]]
    want = [t for t, keep in zip(kept_t["t"], kept_t["keep"]) if keep]
    if got != want:
        raise SystemExit(f"export keep times {got} != kept.json {want}")
    rows_out = []
    for item in detail:
        index = item["index"]
        i0 = int(index[0])
        n_out = -demo["frame"][index][:, :, 2]
        direction = heldout._e_long(n_out, axis)
        _dropped, length = _tangent(axis, n_out[0])
        axis_deg = float(np.degrees(np.arcsin(np.clip(np.reshape(length, ()), 0.0, 1.0))))
        travel = demo["tool_p"][index[min(5, len(index) - 1)]] - demo["tool_p"][i0]
        got_h = rotation_from_tilt(n_out, direction, demo["tilt"][index])
        err = geodesic_deg(got_h, demo["tool_R"][index])
        restart = axis_deg < 20.0
        rows_out.append({
            "t": item["t"],
            "keep": item["keep"],
            "face": item["face"],
            "restart": restart,
            "axis_deg": round(axis_deg, 1),
            "heading_err_deg": round(float(err[0]), 1),
            "e_long_vs_travel_deg": _angle(direction[0], travel),
            "t1_vs_travel_deg": _angle(demo["frame"][i0][:, 0], travel),
            "travel_mm": round(float(np.linalg.norm(travel) * 1e3), 1),
        })
    kept = [row for row in rows_out if row["keep"]]
    restarts = [row for row in rows_out if row["restart"]]
    report = {
        "counts": counts,
        "restart_of_17": len(restarts),
        "kept_that_restart": sum(row["restart"] for row in kept),
        "queries": rows_out,
    }
    path = HERE / "contact_wall_v2" / "heading_c.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"C queries {counts['n']} kept {counts['kept']}  restarts {len(restarts)} of 17  "
          f"kept restarts {report['kept_that_restart']} of {len(kept)}", flush=True)
    print(f"{'t':7} {'keep':5} {'restart':8} {'axis':6} {'head':6} {'e_long':7} {'t1':7} travel_mm", flush=True)
    for row in rows_out:
        print(f"{row['t']:7.3f} {str(row['keep']):5} {str(row['restart']):8} {row['axis_deg']:6.1f} "
              f"{row['heading_err_deg']:6.1f} {row['e_long_vs_travel_deg']:7.1f} "
              f"{row['t1_vs_travel_deg']:7.1f} {row['travel_mm']:6.1f}", flush=True)


if __name__ == "__main__":
    main()
