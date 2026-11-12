#!/usr/bin/env python3
"""dt, ensemble-off, rotation-flip, and press-floor checks. Reads labels only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import replay_gt as rg

OUT = Path("/mnt/data/umidata/ft/replay_exports/gt_dt_ensemble")
DT_LABEL_S = 1.0 / 15.0
DT_RUNNER_S = 2.0 / 15.0
EPISODES = (0, 9)
FOOT_FAR_M = 0.020
LEFT_MESH_M = 0.005


def _basis_dots(ra: np.ndarray, rb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ra = np.asarray(ra, dtype=np.float64)
    rb = np.asarray(rb, dtype=np.float64)
    n_dot = np.sum(ra[..., :, 2] * rb[..., :, 2], axis=-1)
    t_dot = np.sum(ra[..., :, 0] * rb[..., :, 0], axis=-1)
    onto_t2 = np.abs(np.sum(ra[..., :, 0] * rb[..., :, 1], axis=-1))
    return n_dot, t_dot, onto_t2


def t1_flipped(ra: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """True when normals agree and t1 (column 0) has flipped sign."""
    n_dot, t_dot, _onto_t2 = _basis_dots(ra, rb)
    return (n_dot > 0.5) & (t_dot < 0.0)


def t1_swapped(ra: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """True when normals agree and t1 has jumped onto the other tangent."""
    n_dot, t_dot, onto_t2 = _basis_dots(ra, rb)
    return (n_dot > 0.5) & (np.abs(t_dot) < 0.5) & (onto_t2 > 0.9)


def _pct(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"n": 0, "p5": None, "p50": None, "p95": None}
    p5, p50, p95 = np.percentile(values, [5, 50, 95])
    return {
        "n": int(values.size),
        "p5": float(p5),
        "p50": float(p50),
        "p95": float(p95),
    }


def _medmax(values: np.ndarray, mask: np.ndarray | None = None) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool).reshape(-1)]
    if values.size == 0:
        return {"n": 0, "median": None, "max": None}
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def floor_breakdown(group) -> dict:
    """Label δn below -5 mm, split by force, plus raw-TCP depth along n."""
    delta_n = np.asarray(group["fixb_delta_n_0"], dtype=np.float64).reshape(-1)
    robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    force = rg._force_on_robot(group, robot_t)
    tcp = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)[:, :3]
    tcp_sd = rg._mold_field().query(tcp)["sd"]
    clamped = delta_n < rg.PRESS_FLOOR_M
    contact = force >= 8.0
    out = {
        "threshold_m": rg.PRESS_FLOOR_M,
        "contact_n": 8.0,
        "label_dt_s": _pct(np.asarray(group["fixb_dt_next_0"], dtype=np.float64)),
    }
    for name, mask in (
        ("clamped_contact", clamped & contact),
        ("clamped_free", clamped & ~contact),
        ("unclamped_contact", ~clamped & contact),
    ):
        out[name] = {
            "delta_n_m": _pct(delta_n[mask]),
            "tcp_along_n_m": _pct(tcp_sd[mask]),
        }
    return out


def r4_case(group, episode: int, exec_h: int, dt_s: float, scale, offset, runtime) -> dict:
    robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    demo = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    bad = np.asarray(group["bad_span_0"]).reshape(-1) > 0
    long_dt = rg._long_dt_mask(episode, len(robot_t))
    stored_foot = np.asarray(group["mold_x_s_0"], dtype=np.float64)
    stored_n = np.asarray(group["mold_normal_0"], dtype=np.float64)
    stored_dn = np.asarray(group["fixb_delta_n_0"], dtype=np.float64).reshape(-1)
    stored_r = np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
    n_err = float(np.max(np.linalg.norm(stored_r[:, :, 2] - stored_n, axis=1)))
    if n_err > 1e-3:
        raise RuntimeError(f"mold_R column 2 is not n ({n_err})")

    chunks = []
    foot_off, depth_off, rot = [], [], []
    drift_end = []
    exec_idx = []
    flip_stored, swap_stored, flip_step, foot_far, left_mesh, pre_abs = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    n_dot, field_ang = [], []
    field = rg._mold_field()

    for indices in rg._windows(robot_t, rgb_t, exec_h):
        action = rg.roundtrip_action(
            rg._action_rows(group, indices, with_delta_r=True), scale, offset
        )
        head = int(indices[0])
        decoded = runtime.decode_chunk(
            action,
            demo[head, :3],
            rg._pose_R(demo[head])[0],
            np.full(rg.ACTION_HORIZON, dt_s),
            apply_safety=False,
        )
        from mold_aware_runtime import Rp_to_SE3
        import PyriteUtility.spatial_math.spatial_utilities as su

        pose = su.SE3_to_pose7(Rp_to_SE3(decoded["R_d"], decoded["x_d"]))
        stiff = np.asarray(decoded["stiffness"], dtype=np.float64).reshape(-1)
        sl = slice(0, exec_h)
        used = indices[sl]
        foot_err = rg._inplane(decoded["x_s"][sl], stored_foot[used], stored_n[used])
        depth_err = np.abs(np.asarray(decoded["sd"][sl]) - stored_dn[used])
        foot_off.append(foot_err)
        depth_off.append(depth_err)
        drift_end.append(float(foot_err[-1]))
        rot.append(rg.geodesic_deg(decoded["R_d"][sl], rg._pose_R(demo[used])))
        exec_idx.append(used)
        chunks.append({"pose7": pose, "nominal_pose7": pose, "stiffness": stiff})

        r_dec = decoded["R_field"][sl]
        flip_stored.append(t1_flipped(r_dec, stored_r[used]))
        swap_stored.append(t1_swapped(r_dec, stored_r[used]))
        n_dot.append(np.sum(r_dec[:, :, 2] * stored_r[used, :, 2], axis=1))
        field_ang.append(rg.geodesic_deg(r_dec, stored_r[used]))
        step_flip = np.zeros(exec_h, dtype=bool)
        if exec_h > 1:
            step_flip[1:] = t1_flipped(r_dec[1:], r_dec[:-1])
        flip_step.append(step_flip)
        foot_far.append(foot_err > FOOT_FAR_M)

        pre = decoded["x_s"][:-1] + np.einsum(
            "nij,nj->ni", decoded["R_field"][:-1, :, :2], action[:-1, :2]
        ) * dt_s
        pre_sd = np.abs(field.query(pre)["sd"])
        left = np.zeros(exec_h, dtype=bool)
        arrived = np.zeros(exec_h, dtype=np.float64)
        left[1:] = pre_sd[: exec_h - 1] > LEFT_MESH_M
        arrived[1:] = pre_sd[: exec_h - 1]
        left_mesh.append(left)
        pre_abs.append(arrived)

    idx = np.concatenate(exec_idx)
    clean = ~(bad[idx] | long_dt[idx])
    blended = rg.blend_executed(chunks, exec_h)
    queried = field.query(blended["nominal_pose7"][:, :3])
    foot_on = rg._inplane(queried["x_s"], stored_foot[idx], stored_n[idx])
    depth_on = np.abs(queried["sd"] - stored_dn[idx])
    rot_all = np.concatenate(rot)
    big = rot_all > 90.0
    flip_s = np.concatenate(flip_stored)
    swap_s = np.concatenate(swap_stored)
    flip_k = np.concatenate(flip_step)
    far = np.concatenate(foot_far)
    left = np.concatenate(left_mesh)
    arrived = np.concatenate(pre_abs)
    n_agree = np.concatenate(n_dot)
    field_deg = np.concatenate(field_ang)
    frame_jump = flip_s | swap_s | (n_agree < 0.0)
    return {
        "dt_s": float(dt_s),
        "execution_horizon": int(exec_h),
        "n_executed": int(len(idx)),
        "n_clean": int(np.sum(clean)),
        "foot_m": {
            "ensemble_off": _medmax(np.concatenate(foot_off), clean),
            "ensemble_on": _medmax(foot_on, clean),
        },
        "drift_end_m": _medmax(np.asarray(drift_end)),
        "depth_m": {
            "ensemble_off": _medmax(np.concatenate(depth_off), clean),
            "ensemble_on": _medmax(depth_on, clean),
        },
        "rotation_gt_90": {
            "n": int(np.sum(big)),
            "n_executed": int(len(big)),
            "t1_flip_vs_stored": int(np.sum(big & flip_s)),
            "t1_swap_vs_stored": int(np.sum(big & swap_s)),
            "normal_flip_vs_stored": int(np.sum(big & (n_agree < 0.0))),
            "t1_flip_vs_previous_step": int(np.sum(big & flip_k)),
            "field_geodesic_deg_on_outliers": _medmax(field_deg[big]),
            "field_geodesic_deg_on_neither": _medmax(
                field_deg[big & ~frame_jump & ~far]
            ),
            "foot_inplane_gt_20mm": int(np.sum(big & far)),
            "euler_step_left_mesh_gt_5mm": int(np.sum(big & left)),
            "frame_jump_and_foot_far": int(np.sum(big & frame_jump & far)),
            "neither_frame_jump_nor_foot_far": int(np.sum(big & ~frame_jump & ~far)),
            "foot_m_on_outliers": _medmax(np.concatenate(foot_off)[big]),
            "foot_m_on_rest": _medmax(np.concatenate(foot_off)[~big]),
            "euler_abs_sd_m_on_outliers": _medmax(arrived[big]),
            "euler_abs_sd_m_on_rest": _medmax(arrived[~big]),
        },
    }


def analyze() -> dict:
    scale, offset = rg.load_action_scale(rg.R4VT_NORMALIZER, rg.R4VT_SHA)
    runtime = rg._r4_runtime()
    report = {
        "episodes": list(EPISODES),
        "val_episode_ids": rg.val_episode_ids(rg.R4VT_ZARR),
        "label_dt": {
            "file": "UMIFT_Data/surface_frames_scripts/fixb_labels_tracked.py",
            "lines": "79-90",
            "clock": "fixb_dt_next_0 = next RGB sample time minus this robot time",
            "next_control_times": "UMIFT_Data/surface_frames_scripts/fixb_labels.py:48-54",
            "constant_compared_s": DT_LABEL_S,
        },
        "runner_dt": {
            "file": "PyriteEnvSuites/env_runners/umift_env_runner.py",
            "lines": "1878-1884",
            "seconds": DT_RUNNER_S,
        },
        "r4vt": {},
        "r1rel": {},
    }
    for episode in EPISODES:
        group = rg._open_episode(rg.R4VT_ZARR, episode)
        block = {
            "split": rg.split_name(episode, rg.R4VT_ZARR),
            "floor": floor_breakdown(group),
            "cases": [],
        }
        for dt_s in (DT_LABEL_S, DT_RUNNER_S):
            for exec_h in rg.EXEC_HORIZONS:
                block["cases"].append(
                    r4_case(group, episode, exec_h, dt_s, scale, offset, runtime)
                )
        report["r4vt"][str(episode)] = block
        rel = rg.replay_episode("r1rel", episode)
        report["r1rel"][str(episode)] = {
            "split": rel["split"],
            "horizons": {
                key: {
                    "tcp_m": {
                        "ensemble_off": {
                            "median": row["position"]["tcp_m"]["decode_median"],
                            "max": row["position"]["tcp_m"]["decode_max"],
                        },
                        "ensemble_on": {
                            "median": row["position"]["tcp_m"]["ensemble_median"],
                            "max": row["position"]["tcp_m"]["ensemble_max"],
                        },
                    },
                    "orientation_deg": {
                        "ensemble_off": {
                            "median": row["orientation_deg"]["decode_median"],
                            "max": row["orientation_deg"]["decode_max"],
                        },
                        "ensemble_on": {
                            "median": row["orientation_deg"]["ensemble_median"],
                            "max": row["orientation_deg"]["ensemble_max"],
                        },
                    },
                }
                for key, row in rel["horizons"].items()
            },
        }
    return report


def _mm(value) -> str:
    if value is None:
        return "n/a"
    return f"{1e3 * value:.2f}"


def format_report(report: dict) -> str:
    lines = []
    for episode, block in report["r4vt"].items():
        lines.append(f"r4vt episode {episode} ({block['split']})")
        dt = block["floor"]["label_dt_s"]
        lines.append(
            "  label dt p5/50/95 ms: "
            f"{1e3 * dt['p5']:.1f}/{1e3 * dt['p50']:.1f}/{1e3 * dt['p95']:.1f}"
        )
        for case in block["cases"]:
            foot = case["foot_m"]
            depth = case["depth_m"]
            drift = case["drift_end_m"]
            rot = case["rotation_gt_90"]
            lines.append(
                f"  dt {case['dt_s']:.5f}s h {case['execution_horizon']}: "
                f"foot off {_mm(foot['ensemble_off']['median'])}/"
                f"{_mm(foot['ensemble_off']['max'])} "
                f"on {_mm(foot['ensemble_on']['median'])}/"
                f"{_mm(foot['ensemble_on']['max'])} mm, "
                f"drift end {_mm(drift['median'])}/{_mm(drift['max'])} mm, "
                f"depth off {_mm(depth['ensemble_off']['median'])} "
                f"on {_mm(depth['ensemble_on']['median'])} mm, "
                f"rot>90 {rot['n']}/{rot['n_executed']} "
                f"t1-flip {rot['t1_flip_vs_stored']} "
                f"t1-swap {rot['t1_swap_vs_stored']} "
                f"n-flip {rot['normal_flip_vs_stored']} "
                f"step-flip {rot['t1_flip_vs_previous_step']} "
                f"foot>20mm {rot['foot_inplane_gt_20mm']} "
                f"left-mesh {rot['euler_step_left_mesh_gt_5mm']} "
                f"neither {rot['neither_frame_jump_nor_foot_far']} "
                f"field-deg {rot['field_geodesic_deg_on_outliers']['median']}"
            )
        for name in ("clamped_contact", "clamped_free", "unclamped_contact"):
            row = block["floor"][name]
            dn, tcp = row["delta_n_m"], row["tcp_along_n_m"]
            lines.append(
                f"  {name}: n {dn['n']} "
                f"δn p5/50/95 {_mm(dn['p5'])}/{_mm(dn['p50'])}/{_mm(dn['p95'])} mm, "
                f"TCP n p5/50/95 {_mm(tcp['p5'])}/{_mm(tcp['p50'])}/{_mm(tcp['p95'])} mm"
            )
    return "\n".join(lines)


def main() -> None:
    if (OUT / "settings.json").exists():
        raise SystemExit(f"refusing to overwrite {OUT / 'settings.json'}")
    OUT.mkdir(parents=True, exist_ok=True)
    report = analyze()
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "settings.json").write_text(
        json.dumps(
            {
                "episodes": list(EPISODES),
                "dt_s": [DT_LABEL_S, DT_RUNNER_S],
                "apply_safety": False,
                "ensemble_decay_m": rg.DECAY_M,
                "foot_far_m": FOOT_FAR_M,
                "left_mesh_m": LEFT_MESH_M,
            },
            indent=2,
        )
        + "\n"
    )
    print(format_report(report), flush=True)


if __name__ == "__main__":
    main()
