#!/usr/bin/env python3
"""Read-only mold calibration check while the arm is jogged by hand.

Does not load a policy and does not send motion. Each sample is one RTDE
receive of the measured TCP, mapped with the runner's fixture extrinsics,
then passed through MoldAwareRuntime.fill_obs_raw (same tracked-foot query
and hop repair the policy uses).

Square the tool to the surface for the normal check. Tool +z should point
into the mold, along -normal. A layup scrape is tilted about 28 deg off
that axis and will fail the 5 deg check even when the mesh is right.

The six labels and their fixture-frame targets are in mold_calib_points.yaml.
Left is +Z, right is -Z, near is the +X end (closer to the right-arm base).

  source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-surface_frames
  conda activate pyrite
  python UMIFT_Data/surface_frames_scripts/mold_calib_check.py --mold train
  python UMIFT_Data/surface_frames_scripts/mold_calib_check.py --mold heldout --field /path/to/fixb_field.npz
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

_SURFACE_SCRIPTS = Path(__file__).resolve().parent
if str(_SURFACE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SURFACE_SCRIPTS))

from PyriteConfig.tasks.umift.workcell_frames import load_workcell_extrinsics
from fixb_labels import pose7_to_Rp
from mold_aware_runtime import DEFAULT_FIELD, DEFAULT_META, MoldAwareRuntime

# Policy mesh. The October v3.5 and stringer_data_v2_full copies are this file.
TRAIN_FIELD = DEFAULT_FIELD
# A field built from a scan that was not used to train. Not shipped; pass --field.
HELDOUT_FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/heldout_mold/shared/fixb_field.npz"
)
POINTS_PATH = Path(__file__).with_name("mold_calib_points.yaml")
DEFAULT_ROBOT_IP = "192.168.1.2"
DEFAULT_RTDE_PYTHON = Path("/home/boeing/keyence_scan_ws/.venv/bin/python")
SD_FAIL_MM = 3.0
NORMAL_FAIL_DEG = 5.0

CSV_FIELDS = (
    "label",
    "unix_s",
    "sd_mm",
    "foot_x_mm",
    "foot_y_mm",
    "foot_z_mm",
    "n_x",
    "n_y",
    "n_z",
    "angle_n_toolz_deg",
    "normal_error_deg",
    "tcp_x_mm",
    "tcp_y_mm",
    "tcp_z_mm",
    "offset_x_mm",
    "offset_y_mm",
    "offset_z_mm",
    "passed",
    "spot_warn",
    "spot_dist_mm",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_points(path: Path = POINTS_PATH) -> dict:
    """Labels, axis definitions, and per-mold expected xyz."""
    raw = yaml.safe_load(Path(path).read_text())
    points = []
    for item in raw["points"]:
        points.append(
            {
                "label": str(item["label"]),
                "description": str(item["description"]),
                "train_xyz_m": np.asarray(item["train_xyz_m"], dtype=np.float64).reshape(3),
                "heldout_xyz_m": np.asarray(item["heldout_xyz_m"], dtype=np.float64).reshape(3),
            }
        )
    return {
        "spot_warn_mm": float(raw["spot_warn_mm"]),
        "stringer_axis": np.asarray(raw["stringer_axis"], dtype=np.float64).reshape(3),
        "across_axis": np.asarray(raw["across_axis"], dtype=np.float64).reshape(3),
        "crown_normal": np.asarray(raw["crown_normal"], dtype=np.float64).reshape(3),
        "points": points,
    }


def expected_xyz(point: dict, mold: str) -> np.ndarray:
    key = "heldout_xyz_m" if mold == "heldout" else "train_xyz_m"
    return np.asarray(point[key], dtype=np.float64).reshape(3)


def pose7_base_to_fixture(pose7: np.ndarray, T_fixture_from_base: np.ndarray) -> np.ndarray:
    """Same map as umift_env_runner.pose7_batch_to_fixture (right_base → fixture)."""
    from fixb_labels import Rp_to_pose7, pose7_to_Rp

    pose7 = np.asarray(pose7, dtype=np.float64)
    flat = pose7.reshape(-1, 7)
    R, p = pose7_to_Rp(flat)
    T = np.asarray(T_fixture_from_base, dtype=np.float64)
    Rf = np.einsum("ij,njk->nik", T[:3, :3], R)
    pf = (T[:3, :3] @ p.T).T + T[:3, 3]
    return Rp_to_pose7(Rf, pf).reshape(pose7.shape)


def resolve_field(mold: str, field: Optional[Path]) -> Path:
    if field is not None:
        return Path(field).expanduser().resolve()
    if mold == "train":
        return Path(TRAIN_FIELD)
    if mold == "heldout":
        return Path(HELDOUT_FIELD)
    raise ValueError(f"unknown mold {mold!r}")


def rotvec_to_pose7(tcp6: np.ndarray) -> np.ndarray:
    """UR getActualTCPPose [x,y,z,rx,ry,rz] -> pose7 [x,y,z,qw,qx,qy,qz].

    Matches URRTDE::getCartesian, which stores RUT::aa2quat (w, x, y, z).
    """
    tcp6 = np.asarray(tcp6, dtype=np.float64).reshape(6)
    rxryrz = tcp6[3:6]
    angle = float(np.linalg.norm(rxryrz))
    quat = np.zeros(4, dtype=np.float64)
    quat[0] = np.cos(angle / 2.0)
    if angle > 1e-12:
        quat[1:] = np.sin(angle / 2.0) * (rxryrz / angle)
    return np.concatenate([tcp6[:3], quat])


def normal_error_deg(normal: np.ndarray, tool_z: np.ndarray) -> tuple[float, float]:
    """Angle between normal and tool z, and the error from pressing along -normal."""
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    z = np.asarray(tool_z, dtype=np.float64).reshape(3)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    z = z / max(float(np.linalg.norm(z)), 1e-12)
    angle = float(np.degrees(np.arccos(np.clip(np.dot(n, z), -1.0, 1.0))))
    error = float(np.degrees(np.arccos(np.clip(np.dot(-n, z), -1.0, 1.0))))
    return angle, error


def point_passed(sd_mm: float, normal_err_deg: float, sd_fail_mm: float, normal_fail_deg: float) -> bool:
    return abs(sd_mm) <= sd_fail_mm and normal_err_deg <= normal_fail_deg


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _omega_cross_cols(d: np.ndarray) -> np.ndarray:
    """Columns so that M @ ω = ω × d."""
    x, y, z = np.asarray(d, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, z, -y], [-z, 0.0, x], [y, -x, 0.0]],
        dtype=np.float64,
    )


def fit_rigid_offset(
    feet_m: np.ndarray,
    tcp_m: np.ndarray,
    stringer_axis: np.ndarray,
    across_axis: np.ndarray,
    crown_normal: np.ndarray,
) -> dict:
    """Small rigid map tcp ≈ foot + t + ω × (foot - centroid).

    t is the translation of the foot centroid. Yaw is ω about the crown normal.
    """
    feet_m = np.asarray(feet_m, dtype=np.float64).reshape(-1, 3)
    tcp_m = np.asarray(tcp_m, dtype=np.float64).reshape(-1, 3)
    center = feet_m.mean(axis=0)
    blocks = []
    rhs = []
    for foot, tcp in zip(feet_m, tcp_m):
        blocks.append(np.concatenate([np.eye(3), _omega_cross_cols(foot - center)], axis=1))
        rhs.append(tcp - foot)
    sol, *_ = np.linalg.lstsq(np.vstack(blocks), np.concatenate(rhs), rcond=None)
    t = np.asarray(sol[:3], dtype=np.float64)
    omega = np.asarray(sol[3:], dtype=np.float64)
    pred = feet_m + t + np.cross(omega, feet_m - center)
    resid = tcp_m - pred
    resid_rms = float(np.sqrt(np.mean(np.sum(resid * resid, axis=1))))
    crown = _unit(crown_normal)
    return {
        "t_m": t,
        "omega_rad": omega,
        "resid_rms_m": resid_rms,
        "along_m": float(np.dot(t, _unit(stringer_axis))),
        "across_m": float(np.dot(t, _unit(across_axis))),
        "crown_m": float(np.dot(t, crown)),
        "yaw_deg": float(np.degrees(np.dot(omega, crown))),
        "center_m": center,
    }


def spot_distance_mm(foot_m: np.ndarray, expected_m: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(foot_m) - np.asarray(expected_m)) * 1e3)


def measure_contact(runtime, T_fixture_from_base: np.ndarray, pose7_base: np.ndarray) -> dict:
    """One live sample through the runner's fixture map and fill_obs_raw."""
    pose7_base = np.asarray(pose7_base, dtype=np.float64).reshape(7)
    pose_fix = np.asarray(
        pose7_base_to_fixture(pose7_base, T_fixture_from_base), dtype=np.float64
    ).reshape(7)
    obs = {"ts_pose_fb_0": pose_fix.reshape(1, 7)}
    filled = runtime.fill_obs_raw(obs, robot_id=0)
    R_tool, p_fix = pose7_to_Rp(pose_fix.reshape(1, 7))
    normal = np.asarray(filled["normal"][-1], dtype=np.float64)
    foot = np.asarray(filled["x_s"][-1], dtype=np.float64)
    sd_m = float(filled["sd"][-1])
    angle, err = normal_error_deg(normal, R_tool[0, :, 2])
    return {
        "sd_m": sd_m,
        "foot_m": foot,
        "normal": normal,
        "angle_n_toolz_deg": angle,
        "normal_error_deg": err,
        "tcp_fix_m": p_fix[0],
        "offset_m": p_fix[0] - foot,
    }


def read_tcp6_via_rtde(python: Path, robot_ip: str) -> np.ndarray:
    """Actual TCP from RTDE receive only. Does not construct a control interface."""
    import subprocess

    code = (
        "from rtde_receive import RTDEReceiveInterface\n"
        f"r = RTDEReceiveInterface(hostname={robot_ip!r})\n"
        "p = r.getActualTCPPose()\n"
        "print(','.join(str(x) for x in p))\n"
        "r.disconnect()\n"
    )
    proc = subprocess.run(
        [str(python), "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"RTDE receive failed for {robot_ip}: {proc.stderr.strip()}"
        )
    vals = [float(x) for x in proc.stdout.strip().split(",") if x.strip()]
    if len(vals) != 6:
        raise RuntimeError(f"Expected 6 TCP values, got {vals!r}")
    return np.asarray(vals, dtype=np.float64)


def _fmt_vec(v_mm) -> str:
    return " ".join(f"{float(x):+8.2f}" for x in np.asarray(v_mm).reshape(-1))


def print_sample(label: str, row: dict, passed: bool) -> None:
    flag = "PASS" if passed else "FAIL"
    print(
        f"{label}  {flag}  sd={float(row['sd_mm']):+.2f} mm  "
        f"normal_error={float(row['normal_error_deg']):.2f} deg"
    )
    print(
        "  foot fixture mm:    "
        + _fmt_vec([row["foot_x_mm"], row["foot_y_mm"], row["foot_z_mm"]])
    )
    print(
        "  normal fixture:     "
        + _fmt_vec([row["n_x"], row["n_y"], row["n_z"]])
    )
    print(f"  angle(n, tool z):   {float(row['angle_n_toolz_deg']):.2f} deg")


def print_summary(rows: list[dict], catalog: dict) -> None:
    if not rows:
        print("No points recorded.")
        return
    n_pass = sum(1 for r in rows if r["passed"] == "yes")
    n_warn = sum(1 for r in rows if r.get("spot_warn") == "yes")
    feet = np.array(
        [[r["foot_x_mm"], r["foot_y_mm"], r["foot_z_mm"]] for r in rows],
        dtype=np.float64,
    ) / 1000.0
    tcp = np.array(
        [[r["tcp_x_mm"], r["tcp_y_mm"], r["tcp_z_mm"]] for r in rows],
        dtype=np.float64,
    ) / 1000.0
    fit = fit_rigid_offset(
        feet,
        tcp,
        catalog["stringer_axis"],
        catalog["across_axis"],
        catalog["crown_normal"],
    )
    print(f"\n{len(rows)} points  pass={n_pass}  fail={len(rows) - n_pass}  spot_warn={n_warn}")
    print(f"rigid residual rms mm:                 {fit['resid_rms_m'] * 1e3:.2f}")
    print(f"along stringer (+X, near) mm:          {fit['along_m'] * 1e3:+.2f}")
    print(f"across stringer (+Z, left) mm:         {fit['across_m'] * 1e3:+.2f}")
    print(f"crown normal (-Y, up) mm:              {fit['crown_m'] * 1e3:+.2f}")
    print(f"yaw about crown normal deg:            {fit['yaw_deg']:+.3f}")
    print(
        "Along / across / crown are the translation of the foot centroid. "
        "A large residual means the six points are not one rigid offset."
    )


def row_from_sample(label: str, sample: dict, passed: bool, spot_dist_mm: float, spot_warn: bool) -> dict:
    foot = sample["foot_m"] * 1e3
    tcp = sample["tcp_fix_m"] * 1e3
    offset = sample["offset_m"] * 1e3
    return {
        "label": label,
        "unix_s": f"{datetime.now().timestamp():.3f}",
        "sd_mm": f"{sample['sd_m'] * 1e3:.4f}",
        "foot_x_mm": f"{foot[0]:.3f}",
        "foot_y_mm": f"{foot[1]:.3f}",
        "foot_z_mm": f"{foot[2]:.3f}",
        "n_x": f"{sample['normal'][0]:.6f}",
        "n_y": f"{sample['normal'][1]:.6f}",
        "n_z": f"{sample['normal'][2]:.6f}",
        "angle_n_toolz_deg": f"{sample['angle_n_toolz_deg']:.3f}",
        "normal_error_deg": f"{sample['normal_error_deg']:.3f}",
        "tcp_x_mm": f"{tcp[0]:.3f}",
        "tcp_y_mm": f"{tcp[1]:.3f}",
        "tcp_z_mm": f"{tcp[2]:.3f}",
        "offset_x_mm": f"{offset[0]:.3f}",
        "offset_y_mm": f"{offset[1]:.3f}",
        "offset_z_mm": f"{offset[2]:.3f}",
        "passed": "yes" if passed else "no",
        "spot_warn": "yes" if spot_warn else "no",
        "spot_dist_mm": f"{spot_dist_mm:.2f}",
    }


def _print_banner(field_path: Path, extrinsics_path: str, sd_fail_mm: float, normal_fail_deg: float) -> None:
    meta = field_path.with_suffix(".json")
    print(f"mesh:       {field_path}")
    print(f"mesh sha256:{sha256_file(field_path)}")
    if meta.is_file():
        print(f"mesh json:  {meta}")
    print(f"extrinsics: {extrinsics_path}")
    print(f"extrinsics sha256: {sha256_file(Path(extrinsics_path))}")
    print(
        f"fail if |signed distance| > {sd_fail_mm:.1f} mm "
        f"or tool +z is more than {normal_fail_deg:.1f} deg from -normal."
    )
    print("Square the tool to the surface. This script does not move the arm.")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mold", choices=("train", "heldout"), default="train")
    parser.add_argument(
        "--field",
        type=Path,
        default=None,
        help="fixb_field.npz. Overrides --mold. Use this for a held-out mesh.",
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--rtde-python", type=Path, default=DEFAULT_RTDE_PYTHON)
    parser.add_argument("--sd-mm", type=float, default=SD_FAIL_MM)
    parser.add_argument("--normal-deg", type=float, default=NORMAL_FAIL_DEG)
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the mesh and extrinsics paths and hashes, then exit.",
    )
    args = parser.parse_args(argv)

    field_path = resolve_field(args.mold, args.field)
    if not field_path.is_file():
        print(f"Mold field not found: {field_path}", file=sys.stderr)
        if args.mold == "heldout" and args.field is None:
            print(
                "There is no held-out fixb_field.npz yet. The train, October, "
                "and stringer_data_v2_full copies are the same file. "
                "Pass --field /path/to/fixb_field.npz.",
                file=sys.stderr,
            )
        return 1

    extrinsics = load_workcell_extrinsics()
    _print_banner(field_path, extrinsics.path, args.sd_mm, args.normal_deg)
    if args.print_only:
        return 0

    meta_path = field_path.with_name("fixb_meta.json")
    if not meta_path.is_file():
        meta_path = DEFAULT_META
    runtime = MoldAwareRuntime.from_assets(
        run_type="r1_world_world",
        field_path=field_path,
        meta_path=meta_path,
        dry_run=True,
    )
    runtime.reset()

    csv_path = args.csv
    if csv_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Path.cwd() / f"mold_calib_{args.mold}_{stamp}.csv"
    csv_path = csv_path.expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
    print(f"csv: {csv_path}")
    catalog = load_points()
    points = catalog["points"]
    print(
        "Axes: near/far along +X (near is closer to the right-arm base), "
        "left = +Z, right = -Z, crown outward = -Y."
    )
    print(f"{len(points)} touches. Press Enter to record the prompted point. q to finish.\n")
    for i, point in enumerate(points, start=1):
        xyz_mm = expected_xyz(point, args.mold) * 1e3
        print(f"Next {i}/{len(points)}  {point['label']}")
        print(f"  {point['description']}")
        print(f"  expected fixture mm: {_fmt_vec(xyz_mm)}")

    rows: list[dict] = []
    index = 0
    while index < len(points):
        point = points[index]
        try:
            typed = input(f"record {point['label']}> ").strip()
        except EOFError:
            typed = "q"
        if typed.lower() in {"q", "quit", "exit"}:
            break
        tcp6 = read_tcp6_via_rtde(args.rtde_python, args.robot_ip)
        sample = measure_contact(
            runtime, extrinsics.T_fixture_from_right_base, rotvec_to_pose7(tcp6)
        )
        ok = point_passed(
            sample["sd_m"] * 1e3,
            sample["normal_error_deg"],
            args.sd_mm,
            args.normal_deg,
        )
        dist_mm = spot_distance_mm(sample["foot_m"], expected_xyz(point, args.mold))
        warn = dist_mm > catalog["spot_warn_mm"]
        row = row_from_sample(point["label"], sample, ok, dist_mm, warn)
        rows.append(row)
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
        print_sample(point["label"], row, ok)
        if warn:
            print(
                f"  WARN foot is {dist_mm:.1f} mm from the expected spot "
                f"(limit {catalog['spot_warn_mm']:.0f} mm). Likely the wrong corner."
            )
        index += 1

    missing = [p["label"] for p in points[index:]]
    if missing:
        print("Not recorded: " + ", ".join(missing))
    print_summary(rows, catalog)
    print(f"csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
