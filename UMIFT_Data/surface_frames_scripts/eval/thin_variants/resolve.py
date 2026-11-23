#!/usr/bin/env python3
"""Re-solve thin-stringer TCP for four extrinsics. New files only.

A  recorded odom, thin_stringer.yaml fixture
B  odom re-levered from the Nov-18 frame_conventions mount onto the
   training mount (Rx+90, lever -0.01,-0.157,-0.136), same fixture
C  recorded odom, fixture translation 0 and R_FIXTURE_MASTER_TAG
D  B odom + C fixture

C and D are the A and B solves left-multiplied by the fixture change.
That is the same step pipeline.py applies after the graph solve.
One episode is solved again with the C yaml to check the multiply.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
import zarr
from scipy.spatial.transform import Rotation

ROOT = Path("/home/boeing/umi_ft_ws")
CMP = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
sys.path[:0] = [
    str(ROOT / "ee_tag_traj"),
    str(ROOT / "Robotiq-portable-real2sim" / "python"),
    str(CMP / "UMIFT_Data/surface_frames_scripts"),
]

import frame_conventions as fc  # noqa: E402
from ee_tag_traj import pipeline  # noqa: E402
from ee_tag_traj.geometry import T_from_Rt, inv_T, mul_T  # noqa: E402
from ee_tag_traj.io_zarr import load_demo  # noqa: E402
from ee_tag_traj.t265 import interpolate_t265  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402

SRC_YAML = ROOT / "ee_tag_traj/configs/thin_stringer.yaml"
ZARR_DIR = Path("/mnt/data/umidata/ft/datasets/flat_stringer/zarr")
WALL = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/"
    "wall_fix/fixb_field_wall_vertical.npz"
)
OUT = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/variant_resolve"
)
HERE = Path(__file__).resolve().parent
T_THIN = np.array([0.184936, -0.000799, -0.060084])
TRAIN_P = np.array([-0.01, -0.157, -0.136])
TRAIN_E = np.array([90.0, 0.0, 0.0])
NOW_P = np.array([-0.01, -0.172, -0.1454])
NOW_E = np.array([180.0, 0.0, 0.0])
EPISODES = (
    ("A", "2026-11-19-15.43.29"),
    ("B", "2026-11-19-15.45.09"),
    ("C", "2026-11-19-15.49.52"),
    ("D", "2026-11-19-15.51.13"),
)
FORCE_N = 8.0
_LOAD = load_demo
_RIGHT = np.eye(4)


def _check() -> None:
    got = fc.R_from_euler_xyz_deg([90.0, 0.0, 0.0]) @ np.array([0.0, 1.0, 0.0])
    if not np.allclose(got, [0.0, 0.0, 1.0]):
        raise RuntimeError("Rx(+90) is not the training mount rotation")
    if not np.allclose(_lever_right(TRAIN_P, TRAIN_E, TRAIN_P, TRAIN_E), np.eye(4)):
        raise RuntimeError("same mount must be a no-op")
    eye = np.eye(4)
    eye[:3, 3] = [1.0, 2.0, 3.0]
    moved = _drop_fixture_translation(eye[None])[0]
    if not np.allclose(moved[:3, 3], eye[:3, 3] - T_THIN):
        raise RuntimeError("fixture multiply")


def _lever_right(src_p, src_e, dst_p, dst_e) -> np.ndarray:
    """Right-multiply that turns a src-mount TCP pose into a dst-mount TCP pose."""
    src = fc.default_T_tcp_t265(src_p, src_e)
    dst = fc.default_T_tcp_t265(dst_p, dst_e)
    return src @ inv_T(dst)


def _drop_fixture_translation(T: np.ndarray) -> np.ndarray:
    """Thin fixture and the training fixture share R, so this is [I | -t] @ T."""
    out = np.array(T, copy=True)
    out[:, :3, 3] -= T_THIN
    return out


def _write_yamls() -> tuple[Path, Path]:
    cfg = OUT / "cfg"
    cfg.mkdir(parents=True, exist_ok=True)
    text = SRC_YAML.read_text()
    as_is = cfg / "variant_A.yaml"
    train_fix = cfg / "variant_C.yaml"
    if not as_is.exists():
        as_is.write_text(text)
    if not train_fix.exists():
        old = "translation: [0.184936, -0.000799, -0.060084]"
        if old not in text:
            raise SystemExit("thin yaml translation line moved")
        train_fix.write_text(text.replace(old, "translation: [0.0, 0.0, 0.0]", 1))
    return as_is, train_fix


def _relever(demo):
    R_right = Rotation.from_matrix(_RIGHT[:3, :3])
    t_right = _RIGHT[:3, 3]
    old = Rotation.from_quat(demo.t265_quat_xyzw)
    demo.t265_pos = demo.t265_pos + old.apply(t_right)
    demo.t265_quat_xyzw = (old * R_right).as_quat()
    dt = np.diff(demo.t265_t)
    vel = np.zeros_like(demo.t265_pos)
    vel[1:] = (demo.t265_pos[1:] - demo.t265_pos[:-1]) / np.maximum(dt, 1e-4)[:, None]
    vel[0] = vel[1] if len(vel) > 1 else 0.0
    demo.t265_vel = vel
    demo.t265_angvel = R_right.inv().apply(demo.t265_angvel)
    return demo


def _load_maybe(path):
    demo = _LOAD(path)
    if not np.allclose(_RIGHT, np.eye(4)):
        demo = _relever(demo)
    return demo


def _solve(zarr_path: Path, yaml_path: Path, dest: Path, relever: bool) -> None:
    if dest.exists():
        print(f"[variant] keep {dest.name}", flush=True)
        return
    global _RIGHT
    _RIGHT = _lever_right(NOW_P, NOW_E, TRAIN_P, TRAIN_E) if relever else np.eye(4)
    pipeline.load_demo = _load_maybe
    pipeline.run_demo(zarr_path, yaml_path, output_dir=dest.parent, write_overlays=False)
    written = dest.parent / f"{zarr_path.name.replace('_unified.zarr', '')}_ee_master.npz"
    if written != dest:
        if dest.exists():
            raise SystemExit(f"refusing to overwrite {dest}")
        written.rename(dest)


def _force(zarr_path: Path, times: np.ndarray) -> np.ndarray:
    g = zarr.open(zarr_path, "r")["sensors"]["ft_wrench"]
    t = np.asarray(g["timestamp_sec_unified"], dtype=np.float64)
    f = np.column_stack([np.asarray(g[k], dtype=np.float64) for k in ("force_x", "force_y", "force_z")])
    j = np.searchsorted(t, times)
    j = np.clip(j, 1, len(t) - 1)
    left = np.abs(times - t[j - 1]) <= np.abs(times - t[j])
    j = np.where(left, j - 1, j)
    return np.linalg.norm(f[j], axis=1)


def _stats(mm: np.ndarray) -> dict:
    if len(mm) == 0:
        return {"n": 0, "median_mm": None, "p90_mm": None}
    return {
        "n": int(len(mm)),
        "median_mm": round(float(np.median(mm)), 1),
        "p90_mm": round(float(np.percentile(mm, 90)), 1),
    }


def _odom(zarr_path: Path, times: np.ndarray, relever: bool) -> np.ndarray:
    """T265 TCP poses at RGB times. Skips RGB."""
    g = zarr.open(zarr_path, "r")["sensors"]["t265_odom"]
    t = np.asarray(g["timestamp_sec_unified"], dtype=np.float64)
    pos = np.column_stack([np.asarray(g[k], dtype=np.float64) for k in ("pos_x", "pos_y", "pos_z")])
    quat = np.column_stack([np.asarray(g[k], dtype=np.float64) for k in ("orient_x", "orient_y", "orient_z", "orient_w")])
    if relever:
        right = _lever_right(NOW_P, NOW_E, TRAIN_P, TRAIN_E)
        rot = Rotation.from_matrix(right[:3, :3])
        old = Rotation.from_quat(quat)
        pos = pos + old.apply(right[:3, 3])
        quat = (old * rot).as_quat()
    demo = type("D", (), {})()
    demo.t265_t = t
    demo.t265_pos = pos
    demo.t265_quat_xyzw = quat
    demo.t265_vel = np.zeros_like(pos)
    demo.t265_angvel = np.zeros_like(pos)
    T, _, _ = interpolate_t265(demo, times)
    return T


def _drift(zarr_path: Path, solved: np.ndarray, times: np.ndarray, tag: np.ndarray, relever: bool) -> dict:
    T_odom = _odom(zarr_path, times, relever)
    idx = np.flatnonzero(tag >= 1)
    gaps = []
    for i, j in zip(idx[:-1], idx[1:]):
        pred = mul_T(solved[i], mul_T(inv_T(T_odom[i]), T_odom[j]))
        gaps.append(float(np.linalg.norm(pred[:3, 3] - solved[j, :3, 3]) * 1000.0))
    return _stats(np.asarray(gaps, dtype=np.float64))


def _measure(field, zarr_path: Path, npz: Path, relever: bool) -> dict:
    blob = np.load(npz)
    T = np.asarray(blob["T_master_tcp"], dtype=np.float64)
    times = np.asarray(blob["timestamp_sec_unified"], dtype=np.float64)
    tag = np.asarray(blob["n_tags_visible"]).astype(int)
    pos = T[:, :3, 3]
    dist = np.linalg.norm(pos - field.closest(pos)[0], axis=1) * 1000.0
    contact = _force(zarr_path, times) >= FORCE_N
    on_tag = tag >= 1
    return {
        "frames": int(len(times)),
        "tag_frames": int(on_tag.sum()),
        "contact": _stats(dist[contact]),
        "tag": _stats(dist[on_tag]),
        "drift": _drift(zarr_path, T, times, tag, relever),
    }


def main() -> None:
    _check()
    if not WALL.is_file():
        raise SystemExit(f"missing mesh {WALL}")
    as_is, train_fix = _write_yamls()
    # The C yaml is a copy. Confirm the source yaml was not edited.
    src = yaml.safe_load(SRC_YAML.read_text())
    if src["extrinsics"]["T_fixture_master_tag"]["translation"] != T_THIN.tolist():
        raise SystemExit("thin_stringer.yaml translation changed")
    field = FixBMoldField(str(WALL))
    report = {"episodes": {}, "mount": {
        "publisher_cpp_default": {"euler_xyz_deg": [90, 0, 0], "p_t265_in_tcp_m": TRAIN_P.tolist()},
        "frame_conventions_2026_11_18": {"euler_xyz_deg": NOW_E.tolist(), "p_t265_in_tcp_m": NOW_P.tolist()},
        "B_assumes": "bag TCP was published with the Nov-18 frame_conventions mount, then re-levered to the training mount",
    }}
    for letter, stamp in EPISODES:
        zarr_path = ZARR_DIR / f"{stamp}_unified.zarr"
        print(f"[variant] episode {letter} {stamp}", flush=True)
        a_path = OUT / "A" / f"{stamp}_ee_master.npz"
        b_path = OUT / "B" / f"{stamp}_ee_master.npz"
        _solve(zarr_path, as_is, a_path, relever=False)
        _solve(zarr_path, as_is, b_path, relever=True)
        a = np.load(a_path, allow_pickle=True)
        b = np.load(b_path, allow_pickle=True)
        for name, src_T, src_path in (
            ("C", np.asarray(a["T_master_tcp"]), a_path),
            ("D", np.asarray(b["T_master_tcp"]), b_path),
        ):
            dest = OUT / name / f"{stamp}_ee_master.npz"
            if dest.exists():
                raise SystemExit(f"refusing to overwrite {dest}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = a if name == "C" else b
            payload = {k: np.array(src[k]) for k in src.files}
            payload["T_master_tcp"] = _drop_fixture_translation(src_T)
            payload["pos"] = payload["T_master_tcp"][:, :3, 3]
            pose = np.array(payload["pose7"])
            pose[:, :3] = payload["pos"]
            payload["pose7"] = pose
            np.savez_compressed(dest, **payload)
        if letter == "C":
            check = OUT / "_check_C"
            _solve(zarr_path, train_fix, check / f"{stamp}_ee_master.npz", relever=False)
            got = np.load(check / f"{stamp}_ee_master.npz", allow_pickle=True)["pos"]
            ref = np.load(OUT / "C" / f"{stamp}_ee_master.npz")["pos"]
            err = float(np.max(np.abs(got - ref)))
            print(f"[variant] C re-solve vs left-multiply max abs m {err:.3e}", flush=True)
            if err > 1e-6:
                raise SystemExit("fixture left-multiply does not match a re-solve")
        row = {}
        for name, relever in (("A", False), ("B", True), ("C", False), ("D", True)):
            row[name] = _measure(field, zarr_path, OUT / name / f"{stamp}_ee_master.npz", relever)
            c = row[name]["contact"]
            print(
                f"[variant] {letter} {name} contact n={c['n']} med={c['median_mm']} p90={c['p90_mm']} "
                f"tag n={row[name]['tag']['n']} med={row[name]['tag']['median_mm']}",
                flush=True,
            )
        report["episodes"][letter] = {"stamp": stamp, "variants": row}
    dest = HERE / "report.json"
    if dest.exists():
        raise SystemExit(f"refusing to overwrite {dest}")
    dest.write_text(json.dumps(report, indent=2))
    print(f"[variant] wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
