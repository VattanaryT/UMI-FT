"""compliance_command matches the runner's old inline stiffness assembly.

The legacy function below is that inline block, kept only as the oracle.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))

from PyriteEnvSuites.env_runners.stiffness_floor import apply_press_stiffness_floor  # noqa: E402
from PyriteUtility.spatial_math import spatial_utilities as su  # noqa: E402
from mold_aware_runtime import compliance_command  # noqa: E402

E40_LOG = Path("/home/boeing/data/control_log/checkpoints_e40_rollout0_1795128531")
GT_LOG = Path("/home/boeing/data/control_log/gt_replay_ep40_1795121119")


def legacy_compliance_command(
    nominal_se3,
    virtual_se3,
    stiffness,
    measured_pose7,
    stiffness_floor=None,
    test_nominal_target=False,
    test_nominal_target_stiffness=-1.0,
):
    """The runner loop that compliance_command replaced, verbatim."""
    nominal_se3 = np.asarray(nominal_se3).reshape(-1, 4, 4)
    virtual_se3 = np.asarray(virtual_se3).reshape(-1, 4, 4)
    measured_pose7 = np.asarray(measured_pose7).reshape(7)
    SE3_TW = su.SE3_inv(su.pose7_to_SE3(measured_pose7))
    ts_targets_nominal = su.SE3_to_pose7(nominal_se3)
    ts_targets_virtual = su.SE3_to_pose7(virtual_se3)

    ts_stiffnesses = np.zeros([6, 6 * ts_targets_virtual.shape[0]])
    raw_steps = []
    floored_steps = []
    for i in range(ts_targets_virtual.shape[0]):
        SE3_target = nominal_se3[i].reshape([4, 4])
        SE3_virtual_target = virtual_se3[i].reshape([4, 4])

        # stiffness: 1. convert vt to tool frame
        SE3_TVt = SE3_TW @ SE3_virtual_target
        SE3_Ttarget = SE3_TW @ SE3_target

        # stiffness: 2. compute stiffness matrix in the tool frame
        compliance_direction_tool = (
            SE3_TVt[:3, 3] - SE3_Ttarget[:3, 3]
        ).reshape(3)

        if np.linalg.norm(compliance_direction_tool) < 0.001:  #
            compliance_direction_tool = np.array([1.0, 0.0, 0.0])

        compliance_direction_tool /= np.linalg.norm(
            compliance_direction_tool
        )
        X = compliance_direction_tool
        Y = np.cross(X, np.array([0, 0, 1]))
        Y /= np.linalg.norm(Y)
        Z = np.cross(X, Y)

        default_stiffness = 5000
        default_stiffness_rot = 100
        stiffness_matrix_full = np.eye(6) * default_stiffness_rot

        if stiffness is None:
            raw_k = default_stiffness
        else:
            raw_k = stiffness[i]
        _, floored_k = apply_press_stiffness_floor(raw_k, stiffness_floor)
        target_stiffness = float(floored_k[0])
        raw_steps.append(float(np.asarray(raw_k, dtype=np.float64)))
        floored_steps.append(target_stiffness)

        M = np.diag(
            [target_stiffness, default_stiffness, default_stiffness]
        )
        S = np.array([X, Y, Z]).T
        stiffness_matrix = S @ M @ np.linalg.inv(S)
        stiffness_matrix_full[:3, :3] = stiffness_matrix

        if (
            test_nominal_target
            and test_nominal_target_stiffness > 0
        ):
            stiffness_matrix_full[:3, :3] = (
                np.eye(3) * test_nominal_target_stiffness
            )

        ts_stiffnesses[:, 6 * i : 6 * i + 6] = stiffness_matrix_full

    if test_nominal_target:
        scheduled = ts_targets_nominal
    else:
        scheduled = ts_targets_virtual
    return (
        ts_targets_nominal,
        scheduled,
        ts_stiffnesses,
        np.asarray(raw_steps, dtype=np.float64),
        np.asarray(floored_steps, dtype=np.float64),
    )


def _pose7_batch_to_se3(pose7: np.ndarray) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(-1, 7)
    return np.stack([su.pose7_to_SE3(row) for row in pose7], axis=0)


def _max_abs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _compare(nominal_se3, virtual_se3, stiffness, measured, floor, nominal_flag, k_override):
    args = (
        nominal_se3,
        virtual_se3,
        stiffness,
        measured,
        floor,
        nominal_flag,
        k_override,
    )
    old = legacy_compliance_command(*args)
    new = compliance_command(*args)
    names = ("nominal_pose7", "scheduled_pose7", "stiffness_6x6n", "raw", "floored")
    return {name: _max_abs(a, b) for name, a, b in zip(names, old, new)}


def test_matches_legacy_on_a_pressed_chunk():
    measured = np.array([0.1, -0.2, 0.3, 0.9, 0.1, -0.2, 0.3], dtype=np.float64)
    nominal = np.zeros((4, 4, 4), dtype=np.float64)
    virtual = np.zeros_like(nominal)
    for i in range(4):
        nominal[i] = np.eye(4)
        virtual[i] = np.eye(4)
        nominal[i, :3, 3] = [0.01 * i, -0.05, 0.02]
        virtual[i, :3, 3] = nominal[i, :3, 3] + np.array([0.0, 0.004, 0.0])
    stiffness = np.array([800.0, 3000.0, 4500.0, 120.0])
    diff = _compare(nominal, virtual, stiffness, measured, 3000.0, False, -1.0)
    assert max(diff.values()) == 0.0


def test_nominal_target_and_isotropic_override_match_legacy():
    measured = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    nominal = np.eye(4).reshape(1, 4, 4).copy()
    virtual = nominal.copy()
    virtual[0, 1, 3] = -0.01
    diff = _compare(nominal, virtual, np.array([1000.0]), measured, None, True, 1500.0)
    assert max(diff.values()) == 0.0
    scheduled = compliance_command(
        nominal, virtual, np.array([1000.0]), measured, None, True, 1500.0
    )
    assert np.array_equal(scheduled[1], scheduled[0])
    assert np.allclose(scheduled[2][:3, :3], np.eye(3) * 1500.0)


def test_missing_stiffness_uses_5000_and_the_short_axis_fallback():
    measured = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    eye = np.eye(4).reshape(1, 4, 4).copy()
    diff = _compare(eye, eye.copy(), None, measured, None, False, -1.0)
    assert max(diff.values()) == 0.0
    out = compliance_command(eye, eye.copy(), None, measured, None, False, -1.0)
    assert out[3][0] == 5000.0
    assert out[4][0] == 5000.0


def _floor_from_log(root) -> float | None:
    enabled = root.attrs.get("stiffness_floor_enabled", None)
    value = root.attrs.get("stiffness_floor_n_per_m", None)
    if enabled is False or value is None:
        return None
    return float(value)


def _horizons(root):
    names = [key for key in root.group_keys() if key.startswith("horizon_")]
    return sorted(names, key=lambda name: int(name.split("_", 1)[1]))


def compare_log(path: Path) -> dict:
    import zarr

    root = zarr.open_group(str(path), mode="r")
    floor = _floor_from_log(root)
    worst = {name: 0.0 for name in (
        "nominal_pose7", "scheduled_pose7", "stiffness_6x6n", "raw", "floored",
        "vs_log_scheduled", "vs_log_stiffness", "vs_log_floored",
    )}
    n_steps = 0
    for name in _horizons(root):
        action = root[name]["action"]
        nominal_pose = np.asarray(action["ts_nominal_targets_0"], dtype=np.float64)
        scheduled_log = np.asarray(action["ts_virtual_targets_0"], dtype=np.float64)
        logged_k = np.asarray(action["ts_stiffnesses_0"], dtype=np.float64)
        measured = np.asarray(root[name]["obs_inference"]["ts_pose_fb_0"], dtype=np.float64)
        n = nominal_pose.shape[0]
        n_steps += n
        # The log stores the scheduled pose. With the runner default this is the
        # virtual target, which is also the pose the stiffness axis is built from.
        nominal_se3 = _pose7_batch_to_se3(nominal_pose)
        virtual_se3 = _pose7_batch_to_se3(scheduled_log)
        if "stiffness_raw_0" in action:
            raw = np.asarray(action["stiffness_raw_0"], dtype=np.float64).reshape(-1)
        else:
            # Older GT logs stored the 6x6 but not the press-axis scalar. The
            # soft-axis gain is X^T K X because the assembly's S is a rotation.
            raw = np.empty(n, dtype=np.float64)
            measured_se3 = su.pose7_to_SE3(measured)
            tool_from_world = su.SE3_inv(measured_se3)
            for i in range(n):
                delta = (
                    tool_from_world @ virtual_se3[i] - tool_from_world @ nominal_se3[i]
                )[:3, 3]
                if np.linalg.norm(delta) < 0.001:
                    axis = np.array([1.0, 0.0, 0.0])
                else:
                    axis = delta / np.linalg.norm(delta)
                block = logged_k[:3, 6 * i : 6 * i + 3]
                raw[i] = float(axis @ block @ axis)
        diff = _compare(
            nominal_se3, virtual_se3, raw, measured, floor, False, -1.0
        )
        for key, value in diff.items():
            worst[key] = max(worst[key], value)
        new = compliance_command(
            nominal_se3, virtual_se3, raw, measured, floor, False, -1.0
        )
        worst["vs_log_scheduled"] = max(worst["vs_log_scheduled"], _max_abs(new[1], scheduled_log))
        worst["vs_log_stiffness"] = max(worst["vs_log_stiffness"], _max_abs(new[2], logged_k))
        if "stiffness_floored_0" in action:
            floored_log = np.asarray(action["stiffness_floored_0"], dtype=np.float64).reshape(-1)
            worst["vs_log_floored"] = max(
                worst["vs_log_floored"], _max_abs(new[4], floored_log)
            )
    worst["n_steps"] = n_steps
    worst["floor"] = floor
    worst["n_horizons"] = len(_horizons(root))
    return worst


@pytest.mark.skipif(not E40_LOG.is_dir(), reason="e40 control log is not on this machine")
def test_e40_log_matches_legacy_and_the_stored_command():
    report = compare_log(E40_LOG)
    assert report["n_steps"] > 0
    for key in ("nominal_pose7", "scheduled_pose7", "stiffness_6x6n", "raw", "floored"):
        assert report[key] == 0.0
    assert report["vs_log_scheduled"] < 1e-12
    assert report["vs_log_stiffness"] < 1e-9


def main() -> None:
    logs = [E40_LOG, GT_LOG]
    for path in logs:
        if not path.is_dir():
            print(f"missing {path}")
            continue
        report = compare_log(path)
        print(path.name)
        for key, value in report.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
