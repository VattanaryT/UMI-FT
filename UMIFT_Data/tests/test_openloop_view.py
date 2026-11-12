"""Open-loop export checks. Synthetic errors, r4vt label geometry, r1rel pose."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SCRIPTS = Path(__file__).resolve().parents[1] / "surface_frames_scripts"
ROOT = SCRIPTS.parents[1]
for entry in (SCRIPTS, ROOT, ROOT / "PyriteML", ROOT / "PyriteUtility"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import openloop_lib as ol  # noqa: E402
from export_openloop import query_seed  # noqa: E402

R4VT = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_world_fixb_tilt/acp_replay_buffer.zarr"
)
R1REL = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r1rel_world/acp_replay_buffer.zarr"
)
FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)


class FlatField:
    def query(self, p, with_curvature=False):
        p = np.atleast_2d(np.asarray(p, dtype=np.float64))
        n = len(p)
        R = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        return {"x_s": p.copy(), "R": R, "sd": p[:, 2].copy(), "normal": R[:, :, 2].copy()}


def _driver():
    path = ROOT / "PyriteEnvSuites" / "scripts" / "offline_compare_driver.py"
    spec = importlib.util.spec_from_file_location("offline_compare_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_equal_prediction_is_zero_error():
    R = Rotation.from_euler("xyz", [12.0, -7.0, 30.0], degrees=True).as_matrix()
    label = np.zeros((8, 27), dtype=np.float64)
    label[:, 0] = np.linspace(-0.2, 0.2, 8)
    label[:, 1] = 0.05
    label[:, 2] = -0.01
    label[:, 18] = 40.0
    label[:, 21:27] = ol.rot6d_of_rotation(R)
    pred = label.copy()
    force = np.array([1, 9, 2, 12, 0, 8, 3, 20], dtype=np.float64)
    bad = np.array([0, 0, 1, 1, 0, 0, 0, 1], dtype=bool)
    long = np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=bool)
    valid = np.ones(8, dtype=bool)
    table = ol.episode_error_table(pred, label, force, bad, long, valid)
    for dim, groups in table["mae"].items():
        for group, value in groups.items():
            assert value == 0.0, (dim, group, value)


def test_delta_r_is_geodesic_not_rot6d_difference():
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pred = np.zeros((1, 27))
    label = np.zeros((1, 27))
    pred[:, 21:27] = ol.rot6d_of_rotation(R)
    label[:, 21:27] = ol.rot6d_of_rotation(np.eye(3))
    ang = ol.delta_r_deg(pred, label)
    assert abs(float(ang[0]) - 90.0) < 1e-6
    l2 = float(np.linalg.norm(pred[0, 21:27] - label[0, 21:27]))
    assert abs(float(ang[0]) - l2) > 1.0
    driver = _driver()
    got = ol.geodesic_deg(R.reshape(1, 3, 3), np.eye(3).reshape(1, 3, 3))
    ref = driver.geodesic_deg(R.reshape(1, 3, 3), np.eye(3).reshape(1, 3, 3))
    assert np.allclose(got, ref)


def test_query_seed_matches_offline_driver():
    driver = _driver()
    for rgb in (0, 1, 17, 400):
        assert query_seed("r4vt", rgb) == driver.query_seed("r4vt", rgb)


def test_synthetic_r4vt_decode_matches_vt_normal_and_tool_rotation():
    runtime = ol.make_r4_runtime(FlatField(), dry_run=False)
    R_tool = Rotation.from_euler("xyz", [10.0, -20.0, 15.0], degrees=True).as_matrix()
    x_s = np.array([0.1, 0.2, 0.0])
    n = np.array([0.0, 0.0, 1.0])
    p_vt = x_s + np.array([0.03, -0.04, -0.02])
    delta_n = float(np.dot(p_vt - x_s, n))
    action = ol.r4vt_action_row([[0.1, -0.2]], [delta_n], ol.rot6d_of_rotation(R_tool))
    dec = ol.decode_r4vt(runtime, action, x_s, R_tool, np.array([0.0]))
    lhs = ol.normal_offset(dec["x_d"], dec["x_s"], dec["n"])
    rhs = np.sum((p_vt - x_s) * n)
    assert abs(float(lhs[0]) - rhs) <= ol.TOL_POS_M
    assert float(ol.tangential_norm(dec["x_d"] - dec["x_s"], dec["n"])[0]) <= ol.TOL_POS_M
    assert float(ol.geodesic_deg(dec["R_d"], R_tool.reshape(1, 3, 3))[0]) <= ol.TOL_ROT_DEG
    offset = ol.tangential_vt_offset(p_vt.reshape(1, 3), x_s.reshape(1, 3), n.reshape(1, 3))
    expect = float(np.hypot(0.03, 0.04))
    assert abs(offset["median_m"] - expect) < 1e-12
    assert abs(offset["max_m"] - expect) < 1e-12


def test_r1rel_command_reconstructs_tcp():
    """The odom r1rel command is the corrected TCP, so the round-trip is ts_pose_fb."""
    import zarr
    from PyriteConfig.tasks.umift.umift_type_conversions import (
        action21_postprocess,
        action21_to_action_sample,
    )
    from PyriteUtility.spatial_math import spatial_utilities as su

    root = zarr.open(str(R1REL), mode="r")["data"]["episode_0"]
    fb = np.asarray(root["ts_pose_fb_0"], dtype=np.float64)
    command = np.asarray(root["ts_pose_command_0"], dtype=np.float64)
    if not np.array_equal(fb, command):
        raise AssertionError(
            "r1rel ts_pose_command_0 is not ts_pose_fb_0. "
            "The relative action reconstructs ts_pose_command_0, and this test "
            "must compare against that pose instead of loosening the TCP tolerance. "
            f"max |command-fb|={np.max(np.abs(command - fb))}"
        )
    assert ol.R1REL_LABEL_POSE == "ts_pose_fb_0"
    i0 = 200
    horizon = 8
    pose7 = fb[i0 : i0 + horizon]
    pose9 = su.SE3_to_pose9(su.pose7_to_SE3(pose7))
    action = np.zeros((horizon, 21), dtype=np.float64)
    action[:, :9] = pose9
    action[:, 9:18] = pose9
    action[:, 18] = 1.0
    base = [su.pose7_to_SE3(fb[i0])]
    rel = action21_to_action_sample(action, np.zeros((0, 21)), [0], base)["sparse"]
    mats, _vt, _stiff, _eoat = action21_postprocess(rel, base, [0])
    recon = su.SE3_to_pose7(mats[0].reshape(-1, 4, 4))
    pos = np.linalg.norm(recon[:, :3] - fb[i0 : i0 + horizon, :3], axis=1)
    from fixb_labels import pose7_to_Rp

    R_hat, _p = pose7_to_Rp(recon)
    R_fb, _ = pose7_to_Rp(fb[i0 : i0 + horizon])
    ang = ol.geodesic_deg(R_hat, R_fb)
    assert float(np.max(pos)) <= ol.TOL_POS_M
    assert float(np.max(ang)) <= ol.TOL_ROT_DEG


def test_r4vt_labels_match_normal_offset_and_tool_rotation():
    import zarr
    from fixb_field import FixBMoldField
    from fixb_labels import pose7_to_Rp

    r4 = zarr.open(str(R4VT), mode="r")["data"]["episode_0"]
    rel = zarr.open(str(R1REL), mode="r")["data"]["episode_0"]
    x_s = np.asarray(r4["mold_x_s_0"], dtype=np.float64).reshape(-1, 3)
    n = np.asarray(r4["mold_normal_0"], dtype=np.float64).reshape(-1, 3)
    v_par = np.asarray(r4["fixb_v_par_0"], dtype=np.float64)
    delta_n = np.asarray(r4["fixb_delta_n_0"], dtype=np.float64).reshape(-1)
    dR = np.asarray(r4["fixb_delta_R_rot6d_0"], dtype=np.float64).reshape(-1, 6)
    R_tool, _p = pose7_to_Rp(np.asarray(r4["ts_pose_fb_0"]))
    _Rv, p_vt = pose7_to_Rp(np.asarray(rel["ts_pose_virtual_target_0"]))
    m = min(len(x_s), len(p_vt), len(delta_n))
    offset = ol.tangential_vt_offset(p_vt[:m], x_s[:m], n[:m])
    print(
        f"episode 0 tangential VT offset median={offset['median_m']:.6f} m "
        f"max={offset['max_m']:.6f} m"
    )
    runtime = ol.make_r4_runtime(FixBMoldField(FIELD), dry_run=False)
    idx = np.linspace(0, m - 1, 12).astype(int)
    normal_err = []
    parallel = []
    rot_err = []
    for i in idx:
        action = ol.r4vt_action_row(v_par[i : i + 1], delta_n[i : i + 1], dR[i])
        # Anchor on the label foot so R(x_s) is the frame ΔR was written in.
        dec = ol.decode_r4vt(runtime, action, x_s[i], R_tool[i], np.array([0.0]))
        lhs = float(ol.normal_offset(dec["x_d"], dec["x_s"], dec["n"])[0])
        rhs = float(np.sum((p_vt[i] - x_s[i]) * n[i]))
        normal_err.append(abs(lhs - rhs))
        parallel.append(float(ol.tangential_norm(dec["x_d"] - dec["x_s"], dec["n"])[0]))
        rot_err.append(float(ol.geodesic_deg(dec["R_d"], R_tool[i : i + 1])[0]))
    assert max(normal_err) <= ol.TOL_POS_M, max(normal_err)
    assert max(parallel) <= ol.TOL_POS_M, max(parallel)
    assert max(rot_err) <= ol.TOL_ROT_DEG, max(rot_err)


def test_viewer_reads_r4vt_export():
    from view_openloop import episode_payload, load_export

    export = Path("/mnt/data/umidata/ft/open_loop_exports/r4vt_e66520fae9ae")
    loaded = load_export(export)
    payload = episode_payload([loaded], 0)
    assert payload["split"] == "train"
    assert payload["episode"] == 0
    run = payload["runs"][0]
    assert run["q_i"][0] == 0
    assert len(run["q_i"]) == len(run["q_t"]) == len(run["ang"])
    assert run["error"]["mae"]["delta_R_deg"]["clean"] is not None
    assert run["error"]["n"]["bad_span"] == 0
    val = episode_payload([loaded], 9)
    assert val["split"] == "val"
    assert val["runs"][0]["error"]["n"]["valid"] > 0
