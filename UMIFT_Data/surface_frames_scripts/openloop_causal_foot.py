"""Open-loop tool pose for the causal-foot 10-d action.

The foot walk and the tool pose both go through the runner on
``runner_causal_foot``: ``decode_chunk``, ``expand_r4_action``,
``rot6d_rows_to_R``, and ``_command_tangential``. A label action is the demo
TCP stepped one robot sample at a time, which is how those labels were built.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from diffusion_policy.common.sampler_umift import (
    get_val_mask,
    sparse_action_indices_from_rgb_timestamps,
)
from fixb_labels import pose7_to_Rp
from openloop_lib import REGION_NAMES, geodesic_deg, region_code

RUNNER_SCRIPTS = Path(
    "/home/boeing/umi_ft_ws/worktrees/UMI-FT-smooth_frame/UMIFT_Data/surface_frames_scripts"
)
DATA_ROOT = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
CAUSAL_ZARR = DATA_ROOT / "r4vt_causal_foot" / "acp_replay_buffer.zarr"
TRACK_ZARR = DATA_ROOT / "r4vt_smooth_tcp_track_clean" / "acp_replay_buffer.zarr"
FIELD = DATA_ROOT / "shared" / "fixb_field.npz"
LONG_DT = DATA_ROOT / "shared" / "long_dt_gt_100ms.npz"
CAUSAL_NORM = (
    Path(__file__).resolve().parent
    / "assets"
    / "oct_v35_h4s_r4vt_action_normalizer_minmax_causal_foot.pkl"
)
CAUSAL_SHA = "14308b72d55600b4359dfa67d10aedab378e42ad26f2bbcac988eed4d97a1ac2"
TRACK_NORM = (
    RUNNER_SCRIPTS
    / "assets"
    / "oct_v35_h4s_r4vt_action_normalizer_minmax_track_clean.pkl"
)
TRACK_SHA = "29a48f69ff51807df631b1e350087d6991091cba9739f0be88b30e2c85cf3bd4"
TRACK_CKPT = Path(
    "/mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_track_clean"
    "/oct-v35-h4s-r4vt-track-clean_e100_bs64_valtopk_43351238/checkpoints/latest.ckpt"
)
EXPORT_ROOT = Path("/mnt/data/umidata/ft/open_loop_exports")
EXEC_H = 6
RGB_STRIDE = 8
DT_S = 1.0 / 15.0
VAL_SEED = 42
VAL_RATIO = 0.1
EXCLUDE = (35, 44)


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runner():
    """The runner module, under its own name so this checkout's copy stays put."""
    path = RUNNER_SCRIPTS / "mold_aware_runtime.py"
    name = "runner_causal_foot_runtime"
    if name in sys.modules:
        return sys.modules[name]
    inserted = str(RUNNER_SCRIPTS) not in sys.path
    if inserted:
        sys.path.insert(0, str(RUNNER_SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if inserted:
        sys.path.remove(str(RUNNER_SCRIPTS))
    return module


def held_out_episodes(zarr_path: Path = CAUSAL_ZARR) -> list[int]:
    """Seed-42 val split in zarr key order, then episodes 35 and 44 dropped."""
    import zarr

    keys = sorted(
        k for k in zarr.open(str(zarr_path), "r")["data"].keys() if str(k).startswith("episode_")
    )
    ids = [int(str(k).split("_")[-1]) for k in keys]
    mask = get_val_mask(len(ids), VAL_RATIO, VAL_SEED)
    return sorted(ep for ep, flag in zip(ids, mask) if flag and ep not in EXCLUDE)


def _long_dt() -> dict[int, set[int]]:
    blob = np.load(LONG_DT)
    out: dict[int, set[int]] = {}
    for episode, index in zip(blob["flagged_episode"], blob["flagged_index"]):
        out.setdefault(int(episode), set()).add(int(index))
    return out


def action10(v, depth, stiffness, delta_r, index) -> np.ndarray:
    index = np.asarray(index, dtype=np.int64)
    action = np.zeros((len(index), 10), dtype=np.float64)
    action[:, 0:2] = np.asarray(v, dtype=np.float64)[index]
    action[:, 2] = np.asarray(depth, dtype=np.float64).reshape(-1)[index]
    action[:, 3] = np.asarray(stiffness, dtype=np.float64).reshape(-1)[index]
    action[:, 4:10] = np.asarray(delta_r, dtype=np.float64)[index]
    return action


def tool_pose(runner, foot, frame, action, dt: float = DT_S) -> dict:
    """Runner tool pose on an already-known causal frame.

    ``decode_chunk`` does this after the foot step: expand a 10-d row, Gram-Schmidt
    the ΔR, ``R_d = R_field @ ΔR``, and ``x_d = _command_tangential``.
    """
    expanded = runner.expand_r4_action(np.asarray(action, dtype=np.float64))
    delta_r = runner.rot6d_rows_to_R(expanded[:, 21:27])
    frame = np.asarray(frame, dtype=np.float64).reshape(-1, 3, 3)
    foot = np.asarray(foot, dtype=np.float64).reshape(-1, 3)
    rotation = np.einsum("nij,njk->nik", frame, delta_r)
    v_par = expanded[:, 0:2]
    depth = expanded[:, 2]
    commanded = np.stack(
        [
            runner._command_tangential(foot[k], frame[k], v_par[k], depth[k], dt)
            for k in range(len(foot))
        ]
    )
    return {"x_d": commanded, "R_d": rotation, "delta_n": depth, "R_field": frame}


def make_runtime(runner):
    runtime = runner.MoldAwareRuntime.from_assets(
        "r4_world_fixb_tilt",
        FIELD,
        dry_run=False,
        limits=runner.SafetyLimits(),
    )
    if getattr(runtime.field, "_causal_npz", None) is None:
        raise RuntimeError("runner did not attach the causal foot")
    return runtime


def walk_label(runner, group) -> dict:
    """``decode_chunk`` along the demo TCP. One robot sample per step."""
    rotation, position = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
    velocity = np.asarray(group["fixb_v_par_0"], dtype=np.float64)
    depth = np.asarray(group["fixb_delta_n_0"], dtype=np.float64).reshape(-1)
    stiffness = np.asarray(group["stiffness_0"], dtype=np.float64).reshape(-1)
    delta_r = np.asarray(group["fixb_delta_R_rot6d_0"], dtype=np.float64)
    n = len(position)
    hop = np.zeros((n, 3), dtype=np.float64)
    hop[:-1] = np.diff(position, axis=0)
    runtime = make_runtime(runner)
    runtime.reset()
    x_d = np.zeros((n, 3), dtype=np.float64)
    r_d = np.zeros((n, 3, 3), dtype=np.float64)
    x_s = np.zeros((n, 3), dtype=np.float64)
    for start in range(0, n, 1000):
        stop = min(n, start + 1000)
        count = stop - start
        sl = slice(start, stop)
        runtime._tcp_step = hop[sl]
        decoded = runtime.decode_chunk(
            action10(velocity, depth, stiffness, delta_r, np.arange(start, stop)),
            position[start],
            rotation[start],
            np.full(count, DT_S),
            apply_safety=False,
        )
        x_d[sl] = decoded["x_d"]
        r_d[sl] = decoded["R_d"]
        x_s[sl] = decoded["x_s"]
    return {
        "p": position,
        "R": rotation,
        "v": velocity,
        "dn": depth,
        "stiffness": stiffness,
        "dR": delta_r,
        "foot": np.asarray(group["mold_x_s_0"], dtype=np.float64),
        "frame": np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3),
        "normal": np.asarray(group["mold_normal_0"], dtype=np.float64),
        "robot_t": np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1),
        "rgb_t": np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1),
        "bad": np.asarray(group["bad_span_0"], dtype=bool).reshape(-1),
        "wrench": np.asarray(group["wrench_left_0"], dtype=np.float64),
        "wrench_t": np.asarray(group["wrench_time_stamps_0"], dtype=np.float64).reshape(-1),
        "x_d": x_d,
        "R_d": r_d,
        "x_s": x_s,
    }


def query_indices(walked, episode: int, long_idx: set[int], horizon: int = EXEC_H) -> list[tuple[int, np.ndarray]]:
    """First ``horizon`` control steps of every ``RGB_STRIDE`` query, bad spans dropped."""
    del episode
    robot_t = walked["robot_t"]
    rgb_t = walked["rgb_t"]
    bad = walked["bad"]
    found = []
    for rgb_id in range(0, len(rgb_t), RGB_STRIDE):
        index, _times = sparse_action_indices_from_rgb_timestamps(
            robot_t, rgb_t, rgb_id, horizon, 1
        )
        if len(index) < horizon:
            break
        index = np.asarray(index[:horizon], dtype=np.int64)
        if np.any(bad[index]) or any(int(i) in long_idx for i in index):
            continue
        found.append((int(rgb_id), index))
    return found


def _axis_deg(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12, None)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def _mode(delta_r_rot6d, runner) -> np.ndarray:
    rotation = runner.rot6d_rows_to_R(np.asarray(delta_r_rot6d, dtype=np.float64))
    identity = np.broadcast_to(np.eye(3), rotation.shape)
    angle = geodesic_deg(rotation, identity)
    mode = np.full(angle.shape, "mid", dtype=object)
    mode[angle < 45.0] = "near_identity"
    mode[angle > 135.0] = "return_yaw"
    return mode


def errors_against_label(runner, walked, index, pred_action=None, pose_source=None) -> dict:
    """Tool-pose error of an action versus the label at robot ``index``.

    The default action is the label itself. ``pred_action`` is (H, 10) or (H, 27)
    and is posed on ``pose_source`` when given, otherwise on the walked frame.
    Region and ΔR mode stay on ``walked``, so two models share the same bins.
    """
    index = np.asarray(index, dtype=np.int64)
    source = walked if pose_source is None else pose_source
    if pred_action is None:
        x_d = walked["x_d"][index]
        r_d = walked["R_d"][index]
        depth = walked["dn"][index]
    else:
        posed = tool_pose(
            runner, source["foot"][index], source["frame"][index], pred_action
        )
        x_d = posed["x_d"]
        r_d = posed["R_d"]
        depth = posed["delta_n"]
    label_cmd = np.stack(
        [
            runner._command_tangential(
                source["foot"][i], source["frame"][i], source["v"][i, :2], source["dn"][i], DT_S
            )
            for i in index
        ]
    )
    return {
        "position_mm": np.linalg.norm(x_d - label_cmd, axis=1) * 1e3,
        "rotation_deg": geodesic_deg(r_d, walked["R"][index]),
        "delta_n_mm": np.abs(depth - source["dn"][index]) * 1e3,
        "tilt_deg": _axis_deg(r_d[:, :, 2], walked["R"][index][:, :, 2]),
        "region": region_code(walked["normal"][index]),
        "mode": _mode(walked["dR"][index], runner),
        "x_d": x_d,
        "R_d": r_d,
    }


def _pair(values) -> dict | None:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def summarize(rows: list[dict]) -> dict:
    keys = ("position_mm", "rotation_deg", "delta_n_mm", "tilt_deg")
    if not rows:
        return {"all": {key: None for key in keys}}
    region = np.concatenate([row["region"] for row in rows])
    mode = np.concatenate([row["mode"] for row in rows])
    packed = {key: np.concatenate([row[key] for row in rows]) for key in keys}
    groups = {"all": np.ones(region.shape, dtype=bool)}
    for code, name in enumerate(REGION_NAMES):
        groups[name] = region == code
    for name in ("near_identity", "return_yaw"):
        groups[name] = mode == name
    return {
        name: {key: _pair(values[mask]) for key, values in packed.items()}
        for name, mask in groups.items()
    }


def score_label_episode(runner, group, episode: int, long_idx: set[int]) -> dict:
    walked = walk_label(runner, group)
    queries = query_indices(walked, episode, long_idx)
    rows = [errors_against_label(runner, walked, index) for _rgb, index in queries]
    return {"walked": walked, "queries": queries, "rows": rows, "summary": summarize(rows)}


def score_labels(episodes: list[int] | None = None) -> dict:
    if file_sha256(CAUSAL_NORM) != CAUSAL_SHA:
        raise RuntimeError(f"causal normalizer sha is not {CAUSAL_SHA}")
    import zarr

    root = zarr.open(str(CAUSAL_ZARR), "r")["data"]
    if episodes is None:
        episodes = held_out_episodes()
    runner = load_runner()
    long = _long_dt()
    per_episode = {}
    rows = []
    for episode in episodes:
        print(f"[causal-foot] label episode {episode}", flush=True)
        scored = score_label_episode(runner, root[f"episode_{episode}"], episode, long.get(episode, set()))
        per_episode[episode] = scored
        rows.extend(scored["rows"])
    return {"episodes": episodes, "per_episode": per_episode, "summary": summarize(rows)}


def format_table(summary: dict) -> str:
    lines = [
        f"{'split':<16} {'n':>6} {'pos mm':>16} {'rot °':>16} {'δn mm':>16} {'tilt °':>16}"
    ]
    order = ("all",) + REGION_NAMES + ("near_identity", "return_yaw")
    for name in order:
        block = summary.get(name) or {}
        cells = []
        count = ""
        for key in ("position_mm", "rotation_deg", "delta_n_mm", "tilt_deg"):
            pair = block.get(key)
            if not pair:
                cells.append(f"{'—':>16}")
                continue
            count = str(pair["n"])
            cells.append(f"{pair['median']:.3g} / {pair['p90']:.3g}".rjust(16))
        lines.append(f"{name:<16} {count:>6} {''.join(cells)}")
    return "\n".join(lines)
