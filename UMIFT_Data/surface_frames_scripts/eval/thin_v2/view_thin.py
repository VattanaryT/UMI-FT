#!/usr/bin/env python3
"""Thin stringer in the existing open-loop viewer. Does not edit that file.

Imports view_openloop from the surface_frames worktree, writes an export
under eval/thin_v2/, and adds the D405 frame, AprilTag marks, and tilt a/b.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

SURF = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SURF_SCRIPTS = SURF / "UMIFT_Data/surface_frames_scripts"
CMP_SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [
    str(SURF_SCRIPTS),
    str(SURF / "PyriteEnvSuites/scripts"),
    str(SURF / "PyriteML"),
    str(SURF / "PyriteConfig"),
    str(SURF / "PyriteUtility"),
    str(SURF),
]

import view_openloop as viewer  # noqa: E402

sys.path[:0] = [
    str(CMP_SCRIPTS / "eval"),
    str(CMP_SCRIPTS),
    str(ROOT / "PyriteEnvSuites/scripts"),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT),
]

import torch  # noqa: E402
import zarr  # noqa: E402
from PIL import Image  # noqa: E402

import compare_thin as thin  # noqa: E402
import compare_r1rel as base  # noqa: E402
from heading_tilt import e_long_sequence, long_axis, pack_tilt  # noqa: E402
from run_policy_open_loop_val_export import load_policy_and_cfg, predict_sparse_action  # noqa: E402

HERE = Path(__file__).resolve().parent
EXPORT = HERE / "export"
TAG = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/"
    "remap_to_training/tag_traj/2026-11-19-15.49.52_ee_master.npz"
)
MESH = Path("/mnt/data/umidata/ft/held_out_mold/thin_stringer/field/fixb_field.npz")
H = 16
PORT = 8801


def _jpeg(rgb: np.ndarray, size: int, quality: int) -> str:
    image = Image.fromarray(np.asarray(rgb))
    if size != image.size[0]:
        image = image.resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _action(sample) -> np.ndarray:
    act = sample["sparse"] if isinstance(sample, dict) else sample
    act = np.asarray(act, dtype=np.float64)
    if act.ndim == 3:
        act = act[0]
    return act


def _pad(action: np.ndarray, kind: str) -> np.ndarray:
    out = np.zeros((action.shape[0], 27 if kind != "r1rel" else 21), dtype=np.float64)
    if kind == "r1rel":
        n = min(21, action.shape[1])
        out[:, :n] = action[:, :n]
        return out
    out[:, 0:3] = action[:, 0:3]
    if action.shape[1] > 3:
        out[:, 18] = action[:, 3]
    if action.shape[1] >= 10:
        out[:, 21:27] = action[:, 4:10]
    return out


def _load_demo():
    path = thin.OUT / "replay.zarr"
    ep = zarr.open(path, "r")["data"]["episode_0"]
    src = zarr.open(thin.SRC, "r")["data"]["episode_0"]
    robot_t = np.asarray(ep["robot_time_stamps_0"]).reshape(-1)
    rgb_t = np.asarray(ep["rgb_time_stamps_0"]).reshape(-1)
    pose = np.asarray(ep["ts_pose_fb_0"], dtype=np.float64)
    from fixb_labels import pose7_to_Rp

    tool_R, tool_p = pose7_to_Rp(pose)
    side = np.load(TAG)
    tag_t = np.asarray(side["timestamp_sec_unified"], dtype=np.float64).reshape(-1)
    n_tags = np.asarray(side["n_tags_visible"], dtype=np.int32).reshape(-1)
    if len(n_tags) != len(rgb_t) or np.max(np.abs(tag_t - rgb_t)) > 1e-3:
        raise SystemExit("AprilTag sidecar does not line up with rgb_time_stamps_0")
    return {
        "zarr": str(path),
        "mask": str(thin.OUT / "long_dt_empty.npz"),
        "robot_t": robot_t,
        "rgb_t": rgb_t,
        "rgb": np.asarray(src["rgb_0"]),
        "tool_p": tool_p,
        "tool_R": tool_R,
        "frame": np.asarray(ep["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3),
        "foot": np.asarray(ep["mold_x_s_0"], dtype=np.float64).reshape(-1, 3),
        "normal": np.asarray(ep["mold_normal_0"], dtype=np.float64).reshape(-1, 3),
        "wrench_t": np.asarray(ep["wrench_time_stamps_left_0"]).reshape(-1),
        "force": np.linalg.norm(np.asarray(ep["wrench_left_0"], dtype=np.float64)[:, :3], axis=1),
        "n_tags": n_tags,
        "t0": float(rgb_t[0]),
        "cutoff": float(rgb_t[0]) + 0.5 * float(rgb_t[-1] - rgb_t[0]),
    }


def _queries(dataset, demo):
    from diffusion_policy.common.sampler_umift import sparse_action_indices_from_rgb_timestamps

    rows = []
    for idx, (episode, rgb) in enumerate(dataset.sampler.indices):
        rgb = int(rgb)
        if demo["rgb_t"][rgb] >= demo["cutoff"]:
            continue
        index, _times = sparse_action_indices_from_rgb_timestamps(
            demo["robot_t"], demo["rgb_t"], rgb, 64, 1
        )
        index = np.asarray(index, dtype=np.int64)
        if len(index) < 64 or demo["robot_t"][index[-1]] >= demo["cutoff"]:
            continue
        rows.append((idx, rgb, index[:H]))
    if len(rows) < 2:
        raise SystemExit(f"only {len(rows)} thin queries")
    return rows


def _force_at(demo, index: np.ndarray) -> np.ndarray:
    t = demo["robot_t"][index]
    j = np.searchsorted(demo["wrench_t"], t)
    j = np.clip(j, 1, len(demo["wrench_t"]) - 1)
    left = np.abs(t - demo["wrench_t"][j - 1]) <= np.abs(t - demo["wrench_t"][j])
    j = np.where(left, j - 1, j)
    return demo["force"][j]


def _build_one(name, policy, dataset, kind, demo, rows, field):
    q = len(rows)
    dim = 21 if kind == "r1rel" else 27
    pred_a = np.zeros((q, H, dim))
    label_a = np.zeros((q, H, dim))
    pred_pos = np.zeros((q, H, 3))
    pred_R = np.zeros((q, H, 3, 3))
    foot = np.zeros((q, H, 3))
    normal = np.zeros((q, H, 3))
    rgb_idx = np.zeros(q, dtype=np.int64)
    q_time = np.zeros(q)
    robot_idx = np.zeros((q, H), dtype=np.int64)
    force = np.zeros((q, H))
    ab = np.zeros((q, 4))
    for i, (idx, rgb, index) in enumerate(rows):
        obs, sample = dataset.sampler.sample_sequence(idx)
        pred = np.asarray(predict_sparse_action(policy, obs, demo["device"]), dtype=np.float64)
        pred = pred.reshape(-1, pred.shape[-1])[:H]
        label = _action(sample)[:H]
        pred_a[i] = _pad(pred, kind)
        label_a[i] = _pad(label, kind)
        xyz, rot = thin.pose_at(field, pred, index, demo, kind if kind in ("r1rel", "heading") else "field")
        n = min(H, len(xyz))
        pred_pos[i, :n] = xyz[:n]
        pred_R[i, :n] = rot[:n]
        foot[i] = demo["foot"][index]
        normal[i] = demo["normal"][index]
        rgb_idx[i] = rgb
        q_time[i] = demo["rgb_t"][rgb]
        robot_idx[i] = index
        force[i] = _force_at(demo, index)
        if kind == "heading":
            ab[i, 0:2] = np.degrees(pred[0, 6:8])
            packed = pack_tilt(
                demo["tool_R"][index[0]][None],
                demo["normal"][index[0]][None],
                e_long_sequence(demo["normal"][index[0]][None], demo["axis"])[0],
            )[0]
            ab[i, 2:4] = np.degrees(packed[2:4])
        if (i + 1) % 10 == 0:
            print(f"[thin-view] {name} {i + 1}/{q}", flush=True)
    folder = EXPORT / name
    folder.mkdir(parents=True, exist_ok=True)
    pose = np.asarray(zarr.open(demo["zarr"], "r")["data"]["episode_0"]["ts_pose_fb_0"])
    from fixb_labels import pose7_to_Rp

    tcp_R, tcp = pose7_to_Rp(np.asarray(pose, dtype=np.float64))
    np.savez(
        folder / "episode_0.npz",
        split=np.asarray("heldout"),
        query_rgb_idx=rgb_idx,
        query_time_s=q_time,
        n_valid=np.full(q, H, dtype=np.int64),
        pred_action=pred_a,
        label_action=label_a,
        pred_pos=pred_pos,
        pred_R=pred_R,
        raw_pos=pred_pos,
        raw_R=pred_R,
        raw_foot=foot,
        raw_n=normal,
        chunk_pos=pred_pos,
        chunk_R=pred_R,
        chunk_n=np.full(q, H, dtype=np.int64),
        action_time_s=demo["robot_t"][robot_idx],
        action_robot_idx=robot_idx,
        bad_span=np.zeros((q, H), dtype=bool),
        long_dt=np.zeros((q, H), dtype=bool),
        force_n=force,
        tcp_xyz=tcp,
        tcp_R=tcp_R,
        robot_time_s=demo["robot_t"],
        rgb_time_s=demo["rgb_t"],
        force_series_t=demo["wrench_t"],
        force_series=demo["force"],
        bad_robot=np.zeros(len(demo["robot_t"]), dtype=bool),
    )
    epoch = {"causal": 199, "r1rel": 25, "heading": 25}[name]
    (folder / "settings.json").write_text(json.dumps({
        "role": "r1rel" if kind == "r1rel" else "r4vt",
        "dataset": demo["zarr"],
        "execution_horizon": H,
        "query_stride": 4,
        "checkpoint_epoch": epoch,
        "episodes": [0],
    }))
    return ab


def build(demo, field) -> None:
    if (EXPORT / "causal" / "episode_0.npz").is_file() and (EXPORT / "ab.json").is_file():
        print("[thin-view] export exists", flush=True)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demo["device"] = device
    demo["axis"] = long_axis(field.vertices)
    causal_path, causal_policy, causal_cfg, _err = base.load_causal(device)
    r1_policy, r1_cfg = load_policy_and_cfg(str(base.R1REL_CKPT), device)
    head_policy, head_cfg = load_policy_and_cfg(str(thin.HEAD_CKPT), device)
    specs = {
        "causal": (causal_policy, causal_cfg, base.CAUSAL_NORM, "14308b72", 10, "causal"),
        "r1rel": (r1_policy, r1_cfg, base.R1REL_NORM, "2c20fba56851", 21, "r1rel"),
        "heading": (head_policy, head_cfg, thin.HEAD_NORM, "e93b8756", 8, "heading"),
    }
    rows = None
    loc = {}
    datasets = {}
    for name, (policy, cfg, norm, sha, dim, kind) in specs.items():
        datasets[name] = thin.build_dataset(cfg, demo["zarr"], norm, sha, dim, demo["mask"])
        if rows is None:
            rows = _queries(datasets[name], demo)
            print(f"[thin-view] queries {len(rows)} from {causal_path.name}", flush=True)
        loc[name] = {int(rgb): i for i, (_ep, rgb) in enumerate(datasets[name].sampler.indices)}
    shared = []
    for idx, rgb, index in rows:
        if all(rgb in loc[name] for name in specs):
            shared.append((rgb, index, {name: loc[name][rgb] for name in specs}))
    print(f"[thin-view] shared {len(shared)}", flush=True)
    ab = None
    for name, (policy, _cfg, _norm, _sha, _dim, kind) in specs.items():
        use = [(hit[name], rgb, index) for rgb, index, hit in shared]
        got = _build_one(name, policy, datasets[name], kind, demo, use, field)
        if name == "heading":
            ab = got
    (EXPORT / "ab.json").write_text(json.dumps({
        "a_p": ab[:, 0].round(2).tolist(),
        "b_p": ab[:, 1].round(2).tolist(),
        "a_l": ab[:, 2].round(2).tolist(),
        "b_l": ab[:, 3].round(2).tolist(),
    }))


def _images(demo) -> dict:
    small = [_jpeg(frame, 56, 35) for frame in demo["rgb"]]
    return {
        "rgb_t": [round(float(t - demo["t0"]), 3) for t in demo["rgb_t"]],
        "jpg": small,
        "tag": [int(v) for v in demo["n_tags"]],
    }


_FULL = {}


def _patch(images: dict) -> None:
    page = viewer.PAGE
    old_style = "#view3d { height:42vh; }"
    new_style = old_style + """
#wrist { display:flex; gap:8px; align-items:flex-start; padding:4px 12px; }
#wristImg { width:224px; height:224px; background:#000; }
#strip { display:flex; gap:2px; overflow-x:auto; max-width:calc(100vw - 260px); }
#strip img { height:48px; border:2px solid #333; }
#strip img.tag { border-color:#3ddc97; }
#strip img.on { border-color:#ffd23f; }
#tilt { height:18vh; }
"""
    page = page.replace(old_style, new_style, 1)
    page = page.replace(
        '<div id="view3d"></div>',
        '<div id="wrist"><img id="wristImg"><div><div id="wristCap"></div><div id="strip"></div></div></div>'
        '<div id="tilt"></div><div id="view3d"></div>',
        1,
    )
    page = page.replace(
        'applyCamera(cam);',
        'applyCamera(cam);\n  paintWrist(t);',
        1,
    )
    page = page.replace(
        "</script>",
        r"""
function paintWrist(t) {
  const img = document.getElementById("wristImg");
  const cap = document.getElementById("wristCap");
  const strip = document.getElementById("strip");
  if (!data || !data.jpg || !data.jpg.length) return;
  let i = 0;
  let best = 1e9;
  data.rgb_t.forEach((rt, k) => { const d = Math.abs(rt - t); if (d < best) { best = d; i = k; } });
  img.src = "/api/rgb?i=" + i;
  const n = data.tag[i] || 0;
  cap.textContent = data.rgb_t[i].toFixed(2) + " s  " + (n > 0 ? ("AprilTag " + n) : "no AprilTag");
  if (strip._n !== data.jpg.length) {
    strip.innerHTML = data.jpg.map((src, k) =>
      "<img data-k='" + k + "' src='" + src + "'" + (data.tag[k] > 0 ? " class='tag'" : "") + ">").join("");
    strip._n = data.jpg.length;
    strip.onclick = (ev) => {
      const k = ev.target && ev.target.dataset ? +ev.target.dataset.k : NaN;
      if (!Number.isFinite(k) || !data.path_t) return;
      fi = nearest(data.path_t, data.rgb_t[k]);
      document.getElementById("time").value = fi;
      paintCursor();
    };
  }
  strip.querySelectorAll("img").forEach((el, k) => el.classList.toggle("on", k === i));
  const run = data.runs.find(r => r.a_p);
  const box = document.getElementById("tilt");
  if (!run || !run.a_p) { box.style.display = "none"; return; }
  box.style.display = "block";
  const mark = {type: "line", xref: "x", yref: "y domain", x0: t, x1: t, y0: 0, y1: 1, line: {color: "#fff", width: 1}};
  Plotly.react("tilt", [
    {x: run.q_t, y: run.a_l, name: "a label", mode: "lines", line: {color: "#4aa3ff"}},
    {x: run.q_t, y: run.a_p, name: "a pred", mode: "lines", line: {color: "#ff4d6a", dash: "dash"}},
    {x: run.q_t, y: run.b_l, name: "b label", mode: "lines", xaxis: "x2", yaxis: "y2", line: {color: "#3ddc97"}},
    {x: run.q_t, y: run.b_p, name: "b pred", mode: "lines", xaxis: "x2", yaxis: "y2", line: {color: "#ffd23f", dash: "dash"}}
  ], {
    paper_bgcolor: "#111", plot_bgcolor: "#1c1c1c", font: {color: "#ddd"},
    margin: {l: 48, r: 12, t: 18, b: 24}, legend: {orientation: "h", y: 1.2},
    shapes: [mark, Object.assign({}, mark, {xref: "x2", yref: "y2 domain"})],
    xaxis: {anchor: "y", domain: [0, 1]},
    yaxis: {title: "a deg", domain: [0.55, 1]},
    xaxis2: {anchor: "y2", matches: "x", domain: [0, 1]},
    yaxis2: {title: "b deg", domain: [0, 0.45]}
  }, {responsive: true});
}
</script>""",
        1,
    )
    if page == viewer.PAGE or "paintWrist" not in page:
        raise SystemExit("viewer page patch did not apply")
    viewer.PAGE = page
    original = viewer.episode_payload

    def episode_payload(*args, **kwargs):
        out = original(*args, **kwargs)
        out.update(images)
        ab = json.loads((EXPORT / "ab.json").read_text())
        for run in out.get("runs") or []:
            if run.get("label", "").startswith("r4vt ep25"):
                run.update(ab)
        return out

    viewer.episode_payload = episode_payload

    class Handler(viewer.Handler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/rgb":
                query = self.path.split("?", 1)[1] if "?" in self.path else ""
                args = dict(part.split("=", 1) for part in query.split("&") if "=" in part)
                i = int(args.get("i", "0"))
                frame = _FULL.get(i)
                if frame is None:
                    self._send(404, b"", "text/plain")
                    return
                self._send(200, frame, "image/jpeg")
                return
            super().do_GET()

    viewer.Handler = Handler


def _cache_full(demo) -> None:
    for i, frame in enumerate(demo["rgb"]):
        image = Image.fromarray(np.asarray(frame))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=70)
        _FULL[i] = buf.getvalue()


def main() -> None:
    from fixb_field import FixBMoldField

    demo = _load_demo()
    field = FixBMoldField(MESH)
    build(demo, field)
    _cache_full(demo)
    _patch(_images(demo))
    # One payload build proves the export still opens in the untouched viewer.
    loaded = [viewer.load_export(EXPORT / name) for name in ("causal", "r1rel", "heading")]
    payload = viewer.episode_payload(loaded, 0)
    assert payload["runs"] and payload["jpg"] and any(payload["tag"])
    assert any(run.get("a_p") for run in payload["runs"])
    sys.argv = [
        "view_openloop.py",
        "--mesh", str(MESH),
        "--export", str(EXPORT / "causal"),
        "--export", str(EXPORT / "r1rel"),
        "--export", str(EXPORT / "heading"),
        "--host", "127.0.0.1",
        "--port", str(PORT),
    ]
    print(f"[thin-view] http://127.0.0.1:{PORT}", flush=True)
    viewer.main()


if __name__ == "__main__":
    main()
