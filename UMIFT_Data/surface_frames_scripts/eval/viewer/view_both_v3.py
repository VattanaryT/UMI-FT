#!/usr/bin/env python3
"""v2 layout on port 8804. Errors come from score_sign_flip.pose_error.

Does not edit view_both.py or view_both_v2.py. Ports 8802, 8803, and 8766 stay up.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

VIEW = Path(
    "/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/"
    "UMIFT_Data/surface_frames_scripts/eval/viewer"
)
sys.path.insert(0, str(VIEW))

import view_both as both  # noqa: E402
import view_both_v2 as v2  # noqa: E402
import view_openloop as viewer  # noqa: E402

TILT_EVAL = Path(
    "/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/"
    "UMIFT_Data/surface_frames_scripts/eval"
)
sys.path.insert(0, str(TILT_EVAL))
import score_sign_flip as scorer  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from http.server import ThreadingHTTPServer  # noqa: E402

both.PORT = 8804
COLORS = ("#ff4d6a", "#3ddc97")
FIELD_V2 = scorer.TRAIN / "oct_v35_h4s_r4vt_field_v2_k3000" / "checkpoints"
CAUSAL_E199 = (
    scorer.TRAIN
    / "oct_v35_h4s_r4vt_causal_foot"
    / "oct-v35-h4s-r4vt-causal-foot_e200_bs64_valtopk_43390127/checkpoints/latest.ckpt"
)
_LOCK = threading.Lock()
_THIN = None
_TRAIN_AXIS = None
_POLICY = {}
_DS = {}
_VAL = {}


def _models():
    """id, label, kind, path, json run, json which."""
    k = scorer.TRAIN / "oct-v35-h4s-r4vt-causal-foot-carry-k3000_e200_bs64_valtopk_43472139/checkpoints"
    t = scorer.TRAIN / "oct-v35-h4s-r4vt-tilt1_e200_bs64_valtopk_43467749/checkpoints"
    rows = [
        ("score:k3000-e25", "k3000-carry e25", "causal", k / "epoch=0025-val_loss=0.048.ckpt", "k3000-carry", "e25"),
        ("score:k3000-e199", "k3000-carry e199", "causal", k / "e199_final.ckpt", "k3000-carry", "latest"),
        ("score:tilt1-e25", "tilt1 e25", "tilt1", t / "epoch=0025-val_loss=0.064.ckpt", "tilt1", "e25"),
        ("score:tilt1-e180", "tilt1 e180", "tilt1", t / "latest_pulled.ckpt", "tilt1", "latest"),
        ("score:r1rel-e25", "r1rel e25", "r1rel", scorer.TRAIN / "oct_v35_h4s_odom_r1rel_valtopk/checkpoints/epoch=0025-val_loss=0.032.ckpt", "r1rel", "e25"),
        ("score:r1rel-e195", "r1rel e195", "r1rel", scorer.TRAIN / "oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt", "r1rel", "e195"),
        ("score:causal-e199", "causal e199", "causal-e199", CAUSAL_E199, None, None),
    ]
    picked = _field_v2()
    if not picked:
        rows.append(("score:fieldv2-missing", "field_v2 (not pulled)", None, None, None, None))
    for which, path in picked:
        rows.append((f"score:fieldv2-{which}", f"field_v2 {which}", "causal", path, None, None))
    return rows


def _field_v2():
    if not FIELD_V2.is_dir():
        return []
    files = [p for p in FIELD_V2.glob("*.ckpt") if p.is_file() and p.stat().st_size > 0]
    best, latest = None, None
    best_loss, latest_epoch = None, -1
    for path in files:
        name = path.name
        if "val_loss=" in name:
            loss = float(name.split("val_loss=")[1].split(".ckpt")[0])
            if best_loss is None or loss < best_loss:
                best_loss, best = loss, path
        epoch = -1
        if "epoch=" in name:
            epoch = int(name.split("epoch=")[1].split("-")[0])
        if epoch > latest_epoch:
            latest_epoch, latest = epoch, path
        if name in ("latest.ckpt", "latest_pulled.ckpt") or name.endswith("_final.ckpt"):
            latest = path
    out = []
    if best is not None:
        out.append(("best-val", best))
    if latest is not None and latest != best:
        out.append(("latest", latest))
    return out


def _by_id(ckpt_id):
    for row in _models():
        if row[0] == ckpt_id:
            return row
    return None


def _thin():
    global _THIN
    if _THIN is None:
        from fixb_field import FixBMoldField

        field = FixBMoldField(str(scorer.heldout.WALL))
        recordings, axis = scorer._thin(field)
        for rec in recordings.values():
            rec["frame_on"] = np.array(rec["demo"]["frame"], copy=True)
            scorer.sign_flip._flip_demo(rec["demo"])
            rec["frame_off"] = np.array(rec["demo"]["frame"], copy=True)
            rec["demo"]["frame"] = rec["frame_on"]
        _THIN = (recordings, axis)
    return _THIN


def _train_axis():
    global _TRAIN_AXIS
    if _TRAIN_AXIS is None:
        _TRAIN_AXIS = scorer.long_axis(np.load(scorer.sign_flip.TRAIN_FIELD)["vertices"])
    return _TRAIN_AXIS


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _policy(spec):
    key, _label, kind, path, _run, _which = spec
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f"not on disk: {_label}")
    if key not in _POLICY:
        policy, cfg = scorer.load_policy_and_cfg(str(path), _device())
        if kind == "causal-e199":
            try:
                found = scorer.heldout._norm_of(cfg, "causal")
            except SystemExit as exc:
                raise RuntimeError(str(exc)) from exc
            if found is None:
                raise RuntimeError("causal e199 normalizer rejected")
            norm, kind = found[0], "causal"
        else:
            norm = scorer._norm(cfg, kind)
        _POLICY[key] = (policy, cfg, norm, kind)
    return _POLICY[key]


def _bind(rec, flip):
    rec["demo"]["frame"] = rec["frame_on"] if flip else rec["frame_off"]
    rec["demo"].pop("_heading", None)
    return rec["demo"]


def _rows_for(spec, letter, flip):
    recordings, _axis = _thin()
    rec = recordings[letter]
    demo = _bind(rec, flip)
    policy, cfg, norm, kind = _policy(spec)
    cache_key = (spec[0], letter, bool(flip))
    if cache_key not in _DS:
        if kind == "tilt1" and flip:
            dataset, rows = rec["causal_ds"], rec["rows"]
        else:
            dataset = scorer.heldout._dataset(cfg, rec["path"], norm)
            if flip:
                scorer.sign_flip._flip_obs(dataset)
            rows = scorer._filter(scorer.heldout._queries(dataset, demo), rec["kept_i0"])
        _DS[cache_key] = (dataset, rows, kind)
    _dataset, rows, kind = _DS[cache_key]
    return demo, _dataset, rows, kind, _thin()[1]


def _train_bundle(spec):
    key = spec[0]
    if key not in _VAL:
        policy, cfg, norm, kind = _policy(spec)
        val, zarr_path = scorer._train_val(cfg, norm)
        demos, grouped, _n = scorer._train_rows(val, zarr_path)
        flat = [(epi, row) for epi, rows in grouped.items() for row in rows]
        _VAL[key] = (val, demos, flat, kind)
    return _VAL[key]


def _predict(policy, dataset, idx, kind):
    obs, _sample = dataset.sampler.sample_sequence(idx)
    pred = np.asarray(scorer.predict_sparse_action(policy, obs, _device()), dtype=np.float64)
    pred = pred.reshape(-1, pred.shape[-1])[: scorer.heldout.H]
    width = 5 if kind == "tilt1" else (21 if kind == "r1rel" else 10)
    if kind == "r1rel" and pred.shape[1] not in (21, 27):
        raise RuntimeError(f"r1rel action {pred.shape}")
    if kind != "r1rel" and pred.shape[1] != width:
        raise RuntimeError(f"{kind} action {pred.shape}")
    return pred


def _decode(kind, pred, demo, index, axis):
    """Same call the table's _score makes."""
    return scorer.pose_error(kind, pred, demo, index, axis)


def _nearest(rows, demo, t_rel):
    t_abs = float(t_rel) + float(demo["rgb_t"][0])
    i0 = np.array([int(row[2][0]) for row in rows], dtype=np.int64)
    return int(np.argmin(np.abs(demo["robot_t"][i0] - t_abs)))


def _json_step1(run, which, split):
    if not scorer.OUT.is_file() or run is None:
        return None
    for row in json.loads(scorer.OUT.read_text()):
        if row.get("run") == run and row.get("which") == which and split in row:
            cell = row[split]
            return {"pos": cell["pos_mm"]["1"], "rot": cell["rot_deg"]["1"], "n": cell["n"]}
    return None


def score_query(ckpt_id, mold, episode, q, t, flip, ckpt2=""):
    """One or two checkpoints. Step arrays are pose_error, not a second formula."""
    with _LOCK:
        return _score_query(ckpt_id, mold, episode, q, t, flip, ckpt2)


def _score_query(ckpt_id, mold, episode, q, t, flip, ckpt2):
    flip = bool(int(flip)) and mold == "thin"
    specs = []
    for ckpt in (ckpt_id, ckpt2):
        if not ckpt or not str(ckpt).startswith("score:"):
            continue
        spec = _by_id(ckpt)
        if spec is None:
            raise KeyError(ckpt)
        specs.append(spec)
    if not specs:
        raise RuntimeError("pick a score checkpoint")
    letter = {0: "C", 1: "A"}.get(int(episode))
    runs = []
    meta = None
    for spec in specs:
        policy, _cfg, _norm, _stored = _policy(spec)
        if mold == "thin":
            if letter is None:
                raise RuntimeError("sign-flip scores use thin episode C or A")
            demo, dataset, rows, kind, axis = _rows_for(spec, letter, flip)
            if not rows:
                raise RuntimeError(f"no kept {letter} rows")
            slot = int(q) if q is not None else _nearest(rows, demo, t or 0.0)
            slot = int(np.clip(slot, 0, len(rows) - 1))
            idx, _rgb, index = rows[slot]
            split = letter
        else:
            val, demos, flat, kind = _train_bundle(spec)
            if not flat:
                raise RuntimeError("no training windows")
            if q is not None:
                slot = int(np.clip(int(q), 0, len(flat) - 1))
            else:
                same = [i for i, (epi, _row) in enumerate(flat) if epi == int(episode)]
                slot = same[0] if same else 0
            epi, (idx, _rgb, index) = flat[slot]
            demo, dataset, axis = demos[epi], val, _train_axis()
            split = "train"
            letter = "train"
        pred = _predict(policy, dataset, idx, kind)
        pos, rot, xyz = _decode(kind, pred, demo, index, axis)
        if meta is None:
            meta = {
                "slot": slot,
                "i0": int(index[0]),
                "n": len(rows) if mold == "thin" else len(flat),
                "split": split,
                "label_xyz": np.asarray(demo["tool_p"], dtype=np.float64)[index],
                "kind": kind,
                "action": pred,
                "demo": demo,
                "index": index,
                "axis": axis,
            }
        runs.append({
            "name": spec[1],
            "color": COLORS[len(runs) % 2],
            "pos_mm": pos,
            "rot_deg": rot,
            "xyz": xyz,
            "json": _json_step1(spec[4], spec[5], split),
        })
    meta["flip"] = flip
    meta["runs"] = runs
    return meta


def _public(meta):
    def seq(values):
        return [round(float(v), 4) for v in np.asarray(values).reshape(-1)]

    def pts(xyz):
        return [[round(float(p), 5) for p in row] for row in np.asarray(xyz).reshape(-1, 3)]

    runs = []
    for run in meta["runs"]:
        runs.append({
            "name": run["name"],
            "color": run["color"],
            "pos_mm": seq(run["pos_mm"]),
            "rot_deg": seq(run["rot_deg"]),
            "xyz": pts(run["xyz"]),
            "json_step1": run["json"],
        })
    return {
        "flip": meta["flip"],
        "banner": "SIGN FLIP (thin) ON" if meta["flip"] else "",
        "slot": meta["slot"],
        "i0": meta["i0"],
        "n": meta["n"],
        "split": meta["split"],
        "label_xyz": pts(meta["label_xyz"]),
        "runs": runs,
    }


def smoke():
    """Three kept C queries. Viewer step-1 must equal pose_error on the same action."""
    ids = ("score:k3000-e199",)
    print("smoke k3000-carry e199 thin C flip=on", flush=True)
    for q in (0, 1, 2):
        meta = score_query(ids[0], "thin", 0, q, 0.0, 1, "")
        run = meta["runs"][0]
        pos, rot, _xyz = scorer.pose_error(
            meta["kind"], meta["action"], meta["demo"], meta["index"], meta["axis"]
        )
        if not (np.allclose(pos, run["pos_mm"]) and np.allclose(rot, run["rot_deg"])):
            raise SystemExit(f"q {q} viewer diverged from pose_error")
        table = run["json"]
        table_txt = "json n/a" if table is None else f"json mean {table['pos']:.1f} mm / {table['rot']:.1f} deg"
        print(
            f"q{q} i0 {meta['i0']} viewer {float(pos[0]):.4f} mm / {float(rot[0]):.4f} deg"
            f"  pose_error {float(run['pos_mm'][0]):.4f} mm / {float(run['rot_deg'][0]):.4f} deg"
            f"  MATCH  {table_txt}",
            flush=True,
        )
    print("smoke MATCH 3/3", flush=True)


_JS = r"""
function layoutV3() {
  const bar = document.getElementById("topbar");
  if (!bar || document.getElementById("signFlip")) return;
  const flip = document.createElement("label");
  flip.id = "signFlipLabel";
  flip.innerHTML = '<input id="signFlip" type="checkbox" checked> sign flip (thin)';
  const cmp = document.createElement("select");
  cmp.id = "ckpt2";
  cmp.innerHTML = '<option value="">compare: off</option>';
  const banner = document.createElement("span");
  banner.id = "flipBanner";
  banner.textContent = "SIGN FLIP (thin) ON";
  banner.style.cssText = "background:#ffb020;color:#111;font-weight:700;padding:2px 8px;";
  bar.append(flip, cmp, banner);
  const side = document.getElementById("side");
  if (side && !document.getElementById("pErr")) {
    side.insertAdjacentHTML("beforeend", '<div id="pErr" style="flex:1.2 1 0;min-height:80px"></div>');
  }
  document.getElementById("signFlip").onchange = () => paintScore();
  cmp.onchange = () => paintScore();
}
layoutV3();
let scoreCkpts = null;
let scoreSeq = 0;
async function addScoreCkpts(keepScore) {
  if (!scoreCkpts) scoreCkpts = await (await fetch("/api/score_ckpts")).json();
  const ckpt = document.getElementById("ckpt");
  scoreCkpts.forEach(c => {
    if ([...ckpt.options].some(o => o.value === c.id)) return;
    const opt = document.createElement("option");
    opt.value = c.id;
    opt.textContent = c.label;
    ckpt.appendChild(opt);
  });
  if (keepScore && [...ckpt.options].some(o => o.value === keepScore)) ckpt.value = keepScore;
  const cmp = document.getElementById("ckpt2");
  if (!cmp) return;
  const cur = cmp.value;
  cmp.innerHTML = '<option value="">compare: off</option>';
  scoreCkpts.forEach(c => {
    const opt = document.createElement("option");
    opt.value = c.id;
    opt.textContent = c.label;
    cmp.appendChild(opt);
  });
  if ([...cmp.options].some(o => o.value === cur)) cmp.value = cur;
}
function dropScoreTraces() {
  const gd = document.getElementById("view3d");
  if (!gd || !gd.data) return;
  const drop = [];
  gd.data.forEach((tr, i) => {
    if (tr && tr.name && String(tr.name).indexOf("score ") === 0) drop.push(i);
  });
  if (drop.length) Plotly.deleteTraces(gd, drop);
}
async function paintScore() {
  const seq = ++scoreSeq;
  layoutV3();
  const box = document.getElementById("signFlip");
  const on = !!(box && box.checked && mold === "thin");
  const banner = document.getElementById("flipBanner");
  if (banner) {
    banner.textContent = on ? "SIGN FLIP (thin) ON" : "";
    banner.style.background = on ? "#ffb020" : "transparent";
    banner.style.color = on ? "#111" : "#bbb";
  }
  if (!data) return;
  const ckpt = document.getElementById("ckpt").value || "";
  const ckpt2 = (document.getElementById("ckpt2") || {}).value || "";
  dropScoreTraces();
  if (ckpt.indexOf("score:") !== 0 && ckpt2.indexOf("score:") !== 0) {
    const host = document.getElementById("pErr");
    if (host) host.textContent = "";
    return;
  }
  const t = (data.path_t && data.path_t.length) ? data.path_t[Math.max(0, fi | 0)] : 0;
  const qs = new URLSearchParams({
    mold: mold, ep: String(data.episode), t: String(t),
    flip: (box && box.checked) ? "1" : "0", ckpt: ckpt, ckpt2: ckpt2
  });
  let body;
  try {
    body = await (await fetch("/api/score?" + qs.toString())).json();
  } catch (err) {
    return;
  }
  if (seq !== scoreSeq || !body) return;
  const host = document.getElementById("pErr");
  if (body.error) {
    if (host) host.textContent = body.error;
    return;
  }
  const steps = body.runs[0].pos_mm.map((_, i) => i + 1);
  const traces = [];
  body.runs.forEach(run => {
    traces.push({x: steps, y: run.pos_mm, name: run.name + " mm", mode: "lines", line: {color: run.color, width: 1.6}});
    traces.push({x: steps, y: run.rot_deg, name: run.name + " deg", mode: "lines", yaxis: "y2", line: {color: run.color, width: 1.4, dash: "dot"}});
  });
  const json1 = body.runs.map(run => {
    if (!run.json_step1) return run.name;
    return run.name + " json " + run.json_step1.pos + " mm / " + run.json_step1.rot + "°";
  }).join(" · ");
  const step1 = body.runs.map(run => run.name + " " + run.pos_mm[0].toFixed(1) + " mm / " + run.rot_deg[0].toFixed(1) + "°").join(" · ");
  Plotly.react("pErr", traces, {
    paper_bgcolor: "#111", plot_bgcolor: "#111", font: {color: "#ddd", size: 10},
    margin: {l: 36, r: 36, t: 36, b: 24}, legend: {orientation: "h", y: 1.15},
    title: {text: body.split + " q" + body.slot + " i0 " + body.i0 + "<br>" + step1 + "<br>" + json1, font: {size: 11}},
    xaxis: {title: "step", gridcolor: "#333"},
    yaxis: {title: "mm", gridcolor: "#333"},
    yaxis2: {title: "deg", overlaying: "y", side: "right", showgrid: false}
  }, {responsive: true, displayModeBar: false});
  const extra = [scatter(body.label_xyz, "score label", "#4aa3ff", 6)];
  body.runs.forEach(run => extra.push(scatter(run.xyz, "score " + run.name, run.color, 5)));
  dropScoreTraces();
  Plotly.addTraces("view3d", extra);
}
const _ckptChange = document.getElementById("ckpt").onchange;
document.getElementById("ckpt").onchange = () => {
  const v = document.getElementById("ckpt").value;
  if (v.indexOf("score:") === 0) { paintScore(); return; }
  if (_ckptChange) _ckptChange();
};
"""


def _patch_page():
    page = viewer.PAGE
    needle = 'ckpt.innerHTML = data.runs.map(r => "<option>" + r.label + "</option>").join("");'
    if needle not in page:
        raise SystemExit("v3 ckpt dropdown missed")
    page = page.replace(
        needle,
        'const keepScore = ckpt.value;\n  ' + needle + "\n  addScoreCkpts(keepScore);",
        1,
    )
    react = 'footTraces(traces, t);\n  await Plotly.react("view3d", traces, sceneLayout(), {responsive: true});'
    if react not in page:
        raise SystemExit("v3 view3d hook missed")
    page = page.replace(react, react + "\n  paintScore();", 1)
    if "</script>\n</body>" not in page:
        raise SystemExit("v3 script end missed")
    page = page.replace("</script>\n</body>", _JS + "\n</script>\n</body>", 1)
    viewer.PAGE = page


def _patch_handler():
    if getattr(viewer.Handler, "_v3", False):
        return
    original = viewer.Handler.do_GET

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/score_ckpts":
            body = [{"id": row[0], "label": row[1]} for row in _models()]
            self._send(200, json.dumps(body).encode(), "application/json")
            return
        if path == "/api/score":
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            args = dict(part.split("=", 1) for part in query.split("&") if "=" in part)
            from urllib.parse import unquote

            try:
                episode = int(args.get("ep", "0"))
                q = args.get("q")
                q = None if q is None or q == "" else int(q)
                meta = score_query(
                    unquote(args.get("ckpt", "")),
                    args.get("mold", "training"),
                    episode,
                    q,
                    float(args.get("t", "0") or 0),
                    args.get("flip", "0"),
                    unquote(args.get("ckpt2", "")),
                )
                payload = _public(meta)
            except Exception as exc:
                payload = {"error": str(exc)}
            self._send(200, json.dumps(payload).encode(), "application/json")
            return
        original(self)

    viewer.Handler.do_GET = do_GET
    viewer.Handler._v3 = True


_v2_init = ThreadingHTTPServer.__init__


def _init(self, server_address, RequestHandlerClass, bind_and_activate=True):
    _v2_init(self, server_address, RequestHandlerClass, bind_and_activate)
    _patch_page()
    _patch_handler()


ThreadingHTTPServer.__init__ = _init


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke()
    else:
        both.main()
