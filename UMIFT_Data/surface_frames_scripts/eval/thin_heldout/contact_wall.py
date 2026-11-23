#!/usr/bin/env python3
"""Contact-and-wall rescore. Closest-point foot, same as training.

Does not use foot_mode=aligned. Does not rewrite score.json or wall_only/.
Stops before the checkpoints if ground-truth step 1 is outside ~0–20 mm or ~0°.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "viewer"))

import numpy as np  # noqa: E402

import heldout  # noqa: E402
import input_check  # noqa: E402
import foot_overlay  # noqa: E402

import torch  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from heading_tilt import long_axis  # noqa: E402
from run_policy_open_loop_val_export import load_policy_and_cfg  # noqa: E402

import compare_r1rel as fair  # noqa: E402

OUT = HERE / "contact_wall"
TRAIN = Path("/mnt/data/umidata/ft/training_outputs")
FORCE_N = 8.0
# Ground truth on a wall contact should sit on the surface with the label rotation.
POS_MAX_MM = 20.0
ROT_MAX_DEG = 5.0
MODELS = (
    ("causal e199", "causal", TRAIN / "oct_v35_h4s_r4vt_causal_foot/oct-v35-h4s-r4vt-causal-foot_e200_bs64_valtopk_43390127/checkpoints/latest.ckpt"),
    ("heading e100", "heading", TRAIN / "oct-v35-h4s-r4vt-heading-tilt_sc_e200_bs64_valtopk_43448962/checkpoints/latest.ckpt"),
    ("r1rel e25", "r1rel", TRAIN / "oct_v35_h4s_odom_r1rel_valtopk/checkpoints/epoch=0025-val_loss=0.032.ckpt"),
    ("r1rel e195", "r1rel", TRAIN / "oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt"),
    ("robust-ab", "auto", TRAIN / "oct-v35-h4s-r4vt-heading-tilt_sc_robustab_e200_bs64_valtopk_43465423/checkpoints/epoch=0025-val_loss=0.073.ckpt"),
    ("robust-ab", "auto", TRAIN / "oct-v35-h4s-r4vt-heading-tilt_sc_robustab_e200_bs64_valtopk_43465423/checkpoints/latest.ckpt"),
    ("tilt1", "auto", TRAIN / "oct-v35-h4s-r4vt-tilt1_e200_bs64_valtopk_43467749/checkpoints/epoch=0025-val_loss=0.064.ckpt"),
    ("tilt1", "auto", TRAIN / "oct-v35-h4s-r4vt-tilt1_e200_bs64_valtopk_43467749/checkpoints/latest.ckpt"),
    ("k3000-carry", "auto", TRAIN / "oct-v35-h4s-r4vt-causal-foot-carry-k3000_e200_bs64_valtopk_43472139/checkpoints/epoch=0025-val_loss=0.048.ckpt"),
    ("k3000-carry", "auto", TRAIN / "oct-v35-h4s-r4vt-causal-foot-carry-k3000_e200_bs64_valtopk_43472139/checkpoints/latest.ckpt"),
)


def _check() -> None:
    keep, cap, back, off = _flags("wall", np.array([9.0, 8.0]))
    if not keep or cap or back or off:
        raise RuntimeError("wall contact was dropped")
    keep, cap, back, off = _flags("cap", np.array([9.0]))
    if keep or not cap:
        raise RuntimeError("cap contact was kept")
    keep, cap, back, off = _flags("back", np.array([9.0]))
    if keep or not back:
        raise RuntimeError("back contact was kept")
    keep, cap, back, off = _flags("wall", np.array([9.0, 7.0]))
    if keep or not off:
        raise RuntimeError("lift-off was kept")


def _flags(face: str, mag: np.ndarray):
    cap = face == "cap"
    back = face == "back"
    off = bool(np.any(np.asarray(mag) < FORCE_N))
    return (not cap and not back and not off), cap, back, off


def _epoch(path: Path):
    if "epoch=" in path.name:
        return int(path.name.split("epoch=")[1].split("-")[0])
    return fair.ckpt_epoch(path)


def _label(stem: str, path: Path, epoch) -> str:
    if stem in ("causal e199", "heading e100", "r1rel e25", "r1rel e195"):
        return stem
    text = f"{stem} e{epoch}"
    if path.name == "latest.ckpt":
        text += " latest"
    return text


def _recordings(field):
    axis = long_axis(field.vertices)
    t_mid = json.loads((heldout.SCRIPTS / "eval/thin_variants/contact_half.json").read_text())["episodes"]["C"]["t_mid_s"]
    out = {}
    for letter, path in (("C", HERE / "C_first" / "replay.zarr"), ("A", HERE / "A_full" / "replay.zarr")):
        opened = heldout._demo(path, None)
        cutoff = None if letter == "A" else float(opened["rgb_t"][0]) + float(t_mid)
        demo = opened if cutoff is None else heldout._demo(path, cutoff)
        demo["axis"] = axis
        out[letter] = {"path": path, "demo": demo, "episode": 0 if letter == "C" else 1}
    return out, axis


def _rows_for(cfg, norm, rec, cache):
    key = (str(rec["path"]), str(norm))
    if key not in cache:
        cache[key] = heldout._dataset(cfg, rec["path"], norm)
        cache[key + ("q",)] = heldout._queries(cache[key], rec["demo"])
    return cache[key], cache[key + ("q",)]


def _select(demo, rows):
    i0 = np.array([int(index[0]) for _idx, _rgb, index in rows], dtype=np.int64)
    _foot, face, _z = foot_overlay._project(*foot_overlay._load(), demo["tool_p"][i0])
    chosen, detail = [], []
    for row, kind in zip(rows, face):
        _idx, rgb, index = row
        j = heldout._nearest(demo["wrench_t"], demo["robot_t"][index])
        keep, cap, back, off = _flags(str(kind), np.linalg.norm(demo["smooth"][j], axis=1))
        chosen.append(row if keep else None)
        detail.append({
            "rgb": int(rgb),
            "t": round(float(demo["robot_t"][int(index[0])] - demo["rgb_t"][0]), 3),
            "keep": bool(keep),
            "cap": bool(cap),
            "back": bool(back),
            "liftoff": bool(off),
            "face": str(kind),
            "index": index,
        })
    kept = [row for row in chosen if row is not None]
    counts = {
        "n": len(rows),
        "kept": len(kept),
        "dropped": len(rows) - len(kept),
        "cap": int(sum(item["cap"] for item in detail)),
        "back": int(sum(item["back"] for item in detail)),
        "liftoff": int(sum(item["liftoff"] for item in detail)),
    }
    return kept, detail, counts


def _gt(field, demo, axis, detail):
    loaded = input_check._load(demo_path(demo))
    # The sliced demo and the zarr share one timeline. Labels live on the zarr.
    loaded["smooth"] = demo["smooth"]
    queries = [item["index"] for item in detail if item["keep"]]
    if not queries:
        return {"n": 0}
    return input_check._score_gt(field, loaded, axis, queries)


def demo_path(demo):
    return demo["_path"]


def _ok(gt) -> bool:
    if not gt.get("n"):
        return False
    pos = gt["pos"]["mean"]["1"]
    rot = gt["rot"]["mean"]["1"]
    head = gt["head"]["mean"]["1"]
    return pos <= POS_MAX_MM and rot <= ROT_MAX_DEG and head <= ROT_MAX_DEG


def main() -> None:
    _check()
    heldout._check()
    field = FixBMoldField(str(heldout.WALL))
    recordings, axis = _recordings(field)
    for rec in recordings.values():
        rec["demo"]["_path"] = rec["path"]
    # Query list from the causal checkpoint, then the same keep rule for every model.
    causal = MODELS[0][2]
    if not causal.is_file():
        raise SystemExit(f"missing {causal}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, cfg = load_policy_and_cfg(str(causal), device)
    resolved = heldout._norm_of(cfg, "causal")
    if resolved is None:
        raise SystemExit("causal normalizer missing")
    norm, _dim, decode, _sha = resolved
    cache = {}
    page = {}
    report_counts = {}
    gt = {}
    for letter, rec in recordings.items():
        _dataset, rows = _rows_for(cfg, norm, rec, cache)
        kept, detail, counts = _select(rec["demo"], rows)
        rec["kept"] = kept
        report_counts[letter] = counts
        page[str(rec["episode"])] = {
            "t": [item["t"] for item in detail],
            "keep": [item["keep"] for item in detail],
        }
        print(f"[contact] {letter} n {counts['n']} kept {counts['kept']} dropped {counts['dropped']}"
              f"  cap {counts['cap']}  back {counts['back']}  lift-off {counts['liftoff']}", flush=True)
        gt[letter] = _gt(field, rec["demo"], axis, detail)
        block = gt[letter]
        if block.get("n"):
            print(f"[contact] GT {letter} step-1 pos {block['pos']['mean']['1']} mm"
                  f"  rot {block['rot']['mean']['1']}°  heading {block['head']['mean']['1']}°", flush=True)
        else:
            print(f"[contact] GT {letter} no kept queries", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kept.json").write_text(json.dumps({"counts": report_counts, **page}))
    if not all(_ok(gt[letter]) for letter in ("C", "A")):
        (OUT / "gt.json").write_text(json.dumps({"counts": report_counts, "gt": gt}, indent=2))
        print("[contact] ground truth outside 0–20 mm or 0°. Not rescoring.", flush=True)
        return
    tables = {"C": [], "A": []}
    report = {"foot": "closest", "force_n": FORCE_N, "counts": report_counts, "gt": gt, "recordings": {}}
    pending = [(policy, cfg, MODELS[0])]
    for spec in MODELS[1:]:
        pending.append((None, None, spec))
    for policy, cfg, (stem, kind, path) in pending:
        if not path.is_file() or path.stat().st_size < heldout.MIN_CKPT:
            print(f"[contact] missing {path}", flush=True)
            continue
        if policy is None:
            policy, cfg = load_policy_and_cfg(str(path), device)
        epoch = _epoch(path)
        name = _label(stem, path, epoch)
        resolved = heldout._norm_of(cfg, kind)
        if resolved is None:
            print(f"[contact] skip {name}  unsupported normalizer", flush=True)
            del policy
            torch.cuda.empty_cache()
            continue
        norm, _dim, decode, sha = resolved
        scored = {}
        for letter, rec in recordings.items():
            dataset, rows = _rows_for(cfg, norm, rec, cache)
            kept_i0 = {int(index[0]) for _i, _r, index in rec["kept"]}
            rows = [row for row in rows if int(row[2][0]) in kept_i0]
            if len(rows) != len(rec["kept"]):
                raise RuntimeError(f"{name} {letter} kept {len(rows)} vs {len(rec['kept'])}")
            print(f"[contact] score {name} {letter} {len(rows)}", flush=True)
            block = heldout._score_one(decode, policy, dataset, field, rec["demo"], rows, device, axis)
            scored[letter] = block
        for letter in ("C", "A"):
            public = heldout._public(scored[letter])
            tables[letter].append((name, public))
            report["recordings"].setdefault(letter, {})[name] = {
                **public, "ckpt": str(path), "epoch": epoch, "decode": decode, "sha": sha[:12],
            }
        del policy
        torch.cuda.empty_cache()
    (OUT / "score.json").write_text(json.dumps(report, indent=2, default=_json))
    heldout._print_table("C  contact wall", tables["C"])
    heldout._print_table("A  contact wall", tables["A"])
    print(f"[contact] wrote {OUT}", flush=True)


def _json(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value))


if __name__ == "__main__":
    main()
