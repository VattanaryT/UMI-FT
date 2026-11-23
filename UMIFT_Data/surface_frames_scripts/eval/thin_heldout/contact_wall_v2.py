#!/usr/bin/env python3
"""PROVISIONAL. Query |F| >= 8 N and |F| >= 5 N on at least 80% of the horizon.

Drops cap and back feet. Writes contact_wall_v2/ only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import torch

import contact_wall as cw
import heldout
from fixb_field import FixBMoldField
from run_policy_open_loop_val_export import load_policy_and_cfg

QUERY_N = 8.0
FLOOR_N = 5.0
FRAC = 0.80
cw.OUT = cw.HERE / "contact_wall_v2"


def _flags(face: str, mag: np.ndarray):
    mag = np.asarray(mag, dtype=np.float64)
    cap = face == "cap"
    back = face == "back"
    off = not (float(mag[0]) >= QUERY_N and float(np.mean(mag >= FLOOR_N)) >= FRAC)
    return (not cap and not back and not off), cap, back, off


def _check() -> None:
    keep, cap, back, off = _flags("wall", np.array([9.0, 5.0, 5.0, 5.0, 4.0]))
    if not keep or cap or back or off:
        raise RuntimeError("80% floor was dropped")
    keep, cap, back, off = _flags("wall", np.array([9.0, 5.0, 5.0, 4.0, 4.0]))
    if keep or not off:
        raise RuntimeError("under 80% was kept")
    keep, cap, back, off = _flags("wall", np.array([7.0, 9.0, 9.0, 9.0]))
    if keep or not off:
        raise RuntimeError("query under 8 N was kept")
    keep, cap, back, off = _flags("cap", np.array([9.0, 9.0]))
    if keep or not cap:
        raise RuntimeError("cap was kept")
    keep, cap, back, off = _flags("back", np.array([9.0, 9.0]))
    if keep or not back:
        raise RuntimeError("back was kept")


cw._flags = _flags
cw._check = _check
_orig_table = heldout._print_table
_orig_write = Path.write_text


def _print_table(title, rows):
    _orig_table("PROVISIONAL  " + title, rows)


def _write_text(self, data, *args, **kwargs):
    if getattr(self, "parent", None) is not None and self.parent.name == "contact_wall_v2" and self.suffix == ".json":
        payload = json.loads(data)
        payload["label"] = "PROVISIONAL"
        payload["gate"] = "query |F_smooth|>=8 and |F_smooth|>=5 on >=80% of the horizon; drop cap and back"
        data = json.dumps(payload, indent=None if self.name == "kept.json" else 2)
    return _orig_write(self, data, *args, **kwargs)


heldout._print_table = _print_table
Path.write_text = _write_text


A_ONLY = {"heading e100", "robust-ab", "tilt1"}


def score() -> None:
    """Score the kept set. Heading and tilt models are skipped on C."""
    cw._check()
    heldout._check()
    field = FixBMoldField(str(heldout.WALL))
    recordings, axis = cw._recordings(field)
    for rec in recordings.values():
        rec["demo"]["_path"] = rec["path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = {}
    ready = None
    for stem, kind, path in cw.MODELS:
        if stem == "causal e199":
            policy, cfg = load_policy_and_cfg(str(path), device)
            resolved = heldout._norm_of(cfg, kind)
            if resolved is None:
                raise SystemExit("causal normalizer missing")
            ready = (policy, cfg, resolved[0])
            break
    if ready is None:
        raise SystemExit("causal checkpoint missing")
    _policy, cfg, norm = ready
    for letter, rec in recordings.items():
        _dataset, rows = cw._rows_for(cfg, norm, rec, cache)
        kept, _detail, counts = cw._select(rec["demo"], rows)
        rec["kept"] = kept
        print(f"[v2] {letter} kept {counts['kept']}", flush=True)
    if len(recordings["C"]["kept"]) != 9 or len(recordings["A"]["kept"]) != 94:
        raise SystemExit(
            f"kept set changed: C {len(recordings['C']['kept'])} A {len(recordings['A']['kept'])}"
        )
    gt = json.loads((cw.OUT / "gt.json").read_text())["gt"]
    heading_median = {
        "C": gt["C"]["head"]["median"]["1"],
        "A": gt["A"]["head"]["median"]["1"],
    }
    tables = {"C": [], "A": []}
    report = {
        "label": "PROVISIONAL",
        "kept": {"C": 9, "A": 94},
        "heading_gt_median_step1": heading_median,
        "skipped_on_C": "heading and tilt models; C heading GT median is "
        f"{heading_median['C']}°",
        "recordings": {},
    }
    pending = [(ready[0], ready[1], cw.MODELS[0])]
    for spec in cw.MODELS[1:]:
        pending.append((None, None, spec))
    for policy, cfg, (stem, kind, path) in pending:
        letters = ("A",) if stem in A_ONLY else ("C", "A")
        if not path.is_file() or path.stat().st_size < heldout.MIN_CKPT:
            print(f"[v2] missing {path}", flush=True)
            continue
        if policy is None:
            policy, cfg = load_policy_and_cfg(str(path), device)
        epoch = cw._epoch(path)
        name = cw._label(stem, path, epoch)
        resolved = heldout._norm_of(cfg, kind)
        if resolved is None:
            print(f"[v2] skip {name}  unsupported normalizer", flush=True)
            del policy
            torch.cuda.empty_cache()
            continue
        norm, _dim, decode, sha = resolved
        scored = {}
        for letter in letters:
            rec = recordings[letter]
            dataset, rows = cw._rows_for(cfg, norm, rec, cache)
            kept_i0 = {int(index[0]) for _i, _r, index in rec["kept"]}
            rows = [row for row in rows if int(row[2][0]) in kept_i0]
            if len(rows) != len(rec["kept"]):
                raise RuntimeError(f"{name} {letter} kept {len(rows)} vs {len(rec['kept'])}")
            print(f"[v2] score {name} {letter} {len(rows)}", flush=True)
            block = heldout._score_one(decode, policy, dataset, field, rec["demo"], rows, device, axis)
            scored[letter] = block
        for letter in letters:
            public = heldout._public(scored[letter])
            if stem in A_ONLY:
                public = {
                    **public,
                    "flag": f"heading GT median {heading_median['A']}° at step 1",
                }
            tables[letter].append((name, public))
            report["recordings"].setdefault(letter, {})[name] = {
                **public, "ckpt": str(path), "epoch": epoch, "decode": decode, "sha": sha[:12],
            }
        if stem in A_ONLY:
            print(f"[v2] {name} skipped on C  heading GT median {heading_median['C']}°", flush=True)
        del policy
        torch.cuda.empty_cache()
    (cw.OUT / "score.json").write_text(json.dumps(report, indent=2, default=cw._json))
    heldout._print_table("C  contact wall v2", tables["C"])
    heldout._print_table("A  contact wall v2", tables["A"])
    print(f"[v2] wrote {cw.OUT / 'score.json'}", flush=True)


TILT_NORM = Path(
    "/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/UMIFT_Data/surface_frames_scripts/assets/"
    "oct_v35_h4s_r4vt_action_normalizer_minmax_tilt1.pkl"
)


def _score_tilt1(policy, dataset, field, demo, rows, device, axis):
    """5-d tilt1: v1, v2, δn, stiffness, a. Side snap is the tilt1 label rule."""
    import sys
    sys.path.insert(0, "/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/UMIFT_Data/surface_frames_scripts")
    from heading_tilt import geodesic_deg, pack_tilt
    from run_policy_open_loop_val_export import predict_sparse_action
    from tilt1 import rotation_from_tilt1

    pos, rot, dn, tilt_a, tilt_b = [], [], [], [], []
    for n_done, (idx, _rgb, index) in enumerate(rows):
        obs, _sample = dataset.sampler.sample_sequence(idx)
        pred = np.asarray(predict_sparse_action(policy, obs, device), dtype=np.float64)
        pred = pred.reshape(-1, pred.shape[-1])[: heldout.H]
        if pred.shape[1] != 5:
            raise RuntimeError(f"tilt1 action {pred.shape}")
        i0 = int(index[0])
        xyz, frames = heldout._walk(
            field, pred, demo["foot"][i0], demo["frame"][i0], demo["tool_p"][i0], heldout._dts(demo, index)
        )
        n_out = -frames[:, :, 2]
        e_long = heldout._e_long(n_out, axis)
        got_R = rotation_from_tilt1(n_out, e_long, pred[:, 4])
        ab = np.degrees(np.abs(
            pack_tilt(got_R, n_out, e_long)[:, 2:4]
            - pack_tilt(demo["tool_R"][index], n_out, e_long)[:, 2:4]
        ))
        pos.append(np.linalg.norm(xyz - demo["tool_p"][index], axis=1) * 1e3)
        rot.append(geodesic_deg(got_R, demo["tool_R"][index]))
        dn.append(np.abs(pred[:, 2] - heldout._dn(demo, index)) * 1e3)
        tilt_a.append(ab[:, 0])
        tilt_b.append(ab[:, 1])
        if n_done and n_done % 40 == 0:
            print(f"[v2] tilt1 {n_done}/{len(rows)}", flush=True)
    return {
        "n": len(rows),
        "pos_mm": heldout._mean(np.stack(pos)),
        "rot_deg": heldout._mean(np.stack(rot)),
        "dn_mm": heldout._mean(np.stack(dn)),
        "tilt_a_deg": heldout._mean(np.stack(tilt_a)),
        "tilt_b_deg": heldout._mean(np.stack(tilt_b)),
    }


def score_tilt1() -> None:
    """A only. C is skipped with the other tilt models."""
    if not TILT_NORM.is_file():
        raise SystemExit(f"missing {TILT_NORM}")
    field = FixBMoldField(str(heldout.WALL))
    recordings, axis = cw._recordings(field)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    causal = next(item for item in cw.MODELS if item[0] == "causal e199")
    policy, cfg = load_policy_and_cfg(str(causal[2]), device)
    norm = heldout._norm_of(cfg, "causal")[0]
    rec = recordings["A"]
    dataset, rows = cw._rows_for(cfg, norm, rec, {})
    kept, _detail, counts = cw._select(rec["demo"], rows)
    if counts["kept"] != 94:
        raise SystemExit(f"A kept {counts['kept']}")
    kept_i0 = {int(index[0]) for _i, _r, index in kept}
    rows = [row for row in rows if int(row[2][0]) in kept_i0]
    del policy
    torch.cuda.empty_cache()
    gt = json.loads((cw.OUT / "gt.json").read_text())["gt"]["A"]["head"]["median"]["1"]
    report = json.loads((cw.OUT / "score.json").read_text())
    table = []
    for stem, kind, path in cw.MODELS:
        if stem != "tilt1":
            continue
        policy, cfg = load_policy_and_cfg(str(path), device)
        epoch = cw._epoch(path)
        name = cw._label(stem, path, epoch)
        print(f"[v2] score {name} A {len(rows)}", flush=True)
        public = _score_tilt1(policy, dataset, field, rec["demo"], rows, device, axis)
        public["flag"] = f"heading GT median {gt}° at step 1"
        public["decode"] = "tilt1"
        public["epoch"] = epoch
        public["ckpt"] = str(path)
        report["recordings"].setdefault("A", {})[name] = public
        table.append((name, public))
        print(f"[v2] {name} skipped on C  heading GT median 58.048°", flush=True)
        del policy
        torch.cuda.empty_cache()
    (cw.OUT / "score.json").write_text(json.dumps(report, indent=2, default=cw._json))
    heldout._print_table("A  tilt1", table)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tilt1":
        score_tilt1()
    else:
        score()
