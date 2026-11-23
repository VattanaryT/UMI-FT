"""How many action and observation dimensions the training labels use.

Train episodes only (seed-42 val split held out, plus episodes 35 and 44).
One RGB step at a time. Does not load a policy and does not touch the robot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from fixb_labels import pose7_to_Rp
from heading_tilt import e_long_sequence, long_axis, pack_tilt

DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
ZARR = DATA / "r4vt_causal_foot/acp_replay_buffer.zarr"
FIELD = DATA / "shared/fixb_field.npz"
OUT = Path(__file__).resolve().parent
UP = np.array([0.0, -1.0, 0.0])
REGIONS = ("crown", "radius", "flange", "operator wall", "far wall")
EXCLUDE = {35, 44}


# Same held-out episodes the causal-foot loader uses (seed 42, then 35 and 44).
VAL = {9, 12, 28, 37, 41}


def _shares(block: np.ndarray) -> np.ndarray:
    var = np.var(block, axis=0)
    total = float(var.sum())
    if total < 1e-18:
        return np.zeros(block.shape[1])
    return var / total


def _pca(block: np.ndarray) -> np.ndarray:
    centered = block - block.mean(axis=0)
    if len(centered) < 3:
        return np.full(block.shape[1], np.nan)
    _u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    power = s**2
    total = float(power.sum())
    if total < 1e-18:
        return np.zeros(block.shape[1])
    return power / total


def _pca_standard(block: np.ndarray) -> np.ndarray:
    scale = block.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return _pca((block - block.mean(axis=0)) / scale)


def _r2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    design = np.column_stack([x, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-18:
        return 0.0
    return 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot


def _nn_r2(x: np.ndarray, y: np.ndarray, k: int = 8) -> float:
    """Leave-one-out neighbor prediction. A high value means x determines y."""
    rng = np.random.default_rng(0)
    take = rng.choice(len(x), size=min(4000, len(x)), replace=False)
    xs = np.asarray(x[take], dtype=np.float64)
    ys = np.asarray(y[take], dtype=np.float64).reshape(-1)
    scale = xs.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    tree = cKDTree(xs / scale)
    dist, idx = tree.query(xs / scale, k=k + 1)
    neigh = idx[:, 1:]
    pred = ys[neigh].mean(axis=1)
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    if ss_tot < 1e-18:
        return 0.0
    return 1.0 - float(np.sum((ys - pred) ** 2)) / ss_tot


def _region(normal, foot, apron_tree, apron_mask_pts) -> np.ndarray:
    del apron_mask_pts
    normal = np.asarray(normal, dtype=np.float64)
    unit = normal / np.clip(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12, None)
    ang = np.degrees(np.arccos(np.clip(unit @ UP, -1.0, 1.0)))
    dist, _ = apron_tree.query(foot)
    code = np.empty(len(foot), dtype=object)
    code[:] = "radius"
    code[ang < 25.0] = "crown"
    code[ang >= 55.0] = np.where(unit[ang >= 55.0, 2] >= 0.0, "operator wall", "far wall")
    code[dist < 0.015] = "flange"
    return code


def collect():
    field = np.load(FIELD)
    axis = long_axis(field["vertices"])
    apron = field["vertices"][np.asarray(field["is_apron"]).astype(bool)]
    apron_tree = cKDTree(apron)
    root = zarr.open(str(ZARR), "r")["data"]
    rows = {name: [] for name in (
        "field", "world", "r1rel", "tilt", "rel_rot", "region", "geom", "u", "v",
        "wrench", "wrench_world",
    )}
    for ep in range(50):
        if ep in VAL or ep in EXCLUDE:
            continue
        g = root[f"episode_{ep}"]
        R, p = pose7_to_Rp(np.asarray(g["ts_pose_fb_0"], dtype=np.float64))
        rt = np.asarray(g["robot_time_stamps_0"]).reshape(-1)
        rgb_t = np.asarray(g["rgb_time_stamps_0"]).reshape(-1)
        index = np.clip(np.searchsorted(rt, rgb_t), 0, len(p) - 2)
        step_i = np.clip(np.searchsorted(rt, np.r_[rgb_t[1:], rgb_t[-1]]), 0, len(p) - 1)
        dt = rt[step_i] - rt[index]
        ok = (dt > 0.02) & (dt < 0.2) & (step_i > index)
        index, step_i, dt = index[ok], step_i[ok], dt[ok]
        if len(index) == 0:
            continue
        v = np.asarray(g["fixb_v_par_0"], dtype=np.float64)[index]
        dn = np.asarray(g["fixb_delta_n_0"], dtype=np.float64).reshape(-1)[index]
        normal = np.asarray(g["mold_normal_0"], dtype=np.float64).reshape(-1, 3)[index]
        foot = np.asarray(g["mold_x_s_0"], dtype=np.float64).reshape(-1, 3)[index]
        frame = np.asarray(g["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)[index]
        sd = np.asarray(g["mold_signed_distance_0"], dtype=np.float64).reshape(-1)[index]
        r6 = np.asarray(g["mold_R_field_rot6d_0"], dtype=np.float64).reshape(-1, 6)[index]
        delta = p[step_i] - p[index]
        field_step = np.column_stack([v[:, 0] * dt, v[:, 1] * dt, dn])
        world_step = delta
        tool_step = np.einsum("nji,nj->ni", R[index], delta)
        e_long, _ang = e_long_sequence(normal, axis)
        packed = pack_tilt(R[index], normal, e_long)
        psi = np.arctan2(packed[:, 0], packed[:, 1])
        tilt = np.column_stack([psi, packed[:, 2], packed[:, 3]])
        rel = np.einsum("nji,njk->nik", R[index], R[step_i])
        rel_rv = Rotation.from_matrix(rel).as_rotvec()
        along = foot @ axis
        side = np.cross(np.broadcast_to(axis, normal.shape), normal)
        # v is the signed angle of the outward normal about the stringer axis.
        cross_v = np.arctan2(
            np.sum(side * np.cross(axis, UP), axis=1),
            np.sum(normal * UP, axis=1),
        )
        wt = np.asarray(g["wrench_time_stamps_0"]).reshape(-1)
        wr = np.asarray(g["wrench_left_0"], dtype=np.float64).reshape(-1, 6)
        wi = np.clip(np.searchsorted(wt, rgb_t), 0, len(wr) - 1)[ok]
        force = wr[wi, :3]
        force_world = np.einsum("nij,nj->ni", R[index], force)
        geom = np.column_stack([sd, normal, r6])
        code = _region(normal, foot, apron_tree, apron)
        rows["field"].append(field_step)
        rows["world"].append(world_step)
        rows["r1rel"].append(tool_step)
        rows["tilt"].append(tilt)
        rows["rel_rot"].append(rel_rv)
        rows["region"].append(code)
        rows["geom"].append(geom)
        rows["u"].append(along)
        rows["v"].append(cross_v)
        rows["wrench"].append(force)
        rows["wrench_world"].append(force_world)
        print(f"  ep {ep:02d} {len(index)}", flush=True)
    out = {k: np.concatenate(v) for k, v in rows.items()}
    return out, axis


def _block_table(name, axes, block, region) -> list[str]:
    lines = [
        f"### {name}",
        "",
        "| region | n | " + " | ".join(f"{a} %" for a in axes) + " | PCA raw | PCA equalized |",
        "|---|---:|" + "---:|" * len(axes) + "---:|---:|",
    ]
    order = ("all",) + REGIONS
    for reg in order:
        mask = np.ones(len(block), dtype=bool) if reg == "all" else region == reg
        part = block[mask]
        if len(part) < 30:
            lines.append(f"| {reg} | {len(part)} | " + " | ".join(["—"] * (len(axes) + 2)) + " |")
            continue
        share = _shares(part) * 100
        raw = _pca(part) * 100
        eq = _pca_standard(part) * 100
        cells = " | ".join(f"{s:.1f}" for s in share)
        lines.append(
            f"| {reg} | {len(part)} | {cells} | "
            f"{raw[0]:.1f}/{raw[1]:.1f}/{raw[2]:.1f} | "
            f"{eq[0]:.1f}/{eq[1]:.1f}/{eq[2]:.1f} |"
        )
    lines.append("")
    return lines


def _bars(path, region, field, world, r1rel, tilt, rel_rot):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4), dpi=120)
    specs = (
        (axes[0, 0], field, ("v1·dt", "v2·dt", "δn"), "field position"),
        (axes[0, 1], world, ("x", "y", "z"), "world position"),
        (axes[0, 2], r1rel, ("tool x", "tool y", "tool z"), "r1rel position"),
        (axes[1, 0], tilt, ("ψ", "a", "b"), "field rotation"),
        (axes[1, 1], rel_rot, ("rx", "ry", "rz"), "r1rel rotation"),
    )
    colors = ("#4aa3ff", "#ff9f1a", "#ff4d6a")
    x = np.arange(len(REGIONS))
    width = 0.25
    for ax, block, labels, title in specs:
        for j, (label, color) in enumerate(zip(labels, colors)):
            height = []
            for reg in REGIONS:
                part = block[region == reg]
                height.append(0.0 if len(part) < 30 else float(_shares(part)[j] * 100))
            ax.bar(x + (j - 1) * width, height, width, label=label, color=color)
        ax.set_xticks(x, [r.replace(" ", "\n") for r in REGIONS], fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of variance")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, frameon=False)
    ax = axes[1, 2]
    series = (
        ("field pos", field, "#4aa3ff"),
        ("r1rel pos", r1rel, "#ff4d6a"),
        ("tilt ψab", tilt, "#2a9d8f"),
        ("r1rel rot", rel_rot, "#9b5de5"),
    )
    for j, (label, block, color) in enumerate(series):
        height = []
        for reg in REGIONS:
            part = block[region == reg]
            height.append(0.0 if len(part) < 30 else float(_pca_standard(part)[0] * 100))
        ax.bar(x + (j - 1.5) * 0.18, height, 0.18, label=label, color=color)
    ax.set_xticks(x, [r.replace(" ", "\n") for r in REGIONS], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("1st PCA % after equalizing")
    ax.set_title("how one-dimensional", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Share of label variance by mold region", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _self_check():
    block = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    share = _shares(block)
    assert abs(share[0] - 1.0) < 1e-9 and abs(share.sum() - 1.0) < 1e-9
    ratio = _pca(block)
    assert abs(ratio[0] - 1.0) < 1e-6
    assert abs(_r2(np.column_stack([np.arange(20), np.zeros(20)]), np.arange(20) * 2) - 1) < 1e-6


def main():
    _self_check()
    print("[dims] collecting", flush=True)
    data, axis = collect()
    region = data["region"]
    lines = [
        "# Action and observation dimensions",
        "",
        "Training episodes only. Seed-42 validation episodes are out, and so are",
        "episodes 35 and 44. Each row is one RGB step whose next step is 20–200 ms",
        "later. Position steps are in meters. Field position is (v1·dt, v2·dt, δn):",
        "the first two are the tangential step, and δn is the stored virtual-target",
        "depth, not a step. World position is the fixture-frame tool step.",
        "r1rel position is that same step in the current tool frame.",
        "Rotation is in radians: field heading ψ and tilts a, b, versus the",
        "one-step rotation vector in the tool frame.",
        "",
        "Variance % is each axis's variance divided by the sum of the three.",
        "PCA raw is the explained-variance ratio of the three numbers as stored.",
        "PCA equalized divides each axis by its own standard deviation first,",
        "which is what a min-max normalizer roughly does. The three ratios are",
        "written λ1/λ2/λ3.",
        "",
        f"Stringer axis used for u: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}].",
        "Crown is a mold normal within 25° of up. Radius is 25–55°. A wall is",
        "55° or steeper: operator-side if the normal points toward +Z, far-side",
        "if it points toward −Z. Flange is a foot within 15 mm of an apron vertex,",
        "and it wins over the angle bins.",
        "",
    ]
    lines += _block_table("Field position (v1·dt, v2·dt, δn)", ("v1·dt", "v2·dt", "δn"), data["field"], region)
    lines += _block_table("World position (fixture step)", ("x", "y", "z"), data["world"], region)
    lines += _block_table("r1rel position (tool step)", ("tx", "ty", "tz"), data["r1rel"], region)
    lines += _block_table("Field rotation (ψ, a, b)", ("ψ", "a", "b"), data["tilt"], region)
    lines += _block_table("r1rel rotation (tool rotvec)", ("rx", "ry", "rz"), data["rel_rot"], region)
    a = data["tilt"][:, 1]
    b = data["tilt"][:, 2]
    lines += [
        "## Tilt a against b",
        "",
        f"std(a) = {np.degrees(a.std()):.2f}°, std(b) = {np.degrees(b.std()):.2f}°, "
        f"corr(a, b) = {np.corrcoef(a, b)[0, 1]:.3f}.",
        f"|b| exceeds |a| on {100 * np.mean(np.abs(b) > np.abs(a)):.1f}% of steps.",
        f"std of the steps: v1·dt {data['field'][:, 0].std() * 1e3:.2f} mm, "
        f"v2·dt {data['field'][:, 1].std() * 1e3:.2f} mm, "
        f"δn {data['field'][:, 2].std() * 1e3:.1f} mm.",
        "",
        "## Does robot0_mold_geom encode surface (u, v)?",
        "",
        "The 10 numbers are signed distance (1), the outward normal (3), and the",
        "field rotation as rot6d (6). u is the foot's position along the stringer",
        "axis. v is the signed angle of the outward normal about that axis.",
        "",
        f"Linear / neighbor R² of u from all 10: {_r2(data['geom'], data['u']):.3f} / {_nn_r2(data['geom'], data['u']):.3f}.",
        f"Linear R² of u from signed distance only: {_r2(data['geom'][:, :1], data['u']):.3f}.",
        f"Linear R² of u from the normal only: {_r2(data['geom'][:, 1:4], data['u']):.3f}.",
        f"Linear R² of u from the rot6d only: {_r2(data['geom'][:, 4:], data['u']):.3f}.",
        f"Linear / neighbor R² of v from all 10: {_r2(data['geom'], data['v']):.3f} / {_nn_r2(data['geom'], data['v']):.3f}.",
        f"Linear R² of v from the normal only: {_r2(data['geom'][:, 1:4], data['v']):.3f}.",
        "",
        "## Wrench frame",
        "",
    ]
    force = data["wrench"]
    world = data["wrench_world"]
    light = np.linalg.norm(force, axis=1) < 8.0
    g = np.array([0.0, 1.0, 0.0])

    def align(vec):
        nrm = np.linalg.norm(vec, axis=1)
        return np.abs(vec @ g) / np.clip(nrm, 1e-9, None)

    stored_spread = float(force[light].std(axis=0).mean())
    world_spread = float(world[light].std(axis=0).mean())
    lines += [
        "The zarr writer stores `wrench_left_0` as the TCP wrench, and",
        "`sparse_obs_to_obs_sample` multiplies it by the adjoint of the newest",
        "relative pose. That pose is the identity, so the network sees the stored",
        "TCP wrench. It is not rotated into the fixture and not into the field.",
        "",
        f"On {int(light.sum())} steps with |F| < 8 N, the mean per-axis standard",
        f"deviation is {stored_spread:.2f} N as stored and {world_spread:.2f} N "
        "after `R_tool @ F`.",
        f"Mean |cos| with gravity [0, +1, 0] is {align(force[light]).mean():.3f} as stored "
        f"and {align(world[light]).mean():.3f} in the fixture.",
        "Most steps are on the crown or the radius, where the tool is nearly",
        "upright, so the two frames are hard to tell apart. The check does not",
        "override the TCP conversion.",
        "",
        "## Adding field-frame (u, v) and field-frame force",
        "",
        "v is already in the normal: the normal alone has linear R² "
        f"{_r2(data['geom'][:, 1:4], data['v']):.3f} for v. u is only partly there",
        f"(linear {_r2(data['geom'], data['u']):.3f}, neighbor {_nn_r2(data['geom'], data['u']):.3f}),",
        "and that comes from the normal and the rot6d, not from the clearance",
        f"(clearance alone is {_r2(data['geom'][:, :1], data['u']):.3f}). The stringer",
        "bends, so the normal changes along its length. A straight section would",
        "not carry u. Adding u is a new scalar. Adding v repeats the normal.",
        "",
        "The field query that fills `robot0_mold_geom` already returns the foot",
        "`x_s` and the frame `R`. u is `x_s` dotted with the stringer axis.",
        "v is the angle of `R[:, 2]` about that axis. No second closest-point",
        "query. Field-frame force is `R_field.T @ R_tcp @ F` with the stored TCP",
        "force and that same `R`. That is a few lines in the obs converter and",
        "new shape_meta keys. The runner already does this query for mold geom.",
        "",
    ]
    lines += [
        "## Proposal: causal-foot 1-tilt (not submitted)",
        "",
        "One training run. Heading ψ stays on the field frame. The network learns",
        "tilt a. Tilt b is fixed at 0. Position stays v1, v2, δn, and stiffness",
        "stays the fourth channel, as in the 10-d action. Same zarr, same horizon",
        "64, same obs. The 10-d normalizer cannot be loaded as-is (`_pin_r4_10`",
        "requires 10 numbers). Copy its min and max for v1, v2, δn, and stiffness,",
        "and fit a on these same training rows with the same min-max.",
        "",
        f"This is a test, not a free reduction. std(a) = {np.degrees(a.std()):.1f}° and",
        f"std(b) = {np.degrees(b.std()):.1f}°. After equalizing, the rotation PCA is",
        "about 41/33/26, so b is still a quarter of the equalized rotation. It is",
        "small on the operator wall (equalized third component about 7%) and not",
        "small on the radius. δn is 96% of the raw position variance only because",
        "it is a 43 mm offset next to 6–7 mm steps. After equalizing, the three",
        "position axes are about 37/33/30, so all three stay.",
        "",
        "Files that would be added, and not edited here:",
        "",
        "- `config/task/umift_oct_fixb_r4vt_causal_foot_1tilt.yaml` inherits",
        "  `umift_oct_fixb_r4vt_causal_foot` and sets `shape_meta.action.shape: [5]`.",
        "- A dataset subclass next to `fixb_r4vt_dataset.py` that packs",
        "  `[v1, v2, δn, stiffness, a]` using `pack_tilt` on the existing ΔR.",
        "- A 5-d normalizer pickle written by that same packer.",
        "- `eval/causal_foot_1tilt.sbatch` (this folder, not submitted).",
        "",
        "Left unchanged: the causal-foot zarr, `fixb_field.npz`, the long-dt mask,",
        "episodes 35 and 44, the obs (rgb, TCP wrench, 10-d mold geom), horizon 64,",
        "the h4s observation horizons, and `train.py`. `mold_aware_runtime.decode_chunk`",
        "has to grow a 5-d branch before a rollout. It is not needed to train.",
        "",
        "The last causal-foot sbatch asked for 320G, 40 CPUs, and 32 workers.",
        "This sketch asks for 128G, 8 CPUs, and 4 workers. The replay buffer copy",
        "in that run was about 3.6 GB, so four workers are on the order of 15 GB",
        "of buffers plus the model.",
        "",
    ]
    fig = OUT / "action_dims.png"
    _bars(fig, region, data["field"], data["world"], data["r1rel"], data["tilt"], data["rel_rot"])
    (OUT / "action_dims.md").write_text("\n".join(lines))
    summary = {
        "n": int(len(region)),
        "regions": {reg: int(np.sum(region == reg)) for reg in REGIONS},
        "std_a_deg": float(np.degrees(a.std())),
        "std_b_deg": float(np.degrees(b.std())),
    }
    (OUT / "action_dims.json").write_text(json.dumps(summary, indent=2))
    print("[dims] wrote", OUT, summary, flush=True)


if __name__ == "__main__":
    main()
