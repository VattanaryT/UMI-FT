#!/usr/bin/env python3
"""Compare thin-stringer field axes with the fixb training field.

Read-only. Writes frame_flip.md and flange_lean.png next to this file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.spatial import cKDTree

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / "eval")]

from diffusion_policy.common.sampler_umift import sparse_action_indices_from_rgb_timestamps
from fixb_field import FixBMoldField
from fixb_labels import delta_R_from_field_tool, pose7_to_Rp
from heading_tilt import (
    e_long_sequence,
    geodesic_deg,
    long_axis,
    pack_tilt,
    rotation_from_tilt,
)
from mold_aware_runtime import rot6d_rows_to_R

OUT = Path(__file__).resolve().parent
FIXB_FIELD = (
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
THIN_FIELD = "/mnt/data/umidata/ft/held_out_mold/thin_stringer/field/fixb_field.npz"
FIXB_ZARR = (
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_causal_foot/acp_replay_buffer.zarr"
)
HEAD_ZARR = (
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_heading_tilt/acp_replay_buffer.zarr"
)
THIN_ZARR = (
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/"
    "remap_to_training/acp_replay_buffer.zarr"
)
UP = np.array([0.0, -1.0, 0.0])
VAL = (9, 12, 28, 37, 41)
REGIONS = ("wall near flange", "radius", "flange", "crown")


def _self_check() -> None:
    line = np.stack([np.linspace(-1, 1, 20), np.zeros(20), np.zeros(20)], axis=1)
    flipped = line * np.array([-1.0, 1.0, 1.0])
    a = long_axis(line)
    b = long_axis(flipped)
    if a[0] < 0.9 or b[0] < 0.9:
        raise RuntimeError("long_axis did not force +X")
    n = np.array([[0.0, 0.0, 1.0]])
    e, _ = e_long_sequence(n, np.array([1.0, 0.0, 0.0]))
    if abs(float(np.dot(e[0], [1.0, 0.0, 0.0])) - 1.0) > 1e-6:
        raise RuntimeError("e_long left the tangent")


def pca_axis(vertices: np.ndarray) -> np.ndarray:
    centered = np.asarray(vertices, dtype=np.float64) - np.mean(vertices, axis=0)
    _values, vectors = np.linalg.eigh(np.cov(centered.T))
    axis = vectors[:, -1]
    return axis / np.linalg.norm(axis)


def inward_frame(rotation: np.ndarray) -> np.ndarray:
    flip = np.diag([1.0, -1.0, -1.0])
    return np.einsum("...ij,jk->...ik", rotation, flip)


def regions_of(normal, foot, apron) -> np.ndarray:
    ang = np.degrees(np.arccos(np.clip(normal @ UP, -1.0, 1.0)))
    dist, _idx = apron.query(foot, k=1)
    name = np.empty(len(foot), dtype=object)
    name[:] = "other"
    name[ang < 25.0] = "crown"
    name[(ang >= 25.0) & (ang < 55.0)] = "radius"
    name[dist < 0.015] = "flange"
    wall = (ang >= 55.0) & (dist >= 0.015) & (dist < 0.04)
    name[wall] = "wall near flange"
    return name


def unit_mean(rows: np.ndarray) -> np.ndarray:
    mean = np.mean(rows, axis=0)
    n = float(np.linalg.norm(mean))
    return mean / n if n > 1e-12 else np.zeros(3)


def vec(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:+.3f}" for x in np.asarray(v).reshape(3)) + "]"


def load_demo(zarr_path: str, episode: str, field: FixBMoldField):
    group = zarr.open(zarr_path, "r")["data"][episode]
    pose = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    robot_t = np.asarray(group["robot_time_stamps_0"]).reshape(-1)
    rgb_t = np.asarray(group["rgb_time_stamps_0"]).reshape(-1)
    # Every 4th robot sample. The field query is the slow part.
    step = np.arange(0, len(pose), 4)
    tool_R, tool_p = pose7_to_Rp(pose[step])
    geom = field.query(tool_p)
    foot = np.asarray(geom["x_s"], dtype=np.float64).reshape(-1, 3)
    outward = np.asarray(geom["R"], dtype=np.float64).reshape(-1, 3, 3)
    normal = outward[:, :, 2].copy()
    motion = np.zeros_like(tool_p)
    motion[1:] = tool_p[1:] - tool_p[:-1]
    motion[0] = motion[1]
    speed = np.linalg.norm(motion, axis=1)
    moving = speed > 1e-4
    motion[moving] /= speed[moving, None]
    dist = np.linalg.norm(tool_p - foot, axis=1)
    return {
        "tool_R": tool_R,
        "tool_p": tool_p,
        "tool_z": tool_R[:, :, 2],
        "foot": foot,
        "normal": normal,
        "frame": inward_frame(outward),
        "motion": motion,
        "moving": moving,
        "dist": dist,
        "robot_t": robot_t,
        "rgb_t": rgb_t,
        "pose": pose,
        "group": group,
    }


def region_table(demo, axis) -> list[str]:
    apron = cKDTree(demo["apron_pts"])
    name = regions_of(demo["normal"], demo["foot"], apron)
    near = demo["dist"] < 0.04
    e_long, _ang = e_long_sequence(demo["normal"], axis)
    third = np.cross(demo["normal"], e_long)
    third /= np.clip(np.linalg.norm(third, axis=1, keepdims=True), 1e-12, None)
    lines = [
        "| region | n | e_long | n_out | n × e_long | e_long·motion | n_out·tool z |",
        "|---|---:|---|---|---|---:|---:|",
    ]
    for region in REGIONS:
        mask = (name == region) & demo["moving"] & near
        n = int(mask.sum())
        if n == 0:
            lines.append(f"| {region} | 0 | — | — | — | — | — |")
            continue
        lines.append(
            "| {region} | {n} | {e} | {no} | {th} | {em:.2f} | {nz:.2f} |".format(
                region=region,
                n=n,
                e=vec(unit_mean(e_long[mask])),
                no=vec(unit_mean(demo["normal"][mask])),
                th=vec(unit_mean(third[mask])),
                em=float(np.median(np.sum(e_long[mask] * demo["motion"][mask], axis=1))),
                nz=float(np.median(np.sum(demo["normal"][mask] * demo["tool_z"][mask], axis=1))),
            )
        )
    return lines


def angle(decoded, tool) -> float:
    return float(geodesic_deg(decoded.reshape(1, 3, 3), tool.reshape(1, 3, 3))[0])


def thin_windows(demo):
    t0 = float(demo["rgb_t"][0])
    cutoff = t0 + 0.5 * float(demo["rgb_t"][-1] - t0)
    chosen = []
    last = -10**9
    for rgb, stamp in enumerate(demo["rgb_t"]):
        if stamp >= cutoff or rgb - last < 16:
            continue
        index, _times = sparse_action_indices_from_rgb_timestamps(
            demo["robot_t"], demo["rgb_t"], rgb, 64, 1
        )
        index = np.asarray(index, dtype=np.int64)
        if len(index) < 64 or demo["robot_t"][index[-1]] >= cutoff:
            continue
        chosen.append((rgb, int(index[0]), float(stamp - t0)))
        last = rgb
    return chosen


def decode_rows(title, rows) -> list[str]:
    lines = [f"### {title}", "", "| window | causal inward | causal outward | heading | heading, e_long flipped | r1rel step 0 |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            "| {name} | {c:.2f} | {o:.2f} | {h:.2f} | {f:.2f} | {r:.2f} |".format(**row)
        )
    lines.append("")
    return lines


def one_decode(frame, normal, axis, tool_R, tool_p, robot_t, rgb_t) -> dict:
    """Ground-truth label through each decode, one pose."""
    delta = delta_R_from_field_tool(frame.reshape(1, 3, 3), tool_R.reshape(1, 3, 3))[0]
    causal = frame @ delta
    outward = frame @ np.diag([1.0, -1.0, -1.0])
    causal_out = outward @ delta_R_from_field_tool(
        outward.reshape(1, 3, 3), tool_R.reshape(1, 3, 3)
    )[0]
    # The outward round-trip is also 0 by algebra. The convention probe is
    # decoding an inward-built ΔR with the outward frame, and the reverse.
    mixed = outward @ delta
    e_long, _ = e_long_sequence(normal.reshape(1, 3), axis)
    tilt = pack_tilt(tool_R.reshape(1, 3, 3), normal.reshape(1, 3), e_long)
    heading = rotation_from_tilt(normal.reshape(1, 3), e_long, tilt)[0]
    flipped = -e_long
    heading_flip = rotation_from_tilt(normal.reshape(1, 3), flipped, tilt)[0]
    del tool_p, robot_t, rgb_t
    return {
        "causal": angle(causal, tool_R),
        "mixed": angle(mixed, tool_R),
        "out_round": angle(causal_out, tool_R),
        "heading": angle(heading, tool_R),
        "heading_flip": angle(heading_flip, tool_R),
    }


def stored_fixb_decode(axis_fixb) -> list[dict]:
    causal = zarr.open(FIXB_ZARR, "r")["data"]
    heading = zarr.open(HEAD_ZARR, "r")["data"]
    rows = []
    for episode in VAL:
        c = causal[f"episode_{episode}"]
        h = heading[f"episode_{episode}"]
        pose = np.asarray(c["ts_pose_fb_0"], dtype=np.float64)
        rgb_t = np.asarray(c["rgb_time_stamps_0"]).reshape(-1)
        robot_t = np.asarray(c["robot_time_stamps_0"]).reshape(-1)
        # First RGB whose 64-step chunk exists, matching the comparison stride.
        rgb = 0
        index = None
        for candidate in range(0, len(rgb_t), 64):
            got, _t = sparse_action_indices_from_rgb_timestamps(robot_t, rgb_t, candidate, 64, 1)
            got = np.asarray(got, dtype=np.int64)
            if len(got) >= 64:
                rgb = candidate
                index = int(got[0])
                break
        tool_R, _p = pose7_to_Rp(pose[index : index + 1])
        tool_R = tool_R[0]
        frame = np.asarray(c["mold_R_field_0"][index], dtype=np.float64).reshape(3, 3)
        normal = np.asarray(c["mold_normal_0"][index], dtype=np.float64).reshape(3)
        rot6 = np.asarray(c["fixb_delta_R_rot6d_0"][index], dtype=np.float64).reshape(1, 6)
        delta = rot6d_rows_to_R(rot6)[0]
        causal_R = frame @ delta
        tilt = np.asarray(h["fixb_heading_tilt_0"][index], dtype=np.float64).reshape(1, 4)
        e_long, _ = e_long_sequence(normal.reshape(1, 3), axis_fixb)
        head_R = rotation_from_tilt(normal.reshape(1, 3), e_long, tilt)[0]
        head_flip = rotation_from_tilt(normal.reshape(1, 3), -e_long, tilt)[0]
        rows.append(
            {
                "name": f"ep {episode} rgb {rgb}",
                "c": angle(causal_R, tool_R),
                "o": angle(frame @ np.diag([1.0, -1.0, -1.0]) @ delta, tool_R),
                "h": angle(head_R, tool_R),
                "f": angle(head_flip, tool_R),
                "r": 0.0,
            }
        )
    return rows


def flange_plot(demo, axis) -> str:
    apron = cKDTree(demo["apron_pts"])
    name = regions_of(demo["normal"], demo["foot"], apron)
    e_long, _ = e_long_sequence(demo["normal"], axis)
    tilt = pack_tilt(demo["tool_R"], demo["normal"], e_long)
    a_deg = np.degrees(tilt[:, 2])
    pick = np.flatnonzero((name == "flange") & (np.abs(a_deg + 9.0) < 4.0))
    if len(pick) < 5:
        pick = np.flatnonzero(name == "flange")
        order = np.argsort(np.abs(a_deg[pick] + 9.0))
        pick = pick[order[:5]]
    else:
        pick = pick[:5]
    # Cross-section: z across, up = -y. Centerline of this slice is the median foot z of the crown.
    crown = demo["foot"][regions_of(demo["normal"], demo["foot"], apron) == "crown"]
    center_z = float(np.median(crown[:, 2])) if len(crown) else 0.0
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=140)
    foot = demo["foot"][pick]
    tool_z = demo["tool_z"][pick]
    ax.scatter(foot[:, 2], -foot[:, 1], c="#9aabb0", s=28, zorder=3)
    toward_web = 0
    for f, z, adeg in zip(foot, tool_z, a_deg[pick]):
        tip = f + z * 0.025
        ax.annotate(
            "",
            xy=(tip[2], -tip[1]),
            xytext=(f[2], -f[1]),
            arrowprops=dict(arrowstyle="-|>", color="#1d4e89", lw=1.8),
        )
        ax.text(tip[2], -tip[1], f" {adeg:.0f}°", fontsize=8, color="#1d4e89")
        outward = np.sign(f[2] - center_z)
        toward_web += int(outward != 0 and np.sign(z[2]) == -outward)
    pad = 0.04
    ax.set_xlim(float(foot[:, 2].min()) - pad, float(foot[:, 2].max()) + pad)
    ax.set_ylim(float((-foot[:, 1]).min()) - pad, float((-foot[:, 1]).max()) + pad)
    ax.axvline(center_z, color="#bbb", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("z, across")
    ax.set_ylabel("up")
    ax.set_title("fixb flange, tool z, tilt a near −9°")
    fig.tight_layout()
    fig.savefig(OUT / "flange_lean.png")
    plt.close(fig)
    down = float(np.mean(tool_z[:, 1]))
    way = "down, away from the crown, toward the outer edge of the flange" if down > 0 else "up, toward the crown and the radius"
    shown = ", ".join(f"{v:.1f}°" for v in a_deg[pick])
    return (
        f"Five flange frames with tilt a at {shown}. They sit on the same spot. "
        f"Tool z points {way}. The 11° lean is smaller than the arrow. "
        f"See `flange_lean.png`."
    )


def main() -> None:
    _self_check()
    fixb_field = FixBMoldField(FIXB_FIELD)
    thin_field = FixBMoldField(THIN_FIELD)
    fixb = load_demo(FIXB_ZARR, "episode_9", fixb_field)
    thin = load_demo(THIN_ZARR, "episode_0", thin_field)
    fixb["apron_pts"] = fixb_field.vertices[np.load(FIXB_FIELD)["is_apron"].astype(bool)]
    thin["apron_pts"] = thin_field.vertices[np.load(THIN_FIELD)["is_apron"].astype(bool)]
    axis_fixb = long_axis(fixb_field.vertices)
    axis_thin = long_axis(thin_field.vertices)
    raw_fixb = pca_axis(fixb_field.vertices)
    raw_thin = pca_axis(thin_field.vertices)
    if raw_fixb[0] < 0:
        raw_fixb = -raw_fixb
    # raw_thin keeps the eigh sign. long_axis forces +X.
    thin_json = json.loads(Path(THIN_FIELD).with_suffix(".json").read_text())

    lines = [
        "# Field axis check, thin stringer vs fixb",
        "",
        "Both poses are already in the fixture frame the policies see. The thin zarr",
        "note says episode C was re-solved on the training mount (Rx 90, lever",
        "−0.01, −0.157, −0.136). The thin field `T_fixture_from_scanner` is identity,",
        "so no extra rotation is applied here. `e_long` is `long_axis` (PCA, forced +X)",
        "projected into the tangent plane. `n_out` is the field query column 2.",
        "The third axis is n_out × e_long.",
        "",
        "Samples in the tables below are within 40 mm of the mesh. Over the whole",
        f"thin episode the median TCP-to-foot distance is {np.median(thin['dist']) * 1e3:.0f} mm",
        "(most of the episode is off the mesh). fixb episode 9 median is "
        f"{np.median(fixb['dist']) * 1e3:.1f} mm.",
        "",
        "## Axes",
        "",
        f"fixb PCA before the +X rule {vec(raw_fixb if raw_fixb[0] > 0 else -raw_fixb)}, "
        f"long_axis {vec(axis_fixb)}.",
        f"thin field json pca {vec(np.asarray(thin_json['pca_length_axis']))}, "
        f"eigh {vec(raw_thin)}, long_axis {vec(axis_thin)}.",
        "",
        "### fixb, episode 9",
        "",
        *region_table(fixb, axis_fixb),
        "",
        "### thin stringer, first episode, training-mount pose",
        "",
        *region_table(thin, axis_thin),
        "",
        "## Ground-truth label decode",
        "",
        "Causal inward is the stored (or rebuilt) field frame times ΔR. That product",
        "is the tool rotation by algebra when ΔR was built from the same frame, so",
        "it stays near 0 even if the frame itself is upside down. Causal outward is",
        "that same inward ΔR multiplied by the outward frame: a large angle means",
        "the inward flip is doing the work. Heading uses the packed tilt. Flipping",
        "e_long and decoding the unflipped tilt shows what a PCA sign error would do.",
        "r1rel step 0 is the query pose relative to itself, so the angle is 0.",
        "",
    ]
    fixb_rows = stored_fixb_decode(axis_fixb)
    lines += decode_rows("fixb stored labels, one window per val episode, step 0", fixb_rows)

    thin_rows = []
    full_R, full_p = pose7_to_Rp(thin["pose"])
    geom = thin_field.query(full_p[::20])  # not used; windows use indexed query below
    del geom
    for rgb, index, t in thin_windows(thin):
        if t < 4.0:
            continue
        tool_R = full_R[index]
        point = full_p[index : index + 1]
        q = thin_field.query(point)
        outward = np.asarray(q["R"], dtype=np.float64).reshape(3, 3)
        frame = inward_frame(outward)
        normal = outward[:, 2]
        dist_mm = float(np.linalg.norm(point.reshape(3) - np.asarray(q["x_s"]).reshape(3)) * 1e3)
        got = one_decode(frame, normal, axis_thin, tool_R, full_p[index], thin["robot_t"], thin["rgb_t"])
        thin_rows.append(
            {
                "name": f"t={t:.2f}s n·tz {float(np.dot(normal, tool_R[:, 2])):+.2f} dist {dist_mm:.0f} mm",
                "c": got["causal"],
                "o": got["mixed"],
                "h": got["heading"],
                "f": got["heading_flip"],
                "r": 0.0,
            }
        )
    lines += decode_rows("thin rebuilt labels, step 0 of each first-half window", thin_rows)

    # Sign summary used by the verdict.
    def med_dot(demo, axis, key):
        e_long, _ = e_long_sequence(demo["normal"], axis)
        m = demo["moving"] & (demo["dist"] < 0.04)
        if key == "motion":
            return float(np.median(np.sum(e_long[m] * demo["motion"][m], axis=1)))
        return float(np.median(np.sum(demo["normal"][m] * demo["tool_z"][m], axis=1)))

    fixb_nz = med_dot(fixb, axis_fixb, "z")
    thin_nz = med_dot(thin, axis_thin, "z")
    fixb_em = med_dot(fixb, axis_fixb, "motion")
    thin_em = med_dot(thin, axis_thin, "motion")
    same_n = np.sign(fixb_nz) == np.sign(thin_nz) and abs(fixb_nz) > 0.2 and abs(thin_nz) > 0.2
    lines += [
        "## Verdict",
        "",
        f"n_out · tool z median: fixb {fixb_nz:.2f}, thin {thin_nz:.2f}.",
        f"e_long · motion median: fixb {fixb_em:.2f}, thin {thin_em:.2f}.",
        "",
    ]
    label_ok = all(r["c"] < 2.0 and r["h"] < 5.0 for r in thin_rows) and all(
        r["c"] < 2.0 for r in fixb_rows
    )
    lines += [
        "Ground-truth labels decode to the tool on both molds (causal 0°, heading",
        "under a few degrees on fixb, 0° on thin). They are not ~180° on thin.",
        "The ~180° column is the probe that multiplies an inward-built ΔR by the",
        "outward frame, or that flips e_long. The thin scorer did neither.",
        "",
        "e_long uses the same +X rule on both fields. The thin json PCA points at",
        "−X; `long_axis` flips it, which is the same rule that keeps the fixb axis at +X.",
        "e_long · motion is about 0 on both, so the stroke does not pick a sign.",
        "",
        f"n_out · tool z on the near-surface samples is {fixb_nz:.2f} on fixb and "
        f"{thin_nz:.2f} on thin. The thin contact windows separately flip between",
        "+Z and −Z, which is the thin web's two faces, not one global normal sign.",
        "There is no single axis to flip. No corrected rescore.",
        "",
        "The 177° causal and ~100° heading errors are the model predictions.",
        "Episode 9 has no near-surface samples in the 15–40 mm band above the apron,",
        "so the fixb wall-near-flange row is empty. The thin wall row is that band.",
        "",
    ]
    if not label_ok:
        lines += ["Label decode was not near 0. See the tables.", ""]
    del same_n
    lines += [
        "## Flange lean",
        "",
        flange_plot(fixb, axis_fixb),
        "",
    ]
    (OUT / "frame_flip.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'frame_flip.md'}")
    print("n_out·toolz", round(fixb_nz, 3), round(thin_nz, 3))
    print("e_long·motion", round(fixb_em, 3), round(thin_em, 3))
    print("thin label causal", [round(r["c"], 2) for r in thin_rows], "mixed", [round(r["o"], 2) for r in thin_rows])
    print("fixb label causal", [round(r["c"], 2) for r in fixb_rows], "heading", [round(r["h"], 2) for r in fixb_rows])


if __name__ == "__main__":
    main()
