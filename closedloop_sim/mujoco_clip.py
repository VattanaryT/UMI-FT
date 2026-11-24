"""Render one rollout npz to an mp4. Run with the teleop venv, which has mujoco."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)


def _obj(path: Path, vertices, faces) -> None:
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"f {a+1} {b+1} {c+1}" for a, b, c in faces]
    path.write_text("\n".join(lines))


def render(npz_path: Path, demo_path: Path, out_mp4: Path) -> None:
    sim = np.load(npz_path)
    demo = np.load(demo_path)
    field = np.load(FIELD)
    faces = np.asarray(field["faces"])
    if len(faces) > 8000:
        faces = faces[:: max(1, len(faces) // 8000)]
    obj = out_mp4.with_suffix(".obj")
    _obj(obj, field["vertices"], faces)
    center = np.asarray(sim["position"]).mean(axis=0)
    xml = f"""
<mujoco>
  <visual><global offwidth="640" offheight="480"/></visual>
  <asset><mesh name="mold" file="{obj.name}"/></asset>
  <worldbody>
    <light pos="0 -1 1"/>
    <camera name="cam" pos="{center[0]} {center[1]-0.8} {center[2]+0.45}" xyaxes="1 0 0 0 0.45 0.9"/>
    <geom type="mesh" mesh="mold" rgba="0.7 0.7 0.75 0.5" contype="0" conaffinity="0"/>
    <body name="tool" mocap="true">
      <geom type="sphere" size="0.008" rgba="0.9 0.3 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml, {obj.name: obj.read_bytes()})
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    body = model.body("tool").id
    step = max(1, len(sim["position"]) // 400)
    path = np.asarray(sim["position"])[::step]
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480", "-r", "20",
            "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_mp4),
        ],
        stdin=subprocess.PIPE,
    )
    for xyz in path:
        data.mocap_pos[0] = xyz
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="cam")
        proc.stdin.write(np.ascontiguousarray(renderer.render()).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out_mp4}")
    renderer.close()
    print(out_mp4, len(path), flush=True)


if __name__ == "__main__":
    render(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
