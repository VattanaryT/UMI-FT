"""Per-rollout MP4 recorders: wrist (D405 feed) + center Orbbec (V4L)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PyriteEnvSuites.env_runners.frame_rate_watch import FrameRateWatch
from PyriteEnvSuites.env_runners.term_color import print_red

DEFAULT_ORBBEC_V4L = (
    "/dev/v4l/by-id/usb-Orbbec_Orbbec_Femto_Bolt_3D_Camera_CL8K14100WH-video-index0"
)
# The by-id video-index0 link is the Y16 depth node (1024x512). Color is
# another /dev/videoN on the same USB serial, with no by-id symlink.
_DEPTH_FRAME_HW = (512, 1024)


def _serial_from_by_id(path: str) -> Optional[str]:
    """``..._CL8K14100WH-video-index0`` -> ``CL8K14100WH``."""
    name = Path(path).name
    head = name.split("-video-index")[0]
    serial = head.split("_")[-1]
    return serial or None


def _nodes_for_serial(serial: str) -> list:
    """Every /dev/videoN whose USB device serial matches. Stable across replugs."""
    found = []
    root = Path("/sys/class/video4linux")
    if not root.is_dir():
        return found
    for sysdir in sorted(root.glob("video*")):
        cur = (sysdir / "device").resolve()
        device_serial = ""
        for _ in range(6):
            serial_path = cur / "serial"
            if serial_path.is_file():
                device_serial = serial_path.read_text().strip()
                break
            if cur.parent == cur:
                break
            cur = cur.parent
        if device_serial == serial:
            found.append("/dev/" + sysdir.name)
    return found


def _video_index(path: str) -> Optional[int]:
    """``/dev/video8`` -> 8. OpenCV's V4L2 backend rejects by-id paths by name."""
    name = Path(path).name
    if name.startswith("video") and name[5:].isdigit():
        return int(name[5:])
    return None


def _color_device_candidates(path: str) -> list:
    """Color is a different USB interface than the by-id depth symlink."""
    serial = _serial_from_by_id(path)
    nodes = _nodes_for_serial(serial) if serial else []
    if not nodes:
        nodes = [str(path)]
    return nodes


def _open_one(device: str):
    resolved = str(Path(device).resolve())
    index = _video_index(resolved)
    attempts = []
    if index is not None:
        attempts.append(lambda: cv2.VideoCapture(index))
    attempts.append(lambda: cv2.VideoCapture(resolved))
    for make in attempts:
        cap = make()
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if not ok or frame is None or tuple(frame.shape[:2]) == _DEPTH_FRAME_HW:
            cap.release()
            continue
        return cap, resolved, tuple(frame.shape[:2])
    return None, resolved, None


def _open_orbbec_color(path: str):
    """Open the color node for this Bolt. The by-id link is the Y16 depth node."""
    tried = []
    for device in _color_device_candidates(path):
        cap, resolved, shape = _open_one(device)
        tried.append(
            f"{resolved}" + (f" {shape[1]}x{shape[0]}" if shape else " failed")
        )
        if cap is not None:
            return cap, resolved, shape, tried
    return None, None, None, tried


class _Mp4Writer:
    def __init__(self, path: Path, fps: float, size_wh: tuple[int, int]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._w = cv2.VideoWriter(str(self.path), fourcc, float(fps), size_wh)
        if not self._w.isOpened():
            raise RuntimeError(f"VideoWriter failed to open {self.path}")
        self.n = 0

    def write_bgr(self, frame_bgr: np.ndarray) -> None:
        self._w.write(frame_bgr)
        self.n += 1

    def release(self) -> None:
        self._w.release()


class RolloutCamRecorder:
    """Writes wrist_d405.mp4 + orbbec_center.mp4 under ``out_dir``."""

    def __init__(
        self,
        out_dir: Path,
        *,
        orbbec_v4l: str = DEFAULT_ORBBEC_V4L,
        wrist_fps: float = 15.0,
        orbbec_fps: float = 15.0,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.orbbec_v4l = orbbec_v4l
        self.wrist_fps = float(wrist_fps)
        self.orbbec_fps = float(orbbec_fps)
        self._wrist: Optional[_Mp4Writer] = None
        self._orbbec: Optional[_Mp4Writer] = None
        self._orb_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        # Same 15 Hz as D405 collection and the training rgb stream.
        self.wrist_rate = FrameRateWatch("wrist D405", self.wrist_fps)
        self.center_rate = FrameRateWatch("center Orbbec", self.orbbec_fps)

    def start(self) -> None:
        self._stop.clear()
        self._orb_thread = threading.Thread(
            target=self._orbbec_loop, name="OrbbecRolloutRec", daemon=True
        )
        self._orb_thread.start()

    def push_wrist_rgb(self, rgb: np.ndarray) -> None:
        """Accept HxWx3 RGB uint8 (D405 raw or policy-res)."""
        if rgb is None or rgb.size == 0:
            return
        gap = self.wrist_rate.tick()
        if gap:
            print(gap)
        bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        with self._lock:
            if self._wrist is None:
                self._wrist = _Mp4Writer(
                    self.out_dir / "wrist_d405.mp4", self.wrist_fps, (w, h)
                )
            self._wrist.write_bgr(bgr)

    def stop(self) -> dict:
        self._stop.set()
        if self._orb_thread is not None:
            self._orb_thread.join(timeout=3.0)
            self._orb_thread = None
        with self._lock:
            n_w = self._wrist.n if self._wrist else 0
            if self._wrist is not None:
                self._wrist.release()
                self._wrist = None
            n_o = self._orbbec.n if self._orbbec else 0
            if self._orbbec is not None:
                self._orbbec.release()
                self._orbbec = None
        wrist_sum = self.wrist_rate.summary()
        center_sum = self.center_rate.summary()
        print(self.wrist_rate.summary_line())
        print(self.center_rate.summary_line())
        return {
            "wrist_mp4": str(self.out_dir / "wrist_d405.mp4") if n_w else None,
            "orbbec_mp4": str(self.out_dir / "orbbec_center.mp4") if n_o else None,
            "wrist_frames": n_w,
            "orbbec_frames": n_o,
            "wrist_rate": wrist_sum,
            "center_rate": center_sum,
        }

    def _orbbec_loop(self) -> None:
        cap, device, shape, tried = _open_orbbec_color(self.orbbec_v4l)
        if cap is None:
            print_red(
                "[rollout-cam] Orbbec color open failed. Tried: "
                + "; ".join(tried)
            )
            return
        print(
            f"[rollout-cam] Orbbec color: {device} "
            f"{shape[1]}x{shape[0]}"
        )
        period = 1.0 / max(self.orbbec_fps, 1e-3)
        read_fail_announced = False
        last_ok = time.time()
        try:
            while not self._stop.is_set():
                t0 = time.time()
                ok, frame = cap.read()
                if not ok or frame is None:
                    if (
                        not read_fail_announced
                        and time.time() - last_ok > 1.0
                    ):
                        print_red(
                            f"[rollout-cam] Orbbec read failed: {self.orbbec_v4l}"
                        )
                        read_fail_announced = True
                    time.sleep(0.05)
                    continue
                read_fail_announced = False
                last_ok = time.time()
                h, w = frame.shape[:2]
                with self._lock:
                    if self._orbbec is None:
                        self._orbbec = _Mp4Writer(
                            self.out_dir / "orbbec_center.mp4",
                            self.orbbec_fps,
                            (w, h),
                        )
                    self._orbbec.write_bgr(frame)
                gap = self.center_rate.tick()
                if gap:
                    print(gap)
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
        finally:
            cap.release()
