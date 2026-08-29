"""ScanPauseController: periodic mid-episode laser-scan interrupts for a live
diffusion-policy inference loop (bimanual UR5e layup cell).

Problem this solves
--------------------
``run_layup_scan_eval`` in ``umift_env_runner.py`` already pauses RTDE,
scans, restores the right arm, and resumes RTDE -- but only ONCE, between
episodes, and it cold-starts ``run_layup_scan.py`` every time (~3-5s of
laser-settle + pipeline bring-up). This module does the same pause/scan/
restore/resume dance repeatedly *during* a live episode (every ~30s, see
``umift_env_runner.main``'s ``scan_pause_interval_s``), against a
long-running ``scan_daemon.py`` subprocess that stays armed for the whole
episode so each individual sweep is fast.

Why "just don't call the getters" is enough to protect the obs history
------------------------------------------------------------------------
``do_scan()`` is synchronous and is only ever invoked from the single
-threaded live control loop, in between horizons. While it's running, the
calling thread is *inside* this function -- it cannot also be calling
``env.get_observation_from_buffer()`` or ``env.schedule_controls()``, so no
active buffer-freezing is required to keep the policy's observation history
consistent:

* ``D405VisionRingBuffer`` keeps its background capture thread running
  the whole time (nothing in this module touches it) -- it just keeps
  filling its ``deque(maxlen=...)`` with new-but-unqueried frames. Since
  ``env.pause_rtde_control()`` halts trajectory *execution*, not the arm's
  position, the D405 wrist camera keeps seeing essentially the same static
  scene throughout the pause. ``do_scan`` also restores the right arm to
  the *exact* pre-scan joint angles via ``RETURN_RIGHT`` before returning
  RTDE control, so by the time the caller resumes calling
  ``get_last_k_rgb()``/``get_last_k_depth()``, the newest buffered frames
  are visually continuous with what the policy saw right before the pause.
* The pose/wrench buffers are backed by the C++ ``ManipServer`` process,
  which keeps reading RTDE receive-side feedback at 500 Hz regardless of
  whether RTDE *control* is paused -- there is no Python-level way to
  freeze it, and none is needed: we simply never call
  ``get_sparse_observation_from_buffer()`` during the pause, so nothing
  from that window ever enters an inference observation. Once we resume
  querying it, it reflects the arm's true (restored) current state.

So the "signal to the caller" that step 3 of ``do_scan`` needs is nothing
more than: this call is blocking, on the same thread the caller uses to
drive inference. No extra flag, lock, or thread pause is introduced.

Never-paused-forever guarantee
-------------------------------
``do_scan()`` calls ``env.pause_rtde_control()`` exactly once per call. From
the moment that call returns truthy, every subsequent step (SWEEP,
submitting the background compare, RETURN_RIGHT) runs inside a
``try/finally`` whose ``finally`` clause calls ``env.resume_rtde_control()``
unconditionally -- including when the scan daemon crashes, times out, or
returns SCAN_ERROR. A scan going wrong can make a ``ScanOutcome`` with
``success=False``, but it can never leave the arm paused indefinitely.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

# Sentinel pushed onto the stdout line queue when the daemon's stdout pipe
# closes (process exited, crashed, or was killed), so anything blocked in
# _wait_for_prefixes() returns promptly instead of waiting out the full
# timeout.
_EOF_SENTINEL = "\x00__SCAN_DAEMON_EOF__\x00"


@dataclass
class ScanOutcome:
    """Result of one ScanPauseController.do_scan() call.

    ``success`` reflects only the pause/SWEEP/RETURN_RIGHT/resume sequence.
    The background heatmap compare (``heatmap_future``) is fire-and-forget
    by design (see module docstring / do_scan docstring) and does not gate
    ``success`` -- a scan can succeed even if the later compare fails.
    """

    scan_id: int
    success: bool
    session_dir: Optional[Path] = None
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    heatmap_future: Optional[Future] = None
    error: Optional[str] = None


def _read_right_joints_via_rtde(keyence_python: Path, robot_ip: str) -> List[float]:
    """Read the right (task) arm's actual joint angles via a throwaway RTDE
    receive connection.

    Deliberately duplicated from
    ``umift_env_runner.read_robot_joints_via_rtde`` (same approach: a
    subprocess using the keyence venv's ``rtde_receive``) rather than
    imported, to keep this module free of a dependency on
    ``umift_env_runner`` -- which imports ``ScanPauseController`` -- and
    avoid a circular import. Keep the two in sync if the RTDE-read approach
    ever changes.
    """
    code = (
        "from rtde_receive import RTDEReceiveInterface\n"
        f"r = RTDEReceiveInterface(hostname={robot_ip!r})\n"
        "print(','.join(map(str, r.getActualQ())))\n"
        "r.disconnect()\n"
    )
    proc = subprocess.run(
        [str(keyence_python), "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=10.0,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to read joints from {robot_ip}: {proc.stderr.strip()}"
        )
    joints = [float(x) for x in proc.stdout.strip().split(",") if x.strip()]
    if len(joints) != 6:
        raise RuntimeError(f"Expected 6 joints, got {joints!r} from {robot_ip}")
    return joints


def _run_compare_and_load_heatmap(
    *,
    keyence_python: Path,
    keyence_ws: Path,
    compare_script: Path,
    scan_id: int,
    high_ply: Path,
    low_ply: Path,
    png_path: Path,
    score_json: Path,
    heatmap_channels: int,
) -> Optional[np.ndarray]:
    """Blocking compare + load of the small ML-ready heatmap grid.

    Mirrors ``umift_env_runner.run_compare_pointcloud()``'s subprocess
    pattern (invokes the same ``compare_pointcloud_distance.py``), but
    additionally loads the small ML grid it writes
    (``<png-stem>_small.npy``, see ``save_small_ml_grid``) and returns it
    replicated to ``heatmap_channels`` channels as ``(C, H, W)`` float32 --
    ready for ``ManipServerLayupEnv.set_latest_scan_heatmap()``. Safe to run
    in a background thread; called via ``ThreadPoolExecutor.submit`` from
    ``ScanPauseController.do_scan()``.

    Returns None (rather than raising) on any failure, so a bad compare
    never corrupts the live heatmap slot -- the caller's done-callback just
    skips updating it and logs.
    """
    print(f"[ScanPauseController] [bg] scan {scan_id}: comparing point clouds...")
    try:
        cmp_proc = subprocess.run(
            [
                str(keyence_python),
                str(compare_script),
                "--high",
                str(high_ply),
                "--low",
                str(low_ply),
                "--no-show",
                "--png",
                str(png_path),
                "--score-json",
                str(score_json),
            ],
            cwd=str(keyence_ws),
            capture_output=True,
            text=True,
            check=False,
        )
        if cmp_proc.stdout:
            print(cmp_proc.stdout, flush=True)
        if cmp_proc.stderr:
            print(cmp_proc.stderr, flush=True)
        if cmp_proc.returncode != 0:
            raise RuntimeError(
                f"compare_pointcloud_distance failed with code {cmp_proc.returncode}"
            )
        small_path = png_path.with_name(png_path.stem + "_small.npy")
        if not small_path.is_file():
            raise FileNotFoundError(f"Expected small ML grid missing: {small_path}")
        grid_mm = np.load(small_path).astype(np.float32)  # (H, W); NaN where invalid
        grid_mm = np.nan_to_num(grid_mm, nan=0.0)
        heatmap = np.repeat(grid_mm[None, ...], heatmap_channels, axis=0)  # (C, H, W)
        print(
            f"[ScanPauseController] [bg] scan {scan_id}: heatmap ready {heatmap.shape}"
        )
        return heatmap.astype(np.float32)
    except Exception as exc:  # noqa: BLE001 - never let a bad compare propagate
        print(f"[ScanPauseController] [bg] scan {scan_id}: compare FAILED: {exc}")
        return None


class ScanPauseController:
    """Owns the ``scan_daemon.py`` subprocess and orchestrates one
    mid-episode laser-scan interrupt at a time.

    Usage (mirrors the ``start_episode``/per-horizon/``stop_episode``
    lifecycle of an episode in ``umift_env_runner.main``)::

        controller = ScanPauseController(env=env, ...)
        ...
        controller.start_episode()          # once, at episode start
        ...
        outcome = controller.do_scan(scan_id)   # every ~30s, mid-episode
        ...
        controller.stop_episode()           # once, at episode end
    """

    def __init__(
        self,
        *,
        env,
        keyence_python: Path,
        keyence_ws: Path,
        right_arm_ip: str,
        compare_low_ply: Path,
        scan_config: Optional[Path] = None,
        layup_config: Optional[Path] = None,
        heatmap_channels: int = 3,
        daemon_ready_timeout_s: float = 15.0,
        sweep_timeout_s: float = 60.0,
        return_right_timeout_s: float = 30.0,
        stop_timeout_s: float = 10.0,
        compare_executor: Optional[ThreadPoolExecutor] = None,
        daemon_script: Optional[Path] = None,
        compare_script: Optional[Path] = None,
        daemon_extra_args: Optional[List[str]] = None,
        read_joints_fn: Optional[Callable[[], List[float]]] = None,
    ) -> None:
        """
        Args:
            env: the live env; must expose ``pause_rtde_control()`` /
                ``resume_rtde_control()`` (see ``ManipServerEnv``).
            keyence_python: interpreter in the keyence_scan_ws venv, used to
                launch scan_daemon.py and compare_pointcloud_distance.py.
            keyence_ws: keyence_scan_ws root (cwd for the subprocesses).
            right_arm_ip: IP of the right/task arm, for the throwaway RTDE
                receive connection used to snapshot pre-scan joint angles.
            compare_low_ply: reference/"low" surface PLY, passed as
                ``--low`` to compare_pointcloud_distance.py.
            heatmap_channels: channel count for the live heatmap slot passed
                to set_latest_scan_heatmap (3, to match shape_meta's
                scan_heatmap_0: [3, 80, 160] in umift_zucchini.yaml).
            compare_executor: shared ThreadPoolExecutor to submit compares
                to. If omitted, an internal one is created (and shut down
                by ``close()``).
            daemon_script / compare_script: overridable for tests (point at
                a fake stand-in that speaks the same stdout protocol /
                writes the same output files).
            daemon_extra_args: extra argv appended after daemon_script when
                launching it; only useful for tests driving a fake daemon
                that takes args scan_daemon.py doesn't (e.g. a fake
                --session-dir to report). Real scan_daemon.py usage should
                leave this None.
            read_joints_fn: overridable for tests; defaults to a real RTDE
                receive query against ``right_arm_ip``.
        """
        self.env = env
        self.keyence_python = Path(keyence_python)
        self.keyence_ws = Path(keyence_ws)
        self.right_arm_ip = str(right_arm_ip)
        self.compare_low_ply = Path(compare_low_ply)
        self.scan_config = Path(scan_config) if scan_config else None
        self.layup_config = Path(layup_config) if layup_config else None
        self.heatmap_channels = int(heatmap_channels)
        self.daemon_ready_timeout_s = float(daemon_ready_timeout_s)
        self.sweep_timeout_s = float(sweep_timeout_s)
        self.return_right_timeout_s = float(return_right_timeout_s)
        self.stop_timeout_s = float(stop_timeout_s)
        self.daemon_script = (
            Path(daemon_script)
            if daemon_script
            else (self.keyence_ws / "scripts" / "scan_daemon.py")
        )
        self.compare_script = (
            Path(compare_script)
            if compare_script
            else (self.keyence_ws / "utils" / "compare_pointcloud_distance.py")
        )
        self._read_joints_fn = read_joints_fn or (
            lambda: _read_right_joints_via_rtde(self.keyence_python, self.right_arm_ip)
        )
        self._daemon_extra_args = list(daemon_extra_args) if daemon_extra_args else []

        self._own_executor = compare_executor is None
        self._compare_executor = compare_executor or ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="scan-pause-compare"
        )

        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._line_queue: "queue.Queue[str]" = queue.Queue()
        self._available = False

    @property
    def available(self) -> bool:
        """True once start_episode() has succeeded and stop_episode() has
        not yet been called (and the daemon hasn't been observed to die)."""
        return self._available

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start_episode(self) -> bool:
        """Launch scan_daemon.py and wait for SCAN_DAEMON_READY.

        Never raises: on any failure (spawn error, timeout, early exit),
        logs clearly and leaves ``self.available`` False so ``do_scan()``
        becomes a cheap no-op for the rest of the episode instead of
        crashing inference.
        """
        if self._proc is not None and self._proc.poll() is None:
            print("[ScanPauseController] start_episode: daemon already running.")
            return self._available

        cmd = [str(self.keyence_python), str(self.daemon_script)]
        if self.scan_config is not None:
            cmd += ["--scan-config", str(self.scan_config)]
        if self.layup_config is not None:
            cmd += ["--layup-config", str(self.layup_config)]
        cmd += self._daemon_extra_args

        print(f"[ScanPauseController] start_episode: launching {' '.join(cmd)}")
        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(self.keyence_ws),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            print(f"[ScanPauseController] start_episode: FAILED to launch daemon: {exc}")
            self._proc = None
            self._available = False
            return False

        self._line_queue = queue.Queue()
        self._reader_thread = threading.Thread(
            target=self._stdout_reader_loop,
            name="ScanDaemonStdoutReader",
            daemon=True,
        )
        self._reader_thread.start()

        line = self._wait_for_prefixes(
            ("SCAN_DAEMON_READY",), self.daemon_ready_timeout_s
        )
        if line is None:
            print(
                "[ScanPauseController] start_episode: daemon did not report "
                f"SCAN_DAEMON_READY within {self.daemon_ready_timeout_s}s "
                "(or exited early). Scanning UNAVAILABLE for this episode; "
                "inference will continue without periodic scans."
            )
            self._kill_proc()
            self._available = False
            return False

        print("[ScanPauseController] start_episode: daemon ready.")
        self._available = True
        return True

    def stop_episode(self) -> None:
        """Send STOP and wait for SCAN_DAEMON_STOPPED; kill as a fallback.

        Always safe to call, including when start_episode() never
        succeeded (no-op) or was already stopped (no-op).
        """
        self._available = False
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is not None:
            self._proc = None
            return

        try:
            self._send("STOP")
        except Exception as exc:
            print(f"[ScanPauseController] stop_episode: failed to send STOP: {exc}")

        line = self._wait_for_prefixes(("SCAN_DAEMON_STOPPED",), self.stop_timeout_s)
        if line is None:
            print(
                "[ScanPauseController] stop_episode: no SCAN_DAEMON_STOPPED "
                f"within {self.stop_timeout_s}s; killing daemon process."
            )
        self._kill_proc()

    def close(self) -> None:
        """Release owned resources. Only shuts down an internally-created
        compare executor (one passed in by the caller is left alone -- the
        caller owns its lifecycle)."""
        if self._own_executor:
            self._compare_executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # The main operation
    # ------------------------------------------------------------------

    def do_scan(self, scan_id: int) -> ScanOutcome:
        """Pause RTDE, sweep, submit a background heatmap compare, restore
        the right arm to its pre-scan pose, resume RTDE.

        Blocking. See module docstring for why this is sufficient to keep
        the policy's observation history uncorrupted (caller must simply
        not call env.get_observation_from_buffer() / env.schedule_controls()
        again until this returns -- which is automatic, since the caller's
        thread is inside this call the whole time), and for the
        try/finally resume guarantee.
        """
        if self._proc is None or not self._available or self._proc.poll() is not None:
            return ScanOutcome(
                scan_id=scan_id,
                success=False,
                error="scan daemon unavailable (not started / crashed / stopped)",
            )

        # Snapshot pre-scan joints BEFORE pausing anything. If this fails we
        # bail out having touched nothing -- RTDE control was never paused,
        # so there is nothing to resume.
        try:
            pre_scan_joints = self._read_joints_fn()
        except Exception as exc:
            print(
                f"[ScanPauseController] do_scan {scan_id}: failed to read right "
                f"arm joints, aborting scan (RTDE control was never paused): {exc}"
            )
            return ScanOutcome(
                scan_id=scan_id, success=False, error=f"joint read failed: {exc}"
            )

        print(f"[ScanPauseController] do_scan {scan_id}: pausing RTDE control...")
        if not self.env.pause_rtde_control():
            print(
                f"[ScanPauseController] do_scan {scan_id}: pause_rtde_control() "
                "returned False; aborting scan."
            )
            return ScanOutcome(
                scan_id=scan_id, success=False, error="pause_rtde_control failed"
            )

        session_dir: Optional[Path] = None
        t_start: Optional[float] = None
        t_end: Optional[float] = None
        heatmap_future: Optional[Future] = None
        error: Optional[str] = None
        try:
            try:
                self._send(f"SWEEP {scan_id}")
                line = self._wait_for_prefixes(
                    ("SCAN_RESULT", "SCAN_ERROR"), self.sweep_timeout_s
                )
                if line is None:
                    raise RuntimeError(
                        f"no SWEEP response within {self.sweep_timeout_s}s "
                        "(daemon may have crashed)"
                    )
                if line.startswith("SCAN_ERROR"):
                    raise RuntimeError(f"SWEEP failed: {line}")

                session_dir, t_start, t_end = self._parse_scan_result(line)

                png_path = session_dir / "distance_heatmap.png"
                score_json = session_dir / "eval_score.json"
                high_ply = session_dir / "merged.ply"
                heatmap_future = self._compare_executor.submit(
                    _run_compare_and_load_heatmap,
                    keyence_python=self.keyence_python,
                    keyence_ws=self.keyence_ws,
                    compare_script=self.compare_script,
                    scan_id=scan_id,
                    high_ply=high_ply,
                    low_ply=self.compare_low_ply,
                    png_path=png_path,
                    score_json=score_json,
                    heatmap_channels=self.heatmap_channels,
                )

                joints_json = json.dumps([float(q) for q in pre_scan_joints])
                self._send(f"RETURN_RIGHT {joints_json}")
                line2 = self._wait_for_prefixes(
                    ("RETURN_RIGHT_DONE", "SCAN_ERROR"), self.return_right_timeout_s
                )
                if line2 is None:
                    raise RuntimeError(
                        f"no RETURN_RIGHT response within "
                        f"{self.return_right_timeout_s}s"
                    )
                if line2.startswith("SCAN_ERROR"):
                    raise RuntimeError(f"RETURN_RIGHT failed: {line2}")
            except Exception as exc:
                error = str(exc)
                print(f"[ScanPauseController] do_scan {scan_id}: FAILED: {exc}")
        finally:
            # MUST always run, whatever happened above: a scan going wrong
            # must never leave the robot arm paused indefinitely.
            print(f"[ScanPauseController] do_scan {scan_id}: resuming RTDE control...")
            try:
                resumed = self.env.resume_rtde_control()
                if not resumed:
                    resume_err = "resume_rtde_control() returned False"
                    print(f"[ScanPauseController] do_scan {scan_id}: {resume_err}")
                    error = error or resume_err
            except Exception as resume_exc:
                print(
                    f"[ScanPauseController] do_scan {scan_id}: resume_rtde_control "
                    f"raised: {resume_exc}"
                )
                error = error or f"resume_rtde_control raised: {resume_exc}"

        return ScanOutcome(
            scan_id=scan_id,
            success=error is None,
            session_dir=session_dir,
            t_start=t_start,
            t_end=t_end,
            heatmap_future=heatmap_future,
            error=error,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_scan_result(line: str) -> Tuple[Path, float, float]:
        # "SCAN_RESULT scan_id=<id> session_dir=<path> t_start=<f> t_end=<f>"
        fields = {}
        for tok in line.split()[1:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                fields[k] = v
        if not {"session_dir", "t_start", "t_end"} <= fields.keys():
            raise RuntimeError(f"malformed SCAN_RESULT line: {line!r}")
        return Path(fields["session_dir"]), float(fields["t_start"]), float(fields["t_end"])

    def _send(self, line: str) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("scan daemon process/stdin not available")
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"failed writing to scan daemon stdin: {exc}") from exc

    def _stdout_reader_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            for raw_line in self._proc.stdout:
                line = raw_line.rstrip("\n")
                if line:
                    self._line_queue.put(line)
        except Exception:
            pass
        finally:
            self._line_queue.put(_EOF_SENTINEL)

    def _wait_for_prefixes(
        self, prefixes: Tuple[str, ...], timeout_s: float
    ) -> Optional[str]:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                line = self._line_queue.get(timeout=remaining)
            except queue.Empty:
                return None
            if line == _EOF_SENTINEL:
                print("[ScanPauseController] daemon stdout closed (process exited).")
                return None
            if any(line.startswith(p) for p in prefixes):
                return line
            print(f"[ScanPauseController] (ignoring unexpected daemon line: {line!r})")

    def _kill_proc(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
