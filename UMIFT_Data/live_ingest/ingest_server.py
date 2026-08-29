#!/usr/bin/env python3
"""
Live ethernet ingest -- PC-side server.

Started manually before a collection session (see
bash/start_ingest_server.sh), not as an always-on systemd service. Accepts
control-channel connections from Robotiq-portable-real2sim/python/live_ingest/uplink_daemon.py
running on the data collection device. After each trajectory's checksums
match, it sends VERIFIED so the device can start rsync of the next file,
then runs process_local_recordings.sh (unified zarr + AprilTag transform)
for that prefix before sending SAFE_TO_DELETE.

Does NOT run the whole-dataset combine step (bash/combine_umift_dataset.sh)
-- that stays a manual step you run yourself once collection is done.

Usage:
  python3 live_ingest/ingest_server.py --folder-name <name>
"""

from __future__ import annotations

import argparse
import os
import queue
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Optional

# Reach across to the Pi-side package for the shared wire protocol. Both
# repos are checked out side by side in the umi_ft_ws workspace on the PC
# (the device only has Robotiq-portable-real2sim, which is why the
# protocol module lives there and this side imports it via path, same
# cross-repo PYTHONPATH convention already used by the bash scripts here).
_PROTOCOL_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..",
        "Robotiq-portable-real2sim", "python", "live_ingest",
    )
)
if _PROTOCOL_DIR not in sys.path:
    sys.path.insert(0, _PROTOCOL_DIR)
from protocol import (  # noqa: E402
    Connection,
    DEFAULT_PORT,
    Dispatcher,
    MessageType,
    ProtocolError,
    ProtocolTimeout,
    sha256_file,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESS_LOCAL_SCRIPT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "bash", "process_local_recordings.sh"))

# Hard cap on one process_local_recordings.sh invocation (Docker zarr
# convert + AprilTag transform for one trajectory). Was previously
# unbounded -- a wedged Docker container would hang the single processing
# worker forever, which in turn wedges every device connection waiting on
# it (job.done_event.wait() had no timeout either -- see below).
PROCESS_SCRIPT_TIMEOUT_S = 1800  # 30 min; generous for one ~5GB trajectory
# How often to heartbeat a connection that's blocked waiting on the
# processing worker, so the device's Connection idle timeout (90s default,
# see protocol.IDLE_TIMEOUT_S) doesn't fire while a multi-minute Docker job
# is still legitimately running.
JOB_WAIT_HEARTBEAT_S = 20.0


@dataclass
class ProcessingJob:
    prefix: str
    done_event: threading.Event = field(default_factory=threading.Event)
    success: Optional[bool] = None
    stage: str = "process_local_recordings.sh"
    error: Optional[str] = None


class IngestServer:
    def __init__(self, local_dir: str, folder_name: str, host: str, port: int, process_script: str = PROCESS_LOCAL_SCRIPT):
        self.local_dir = local_dir
        self.folder_name = folder_name
        self.host = host
        self.port = port
        self.process_script = process_script
        self.job_queue: "queue.Queue[Optional[ProcessingJob]]" = queue.Queue()

        os.makedirs(self.local_dir, exist_ok=True)

        self.dispatcher = Dispatcher()
        self.dispatcher.register(MessageType.HELLO, self._handle_hello)
        self.dispatcher.register(MessageType.TRAJECTORY_READY, self._handle_trajectory_ready)
        self.dispatcher.register(MessageType.TRANSFER_DONE, self._handle_transfer_done)
        self.dispatcher.register(MessageType.COLLECTION_STARTED, self._handle_collection_started)

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)

    # ---- message handlers ----
    def _handle_hello(self, conn: Connection, payload: dict) -> None:
        device_folder = payload.get("folder_name")
        print(
            f"[ingest] HELLO from {payload.get('device_id', '?')} "
            f"(folder_name={device_folder!r}, protocol_version={payload.get('protocol_version')})"
        )
        if device_folder and device_folder != self.folder_name:
            print(
                f"[ingest] WARNING: device folder_name {device_folder!r} != "
                f"server folder_name {self.folder_name!r} -- files still land under {self.local_dir}"
            )

    def _handle_collection_started(self, conn: Connection, payload: dict) -> None:
        # Fire-and-forget notification, no reply. Reserved hook for future
        # laser-scan integration (see protocol.py) -- today, log only.
        print(f"[ingest] Collection started: prefix={payload.get('prefix')!r}")

    def _handle_trajectory_ready(self, conn: Connection, payload: dict) -> None:
        prefix = payload["prefix"]
        files = payload["files"]
        print(f"[ingest] {prefix}: trajectory ready ({len(files)} file(s)), requesting transfer")
        # One trajectory is handled per connection (the daemon reconnects
        # fresh for each), so stashing the manifest on the connection object
        # is safe -- no risk of a second trajectory overwriting it mid-flight.
        conn.pending_manifest = {f["rel_path"]: f for f in files}  # type: ignore[attr-defined]
        conn.send(MessageType.START_TRANSFER, {"dest": self.local_dir})
        print(f"[ingest] {prefix}: START_TRANSFER sent, awaiting rsync (TRANSFER_DONE)")

    def _handle_transfer_done(self, conn: Connection, payload: dict) -> None:
        prefix = payload["prefix"]
        print(f"[ingest] {prefix}: TRANSFER_DONE received, verifying checksums")
        manifest = getattr(conn, "pending_manifest", None)
        if manifest is None:
            print(f"[ingest] {prefix}: TRANSFER_DONE with no prior TRAJECTORY_READY on this connection; rejecting")
            conn.send(MessageType.TRANSFER_RETRY, {"prefix": prefix, "reason": "no manifest on this connection"})
            return

        mismatch = self._verify_manifest(manifest)
        if mismatch:
            print(f"[ingest] {prefix}: checksum verification failed: {mismatch}")
            conn.send(MessageType.TRANSFER_RETRY, {"prefix": prefix, "reason": mismatch})
            return

        print(f"[ingest] {prefix}: transfer verified")
        conn.send(MessageType.VERIFIED, {"prefix": prefix})

        job = ProcessingJob(prefix=prefix)
        self.job_queue.put(job)
        # This connection waits for processing; other connections can rsync
        # the next trajectory meanwhile. Heartbeat while waiting so the
        # device's idle timeout doesn't fire during a legitimately slow
        # (but alive) multi-minute Docker job -- see JOB_WAIT_HEARTBEAT_S.
        conn.start_heartbeat(JOB_WAIT_HEARTBEAT_S)
        try:
            job.done_event.wait()
        finally:
            conn.stop_heartbeat()

        if job.success:
            print(f"[ingest] {prefix}: per-trajectory processing succeeded -- safe to delete on device")
            conn.send(MessageType.SAFE_TO_DELETE, {"prefix": prefix})
        else:
            print(f"[ingest] {prefix}: per-trajectory processing FAILED ({job.stage}: {job.error})")
            conn.send(MessageType.PROCESSING_FAILED, {"prefix": prefix, "stage": job.stage, "error": job.error})

    def _verify_manifest(self, manifest: dict) -> Optional[str]:
        for rel_path, meta in manifest.items():
            local_path = os.path.join(self.local_dir, rel_path)
            if not os.path.isfile(local_path):
                return f"missing file after transfer: {rel_path}"
            actual_size = os.path.getsize(local_path)
            if actual_size != meta["size"]:
                return f"size mismatch for {rel_path}: expected {meta['size']}, got {actual_size}"
            if sha256_file(local_path) != meta["sha256"]:
                return f"sha256 mismatch for {rel_path}"
        return None

    # ---- single-worker processing queue ----
    def _worker_loop(self) -> None:
        while True:
            job = self.job_queue.get()
            if job is None:
                return
            print(f"[ingest] {job.prefix}: running {os.path.basename(self.process_script)} ...")
            try:
                result = subprocess.run(
                    [
                        "bash",
                        self.process_script,
                        self.folder_name,
                        "--skip-existing",
                        "--prefix",
                        job.prefix,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=PROCESS_SCRIPT_TIMEOUT_S,
                )
                if result.returncode != 0:
                    tail = (result.stderr.strip() or result.stdout.strip())[-2000:]
                    raise RuntimeError(f"exit {result.returncode}: {tail}")
                job.success = True
            except subprocess.TimeoutExpired:
                # Previously unbounded: a wedged Docker container hung this
                # single worker forever, which hung every device connection
                # waiting on job.done_event (D6). Now it times out, reports
                # failure (raw data stays on the device, PROCESSING_FAILED
                # -> retried later), and the worker moves on to the next job.
                job.success = False
                job.error = f"timed out after {PROCESS_SCRIPT_TIMEOUT_S}s"
            except Exception as e:  # noqa: BLE001 - report to device, keep worker alive for the next job
                job.success = False
                job.error = str(e)
            finally:
                job.done_event.set()

    # ---- connection handling ----
    def _handle_connection(self, sock: socket.socket, addr) -> None:
        print(f"[ingest] Connection from {addr}")
        conn = Connection(sock)
        try:
            self.dispatcher.serve_forever(conn)
        except (ProtocolError, ProtocolTimeout) as e:
            print(f"[ingest] Protocol error from {addr}: {e}")
        finally:
            conn.close()
            print(f"[ingest] Connection from {addr} closed")

    def serve_forever(self) -> None:
        self._worker_thread.start()
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self.host, self.port))
        server_sock.listen(8)
        print(f"[ingest] Listening on {self.host}:{self.port} (folder_name={self.folder_name!r}, local_dir={self.local_dir})")
        try:
            while True:
                client_sock, addr = server_sock.accept()
                thread = threading.Thread(target=self._handle_connection, args=(client_sock, addr), daemon=True)
                thread.start()
        except KeyboardInterrupt:
            print("\n[ingest] Shutting down...")
        finally:
            self.job_queue.put(None)
            server_sock.close()


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)  # keep logs live when redirected to a file
    parser = argparse.ArgumentParser(description="Live ethernet ingest -- PC-side server")
    parser.add_argument("--folder-name", required=True, help="Session/folder name; data lands in <local-ft-root>/<folder_name>/")
    parser.add_argument("--local-ft-root", default=os.environ.get("LOCAL_FT_ROOT", "/mnt/data/umidata/ft"))
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Control port (default {DEFAULT_PORT})")
    parser.add_argument(
        "--process-script",
        default=PROCESS_LOCAL_SCRIPT,
        help="Override the per-trajectory processing script (mainly for testing with a stub)",
    )
    args = parser.parse_args()

    local_dir = os.path.join(args.local_ft_root, args.folder_name)
    server = IngestServer(
        local_dir=local_dir,
        folder_name=args.folder_name,
        host=args.host,
        port=args.port,
        process_script=args.process_script,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
