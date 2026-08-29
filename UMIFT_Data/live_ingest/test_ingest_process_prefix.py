#!/usr/bin/env python3
"""Tests that ingest_server invokes process_local_recordings with --prefix."""

from __future__ import annotations

import stat
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ingest_server import IngestServer, ProcessingJob  # noqa: E402


class ProcessPrefixArgTests(unittest.TestCase):
    def test_worker_passes_prefix_to_process_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            argv_log = tmp_path / "argv.txt"
            stub = tmp_path / "stub.sh"
            stub.write_text(
                "#!/usr/bin/env bash\n"
                f'printf "%s\\n" "$@" > "{argv_log}"\n'
            )
            stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

            server = IngestServer(
                local_dir=str(tmp_path / "data"),
                folder_name="session",
                host="127.0.0.1",
                port=0,
                process_script=str(stub),
            )
            job = ProcessingJob(prefix="traj_a")
            server.job_queue.put(job)
            server.job_queue.put(None)
            server._worker_loop()

            self.assertTrue(job.success, job.error)
            recorded = argv_log.read_text().splitlines()
            self.assertIn("--skip-existing", recorded)
            self.assertIn("--prefix", recorded)
            prefix_idx = recorded.index("--prefix")
            self.assertEqual(recorded[prefix_idx + 1], "traj_a")
            self.assertIn("session", recorded)


if __name__ == "__main__":
    unittest.main()
