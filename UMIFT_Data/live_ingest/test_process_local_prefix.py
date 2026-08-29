#!/usr/bin/env python3
"""CLI tests for process_local_recordings.sh --prefix."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "bash"
    / "process_local_recordings.sh"
)


class ProcessLocalPrefixCliTests(unittest.TestCase):
    def test_prefix_is_accepted(self) -> None:
        result = subprocess.run(
            ["bash", str(SCRIPT), "no_such_folder", "--prefix", "traj_a"],
            capture_output=True,
            text=True,
        )
        combined = result.stdout + result.stderr
        self.assertNotIn("Unknown option", combined)
        self.assertIn("not found", combined.lower())
        self.assertNotEqual(result.returncode, 0)

    def test_help_mentions_prefix(self) -> None:
        result = subprocess.run(
            ["bash", str(SCRIPT), "ignored", "-h"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--prefix", result.stdout)


if __name__ == "__main__":
    unittest.main()
