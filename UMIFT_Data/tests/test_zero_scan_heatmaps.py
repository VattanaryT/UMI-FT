import importlib.util
import pathlib
import tempfile

import numpy as np
import zarr


SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "zero_scan_heatmaps.py"
)
SPEC = importlib.util.spec_from_file_location("zero_scan_heatmaps", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_zero_scan_heatmaps_leaves_maps_and_gaps():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "buf.zarr"
        root = zarr.open(str(path), mode="w")
        data = root.create_group("data")
        ep = data.create_group("episode_0")
        heatmap = np.arange(2 * 4 * 8 * 3, dtype=np.float32).reshape(2, 4, 8, 3)
        ep.create_dataset("scan_heatmap_0", data=heatmap)
        ep.create_dataset("map_to_scan_idx_0", data=np.array([0, 0, 1, 1], dtype=np.int64))
        ep.create_dataset("scan_gap_intervals_0", data=np.array([[1.0, 2.0]], dtype=np.float32))
        data.create_group("episode_1")  # no scan keys

        n = MODULE.zero_scan_heatmaps(zarr.open(str(path), mode="r+"))
        out = zarr.open(str(path), mode="r")
        assert n == 1
        assert np.all(out["data/episode_0/scan_heatmap_0"][:] == 0)
        assert np.array_equal(
            out["data/episode_0/map_to_scan_idx_0"][:], [0, 0, 1, 1]
        )
        assert np.array_equal(
            out["data/episode_0/scan_gap_intervals_0"][:], [[1.0, 2.0]]
        )
