"""TCP position in the mold PCA frame. New file. Does not change the parent dataset."""

from __future__ import annotations

import numpy as np
import zarr

from fixb_r4vt_dataset import UmiFTDatasetR4VT


def mold_local(position: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """axes columns are the mold axes in the fixture frame."""
    return (np.asarray(position, dtype=np.float64) - origin) @ axes


class UmiFTDatasetFieldV2(UmiFTDatasetR4VT):
    """Same samples as the causal-foot dataset. robot0_eef_pos is mold-local."""

    def raw_episodes_conversion(self, replay_buffer_raw, shape_meta):
        replay = super().raw_episodes_conversion(replay_buffer_raw, shape_meta)
        spec = zarr.open(self._dataset_path, "r").attrs["field_v2_axes"]
        origin = np.asarray(spec["origin"], dtype=np.float64)
        axes = np.asarray(spec["axes"], dtype=np.float64)
        for episode in replay["data"].values():
            pos = episode["obs"]["robot0_eef_pos"]
            episode["obs"]["robot0_eef_pos"] = mold_local(pos, origin, axes).astype(np.float32)
        return replay


def _check() -> None:
    origin = np.zeros(3)
    axes = np.eye(3)
    got = mold_local(np.array([[2.0, 0.0, 0.0]]), origin, axes)
    if not np.allclose(got, [[2.0, 0.0, 0.0]]):
        raise RuntimeError("mold frame did not keep a point on its x axis")


if __name__ == "__main__":
    _check()
    print("field_v2_dataset ok")
