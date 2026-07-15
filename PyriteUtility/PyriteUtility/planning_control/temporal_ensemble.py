"""Temporal ensemble of overlapping action-chunk predictions.

Each MPC query predicts a full sparse action horizon but only advances by
``sparse_execution_horizon`` steps. Keeping the unused future steps and blending
them with later overlapping queries removes hard chunk-boundary discontinuities
without adding execution latency.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np


def make_horizon_weights(
    horizon: int,
    mode: str = "exp",
    decay_m: float = 0.0,
) -> np.ndarray:
    """Per-horizon-position weights for a freshly predicted action chunk.

    Args:
        horizon: Number of sparse action steps in one prediction.
        mode: ``"exp"`` uses ``w[i] = exp(-decay_m * i)`` (near-term trusted more).
            ``"uniform"`` uses ones.
        decay_m: Exponential decay rate when ``mode == "exp"``. ``0`` is uniform.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    if mode == "uniform" or decay_m == 0.0:
        return np.ones(horizon, dtype=np.float64)
    if mode != "exp":
        raise ValueError(f"Unsupported weight mode: {mode}")
    i = np.arange(horizon, dtype=np.float64)
    return np.exp(-float(decay_m) * i)


def _align_quat_sign(quat: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip ``quat`` if needed so it lies in the same hemisphere as ``reference``."""
    if float(np.dot(quat, reference)) < 0.0:
        return -quat
    return quat


class PoseChunkEnsembler:
    """Accumulate overlapping pose/scalar action predictions and pop blended steps.

    Pose fields are expected as ``(H, 7)`` arrays ``[x, y, z, qw, qx, qy, qz]``.
    Quaternion contributions are sign-disambiguated against the running sum, then
    re-normalized when popped. Linear fields (stiffness, gripper, ...) are
    weighted-sum / weight-sum averaged.
    """

    def __init__(
        self,
        pose_keys: Sequence[str] = ("pose7",),
        linear_keys: Sequence[str] = (),
        weight_mode: str = "exp",
        decay_m: float = 0.0,
    ):
        self.pose_keys = tuple(pose_keys)
        self.linear_keys = tuple(linear_keys)
        self.weight_mode = weight_mode
        self.decay_m = float(decay_m)

        # Absolute step -> accumulators.
        self._weights: Dict[int, float] = {}
        self._pose_sums: Dict[str, Dict[int, np.ndarray]] = {
            k: {} for k in self.pose_keys
        }
        self._linear_sums: Dict[str, Dict[int, np.ndarray]] = {
            k: {} for k in self.linear_keys
        }
        self._next_pop_step = 0

    def reset(self) -> None:
        self._weights.clear()
        for d in self._pose_sums.values():
            d.clear()
        for d in self._linear_sums.values():
            d.clear()
        self._next_pop_step = 0

    def push(
        self,
        start_step: int,
        fields: Mapping[str, Optional[np.ndarray]],
        weights: Optional[np.ndarray] = None,
    ) -> None:
        """Accumulate one predicted chunk starting at absolute ``start_step``.

        Args:
            start_step: Absolute sparse-step index of the chunk's first row.
            fields: Mapping of field name -> array. Pose keys: ``(H, 7)``.
                Linear keys: ``(H,)`` or ``(H, D)``. Missing / ``None`` values
                are skipped for that chunk.
            weights: Optional length-``H`` weights. Defaults to
                :func:`make_horizon_weights` for the chunk length.
        """
        # Infer horizon from the first available field.
        horizon = None
        for key in list(self.pose_keys) + list(self.linear_keys):
            arr = fields.get(key)
            if arr is not None:
                arr = np.asarray(arr)
                horizon = int(arr.shape[0])
                break
        if horizon is None or horizon <= 0:
            raise ValueError("push() requires at least one non-empty field")

        if weights is None:
            w = make_horizon_weights(horizon, mode=self.weight_mode, decay_m=self.decay_m)
        else:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w.shape[0] != horizon:
                raise ValueError(
                    f"weights length {w.shape[0]} != chunk horizon {horizon}"
                )

        for i in range(horizon):
            step = int(start_step + i)
            wi = float(w[i])
            if wi <= 0.0:
                continue
            self._weights[step] = self._weights.get(step, 0.0) + wi

            for key in self.pose_keys:
                arr = fields.get(key)
                if arr is None:
                    continue
                pose = np.asarray(arr[i], dtype=np.float64).reshape(7)
                pos = pose[:3]
                quat = pose[3:7].copy()
                nq = float(np.linalg.norm(quat))
                if nq > 0.0:
                    quat /= nq

                pose_map = self._pose_sums[key]
                if step not in pose_map:
                    pose_map[step] = np.concatenate([wi * pos, wi * quat])
                else:
                    acc = pose_map[step]
                    acc[:3] += wi * pos
                    # Keep quaternions in a consistent hemisphere before summing.
                    quat = _align_quat_sign(quat, acc[3:7])
                    acc[3:7] += wi * quat

            for key in self.linear_keys:
                arr = fields.get(key)
                if arr is None:
                    continue
                val = np.asarray(arr[i], dtype=np.float64)
                lin_map = self._linear_sums[key]
                if step not in lin_map:
                    lin_map[step] = wi * val
                else:
                    lin_map[step] = lin_map[step] + wi * val

    def pop(self, n: int) -> Dict[str, np.ndarray]:
        """Finalize and return the next ``n`` absolute steps, then evict them."""
        if n <= 0:
            raise ValueError(f"pop n must be > 0, got {n}")

        out: Dict[str, np.ndarray] = {}
        pose_bufs = {
            k: np.zeros((n, 7), dtype=np.float64) for k in self.pose_keys
        }
        linear_bufs: Dict[str, Optional[np.ndarray]] = {
            k: None for k in self.linear_keys
        }

        for i in range(n):
            step = self._next_pop_step + i
            if step not in self._weights:
                raise KeyError(
                    f"Cannot pop absolute step {step}: no accumulated predictions. "
                    f"Did you call push() first? next_pop_step={self._next_pop_step}"
                )
            w_sum = self._weights[step]
            if w_sum <= 0.0:
                raise RuntimeError(f"Non-positive weight sum at step {step}: {w_sum}")

            for key in self.pose_keys:
                acc = self._pose_sums[key][step]
                pos = acc[:3] / w_sum
                quat = acc[3:7].copy()
                nq = float(np.linalg.norm(quat))
                if nq > 0.0:
                    quat /= nq
                else:
                    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                pose_bufs[key][i] = np.concatenate([pos, quat])

            for key in self.linear_keys:
                if step not in self._linear_sums[key]:
                    continue
                val = self._linear_sums[key][step] / w_sum
                if linear_bufs[key] is None:
                    linear_bufs[key] = np.zeros((n,) + val.shape, dtype=np.float64)
                linear_bufs[key][i] = val

            # Evict.
            del self._weights[step]
            for key in self.pose_keys:
                self._pose_sums[key].pop(step, None)
            for key in self.linear_keys:
                self._linear_sums[key].pop(step, None)

        self._next_pop_step += n
        out.update(pose_bufs)
        for key, buf in linear_bufs.items():
            if buf is not None:
                out[key] = buf
        return out

    @property
    def buffered_steps(self) -> int:
        return len(self._weights)

    @property
    def next_pop_step(self) -> int:
        return self._next_pop_step


def ensemble_chunks_offline(
    chunks: Sequence[Mapping[str, Optional[np.ndarray]]],
    execution_horizon: int,
    pose_keys: Sequence[str] = ("pose7",),
    linear_keys: Sequence[str] = (),
    weight_mode: str = "exp",
    decay_m: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Convenience helper: push a sequence of full chunks and pop executed steps.

    Each entry in ``chunks`` is a full prediction of length H. Only the first
    ``execution_horizon`` blended steps are retained after each push (same
    zero-added-latency schedule as online MPC).
    """
    ensembler = PoseChunkEnsembler(
        pose_keys=pose_keys,
        linear_keys=linear_keys,
        weight_mode=weight_mode,
        decay_m=decay_m,
    )
    executed: Dict[str, list] = {}
    start_step = 0
    for chunk in chunks:
        ensembler.push(start_step=start_step, fields=chunk)
        blended = ensembler.pop(execution_horizon)
        for key, arr in blended.items():
            executed.setdefault(key, []).append(arr)
        start_step += execution_horizon

    return {k: np.concatenate(v, axis=0) for k, v in executed.items()}
