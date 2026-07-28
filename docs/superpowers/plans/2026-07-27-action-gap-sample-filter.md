# Action-Gap Sample Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exclude `SequenceSamplerUmiFT` training queries whose sparse action-target timestamps contain a gap larger than `3 × median(rgb_dt) × action_rgb_frame_stride`, and log how many queries were discarded.

**Architecture:** Add pure helpers next to `sparse_action_indices_from_rgb_timestamps` in `sampler_umift.py`. Filter candidate `rgb_id`s while building `self.indices` in `SequenceSamplerUmiFT.__init__`. Thread an optional multiplier through `UmiFTDataset` (default `3.0`; `<= 0` disables). Observation-history gaps are ignored.

**Tech Stack:** Python 3, NumPy, pytest, existing UMI-FT `SequenceSamplerUmiFT` / `UmiFTDataset`.

## Global Constraints

- Detect jumps via **timestamp gaps on action targets only** (not pose magnitude, not obs history).
- Default multiplier: **`3.0`**.
- Expected step: **`median(diff(rgb_times)) * shape_meta["sample"]["action"]["sparse"]["down_sample_steps"]`** (this config field is an RGB frame stride in UMI-FT).
- Log discarded vs total queries at sampler init.
- `action_gap_threshold_mult <= 0` disables filtering.
- Do **not** commit unless the user explicitly asks for a commit.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `UMI-FT/PyriteML/diffusion_policy/common/sampler_umift.py` | Gap helpers + filter during index build + logging |
| `UMI-FT/PyriteML/diffusion_policy/dataset/umift_dataset.py` | Pass `action_gap_threshold_mult` into sampler (train + val) |
| `UMI-FT/PyriteML/tests/test_sampler_umift_action_timing.py` | Unit tests for helpers / filter behavior |

No YAML change required for default behavior; optional later config knob can use the new dataset ctor arg.

---

### Task 1: Pure gap-check helpers + unit tests

**Files:**
- Modify: `UMI-FT/PyriteML/diffusion_policy/common/sampler_umift.py` (after `sparse_action_indices_from_rgb_timestamps`, before `get_val_mask`)
- Modify: `UMI-FT/PyriteML/tests/test_sampler_umift_action_timing.py`

**Interfaces:**
- Consumes: `sparse_action_indices_from_rgb_timestamps(action_times, rgb_times, rgb_id, action_horizon, rgb_frame_stride) -> (indices, target_times)`
- Produces:
  - `action_targets_have_timestamp_gap(target_times: np.ndarray, expected_dt: float, mult: float) -> bool`
  - `filter_rgb_ids_without_action_gaps(rgb_ids, action_times, rgb_times, action_horizon, rgb_frame_stride, mult) -> tuple[np.ndarray, int]`  
    returns `(kept_rgb_ids, n_discarded)`

- [ ] **Step 1: Write the failing tests**

Append to `UMI-FT/PyriteML/tests/test_sampler_umift_action_timing.py`:

```python
from diffusion_policy.common.sampler_umift import (
    action_targets_have_timestamp_gap,
    filter_rgb_ids_without_action_gaps,
    sparse_action_indices_from_rgb_timestamps,
)


def test_action_targets_have_timestamp_gap_detects_large_step():
    # expected_dt = 0.133s; 3x => 0.4s threshold
    target_times = np.array([0.0, 0.133, 0.266, 1.266])  # last step = 1.0s
    assert action_targets_have_timestamp_gap(target_times, expected_dt=0.133, mult=3.0)
    assert not action_targets_have_timestamp_gap(
        np.array([0.0, 0.133, 0.266, 0.399]), expected_dt=0.133, mult=3.0
    )


def test_action_targets_have_timestamp_gap_short_horizon_is_safe():
    assert not action_targets_have_timestamp_gap(
        np.array([0.0]), expected_dt=0.133, mult=3.0
    )
    assert not action_targets_have_timestamp_gap(
        np.array([0.0, 0.133]), expected_dt=0.133, mult=3.0
    )


def test_filter_rgb_ids_drops_queries_whose_horizon_spans_rgb_hole():
    # 15 Hz RGB with a 1.0s hole after frame 10
    rgb_normal = np.arange(0.0, 10.1 / 15.0, 1.0 / 15.0)  # frames 0..10
    rgb_after = rgb_normal[-1] + 1.0 + np.arange(1, 21) / 15.0
    rgb_times = np.concatenate([rgb_normal, rgb_after])
    action_times = np.arange(rgb_times[0], rgb_times[-1] + 1e-3, 0.005)

    # horizon=4, stride=2 spans 6 RGB frames of nominal time
    candidate_ids = np.arange(0, len(rgb_times) - 8)
    kept, n_discarded = filter_rgb_ids_without_action_gaps(
        rgb_ids=candidate_ids,
        action_times=action_times,
        rgb_times=rgb_times,
        action_horizon=4,
        rgb_frame_stride=2,
        mult=3.0,
    )
    assert n_discarded > 0
    assert len(kept) + n_discarded == len(candidate_ids)

    # A query far before the hole should be kept
    assert 0 in kept
    # A query whose targets cross the hole should be discarded
    # rgb_id near 8 with stride 2 reaches into post-hole frames
    assert 8 not in kept


def test_filter_rgb_ids_disabled_when_mult_nonpositive():
    rgb_times = np.array([0.0, 0.067, 0.134, 1.134, 1.201, 1.268])
    action_times = np.arange(0.0, 2.0, 0.005)
    rgb_ids = np.array([0, 1])
    kept, n_discarded = filter_rgb_ids_without_action_gaps(
        rgb_ids=rgb_ids,
        action_times=action_times,
        rgb_times=rgb_times,
        action_horizon=3,
        rgb_frame_stride=1,
        mult=0.0,
    )
    np.testing.assert_array_equal(kept, rgb_ids)
    assert n_discarded == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run from `UMI-FT/PyriteML`:

```bash
python -m pytest tests/test_sampler_umift_action_timing.py::test_action_targets_have_timestamp_gap_detects_large_step tests/test_sampler_umift_action_timing.py::test_filter_rgb_ids_drops_queries_whose_horizon_spans_rgb_hole -v
```

Expected: FAIL with `ImportError` / `not defined` for the new helpers.

- [ ] **Step 3: Implement the helpers**

Add to `sampler_umift.py` immediately after `sparse_action_indices_from_rgb_timestamps`:

```python
def action_targets_have_timestamp_gap(
    target_times: np.ndarray,
    expected_dt: float,
    mult: float,
) -> bool:
    """Return True if any consecutive action-target dt exceeds mult * expected_dt."""
    if mult <= 0:
        return False
    target_times = np.asarray(target_times, dtype=np.float64).reshape(-1)
    if target_times.size < 2:
        return False
    if not np.isfinite(expected_dt) or expected_dt <= 0:
        raise ValueError("expected_dt must be a positive finite float.")
    return bool(np.any(np.diff(target_times) > mult * expected_dt))


def filter_rgb_ids_without_action_gaps(
    rgb_ids: np.ndarray,
    action_times: np.ndarray,
    rgb_times: np.ndarray,
    action_horizon: int,
    rgb_frame_stride: int,
    mult: float,
) -> tuple[np.ndarray, int]:
    """Keep rgb query ids whose sparse action target times have no large gap.

    Returns (kept_rgb_ids, n_discarded).
    """
    rgb_ids = np.asarray(rgb_ids, dtype=np.int64).reshape(-1)
    if mult <= 0 or rgb_ids.size == 0:
        return rgb_ids.copy(), 0

    rgb_times = np.asarray(rgb_times, dtype=np.float64).reshape(-1)
    action_times = np.asarray(action_times, dtype=np.float64).reshape(-1)
    expected_dt = float(np.median(np.diff(rgb_times))) * float(rgb_frame_stride)

    kept = []
    n_discarded = 0
    for rgb_id in rgb_ids:
        _, target_times = sparse_action_indices_from_rgb_timestamps(
            action_times=action_times,
            rgb_times=rgb_times,
            rgb_id=int(rgb_id),
            action_horizon=action_horizon,
            rgb_frame_stride=rgb_frame_stride,
        )
        if action_targets_have_timestamp_gap(target_times, expected_dt, mult):
            n_discarded += 1
        else:
            kept.append(int(rgb_id))
    return np.asarray(kept, dtype=np.int64), n_discarded
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_sampler_umift_action_timing.py -v
```

Expected: all tests PASS. If `test_filter_rgb_ids_drops_queries_whose_horizon_spans_rgb_hole` fails on the exact `8 not in kept` assertion, adjust the asserted `rgb_id` to one that demonstrably spans the hole (print which discarded ids during debug), then lock that id into the test.

- [ ] **Step 5: Commit (only if user requested commits)**

```bash
# skip unless user asked to commit
```

---

### Task 2: Apply filter while building `self.indices`

**Files:**
- Modify: `UMI-FT/PyriteML/diffusion_policy/common/sampler_umift.py` (`SequenceSamplerUmiFT.__init__`)

**Interfaces:**
- Consumes: `filter_rgb_ids_without_action_gaps`, `shape_meta["sample"]["action"]["sparse"]["horizon"|"down_sample_steps"]`
- Produces: `self.indices` without action-gap queries; startup log of discard counts

- [ ] **Step 1: Add ctor arg**

In `SequenceSamplerUmiFT.__init__`, add after `correction_horizon`:

```python
        action_gap_threshold_mult: float = 3.0,
```

Store it:

```python
        self.action_gap_threshold_mult = float(action_gap_threshold_mult)
```

(near where `self.action_padding` is set is fine; can also set earlier before the index loop).

- [ ] **Step 2: Filter normal (non-mask) candidate ids before extending `epi_id`/`ids`**

Replace the block that currently does:

```python
                epi_id.extend(
                    ep_id_to_be_added[::sparse_query_frequency_down_sample_steps]
                )
                ids.extend(ids_to_be_added[::sparse_query_frequency_down_sample_steps])
```

with logic equivalent to:

```python
                candidate_rgb_ids = ids_to_be_added[
                    ::sparse_query_frequency_down_sample_steps
                ]
                action_horizon = shape_meta["sample"]["action"]["sparse"]["horizon"]
                action_rgb_frame_stride = shape_meta["sample"]["action"]["sparse"][
                    "down_sample_steps"
                ]
                action_times = np.squeeze(
                    replay_buffer["data"][key]["action_time_stamps"]
                )
                rgb_times = np.squeeze(
                    replay_buffer["data"][key]["obs"]["rgb_time_stamps_0"]
                )
                kept_rgb_ids, n_discarded = filter_rgb_ids_without_action_gaps(
                    rgb_ids=candidate_rgb_ids,
                    action_times=action_times,
                    rgb_times=rgb_times,
                    action_horizon=action_horizon,
                    rgb_frame_stride=action_rgb_frame_stride,
                    mult=self.action_gap_threshold_mult,
                )
                # accumulate for summary log
                gap_discard_total += n_discarded
                gap_candidate_total += len(candidate_rgb_ids)
                if n_discarded > 0:
                    gap_discard_by_episode[episode_index] = (
                        n_discarded,
                        len(candidate_rgb_ids),
                    )

                epi_id.extend([episode_index] * len(kept_rgb_ids))
                ids.extend(kept_rgb_ids.tolist())
```

Initialize before the episode loop:

```python
        gap_discard_total = 0
        gap_candidate_total = 0
        gap_discard_by_episode = {}
```

**Important:** The existing `weighted_sampling` / correction duplication block still uses `ids_to_be_added[::sparse_query_frequency_down_sample_steps]` as its base. Change it to operate on `kept_rgb_ids` (gap-filtered), not the pre-filter candidates, so duplicated correction samples also never include action gaps. Apply the same gap filter to any additional correction-horizon ids before extending.

- [ ] **Step 3: Filter mask-based query ids the same way**

In the `else` branch that builds `rgb_id_at_mask_start`, before `epi_id.extend` / `ids.extend`, run `filter_rgb_ids_without_action_gaps` on those rgb ids and accumulate the same counters.

- [ ] **Step 4: Print summary log after `indices = list(zip(epi_id, ids))`**

```python
        if self.action_gap_threshold_mult > 0:
            pct = (
                100.0 * gap_discard_total / gap_candidate_total
                if gap_candidate_total > 0
                else 0.0
            )
            print(
                f"[SequenceSamplerUmiFT] action-gap filter "
                f"(mult={self.action_gap_threshold_mult}): "
                f"discarded {gap_discard_total}/{gap_candidate_total} queries "
                f"({pct:.1f}%)"
            )
            for ep_idx, (n_disc, n_cand) in sorted(gap_discard_by_episode.items()):
                print(
                    f"  episode_{ep_idx}: {n_disc}/{n_cand}"
                )
```

- [ ] **Step 5: Smoke-check helper still importable; run unit tests**

```bash
python -m pytest tests/test_sampler_umift_action_timing.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit (only if user requested commits)**

---

### Task 3: Thread ctor arg through `UmiFTDataset`

**Files:**
- Modify: `UMI-FT/PyriteML/diffusion_policy/dataset/umift_dataset.py`

**Interfaces:**
- Consumes: `SequenceSamplerUmiFT(..., action_gap_threshold_mult=...)`
- Produces: dataset ctor arg `action_gap_threshold_mult: float = 3.0` applied to train and val samplers

- [ ] **Step 1: Add dataset parameter**

In `UmiFTDataset.__init__` signature, add:

```python
        action_gap_threshold_mult: float = 3.0,
```

Save on `self`:

```python
        self.action_gap_threshold_mult = action_gap_threshold_mult
```

- [ ] **Step 2: Pass into both sampler constructions**

Train sampler construction (~line 120) and `get_validation_dataset` sampler construction (~line 210) both need:

```python
            action_gap_threshold_mult=self.action_gap_threshold_mult,
```

(or the local arg name during `__init__` before `self` is set).

- [ ] **Step 3: Optional config exposure (only if trivial)**

If desired in the same change, add under `dataset:` in `umift_zucchini.yaml`:

```yaml
  action_gap_threshold_mult: 3.0
```

Not required for correctness (default is 3.0). Prefer adding it so the user can bump to `5.0` without code edits.

- [ ] **Step 4: Run unit tests again**

```bash
python -m pytest tests/test_sampler_umift_action_timing.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit (only if user requested commits)**

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Timestamp-gap filter on action targets only | Task 1–2 |
| Threshold `3 × expected_dt` with expected_dt = median rgb_dt × RGB frame stride | Task 1–2 |
| Filter at index build in `__init__` | Task 2 |
| Ignore obs-history gaps | Task 1–2 (never inspected) |
| Log discarded/total (+ per-episode when discarded) | Task 2 |
| `mult <= 0` disables | Task 1–2 |
| Wire through `UmiFTDataset` | Task 3 |
| Unit tests for hole spanning / disable | Task 1 |
| Mask / weighted-sampling paths also filtered | Task 2 |

## Self-review notes

- No placeholders left in steps.
- Helper names are consistent across tasks.
- Commit steps are gated on explicit user request (workspace rule).
