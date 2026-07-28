# Action-gap sample filter for `SequenceSamplerUmiFT`

## Problem

Some stringer / ACP episodes contain 1–2 s recording drops. Pose action labels are sampled on the RGB time grid, so a gap inside the **future action horizon** produces an oversized relative action and poisons policy training.

Jumps that appear only inside the **observation history** are acceptable and must not discard the sample.

## Goal

Exclude training queries whose sparse action-target timestamps contain a large consecutive gap. Log how many queries are discarded so the multiplier can be tuned (3× → 5×) if needed.

## Non-goals

- Filtering observation-history gaps
- Pose-magnitude jump detection
- Offline zarr rewriting / gap masks baked into the dataset
- Changing inference / temporal ensemble behavior

## Approach

Filter when building `self.indices` in `SequenceSamplerUmiFT.__init__` (one-time, keeps `__len__` correct).

For each candidate `(episode, rgb_id)` after the existing start/end windowing and query downsampling:

1. Resolve sparse action target times with `sparse_action_indices_from_rgb_timestamps` (same helper used at sample time).
2. Compute `expected_dt = median(diff(rgb_times)) * action_rgb_frame_stride`.
3. If any consecutive `diff(target_times) > action_gap_threshold_mult * expected_dt`, discard the query.
4. Do **not** inspect observation history timestamps.

Default `action_gap_threshold_mult = 3.0` (~0.4 s at 15 fps with stride 2).

## API / config

- Add optional constructor arg `action_gap_threshold_mult: float = 3.0` to `SequenceSamplerUmiFT`.
- Thread it from `UmiFTDataset` (default unchanged for callers).
- Set `action_gap_threshold_mult <= 0` to disable the filter (useful for ablations).

## Logging

After index construction, print a single summary, e.g.:

```text
[SequenceSamplerUmiFT] action-gap filter (mult=3.0): discarded 120/4500 queries (2.7%)
  episode_0: 12/800
  episode_1: 40/820
  ...
```

Only episodes with discards need per-episode lines (or always print if cheap). Total discarded/kept is required.

## Edge cases

- Horizon shorter than 2 targets after end-of-episode trimming: keep (no consecutive gap to check).
- Queries already excluded by `action_padding=False` end windowing remain excluded.
- Mask-based / weighted-correction index paths: apply the same gap check before accepting a query into `self.indices`.
- Missing or empty action/rgb timestamps: keep existing hard failures (do not silently treat as gap).

## Tests

Add unit tests in `PyriteML/tests/test_sampler_umift_action_timing.py` (or a sibling):

1. Synthetic RGB times with a 1 s hole; queries whose action horizon spans the hole are discarded; queries that do not span it are kept.
2. With `action_gap_threshold_mult <= 0`, no queries are discarded for gaps.
3. Discard count logging path is covered indirectly by assertable return of filter helper counts (prefer a small pure helper for testability).

## Implementation sketch

1. Extract a pure helper, e.g. `action_targets_have_timestamp_gap(target_times, expected_dt, mult) -> bool`.
2. Extract `filter_rgb_ids_without_action_gaps(...)` that returns kept ids + discard count.
3. Call it when appending candidate ids in `__init__`.
4. Wire ctor arg through `UmiFTDataset`.
5. Add tests; run the existing action-timing tests.

## Success criteria

- Training never sees sparse action targets with consecutive `dt > 3 × expected_dt` (default).
- Startup log shows discarded vs total queries.
- Observation-only gaps do not remove samples.
- Existing timing tests still pass; new gap-filter tests pass.
