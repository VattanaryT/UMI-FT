# Start-of-episode padding to fix train/inference OOD at episode reset

## Problem

`SequenceSamplerUmiFT` only creates a trainable query at index `rgb_id` once every
modality has a *full* backward observation history available:

```
rgb_start_id = (horizon * down_sample_steps) + 1   # sampler_umift.py:194-208
```

For the current `umift_zucchini` task, `sparse_obs_low_dim_horizon` /
`sparse_obs_wrench_horizon` are both 32 steps @ 10 Hz (`down_sample_steps=1`),
so no query is ever built inside the first ~3.3 s of any episode
(`sampler_umift.py:159-408`). At inference time, the real rollout starts
exactly there: t=0, with the observation buffer having nothing but the
reset/start state repeated for history. The policy is *never* trained on that
situation, even though it's the one state every rollout is guaranteed to
visit.

The `sample_sequence` first-frame-repeat padding (`sampler_umift.py:732-735`)
exists for this, but in practice never fires for a real query, because
`episodes_start`/`episodes_length_for_query` already excludes every index
that would need it.

## Goal

Make "observation history = repeated start state, action = whatever really
happens starting from t=0" part of the trainable data, by adding real
timesteps to the front of every episode at data-prep time, rather than by
changing sampling/exclusion logic. This generalizes to any future horizon
config without needing to keep two systems (skip-window + pad-fallback) in
sync.

## Non-goals

- Changing `SequenceSamplerUmiFT` / `sample_sequence` padding logic. It's
  already correct; it just currently gets no practical exercise, and will
  behave correctly for the pad region without modification.
- Changing the virtual-target / stiffness estimator, merge script, or
  training-time `raw_to_action*` conversion. The pad is applied upstream of
  all of these; they simply see a longer episode.
- An in-place / streaming-append converter. Stage C already always does a
  full-store rebuild per run; that doesn't change here.
- A standalone "pad an already-converted zarr in place" tool. The user will
  re-run the existing convert → VT-postprocess → merge pipeline against
  padding-enabled converters for existing datasets.

## Approach

Add a pure helper, `pad_episode_start(ep_data, pad_duration_s, id_list=(0,))`,
in a new module `UMIFT_Data/umift/processing/pad_episode_start.py` (sibling
to `umift_image_prep.py`, same "shared per-episode dict transform" shape).
Call it from both Stage C converters, on the in-memory `ep_data: Dict[str,
np.ndarray]` dict, right before each episode's arrays are written to zarr.

### `pad_episode_start` semantics

Given an episode dict following the UMIFT naming convention
(`<stream>_<id>`, `<family>_time_stamps_<id>`, `map_to_<uw|d>_idx_<id>`):

1. For each `*_time_stamps_<id>` array present, compute that stream's own
   median `dt` and prepend `n_pad = round(pad_duration_s / dt)` new
   timestamps continuing backward at that same `dt`, immediately before the
   existing first sample. `n_pad` is tracked per timestamp key.
2. For each per-timestep data array (`rgb_0`, `ts_pose_fb_0`,
   `wrench_left_0`, `gripper_0`, `stiffness_0`, `ts_pose_virtual_target_0`,
   `ts_pose_command_0`, coinft variants, etc.), look up its owning
   timestamp key's `n_pad` and prepend that many literal copies of row 0
   (the true first sample, verbatim — including wrench, per product
   decision: no baseline/free-space assumption is injected here).
3. For `map_to_uw_idx_0` / `map_to_d_idx_0`, prepend zeros (pointing at the
   padded region's first, also-repeated, target frame) and shift the
   original indices by the target stream's own `n_pad` so they still
   resolve into the now-longer target array.
4. Re-zero every timestamp array by the global minimum first sample (reuses
   the same convention `normalize_timestamps` already establishes in
   `convert_unified_t265_wild_to_umift_zarr.py`), so the episode still
   starts at t≈0 after padding.
5. `pad_duration_s <= 0` returns a shallow copy of the input unchanged — the
   call site never needs its own conditional.
6. Unknown/missing keys are skipped silently (schema variation between the
   two converters and pre/post-VT is expected; this function only acts on
   keys it recognizes and that are present).

This is pure-numpy, dict-in/dict-out — no zarr dependency — so it is testable
in isolation and reusable from both converters without any zarr-specific
plumbing.

### Wiring

- `convert_unified_t265_wild_to_umift_zarr.py`: new `--start-pad-seconds`
  float CLI arg, **default `5.0`**. Call `pad_episode_start(...)` in
  `convert()` right after `_postprocess_episode(...)` (so it composes with
  `--resample-10hz` either way) and before the per-key `create_dataset`
  loop / `_episode_len(...)` calls (so `meta/episode_*_len` reflect the
  padded lengths automatically). Record `root.attrs["start_pad_seconds"]`
  and `ep_group.attrs["start_pad_seconds"]` for provenance.
- `convert_lerobot_teleop_to_umift_zarr.py`: same flag/default, applied
  right after `build_episode_arrays_from_tables(...)`.
- `UMIFT_Data/bash/combine_umift_dataset.sh`: new optional `--start-pad-seconds`
  passthrough flag (same pattern as the existing `--pose-source`/`--output`
  flags), forwarded to `convert_unified_t265_wild_to_umift_zarr.py` only
  when explicitly given; otherwise the python script's own default applies.
- No changes to `postprocessing_add_virtual_target_label_umift.py` or
  `merge_umift_zarr_datasets.py` — both operate generically over whatever
  arrays/lengths an episode has.

### Why 5.0 s default

Covers the largest current obs horizon (32 steps @ 10 Hz = 3.2 s) with
margin, so the very first trainable query of every episode lands fully
inside the padded region (frozen-state history, real action target once the
query time crosses into the true recording). If horizons grow later, this
constant should grow with them (documented in the CLI help string).

### Reprocessing existing datasets

Not implemented as a new tool — the user will re-run their existing
Stage C → VT-postprocess → merge commands (see execution plan / PR
description for the exact commands per dataset folder), and padding is
applied automatically since it defaults on in the converters.

## Testing

Add `UMI-FT/UMIFT_Data/tests/test_pad_episode_start.py` (same
importlib-free style as `test_umift_image_prep.py`, importing from
`umift.processing.pad_episode_start`):

1. `pad_duration_s=0` returns an equal (unpadded) episode.
2. Uniform-grid synthetic episode (single dt shared by all streams, like
   `test_resample_episode_to_hz_20_to_10`): every per-timestep array grows
   by the same `n_pad`; the first `n_pad` rows of every data array equal
   the original row 0; timestamps stay monotonic and still start at 0;
   `map_to_uw_idx_0`/`map_to_d_idx_0` prefix is `0` and the shifted tail
   still resolves to the correct (also-repeated) target rows.
3. Heterogeneous-rate episode (distinct dt for rgb vs. robot vs. wrench,
   like the pre-resample wild-converter output): each stream's own `n_pad`
   is derived from its own dt; post-pad timestamps across streams are still
   mutually consistent within the sampler's existing tolerance.
4. A key family that's absent (e.g. no `ts_pose_virtual_target_0` — the
   pre-VT case) is simply skipped without error.
5. `test_main_parser_has_resample_flag`-style tests for the new
   `--start-pad-seconds` arg/default in both converter scripts.

Run: `cd UMI-FT/UMIFT_Data && PYTHONPATH="" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q`
(the ambient ROS-sourced shell otherwise drags in a broken `launch_testing`
pytest plugin).

## Success criteria

- `pad_episode_start` unit tests pass; full existing `UMIFT_Data/tests/`
  suite still passes unmodified.
- Running either converter with the new default produces episodes whose
  `meta/episode_*_len` are each exactly `n_pad` longer than before (verified
  via the existing `validate_merged_zarr`-style length checks after a
  convert → VT-postprocess → merge dry run on a small synthetic/sample
  dataset).
- No changes required to `SequenceSamplerUmiFT`, VT postprocessing, or
  merge scripts.
