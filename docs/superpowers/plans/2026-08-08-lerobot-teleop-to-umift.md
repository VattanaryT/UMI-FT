# LeRobot Teleop → UMIFT Zarr Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert LeRobot teleop (`08_03_..._combined`) into a UMIFT episode zarr (224 images, zero depth, right arm, 10 Hz), extend unified→UMIFT with always-on 224 resize and optional `--resample-10hz`, and retune train/infer configs for 10 Hz (`ds=1`, history 16, action 32, query stride 1).

**Architecture:** New teleop converter writes stringer-compatible episode zarr geometry; existing VT postprocess stays unchanged. Unified converter gains shared resize helpers and an opt-in 10 Hz resample path. Zucchini + env-runner sampling math is updated for a 10 Hz policy rate. Query-relative math is untouched. Sample-time win is **pre-resized 224** only (no action21/pose9/float-CHW on disk).

**Tech Stack:** Python 3, NumPy, SciPy, OpenCV/existing `get_image_transform_with_border`, zarr v2 + numcodecs, pyarrow (LeRobot parquet), OpenCV/PyAV for mp4 decode, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-08-lerobot-teleop-to-umift-design.md`
- Do **not** modify bags or source LeRobot / `*_unified.zarr` in place.
- Teleop: always 10 Hz, 224, zero depth, right arm only; VT via existing postprocess (fb + wrench).
- Unified: always resize 224; `--resample-10hz` default **false**.
- Config: histories horizon **16**, action horizon **32**, most `down_sample_steps` **1**, query stride **1**.
- Do **not** commit unless the user explicitly asks.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `UMIFT_Data/umift/processing/umift_image_prep.py` | Shared resize (+ optional 10 Hz resample helpers) |
| `UMIFT_Data/scripts/convert_unified_t265_wild_to_umift_zarr.py` | Call resize; wire `--resample-10hz` |
| `UMIFT_Data/tests/test_umift_image_prep.py` | Unit tests for resize / resample helpers |
| `UMIFT_Data/tests/test_convert_unified_t265_wild_to_umift_zarr.py` | Flag / convert integration smoke on tiny synthetic episode |
| `UMIFT_Data/scripts/convert_lerobot_teleop_to_umift_zarr.py` | LeRobot → UMIFT episode zarr |
| `UMIFT_Data/tests/test_convert_lerobot_teleop_to_umift_zarr.py` | Pose/wrench/mapping unit tests |
| `PyriteML/diffusion_policy/config/task/umift_zucchini.yaml` | 10 Hz horizons / ds |
| `PyriteEnvSuites/env_runners/umift_env_runner.py` | `raw_time_step_s` + downsample corrections for 10 Hz train rate |

---

### Task 1: Shared 224 resize (+ 10 Hz resample helpers)

**Files:**
- Create: `UMI-FT/UMIFT_Data/umift/processing/umift_image_prep.py`
- Create: `UMI-FT/UMIFT_Data/tests/test_umift_image_prep.py`

**Interfaces:**
- Produces:
  - `resize_hwc_uint8(frames: np.ndarray, out_hw: tuple[int,int]=(224,224)) -> np.ndarray`  # (T,H,W,3) uint8
  - `resize_hwc_float(frames: np.ndarray, out_hw: tuple[int,int]=(224,224)) -> np.ndarray`  # depth float16/float32
  - `make_uniform_time_grid(t0: float, t1: float, hz: float) -> np.ndarray`  # 1d seconds
  - `nearest_indices(src_t: np.ndarray, query_t: np.ndarray) -> np.ndarray`  # int64
  - `resample_episode_to_hz(ep: dict, hz: float=10.0) -> dict`  # images nearest; pose via existing `_resample_pose7` pattern; wrench nearest; rewrite timestamps + identity maps

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from umift.processing.umift_image_prep import (
    resize_hwc_uint8,
    resize_hwc_float,
    make_uniform_time_grid,
    nearest_indices,
)

def test_resize_hwc_uint8_to_224():
    frames = np.zeros((3, 48, 64, 3), dtype=np.uint8)
    frames[:, 10:20, 10:20] = 255
    out = resize_hwc_uint8(frames, (224, 224))
    assert out.shape == (3, 224, 224, 3)
    assert out.dtype == np.uint8

def test_resize_hwc_float_preserves_dtype_float16():
    depth = np.ones((2, 40, 50, 3), dtype=np.float16) * 0.25
    out = resize_hwc_float(depth, (224, 224))
    assert out.shape == (2, 224, 224, 3)
    assert out.dtype == np.float16

def test_make_uniform_time_grid_10hz():
    g = make_uniform_time_grid(0.0, 1.0, 10.0)
    assert len(g) == 11  # 0.0 .. 1.0 inclusive at 0.1
    assert np.isclose(g[1] - g[0], 0.1)

def test_nearest_indices_picks_closest():
    src = np.array([0.0, 0.1, 0.2])
    query = np.array([0.0, 0.09, 0.25])
    idx = nearest_indices(src, query)
    assert list(idx) == [0, 1, 2]
```

- [ ] **Step 2: Run tests (expect fail)**

```bash
cd UMI-FT/UMIFT_Data && python -m pytest tests/test_umift_image_prep.py -v
```

Expected: import/module missing.

- [ ] **Step 3: Implement helpers**

Use `get_image_transform_with_border` from `PyriteUtility.computer_vision.computer_vision_utility` (same as sample-time path) for uint8 RGB. For depth float, use `cv2.resize(..., INTER_NEAREST)` then cast back to float16. For `resample_episode_to_hz`, expect keys matching unified converter episode dict (`rgb_0`, `depth_0`, `ultrawide_0`, `ts_pose_fb_0`, wrench keys, timestamps); rewrite all stream lengths to the grid; set `map_to_uw_idx_0` / `map_to_d_idx_0` to identity.

- [ ] **Step 4: Run tests (expect pass)**

```bash
cd UMI-FT/UMIFT_Data && python -m pytest tests/test_umift_image_prep.py -v
```

---

### Task 2: Unified converter — always 224, optional `--resample-10hz`

**Files:**
- Modify: `UMI-FT/UMIFT_Data/scripts/convert_unified_t265_wild_to_umift_zarr.py`
- Modify: `UMI-FT/UMIFT_Data/tests/test_convert_unified_t265_wild_to_umift_zarr.py`

**Interfaces:**
- Consumes: `resize_hwc_uint8`, `resize_hwc_float`, `resample_episode_to_hz`
- Changes: `convert(..., resample_10hz: bool = False)`; CLI `--resample-10hz` store_true
- After `_read_episode_from_unified`, always resize rgb/ultrawide/depth to 224; if flag, call `resample_episode_to_hz(ep, 10.0)` before write
- Root attrs: add `"image_size_hw": [224, 224]` and `"resample_hz": 10.0 | null`

- [ ] **Step 1: Extend unit tests**

Add tests that import the convert module and call a thin wrapper if needed; at minimum test that argparse accepts `--resample-10hz` and that a pure function path resizes:

```python
def test_main_parser_has_resample_flag():
    # parse with MODULE's argparse by extracting or calling a build_parser() helper
    parser = MODULE.build_parser()
    args = parser.parse_args(["--input", "x", "--output", "y", "--resample-10hz"])
    assert args.resample_10hz is True
    args2 = parser.parse_args(["--input", "x", "--output", "y"])
    assert args2.resample_10hz is False
```

Refactor `main()` to use `build_parser()` returning the ArgumentParser.

- [ ] **Step 2: Run test (expect fail on missing flag/parser)**

```bash
cd UMI-FT/UMIFT_Data && python -m pytest tests/test_convert_unified_t265_wild_to_umift_zarr.py -v
```

- [ ] **Step 3: Wire resize + flag into `convert()` / `main()`**

```python
# after ep_data, pose_used, frame_conv = _read_episode_from_unified(...)
from umift.processing.umift_image_prep import (
    resize_hwc_uint8,
    resize_hwc_float,
    resample_episode_to_hz,
)
ep_data["rgb_0"] = resize_hwc_uint8(ep_data["rgb_0"], (224, 224))
ep_data["ultrawide_0"] = resize_hwc_uint8(ep_data["ultrawide_0"], (224, 224))
ep_data["depth_0"] = resize_hwc_float(ep_data["depth_0"], (224, 224))
if resample_10hz:
    ep_data = resample_episode_to_hz(ep_data, hz=10.0)
```

Pass `resample_10hz` from CLI. Do not touch bags / input unified stores.

- [ ] **Step 4: Run existing + new convert tests**

```bash
cd UMI-FT/UMIFT_Data && python -m pytest tests/test_convert_unified_t265_wild_to_umift_zarr.py tests/test_umift_image_prep.py -v
```

Expected: PASS.

---

### Task 3: LeRobot teleop → UMIFT converter (geometry only)

**Files:**
- Create: `UMI-FT/UMIFT_Data/scripts/convert_lerobot_teleop_to_umift_zarr.py`
- Create: `UMI-FT/UMIFT_Data/tests/test_convert_lerobot_teleop_to_umift_zarr.py`

**Interfaces:**
- CLI: `--input` (LeRobot root), `--output` (episode zarr path), `--image-size` default 224
- Produces episode keys per spec; **does not** write VT/stiffness
- Helpers (testable without video):
  - `rot6d_to_quat_wxyz(rot6: np.ndarray) -> np.ndarray`  # (...,6)->(...,4)
  - `right_state_to_pose7_wrench(state30: np.ndarray) -> tuple[np.ndarray, np.ndarray]`  # pose7 (T,7), wrench (T,6)
  - `build_episode_arrays_from_tables(...)` assembling timestamps, zeros depth, identity maps, command=fb, gripper zeros, wrench_right zeros

**Mapping (locked):**
- `observation.state` right `[0:9]` → `ts_pose_fb_0` / `ts_pose_command_0`
- right wrench `[9:15]` → `wrench_left_0`; `wrench_right_0 = 0`
- `wrist_cam_0` → `rgb_0` @ 224; `depth_0` zeros float16 3ch; `ultrawide_0` zeros uint8 3ch
- all timestamps = LeRobot episode timestamps (10 Hz)

- [ ] **Step 1: Write failing unit tests for pose/wrench helpers**

```python
import numpy as np
from convert_lerobot_teleop_to_umift_zarr import (  # or importlib like unified tests
    rot6d_to_quat_wxyz,
    right_state_to_pose7_wrench,
)

def test_rot6d_identity_to_quat():
    r6 = np.array([1.0, 0, 0, 0, 1, 0], dtype=np.float64)
    q = rot6d_to_quat_wxyz(r6)
    assert q.shape == (4,)
    assert np.allclose(np.abs(q), [1, 0, 0, 0], atol=1e-5) or np.allclose(q, [1, 0, 0, 0], atol=1e-5)

def test_right_state_unpack():
    state = np.zeros((2, 30), dtype=np.float32)
    state[0, 0:3] = [0.1, 0.2, 0.3]
    state[0, 3:9] = [1, 0, 0, 0, 1, 0]
    state[0, 9:15] = [1, 2, 3, 4, 5, 6]
    pose7, wrench = right_state_to_pose7_wrench(state)
    assert pose7.shape == (2, 7)
    assert wrench.shape == (2, 6)
    assert np.allclose(pose7[0, :3], [0.1, 0.2, 0.3])
    assert np.allclose(wrench[0], [1, 2, 3, 4, 5, 6])
```

- [ ] **Step 2: Run tests (expect fail)**

```bash
cd UMI-FT/UMIFT_Data && python -m pytest tests/test_convert_lerobot_teleop_to_umift_zarr.py -v
```

- [ ] **Step 3: Implement converter**

Read `meta/info.json`, `meta/episodes/...parquet`, `data/...parquet`. For each episode, slice parquet rows by `dataset_from_index`/`dataset_to_index`. Decode `observation.images.wrist_cam_0_color_image_rect_raw_compressed` mp4 using episode `from_timestamp`/`to_timestamp` (OpenCV `VideoCapture` + fps seek, or PyAV). Resize with `resize_hwc_uint8`. Write zarr v2 DirectoryStore like unified converter (`data/episode_k`, cumulative `meta/episode_*_len`). Attrs: `source_type=lerobot_teleop`, `arm=right`, `depth_source=zeros`, `image_size_hw=[224,224]`, `resample_hz=10`.

- [ ] **Step 4: Run unit tests**

```bash
cd UMI-FT/UMIFT_Data && python -m pytest tests/test_convert_lerobot_teleop_to_umift_zarr.py -v
```

- [ ] **Step 5: Smoke convert (manual, full data)**

```bash
python UMIFT_Data/scripts/convert_lerobot_teleop_to_umift_zarr.py \
  --input /mnt/data/umidata/ft/teleop/lerobot_dataset_08_03_stringer_bimanual_10hz_vention_frame_obsrot6d_actdelta_desired_combined \
  --output /mnt/data/umidata/ft/teleop_umift_08_03_right/acp_replay_buffer.zarr

# then existing VT postprocess on that output path (same invocation as stringer)
```

Verify: 47 episodes; `rgb_0.shape[1:]= (224,224,3)`; `depth_0` all ~0; `wrench_right_0` all 0; equal stream lengths.

---

### Task 4: Retune zucchini + inference for 10 Hz

**Files:**
- Modify: `UMI-FT/PyriteML/diffusion_policy/config/task/umift_zucchini.yaml`
- Modify: `UMI-FT/PyriteEnvSuites/env_runners/umift_env_runner.py` (layup `control_para` / `pipeline_para`)

**Config values (exact):**

```yaml
sparse_obs_rgb_down_sample_steps: 1
sparse_obs_rgb_horizon: 16
sparse_obs_ultrawide_down_sample_steps: 1
sparse_obs_ultrawide_horizon: 16
sparse_obs_depth_down_sample_steps: 1
sparse_obs_depth_horizon: 16
sparse_obs_low_dim_down_sample_steps: 1
sparse_obs_low_dim_horizon: 16
sparse_obs_wrench_down_sample_steps: 1
sparse_obs_wrench_horizon: 16
sparse_action_down_sample_steps: 1
sparse_action_horizon: 32
# dataset:
sparse_query_frequency_down_sample_steps: 1
dataset_path: /mnt/data/umidata/ft/teleop_umift_08_03_right/acp_replay_buffer.zarr  # after convert+VT
```

Inference layup defaults:

```python
control_para["raw_time_step_s"] = 1.0 / 10.0
# training policy rate = 10 Hz; live RTDE ~500 Hz, wrench ~28 Hz (keep same live rates)
pipeline_para["robot_downsample_correction"] = 500.0 / 10.0
pipeline_para["wrench_downsample_correction"] = 28.0 / 10.0  # or keep measured live wrench Hz / 10
# if D405 still ~15 Hz live: depth/rgb correction = 15/10 = 1.5 (use int floor/ceil consistently with existing //= pattern)
```

Read existing correction block (~lines 511–675) and update comments that still say 15 Hz / 200 Hz training assumptions.

- [ ] **Step 1: Edit yaml horizons/ds/path**
- [ ] **Step 2: Edit env_runner time step + corrections**
- [ ] **Step 3: Sanity-check math**

```bash
python - <<'PY'
# rgb history: 16*1/10 = 1.6s; action: 32*1/10 = 3.2s
# infer robot ds after correction: 1 * (500/10) = 50 → 50/500 = 0.1s stride
print('ok')
PY
```

---

### Task 5: End-to-end validation checklist

No new code required beyond Tasks 1–4. Run after convert + VT:

- [ ] **Step 1: Schema checks**

```bash
python - <<'PY'
import zarr, numpy as np
p='/mnt/data/umidata/ft/teleop_umift_08_03_right/acp_replay_buffer.zarr'
r=zarr.open(p,'r')
eps=sorted([k for k in r['data']], key=lambda x: int(x.split('_')[1]))
assert len(eps)==47
ep=r['data'][eps[0]]
assert ep['rgb_0'].shape[1:3]==(224,224)
assert float(np.max(np.abs(ep['wrench_right_0'][:])))==0.0
assert float(np.max(np.abs(ep['depth_0'][:])))==0.0
assert 'ts_pose_virtual_target_0' in ep and 'stiffness_0' in ep
print('episode0 lens', {k: ep[k].shape[0] for k in ['rgb_0','ts_pose_fb_0','wrench_left_0']})
PY
```

- [ ] **Step 2: Dataset smoke**

Construct `UmiFTDataset` with retuned zucchini `shape_meta` and `dataset_path`; `next(iter(DataLoader(ds, batch_size=2)))` succeeds; rgb tensor shape `[B,16,3,224,224]`.

---

## Spec coverage (self-review)

| Spec requirement | Task |
|---|---|
| LeRobot A → UMIFT zarr, right arm, drop center/left cams | Task 3 |
| Zero depth kept in schema | Task 3 |
| 224 resize in converters | Tasks 1–3 |
| Teleop always 10 Hz | Task 3 |
| Unified `--resample-10hz` opt-in; default off; always 224 | Task 2 |
| Bags / unified inputs untouched | Tasks 2–3 |
| VT via existing postprocess | Task 3 step 5 + ops |
| Zucchini ds=1, hist 16, action 32, query 1 | Task 4 |
| Inference 10 Hz wall-clock corrections | Task 4 |
| No action21/pose9 preprocess | (non-goal; omitted) |
| Query-relative unchanged | (no code) |
