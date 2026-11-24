from omegaconf import OmegaConf

from diffusion_policy.common.drop_rgb_inputs import apply_drop_rgb_inputs


def _cfg(flag):
    shape = {
        "raw": {"rgb_0": {"shape": [3, 224, 224], "type": "rgb"}},
        "obs": {
            "rgb_0": {"shape": [3, 224, 224], "type": "rgb"},
            "rgb_time_stamps_0": {"shape": [1], "type": "timestamp"},
            "robot0_eef_wrench_left": {"shape": [6], "type": "low_dim"},
            "robot0_mold_geom": {"shape": [10], "type": "low_dim"},
        },
        "sample": {
            "obs": {
                "sparse": {
                    "rgb_0": {"horizon": 16},
                    "robot0_eef_wrench_left": {"horizon": 118},
                    "robot0_mold_geom": {"horizon": 51},
                }
            }
        },
    }
    return OmegaConf.create(
        {
            "task": {
                "drop_rgb_inputs": flag,
                "shape_meta": shape,
                "dataset": {"shape_meta": shape},
            },
            "policy": {"shape_meta": shape, "obs_encoder": {"shape_meta": shape}},
        }
    )


def test_flag_removes_policy_rgb_and_keeps_the_sampler_clock():
    cfg = _cfg(True)
    assert apply_drop_rgb_inputs(cfg) is True
    for shape_meta in (
        cfg.task.shape_meta,
        cfg.task.dataset.shape_meta,
        cfg.policy.shape_meta,
        cfg.policy.obs_encoder.shape_meta,
    ):
        assert "rgb_0" not in shape_meta.obs
        assert "rgb_time_stamps_0" not in shape_meta.obs
        assert "robot0_eef_wrench_left" in shape_meta.obs
        assert "robot0_mold_geom" in shape_meta.obs
        assert "rgb_0" in shape_meta.raw
        assert "rgb_0" in shape_meta.sample.obs.sparse


def test_flag_off_is_a_noop():
    cfg = _cfg(False)
    assert apply_drop_rgb_inputs(cfg) is False
    assert "rgb_0" in cfg.task.shape_meta.obs
