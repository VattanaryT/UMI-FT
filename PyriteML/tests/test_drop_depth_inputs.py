from omegaconf import OmegaConf

from diffusion_policy.common.drop_depth_inputs import apply_drop_depth_inputs


def _cfg(flag):
    shape = {
        "raw": {"rgb_0": {"shape": [3, 224, 224]}, "depth_0": {"shape": [3, 224, 224]}},
        "obs": {
            "rgb_0": {"shape": [3, 224, 224], "type": "rgb"},
            "depth_0": {"shape": [3, 224, 224], "type": "rgb"},
            "depth_time_stamps_0": {"shape": [1], "type": "timestamp"},
            "robot0_mold_geom": {"shape": [10], "type": "low_dim"},
        },
        "sample": {
            "obs": {
                "sparse": {
                    "rgb_0": {"horizon": 16},
                    "depth_0": {"horizon": 16},
                    "robot0_mold_geom": {"horizon": 51},
                }
            }
        },
    }
    return OmegaConf.create(
        {
            "task": {
                "drop_depth_inputs": flag,
                "shape_meta": shape,
                "dataset": {"shape_meta": shape},
            },
            "policy": {"shape_meta": shape, "obs_encoder": {"shape_meta": shape}},
        }
    )


def test_flag_removes_depth_and_keeps_rgb_and_mold():
    cfg = _cfg(True)
    assert apply_drop_depth_inputs(cfg) is True
    for shape_meta in (
        cfg.task.shape_meta,
        cfg.task.dataset.shape_meta,
        cfg.policy.shape_meta,
        cfg.policy.obs_encoder.shape_meta,
    ):
        assert "depth_0" not in shape_meta.obs
        assert "depth_time_stamps_0" not in shape_meta.obs
        assert "depth_0" not in shape_meta.raw
        assert "depth_0" not in shape_meta.sample.obs.sparse
        assert "rgb_0" in shape_meta.obs
        assert "robot0_mold_geom" in shape_meta.sample.obs.sparse
    assert apply_drop_depth_inputs(cfg) is True


def test_flag_off_leaves_depth():
    cfg = _cfg(False)
    assert apply_drop_depth_inputs(cfg) is False
    assert "depth_0" in cfg.task.shape_meta.obs
