"""Remove wrist-depth observations from a composed training config.

Hydra merges task configs, so a variant cannot delete ``depth_0`` by
omission. ``task.drop_depth_inputs`` strips every depth key from
``shape_meta`` before the policy is built. The vision encoder is created
only for the remaining rgb keys, so the checkpoint has no depth branch.
"""

from omegaconf import OmegaConf, open_dict

_SHAPE_META_PATHS = (
    "task.shape_meta",
    "task.dataset.shape_meta",
    "shape_meta",
    "policy.shape_meta",
    "policy.obs_encoder.shape_meta",
)


def _is_depth_shape_key(key: str) -> bool:
    return key == "depth_0" or key.startswith("depth_")


def drop_depth_inputs(shape_meta) -> list:
    """Delete depth observation keys from one shape_meta. Returns removed names."""
    if shape_meta is None or not OmegaConf.is_config(shape_meta):
        return []
    removed = []
    with open_dict(shape_meta):
        for section in ("raw", "obs"):
            if section not in shape_meta:
                continue
            block = shape_meta[section]
            with open_dict(block):
                for key in list(block.keys()):
                    if _is_depth_shape_key(key):
                        del block[key]
                        removed.append(f"{section}.{key}")
        sparse = OmegaConf.select(shape_meta, "sample.obs.sparse")
        if sparse is not None:
            with open_dict(sparse):
                for key in list(sparse.keys()):
                    if _is_depth_shape_key(key):
                        del sparse[key]
                        removed.append(f"sample.obs.sparse.{key}")
    return removed


def apply_drop_depth_inputs(cfg) -> bool:
    """Strip depth inputs when ``task.drop_depth_inputs`` is set.

    Safe to call before or after ``OmegaConf.resolve``. Missing keys are
    skipped, so a second call is a no-op.
    """
    if not bool(OmegaConf.select(cfg, "task.drop_depth_inputs", default=False)):
        return False
    removed = []
    seen = set()
    for path in _SHAPE_META_PATHS:
        shape_meta = OmegaConf.select(cfg, path)
        if shape_meta is None:
            continue
        node_id = id(shape_meta)
        if node_id in seen:
            continue
        seen.add(node_id)
        removed.extend(drop_depth_inputs(shape_meta))
    if removed:
        print("[no_depth] removed depth inputs from shape_meta:", ", ".join(removed))
    return True
