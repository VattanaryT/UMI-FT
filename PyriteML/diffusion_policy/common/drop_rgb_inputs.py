"""Remove wrist-RGB observations from a composed training config.

The sampler still requires ``rgb_0`` in ``sample.obs.sparse`` as the query
clock, and ``raw.rgb_0`` so the episode store can be built. Deleting those
would change which timesteps are trained or crash the sampler. This only
removes ``rgb_*`` from ``shape_meta.obs``, which is the policy observation.
"""

from omegaconf import OmegaConf, open_dict

_SHAPE_META_PATHS = (
    "task.shape_meta",
    "task.dataset.shape_meta",
    "shape_meta",
    "policy.shape_meta",
    "policy.obs_encoder.shape_meta",
)


def _is_wrist_rgb_obs_key(key: str) -> bool:
    if not key.startswith("rgb_"):
        return False
    suffix = key[4:]
    return suffix.isdigit()


def drop_rgb_inputs(shape_meta) -> list:
    """Delete wrist RGB keys from shape_meta.obs. Returns removed names."""
    if shape_meta is None or not OmegaConf.is_config(shape_meta):
        return []
    if "obs" not in shape_meta:
        return []
    removed = []
    with open_dict(shape_meta):
        block = shape_meta["obs"]
        with open_dict(block):
            for key in list(block.keys()):
                if _is_wrist_rgb_obs_key(key):
                    del block[key]
                    removed.append(f"obs.{key}")
    return removed


def apply_drop_rgb_inputs(cfg) -> bool:
    """Strip wrist RGB from policy obs when ``task.drop_rgb_inputs`` is set.

    Safe to call before or after ``OmegaConf.resolve``. Missing keys are
    skipped, so a second call is a no-op.
    """
    if not bool(OmegaConf.select(cfg, "task.drop_rgb_inputs", default=False)):
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
        removed.extend(drop_rgb_inputs(shape_meta))
    if removed:
        print("[no_rgb] removed wrist RGB from policy obs:", ", ".join(removed))
    return True
