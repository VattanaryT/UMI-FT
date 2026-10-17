"""Remove laser-scan observations from a composed training config.

Hydra merges task configs, so a variant cannot delete keys it inherits.
``umift_ss_v2_no_scan`` sets ``task.drop_scan_inputs`` and this strips
``scan_heatmap_*`` / ``map_to_scan_idx_*`` from every ``shape_meta`` before
the policy is constructed. The scan CNN is built only for those keys, so
the checkpoint's model has no scan branch.
"""

from omegaconf import OmegaConf, open_dict

_SHAPE_META_PATHS = (
    "task.shape_meta",
    "task.dataset.shape_meta",
    "shape_meta",
    "policy.shape_meta",
    "policy.obs_encoder.shape_meta",
)


def _is_scan_shape_key(key: str) -> bool:
    return "scan_heatmap" in key or key.startswith("map_to_scan_idx_")


def drop_scan_inputs(shape_meta) -> list:
    """Delete scan observation keys from one shape_meta. Returns removed names."""
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
                    if _is_scan_shape_key(key):
                        del block[key]
                        removed.append(f"{section}.{key}")
        sparse = OmegaConf.select(shape_meta, "sample.obs.sparse")
        if sparse is not None:
            with open_dict(sparse):
                for key in list(sparse.keys()):
                    if _is_scan_shape_key(key):
                        del sparse[key]
                        removed.append(f"sample.obs.sparse.{key}")
    return removed


def apply_drop_scan_inputs(cfg) -> bool:
    """Strip scan inputs when ``task.drop_scan_inputs`` is set.

    Safe to call before or after ``OmegaConf.resolve``. Missing keys are
    skipped, so a second call is a no-op.
    """
    if not bool(OmegaConf.select(cfg, "task.drop_scan_inputs", default=False)):
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
        removed.extend(drop_scan_inputs(shape_meta))
    if removed:
        print("[no_scan] removed scan inputs from shape_meta:", ", ".join(removed))
    return True
