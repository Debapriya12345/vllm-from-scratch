"""Cache compatibility shims (transformers v4 and v5)."""

from __future__ import annotations

import transformers

try:
    from transformers import DynamicCache
except ImportError:
    DynamicCache = None

TF_MAJOR = int(transformers.__version__.split(".")[0])


def to_legacy(cache):
    """Convert any transformers Cache -> tuple of (k, v) per layer."""
    if cache is None:
        return None
    if isinstance(cache, tuple):
        return cache
    # v4
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    # v5: iterate layers
    if hasattr(cache, "layers"):
        result = []
        for layer in cache.layers:
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if k is None or v is None:
                if isinstance(layer, (tuple, list)) and len(layer) == 2:
                    k, v = layer
                else:
                    raise TypeError(f"Cannot extract K/V from layer: {type(layer)}")
            result.append((k, v))
        return tuple(result)
    # v4 fallback
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return tuple(zip(cache.key_cache, cache.value_cache))
    raise TypeError(f"Unsupported cache type: {type(cache)}")


def make_cache(legacy_kv):
    """Wrap tuple-of-tuples as a Cache object (v5), or fall back for v4."""
    if DynamicCache is None:
        return legacy_kv
    try:
        return DynamicCache(ddp_cache_data=legacy_kv)  # v5
    except TypeError:
        return DynamicCache.from_legacy_cache(legacy_kv)  # v4
