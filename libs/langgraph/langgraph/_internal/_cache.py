from __future__ import annotations

import pickle
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

_PRIMITIVE_TYPES = {str, int, float, bool, bytes, type(None), frozenset}


def _freeze(obj: Any, depth: int = 10) -> Hashable:
    # Fast path for common immutable hashable built-in types
    if type(obj) in _PRIMITIVE_TYPES or depth <= 0:
        return obj

    # Fallback check for hashability for custom/rare types
    if isinstance(obj, Hashable):
        return obj

    if isinstance(obj, Mapping):
        # sort keys once, then produce tuple of key, value
        keys = sorted(obj)
        return tuple((k, _freeze(obj[k], depth - 1)) for k in keys)

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return tuple(_freeze(x, depth - 1) for x in obj)

    # numpy / pandas etc. can provide their own .tobytes()
    tobytes = getattr(obj, "tobytes", None)
    if callable(tobytes):
        shape = getattr(obj, "shape", None)
        return (
            type(obj).__name__,
            tobytes(),
            shape,
        )
    return obj  # for e.g. dataclasses with frozen=True etc.


def default_cache_key(*args: Any, **kwargs: Any) -> str | bytes:
    """Default cache key function that uses the arguments and keyword arguments to generate a hashable key."""
    # protocol 5 strikes a good balance between speed and size
    return pickle.dumps((_freeze(args), _freeze(kwargs)), protocol=5, fix_imports=False)
