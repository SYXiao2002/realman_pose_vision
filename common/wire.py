"""Bounded MessagePack transport for image and mask arrays."""

import math

import msgpack
import numpy as np

MAX_MESSAGE_BYTES = 32 * 1024 * 1024
_DTYPES = {"|u1", "|b1", "<i4", "<f4", "<f8"}


def _pack_default(value):
    if isinstance(value, np.ndarray):
        if value.dtype.str not in _DTYPES or value.nbytes > MAX_MESSAGE_BYTES:
            raise ValueError("unsupported or oversized image array")
        return {
            "__ndarray__": True,
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": np.ascontiguousarray(value).tobytes(),
        }
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported MessagePack value: {type(value).__name__}")


def _unpack_hook(value):
    if "__ndarray__" not in value:
        return value
    if set(value) != {"__ndarray__", "dtype", "shape", "data"} or value["__ndarray__"] is not True:
        raise ValueError("invalid ndarray envelope")
    dtype, shape, data = value["dtype"], value["shape"], value["data"]
    if not isinstance(dtype, str) or dtype not in _DTYPES:
        raise ValueError("unsupported array dtype")
    if not isinstance(shape, list) or not 1 <= len(shape) <= 3:
        raise ValueError("invalid array shape")
    if any(type(size) is not int or not 0 < size <= 4096 for size in shape):
        raise ValueError("array shape is out of range")
    expected = math.prod(shape) * np.dtype(dtype).itemsize
    if not isinstance(data, bytes) or expected != len(data) or expected > MAX_MESSAGE_BYTES:
        raise ValueError("array byte count mismatch")
    return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)


def packb(value):
    payload = msgpack.packb(value, default=_pack_default, use_bin_type=True)
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("message exceeds byte limit")
    return payload


def unpackb(payload):
    if not isinstance(payload, bytes) or len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("expected a bounded binary MessagePack message")
    return msgpack.unpackb(
        payload,
        raw=False,
        object_hook=_unpack_hook,
        strict_map_key=True,
        max_bin_len=MAX_MESSAGE_BYTES,
        max_str_len=4096,
        max_array_len=4096,
        max_map_len=128,
        max_ext_len=0,
    )
