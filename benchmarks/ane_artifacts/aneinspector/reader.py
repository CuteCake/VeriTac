"""Bounded file reading and hashing.

Only stdlib.  All reads are bounded by explicit limits so that very large
artifacts (multi-GB weight banks) are handled by stream hashing and capped
region reads rather than being slurped into memory.
"""

import hashlib
import os

HASH_CHUNK = 1 << 20  # 1 MiB


class ReadLimitExceeded(Exception):
    """Raised when a bounded region read exceeds its cap."""


def file_length(path):
    return os.path.getsize(path)


def stream_sha256(fh, limit=None):
    """Stream the whole file (or up to ``limit`` bytes) through SHA-256."""
    digest = hashlib.sha256()
    remaining = limit
    while True:
        n = HASH_CHUNK
        if remaining is not None:
            n = min(n, remaining)
            if n <= 0:
                break
        block = fh.read(n)
        if not block:
            break
        digest.update(block)
        if remaining is not None:
            remaining -= len(block)
    return digest.hexdigest()


def read_exact(fh, offset, length):
    """Read exactly ``length`` bytes at ``offset``; return bytes or None if
    the file is too short (truncated)."""
    if offset < 0 or length < 0:
        return None
    try:
        fh.seek(offset)
        data = fh.read(length)
    except OSError:
        return None
    if data is None or len(data) < length:
        return None
    return data


def read_bounded(fh, offset, length, cap):
    """Read at most ``length`` bytes but never more than ``cap`` bytes.

    Returns ``(data, was_truncated)``.  ``was_truncated`` is True when the
    requested ``length`` exceeded ``cap``, so the caller can tell that the
    returned region is a clipped prefix of what the format described.
    ``data`` is None when the file is too short (truncated on disk).
    Raises ReadLimitExceeded if length is below 0."""
    if length < 0:
        raise ReadLimitExceeded("negative region length")
    was_truncated = length > cap
    bounded = min(length, cap)
    data = read_exact(fh, offset, bounded)
    if data is None:
        return None, was_truncated
    return data, was_truncated


def bounded_strings(data, min_len=4, max_strings=512, max_len=256):
    """Extract printable ASCII runs of length >= min_len from a byte region.

    Bounded: at most ``max_strings`` strings, each truncated to ``max_len``.
    """
    out = []
    current = bytearray()
    for byte in data:
        if 32 <= byte < 127:
            current.append(byte)
        else:
            if len(current) >= min_len:
                out.append(bytes(current[:max_len]).decode("ascii", "replace"))
                if len(out) >= max_strings:
                    break
            current = bytearray()
    if len(current) >= min_len and len(out) < max_strings:
        out.append(bytes(current[:max_len]).decode("ascii", "replace"))
    return out
