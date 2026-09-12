"""Inspect a single explicit artifact file (bounded, read-only).

Produces a structured dict: file identity (length, hash), magic, and
container-level metadata when the file is Mach-O / HWX / fat shaped, plus
bounded string and JSON / plist metadata extraction.  No instruction-level
decode is attempted; semantic scope is reported as container_only.
"""

import datetime
import json
import os
import plistlib
import struct

from . import macho
from .reader import bounded_strings, read_bounded, stream_sha256
from .version import __version__

# Bounded caps for the non-container string / metadata passes.
STRING_REGION_CAP = 8 << 20          # 8 MiB per region scanned for strings
STRING_CAP_TOTAL = 32 << 20          # 32 MiB aggregate string scan
METADATA_FILE_CAP = 16 << 20         # only try JSON/plist below this size
SYMBOL_CAP = 8 << 20                 # symtab string-table cap
MAX_COMPILER_HINTS = 64

# Conservative, heuristic token patterns for identifying the compiler /
# target from extracted strings.  These are *hints* with no claim of vendor
# independence, and are never used to dispatch opcode decoding.
_HINT_PATTERNS = (
    "zin_ane_compiler",
    "anecompiler",
    "ANECompilerFramework",
    "com.apple.ANECompilerFramework",
    "zin_ane",
    "h13g",
    "h15g",
    "h17g",
)
_HINT_TARGETS = ("h13", "h14", "h15", "h16", "h17", "m1", "m2", "m3")


class _InspectorError(Exception):
    def __init__(self, reason, detail=""):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


def _json_safe(obj, depth=0, max_depth=24, max_items=2048):
    """Recursively normalize a parsed metadata value so it is JSON
    serializable: bytes -> hex, datetime -> isoformat, bound depth/size."""
    if depth > max_depth:
        return "<max_depth>"
    if isinstance(obj, dict):
        out = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                break
            out[str(k)] = _json_safe(v, depth + 1, max_depth, max_items)
        return out
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x, depth + 1, max_depth, max_items)
                for x in obj[:max_items]]
    if isinstance(obj, bytes):
        return "0x" + obj[:64].hex() + ("..." if len(obj) > 64 else "")
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def _detect_text_metadata(fh, file_len):
    """Return (kind, parsed) for JSON / plist metadata files, else (None,
    None).  Bounded to METADATA_FILE_CAP bytes and normalized to JSON-safe."""
    if file_len > METADATA_FILE_CAP or file_len == 0:
        return None, None
    data, _ = read_bounded(fh, 0, file_len, METADATA_FILE_CAP)
    if data is None:
        return None, None
    try:
        if data[:1] in (b"{", b"["):
            return "json", _json_safe(json.loads(data.decode("utf-8", "replace")))
        if data.startswith(b"bplist") or data.startswith(b"<?xml"):
            return "plist", _json_safe(plistlib.loads(data))
    except (ValueError, UnicodeDecodeError, plistlib.InvalidFileException,
            struct.error, OverflowError):
        return None, None
    return None, None


def _compiler_hints(all_strings):
    hints = []
    for pat in _HINT_PATTERNS:
        for s in all_strings:
            if pat in s.lower():
                hints.append(s)
                break
    for tgt in _HINT_TARGETS:
        for s in all_strings:
            if tgt in s.lower():
                hints.append(s)
                break
    seen = set()
    uniq = []
    for h in hints:
        if h not in seen:
            seen.add(h)
            uniq.append(h)
    return uniq[:MAX_COMPILER_HINTS]


def _scan_segment_strings(fh, file_len, container, budget):
    """Scan file-backed segment regions for printable strings, bounded by the
    aggregate ``budget``.  Returns (strings, bytes_consumed)."""
    found = []
    consumed = 0
    for seg in container.get("segments", []):
        fileoff = seg.get("fileoff")
        filesize = seg.get("filesize")
        if fileoff is None or filesize is None or filesize <= 0:
            continue  # zero-fill / virtual segments have no file bytes
        if fileoff < 0 or fileoff + filesize > file_len:
            continue  # out-of-file; skip scan
        take = min(filesize, STRING_REGION_CAP, budget)
        if take <= 0:
            break
        region, _ = read_bounded(fh, fileoff, take, take)
        if region is None:
            continue
        found.extend(bounded_strings(region))
        consumed += len(region)
        budget -= len(region)
        if budget <= 0:
            break
    return found, consumed


def _scan_load_command_strings(fh, file_len, container, budget):
    hdr_len = container["header"]["header_length"]
    sizeofcmds = container["header"]["sizeofcmds"]
    if hdr_len + sizeofcmds > file_len:
        sizeofcmds = max(0, file_len - hdr_len)
    take = min(sizeofcmds, STRING_REGION_CAP, budget)
    if take <= 0:
        return [], 0
    region, _ = read_bounded(fh, hdr_len, take, take)
    if region is None:
        return [], 0
    return bounded_strings(region), len(region)


def inspect_file(path):
    """Inspect one explicit file.  Returns a dict.  Raises _InspectorError on
    unreadable / malformed input."""
    try:
        file_len = os.path.getsize(path)
        with open(path, "rb") as fh:
            sha256 = stream_sha256(fh)
            fh.seek(0)
            return _inspect_open(path, file_len, sha256, fh)
    except OSError as exc:
        raise _InspectorError("unreadable", str(exc)) from exc


def _coverage(variant):
    """Semantic coverage declaration.  The shell is container_only; instruction
    / task-descriptor semantics are deliberately unmodeled on the current
    (non-M1) target."""
    return {
        "semantic_scope": "container_only",
        "instruction_semantics": "instruction_semantics_unmodeled",
        "decoder_provenance": "none; generation-specific offsets not assumed",
        "notes": [
            "No instruction or task-descriptor opcode decoding is performed.",
            "M1 v10 / H13 register offsets are NOT applied to this target.",
            "Unknown load commands are preserved raw (type/offset/size/hash).",
            "Generic Mach-O-shaped HWX interpretations are provisional only.",
        ],
    }


def _inspect_open(path, file_len, sha256, fh):
    result = {
        "path": path,
        "file_length": file_len,
        "sha256": sha256,
        "status": "ok",
    }

    name, variant, width, endian, hdr_len = macho.detect_magic(fh, file_len)
    if name is not None:
        result["magic"] = {
            "name": name,
            "variant": variant,
            "width": width,
            "endian": endian,
            "header_length": hdr_len,
        }
        result["coverage"] = _coverage(variant)

    if name is not None and variant != "fat":
        try:
            container = macho.parse_container(fh, file_len,
                                              (name, variant, width, endian,
                                               hdr_len))
        except macho.ParseError as exc:
            result["status"] = "malformed"
            result["error"] = {"reason": exc.reason, "detail": exc.detail}
            return result
        result["container"] = container

        budget = STRING_CAP_TOTAL
        strings = []
        s, used = _scan_load_command_strings(fh, file_len, container, budget)
        strings += s
        budget -= used
        s, used = _scan_segment_strings(fh, file_len, container, budget)
        strings += s
        budget -= used
        names = macho.extract_symbol_names(fh, file_len, container, cap=SYMBOL_CAP)
        result["strings"] = {
            "count": len(strings),
            "extracted": strings[:128],
        }
        if names:
            result["symbol_names"] = {
                "count": len(names),
                "sample": names[:64],
            }
        hints = _compiler_hints(strings)
        if hints:
            result["compiler_hints"] = hints
        return result

    if variant == "fat":
        try:
            container = macho.parse_container(fh, file_len,
                                              (name, variant, width, endian,
                                               hdr_len))
            result["container"] = container
            return result
        except macho.ParseError as exc:
            result["status"] = "malformed"
            result["error"] = {"reason": exc.reason, "detail": exc.detail}
            return result

    # Not a recognized container.
    kind, parsed = _detect_text_metadata(fh, file_len)
    if kind is not None:
        result["metadata"] = {"kind": kind, "value": parsed}
        result["status"] = "metadata"
        result["coverage"] = _coverage(None)
    else:
        result["status"] = "unknown_format"
        result["error"] = {
            "reason": "unknown_format",
            "detail": "not a Mach-O / HWX / fat container, JSON, or plist",
        }
        result["coverage"] = _coverage(None)
    return result
