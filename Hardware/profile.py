"""Hardware profiles for VeriTac.

Collects device limits into a normalized, unit-bearing target description,
serializes it to canonical JSON with a SHA-256 hash, and renders a deterministic
prompt fragment.  Probing is best-effort: any backend or field that cannot be
observed stays ``None`` (unknown) and an unknown value can never authorize a
feature.  No external Python dependencies are required.

Schema (``schema_version = 1``) for a normalized target::

    {
      "schema_version": 1,
      "backend": "metal" | "cuda",
      "device_name": str,
      "max_threads_per_threadgroup": int,      # normalized max_threads_per_block
      "max_threadgroup_memory_bytes": int,     # normalized default/opt-in shared mem
      "simd_width": int,                       # Metal threadExecutionWidth / CUDA warp size
      "toolchain": {"name": str, "version": str},
      "provenance": {"source": str, "host": str},
      "features": {<name>: bool | None}
    }

Backend-specific names are retained alongside the normalized names so the saved
profile preserves the observed units.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile

SCHEMA_VERSION = 1

_PROBE_METAL_SOURCE = r"""
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static void print_string(const char *key, NSString *s) {
    if (!s) { printf("\"%s\":null", key); return; }
    printf("\"%s\":\"%s\"", key, [s UTF8String]);
}

int main(void) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { printf("{\"error\":\"no_default_device\"}\n"); return 1; }

    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@"kernel void k(void){}"
                                            options:nil error:&err];
    if (!lib) { printf("{\"error\":\"library_compile\"}\n"); return 2; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"k"];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { printf("{\"error\":\"pso_compile\"}\n"); return 3; }

    printf("{");
    print_string("device_name", [dev name]);
    printf(",");
    printf("\"thread_execution_width\":%lu", (unsigned long)[pso threadExecutionWidth]);
    printf(",");
    printf("\"max_total_threads_per_threadgroup\":%lu", (unsigned long)[pso maxTotalThreadsPerThreadgroup]);
    printf(",");
    printf("\"max_threadgroup_memory_bytes\":%lu", (unsigned long)[dev maxThreadgroupMemoryLength]);
    MTLSize m = [dev maxThreadsPerThreadgroup];
    printf(",");
    printf("\"max_threads_per_threadgroup_width\":%lu", (unsigned long)m.width);
    printf(",");
    printf("\"max_threads_per_threadgroup_height\":%lu", (unsigned long)m.height);
    printf(",");
    printf("\"max_threads_per_threadgroup_depth\":%lu", (unsigned long)m.depth);
    printf("}\n");
    return 0;
}
"""


def _hostname() -> str:
    try:
        return os.uname().nodename or "unknown"
    except Exception:
        return "unknown"


def _metal_sdk_version() -> str:
    try:
        out = subprocess.run(
            ["xcrun", "--sdk", "macosx", "--show-sdk-version"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _probe_metal() -> dict | None:
    """Probe the default Metal device via a compiled C helper (no Python deps)."""
    if platform.system() != "Darwin":
        return None
    if shutil.which("clang") is None:
        return None
    cache_dir = os.environ.get("VERITAC_CACHE_DIR")
    binary = None
    tmp = None
    try:
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            binary = os.path.join(cache_dir, "veritac_probe_metal")
            if not os.path.exists(binary):
                tmp = os.path.join(cache_dir, "veritac_probe_metal.m")
                with open(tmp, "w") as f:
                    f.write(_PROBE_METAL_SOURCE)
                _cc_metal(binary, tmp)
        else:
            tmpdir = tempfile.mkdtemp(prefix="veritac_metal_")
            tmp = os.path.join(tmpdir, "probe.m")
            binary = os.path.join(tmpdir, "probe")
            with open(tmp, "w") as f:
                f.write(_PROBE_METAL_SOURCE)
            _cc_metal(binary, tmp)
        out = subprocess.run([binary], capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            return None
        data = json.loads(out.stdout)
        if "error" in data:
            return None
        return data
    except Exception:
        return None
    finally:
        if not cache_dir and tmp is not None:
            shutil.rmtree(os.path.dirname(tmp), ignore_errors=True)


def _cc_metal(binary: str, src: str) -> None:
    subprocess.run(
        ["clang", "-x", "objective-c", "-fobjc-arc",
         "-framework", "Foundation", "-framework", "Metal",
         "-o", binary, src],
        capture_output=True, text=True, check=True, timeout=120,
    )


def _nvcc_path() -> str | None:
    p = shutil.which("nvcc")
    if p is None:
        cand = "/usr/local/cuda/bin/nvcc"
        if os.path.exists(cand):
            p = cand
    return p


def _nvcc_version() -> str:
    path = _nvcc_path()
    if path is None:
        return "unknown"
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30)
        for line in out.stdout.splitlines():
            if "release" in line and "," in line:
                return line.split("release")[1].split(",")[0].strip()
    except Exception:
        pass
    return "unknown"


def _probe_cuda() -> dict | None:
    """Probe the first CUDA device via the libcuda driver API (ctypes).

    Uses ``cuInit``/``cuDeviceGetCount``/``cuDeviceGet``/``cuDeviceGetName``,
    ``cuDeviceComputeCapability``, ``cuDeviceGetAttribute`` with the documented
    attribute IDs, and ``cuDeviceTotalMem_v2`` when present.  Every return code
    is checked; optional attributes that are unavailable become ``None``.
    Returns None if the driver is absent or no device is present.
    """
    libcuda = None
    for cand in ("libcuda.so.1", "libcuda.dylib", "libcuda.so"):
        try:
            libcuda = ctypes.CDLL(cand)
            break
        except OSError:
            continue
    if libcuda is None:
        return None

    def set_sig(name, argtypes, restype):
        fn = getattr(libcuda, name, None)
        if fn is not None:
            fn.argtypes = argtypes
            fn.restype = restype
        return fn

    set_sig("cuInit", [ctypes.c_uint], ctypes.c_int)
    set_sig("cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)], ctypes.c_int)
    set_sig("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.c_int)
    set_sig("cuDeviceGetName", [ctypes.c_char_p, ctypes.c_int, ctypes.c_int], ctypes.c_int)
    set_sig("cuDeviceComputeCapability",
            [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int],
            ctypes.c_int)
    set_sig("cuDeviceGetAttribute",
            [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int], ctypes.c_int)
    cu_total = set_sig("cuDeviceTotalMem_v2",
                       [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int], ctypes.c_int)

    if not hasattr(libcuda, "cuInit"):
        return None
    try:
        if libcuda.cuInit(0) != 0:
            return None
        count = ctypes.c_int()
        if libcuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return None
        if count.value < 1:
            return None
        dev = ctypes.c_int()
        if libcuda.cuDeviceGet(ctypes.byref(dev), 0) != 0:
            return None

        name_buf = ctypes.create_string_buffer(256)
        if libcuda.cuDeviceGetName(name_buf, 256, dev.value) != 0:
            return None
        name = name_buf.value.decode("utf-8", "replace").rstrip("\x00")

        major = ctypes.c_int()
        minor = ctypes.c_int()
        if libcuda.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), dev.value) != 0:
            return None

        def attr(attribute_id):
            val = ctypes.c_int()
            if libcuda.cuDeviceGetAttribute(ctypes.byref(val), attribute_id, dev.value) != 0:
                return None
            return val.value

        # Documented cudaDevAttr* IDs.
        max_threads_per_block = attr(1)                     # cudaDevAttrMaxThreadsPerBlock
        shared_mem_per_block_default = attr(8)              # cudaDevAttrMaxSharedMemoryPerBlock
        warp_size = attr(10)                                # cudaDevAttrWarpSize
        sm_count = attr(16)                                 # cudaDevAttrMultiProcessorCount
        l2_cache_size = attr(38)                            # cudaDevAttrL2CacheSize
        shared_mem_per_multiprocessor = attr(81)            # cudaDevAttrMaxSharedMemoryPerMultiprocessor
        shared_mem_per_block_optin = attr(97)               # cudaDevAttrMaxSharedMemoryPerBlockOptin

        total_device_memory_bytes = None
        if cu_total is not None:
            total_mem = ctypes.c_size_t()
            if libcuda.cuDeviceTotalMem_v2(ctypes.byref(total_mem), dev.value) == 0:
                total_device_memory_bytes = int(total_mem.value)

        return {
            "device_name": name,
            "compute_capability": [major.value, minor.value],
            "total_device_memory_bytes": total_device_memory_bytes,
            "warp_size": warp_size,
            "max_threads_per_block": max_threads_per_block,
            "shared_mem_per_block_default_bytes": shared_mem_per_block_default,
            "shared_mem_per_block_optin_bytes": shared_mem_per_block_optin,
            "sm_count": sm_count,
            "l2_cache_size_bytes": l2_cache_size,
            "shared_mem_per_multiprocessor_bytes": shared_mem_per_multiprocessor,
        }
    except Exception:
        return None


def _normalize_metal(raw: dict) -> dict:
    host = _hostname()
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "metal",
        "device_name": raw.get("device_name"),
        "max_threads_per_threadgroup": raw.get("max_total_threads_per_threadgroup"),
        "max_threadgroup_memory_bytes": raw.get("max_threadgroup_memory_bytes"),
        "max_threads_per_threadgroup_size": {
            "width": raw.get("max_threads_per_threadgroup_width"),
            "height": raw.get("max_threads_per_threadgroup_height"),
            "depth": raw.get("max_threads_per_threadgroup_depth"),
        },
        "simd_width": raw.get("thread_execution_width"),
        "toolchain": {"name": "metal_sdk", "version": _metal_sdk_version()},
        "provenance": {"source": "metal_device_query", "host": host},
        "features": {
            "matrix_ops": None,
            "async_copy": None,
            "fp16": None,
            "bfloat16": None,
        },
    }


def _performance_hints_cuda(raw: dict) -> dict:
    optin = raw.get("shared_mem_per_block_optin_bytes")
    default = raw.get("shared_mem_per_block_default_bytes")
    optin_available = None
    if optin is not None and default is not None:
        optin_available = optin > default
    return {
        "optin_shared_memory_available": optin_available,
        "sm_count": raw.get("sm_count"),
        "l2_cache_size_bytes": raw.get("l2_cache_size_bytes"),
    }


def _normalize_cuda(raw: dict) -> dict:
    host = _hostname()
    max_threads_per_threadgroup = raw.get("max_threads_per_block")
    optin = raw.get("shared_mem_per_block_optin_bytes")
    default = raw.get("shared_mem_per_block_default_bytes")
    if isinstance(optin, int) and not isinstance(optin, bool) and optin > 0:
        max_threadgroup_memory_bytes = optin
    elif isinstance(default, int) and not isinstance(default, bool) and default > 0:
        max_threadgroup_memory_bytes = default
    else:
        max_threadgroup_memory_bytes = None
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "cuda",
        "device_name": raw.get("device_name"),
        "max_threads_per_threadgroup": max_threads_per_threadgroup,
        "max_threadgroup_memory_bytes": max_threadgroup_memory_bytes,
        "max_threads_per_block": raw.get("max_threads_per_block"),
        "shared_mem_per_block_default_bytes": raw.get("shared_mem_per_block_default_bytes"),
        "shared_mem_per_block_optin_bytes": raw.get("shared_mem_per_block_optin_bytes"),
        "simd_width": raw.get("warp_size"),
        "compute_capability": raw.get("compute_capability"),
        "total_device_memory_bytes": raw.get("total_device_memory_bytes"),
        "performance_hints": _performance_hints_cuda(raw),
        "toolchain": {
            "name": "nvcc",
            "version": _nvcc_version(),
            "code_target": "sm_{}{}".format(raw["compute_capability"][0], raw["compute_capability"][1])
            if raw.get("compute_capability") else None,
        },
        "provenance": {"source": "cuda_device_query", "host": host},
        "features": {
            "matrix_ops": None,
            "async_copy": None,
            "fp16": None,
            "bfloat16": None,
        },
    }


def probe(backend: str | None = None) -> dict | None:
    """Probe a backend and return a normalized target profile, or None.

    ``backend`` is ``"metal"`` or ``"cuda"``; when omitted, Metal is probed on
    macOS and CUDA is probed when present (Metal takes precedence on macOS).
    """
    candidates = []
    if backend is not None:
        candidates = [backend]
    else:
        if platform.system() == "Darwin":
            candidates = ["metal", "cuda"]
        else:
            candidates = ["cuda", "metal"]

    for b in candidates:
        if b == "metal":
            raw = _probe_metal()
            if raw is not None:
                normalized = _normalize_metal(raw)
                try:
                    validate_schema(normalized)
                    return normalized
                except ValueError:
                    return None
        elif b == "cuda":
            raw = _probe_cuda()
            if raw is not None:
                normalized = _normalize_cuda(raw)
                try:
                    validate_schema(normalized)
                    return normalized
                except ValueError:
                    return None
    return None


def _sort_key(item):
    key, _ = item
    return key


def canonical_json(profile: dict, validate: bool = True) -> str:
    """Serialize a profile to deterministic, canonical JSON (sorted keys, compact).

    External profiles are validated against the schema unless ``validate`` is
    explicitly disabled (internal normalized probes).
    """
    if validate:
        validate_schema(profile)
    return json.dumps(profile, sort_keys=True, separators=(",", ":"))


def profile_hash(profile: dict, validate: bool = True) -> str:
    """SHA-256 of the canonical JSON representation."""
    return hashlib.sha256(
        canonical_json(profile, validate=validate).encode("utf-8")).hexdigest()


def render_prompt(profile: dict, validate: bool = True) -> str:
    """Render a deterministic, unit-bearing profile fragment for a prompt.

    Unknown fields render as ``unknown`` and are explicitly marked as unable to
    authorize a feature.
    """
    if validate:
        validate_schema(profile)
    lines = ["[hardware_profile]"]
    lines.append("schema_version: {}".format(profile.get("schema_version", "unknown")))
    lines.append("backend: {}".format(profile.get("backend", "unknown")))
    lines.append("device_name: {}".format(profile.get("device_name") or "unknown"))
    lines.append("max_threads_per_threadgroup: {}".format(
        _fmt_unknown(profile.get("max_threads_per_threadgroup"))))
    lines.append("max_threadgroup_memory_bytes: {}".format(
        _fmt_unknown(profile.get("max_threadgroup_memory_bytes"))))
    lines.append("simd_width: {}".format(_fmt_unknown(profile.get("simd_width"))))
    tc = profile.get("toolchain") or {}
    lines.append("toolchain: {} {}".format(
        tc.get("name") or "unknown", tc.get("version") or "").strip())
    prov = profile.get("provenance") or {}
    lines.append("provenance: {} on {}".format(
        prov.get("source") or "unknown", prov.get("host") or "unknown"))
    lines.append("features:")
    for name in sorted((profile.get("features") or {}).keys()):
        val = (profile.get("features") or {}).get(name)
        if val is True:
            status = "supported"
        elif val is False:
            status = "unsupported"
        else:
            status = "unknown (cannot authorize)"
        lines.append("  {}: {}".format(name, status))
    return "\n".join(lines) + "\n"


def _fmt_unknown(v):
    return "unknown" if v is None else str(v)


def feature_authorized(profile: dict, name: str) -> bool:
    """A feature is authorized only if the profile records it explicitly as True.

    Unknown (``None``) support never authorizes a feature.
    """
    validate_schema(profile)
    return (profile.get("features") or {}).get(name) is True


def save(profile: dict, path: str) -> str:
    """Write a canonical profile to ``path``; returns the SHA-256 hash."""
    text = canonical_json(profile)
    with open(path, "w") as f:
        f.write(text + "\n")
    return profile_hash(profile)


def load(path: str) -> dict:
    """Load a saved profile, validating the schema."""
    with open(path) as f:
        profile = json.load(f)
    validate_schema(profile)
    return profile


def validate_schema(profile) -> None:
    """Validate a normalized profile against schema_version 1.

    Raises ``ValueError`` on any malformed or missing required field.  Unknown
    (``None``) feature values are permitted but never authorize a feature.
    """
    if not isinstance(profile, dict):
        raise ValueError("profile must be a dict, got {}".format(type(profile).__name__))
    if profile.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "unsupported schema_version {} (expected {})".format(
                profile.get("schema_version"), SCHEMA_VERSION))

    backend = profile.get("backend")
    if backend not in ("metal", "cuda"):
        raise ValueError("backend must be 'metal' or 'cuda', got {!r}".format(backend))

    device_name = profile.get("device_name")
    if not isinstance(device_name, str) or not device_name.strip():
        raise ValueError("device_name must be a non-empty string")

    for key in ("max_threads_per_threadgroup", "max_threadgroup_memory_bytes", "simd_width"):
        val = profile.get(key)
        if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
            raise ValueError(
                "{} must be a positive int, got {!r}".format(key, val))

    _validate_string_fields(profile.get("toolchain"), ("name", "version"), "toolchain")
    _validate_string_fields(profile.get("provenance"), ("source", "host"), "provenance")

    features = profile.get("features")
    if not isinstance(features, dict):
        raise ValueError("features must be a dict, got {!r}".format(type(features).__name__))
    for name, val in features.items():
        if val is not None and not isinstance(val, bool):
            raise ValueError(
                "feature {!r} must be bool or None, got {!r}".format(name, val))


def _validate_string_fields(obj, required, label):
    if not isinstance(obj, dict):
        raise ValueError("{} must be a dict, got {!r}".format(label, type(obj).__name__))
    for field in required:
        val = obj.get(field)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(
                "{} . {} must be a non-empty string, got {!r}".format(label, field, val))


def make_metal_profile(device_name, max_threads_per_threadgroup,
                       max_threadgroup_memory_bytes, simd_width,
                       toolchain_version="26.5", host="localhost"):
    """Construct a normalized Metal profile dict (used by tests)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "metal",
        "device_name": device_name,
        "max_threads_per_threadgroup": max_threads_per_threadgroup,
        "max_threadgroup_memory_bytes": max_threadgroup_memory_bytes,
        "simd_width": simd_width,
        "toolchain": {"name": "metal_sdk", "version": toolchain_version},
        "provenance": {"source": "metal_device_query", "host": host},
        "features": {
            "matrix_ops": None,
            "async_copy": None,
            "fp16": None,
            "bfloat16": None,
        },
    }


if __name__ == "__main__":
    p = probe()
    if p is None:
        print("no backend probed")
    else:
        print(canonical_json(p))
        print("hash:", profile_hash(p))
        print(render_prompt(p), end="")
