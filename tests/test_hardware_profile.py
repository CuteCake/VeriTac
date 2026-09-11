"""Unit tests for Hardware/profile.py.

The CUDA driver probe and helper lookups are exercised with mocked libraries and
subprocesses so the tests are deterministic and run without real hardware.  A
live Metal probe is run only when explicitly enabled via the ``LIVE_METAL``
environment variable.
"""

import ctypes
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from Hardware import profile

# cudaDevAttr* IDs used by the driver probe.
MAX_THREADS_PER_BLOCK = 1
MAX_SHARED_PER_BLOCK = 8
WARP_SIZE = 10
MULTI_PROCESSOR_COUNT = 16
L2_CACHE_SIZE = 38
MAX_SHARED_PER_MULTIPROCESSOR = 81
MAX_SHARED_PER_BLOCK_OPTIN = 97

STANDARD_ATTRS = {
    MAX_THREADS_PER_BLOCK: 1024,
    MAX_SHARED_PER_BLOCK: 49152,
    WARP_SIZE: 32,
    MULTI_PROCESSOR_COUNT: 80,
    L2_CACHE_SIZE: 6 * 1024 * 1024,
    MAX_SHARED_PER_MULTIPROCESSOR: 1024 * 1024,
    MAX_SHARED_PER_BLOCK_OPTIN: 196608,
}


class FakeCUDALib:
    """Minimal stand-in for the libcuda driver API used by the probe.

    Driver entry points are attached as plain function attributes (closures) so
    that ``set_sig`` can assign ``argtypes``/``restype`` onto them just like a
    real ``ctypes`` library object.
    """

    def __init__(self, count=1, name="Fake GPU", major=8, minor=0,
                 attrs=None, total_mem=8 * 1024**3,
                 init_rc=0, get_count_rc=0, get_rc=0, name_rc=0, cc_rc=0,
                 total_rc=0, has_total=True):
        self.attrs = dict(attrs) if attrs is not None else dict(STANDARD_ATTRS)
        self.total_mem = total_mem
        self.has_total = has_total

        def _set_sig(fn):
            fn.argtypes = []
            fn.restype = ctypes.c_int
            return fn

        def _c_int_ptr(p):
            return ctypes.cast(p, ctypes.POINTER(ctypes.c_int))

        self.cuInit = _set_sig(lambda flags: init_rc)

        def cuDeviceGetCount(out):
            _c_int_ptr(out).contents.value = count
            return get_count_rc
        self.cuDeviceGetCount = _set_sig(cuDeviceGetCount)

        def cuDeviceGet(out, ordinal):
            _c_int_ptr(out).contents.value = ordinal
            return get_rc
        self.cuDeviceGet = _set_sig(cuDeviceGet)

        def cuDeviceGetName(buf, length, dev):
            if name_rc:
                return name_rc
            buf.value = name.encode("utf-8")
            return 0
        self.cuDeviceGetName = _set_sig(cuDeviceGetName)

        def cuDeviceComputeCapability(major_out, minor_out, dev):
            if cc_rc:
                return cc_rc
            _c_int_ptr(major_out).contents.value = major
            _c_int_ptr(minor_out).contents.value = minor
            return 0
        self.cuDeviceComputeCapability = _set_sig(cuDeviceComputeCapability)

        def cuDeviceGetAttribute(out, attribute_id, dev):
            if attribute_id not in self.attrs:
                return 1  # cudaErrorInvalidValue
            _c_int_ptr(out).contents.value = self.attrs[attribute_id]
            return 0
        self.cuDeviceGetAttribute = _set_sig(cuDeviceGetAttribute)

        def cuDeviceTotalMem_v2(out, dev):
            if total_rc:
                return total_rc
            ctypes.cast(out, ctypes.POINTER(ctypes.c_size_t)).contents.value = self.total_mem
            return 0
        if has_total:
            self.cuDeviceTotalMem_v2 = _set_sig(cuDeviceTotalMem_v2)


def patch_cuda(lib):
    """Return a context manager that makes ``_probe_cuda`` see ``lib``."""
    def fake_cdll(name, *a, **k):
        if "cuda" not in name:
            raise OSError("not a cuda lib: " + name)
        return lib
    return mock.patch.object(profile.ctypes, "CDLL", side_effect=fake_cdll)


def valid_metal_profile():
    return profile.make_metal_profile(
        device_name="Apple M1",
        max_threads_per_threadgroup=1024,
        max_threadgroup_memory_bytes=32768,
        simd_width=32,
    )


def valid_cuda_profile():
    return {
        "schema_version": profile.SCHEMA_VERSION,
        "backend": "cuda",
        "device_name": "NVIDIA Fake",
        "max_threads_per_threadgroup": 1024,
        "max_threadgroup_memory_bytes": 196608,
        "simd_width": 32,
        "toolchain": {"name": "nvcc", "version": "12.3"},
        "provenance": {"source": "cuda_device_query", "host": "localhost"},
        "features": {"matrix_ops": True, "async_copy": None},
    }


class ValidateSchemaTest(unittest.TestCase):
    def _valid(self):
        return valid_metal_profile()

    def test_accepts_valid(self):
        profile.validate_schema(self._valid())
        profile.validate_schema(valid_cuda_profile())

    def test_rejects_non_dict(self):
        for bad in (None, [], "metal", 42, 3.14):
            with self.assertRaises(ValueError):
                profile.validate_schema(bad)

    def test_rejects_wrong_schema_version(self):
        p = self._valid()
        p["schema_version"] = 2
        with self.assertRaises(ValueError):
            profile.validate_schema(p)

    def test_rejects_bad_backend(self):
        for backend in (None, "opencl", "cpu", 7):
            p = self._valid()
            p["backend"] = backend
            with self.assertRaises(ValueError):
                profile.validate_schema(p)

    def test_rejects_bad_device_name(self):
        for device_name in (None, "", "   ", 123, ["gpu"]):
            p = self._valid()
            p["device_name"] = device_name
            with self.assertRaises(ValueError):
                profile.validate_schema(p)

    def test_rejects_nonpositive_or_bool_limits(self):
        for key in ("max_threads_per_threadgroup",
                    "max_threadgroup_memory_bytes", "simd_width"):
            for bad in (0, -1, True, False, 1.5, "1024", None):
                p = self._valid()
                p[key] = bad
                with self.assertRaises(ValueError):
                    profile.validate_schema(p)

    def test_rejects_malformed_toolchain(self):
        for tc in (None, "nvcc", {}, {"name": "nvcc"}, {"name": "", "version": "x"},
                   {"name": "nvcc", "version": ""}, {"name": 1, "version": "x"}):
            p = self._valid()
            p["toolchain"] = tc
            with self.assertRaises(ValueError):
                profile.validate_schema(p)

    def test_rejects_malformed_provenance(self):
        for prov in (None, "host", {}, {"source": "s"}, {"source": "", "host": "h"},
                     {"source": "s", "host": ""}, {"source": "s", "host": None}):
            p = self._valid()
            p["provenance"] = prov
            with self.assertRaises(ValueError):
                profile.validate_schema(p)

    def test_rejects_bad_feature_values(self):
        p = self._valid()
        p["features"] = "nope"
        with self.assertRaises(ValueError):
            profile.validate_schema(p)
        for val in ("yes", 1, 0, 1.5, [], {}):
            q = self._valid()
            q["features"]["matrix_ops"] = val
            with self.assertRaises(ValueError):
                profile.validate_schema(q)

    def test_accepts_bool_and_none_features(self):
        p = self._valid()
        p["features"]["matrix_ops"] = True
        p["features"]["async_copy"] = False
        p["features"]["fp16"] = None
        profile.validate_schema(p)


class SerializeValidateTest(unittest.TestCase):
    def test_canonical_json_validates(self):
        with self.assertRaises(ValueError):
            profile.canonical_json({"schema_version": 2})
        # internal probes can disable validation
        self.assertEqual(profile.canonical_json({"x": 1}, validate=False), '{"x":1}')

    def test_canonical_json_deterministic(self):
        a = valid_metal_profile()
        b = json.loads(json.dumps(a))
        self.assertEqual(profile.canonical_json(a), profile.canonical_json(b))

    def test_profile_hash_validates(self):
        with self.assertRaises(ValueError):
            profile.profile_hash({"backend": "bogus"})
        self.assertEqual(
            profile.profile_hash({"x": 1}, validate=False),
            profile.profile_hash({"x": 1}, validate=False))
        self.assertEqual(
            profile.profile_hash({"x": 1}, validate=False),
            "5041bf1f713df204784353e82f6a4a535931cb64f1f4b4a5aeaffcb720918b22")

    def test_canonical_json_nonserializable_raises_type_error(self):
        with self.assertRaises(TypeError):
            profile.canonical_json({"x": object()}, validate=False)

    def test_render_validates_and_marks_unknown(self):
        with self.assertRaises(ValueError):
            profile.render_prompt({"schema_version": 3})
        out = profile.render_prompt(valid_metal_profile())
        self.assertIn("[hardware_profile]", out)
        self.assertIn("unknown (cannot authorize)", out)
        self.assertNotIn("supported", out)

    def test_save_and_load_roundtrip(self):
        p = valid_metal_profile()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "profile.json")
            h = profile.save(p, path)
            loaded = profile.load(path)
            self.assertEqual(loaded, p)
            self.assertEqual(h, profile.profile_hash(p))

    def test_save_rejects_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "profile.json")
            with self.assertRaises(ValueError):
                profile.save({"schema_version": 99}, path)

    def test_load_rejects_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "profile.json")
            with open(path, "w") as f:
                json.dump({"schema_version": 99}, f)
            with self.assertRaises(ValueError):
                profile.load(path)


class FeatureAuthorizedTest(unittest.TestCase):
    def test_unknown_never_authorizes(self):
        p = valid_metal_profile()
        self.assertFalse(profile.feature_authorized(p, "matrix_ops"))
        self.assertFalse(profile.feature_authorized(p, "missing"))

    def test_only_explicit_true_authorizes(self):
        p = valid_metal_profile()
        p["features"]["matrix_ops"] = True
        self.assertTrue(profile.feature_authorized(p, "matrix_ops"))
        p["features"]["async_copy"] = False
        self.assertFalse(profile.feature_authorized(p, "async_copy"))

    def test_malformed_profile_cannot_authorize(self):
        with self.assertRaises(ValueError):
            profile.feature_authorized({"schema_version": 99}, "matrix_ops")


class CudaProbeTest(unittest.TestCase):
    def test_returns_none_when_libcuda_absent(self):
        def fail(name, *a, **k):
            raise OSError("no libcuda")
        with mock.patch.object(profile.ctypes, "CDLL", side_effect=fail):
            self.assertIsNone(profile._probe_cuda())

    def test_returns_none_when_cuinit_fails(self):
        lib = FakeCUDALib(init_rc=1)
        with patch_cuda(lib):
            self.assertIsNone(profile._probe_cuda())

    def test_returns_none_when_no_devices(self):
        lib = FakeCUDALib(count=0)
        with patch_cuda(lib):
            self.assertIsNone(profile._probe_cuda())

    def test_returns_none_when_get_count_fails(self):
        lib = FakeCUDALib(get_count_rc=2)
        with patch_cuda(lib):
            self.assertIsNone(profile._probe_cuda())

    def test_returns_none_when_device_get_fails(self):
        lib = FakeCUDALib(get_rc=2)
        with patch_cuda(lib):
            self.assertIsNone(profile._probe_cuda())

    def test_returns_none_when_name_fails(self):
        lib = FakeCUDALib(name_rc=3)
        with patch_cuda(lib):
            self.assertIsNone(profile._probe_cuda())

    def test_returns_none_when_compute_capability_fails(self):
        lib = FakeCUDALib(cc_rc=3)
        with patch_cuda(lib):
            self.assertIsNone(profile._probe_cuda())

    def test_returns_probe_with_attributes(self):
        lib = FakeCUDALib()
        with patch_cuda(lib):
            raw = profile._probe_cuda()
        self.assertIsNotNone(raw)
        self.assertEqual(raw["device_name"], "Fake GPU")
        self.assertEqual(raw["compute_capability"], [8, 0])
        self.assertEqual(raw["max_threads_per_block"], STANDARD_ATTRS[MAX_THREADS_PER_BLOCK])
        self.assertEqual(raw["shared_mem_per_block_default_bytes"], STANDARD_ATTRS[MAX_SHARED_PER_BLOCK])
        self.assertEqual(raw["shared_mem_per_block_optin_bytes"], STANDARD_ATTRS[MAX_SHARED_PER_BLOCK_OPTIN])
        self.assertEqual(raw["warp_size"], 32)
        self.assertEqual(raw["sm_count"], 80)
        self.assertEqual(raw["l2_cache_size_bytes"], STANDARD_ATTRS[L2_CACHE_SIZE])
        self.assertEqual(raw["shared_mem_per_multiprocessor_bytes"], STANDARD_ATTRS[MAX_SHARED_PER_MULTIPROCESSOR])
        self.assertEqual(raw["total_device_memory_bytes"], 8 * 1024**3)

    def test_total_mem_none_when_total_v2_absent(self):
        lib = FakeCUDALib(has_total=False)
        with patch_cuda(lib):
            raw = profile._probe_cuda()
        self.assertIsNotNone(raw)
        self.assertIsNone(raw["total_device_memory_bytes"])

    def test_total_mem_none_when_total_v2_fails(self):
        lib = FakeCUDALib(total_rc=1)
        with patch_cuda(lib):
            raw = profile._probe_cuda()
        self.assertIsNotNone(raw)
        self.assertIsNone(raw["total_device_memory_bytes"])

    def test_unavailable_attribute_becomes_none(self):
        attrs = dict(STANDARD_ATTRS)
        del attrs[L2_CACHE_SIZE]
        lib = FakeCUDALib(attrs=attrs)
        with patch_cuda(lib):
            raw = profile._probe_cuda()
        self.assertIsNotNone(raw)
        self.assertIsNone(raw["l2_cache_size_bytes"])

    def test_normalize_cuda_includes_extra_fields(self):
        lib = FakeCUDALib()
        with patch_cuda(lib), \
             mock.patch.object(profile, "_hostname", return_value="mock-host"), \
             mock.patch.object(profile, "_nvcc_version", return_value="12.3"):
            normalized = profile._normalize_cuda(profile._probe_cuda())
        self.assertEqual(normalized["backend"], "cuda")
        self.assertEqual(normalized["compute_capability"], [8, 0])
        self.assertEqual(normalized["total_device_memory_bytes"], 8 * 1024**3)
        self.assertEqual(normalized["toolchain"]["code_target"], "sm_80")
        self.assertEqual(normalized["performance_hints"]["optin_shared_memory_available"], True)
        profile.validate_schema(normalized)

    def test_probe_chooses_optin_shared_memory(self):
        lib = FakeCUDALib()
        with patch_cuda(lib), \
             mock.patch.object(profile, "_hostname", return_value="mock-host"), \
             mock.patch.object(profile, "_nvcc_version", return_value="12.3"):
            normalized = profile.probe(backend="cuda")
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["max_threadgroup_memory_bytes"],
                         STANDARD_ATTRS[MAX_SHARED_PER_BLOCK_OPTIN])

    def test_probe_falls_back_to_default_shared_memory(self):
        attrs = dict(STANDARD_ATTRS)
        attrs[MAX_SHARED_PER_BLOCK_OPTIN] = 0
        lib = FakeCUDALib(attrs=attrs)
        with patch_cuda(lib), \
             mock.patch.object(profile, "_hostname", return_value="mock-host"), \
             mock.patch.object(profile, "_nvcc_version", return_value="12.3"):
            normalized = profile.probe(backend="cuda")
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["max_threadgroup_memory_bytes"],
                         STANDARD_ATTRS[MAX_SHARED_PER_BLOCK])

    def test_probe_returns_none_on_invalid_shared_memory(self):
        attrs = dict(STANDARD_ATTRS)
        attrs[MAX_SHARED_PER_BLOCK_OPTIN] = 0
        attrs[MAX_SHARED_PER_BLOCK] = 0
        lib = FakeCUDALib(attrs=attrs)
        with patch_cuda(lib), \
             mock.patch.object(profile, "_hostname", return_value="mock-host"), \
             mock.patch.object(profile, "_nvcc_version", return_value="12.3"):
            self.assertIsNone(profile.probe(backend="cuda"))


class NvccLookupTest(unittest.TestCase):
    def test_nvcc_version_falls_back_to_usr_local(self):
        with mock.patch.object(profile.shutil, "which", return_value=None), \
             mock.patch.object(profile.os.path, "exists", return_value=True), \
             mock.patch.object(profile.subprocess, "run") as run:
            run.return_value = mock.Mock(
                returncode=0, stdout="Cuda compilation tools, release 12.3, V12.3.0\n")
            self.assertEqual(profile._nvcc_version(), "12.3")
            self.assertEqual(run.call_args[0][0], ["/usr/local/cuda/bin/nvcc", "--version"])

    def test_nvcc_version_unknown_when_missing(self):
        with mock.patch.object(profile.shutil, "which", return_value=None), \
             mock.patch.object(profile.os.path, "exists", return_value=False):
            self.assertEqual(profile._nvcc_version(), "unknown")


class MetalProbeCleanupTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("LIVE_METAL"), "set LIVE_METAL to run live Metal probe")
    def test_live_metal_probe(self):
        raw = profile._probe_metal()
        self.assertIsNotNone(raw)
        self.assertIn("device_name", raw)
        self.assertIn("max_total_threads_per_threadgroup", raw)

    def test_tempdir_removes_source_and_binary(self):
        # Force a non-cached probe that compiles then fails to run the binary.
        fake_src = os.path.join(tempfile.gettempdir(), "veritac_metal_probe_dummy.m")

        def fake_cc(binary, src):
            with open(binary, "w") as f:
                f.write("#!/bin/sh\n")
            os.chmod(binary, 0o755)

        def fake_run(cmd, **kw):
            if cmd and os.path.basename(cmd[0]) == "probe":
                # Binary exists (created by fake_cc) and would run; make it fail.
                return mock.Mock(returncode=1, stdout="", stderr="")
            return mock.Mock(returncode=0, stdout="{}")

        with mock.patch.object(profile, "_cc_metal", side_effect=fake_cc), \
             mock.patch.object(profile, "tempfile", spec=tempfile) as tmp_mock, \
             mock.patch.object(profile.platform, "system", return_value="Darwin"), \
             mock.patch.object(profile.shutil, "which", return_value="/usr/bin/clang"):
            tmpdir = tempfile.mkdtemp(prefix="veritac_metal_")
            tmp_mock.mkdtemp.return_value = tmpdir
            src = os.path.join(tmpdir, "probe.m")
            binary = os.path.join(tmpdir, "probe")
            with open(src, "w") as f:
                f.write(_METAL_SOURCE_SENTINEL)
            with mock.patch.object(profile.subprocess, "run", side_effect=fake_run):
                profile._probe_metal()
            # Source and binary must both be removed, and the tempdir gone.
            self.assertFalse(os.path.exists(src))
            self.assertFalse(os.path.exists(binary))
            self.assertFalse(os.path.exists(tmpdir))


_METAL_SOURCE_SENTINEL = "int main(void){return 0;}\n"


if __name__ == "__main__":
    unittest.main()
