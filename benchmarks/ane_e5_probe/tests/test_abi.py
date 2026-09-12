"""Pure (no-hardware) validation tests for ane_e5_probe.

Validates that the ctypes ABI binding matches the authoritative e5rt_api.h,
plus pure CLI/journal/inventory logic and a full fake-functions compile /
prepare / cleanup flow. Nothing here dlopens or calls the private Espresso
framework (no hardware calls).

Run from the repo root:
    PYTHONPATH=. python3 -m unittest \
        benchmarks/ane_e5_probe/tests/test_abi.py -v
"""

import ctypes as C
import importlib.util
import json
import os
import re
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "..", "ane_e5_probe.py")
DEFAULT_HEADER = os.path.join(HERE, "..", "reference", "e5rt_api.h")
HEADER = os.environ.get("ANE_E5RT_API_H", DEFAULT_HEADER)


def load_probe():
    spec = importlib.util.spec_from_file_location("ane_e5_probe", PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = load_probe()

P = probe.P
PP = probe.PP
CHAR = probe.CHAR
CU64 = probe.CU64
CINT = probe.CINT

TYPEDEF_RE = re.compile(
    r"typedef\s+e5rt_error_code_t\s+\(\*(\w+_fn)\)\((.*?)\);", re.S)


def ctypes_for_param(param):
    """Map a header parameter declaration to the expected ctypes argtype."""
    m = re.match(r"\s*(.*?)\s*(\w+)\s*$", param)
    if not m:
        raise AssertionError("cannot parse param: %r" % param)
    t, _name = m.groups()
    stars = t.count("*")
    base = t.replace("*", "").replace("const", "").replace(" ", "").strip()
    if base == "void":
        return PP if stars == 2 else P
    if base == "char":
        return C.POINTER(CHAR) if stars == 2 else CHAR
    if base == "uint64_t":
        return C.POINTER(CU64) if stars >= 1 else CU64
    if base == "int":
        return CINT
    raise AssertionError("unknown base type in %r" % t)


def header_argtypes():
    """Return {symbol: [argtypes]} parsed from the authoritative header,
    restricted to symbols actually bound in probe.ABI."""
    with open(HEADER) as f:
        text = f.read()
    out = {}
    for fnname, params in TYPEDEF_RE.findall(text):
        sym = fnname[:-3]  # strip "_fn"
        if sym not in probe.ABI:
            continue
        if params.strip() == "void":
            out[sym] = []
            continue
        out[sym] = [ctypes_for_param(p) for p in params.split(",")]
    return out


def fake_rt(include_prepare=False):
    """Build ctypes-wrapped fake e5rt functions that simulate a successful
    compile/library/prepare session without any hardware. Returns (fns, calls)."""
    calls = []
    addr = [0x1000]
    fns = {}

    def add(sym, impl):
        restype, argtypes = probe.ABI[sym]
        fns[sym] = C.CFUNCTYPE(restype, *argtypes)(impl)

    def rec(sym):
        def impl(*a):
            calls.append(sym)
            return 0
        return impl

    for sym in probe.COMPILE_SYMBOLS:
        add(sym, rec(sym))

    def out_first(*args):
        calls.append("out_first")
        args[0][0] = addr[0] + 1
        return 0

    def out_last(*args):
        calls.append("out_last")
        args[-1][0] = addr[0] + 1
        return 0

    # Out-FIRST creates
    add("e5rt_e5_compiler_config_options_create", out_first)
    add("e5rt_e5_compiler_options_create", out_first)
    add("e5rt_e5_compiler_create_with_config", out_first)  # (out, config)
    add("e5rt_program_library_create", out_first)          # (out, path)
    # Compile: out-LAST (compiler, path, options, PP out)
    add("e5rt_e5_compiler_compile", out_last)
    # get_num_functions: (library, uint64_t* out)
    add("e5rt_program_library_get_num_functions",
        lambda lib, out: (calls.append("get_num_functions"), out.__setitem__(0, 1), 0)[2])
    # get_function_names: (library, count, const char** names)
    add("e5rt_program_library_get_function_names",
        lambda lib, n, names: (calls.append("get_function_names"), names.__setitem__(0, b"main"), 0)[2])

    if include_prepare:
        for sym in probe.PREPARE_SYMBOLS:
            add(sym, rec(sym))
        add("e5rt_program_library_retain_program_function", out_last)  # (library, name, out)
        add("e5rt_precompiled_compute_op_create_options_create_with_program_function", out_first)
        add("e5rt_execution_stream_operation_create_precompiled_compute_operation_with_options", out_first)

    return fns, calls


@unittest.skipUnless(os.path.isfile(HEADER), "e5rt_api.h not found (%s)" % HEADER)
class TestAbiAgainstHeader(unittest.TestCase):
    def setUp(self):
        self.header = header_argtypes()

    def test_every_bound_symbol_matches_header(self):
        for sym, (restype, argtypes) in probe.ABI.items():
            self.assertIn(sym, self.header, "symbol %s not in header" % sym)
            self.assertEqual(
                list(argtypes), self.header[sym],
                "ABI argtypes for %s diverge from header: %s vs %s"
                % (sym, argtypes, self.header[sym]))

    def test_restype_is_error_code(self):
        for sym, (restype, _argtypes) in probe.ABI.items():
            self.assertEqual(restype, C.c_int64, sym)

    def test_create_with_config_is_out_first_config_second(self):
        self.assertEqual(probe.ABI["e5rt_e5_compiler_create_with_config"][1], (PP, P))

    def test_releases_take_void_star_star(self):
        for sym in probe.ABI:
            if sym.endswith("_release"):
                self.assertEqual(probe.ABI[sym][1], (PP,), sym)

    def test_program_library_create_is_out_first(self):
        self.assertEqual(probe.ABI["e5rt_program_library_create"][1], (PP, CHAR))

    def test_retain_program_function_is_out_last(self):
        self.assertEqual(probe.ABI["e5rt_program_library_retain_program_function"][1],
                         (P, CHAR, PP))

    def test_prepare_creates_are_out_first(self):
        self.assertEqual(
            probe.ABI["e5rt_precompiled_compute_op_create_options_create_with_program_function"][1],
            (PP, P))
        self.assertEqual(
            probe.ABI["e5rt_execution_stream_operation_create_precompiled_compute_operation_with_options"][1],
            (PP, P))

    def test_no_error_code_get_string_used(self):
        for sym in probe.ABI:
            self.assertNotIn("get_string", sym)


class TestPureLogic(unittest.TestCase):
    def test_make_outdir_atomic_reject(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(probe.ProbeError):
                probe.make_outdir(d)
            with self.assertRaises(probe.ProbeError):
                probe.make_outdir(d)

    def test_make_outdir_creates_fresh(self):
        with tempfile.TemporaryDirectory() as d:
            probe.make_outdir(os.path.join(d, "sub"))
            self.assertTrue(os.path.isdir(os.path.join(d, "sub")))

    def test_inventory_hashes_own_dir_skips_symlinks(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a.bin"), "wb") as f:
                f.write(b"\x00\x01")
            outside = os.path.join(d, "..", "outside.bin")
            with open(outside, "wb") as f:
                f.write(b"secret")
            os.symlink(outside, os.path.join(d, "link.bin"))
            os.mkdir(os.path.join(d, "sub"))
            os.symlink(d, os.path.join(d, "sub", "loop"))
            inv = probe.inventory_dir(d)
            self.assertEqual([i["path"] for i in inv], ["a.bin"])
            self.assertEqual(inv[0]["sha256"], probe.sha256_file(os.path.join(d, "a.bin")))

    def test_journal_stage_after_fail(self):
        with tempfile.TemporaryDirectory() as d:
            j = probe.Journal(os.path.join(d, "journal.json"), {"meta": 1})
            j.stage("compile")
            j.after(5)
            j.fail("boom")
            doc = json.load(open(os.path.join(d, "journal.json")))
            self.assertEqual(doc["status"], "failed")
            self.assertEqual(doc["error"], "boom")
            self.assertEqual(doc["stages"][-1]["stage"], "compile")
            self.assertEqual(doc["stages"][-1]["rc"], 5)

    def test_failure_final_json_has_inventory_and_error(self):
        with tempfile.TemporaryDirectory() as d:
            outdir = os.path.join(d, "out")
            os.makedirs(outdir)
            with open(os.path.join(outdir, "x.bin"), "wb") as f:
                f.write(b"\x00")
            j = probe.Journal(os.path.join(outdir, "journal.json"), {})
            j.stage("compile")
            j.after(7)
            probe._write_failure(outdir, j, "compile returned rc=7")
            doc = json.load(open(os.path.join(outdir, "final.json")))
            self.assertEqual(doc["status"], "failed")
            self.assertEqual(doc["error"], "compile returned rc=7")
            self.assertTrue(doc["inventory_newdir"])
            self.assertTrue(doc["preserved_failed_artifacts"])

    def test_capability_map_covers_all_abi(self):
        caps = probe.capability_map()
        self.assertEqual(set(caps["symbols"]), set(probe.ABI))


class TestFakeCompileFlow(unittest.TestCase):
    def _setup(self):
        d = tempfile.TemporaryDirectory()
        outdir = os.path.join(d.name, "out")
        os.makedirs(outdir)
        mil = os.path.join(d.name, "model.mil")
        with open(mil, "w") as f:
            f.write("program(1.3)\n")
        j = probe.Journal(os.path.join(outdir, "journal.json"), {"m": 1})
        return d, outdir, mil, j

    def test_compile_cleanup_reverse_order_and_completion(self):
        fns, calls = fake_rt()
        d, outdir, mil, j = self._setup()
        try:
            res = probe.run_compile(fns, j, mil, outdir, time.monotonic() + 100, False)
            self.assertEqual(res["status"], "completed")
            self.assertEqual(res["function_names"], ["main"])
            doc = json.load(open(os.path.join(outdir, "journal.json")))
            self.assertEqual(doc["status"], "completed")  # only after cleanup
            rel = [c for c in calls if c.endswith("_release")]
            self.assertEqual(rel, [
                "e5rt_program_library_release",
                "e5rt_e5_compiler_options_release",
                "e5rt_e5_compiler_release",
                "e5rt_e5_compiler_config_options_release",
            ])
        finally:
            d.cleanup()

    def test_compile_prepare_creates_op_and_releases_reverse(self):
        fns, calls = fake_rt(include_prepare=True)
        d, outdir, mil, j = self._setup()
        try:
            res = probe.run_compile(fns, j, mil, outdir, time.monotonic() + 100, True)
            self.assertEqual(res["prepare"]["status"], "created")
            rel = [c for c in calls if c.endswith("_release")]
            self.assertEqual(rel, [
                "e5rt_execution_stream_operation_release",
                "e5rt_precompiled_compute_op_create_options_release",
                "e5rt_program_function_release",
                "e5rt_program_library_release",
                "e5rt_e5_compiler_options_release",
                "e5rt_e5_compiler_release",
                "e5rt_e5_compiler_config_options_release",
            ])
        finally:
            d.cleanup()

    def test_cleanup_attempts_all_when_one_release_fails(self):
        fns, calls = fake_rt()
        # make program_library_release fail (rc=1)
        fns["e5rt_program_library_release"] = C.CFUNCTYPE(
            C.c_int64, PP)(lambda obj: (calls.append("e5rt_program_library_release"), 1)[1])
        d, outdir, mil, j = self._setup()
        try:
            with self.assertRaises(probe.ProbeError):
                probe.run_compile(fns, j, mil, outdir, time.monotonic() + 100, False)
            # all four releases still attempted even though the first failed
            rel = [c for c in calls if c.endswith("_release")]
            self.assertEqual(len(rel), 4)
            doc = json.load(open(os.path.join(outdir, "journal.json")))
            self.assertEqual(doc["status"], "in_progress")  # not completed on cleanup error
        finally:
            d.cleanup()

    def test_function_count_cap_and_null_name_rejected(self):
        def many(lib, out):
            out[0] = 65
            return 0
        fns, _ = fake_rt()
        fns["e5rt_program_library_get_num_functions"] = C.CFUNCTYPE(
            C.c_int64, P, C.POINTER(C.c_uint64))(many)
        d, outdir, mil, j = self._setup()
        try:
            with self.assertRaisesRegex(probe.ProbeError, "exceeds cap"):
                probe.run_compile(fns, j, mil, outdir, time.monotonic() + 100, False)
        finally:
            d.cleanup()

        def one(lib, out):
            out[0] = 1
            return 0

        def nullname(lib, n, names):
            names[0] = None
            return 0
        fns2, _ = fake_rt()
        fns2["e5rt_program_library_get_num_functions"] = C.CFUNCTYPE(
            C.c_int64, P, C.POINTER(C.c_uint64))(one)
        fns2["e5rt_program_library_get_function_names"] = C.CFUNCTYPE(
            C.c_int64, P, C.c_uint64, C.POINTER(CHAR))(nullname)
        d, outdir, mil, j = self._setup()
        try:
            with self.assertRaisesRegex(probe.ProbeError, "NULL function name"):
                probe.run_compile(fns2, j, mil, outdir, time.monotonic() + 100, False)
        finally:
            d.cleanup()


if __name__ == "__main__":
    unittest.main()
