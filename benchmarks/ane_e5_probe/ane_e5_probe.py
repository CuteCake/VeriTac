#!/usr/bin/env python3
"""ane_e5_probe.py - bounded E5 compiled-bundle export/reopen probe.

Binds Apple's private Espresso ``e5rt_*`` C ABI through ctypes (dlsym only),
then runs one of three subcommands:

  capabilities  -- report which e5rt_* symbols resolve (may run; no hardware)
  compile       -- compile a MIL into an ANE E5 bundle, record the library's
                   function names, then release. No load/eval.
  reopen        -- independently open a compiled bundle via
                   e5rt_program_library_create + list its functions
                   (no compiler-API calls).

Optional ``--prepare`` (default off) additionally retains program function
"main", builds precompiled-op create options and creates the precompiled
operation, then releases all in reverse. This distinguishes *library reopen*
from *hardware preparation*; it never executes, creates a stream, or binds
inputs. See README for the root-run positive (same-process) / negative
(fresh-process) control procedure.

ABI is taken verbatim from ``aneforge/_lib/e5rt_api.h`` (pinned ANEForge
67c1d4861c562b0168d60c7a5677840e2cad44ea). The typedefs are authoritative
individually: ``create_with_config`` is (out FIRST, config SECOND),
``create_with_program_function``/``...with_options`` are out FIRST, and every
``release`` takes ``void **``. We never call ``e5rt_error_code_get_string``
(it has a C++ std::string ABI).

Every call checks rc and non-NULL out, cleans up in reverse (attempting all
handles even if one release fails) and flushes a JSON stage journal
before/after each call so a crash inside a private API call is attributable
to the exact stage. Only own-NEWdir and the explicit bundle are hashed
(regular files only; symlinks/non-regular files are skipped). No private-cache
scans, no signing, no sudo.

DO NOT run compile/reopen until root review; root orchestrates the parent
timeout process. This file is import-safe (no framework is touched at import)
so the pure ABI/CLI logic can be unit-tested without hardware.
"""

import argparse
import ctypes as C
import hashlib
import json
import os
import shutil
import sys
import time

FRAMEWORK = "/System/Library/PrivateFrameworks/Espresso.framework/Espresso"
DEFAULT_DEADLINE = 300.0
DEVICE_MASK_ANE = 4
SEGMENTER_GRAPH = "graph"
MAX_FUNCTIONS = 64

P = C.c_void_p
PP = C.POINTER(C.c_void_p)
CHAR = C.c_char_p
CU64 = C.c_uint64
CINT = C.c_int

# --- authoritative ABI (from e5rt_api.h; each signature individually) -------
# value = (restype, argtypes). PP == "void**" out/obj, P == "void*", CHAR path,
# CU64 mask/count, CINT int. create_* are out-FIRST; retain is out-LAST.
ABI = {
    "e5rt_e5_compiler_config_options_create": (C.c_int64, (PP,)),
    "e5rt_e5_compiler_config_options_release": (C.c_int64, (PP,)),
    "e5rt_e5_compiler_config_options_set_cache_bundle_location": (C.c_int64, (P, CHAR)),
    "e5rt_e5_compiler_create_with_config": (C.c_int64, (PP, P)),
    "e5rt_e5_compiler_release": (C.c_int64, (PP,)),
    "e5rt_e5_compiler_options_create": (C.c_int64, (PP,)),
    "e5rt_e5_compiler_options_release": (C.c_int64, (PP,)),
    "e5rt_e5_compiler_options_set_compute_device_types_mask": (C.c_int64, (P, CU64)),
    "e5rt_e5_compiler_options_set_force_recompilation": (C.c_int64, (P, CINT)),
    "e5rt_e5_compiler_options_set_segmenter": (C.c_int64, (P, CHAR)),
    "e5rt_e5_compiler_compile": (C.c_int64, (P, CHAR, P, PP)),
    "e5rt_program_library_create": (C.c_int64, (PP, CHAR)),
    "e5rt_program_library_release": (C.c_int64, (PP,)),
    "e5rt_program_library_get_num_functions": (C.c_int64, (P, C.POINTER(CU64))),
    "e5rt_program_library_get_function_names": (C.c_int64, (P, CU64, C.POINTER(CHAR))),
    "e5rt_program_library_retain_program_function": (C.c_int64, (P, CHAR, PP)),
    "e5rt_program_function_release": (C.c_int64, (PP,)),
    "e5rt_precompiled_compute_op_create_options_create_with_program_function": (C.c_int64, (PP, P)),
    "e5rt_precompiled_compute_op_create_options_release": (C.c_int64, (PP,)),
    "e5rt_precompiled_compute_op_create_options_set_operation_name": (C.c_int64, (P, CHAR)),
    "e5rt_precompiled_compute_op_create_options_set_allocate_intermediate_buffers": (C.c_int64, (P, CINT)),
    "e5rt_execution_stream_operation_create_precompiled_compute_operation_with_options": (C.c_int64, (PP, P)),
    "e5rt_execution_stream_operation_release": (C.c_int64, (PP,)),
}

COMPILE_SYMBOLS = (
    "e5rt_e5_compiler_config_options_create",
    "e5rt_e5_compiler_config_options_release",
    "e5rt_e5_compiler_config_options_set_cache_bundle_location",
    "e5rt_e5_compiler_create_with_config",
    "e5rt_e5_compiler_release",
    "e5rt_e5_compiler_options_create",
    "e5rt_e5_compiler_options_release",
    "e5rt_e5_compiler_options_set_compute_device_types_mask",
    "e5rt_e5_compiler_options_set_force_recompilation",
    "e5rt_e5_compiler_options_set_segmenter",
    "e5rt_e5_compiler_compile",
    "e5rt_program_library_create",
    "e5rt_program_library_release",
    "e5rt_program_library_get_num_functions",
    "e5rt_program_library_get_function_names",
)

REOPEN_SYMBOLS = (
    "e5rt_program_library_create",
    "e5rt_program_library_release",
    "e5rt_program_library_get_num_functions",
    "e5rt_program_library_get_function_names",
)

PREPARE_SYMBOLS = (
    "e5rt_program_library_retain_program_function",
    "e5rt_program_function_release",
    "e5rt_precompiled_compute_op_create_options_create_with_program_function",
    "e5rt_precompiled_compute_op_create_options_release",
    "e5rt_precompiled_compute_op_create_options_set_operation_name",
    "e5rt_precompiled_compute_op_create_options_set_allocate_intermediate_buffers",
    "e5rt_execution_stream_operation_create_precompiled_compute_operation_with_options",
    "e5rt_execution_stream_operation_release",
)


class ProbeError(Exception):
    pass


def now_ms():
    return int(time.time() * 1000)


class Journal:
    """Stage journal flushed before/after every e5rt call for crash attribution."""

    def __init__(self, path, meta):
        self.path = path
        self.meta = meta
        self.entries = []
        self.status = "in_progress"
        self.error = None

    def _flush(self):
        doc = {"meta": self.meta, "status": self.status, "error": self.error,
               "stages": self.entries}
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=2, sort_keys=True)
        os.replace(tmp, self.path)

    def stage(self, name):
        self.entries.append({"stage": name, "before_ms": now_ms()})
        self._flush()

    def after(self, rc, obj=None):
        e = self.entries[-1]
        e["after_ms"] = now_ms()
        e["rc"] = rc
        if obj is not None:
            e["obj"] = obj
        self._flush()

    def fail(self, msg):
        self.status = "failed"
        self.error = msg
        self._flush()

    def ok(self):
        self.status = "completed"
        self._flush()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory_dir(root):
    """SHA-256 inventory of a directory tree (own NEWdir / explicit bundle).

    Only regular files are hashed; symlinks and other non-regular files are
    skipped so nothing outside the explicit root is ever read.
    """
    out = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for fn in sorted(filenames):
            p = os.path.join(dirpath, fn)
            try:
                if os.path.islink(p) or not os.path.isfile(p):
                    continue
                out.append({"path": os.path.relpath(p, root),
                            "bytes": os.path.getsize(p), "sha256": sha256_file(p)})
            except OSError as e:
                out.append({"path": os.path.relpath(p, root), "error": str(e)})
    return sorted(out, key=lambda d: d["path"])


def make_outdir(outdir):
    try:
        os.makedirs(outdir, exist_ok=False)
    except FileExistsError:
        raise ProbeError("output dir already exists (rejecting, not overwriting): %s" % outdir)


def load_functions(which, deadline, prepare=False):
    """dlsym-bound e5rt_* callables (exact argtypes). Returns (fns, error)."""
    try:
        dll = C.CDLL(FRAMEWORK)
    except Exception as e:  # noqa: BLE001
        return None, "dlopen failed: %s" % e
    base = COMPILE_SYMBOLS if which == "compile" else REOPEN_SYMBOLS
    required = list(base) + (list(PREPARE_SYMBOLS) if prepare else [])
    fns = {}
    missing = []
    for name in required:
        fn = getattr(dll, name, None)
        if fn is None:
            fn = getattr(dll, "_" + name, None)
        if fn is None:
            missing.append(name)
            continue
        restype, argtypes = ABI[name]
        fn.restype = restype
        fn.argtypes = argtypes
        fns[name] = fn
    if missing:
        return fns, "missing symbols: %s" % ", ".join(missing)
    return fns, None


def capability_map():
    """Report resolution for every symbol in ABI without calling anything."""
    dll = None
    dlopen_error = None
    try:
        dll = C.CDLL(FRAMEWORK)
    except Exception as e:  # noqa: BLE001
        dlopen_error = str(e)
    out = {"framework": FRAMEWORK, "dlopen": "ok" if dll else "failed",
           "dlopen_error": dlopen_error, "symbols": {}}
    if dll:
        for name in ABI:
            fn = getattr(dll, name, None)
            if fn is None:
                fn = getattr(dll, "_" + name, None)
            out["symbols"][name] = fn is not None
    else:
        for name in ABI:
            out["symbols"][name] = False
    return out


def check(rc, objp, what, journal, deadline):
    """Validate an e5rt return. ``objp`` is a c_void_p object (or None for
    rc-only setters). Uses ``objp.value``; a None ``objp`` skips the NULL
    out-check entirely."""
    if time.monotonic() > deadline:
        raise ProbeError("deadline exceeded before %s" % what)
    has_out = objp is not None
    obj = objp.value if has_out else None
    journal.after(int(rc), (obj is not None) if has_out else None)
    if rc != 0:
        raise ProbeError("%s returned rc=%d" % (what, rc))
    if has_out and obj is None:
        raise ProbeError("%s returned NULL out (rc=%d)" % (what, rc))
    return obj


def release(fns, journal, objp, sym):
    """Release a handle. ``objp`` is a c_void_p; release takes void**, so pass
    ``byref(objp)`` and clear ``objp.value`` afterwards."""
    if objp.value is None:
        return
    journal.stage("release_%s" % sym)
    rc = fns[sym](C.byref(objp))
    objp.value = None
    journal.after(int(rc), False)
    if rc != 0:
        raise ProbeError("release %s returned rc=%d" % (sym, rc))


def cleanup_all(fns, journal, created):
    """Attempt reverse cleanup of all handles; collect and raise on any error
    so no handle is skipped even if one release fails."""
    errors = []
    for sym, objp in reversed(created):
        try:
            release(fns, journal, objp, sym)
        except Exception as e:  # noqa: BLE001
            errors.append("%s: %s" % (sym, e))
    if errors:
        raise ProbeError("cleanup errors: %s" % "; ".join(errors))


def list_functions(fns, journal, library, deadline):
    n = C.c_uint64(0)
    journal.stage("get_num_functions")
    rc = fns["e5rt_program_library_get_num_functions"](library, C.byref(n))
    journal.after(int(rc))
    if rc != 0:
        raise ProbeError("get_num_functions returned rc=%d" % rc)
    if n.value > MAX_FUNCTIONS:
        raise ProbeError("function count %d exceeds cap %d" % (n.value, MAX_FUNCTIONS))
    names = (CHAR * int(n.value))()
    if n.value:
        journal.stage("get_function_names")
        rc = fns["e5rt_program_library_get_function_names"](library, n.value, names)
        journal.after(int(rc))
        if rc != 0:
            raise ProbeError("get_function_names returned rc=%d" % rc)
    out = []
    for i in range(int(n.value)):
        raw = names[i]
        if not raw:
            raise ProbeError("NULL function name at index %d" % i)
        out.append(raw.decode("utf-8", "replace"))
    return out


def prepare_from_library(fns, journal, library, deadline, created):
    """Retain function "main" and build a precompiled operation (no execute,
    no stream, no input binding). Appends handles to the caller's ``created``
    list as they are allocated so a partial failure is cleaned up by the
    caller's reverse cleanup (no local leak)."""
    func = C.c_void_p()
    journal.stage("retain_program_function")
    rc = fns["e5rt_program_library_retain_program_function"](library, b"main", C.byref(func))
    check(rc, func, "retain_program_function", journal, deadline)
    created.append(("e5rt_program_function_release", func))

    op_options = C.c_void_p()
    journal.stage("op_options_create")
    rc = fns["e5rt_precompiled_compute_op_create_options_create_with_program_function"](
        C.byref(op_options), func)
    check(rc, op_options, "op_options_create", journal, deadline)
    created.append(("e5rt_precompiled_compute_op_create_options_release", op_options))

    journal.stage("set_operation_name")
    rc = fns["e5rt_precompiled_compute_op_create_options_set_operation_name"](op_options, b"main")
    check(rc, None, "set_operation_name", journal, deadline)
    journal.stage("set_allocate_intermediate_buffers")
    rc = fns["e5rt_precompiled_compute_op_create_options_set_allocate_intermediate_buffers"](
        op_options, 1)
    check(rc, None, "set_allocate_intermediate_buffers", journal, deadline)

    operation = C.c_void_p()
    journal.stage("op_create")
    rc = fns["e5rt_execution_stream_operation_create_precompiled_compute_operation_with_options"](
        C.byref(operation), op_options)
    check(rc, operation, "op_create", journal, deadline)
    created.append(("e5rt_execution_stream_operation_release", operation))
    return created


def snapshot_bundle(outdir, cache, result, done):
    """Copy the compiled cache to ``outdir/bundle_snapshot`` (symlinks copied,
    not followed) and inventory it + the live cache, while library/operation
    handles are still alive. Runs once (success path, or finally on failure)."""
    if done[0]:
        return
    done[0] = True
    snap = os.path.join(outdir, "bundle_snapshot")
    try:
        shutil.copytree(cache, snap, symlinks=True)
    except FileExistsError:
        pass
    result["cache_inventory_alive"] = inventory_dir(cache)
    result["bundle_snapshot_inventory"] = inventory_dir(snap)


def run_compile(fns, journal, mil_path, outdir, deadline, prepare):
    cache = os.path.join(outdir, "cache")
    os.makedirs(cache, exist_ok=True)
    created = []
    library = C.c_void_p()
    result = {}
    snapshot_done = [False]
    try:
        journal.stage("config_create")
        config = C.c_void_p()
        rc = fns["e5rt_e5_compiler_config_options_create"](C.byref(config))
        check(rc, config, "config_create", journal, deadline)
        created.append(("e5rt_e5_compiler_config_options_release", config))

        journal.stage("set_cache_bundle_location")
        rc = fns["e5rt_e5_compiler_config_options_set_cache_bundle_location"](
            config, cache.encode())
        check(rc, None, "set_cache_bundle_location", journal, deadline)

        journal.stage("compiler_create_with_config")
        compiler = C.c_void_p()
        rc = fns["e5rt_e5_compiler_create_with_config"](C.byref(compiler), config)
        check(rc, compiler, "compiler_create_with_config", journal, deadline)
        created.append(("e5rt_e5_compiler_release", compiler))

        journal.stage("options_create")
        options = C.c_void_p()
        rc = fns["e5rt_e5_compiler_options_create"](C.byref(options))
        check(rc, options, "options_create", journal, deadline)
        created.append(("e5rt_e5_compiler_options_release", options))

        journal.stage("set_device_mask")
        rc = fns["e5rt_e5_compiler_options_set_compute_device_types_mask"](
            options, DEVICE_MASK_ANE)
        check(rc, None, "set_device_mask", journal, deadline)
        journal.stage("set_force_recompilation")
        rc = fns["e5rt_e5_compiler_options_set_force_recompilation"](options, 1)
        check(rc, None, "set_force_recompilation", journal, deadline)
        journal.stage("set_segmenter")
        rc = fns["e5rt_e5_compiler_options_set_segmenter"](options, SEGMENTER_GRAPH.encode())
        check(rc, None, "set_segmenter", journal, deadline)

        journal.stage("compile")
        rc = fns["e5rt_e5_compiler_compile"](compiler, mil_path.encode(), options, C.byref(library))
        check(rc, library, "compile", journal, deadline)
        created.append(("e5rt_program_library_release", library))

        func_names = list_functions(fns, journal, library, deadline)
        result = {"status": "in_progress", "function_names": func_names,
                  "device_mask": DEVICE_MASK_ANE, "segmenter": SEGMENTER_GRAPH,
                  "cache_bundle_location": cache}

        if prepare:
            prepare_from_library(fns, journal, library, deadline, created)
            result["prepare"] = {"status": "created", "operation": "main"}
        snapshot_bundle(outdir, cache, result, snapshot_done)  # while alive
    finally:
        try:
            snapshot_bundle(outdir, cache, result, snapshot_done)  # failure path
        finally:
            cleanup_all(fns, journal, created)
    result["status"] = "completed"
    journal.ok()
    return result


def run_reopen(fns, journal, bundle_path, deadline, prepare):
    created = []
    library = C.c_void_p()
    result = {}
    try:
        journal.stage("program_library_create")
        rc = fns["e5rt_program_library_create"](C.byref(library), bundle_path.encode())
        check(rc, library, "program_library_create", journal, deadline)
        created.append(("e5rt_program_library_release", library))

        func_names = list_functions(fns, journal, library, deadline)
        result = {"status": "in_progress", "function_names": func_names}

        if prepare:
            prepare_from_library(fns, journal, library, deadline, created)
            result["prepare"] = {"status": "created", "operation": "main"}
    finally:
        cleanup_all(fns, journal, created)
    result["status"] = "completed"
    journal.ok()
    return result


def write_final(outdir, doc):
    with open(os.path.join(outdir, "final.json"), "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ane_e5_probe", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("capabilities", help="report which e5rt symbols resolve")
    pc.add_argument("--output", required=True)

    pco = sub.add_parser("compile", help="compile MIL into ANE E5 bundle + list functions")
    pco.add_argument("--mil", required=True)
    pco.add_argument("--output", required=True)
    pco.add_argument("--prepare", action="store_true",
                     help="also retain main + create precompiled op (hardware preparation, no execute)")

    pr = sub.add_parser("reopen", help="open a compiled bundle + list functions (no compiler)")
    pr.add_argument("--bundle", required=True)
    pr.add_argument("--output", required=True)
    pr.add_argument("--prepare", action="store_true")

    for p in (pc, pco, pr):
        p.add_argument("--deadline", type=float, default=DEFAULT_DEADLINE)

    args = ap.parse_args(argv)
    outdir = os.path.abspath(args.output)
    deadline = time.monotonic() + args.deadline
    meta = {"subcommand": args.cmd, "argv": list(sys.argv[1:])}

    try:
        make_outdir(outdir)
    except ProbeError as e:
        print("error: %s" % e, file=sys.stderr)
        return 2

    if args.cmd == "capabilities":
        caps = capability_map()
        write_final(outdir, caps)
        for name in sorted(caps["symbols"]):
            print("%s %s" % ("present" if caps["symbols"][name] else "missing", name))
        return 0

    prepare = bool(getattr(args, "prepare", False))
    journal = Journal(os.path.join(outdir, "journal.json"), meta)
    fns, err = load_functions(args.cmd, deadline, prepare=prepare)
    if err:
        journal.fail(err)
        print("error: %s" % err, file=sys.stderr)
        return 2

    try:
        if args.cmd == "compile":
            mil_path = os.path.abspath(args.mil)
            if not os.path.isfile(mil_path):
                raise ProbeError("mil path not found: %s" % mil_path)
            warnings = []
            weights = os.path.join(os.path.dirname(mil_path), "weights", "weight.bin")
            if not os.path.isfile(weights):
                warnings.append("weights not staged at %s; root must stage "
                                "weight.bin at weights/weight.bin beside the mil" % weights)
            result = run_compile(fns, journal, mil_path, outdir, deadline, prepare)
            result["warnings"] = warnings
            result["bundle_dir"] = os.path.join(outdir, "cache")
            result["inventory_newdir"] = inventory_dir(outdir)
        else:  # reopen
            bundle_path = os.path.abspath(args.bundle)
            if not os.path.isdir(bundle_path):
                raise ProbeError("bundle path not found: %s" % bundle_path)
            result = run_reopen(fns, journal, bundle_path, deadline, prepare)
            result["bundle_path"] = bundle_path
            result["inventory_newdir"] = inventory_dir(outdir)
            result["inventory_bundle"] = inventory_dir(bundle_path)
        result["journal"] = journal.path
        result["preserved_failed_artifacts"] = True
        write_final(outdir, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except ProbeError as e:
        journal.fail(str(e))
        _write_failure(outdir, journal, str(e))
        print("error: %s" % e, file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 - preserve artifacts on any failure
        journal.fail("unexpected: %r" % e)
        _write_failure(outdir, journal, "unexpected: %r" % e)
        print("error: unexpected %r" % e, file=sys.stderr)
        return 1


def _write_failure(outdir, journal, error):
    try:
        write_final(outdir, {"status": "failed", "error": error,
                             "journal": journal.path,
                             "inventory_newdir": inventory_dir(outdir),
                             "preserved_failed_artifacts": True})
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    sys.exit(main())
