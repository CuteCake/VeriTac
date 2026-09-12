#!/usr/bin/env python3
"""execute_control.py - decisive same-process vs fresh-bundle E5 execution control.

Runs a FIXED identity+ReLU workload of shape [1, 512, 1, 64]: input 65536 fp16
bytes (32768 elements), expected output ``np.maximum(input, 0)``, checked with
atol = rtol = 0 and a full finiteness check. The graph is executed EXACTLY ONCE
via the e5rt buffer / port / stream path (no timing loops, no
``load_for_execution``).

  --mode compile  compile --path <model.mil> then prepare and execute in the
                  SAME process before any cleanup (decisive same-process run).
  --mode reopen   open --path <bundle-dir> fresh (library create + prepare +
                  execute; absolutely no compiler-API calls), independent
                  fresh-bundle run.

ABI for compile/prepare is reused from ane_e5_probe; the additional buffer/
port/stream symbols are bound with EXACT signatures from
reference/e5rt_api.h. Reuses ane_e5_probe.Journal / check / release /
cleanup_all / prepare_from_library / snapshot_bundle / list_functions.

Scope note: TEST-ONLY against the fixed workload above; not a general ANE
driver. No real hardware runs are made by the worker here - only py_compile /
mock validation. Root runs compile once, then reopen once, under a parent
timeout. Every e5rt call is journaled before/after, checked for rc + non-NULL
out, and all handles are released in reverse on every exit path.
"""

import argparse
import ctypes as C
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import ane_e5_probe as probe  # noqa: E402

P, PP, CHAR, CU64 = probe.P, probe.PP, probe.CHAR, probe.CU64
U32, I64 = C.c_uint32, C.c_int64

INPUT_BYTES = 65536
INPUT_PORT, OUTPUT_PORT = "x", "out"
BUFFER_TYPE_CPU = 0

EXEC_ABI = {  # exact from reference/e5rt_api.h
    "e5rt_buffer_object_alloc": (I64, (PP, CU64, U32)),
    "e5rt_buffer_object_release": (I64, (PP,)),
    "e5rt_buffer_object_get_data_ptr": (I64, (P, PP)),
    "e5rt_buffer_object_get_size": (I64, (P, C.POINTER(CU64))),
    "e5rt_execution_stream_operation_retain_input_port": (I64, (P, CHAR, PP)),
    "e5rt_execution_stream_operation_retain_output_port": (I64, (P, CHAR, PP)),
    "e5rt_io_port_release": (I64, (PP,)),
    "e5rt_io_port_bind_buffer_object": (I64, (P, P)),
    "e5rt_execution_stream_create": (I64, (PP,)),
    "e5rt_execution_stream_release": (I64, (PP,)),
    "e5rt_execution_stream_operation_prepare_op_for_encode": (I64, (P,)),
    "e5rt_execution_stream_encode_operation": (I64, (P, P)),
    "e5rt_execution_stream_execute_sync": (I64, (P,)),
}


def _abi_for(name):
    return probe.ABI.get(name) or EXEC_ABI[name]


def load(mode):
    dll = C.CDLL(probe.FRAMEWORK)
    base = probe.COMPILE_SYMBOLS if mode == "compile" else probe.REOPEN_SYMBOLS
    required = list(base) + list(probe.PREPARE_SYMBOLS) + list(EXEC_ABI)
    fns, missing = {}, []
    for name in required:
        fn = getattr(dll, name, None) or getattr(dll, "_" + name, None)
        if fn is None:
            missing.append(name)
            continue
        restype, argtypes = _abi_for(name)
        fn.restype, fn.argtypes = restype, argtypes
        fns[name] = fn
    if missing:
        raise probe.ProbeError("missing symbols: %s" % ", ".join(missing))
    return fns


def _call(fns, journal, deadline, sym, out, *args):
    journal.stage(sym)
    return probe.check(fns[sym](*args), out, sym, journal, deadline)


def _build_compile(fns, journal, mil_path, cache, deadline, created):
    library = C.c_void_p()
    config, compiler, options = P(), P(), P()
    _call(fns, journal, deadline, "e5rt_e5_compiler_config_options_create",
          config, C.byref(config))
    created.append(("e5rt_e5_compiler_config_options_release", config))
    _call(fns, journal, deadline, "e5rt_e5_compiler_config_options_set_cache_bundle_location",
          None, config, cache.encode())
    _call(fns, journal, deadline, "e5rt_e5_compiler_create_with_config",
          compiler, C.byref(compiler), config)
    created.append(("e5rt_e5_compiler_release", compiler))
    _call(fns, journal, deadline, "e5rt_e5_compiler_options_create",
          options, C.byref(options))
    created.append(("e5rt_e5_compiler_options_release", options))
    _call(fns, journal, deadline, "e5rt_e5_compiler_options_set_compute_device_types_mask",
          None, options, probe.DEVICE_MASK_ANE)
    _call(fns, journal, deadline, "e5rt_e5_compiler_options_set_force_recompilation",
          None, options, 1)
    _call(fns, journal, deadline, "e5rt_e5_compiler_options_set_segmenter",
          None, options, probe.SEGMENTER_GRAPH.encode())
    _call(fns, journal, deadline, "e5rt_e5_compiler_compile",
          library, compiler, mil_path.encode(), options, C.byref(library))
    created.append(("e5rt_program_library_release", library))
    probe.prepare_from_library(fns, journal, library, deadline, created)
    return library


def _build_reopen(fns, journal, bundle_path, deadline, created):
    library = C.c_void_p()
    _call(fns, journal, deadline, "e5rt_program_library_create",
          library, C.byref(library), bundle_path.encode())
    created.append(("e5rt_program_library_release", library))
    probe.prepare_from_library(fns, journal, library, deadline, created)
    return library


def _check_size(fns, journal, buffer, deadline):
    size = CU64()
    _call(fns, journal, deadline, "e5rt_buffer_object_get_size", None,
          buffer, C.byref(size))
    if size.value < INPUT_BYTES:
        raise probe.ProbeError("buffer allocation is smaller than fixed tensor")


def _execute(fns, journal, created, in_data, deadline):
    operation = created[-1][1].value
    if not operation:
        raise probe.ProbeError("no prepared operation")
    in_buffer, out_buffer = C.c_void_p(), C.c_void_p()
    in_port, out_port, stream = C.c_void_p(), C.c_void_p(), C.c_void_p()

    _call(fns, journal, deadline, "e5rt_buffer_object_alloc", in_buffer,
          C.byref(in_buffer), INPUT_BYTES, BUFFER_TYPE_CPU)
    created.append(("e5rt_buffer_object_release", in_buffer))
    in_ptr = C.c_void_p()
    _call(fns, journal, deadline, "e5rt_buffer_object_get_data_ptr", in_ptr,
          in_buffer, C.byref(in_ptr))
    _check_size(fns, journal, in_buffer, deadline)
    C.memmove(in_ptr, in_data, INPUT_BYTES)

    _call(fns, journal, deadline, "e5rt_execution_stream_operation_retain_input_port",
          in_port, operation, INPUT_PORT.encode(), C.byref(in_port))
    created.append(("e5rt_io_port_release", in_port))
    _call(fns, journal, deadline, "e5rt_io_port_bind_buffer_object", None,
          in_port, in_buffer)

    _call(fns, journal, deadline, "e5rt_buffer_object_alloc", out_buffer,
          C.byref(out_buffer), INPUT_BYTES, BUFFER_TYPE_CPU)
    created.append(("e5rt_buffer_object_release", out_buffer))
    out_ptr = C.c_void_p()
    _call(fns, journal, deadline, "e5rt_buffer_object_get_data_ptr", out_ptr,
          out_buffer, C.byref(out_ptr))
    _check_size(fns, journal, out_buffer, deadline)
    C.memset(out_ptr, 0xFF, INPUT_BYTES)  # NaN sentinel for unwritten output.

    _call(fns, journal, deadline, "e5rt_execution_stream_operation_retain_output_port",
          out_port, operation, OUTPUT_PORT.encode(), C.byref(out_port))
    created.append(("e5rt_io_port_release", out_port))
    _call(fns, journal, deadline, "e5rt_io_port_bind_buffer_object", None,
          out_port, out_buffer)

    _call(fns, journal, deadline, "e5rt_execution_stream_create", stream,
          C.byref(stream))
    created.append(("e5rt_execution_stream_release", stream))

    # A fresh operation needs no reset before its first encode.
    _call(fns, journal, deadline, "e5rt_execution_stream_encode_operation", None,
          stream, operation)
    _call(fns, journal, deadline, "e5rt_execution_stream_execute_sync", None, stream)

    return C.string_at(out_ptr, INPUT_BYTES)


def _expected_bytes(in_data):
    return np.maximum(
        np.frombuffer(in_data, dtype=np.float16).astype(np.float32), 0
    ).astype(np.float16).tobytes()


def control(mode, fns, journal, path, outdir, in_data, deadline):
    created = []
    result = {}
    try:
        if mode == "compile":
            cache = os.path.join(outdir, "cache")
            os.makedirs(cache, exist_ok=True)
            library = _build_compile(fns, journal, path, cache, deadline, created)
        else:
            library = _build_reopen(fns, journal, path, deadline, created)
        result["function_names"] = probe.list_functions(fns, journal, library, deadline)

        out_bytes = _execute(fns, journal, created, in_data, deadline)
        exp_bytes = _expected_bytes(in_data)
        result["output_sha256"] = hashlib.sha256(out_bytes).hexdigest()
        result["output_bytes"] = len(out_bytes)
        result["finite"] = bool(np.isfinite(np.frombuffer(out_bytes, dtype=np.float16)).all())
        result["exact_equal"] = (out_bytes == exp_bytes)
        with open(os.path.join(outdir, "output.fp16.bin"), "wb") as f:
            f.write(out_bytes)
        result["status"] = "in_progress"
    finally:
        try:
            if mode == "compile":
                probe.snapshot_bundle(outdir, cache, result, [False])
        finally:
            probe.cleanup_all(fns, journal, created)
    result["status"] = "completed"
    journal.ok()
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(prog="execute_control", description=__doc__)
    ap.add_argument("--mode", required=True, choices=("compile", "reopen"))
    ap.add_argument("--path", required=True, help="MIL (compile) or bundle dir (reopen)")
    ap.add_argument("--input", required=True, help="65536-byte fp16 input file")
    ap.add_argument("--output", required=True, help="new output dir (must not exist)")
    ap.add_argument("--deadline", type=float, default=probe.DEFAULT_DEADLINE)
    args = ap.parse_args(argv)

    outdir = os.path.abspath(args.output)
    try:
        probe.make_outdir(outdir)
    except probe.ProbeError as e:
        print("error: %s" % e, file=sys.stderr)
        return 2

    with open(args.input, "rb") as f:
        in_data = f.read()
    if len(in_data) != INPUT_BYTES:
        print("error: input must be exactly %d bytes (got %d)" % (INPUT_BYTES, len(in_data)),
              file=sys.stderr)
        return 2

    deadline = time.monotonic() + args.deadline
    journal = probe.Journal(os.path.join(outdir, "journal.json"),
                            {"subcommand": "execute_control", "mode": args.mode})
    try:
        fns = load(args.mode)
    except probe.ProbeError as e:
        journal.fail(str(e))
        print("error: %s" % e, file=sys.stderr)
        return 2

    try:
        result = control(args.mode, fns, journal, os.path.abspath(args.path),
                         outdir, in_data, deadline)
        result.update(mode=args.mode,
                      input_sha256=hashlib.sha256(in_data).hexdigest(),
                      expected_output_sha256=hashlib.sha256(_expected_bytes(in_data)).hexdigest(),
                      journal=journal.path)
        with open(os.path.join(outdir, "final.json"), "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["finite"] and result["exact_equal"] else 1
    except probe.ProbeError as e:
        journal.fail(str(e))
        probe._write_failure(outdir, journal, str(e))
        print("error: %s" % e, file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        journal.fail("unexpected: %r" % e)
        probe._write_failure(outdir, journal, "unexpected: %r" % e)
        print("error: unexpected %r" % e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
