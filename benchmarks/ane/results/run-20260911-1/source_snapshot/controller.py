#!/usr/bin/env python3
"""Bounded Apple Neural Engine feasibility probe — controller / harness.

Owns benchmarks/ane only. Orchestrates the native `ane_probe` binary:

  * generates deterministic canonical fp16 input bytes
  * independently builds the MIL text and weight blob (to cross-check the
    probe's generation byte-for-byte)
  * computes an independent float64 reference oracle for each weight pattern
  * launches the probe with an ISOLATED absolute TMPDIR (task-owned
    run/framework_tmp) so the ANE compiler's per-model temp dir stays scoped
  * merges the probe's structured results with controller-side hashes, the
    numeric oracle, and verification; records raw logs and artifacts
  * never overwrites prior run evidence (each case gets its own subdir)

Usage (from project root):
  PYTHONPATH=. python3 benchmarks/ane/controller/ane_probe.py run \
      --results-dir results/ane/20260911-run1 \
      --cases identity,two-tap --channels 512 --spatial 64 --seed 1234

Tolerances are frozen and recorded for FP16 (atol=2e-3, rtol=2e-2).
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import datetime
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SRC_DIR = HERE.parent / "src"
BIN = HERE.parent / "ane_probe"

# Frozen comparison tolerance. Inputs are dyadic (multiples of 1/64, exactly
# representable in fp16) and both weight patterns (identity: 1.0*x;
# two-tap: 0.5*x - 0.25*x) yield results exactly representable in fp16, so
# the comparison is EXACT (atol=rtol=0). A nonzero error is a real mismatch.
ATOL = 0.0
RTOL = 0.0


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def make_input(C: int, S: int, seed: int) -> bytes:
    """Deterministic dyadic mixed-sign fp16 input (multiples of 1/64),
    little-endian bytes. Values are exactly representable in fp16."""
    rng = np.random.default_rng(seed)
    x = rng.integers(-64, 65, C * S).astype(np.float64) / 64.0
    return x.astype("<f2").tobytes()


def build_mil(C: int, S: int) -> str:
    return (
        "program(1.3)\n"
        '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
        '{"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, '
        '{"coremltools-version", "9.0"}})]\n'
        "{\n"
        f"    func main<ios18>(tensor<fp16, [1, {C}, 1, {S}]> x) {{\n"
        '        string pt = const()[name = string("pt"), val = string("valid")];\n'
        '        tensor<int32, [2]> st = const()[name = string("st"), val = tensor<int32, [2]>([1, 1])];\n'
        '        tensor<int32, [4]> pd = const()[name = string("pd"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n'
        '        tensor<int32, [2]> dl = const()[name = string("dl"), val = tensor<int32, [2]>([1, 1])];\n'
        '        int32 gr = const()[name = string("gr"), val = int32(1)];\n'
        f"        tensor<fp16, [{C}, {C}, 1, 1]> W = const()[name = string(\"W\"), "
        f"val = tensor<fp16, [{C}, {C}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        f"        tensor<fp16, [1, {C}, 1, {S}]> c = conv(dilations = dl, groups = gr, pad = pd, "
        f"pad_type = pt, strides = st, weight = W, x = x)[name = string(\"conv\")];\n"
        f"        tensor<fp16, [1, {C}, 1, {S}]> out = relu(x = c)[name = string(\"relu\")];\n"
        "    } -> (out);\n"
        "}\n"
    )


def build_weight_blob(C: int, kind: str) -> bytes:
    """Independent weight blob: 128-byte header + row-major fp16 [out,in]."""
    diag = 1.0 if kind == "identity" else 0.5
    offdiag = -0.25 if kind == "two-tap" else 0.0
    w = np.zeros((C, C), dtype="<f2")
    for o in range(C):
        w[o, o] = np.float16(diag)
        if kind == "two-tap":
            w[o, (o + 1) % C] = np.float16(offdiag)
    blob = bytearray(128)
    blob[0] = 0x01
    blob[4] = 0x02
    blob[64:68] = bytes([0xEF, 0xBE, 0xAD, 0xDE])
    blob[68] = 0x01
    wsize = C * C * 2
    blob[72:76] = int(wsize).to_bytes(4, "little")
    blob[80:84] = (128).to_bytes(4, "little")
    blob += w.tobytes()
    return bytes(blob)


def compute_reference(x_f16: np.ndarray, C: int, S: int, kind: str) -> np.ndarray:
    """x_f16: [C, S] fp16; returns float64 [C, S] = relu(W x)."""
    xs = x_f16.astype(np.float64)  # [C, S]
    diag = 1.0 if kind == "identity" else 0.5
    offdiag = -0.25 if kind == "two-tap" else 0.0
    y = diag * xs
    if kind == "two-tap":
        # out[o,s] = 0.5*x[o,s] - 0.25*x[(o+1)%C, s]
        y = y + offdiag * np.roll(xs, -1, axis=0)
    return np.maximum(y, 0.0)


def verify(out_bytes: bytes, ref: np.ndarray, C: int, S: int):
    out = np.frombuffer(out_bytes, dtype="<f2").astype(np.float64).reshape(C, S)
    ref = ref.reshape(C, S)
    finite_out = bool(np.all(np.isfinite(out)))
    finite_ref = bool(np.all(np.isfinite(ref)))
    shape_ok = out.shape == ref.shape
    diff = np.abs(out - ref)
    within = np.abs(out - ref) <= ATOL + RTOL * np.abs(ref)
    return {
        "shape_ok": shape_ok,
        "output_shape": list(out.shape),
        "reference_shape": list(ref.shape),
        "finite_out": finite_out,
        "finite_ref": finite_ref,
        "max_abs_error": float(diff.max()) if diff.size else None,
        "rms_error": float(np.sqrt(np.mean(diff ** 2))) if diff.size else None,
        "fraction_within_tolerance": float(within.mean()) if within.size else None,
        "within_tolerance": bool(shape_ok and finite_out and finite_ref and within.all()),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
    }


def record_source_hashes(results_dir: pathlib.Path) -> dict:
    h = {}
    for name, p in [
        ("ane_probe.m", SRC_DIR / "ane_probe.m"),
        ("ane_bridge.m", SRC_DIR / "ane_bridge.m"),
        ("ane_bridge.h", SRC_DIR / "ane_bridge.h"),
        ("Makefile", HERE.parent / "Makefile"),
        ("controller.py", __file__),
    ]:
        h[name] = sha256_file(p)
    if BIN.exists():
        h["ane_probe_binary"] = sha256_file(BIN)
    return h


def run_case(case_dir: pathlib.Path, kind: str, C: int, S: int, seed: int,
             samples: int, warmup: int, budget: int, timeout: float):
    if case_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing evidence: {case_dir}")
    case_dir.mkdir(parents=True)
    in_bytes = make_input(C, S, seed)
    input_path = case_dir / "input.fp16.bin"
    input_path.write_bytes(in_bytes)

    # Independent MIL + blob for byte-for-byte cross-check.
    mil = build_mil(C, S)
    blob = build_weight_blob(C, kind)
    (case_dir / "model.mil.controller").write_text(mil)
    (case_dir / "weight.bin.controller").write_bytes(blob)

    # Isolated absolute TMPDIR for the ANE compiler's per-model temp dir.
    fw_tmp = (case_dir / "framework_tmp").resolve()
    fw_tmp.mkdir(parents=True)

    # Canonical float64 reference oracle; save its bytes.
    x = np.frombuffer(in_bytes, dtype="<f2").reshape(C, S)
    ref = compute_reference(x, C, S, kind)
    (case_dir / "reference.f64.bin").write_bytes(ref.astype("<f8").tobytes())

    env = dict(os.environ)
    env["TMPDIR"] = str(fw_tmp)

    results_json = case_dir / "results.json"
    stdout_path = case_dir / "probe.stdout.log"
    stderr_path = case_dir / "probe.stderr.log"

    cmd = [
        str(BIN), "--mode", "compile-and-run",
        "--channels", str(C), "--spatial", str(S),
        "--weight", kind, "--seed", str(seed),
        "--samples", str(samples), "--warmup", str(warmup),
        "--compile-budget", str(budget),
        "--run-dir", str(case_dir),
        "--input", str(input_path),
        "--out", str(results_json),
    ]
    start = time.time()
    rc = None
    stdout = ""
    stderr = ""
    timed_out = False
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=timeout, cwd=str(HERE.parent),
        )
        rc = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as ex:
        timed_out = True
        rc = -1
        out = ex.stdout or ""
        err = ex.stderr or ""
        # decode in case bytes were returned despite text=True
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
        stdout = out
        stderr = err + f"\n[controller] TIMEOUT after {timeout}s\n"
    wall_ms = (time.time() - start) * 1000.0

    stdout_path.write_text(stdout)
    stderr_path.write_text(stderr)

    merged = {
        "controller": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "case": kind,
            "channels": C, "spatial": S, "seed": seed,
            "samples": samples, "warmup": warmup, "compile_budget": budget,
            "probe_rc": rc,
            "timed_out": timed_out,
            "wall_ms": wall_ms,
            "timeout_s": timeout,
            "probe_cmd": " ".join(cmd),
            "isolated_tmpdir": str(fw_tmp),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "inputs_sha256": {
                "input_fp16_bin": sha256_bytes(in_bytes),
            },
            "controller_mil_sha256": sha256_bytes(mil.encode()),
            "controller_weight_blob_sha256": sha256_bytes(blob),
            "reference_sha256": sha256_bytes(ref.astype("<f8").tobytes()),
            "reference_f64_path": str(case_dir / "reference.f64.bin"),
            "source_hashes": record_source_hashes(case_dir),
        },
        "verification": None,
        "probe": None,
    }

    # Parse the probe's structured results EVEN on nonzero rc (to preserve
    # failure-stage details).
    if results_json.exists():
        try:
            probe = json.loads(results_json.read_text())
            merged["probe"] = probe
            pmil = probe.get("hashes", {}).get("mil_sha256")
            pblob = probe.get("hashes", {}).get("weight_blob_sha256")
            merged["controller"]["mil_matches_probe"] = (
                pmil == merged["controller"]["controller_mil_sha256"])
            merged["controller"]["weight_blob_matches_probe"] = (
                pblob == merged["controller"]["controller_weight_blob_sha256"])
            merged["exact_mil_text"] = probe.get("workload", {}).get("mil_text")
        except Exception as ex:  # noqa: BLE001
            merged["controller"]["probe_parse_error"] = str(ex)

    # Numeric verification only when dispatch produced a full output.
    out_path = case_dir / "output.fp16.bin"
    if out_path.exists() and out_path.stat().st_size == C * S * 2:
        out_bytes = out_path.read_bytes()
        merged["controller"]["output_sha256"] = sha256_bytes(out_bytes)
        merged["verification"] = verify(out_bytes, ref, C, S)

    (case_dir / "merged.json").write_text(json.dumps(merged, indent=2))
    return merged


def main():
    ap = argparse.ArgumentParser(description="ANE bounded feasibility probe controller")
    sub = ap.add_subparsers(dest="command", required=True)

    runp = sub.add_parser("run", help="run compile-and-run cases")
    runp.add_argument("--results-dir", required=True)
    runp.add_argument("--cases", default="identity,two-tap")
    runp.add_argument("--channels", type=int, default=512)
    runp.add_argument("--spatial", type=int, default=64)
    runp.add_argument("--seed", type=int, default=1234)
    runp.add_argument("--samples", type=int, default=3)
    runp.add_argument("--warmup", type=int, default=1)
    runp.add_argument("--compile-budget", type=int, default=8)
    runp.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    # Strict validation.
    if args.channels not in (256, 512, 1024):
        ap.error("--channels must be 256|512|1024")
    if args.spatial not in (16, 64):
        ap.error("--spatial must be 16|64")
    if not 1 <= args.samples <= 100:
        ap.error("--samples must be in [1,100]")
    if not 0 <= args.warmup <= 10:
        ap.error("--warmup must be in [0,10]")
    if not 1 <= args.compile_budget <= 8:
        ap.error("--compile-budget must be in [1,8]")
    if args.timeout <= 0:
        ap.error("--timeout must be > 0")
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    if not cases:
        ap.error("--cases must be non-empty")
    if len(set(cases)) != len(cases):
        ap.error("--cases must be unique")
    if len(cases) > args.compile_budget:
        ap.error("--cases count exceeds --compile-budget")
    for c in cases:
        if c not in ("identity", "two-tap"):
            ap.error(f"invalid case '{c}' (identity|two-tap)")

    results_dir = pathlib.Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    if not BIN.exists():
        ap.error(f"probe binary not found: {BIN} (run 'make' in benchmarks/ane)")

    # Failfast: stop on the first failed case, no retries.
    all_merged = []
    failed = None
    for kind in cases:
        subdir = results_dir / f"case-{kind}-c{args.channels}s{args.spatial}"
        try:
            m = run_case(subdir, kind, args.channels, args.spatial, args.seed,
                         args.samples, args.warmup, args.compile_budget, args.timeout)
        except FileExistsError as ex:
            failed = {"case": kind, "error": str(ex)}
            print(f"[controller] FAIL (evidence exists): {kind}", file=sys.stderr)
            break
        all_merged.append(m)

        v = m.get("verification")
        ok = (m["controller"]["probe_rc"] == 0
              and m["controller"].get("mil_matches_probe") is True
              and m["controller"].get("weight_blob_matches_probe") is True
              and v is not None
              and v["within_tolerance"] is True)
        print(f"[controller] {kind}: probe_rc={m['controller']['probe_rc']} "
              f"mil_match={m['controller'].get('mil_matches_probe')} "
              f"blob_match={m['controller'].get('weight_blob_matches_probe')} "
              f"within_tolerance={v and v['within_tolerance']}")
        if not ok:
            failed = {"case": kind, "probe_rc": m["controller"]["probe_rc"],
                      "verification": v}
            print(f"[controller] FAIL: {kind}", file=sys.stderr)
            break

    (results_dir / "index.json").write_text(json.dumps(
        {"results_dir": str(results_dir),
         "cases": cases,
         "failed": failed,
         "merged_files": [str(r) for r in all_merged]}, indent=2))

    if failed:
        print(f"[controller] stopped after failure: {failed}", file=sys.stderr)
        sys.exit(1)
    print(f"[controller] done -> {results_dir}")
    sys.exit(0)


if __name__ == "__main__":
    main()
