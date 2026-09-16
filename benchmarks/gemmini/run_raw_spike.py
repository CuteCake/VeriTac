"""Execute the EXACT certified raw Gemmini kernel bytes under upstream Spike.

Unlike run_spike.py (which compiles generated C sources), this harness embeds
the certified .bin verbatim into an executable ELF section (.incbin) and jumps
to it. The body is never regenerated or recompiled; A/B/C buffers are placed at
the bases declared in request.json via dedicated linker sections.

Run on the Linux host prepared by setup_spike.sh (Spark3). Spike cycle counts
are NOT hardware latency; only dynamic instruction/DMA counts are reported.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

SCHEMA = "veritac_program_request_v1"
ENCODING_FORMAT = "veritac_gemmini_bytes_v2"
MODES = ("zeros", "extrema", "lcg")
PASS_MARKER = "VERITAC_GEMMINI_RAW_PASS"


class CaseError(Exception):
    """Fail-closed validation error."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_request_dict(request: dict) -> dict:
    if request.get("schema") != SCHEMA:
        raise CaseError(f"unexpected request schema {request.get('schema')!r}")
    encoding = request.get("encoding")
    if not isinstance(encoding, dict) or encoding.get("format") != ENCODING_FORMAT:
        raise CaseError("missing veritac_gemmini_bytes_v2 encoding block")
    if encoding.get("word_size") != 4 or encoding.get("endianness") != "little":
        raise CaseError("unsupported word encoding")
    hex_text = encoding.get("bytes_hex")
    if not isinstance(hex_text, str) or not hex_text or len(hex_text) % 8 != 0:
        raise CaseError("bytes_hex must be a multiple of 32-bit words")
    try:
        bytes.fromhex(hex_text)
    except ValueError as error:
        raise CaseError("bytes_hex is not valid hex") from error
    for section in ("plan", "commands"):
        if not isinstance(request.get(section), (dict, list)):
            raise CaseError(f"missing {section}")
    return request


def load_request(path: Path) -> dict:
    try:
        request = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CaseError(f"cannot read request: {error}") from error
    return validate_request_dict(request)


def encoding_bytes(request: dict) -> bytes:
    return bytes.fromhex(request["encoding"]["bytes_hex"])


def declared_layout(request: dict) -> tuple[dict, dict, dict]:
    """Return (bases, sizes, plan) after fail-closed consistency checks."""
    encoding = request["encoding"]
    bases = encoding.get("bases")
    sizes = encoding.get("sizes")
    plan = request.get("plan")
    if not isinstance(bases, dict) or set(bases) != {"A", "B", "C"}:
        raise CaseError("bases must declare exactly A, B, C")
    if not isinstance(sizes, dict) or set(sizes) != {"A", "B", "C"}:
        raise CaseError("sizes must declare exactly A, B, C")
    plan = request.get("plan")
    if not isinstance(plan, dict):
        raise CaseError("missing plan")
    try:
        m, n, k = plan["m"], plan["n"], plan["k"]
        base_a, base_b, base_c = bases["A"], bases["B"], bases["C"]
        size_a, size_b, size_c = sizes["A"], sizes["B"], sizes["C"]
    except (KeyError, TypeError) as error:
        raise CaseError(f"missing plan/size field: {error}") from error
    if any(type(v) is not int or v <= 0 or v % 16 for v in (m, n, k)):
        raise CaseError("matrix dimensions must be positive multiples of 16")
    if plan.get("dim") != 16 or k * 16384 > 2147483647:
        raise CaseError("unsupported DIM or int32 overflow contract")
    if any(type(v) is not int or v <= 0 for v in (size_a, size_b, size_c)):
        raise CaseError("buffer sizes must be positive integers")
    for name, value in (("A", base_a), ("B", base_b), ("C", base_c)):
        if type(value) is not int or value < 0 or value % 4:
            raise CaseError(f"base {name} must be a 4-byte-aligned address")
    if any(base + size > 1 << 64 for base, size in ((base_a, size_a), (base_b, size_b), (base_c, size_c))):
        raise CaseError("buffer extent exceeds uint64")
    expected = {"A": m * k, "B": k * n, "C": m * n * 4}
    for name in ("A", "B", "C"):
        if sizes[name] != expected[name]:
            raise CaseError(
                f"declared size {sizes[name]} for {name} != derived {expected[name]}")
    ordered = sorted([(base_a, size_a), (base_b, size_b), (base_c, size_c)])
    for (_, _, lo_end), (hi_start, _) in zip(
            [(b, s, b + s) for b, s in ordered], ordered[1:]):
        if lo_end > hi_start:
            raise CaseError("buffer bases overlap")
    return bases, sizes, plan


def validate_pair(request: dict, binary: bytes) -> tuple[dict, dict, dict]:
    """The executed bytes must be EXACTLY the certified bytes."""
    certified = encoding_bytes(request)
    if binary != certified:
        raise CaseError(
            f"binary bytes {sha256_bytes(binary)[:16]}… do not match certified "
            f"bytes_hex {sha256_bytes(certified)[:16]}…")
    return declared_layout(request)


# ---------------------------------------------------------------------------
# Deterministic input patterns (identical definitions exist in the C harness).
# ---------------------------------------------------------------------------

def lcg8_stream(count: int, seed: int = 0x12345678) -> list[int]:
    values, state = [], seed & 0xFFFFFFFF
    for _ in range(count):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        values.append((state >> 16) & 0xFF)
    return values


def fill_inputs(mode: str, size_a: int, size_b: int) -> tuple[bytes, bytes]:
    if mode == "zeros":
        return bytes(size_a), bytes(size_b)
    if mode == "extrema":
        a = bytes(0x80 if i % 2 == 0 else 0x7F for i in range(size_a))
        b = bytes(0x7F if i % 2 == 0 else 0x80 for i in range(size_b))
        return a, b
    if mode == "lcg":
        stream = lcg8_stream(size_a + size_b)
        return bytes(stream[:size_a]), bytes(stream[size_a:])
    raise CaseError(f"unknown mode {mode}")


def to_int8(value: int) -> int:
    return value - 256 if value >= 128 else value


def reference_matmul(a: bytes, b: bytes, m: int, n: int, k: int) -> list[int]:
    return [
        sum(to_int8(a[r * k + kk]) * to_int8(b[kk * n + c]) for kk in range(k))
        for r in range(m) for c in range(n)
    ]


# ---------------------------------------------------------------------------
# Harness generation. The kernel bytes are embedded verbatim (.incbin); this
# never regenerates or recompiles kernel instructions.
# ---------------------------------------------------------------------------

KERNEL_ASM_TEMPLATE = '''.section .kernel_body, "ax", @progbits
.p2align 2
.global kernel_body_start
kernel_body_start:
.incbin "{binary_name}"
.p2align 2
.global kernel_body_end
kernel_body_end:
'''

HARNESS_C_TEMPLATE = '''/* Auto-generated by run_raw_spike.py. The kernel body is embedded verbatim
 * from the certified .bin by kernel_body.S; nothing here generates or
 * compiles kernel instructions. Runtime assertions fail closed. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include "include/gemmini.h"

#define A_BASE 0x{a:08x}u
#define B_BASE 0x{b:08x}u
#define C_BASE 0x{c:08x}u
#define A_BYTES {a_bytes}u
#define B_BYTES {b_bytes}u
#define C_BYTES {c_bytes}u
#define M_DIM {m}u
#define N_DIM {n}u
#define K_DIM {k}u
#define BODY_BYTES {body_bytes}u
_Static_assert(DIM == 16 && ADDR_LEN == 32, "unsupported Gemmini configuration");
_Static_assert(_Generic((elem_t)0, int8_t: 1, default: 0), "int8 input required");
_Static_assert(_Generic((acc_t)0, int32_t: 1, default: 0), "int32 output required");
_Static_assert(BANK_NUM * BANK_ROWS >= {scratchpad_rows}, "scratchpad capacity mismatch");
_Static_assert(ACC_ROWS >= {accumulator_rows}, "accumulator capacity mismatch");

extern const uint8_t kernel_body_start[];
extern const uint8_t kernel_body_end[];
extern uint8_t array_a_start[];
extern uint8_t array_b_start[];
extern uint8_t array_c_start[];

__attribute__((section(".array_a"))) uint8_t array_a_start[A_BYTES];
__attribute__((section(".array_b"))) uint8_t array_b_start[B_BYTES];
__attribute__((section(".array_c"))) uint8_t array_c_start[C_BYTES];

static const uint8_t expected_body[BODY_BYTES] = {{ {expected_bytes} }};
static int32_t ref_c[M_DIM * N_DIM];

static int fail(const char *message) {{
    printf("RAW_SPIKE_FAIL %s\\n", message);
    return 1;
}}

static uint32_t lcg_state = 0x12345678u;
static uint8_t lcg8(void) {{
    lcg_state = lcg_state * 1103515245u + 12345u;
    return (uint8_t)((lcg_state >> 16) & 0xffu);
}}

static void fill_inputs(int mode) {{
    uint32_t i;
    if (mode == 0) {{
        memset(array_a_start, 0, A_BYTES);
        memset(array_b_start, 0, B_BYTES);
    }} else if (mode == 1) {{
        for (i = 0; i < A_BYTES; i++) array_a_start[i] = (i & 1) ? 0x7f : 0x80;
        for (i = 0; i < B_BYTES; i++) array_b_start[i] = (i & 1) ? 0x80 : 0x7f;
    }} else {{
        for (i = 0; i < A_BYTES; i++) array_a_start[i] = lcg8();
        for (i = 0; i < B_BYTES; i++) array_b_start[i] = lcg8();
    }}
    for (i = 0; i < C_BYTES; i++) array_c_start[i] = 0x5au;
}}

static void compute_ref(void) {{
    uint32_t r, c, kk;
    for (r = 0; r < M_DIM; r++)
        for (c = 0; c < N_DIM; c++) {{
            int32_t acc = 0;
            for (kk = 0; kk < K_DIM; kk++)
                acc += (int32_t)(int8_t)array_a_start[r * K_DIM + kk]
                     * (int32_t)(int8_t)array_b_start[kk * N_DIM + c];
            ref_c[r * N_DIM + c] = acc;
        }}
}}

int main(void) {{
    const size_t body_len = (size_t)(kernel_body_end - kernel_body_start);
    uint32_t i;
    int mode;
    FILE *results = fopen("raw_spike_outputs.txt", "w");
    if (results == NULL) return fail("cannot_open_results_file");
    if ((uintptr_t)array_a_start != A_BASE) return fail("a_base_mismatch");
    if ((uintptr_t)array_b_start != B_BASE) return fail("b_base_mismatch");
    if ((uintptr_t)array_c_start != C_BASE) return fail("c_base_mismatch");
    if (body_len != BODY_BYTES) return fail("body_length_mismatch");
    if (((uintptr_t)kernel_body_start & 3u) != 0) return fail("body_unaligned");
    if (memcmp(expected_body, kernel_body_start, body_len) != 0)
        return fail("loaded_body_bytes_mismatch");
    asm volatile("fence.i" ::: "memory");

    for (mode = 0; mode < 3; mode++) {{
        int mismatch = 0;
        fill_inputs(mode);
        compute_ref();
        gemmini_flush(0);
        ((void (*)(void))kernel_body_start)();
        if (memcmp(expected_body, kernel_body_start, body_len) != 0)
            return fail("body_modified_during_execution");
        /* The simulator extension dumps DMA traces to stdout and interleaves
         * with printf, so the full output payload goes to a dedicated file. */
        fprintf(results, "COUT %d ", mode);
        for (i = 0; i < M_DIM * N_DIM; i++) {{
            int32_t got = ((const int32_t *)array_c_start)[i];
            if (got != ref_c[i]) mismatch = 1;
            fprintf(results, "%08" PRIx32, (uint32_t)got);
        }}
        fprintf(results, "\\n");
        fflush(results);
        printf("MODE %d %s\\n", mode, mismatch ? "FAIL" : "PASS");
        fflush(stdout);
        if (mismatch) return fail("output_mismatch");
    }}
    fclose(results);
    printf("{pass_marker}\\n");
    return 0;
}}
'''


def make_kernel_asm(binary_path: str) -> str:
    return KERNEL_ASM_TEMPLATE.format(binary_name=Path(binary_path).name)


def make_harness_c(bases: dict, sizes: dict, plan: dict, body_bytes: int,
                   expected_body: bytes | None = None) -> str:
    expected_body = bytes(body_bytes) if expected_body is None else expected_body
    if len(expected_body) != body_bytes:
        raise CaseError("expected-body length mismatch")
    return HARNESS_C_TEMPLATE.format(
        a=bases["A"], b=bases["B"], c=bases["C"],
        a_bytes=sizes["A"], b_bytes=sizes["B"], c_bytes=sizes["C"],
        m=plan["m"], n=plan["n"], k=plan["k"], body_bytes=body_bytes,
        pass_marker=PASS_MARKER,
        expected_bytes=",".join(f"0x{v:02x}" for v in expected_body),
        scratchpad_rows=plan.get("scratchpad_rows", 32),
        accumulator_rows=plan.get("accumulator_rows", 16),
    )


def section_start_flags(bases: dict) -> list[str]:
    flags = []
    for name in ("A", "B", "C"):
        flags.append(f"-Wl,--section-start=.array_{name.lower()}=0x{bases[name]:08x}")
    return flags


# ---------------------------------------------------------------------------
# Output parsing.
# ---------------------------------------------------------------------------

def parse_dma_trace(stdout: str) -> dict:
    counts: Counter = Counter()
    loads: Counter = Counter()
    for line in stdout.splitlines():
        match = re.match(
            r"GEMMINI: mvin - (0x[0-9a-f]+) cols and (0x[0-9a-f]+) rows .* to addr (0x[0-9a-f]+)$",
            line)
        if match:
            cols, rows, address = (int(value, 16) for value in match.groups())
            counts["mvin"] += 1
            loads[f"0x{address:08x}"] += 1
            if not address & (1 << 31):
                counts["scratchpad_input_bytes"] += cols * rows
        elif line.startswith("GEMMINI: mvout - "):
            counts["mvout"] += 1
        elif line.startswith("GEMMINI: preload - "):
            counts["preload"] += 1
        elif line.startswith("GEMMINI: compute - preload = "):
            counts["compute"] += 1
    return {"operations": dict(counts), "mvin_by_local_address": dict(loads)}


def check_command_counts(request: dict, observed: dict, modes: int = len(MODES)) -> dict:
    expected = Counter()
    loads = Counter()
    for command in request["commands"]:
        kind = command["kind"]
        if kind in ("mvin", "preload", "compute", "mvout"):
            expected[kind] += modes
        if kind == "mvin":
            expected["scratchpad_input_bytes"] += modes * command["cols"] * command["rows"]
            loads[f"0x{command['spad_addr']:08x}"] += modes
    actual = observed.get("operations", {})
    ok = all(actual.get(k, 0) == count for k, count in expected.items())
    ok = ok and observed.get("mvin_by_local_address", {}) == dict(loads)
    return {"pass": ok, "expected_operations": dict(expected),
            "expected_mvin_by_local_address": dict(loads)}


KERNEL_COMMIT = re.compile(r"^core\s+\d+:\s+\d+\s+0x([0-9a-f]+)\s+\(0x([0-9a-f]+)\)")


def check_kernel_trace(path: Path, start: int, binary: bytes, modes: int = len(MODES)) -> dict:
    """Check every committed body opcode and exactly one visit per word/call."""
    visits = [0] * (len(binary) // 4)
    mismatches = []
    with path.open(errors="replace") as stream:
        for line in stream:
            match = KERNEL_COMMIT.match(line)
            if not match:
                continue
            pc, word = (int(x, 16) for x in match.groups())
            if not start <= pc < start + len(binary):
                continue
            offset = pc - start
            if offset % 4 or word != int.from_bytes(binary[offset:offset + 4], "little"):
                if len(mismatches) < 8:
                    mismatches.append({"pc": hex(pc), "instruction": hex(word)})
                continue
            visits[offset // 4] += 1
    return {"pass": bool(visits) and not mismatches and all(n == modes for n in visits),
            "kernel_address": hex(start), "instruction_words": len(visits),
            "committed_kernel_instructions": sum(visits),
            "expected_kernel_instructions": len(visits) * modes,
            "minimum_visits": min(visits, default=0), "maximum_visits": max(visits, default=0),
            "opcode_mismatches": mismatches}


COMMIT_LINE = re.compile(r"^core\s+\d+:\s+\d+\s+0x")


def count_committed(log_text: str) -> int:
    return sum(1 for line in log_text.splitlines() if COMMIT_LINE.match(line))


COUT_LINE = re.compile(r"^COUT (\d+) ([0-9a-f]+)$")


def parse_cout(results_text: str) -> dict[int, list[int]]:
    outputs: dict[int, list[int]] = {}
    for line in results_text.splitlines():
        match = COUT_LINE.match(line)
        if not match:
            continue
        mode, hex_text = match.groups()
        if len(hex_text) % 8:
            raise CaseError(f"malformed COUT payload for mode {mode}")
        values = []
        for i in range(0, len(hex_text), 8):
            word = int(hex_text[i:i + 8], 16)  # matches C printf("%08x")
            values.append(word - (1 << 32) if word >= (1 << 31) else word)
        outputs[int(mode)] = values
    return outputs


def check_outputs(request: dict, stdout: str, results_text: str) -> tuple[bool, list[str]]:
    """Full-output comparison of every mode against the scalar reference.

    COUT payloads live in raw_spike_outputs.txt because the simulator
    extension interleaves its DMA dumps with harness printf on stdout;
    markers are checked as substrings to tolerate any residual interleaving.
    """
    bases, sizes, plan = declared_layout(request)
    m, n, k = plan["m"], plan["n"], plan["k"]
    outputs = parse_cout(results_text)
    problems: list[str] = []
    for index, mode in enumerate(MODES):
        if index not in outputs:
            problems.append(f"missing COUT for mode {mode}")
            continue
        a, b = fill_inputs(mode, sizes["A"], sizes["B"])
        expected = reference_matmul(a, b, m, n, k)
        if outputs[index] != expected:
            problems.append(f"mode {mode}: C output differs from scalar reference")
        if f"MODE {index} PASS" not in stdout:
            problems.append(f"mode {mode}: harness did not report PASS")
    return not problems, problems


# ---------------------------------------------------------------------------
# Remote driver.
# ---------------------------------------------------------------------------

def readelf_sections(readelf: Path, binary: Path, env=None) -> dict[str, dict]:
    result = subprocess.run([str(readelf), "-SW", str(binary)],
                            capture_output=True, text=True, check=True, env=env)
    sections: dict[str, dict] = {}
    for line in result.stdout.splitlines():
        match = re.match(
            r"\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)",
            line)
        if match and match.group(1) != "NULL":
            sections[match.group(1)] = {
                "addr": int(match.group(2), 16),
                "offset": int(match.group(3), 16),
                "size": int(match.group(4), 16),
            }
    return sections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="veritac-gemmini cache root with prefix/, pk-build/, gemmini-rocc-tests/")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True,
                        help="certified raw kernel .bin (embedded verbatim)")
    parser.add_argument("--output", type=Path, required=True,
                        help="artifact directory; must not already exist (fail closed)")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()

    try:
        if out.exists():
            raise CaseError(f"output directory already exists: {out}")
        request = load_request(args.request)
        binary = args.binary.read_bytes()
        bases, sizes, plan = validate_pair(request, binary)
    except (CaseError, OSError) as error:
        print(f"FAIL CLOSED: {error}", file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=False)
    prefix = root / "prefix"
    upstream = root / "gemmini-rocc-tests"
    multiarch = subprocess.check_output(
        ["dpkg-architecture", "-qDEB_HOST_MULTIARCH"], text=True).strip()
    env = dict(os.environ)
    env["PATH"] = f"{prefix / 'usr/bin'}:{prefix / 'bin'}:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{prefix / 'lib'}:{prefix / 'usr/lib' / multiarch}:" + env.get("LD_LIBRARY_PATH", ""))
    cc = prefix / "usr/bin/riscv64-linux-gnu-gcc-13"
    objcopy = prefix / "usr/bin/riscv64-linux-gnu-objcopy-13"
    if not objcopy.exists():
        objcopy = prefix / "usr/bin/riscv64-linux-gnu-objcopy"
    readelf = prefix / "usr/bin/riscv64-linux-gnu-readelf-13"
    if not readelf.exists():
        readelf = prefix / "usr/bin/riscv64-linux-gnu-readelf"
    spike = prefix / "bin/spike"
    pk = root / "pk-build/pk"

    report: dict = {
        "schema_version": 1,
        "backend": "upstream_gemmini_spike_raw_bytes",
        "performance_claim": "none; functional simulator; dynamic instruction/DMA counts only, not hardware latency",
        "success": False,
        "request": str(args.request.resolve()),
        "binary_sha256": sha256_bytes(binary),
        "binary_bytes": len(binary),
        "bases": {k: f"0x{v:08x}" for k, v in bases.items()},
        "sizes": sizes,
        "plan": plan,
        "artifacts": {},
    }
    harness_c = out / "raw_harness.c"
    kernel_asm = out / "kernel_body.S"
    kernel_bin = out / "kernel.bin"
    elf = out / "raw_harness"
    harness_c.write_text(make_harness_c(bases, sizes, plan, len(binary), binary))
    kernel_asm.write_text(make_kernel_asm(kernel_bin.name))
    kernel_bin.write_bytes(binary)  # verbatim certified bytes
    report["artifacts"] = {
        "harness_c": sha256_file(harness_c),
        "kernel_asm": sha256_file(kernel_asm),
        "kernel_bin": sha256_file(kernel_bin),
    }
    compile_cmd = [
        str(cc), f"--sysroot={prefix}", "-static", "-O2",
        "-march=rv64gc", "-mabi=lp64d", "-DBAREMETAL",
        "-I", str(upstream), str(harness_c), str(kernel_asm),
        *section_start_flags(bases), "-o", str(elf),
    ]
    run_cmd = [
        str(spike), f"--log={out / 'spike.log'}", "--log-commits",
        "--extension=gemmini", str(pk), str(elf),
    ]
    report["compile_command"] = compile_cmd
    report["run_command"] = run_cmd
    try:
        provenance = {
            "compiler": cc, "spike": spike, "proxy_kernel": pk,
            "extension": prefix / "lib/libgemmini.so",
            "compiler_parameters": upstream / "include/gemmini_params.h",
            "simulator_parameters": root / "libgemmini/gemmini_params.h",
            "gemmini_header": upstream / "include/gemmini.h",
            "xcustom_header": upstream / "rocc-software/src/xcustom.h",
        }
        report["toolchain"] = {name: {"path": str(path), "sha256": sha256_file(path)}
                               for name, path in provenance.items()}
        if sha256_file(provenance["compiler_parameters"]) != sha256_file(provenance["simulator_parameters"]):
            raise RuntimeError("compiler and simulator configurations differ")
        compiled = subprocess.run(compile_cmd, cwd=out, env=env, capture_output=True,
                                  text=True, timeout=args.timeout)
        (out / "compile.stdout").write_text(compiled.stdout)
        (out / "compile.stderr").write_text(compiled.stderr)
        report["compile_exit_code"] = compiled.returncode
        if compiled.returncode:
            raise RuntimeError("harness compilation failed; see compile.stderr")
        report["elf_sha256"] = sha256_file(elf)

        # ELF-level byte equality: the embedded section must equal the .bin.
        extracted = out / "body_section.bin"
        subprocess.run([str(objcopy), "-O", "binary",
                        "--only-section=.kernel_body", str(elf), str(extracted)],
                       check=True, capture_output=True, timeout=args.timeout, env=env)
        loaded = extracted.read_bytes()
        report["loaded_body_sha256"] = sha256_bytes(loaded)
        if loaded != binary:
            raise RuntimeError("loaded .kernel_body bytes differ from certified .bin")

        # ELF-level placement: array sections must sit at the declared bases.
        sections = readelf_sections(readelf, elf, env)
        placements = {}
        for name, base in bases.items():
            section = sections.get(f".array_{name.lower()}")
            if not section:
                raise RuntimeError(f"missing .array_{name.lower()} section in ELF")
            if section["addr"] != base:
                raise RuntimeError(
                    f".array_{name.lower()} at 0x{section['addr']:08x}, expected 0x{base:08x}")
            if section["size"] != sizes[name]:
                raise RuntimeError(f".array_{name.lower()} size mismatch")
            placements[name] = section
        report["section_placements"] = placements
        body_section = sections.get(".kernel_body")
        if not body_section or body_section["size"] != len(binary):
            raise RuntimeError("kernel section placement/size is invalid")
        report["kernel_section"] = body_section

        ran = subprocess.run(run_cmd, env=env, capture_output=True, text=True,
                             timeout=args.timeout, cwd=out)
        (out / "spike.stdout").write_text(ran.stdout)
        (out / "spike.stderr").write_text(ran.stderr)
        report["exit_code"] = ran.returncode
        report["executed_commands"] = parse_dma_trace(ran.stdout)
        report["command_count_checks"] = check_command_counts(request, report["executed_commands"])
        report["stdout_sha256"] = sha256_file(out / "spike.stdout")
        report["stderr_sha256"] = sha256_file(out / "spike.stderr")
        results_file = out / "raw_spike_outputs.txt"
        if not results_file.exists():
            raise RuntimeError("harness produced no raw_spike_outputs.txt")
        report["outputs_file_sha256"] = sha256_file(results_file)
        if (out / "spike.log").exists():
            report["spike_log_sha256"] = sha256_file(out / "spike.log")
            report["dynamic_instructions"] = count_committed(
                (out / "spike.log").read_text(errors="replace"))
        report["kernel_trace_checks"] = check_kernel_trace(out / "spike.log", body_section["addr"], binary)
        if not report["command_count_checks"]["pass"]:
            raise RuntimeError("dynamic Gemmini command counts differ from the submitted program")
        if not report["kernel_trace_checks"]["pass"]:
            raise RuntimeError("committed kernel instructions differ from the certified body")
        if ran.returncode:
            raise RuntimeError("Spike execution failed; see spike.stderr/stdout")
        if PASS_MARKER not in ran.stdout:
            raise RuntimeError("harness did not emit the full-pass marker")
        if not report["executed_commands"]["operations"].get("compute"):
            raise RuntimeError("no executed Gemmini compute instructions in trace")
        ok, problems = check_outputs(request, ran.stdout,
                                     results_file.read_text())
        report["output_checks"] = {"pass": ok, "problems": problems,
                                   "modes": list(MODES)}
        if not ok:
            raise RuntimeError("; ".join(problems))
        report["success"] = True
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        report["error"] = str(error)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
