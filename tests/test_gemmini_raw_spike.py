"""Tests for benchmarks/gemmini/run_raw_spike.py (raw certified-bytes harness).

Covers request/binary binding, fail-closed validation, harness generation
(embeds kernel bytes verbatim, never recompiles them), output comparison, and
trace parsing. Pure logic runs locally; the actual Spike execution happens on
the Spark3 host and is reported separately.

Run from the project root:  PYTHONPATH=. python3 -m unittest tests.test_gemmini_raw_spike
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO / "benchmarks" / "gemmini" / "run_raw_spike.py"

_spec = importlib.util.spec_from_file_location("run_raw_spike", RUNNER_PATH)
rrs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rrs)


def synth_request(m=16, n=16, k=16, bases=None, body_words=None):
    """A minimal-but-schema-valid request whose body is fence + ret."""
    bases = bases or {"A": 0x10000000, "B": 0x20000000, "C": 0x30000000}
    body = body_words if body_words is not None else [0x0FF0000F, 0x00008067]
    return {
        "schema": "veritac_program_request_v1",
        "plan": {"m": m, "n": n, "k": k, "dim": 16, "schedule": "baseline",
                 "scratchpad_rows": 32, "accumulator_rows": 16},
        "commands": [{"kind": "fence"}],
        "encoding": {
            "format": "veritac_gemmini_bytes_v2",
            "word_size": 4,
            "endianness": "little",
            "regs": {"rs1": 5, "rs2": 6, "temp": 31},
            "bases": bases,
            "sizes": {"A": m * k, "B": k * n, "C": m * n * 4},
            "bytes_hex": "".join((w & 0xFFFFFFFF).to_bytes(4, "little").hex() for w in body),
        },
    }


class BindingTests(unittest.TestCase):
    def test_binary_must_equal_certified_bytes(self):
        request = synth_request()
        rrs.validate_pair(request, bytes.fromhex(request["encoding"]["bytes_hex"]))
        with self.assertRaises(rrs.CaseError):
            rrs.validate_pair(request, b"\x00" * 8)

    def test_single_flipped_byte_rejected(self):
        request = synth_request()
        good = bytearray(bytes.fromhex(request["encoding"]["bytes_hex"]))
        good[0] ^= 1
        with self.assertRaises(rrs.CaseError):
            rrs.validate_pair(request, bytes(good))

    def test_bad_schema_rejected(self):
        request = synth_request()
        request["schema"] = "something_else"
        with self.assertRaises(rrs.CaseError):
            rrs.validate_request_dict(request)

    def test_bad_format_rejected(self):
        request = synth_request()
        request["encoding"]["format"] = "v1"
        with self.assertRaises(rrs.CaseError):
            rrs.validate_request_dict(request)

    def test_odd_hex_rejected(self):
        request = synth_request()
        request["encoding"]["bytes_hex"] = "0f0"
        with self.assertRaises(rrs.CaseError):
            rrs.validate_request_dict(request)

    def test_non_hex_rejected(self):
        request = synth_request()
        request["encoding"]["bytes_hex"] = "zzzzzzzz"
        with self.assertRaises(rrs.CaseError):
            rrs.validate_request_dict(request)

    def test_missing_plan_rejected(self):
        request = synth_request()
        del request["plan"]
        with self.assertRaises(rrs.CaseError):
            rrs.declared_layout(request)

    def test_wrong_derived_sizes_rejected(self):
        request = synth_request()
        request["encoding"]["sizes"]["C"] = 999
        with self.assertRaises(rrs.CaseError):
            rrs.declared_layout(request)

    def test_unaligned_base_rejected(self):
        request = synth_request(bases={"A": 0x10000002, "B": 0x20000000, "C": 0x30000000})
        with self.assertRaises(rrs.CaseError):
            rrs.declared_layout(request)

    def test_overlapping_bases_rejected(self):
        request = synth_request(bases={"A": 0x10000000, "B": 0x10000040, "C": 0x30000000})
        with self.assertRaises(rrs.CaseError):
            rrs.declared_layout(request)

    def test_missing_bases_rejected(self):
        request = synth_request()
        del request["encoding"]["bases"]["C"]
        with self.assertRaises(rrs.CaseError):
            rrs.declared_layout(request)


class HarnessGenerationTests(unittest.TestCase):
    def test_kernel_asm_embeds_verbatim_incbin(self):
        asm = rrs.make_kernel_asm("kernel.bin")
        self.assertIn('.incbin "kernel.bin"', asm)
        self.assertIn('.section .kernel_body, "ax", @progbits', asm)
        self.assertIn("kernel_body_start:", asm)
        self.assertIn("kernel_body_end:", asm)

    def test_harness_covers_bases_sizes_and_markers(self):
        request = synth_request()
        bases, sizes, plan = rrs.declared_layout(request)
        text = rrs.make_harness_c(bases, sizes, plan, body_bytes=8)
        self.assertIn("#define A_BASE 0x10000000u", text)
        self.assertIn("#define B_BASE 0x20000000u", text)
        self.assertIn("#define C_BASE 0x30000000u", text)
        self.assertIn("#define A_BYTES 256u", text)
        self.assertIn("#define C_BYTES 1024u", text)
        self.assertIn('section(".array_a")', text)
        self.assertIn("a_base_mismatch", text)
        self.assertIn("body_modified_during_execution", text)
        self.assertIn("memcmp(expected_body, kernel_body_start, body_len)", text)
        self.assertIn(rrs.PASS_MARKER, text)
        self.assertIn('fopen("raw_spike_outputs.txt", "w")', text)
        self.assertIn('fprintf(results, "COUT %d ", mode)', text)
        self.assertIn("gemmini_flush(0)", text)
        self.assertIn('asm volatile("fence.i"', text)
        self.assertIn("#include \"include/gemmini.h\"", text)

    def test_section_start_flags(self):
        flags = rrs.section_start_flags({"A": 0x10000000, "B": 0x20000000, "C": 0x30000000})
        self.assertEqual(flags, [
            "-Wl,--section-start=.array_a=0x10000000",
            "-Wl,--section-start=.array_b=0x20000000",
            "-Wl,--section-start=.array_c=0x30000000",
        ])

    def test_kernel_bin_copy_is_verbatim(self):
        request = synth_request()
        raw = bytes.fromhex(request["encoding"]["bytes_hex"])
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "certified.bin"
            source.write_bytes(raw)
            kernel_bin = Path(tmp) / "out" / "kernel.bin"
            kernel_bin.parent.mkdir()
            kernel_bin.write_bytes(source.read_bytes())
            self.assertEqual(rrs.sha256_file(kernel_bin), rrs.sha256_file(source))
            asm = rrs.make_kernel_asm(kernel_bin.name)
            self.assertIn('.incbin "kernel.bin"', asm)


class PatternAndOutputTests(unittest.TestCase):
    def test_lcg_matches_c_definition(self):
        stream = rrs.lcg8_stream(6)
        state = 0x12345678
        expected = []
        for _ in range(6):
            state = (state * 1103515245 + 12345) & 0xFFFFFFFF
            expected.append((state >> 16) & 0xFF)
        self.assertEqual(stream, expected)
        self.assertEqual(len(set(stream[:64])) > 1, True)

    def test_zeros_reference(self):
        expected = [0] * (2 * 2)
        self.assertEqual(rrs.reference_matmul(bytes(4), bytes(4), 2, 2, 2), expected)

    def test_extrema_reference_matches_c_fill_order(self):
        a, b = rrs.fill_inputs("extrema", 4, 4)
        self.assertEqual(list(a), [0x80, 0x7F, 0x80, 0x7F])
        self.assertEqual(list(b), [0x7F, 0x80, 0x7F, 0x80])
        # Hand-computed for m=n=k=2: rows alternate sign/magnitude, only the
        # c=0 column cancels; c=1 pairs (-128,-128) with (127,-128).
        self.assertEqual(rrs.reference_matmul(a, b, 2, 2, 2),
                         [-127, 128, -127, 128])

    def test_small_random_reference_by_hand(self):
        a = bytes([1, 2, 3, 4])
        b = bytes([5, 256 - 6, 7, 256 - 8])
        self.assertEqual(rrs.reference_matmul(a, b, 2, 2, 2),
                         [1 * 5 + 2 * 7, 1 * (-6) + 2 * (-8),
                          3 * 5 + 4 * 7, 3 * (-6) + 4 * (-8)])

    def test_unknown_mode_rejected(self):
        with self.assertRaises(rrs.CaseError):
            rrs.fill_inputs("bogus", 4, 4)

    def test_check_outputs_detects_mismatch(self):
        request = synth_request(m=16, n=16, k=16)

        def cout_line(mode, values):
            return "COUT %d " % mode + "".join(f"{v & 0xFFFFFFFF:08x}" for v in values)

        expected = [rrs.reference_matmul(*rrs.fill_inputs(m, 256, 256), 16, 16, 16)
                    for m in rrs.MODES]
        cout, marks = [], []
        for index in range(3):
            cout.append(cout_line(index, expected[index]))
            marks.append(f"MODE {index} PASS")
        results_text = "\n".join(cout) + "\n"
        stdout = "\n".join(marks + [rrs.PASS_MARKER]) + "\n"
        ok, problems = rrs.check_outputs(request, stdout, results_text)
        self.assertTrue(ok, problems)
        bad_stdout = stdout.replace("MODE 0 PASS", "MODE 0 FAIL")
        ok, problems = rrs.check_outputs(request, bad_stdout, results_text)
        self.assertFalse(ok)
        self.assertTrue(any("harness did not report PASS" in p for p in problems))
        # One flipped output word must fail the comparison.
        flipped = list(cout)
        line = flipped[0].split(" ")
        line[2] = "01" + line[2][2:]
        flipped[0] = " ".join(line)
        ok, problems = rrs.check_outputs(request, stdout, "\n".join(flipped) + "\n")
        self.assertFalse(ok)
        self.assertTrue(any("differs from scalar reference" in p for p in problems))
        # Even a stdout interleaved with junk must not break marker checks.
        ok, _ = rrs.check_outputs(request, "GEMMINI: junk\n" + stdout + "trailer\n",
                                  results_text)
        self.assertTrue(ok)

    def test_missing_cout_fails_closed(self):
        request = synth_request(m=16, n=16, k=16)
        ok, problems = rrs.check_outputs(request, rrs.PASS_MARKER + "\n", "")
        self.assertFalse(ok)
        self.assertEqual(len(problems), 3)

    def test_malformed_cout_rejected(self):
        with self.assertRaises(rrs.CaseError):
            rrs.parse_cout("COUT 0 abc\n")


class TraceParsingTests(unittest.TestCase):
    def test_dma_trace_counts(self):
        stdout = (
            "GEMMINI: mvin - 0x10 cols and 0x10 rows at scale 1 to addr 0x00000000\n"
            "GEMMINI: mvin - 0x10 cols and 0x10 rows at scale 1 to addr 0x10000010\n"
            "GEMMINI: preload - 0x10 cols and 0x10 rows\n"
            "GEMMINI: compute - preload = 0x0, compute = 0x1\n"
            "GEMMINI: mvout - 0x10 cols and 0x10 rows from addr 0xe0000000\n"
        )
        trace = rrs.parse_dma_trace(stdout)
        self.assertEqual(trace["operations"]["mvin"], 2)
        self.assertEqual(trace["operations"]["scratchpad_input_bytes"], 512)
        self.assertEqual(trace["operations"]["compute"], 1)
        self.assertEqual(trace["operations"]["mvout"], 1)
        self.assertEqual(trace["mvin_by_local_address"]["0x00000000"], 1)

    def test_committed_count(self):
        log = (
            "core   0: 3 0x0000000000001000 (0x00000297) x5  0x0000000000001000\n"
            "this is not a commit line\n"
            "core   0: 4 0x0000000000001004 (0x00028293) x11 0x0000000000001020\n"
        )
        self.assertEqual(rrs.count_committed(log), 2)


class CliFailClosedTests(unittest.TestCase):
    def run_cli(self, request_path, binary_path, output_path):
        return subprocess.run(
            [sys.executable, str(RUNNER_PATH), "--root", "/nonexistent",
             "--request", str(request_path), "--binary", str(binary_path),
             "--output", str(output_path)],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(REPO)})

    def test_rejects_existing_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            result = self.run_cli(Path(tmp) / "r.json", Path(tmp) / "b.bin", out)
            self.assertEqual(result.returncode, 1)
            self.assertIn("FAIL CLOSED", result.stderr)

    def test_rejects_binary_mismatch_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = synth_request()
            request_path = Path(tmp) / "r.json"
            request_path.write_text(json.dumps(request))
            binary_path = Path(tmp) / "bad.bin"
            binary_path.write_bytes(b"\x00\x01\x02\x03")
            out = Path(tmp) / "out"
            result = self.run_cli(request_path, binary_path, out)
            self.assertEqual(result.returncode, 1)
            self.assertIn("FAIL CLOSED", result.stderr)
            self.assertFalse(out.exists())

    def test_rejects_missing_request_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_cli(Path(tmp) / "nope.json", Path(tmp) / "b.bin",
                                  Path(tmp) / "out")
            self.assertEqual(result.returncode, 1)
            self.assertFalse((Path(tmp) / "out").exists())


class ExactTraceTests(unittest.TestCase):
    def test_every_committed_word_is_bound_to_body(self):
        import tempfile
        words = [0x000002B7, 0x00008067]
        body = b"".join(w.to_bytes(4, "little") for w in words)
        text = "".join(f"core 0: 0 0x{0x10000 + i * 4:016x} (0x{w:08x})\n"
                       for _ in range(3) for i, w in enumerate(words))
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "trace"
            log.write_text(text)
            self.assertTrue(rrs.check_kernel_trace(log, 0x10000, body)["pass"])
            log.write_text(text.replace("0x000002b7", "0x000002b6", 1))
            self.assertFalse(rrs.check_kernel_trace(log, 0x10000, body)["pass"])

    def test_missing_instruction_visit_is_rejected(self):
        import tempfile
        body = (0x00008067).to_bytes(4, "little")
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "trace"
            log.write_text("core 0: 0 0x0000000000010000 (0x00008067)\n")
            self.assertFalse(rrs.check_kernel_trace(log, 0x10000, body)["pass"])

    def test_trace_counts_must_match_submitted_commands(self):
        request = synth_request()
        request["commands"] = [{"kind": "mvin", "cols": 16, "rows": 16, "spad_addr": 0},
                               {"kind": "compute"}, {"kind": "fence"}]
        observed = {"operations": {"mvin": 3, "compute": 3, "scratchpad_input_bytes": 768},
                    "mvin_by_local_address": {"0x00000000": 3}}
        self.assertTrue(rrs.check_command_counts(request, observed)["pass"])
        observed["operations"]["compute"] = 2
        self.assertFalse(rrs.check_command_counts(request, observed)["pass"])

    def test_harness_compares_initial_bytes_against_independent_copy(self):
        request = synth_request()
        bases, sizes, plan = rrs.declared_layout(request)
        body = bytes.fromhex(request["encoding"]["bytes_hex"])
        text = rrs.make_harness_c(bases, sizes, plan, len(body), body)
        self.assertIn("loaded_body_bytes_mismatch", text)
        self.assertIn("0x0f,0x00,0xf0,0x0f", text)
        self.assertNotIn(chr(92) + chr(92) + "n", text)


if __name__ == "__main__":
    unittest.main()
