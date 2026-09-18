"""Tests for the Tenstorrent Metalium emitter + runner
(specializations/tenstorrent_protocol/runtime).

Protocol semantics are owned by specializations.tenstorrent_protocol.protocol
(the native checker); these tests bind the emitted artifacts back to it:
the emitted program table is decoded and re-checked, the deterministic
input generator is compiled and compared against the Python reference, and
the runner pipeline is exercised offline (emit-only).
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specializations.tenstorrent_protocol import protocol
from specializations.tenstorrent_protocol.runtime import emit_cpp, runner


def make_task(**over):
    obj = {
        "id": "test-task",
        "input_pages": 4,
        "expected": [2, 0, 3, 1],
        "slots": 2,
        "page_bytes": 2048,
    }
    obj.update(over)
    return protocol.parse_task(obj)


def serial_proposal(task):
    return protocol.serial_baseline(task)


class SchemaReusesNativeChecker(unittest.TestCase):
    def test_bad_tasks_rejected(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.parse_task({"id": "x", "input_pages": 4, "expected": [0],
                                 "slots": 1, "page_bytes": 33})  # not multiple of 32
        with self.assertRaises(protocol.ProtocolError):
            protocol.parse_task({"id": "x", "input_pages": 1, "expected": [1],
                                 "slots": 1, "page_bytes": 32})  # src out of range
        with self.assertRaises(protocol.ProtocolError):
            protocol.parse_task({"id": "x", "input_pages": 1, "expected": [0],
                                 "slots": 1, "page_bytes": 32, "extra": 1})

    def test_bad_proposals_rejected(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.parse_proposal({"program": [{"op": "steal", "slot": 0}],
                                     "rationale": "r"})
        with self.assertRaises(protocol.ProtocolError):
            protocol.parse_proposal({"program": [{"op": "read", "slot": 0}],
                                     "rationale": "r"})  # missing src


class EncodingBindsToProtocolIR(unittest.TestCase):
    def test_serial_baseline_encoding(self):
        task = make_task()
        prop = serial_proposal(task)
        words = emit_cpp.encode_program(prop.program)
        self.assertEqual(words[:12], [0, 0, 0, 1, 0, 2, 2, 0, 0, 3, 0, 0])
        self.assertEqual(len(words), 3 * len(prop.program))

    def test_mutation_changes_encoding(self):
        task = make_task()
        prop = serial_proposal(task)
        words_a = emit_cpp.encode_program(prop.program)
        mutated = list(prop.program)
        mutated[1] = protocol.Read(slot=0, src=(mutated[1].src + 1) % task.input_pages)
        words_b = emit_cpp.encode_program(mutated)
        self.assertNotEqual(words_a, words_b)
        self.assertEqual(words_a[4:6], [0, 2])  # read slot 0, src 2 in baseline
        self.assertEqual(words_b[4:6], [0, 3])

    def test_decode_roundtrip_rechecks_accepted(self):
        task = make_task()
        prop = serial_proposal(task)
        back = emit_cpp.decode_program(emit_cpp.encode_program(prop.program))
        prop2 = protocol.parse_proposal({"program": back, "rationale": "roundtrip"})
        self.assertEqual(protocol.check(task, prop2).status, protocol.STATUS_ACCEPTED)

    def test_decode_rejects_unknown_opcode(self):
        with self.assertRaises(ValueError):
            emit_cpp.decode_program([99, 0, 0])
        with self.assertRaises(ValueError):
            emit_cpp.decode_program([0, 0])  # not a triple


def extract_const_array(text, name):
    m = re.search(
        r"constexpr\s+uint32_t\s+" + re.escape(name) +
        r"\[[^]]*\]\s*=\s*\{([^;]*)\};", text, re.S)
    if not m:
        raise AssertionError(f"array {name} not found in emitted host")
    body = m.group(1)
    body = re.sub(r"//[^\n]*", "", body)
    return [int(w.strip().rstrip("u")) for w in body.split(",") if w.strip()]


class EmittedHostBindsToChecker(unittest.TestCase):
    def test_program_table_decodes_and_is_accepted(self):
        task = make_task()
        prop = serial_proposal(task)
        host = emit_cpp.emit_host(task, prop.program)
        words = extract_const_array(host, "kProgram")
        self.assertEqual(words, emit_cpp.encode_program(prop.program))
        back = emit_cpp.decode_program(words)
        checked = protocol.check(task, protocol.parse_proposal(
            {"program": back, "rationale": "decoded"}))
        self.assertEqual(checked.status, protocol.STATUS_ACCEPTED)

    def test_expected_table_binds_to_task(self):
        task = make_task()
        host = emit_cpp.emit_host(task, serial_proposal(task).program)
        self.assertEqual(extract_const_array(host, "kExpected"), list(task.expected))

    def test_tampered_wait_becomes_unsafe(self):
        """Binding proof: mutate ONE word of the emitted table (wait_reads ->
        read on the same busy slot) and the native checker rejects the
        program reconstructed from the emitted text."""
        task = make_task()
        prop = serial_proposal(task)
        host = emit_cpp.emit_host(task, prop.program)
        words = extract_const_array(host, "kProgram")
        i = words.index(emit_cpp.OPCODES["wait_reads"])  # first wait_reads
        words[i] = emit_cpp.OPCODES["read"]
        words[i + 1] = 0  # slot 0 still has its read pending -> unsafe
        words[i + 2] = prop.program[1].src
        tampered = protocol.check(task, protocol.parse_proposal(
            {"program": emit_cpp.decode_program(words), "rationale": "t"}))
        self.assertEqual(tampered.status, protocol.STATUS_UNSAFE)


class EmittedKernelOpMap(unittest.TestCase):
    def test_every_opcode_case_uses_required_calls(self):
        kernel = emit_cpp.emit_kernel()
        cases = dict(re.findall(r"case (OP_\w+):\s*(.*?)\s*break;", kernel, re.S))
        self.assertEqual(len(cases), 8)
        required = {
            "OP_RESERVE": ["cb_reserve_back"],
            "OP_READ": ["get_write_ptr", "noc_async_read", "src_acc.get_noc_addr"],
            "OP_WAIT_READS": ["noc_async_read_barrier"],
            "OP_PUBLISH": ["cb_push_back"],
            "OP_ACQUIRE": ["cb_wait_front"],
            "OP_WRITE": ["get_read_ptr", "noc_async_write", "dst_acc.get_noc_addr"],
            "OP_WAIT_WRITES": ["noc_async_write_barrier"],
            "OP_RELEASE": ["cb_pop_front"],
        }
        for case, calls in required.items():
            for call in calls:
                self.assertIn(call, cases[case], f"{case} missing {call}")
        # Banned in code (comments stripped): no SFPU/format conversion.
        code_only = re.sub(r"//[^\n]*", "", kernel)
        for banned in ("llk_math", "SFPU", "DataFormat::Float16", "unpack_"):
            self.assertNotIn(banned, code_only)


class CpuReferenceParity(unittest.TestCase):
    def test_reference_matches_expected_mapping(self):
        task = make_task()
        ref = runner.compute_reference(task)
        inp, pb = ref["input"], task.page_bytes
        pages = [inp[i * pb : (i + 1) * pb] for i in range(task.input_pages)]
        self.assertEqual(ref["expected_output"],
                         b"".join(pages[src] for src in task.expected))
        self.assertNotEqual(ref["input_checksum"], ref["expected_output_checksum"])

    def test_fnv1a64_known_vector(self):
        self.assertEqual(runner.fnv1a64(b"abc"), 0xE71FA2190541574B)
        self.assertEqual(runner.fnv1a64(b""), 0xCBF29CE484222325)

    def test_fill_is_deterministic(self):
        self.assertEqual(runner.fill_input(256, seed=1), runner.fill_input(256, seed=1))
        self.assertNotEqual(runner.fill_input(256, seed=1), runner.fill_input(256, seed=2))


class InputGenHeaderCompiles(unittest.TestCase):
    def test_compiled_cpp_matches_python_reference(self):
        cc = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if cc is None:
            self.skipTest("no C++ compiler on PATH")
        with tempfile.TemporaryDirectory() as tmp:
            hpp = Path(tmp) / "input_gen.hpp"
            hpp.write_text(emit_cpp.emit_input_gen())
            main = Path(tmp) / "main.cpp"
            main.write_text(
                '#include "input_gen.hpp"\n'
                "#include <cstdio>\n"
                "int main() {\n"
                "    std::vector<uint8_t> v(2048 * 4);\n"
                "    veritac_fill_input(v, 0x12345678u);\n"
                "    std::printf(\"%016llx\\n\", (unsigned long long)veritac_fnv1a64(v.data(), v.size()));\n"
                "    return 0;\n"
                "}\n")
            exe = Path(tmp) / "gen"
            build = subprocess.run([cc, "-std=c++20", "-Wall", str(main), "-o", str(exe)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            out = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            expected = runner.fnv1a64(runner.fill_input(2048 * 4))
            self.assertEqual(out.stdout.strip(), format(expected, "016x"))


class RunnerPipelineOffline(unittest.TestCase):
    def _write_inputs(self, tmp):
        task = make_task()
        prop = serial_proposal(task)
        task_p = Path(tmp) / "task.json"
        prop_p = Path(tmp) / "proposal.json"
        task_p.write_text(emit_cpp.task_to_dict_json(task))
        prop_p.write_text(emit_cpp.proposal_to_dict_json(prop))
        return task_p, prop_p

    def test_emit_only_produces_bundle_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_p, prop_p = self._write_inputs(tmp)
            out = Path(tmp) / "artifacts"
            code = runner.main(["--task", str(task_p), "--proposal", str(prop_p),
                                "--out", str(out), "--emit-only"])
            self.assertEqual(code, 0)
            results = (out / "results.json")
            import json
            report = json.loads(results.read_text())
            self.assertEqual(report["status"], "emitted-not-run")
            self.assertEqual(report["protocol_check"]["status"], "accepted")
            for rel in ("host_main.cpp", "input_gen.hpp",
                        "kernels/dm_slot_copy.cpp", "CMakeLists.txt"):
                self.assertIn(rel, report["emitted_artifacts"])
                self.assertTrue((out / "src" / rel).is_file())
            # Bundle JSON round-trips through the native checker.
            task2 = protocol.parse_task(json.loads((out / "src" / "task.json").read_text()))
            prop2 = protocol.parse_proposal(
                json.loads((out / "src" / "proposal.json").read_text()))
            self.assertEqual(protocol.check(task2, prop2).status, protocol.STATUS_ACCEPTED)

    def test_unsafe_proposal_is_rejected_with_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = make_task()
            program = list(serial_proposal(task).program)
            # Drop the first wait_reads: publish while the read is pending.
            del program[2]
            prop_p = Path(tmp) / "proposal.json"
            task_p = Path(tmp) / "task.json"
            task_p.write_text(emit_cpp.task_to_dict_json(task))
            prop_p.write_text(emit_cpp.proposal_to_dict_json(
                protocol.Proposal(program=tuple(program), rationale="mutated")))
            out = Path(tmp) / "artifacts"
            code = runner.main(["--task", str(task_p), "--proposal", str(prop_p),
                                "--out", str(out), "--emit-only"])
            self.assertEqual(code, runner.EXIT_PROTOCOL)
            import json
            report = json.loads((out / "results.json").read_text())
            self.assertEqual(report["protocol_check"]["status"], protocol.STATUS_UNSAFE)

    def test_missing_tt_metal_env_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_p, prop_p = self._write_inputs(tmp)
            out = Path(tmp) / "artifacts"
            env = {k: v for k, v in os.environ.items()
                   if k not in ("TT_METAL_HOME", "TT_METAL_SOURCE",
                                "TT_METAL_INSTALL_PREFIX")}
            code = subprocess.run(
                [sys.executable, "-m",
                 "specializations.tenstorrent_protocol.runtime.runner",
                 "--task", str(task_p), "--proposal", str(prop_p),
                 "--out", str(out), "--tt-metal-home", os.path.join(tmp, "nope")],
                capture_output=True, text=True, env=env,
                cwd=str(Path(__file__).resolve().parents[1]))
            self.assertEqual(code.returncode, runner.EXIT_NO_ENV)
            self.assertIn("nothing built/run", code.stdout + code.stderr)


if __name__ == "__main__":
    unittest.main()
