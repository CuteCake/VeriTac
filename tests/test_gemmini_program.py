"""Concrete Gemmini program acceptance tests.

These exercise the separate gemmini_program_check Lean CLI: it must accept the
actual generated command stream plus emitted executable bytes for a valid
plan, and reject every mutation of offsets, strides, accumulate flags,
preload/reuse structure, fences, encodings, missing outputs, unknown ops, and
out-of-bound memory -- plus strict-parse failures and missing checkers.

Plan acceptance (gemmini_check) alone is NOT program acceptance; a dedicated
test below proves the distinction.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from CodeGen import gemmini as g

BIN = ROOT / ".lake/build/bin/gemmini_program_check"
PLAN_BIN = ROOT / ".lake/build/bin/gemmini_check"


def base_plan(schedule="baseline", m=16, n=16, k=16):
    return g.make_plan(m, n, k, schedule)


def valid_request(schedule="baseline", m=16, n=16, k=16):
    plan = base_plan(schedule, m, n, k)
    commands = g.gen_commands(plan)
    encoding, why = g.emit_encoding(plan, commands)
    if encoding is None:
        raise unittest.SkipTest("encoding backend unavailable: %s" % why)
    return g.build_program_request(plan, commands, encoding)


def run_cli(request, binary=BIN):
    if not binary.is_file():
        raise unittest.SkipTest("binary missing: %s" % binary)
    proc = subprocess.run([str(binary)], input=json.dumps(request),
                          text=True, capture_output=True, timeout=60)
    if proc.returncode:
        raise AssertionError(proc.stderr or proc.stdout)
    return json.loads(proc.stdout)


def assert_rejected(test, request, label):
    out = run_cli(request)
    unittest.TestCase().assertIs(out["accepted"], False,
                                 "%s was accepted (BUG)" % label)
    unittest.TestCase().assertIsInstance(out.get("reason"), str)
    unittest.TestCase().assertNotIn("program", out)
    unittest.TestCase().assertNotIn("certificate", out)


class ProgramAcceptanceTests(unittest.TestCase):
    def test_valid_baseline_accepted(self):
        req = valid_request("baseline")
        out = run_cli(req)
        self.assertIs(out["accepted"], True, out.get("reason"))
        self.assertEqual(out["program"]["plan"], req["plan"])
        self.assertEqual(out["program"]["commands"], req["commands"])
        self.assertEqual(out["program"]["encoding"], req["encoding"])
        self.assertIsInstance(out.get("certificate"), dict)

    def test_valid_reuse_b_accepted(self):
        req = valid_request("reuse_b")
        out = run_cli(req)
        self.assertIs(out["accepted"], True, out.get("reason"))

    def test_parameterized_64cube_is_accepted(self):
        req = valid_request("baseline", 64, 64, 64)
        out = run_cli(req)
        self.assertIs(out["accepted"], True, out["reason"])



class MutationRejectionTests(unittest.TestCase):
    def build_mutation(self, mutate_commands=None, mutate_plan=None,
                       mutate_encoding=None, drop_encoding=False):
        req = valid_request("baseline")
        req = copy.deepcopy(req)
        if mutate_commands:
            mutate_commands(req["commands"])
        if mutate_plan:
            req["plan"] = mutate_plan(req["plan"])
        if mutate_encoding:
            mutate_encoding(req["encoding"])
        if drop_encoding:
            del req["encoding"]
        return req

    def test_mvin_offset_mutation_rejected(self):
        def mutate(cmds):
            for c in cmds:
                if c["kind"] == "mvin" and c["slot"] == 0:
                    c["offset"] += 16
                    return
        self.assertIs(run_cli(self.build_mutation(mutate))["accepted"], False)

    def test_config_ld_stride_mutation_rejected(self):
        def mutate(cmds):
            for c in cmds:
                if c["kind"] == "config_ld" and c["slot"] == 0:
                    c["stride_bytes"] += 1
                    return
        self.assertIs(run_cli(self.build_mutation(mutate))["accepted"], False)

    def test_accumulate_flag_mutation_rejected(self):
        def mutate(cmds):
            for c in cmds:
                if c["kind"] == "preload":
                    c["out_addr"] ^= (1 << 30)
                    return
        self.assertIs(run_cli(self.build_mutation(mutate))["accepted"], False)

    def test_preload_source_mutation_rejected(self):
        def mutate(cmds):
            for c in cmds:
                if c["kind"] == "preload" and c["bd_spad_addr"] != g.GARBAGE_ADDR:
                    c["bd_spad_addr"] = g.GARBAGE_ADDR
                    return
        self.assertIs(run_cli(self.build_mutation(mutate))["accepted"], False)

    def test_reuse_compute_flag_mutation_rejected(self):
        req = valid_request("reuse_b", 32, 16, 16)
        self.assertTrue(run_cli(req)["accepted"])
        retained = next(c for c in req["commands"] if c["kind"] == "compute" and c["accumulated"])
        retained["accumulated"] = False
        self.assertIs(run_cli(req)["accepted"], False)

    def test_missing_fence_rejected(self):
        def mutate(cmds):
            cmds[:] = [c for c in cmds if c["kind"] != "fence"]
        self.assertIs(run_cli(self.build_mutation(mutate))["accepted"], False)

    def test_missing_mvout_rejected(self):
        def mutate(cmds):
            for idx in range(len(cmds) - 1, -1, -1):
                if cmds[idx]["kind"] == "mvout":
                    del cmds[idx]
                    return
        self.assertIs(run_cli(self.build_mutation(mutate))["accepted"], False)

    def test_unknown_op_rejected(self):
        def mutate(cmds):
            cmds.insert(0, {"kind": "flush"})
        self.assertIs(run_cli(self.build_mutation(mutate))["accepted"], False)

    def test_out_of_bound_memory_rejected(self):
        def mutate(cmds):
            for c in cmds:
                if c["kind"] == "mvin" and c["buf"] == "A":
                    c["offset"] = 10 ** 9
                    return
        self.assertIs(run_cli(self.build_mutation(mutate))["accepted"], False)

    def test_trailing_extra_command_rejected(self):
        def mutate(cmds):
            cmds.append(copy.deepcopy(cmds[-1]))
        self.assertIs(run_cli(self.build_mutation(mutate))["accepted"], False)

    def test_reorder_rejected(self):
        def mutate(cmds):
            cmds[0], cmds[1] = cmds[1], cmds[0]
        self.assertIs(run_cli(self.build_mutation(mutate))["accepted"], False)

    def test_bytes_bit_flip_rejected(self):
        def mutate(enc):
            data = bytearray.fromhex(enc["bytes_hex"])
            data[0] ^= 1
            enc["bytes_hex"] = data.hex()
        self.assertIs(run_cli(self.build_mutation(mutate_encoding=mutate))["accepted"], False)

    def test_commands_bytes_mismatch_rejected(self):
        req = valid_request("baseline")
        req["commands"] = req["commands"][:-1]
        self.assertIs(run_cli(req)["accepted"], False)

    def test_unknown_encoding_format_rejected(self):
        def mutate(enc):
            enc["format"] = "unknown_format_v99"
        self.assertIs(run_cli(self.build_mutation(mutate_encoding=mutate))["accepted"], False)

    def test_missing_encoding_rejected(self):
        self.assertIs(run_cli(self.build_mutation(drop_encoding=True))["accepted"], False)


class StrictParsingTests(unittest.TestCase):
    def test_missing_and_unknown_fields_rejected(self):
        req = valid_request("baseline")
        for key in ("schema", "plan", "commands", "encoding"):
            broken = copy.deepcopy(req)
            del broken[key]
            self.assertIs(run_cli(broken)["accepted"], False, key)
        broken = copy.deepcopy(req)
        broken["surprise"] = 1
        self.assertIs(run_cli(broken)["accepted"], False)
        broken = copy.deepcopy(req)
        broken["plan"]["surprise"] = 1
        self.assertIs(run_cli(broken)["accepted"], False)
        broken = copy.deepcopy(req)
        broken["commands"][0]["surprise"] = 1
        self.assertIs(run_cli(broken)["accepted"], False)

    def test_malformed_json_rejected(self):
        if not BIN.is_file():
            self.skipTest("binary missing")
        proc = subprocess.run([str(BIN)], input="{broken\n", text=True,
                              capture_output=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIs(json.loads(proc.stdout)["accepted"], False)

    def test_non_object_input_rejected(self):
        for value in (None, [], "req", 42, True):
            with self.subTest(value=value):
                out = run_cli(value)
                self.assertIs(out["accepted"], False)

    def test_empty_input_rejected(self):
        if not BIN.is_file():
            self.skipTest("binary missing")
        proc = subprocess.run([str(BIN)], input="", text=True,
                              capture_output=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIs(json.loads(proc.stdout)["accepted"], False)


class FailClosedTests(unittest.TestCase):
    def test_missing_checker_binary_fails_closed(self):
        env_bin = "/nonexistent/gemmini_program_check"
        request = {"schema": g.PROGRAM_CHECK_SCHEMA, "plan": base_plan(),
                   "commands": [], "encoding": {}}
        saved = os.environ.get("VERITAC_GEMMINI_PROGRAM_CHECK_BIN")
        os.environ["VERITAC_GEMMINI_PROGRAM_CHECK_BIN"] = env_bin
        try:
            ok, reason, cert = g.gemmini_program_check(request)
        finally:
            if saved is None:
                os.environ.pop("VERITAC_GEMMINI_PROGRAM_CHECK_BIN", None)
            else:
                os.environ["VERITAC_GEMMINI_PROGRAM_CHECK_BIN"] = saved
        self.assertIs(ok, False)
        self.assertIn("fail-closed", reason)
        self.assertIsNone(cert)

    def test_encoding_backend_error_fails_closed(self):
        from unittest.mock import patch
        plan = base_plan()
        with patch("CodeGen.gemmini_encoding.emit_encoding", side_effect=RuntimeError("encoder failed")):
            ok, reason, cert = g.check_program(plan, g.gen_commands(plan))
        self.assertIs(ok, False)
        self.assertIn("fail-closed", reason)
        self.assertIsNone(cert)


class PlanVsProgramTests(unittest.TestCase):
    def test_plan_acceptance_is_not_program_acceptance(self):
        """A valid plan with a corrupted command stream: plan check accepts,
        program check must reject.  Hash equality is likewise insufficient."""
        plan = base_plan("baseline")
        ok, reason, _ = g.gemmini_check(plan)
        if not ok:
            self.skipTest("plan checker unavailable")
        commands = g.gen_commands(plan)
        encoding, why = g.emit_encoding(plan, commands)
        if encoding is None:
            self.skipTest("encoding backend unavailable: %s" % why)
        mutated = copy.deepcopy(g.serialize_commands(commands))
        for c in mutated:
            if c["kind"] == "mvin" and c["slot"] == 1:
                c["offset"] += 8
                break
        req = g.build_program_request(plan, mutated, encoding)
        self.assertIs(run_cli(req)["accepted"], False)
        # And the honest request still passes.
        req_ok = g.build_program_request(plan, g.serialize_commands(commands),
                                         encoding)
        self.assertIs(run_cli(req_ok)["accepted"], True)


def placeholder_encoding():
    """Schema-valid encoding artifact (schema per encoding-contract.md); its
    bytes are not expected to decode to the commands, which is fine -- the CLI
    must still fail closed on checker unavailability."""
    return {"format": "veritac_gemmini_bytes_v2",
            "word_size": 4, "endianness": "little",
            "regs": {"rs1": 5, "rs2": 6, "temp": 31},
            "bases": {"A": 4096, "B": 8192, "C": 16384},
            "sizes": {"A": 1024, "B": 1024, "C": 4096},
            "bytes_hex": "0ff0000f" "00008067"}


class CliParsingTests(unittest.TestCase):
    """CLI paths reachable without the encoding backend (placeholder encoding
    object passes request parsing; the proved decoder is still required for
    acceptance)."""

    @staticmethod
    def placeholder_request(schedule="baseline"):
        plan = base_plan(schedule)
        return {"schema": g.PROGRAM_CHECK_SCHEMA, "plan": plan,
                "commands": g.serialize_commands(g.gen_commands(plan)),
                "encoding": placeholder_encoding()}

    def test_invalid_plan_rejected_with_plan_reason(self):
        req = self.placeholder_request()
        req["plan"]["m"] = 31  # not divisible by dim
        out = run_cli(req)
        self.assertIs(out["accepted"], False)
        self.assertNotIn("unavailable", out["reason"])
        self.assertIn("divisible", out["reason"])

    def test_complete_request_accepts_with_composed_theorem(self):
        out = run_cli(valid_request())
        self.assertIs(out["accepted"], True, out["reason"])
        self.assertEqual(out["certificate"]["soundness"], "VeriTac.Gemmini.checkExecutable_sound")

    def test_placeholder_bytes_do_not_accept(self):
        """Schema-valid placeholder bytes (fence+ret only) must never be
        accepted for a real program."""
        out = run_cli(self.placeholder_request())
        self.assertIs(out["accepted"], False)


if __name__ == "__main__":
    unittest.main()
