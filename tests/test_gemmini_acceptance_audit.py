"""Bounded final-acceptance audit tests (independent audit worker).

Adversarial coverage of the Gemmini program-acceptance integration:
  - metadata (sizes/bases) tampering must be rejected,
  - exact candidate byte mutations with valid metadata must not verify,
  - command/byte disagreement must be rejected,
  - a stale or fake Lean checker response must never yield a verified
    artifact, because the kernel-checked concrete certificate re-decides
    the actual bytes,
  - the emitted .bin must be exactly the certificate code,
  - the default 32x16x16 reuse_b winner and the baseline fallback must
    genuinely produce kernel-checked certificates.

Temp artifacts live under .lake/gemmini_audit to avoid external-dir
permission issues. Run: PYTHONPATH=. python3 tests/test_gemmini_acceptance_audit.py
"""
import json
import os
import stat
import unittest
from pathlib import Path

from CodeGen import gemmini as g

AUDIT_DIR = Path(".lake/gemmini_audit/acceptance")
STALE_RESPONSE = AUDIT_DIR / "stale_response.json"
STALE_CHECKER = AUDIT_DIR / "stale_checker.sh"
FAKE_CHECKER = AUDIT_DIR / "fake_accept_checker.py"


def hex_to_bytes(hexstr):
    return bytearray(bytes.fromhex(hexstr))


def bytes_to_hex(bs):
    return bytes(bs).hex()


class AuditAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        cls.binpath = g.program_checker_binary_path()
        assert os.path.isfile(cls.binpath), cls.binpath

    def build_request(self, plan):
        commands = g.gen_commands(plan)
        encoding, why = g.emit_encoding(plan, commands)
        self.assertIsNotNone(encoding, why)
        return g.build_program_request(plan, commands, encoding), commands, encoding

    def lean_reject(self, request, expected_substring):
        accepted, reason, cert = g.gemmini_program_check(request)
        self.assertFalse(accepted, "tampered request unexpectedly accepted")
        self.assertIn(expected_substring, reason)
        self.assertIsNone(cert)
        return reason

    # -- metadata tampering ------------------------------------------------

    def test_size_metadata_mismatch_rejected(self):
        plan = g.make_plan(16, 16, 16, g.SCHEDULE_BASELINE)
        request, _, encoding = self.build_request(plan)
        bad = dict(encoding, sizes=dict(encoding["sizes"], A=encoding["sizes"]["A"] * 2))
        tampered = dict(request, encoding=bad)
        self.lean_reject(tampered, "buffer size metadata does not match plan")

    def test_base_alias_metadata_rejected(self):
        plan = g.make_plan(16, 16, 16, g.SCHEDULE_BASELINE)
        request, _, encoding = self.build_request(plan)
        # Claim C aliases the A buffer; bytes still reference the true bases.
        bad = dict(encoding, bases=dict(encoding["bases"], C=encoding["bases"]["A"]))
        tampered = dict(request, encoding=bad)
        self.lean_reject(tampered, "executable rejected")

    # -- byte-level tampering with valid metadata ---------------------------

    def test_funct_byte_mutation_rejected(self):
        plan = g.make_plan(16, 16, 16, g.SCHEDULE_BASELINE)
        request, _, encoding = self.build_request(plan)
        raw = hex_to_bytes(encoding["bytes_hex"])
        # First packet = config_ex, funct 0: word 16 of the packet carries
        # funct in bits 25-31 (byte 3, bit 1 upward). Flip funct bit -> funct 1.
        idx = 16 * 4 + 3
        raw[idx] ^= 0x02
        bad = dict(encoding, bytes_hex=bytes_to_hex(raw))
        self.lean_reject(dict(request, encoding=bad), "executable rejected")

    def test_opcode_byte_mutation_rejected(self):
        plan = g.make_plan(16, 16, 16, g.SCHEDULE_BASELINE)
        request, _, encoding = self.build_request(plan)
        raw = hex_to_bytes(encoding["bytes_hex"])
        # Word 0 of packet 2 (packet index 1) is an rs1 LUI; byte 0 holds the
        # opcode bits. Any change makes the instruction stream non-decodable
        # or differently-decoded while metadata stays valid.
        idx = 17 * 4 + 0
        raw[idx] ^= 0x01
        bad = dict(encoding, bytes_hex=bytes_to_hex(raw))
        self.lean_reject(dict(request, encoding=bad), "executable rejected")

    # -- command/byte disagreement ------------------------------------------

    def test_command_stream_disagreement_rejected(self):
        plan_a = g.make_plan(16, 16, 16, g.SCHEDULE_BASELINE)
        plan_b = g.make_plan(16, 16, 16, g.SCHEDULE_REUSE_B)
        req_a, _, enc_a = self.build_request(plan_a)
        commands_b = g.gen_commands(plan_b)
        # Bytes from baseline, commands claimed from reuse_b.
        mixed = g.build_program_request(plan_b, commands_b, enc_a)
        self.lean_reject(mixed, "executable rejected")

    def test_single_command_field_disagreement_rejected(self):
        plan = g.make_plan(16, 16, 16, g.SCHEDULE_BASELINE)
        request, commands, encoding = self.build_request(plan)
        tampered_cmds = [dict(c) for c in request["commands"]]
        for c in tampered_cmds:
            if c["kind"] == "mvin":
                c["offset"] = c["offset"] + 16
                break
        else:
            self.fail("no mvin command found")
        mixed = dict(request, commands=tampered_cmds)
        self.lean_reject(mixed, "instruction execution rejected")

    # -- stale checker response ---------------------------------------------

    def test_stale_checker_response_cannot_verify(self):
        plan_a = g.make_plan(16, 16, 16, g.SCHEDULE_BASELINE)
        req_a, _, _ = self.build_request(plan_a)
        proc = g.gemmini_program_check(req_a)
        self.assertTrue(proc[0], proc[1])
        STALE_RESPONSE.write_text(
            json.dumps({"accepted": True, "reason": "accepted",
                        "program": {"plan": req_a["plan"],
                                    "commands": req_a["commands"],
                                    "encoding": req_a["encoding"]},
                        "certificate": proc[2]}))
        STALE_CHECKER.write_text("#!/bin/sh\ncat > /dev/null\ncat %s\n"
                                 % STALE_RESPONSE.resolve())
        STALE_CHECKER.chmod(STALE_CHECKER.stat().st_mode | stat.S_IEXEC)
        old = os.environ.get("VERITAC_GEMMINI_PROGRAM_CHECK_BIN")
        os.environ["VERITAC_GEMMINI_PROGRAM_CHECK_BIN"] = str(STALE_CHECKER.resolve())
        try:
            plan_b = g.make_plan(32, 16, 16, g.SCHEDULE_REUSE_B)
            req_b, _, _ = self.build_request(plan_b)
            accepted, reason, cert = g.gemmini_program_check(req_b)
            self.assertFalse(accepted)
            self.assertIn("echoed different plan/commands/bytes", reason)
        finally:
            if old is None:
                del os.environ["VERITAC_GEMMINI_PROGRAM_CHECK_BIN"]
            else:
                os.environ["VERITAC_GEMMINI_PROGRAM_CHECK_BIN"] = old

    # -- fake checker cannot bypass the kernel certificate -------------------

    def test_fake_checker_cannot_yield_verified_artifact(self):
        plan = g.make_plan(32, 16, 16, g.SCHEDULE_REUSE_B)
        commands = g.gen_commands(plan)
        encoding, why = g.emit_encoding(plan, commands)
        self.assertIsNotNone(encoding, why)
        raw = hex_to_bytes(encoding["bytes_hex"])
        raw[16 * 4 + 3] ^= 0x02  # semantic packet mutation, metadata valid
        bad = dict(encoding, bytes_hex=bytes_to_hex(raw))
        FAKE_CHECKER.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "req = json.load(sys.stdin)\n"
            "print(json.dumps({'accepted': True, 'reason': 'accepted',"
            " 'program': {'plan': req['plan'], 'commands': req['commands'],"
            " 'encoding': req['encoding']},"
            " 'certificate': {'program_checker': 'fake'}}))\n")
        FAKE_CHECKER.chmod(FAKE_CHECKER.stat().st_mode | stat.S_IEXEC)
        # Compromise the encoding backend too: emit_program_artifacts must
        # still fail because the kernel certificate decides the real bytes.
        import unittest.mock
        orig_emit = g.emit_encoding
        tampered = dict(encoding, bytes_hex=bytes_to_hex(raw))

        def tampered_emit(plan_, commands_, bases=None):
            return tampered, "ok"
        old = os.environ.get("VERITAC_GEMMINI_PROGRAM_CHECK_BIN")
        os.environ["VERITAC_GEMMINI_PROGRAM_CHECK_BIN"] = str(FAKE_CHECKER.resolve())
        try:
            with unittest.mock.patch.object(g, "emit_encoding", tampered_emit):
                out_dir = AUDIT_DIR / "fake_checker_out"
                result = g.emit_program_artifacts(plan, str(out_dir))
        finally:
            if old is None:
                del os.environ["VERITAC_GEMMINI_PROGRAM_CHECK_BIN"]
            else:
                os.environ["VERITAC_GEMMINI_PROGRAM_CHECK_BIN"] = old
        # The fake Lean response was accepted, but the kernel certificate
        # re-decides the real bytes and must reject them.
        self.assertFalse(result["program_accepted"])
        self.assertFalse(result["verified"])
        self.assertEqual(result["program_reason"], "Lean certificate rejected")
        kc = result["kernel_certificate"]
        self.assertFalse(kc["accepted"])
        log = Path(kc["log"]).read_text()
        # decide proved the mutated bytes do NOT satisfy checkExecutable; the
        # sorryAx mention is the failed certificate's rejection evidence.
        self.assertIn("is false", log)
        self.assertIn("depends on axioms: [propext, sorryAx]", log)
        # The emitted .bin is exactly the (mutated) certificate code.
        binary = Path(kc["binary"]).read_bytes()
        self.assertEqual(binary, bytes.fromhex(bad["bytes_hex"]))
        self.assertEqual(kc["binary_sha256"],
                         __import__("hashlib").sha256(binary).hexdigest())

    # -- genuine acceptance: default winner + baseline fallback --------------

    def assert_genuine_acceptance(self, plan, out_dir):
        result = g.emit_program_artifacts(plan, str(out_dir))
        self.assertTrue(result["program_accepted"], result["program_reason"])
        self.assertTrue(result["verified"])
        self.assertEqual(result["program_reason"],
                         "kernel-checked concrete executable certificate")
        kc = result["kernel_certificate"]
        self.assertTrue(kc["accepted"])
        proof = Path(kc["proof"]).read_text()
        self.assertIn("checkExecutable_sound plan program bases code accepted",
                      proof)
        for checked_link in (
            "Symbolic.check plan program = true := by decide +kernel",
            "Encoding.executeBytes code = .ok packets := by decide +kernel",
            "commandsOf bases packets = some program := by decide +kernel",
            "checkExecutable plan program bases code = true := by"):
            self.assertIn(checked_link, proof)
        log = Path(kc["log"]).read_text()
        self.assertIn("'SubmittedKernel.correct' depends on axioms:", log)
        self.assertNotIn("sorryAx", log)
        self.assertNotIn("Lean.ofReduceBool", log)
        # .bin is exactly the accepted bytes.
        binary = Path(kc["binary"]).read_bytes()
        self.assertEqual(binary, bytes.fromhex(result["encoding"]["bytes_hex"]))
        self.assertEqual(kc["binary_sha256"],
                         __import__("hashlib").sha256(binary).hexdigest())
        # Independent cross-check: run the actual machine bytes with the
        # untrusted-RV64 interpreter and compare packets to the pinned macros.
        from tests.test_gemmini_machine_bytes import (
            execute_register_program, expected_packets)
        from CodeGen.gemmini_encoding import command_dict
        packets = execute_register_program(binary)
        expected = expected_packets(
            [command_dict(c) for c in g.gen_commands(plan)],
            result["encoding"]["bases"])
        self.assertEqual(packets, expected)
        return result

    def test_default32x16x16_reuse_b_certificate_passes(self):
        plan = g.make_plan(32, 16, 16, g.SCHEDULE_REUSE_B)
        self.assert_genuine_acceptance(plan, AUDIT_DIR / "default32_reuse_b")

    def test_baseline_fallback_certificate_passes(self):
        plan = g.make_plan(16, 16, 16, g.SCHEDULE_BASELINE)
        self.assert_genuine_acceptance(plan, AUDIT_DIR / "baseline16_fallback")


if __name__ == "__main__":
    unittest.main(verbosity=2)
