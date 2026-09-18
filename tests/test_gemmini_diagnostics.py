"""Differential tests: advisory Gemmini diagnostics vs the Lean checkers.

The Lean binaries are authoritative.  For every case below, the existing
Lean checker's accepted/reason output is compared with the advisory
report from specializations/gemmini_gemm/diagnostics.diagnose:

  * Lean accepts            =>  diagnose must report no_diagnosed_failure
                                (which is explicitly NOT acceptance).
  * Lean rejects at the     =>  diagnose must report a failure status
    plan/symbolic level         and never no_diagnosed_failure.

The advisory module must never claim acceptance and always carry
advisory_only: true.
"""
import copy
import json
import random
import unittest

from specializations.gemmini_gemm import backend as g
from specializations.gemmini_gemm import diagnostics as d
from specializations.gemmini_gemm.diagnostics import diagnose

PLAN = {"m": 32, "n": 32, "k": 32, "dim": 16, "scratchpad_rows": 32,
        "accumulator_rows": 32, "schedule": "reuse_b"}
BASELINE_PLAN = dict(PLAN, schedule="baseline")

CMDS = g.serialize_commands(g.gen_commands(PLAN))
ENCODING = g.emit_encoding(PLAN, g.gen_commands(PLAN))[0]


def find(pred):
    for i, c in enumerate(CMDS):
        if pred(c):
            return i
    raise AssertionError("no matching command")


IDX_A_MVIN = find(lambda c: c["kind"] == "mvin" and c["slot"] == 0)
IDX_B_MVIN = find(lambda c: c["kind"] == "mvin" and c["slot"] == 1)
IDX_PRELOAD = find(lambda c: c["kind"] == "preload")
IDX_COMPUTE = find(lambda c: c["kind"] == "compute")
IDX_MVOUT = find(lambda c: c["kind"] == "mvout")


def lean_program(plan, commands):
    """Authoritative Lean program acceptance on the given concrete stream."""
    request = g.build_program_request(plan, commands, ENCODING)
    return g.gemmini_program_check(request)


def lean_plan(plan):
    """Authoritative Lean plan legality."""
    return g.gemmini_check(plan)


def mutate(index, **changes):
    commands = copy.deepcopy(CMDS)
    commands[index].update(changes)
    return commands


class AdvisoryContractTests(unittest.TestCase):
    def test_report_is_json_and_advisory(self):
        for report in (diagnose(PLAN, CMDS), diagnose({}, []),
                       diagnose(PLAN, [{"kind": "bogus"}]),
                       diagnose(dict(PLAN, m=0), CMDS)):
            self.assertTrue(report["advisory_only"])
            self.assertNotIn("accepted", report)
            json.dumps(report)

    def test_no_failure_is_not_acceptance(self):
        report = diagnose(PLAN, CMDS)
        self.assertEqual(report["status"], d.STATUS_NO_FAILURE)
        self.assertIn("NOT acceptance", report["note"])

    def test_status_vocabulary(self):
        statuses = {d.STATUS_PLAN_REJECTED, d.STATUS_SYNTAX_UNSUPPORTED,
                    d.STATUS_EXECUTION_REJECTED, d.STATUS_COMPLETION_MISSING,
                    d.STATUS_OUTPUT_MISMATCH, d.STATUS_NO_FAILURE,
                    d.STATUS_INCONCLUSIVE}
        self.assertEqual(len(statuses), 7)


class AgreementAcceptedTests(unittest.TestCase):
    """Lean accepts => advisory reports no_diagnosed_failure."""

    def test_reuse_b_accepted_by_lean(self):
        accepted, reason, _ = g.check_program(PLAN, g.gen_commands(PLAN))
        self.assertTrue(accepted, reason)
        self.assertEqual(diagnose(PLAN, CMDS)["status"], d.STATUS_NO_FAILURE)

    def test_baseline_accepted_by_lean(self):
        commands = g.serialize_commands(g.gen_commands(BASELINE_PLAN))
        accepted, reason, _ = g.check_program(BASELINE_PLAN, commands)
        self.assertTrue(accepted, reason)
        self.assertEqual(diagnose(BASELINE_PLAN, commands)["status"],
                         d.STATUS_NO_FAILURE)

    def test_lean_acceptance_and_advisory_agree(self):
        accepted, reason, _ = g.check_program(PLAN, g.gen_commands(PLAN))
        self.assertTrue(accepted, reason)
        report = diagnose(PLAN, CMDS)
        self.assertEqual(report["status"], d.STATUS_NO_FAILURE)
        self.assertTrue(report["advisory_only"])


class DifferentialMutationTests(unittest.TestCase):
    """Lean rejects the mutated stream => advisory reports a failure with
    the instruction index/kind, and never no_diagnosed_failure."""

    def assert_rejected(self, commands, status, index=None, kind=None):
        accepted, reason, _ = lean_program(PLAN, commands)
        self.assertFalse(accepted, "Lean unexpectedly accepted: %s" % reason)
        report = diagnose(PLAN, commands)
        self.assertEqual(report["status"], status,
                         (report, reason))
        self.assertTrue(report["advisory_only"])
        if index is not None or kind is not None:
            self.assertEqual(report["instruction"]["index"], index)
            self.assertEqual(report["instruction"]["kind"], kind)
        if status == d.STATUS_EXECUTION_REJECTED:
            self.assertIn("failed_obligation", report)
            self.assertIsInstance(report["facts"], dict)
        return report

    def mutate(self, index, **changes):
        commands = copy.deepcopy(CMDS)
        commands[index].update(changes)
        return commands

    def drop(self, index):
        commands = copy.deepcopy(CMDS)
        del commands[index]
        return commands

    def test_config_ex_not_weight_stationary(self):
        self.assert_rejected(mutate(0, dataflow=0),
                             d.STATUS_EXECUTION_REJECTED, 0, "config_ex")

    def test_config_st_zero_stride(self):
        self.assert_rejected(mutate(1, stride_bytes=0),
                             d.STATUS_EXECUTION_REJECTED, 1, "config_st")

    def test_config_st_stride_not_multiple_of_four(self):
        self.assert_rejected(mutate(1, stride_bytes=5),
                             d.STATUS_EXECUTION_REJECTED, 1, "config_st")

    def test_config_ld_slot_two_parses_but_execution_rejects(self):
        self.assert_rejected(mutate(2, slot=2),
                             d.STATUS_EXECUTION_REJECTED, 2, "config_ld")

    def test_mvin_before_positive_slot_stride(self):
        self.assert_rejected(mutate(2, stride_bytes=0),
                             d.STATUS_EXECUTION_REJECTED, 5, "mvin")

    def test_mvin_partial_tile_cols(self):
        self.assert_rejected(mutate(IDX_A_MVIN, cols=15),
                             d.STATUS_EXECUTION_REJECTED, 5, "mvin")

    def test_mvin_unaligned_scratchpad_row(self):
        self.assert_rejected(mutate(IDX_A_MVIN, spad_addr=3),
                             d.STATUS_EXECUTION_REJECTED, 5, "mvin")

    def test_mvin_host_range(self):
        self.assert_rejected(mutate(IDX_A_MVIN, offset=10 ** 6),
                             d.STATUS_EXECUTION_REJECTED, 5, "mvin")

    def test_preload_output_address_missing_bits(self):
        self.assert_rejected(mutate(IDX_PRELOAD, out_addr=0),
                             d.STATUS_EXECUTION_REJECTED, 6, "preload")

    def test_preload_output_address_missing_bit29(self):
        self.assert_rejected(mutate(IDX_PRELOAD, out_addr=2 ** 31),
                             d.STATUS_EXECUTION_REJECTED, 6, "preload")

    def test_preload_bd_unaligned(self):
        self.assert_rejected(mutate(IDX_PRELOAD, bd_spad_addr=1),
                             d.STATUS_EXECUTION_REJECTED, 6, "preload")

    def test_compute_bd_not_garbage(self):
        self.assert_rejected(mutate(IDX_COMPUTE, bd_spad_addr=16),
                             d.STATUS_EXECUTION_REJECTED, 7, "compute")

    def test_compute_accumulated_without_retained_weights(self):
        self.assert_rejected(mutate(IDX_COMPUTE, accumulated=True),
                             d.STATUS_EXECUTION_REJECTED, 7, "compute")

    def test_compute_without_preload(self):
        self.assert_rejected(self.drop(IDX_PRELOAD),
                             d.STATUS_EXECUTION_REJECTED, IDX_COMPUTE - 1, "compute")

    def test_mvout_source_address_missing_bits(self):
        self.assert_rejected(mutate(IDX_MVOUT, acc_addr=0),
                             d.STATUS_EXECUTION_REJECTED, IDX_MVOUT, "mvout")

    def test_mvout_host_range(self):
        self.assert_rejected(mutate(IDX_MVOUT, buf_offset=10 ** 6),
                             d.STATUS_EXECUTION_REJECTED, IDX_MVOUT, "mvout")

    def test_missing_final_fence_is_completion_failure(self):
        commands = copy.deepcopy(CMDS)
        del commands[-1]
        accepted, reason, _ = lean_program(PLAN, commands)
        self.assertFalse(accepted)
        report = self.assert_rejected(commands, d.STATUS_COMPLETION_MISSING)
        self.assertIs(report["facts"]["drained"], False)

    def test_missing_mvout_leaves_output_uncovered(self):
        commands = copy.deepcopy(CMDS)
        del commands[IDX_MVOUT]
        accepted, reason, _ = lean_program(PLAN, commands)
        self.assertFalse(accepted)
        self.assertIn("output obligation", reason)
        report = self.assert_rejected(commands, d.STATUS_OUTPUT_MISMATCH)
        self.assertIs(report["facts"]["written"], False)

    def test_wrong_b_tile_is_output_obligation_mismatch(self):
        commands = copy.deepcopy(CMDS)
        commands[IDX_B_MVIN]["offset"] = 16  # in range, wrong tile
        accepted, reason, _ = lean_program(PLAN, commands)
        self.assertFalse(accepted)
        self.assertIn("output obligation", reason)
        report = self.assert_rejected(commands, d.STATUS_OUTPUT_MISMATCH)
        self.assertIs(report["facts"]["written"], True)
        self.assertIs(report["facts"]["product_sequence_matches"], False)


class SyntaxDifferentialTests(unittest.TestCase):
    """Parser-level mutations: Lean rejects the request and the advisory
    module must classify the stream as syntax_unsupported."""

    def assert_syntax(self, commands):
        accepted, reason, _ = lean_program(PLAN, commands)
        self.assertFalse(accepted)
        report = diagnose(PLAN, commands)
        self.assertEqual(report["status"], d.STATUS_SYNTAX_UNSUPPORTED,
                         (report, reason))
        self.assertTrue(report["advisory_only"])

    def test_unknown_kind(self):
        self.assert_syntax([{"kind": "mvini", "slot": 0, "buf": "A",
                             "offset": 0, "spad_addr": 0, "cols": 16,
                             "rows": 16}] + CMDS[1:])

    def test_missing_field(self):
        commands = copy.deepcopy(CMDS)
        del commands[5]["cols"]
        self.assert_syntax(commands)

    def test_extra_field(self):
        commands = copy.deepcopy(CMDS)
        commands[5]["junk"] = 1
        self.assert_syntax(commands)

    def test_negative_operand(self):
        self.assert_syntax(mutate(IDX_A_MVIN, offset=-1))

    def test_noninteger_operand(self):
        self.assert_syntax(mutate(IDX_A_MVIN, cols=16.5))

    def test_boolean_operand_rejected(self):
        self.assert_syntax(mutate(IDX_A_MVIN, cols=True))

    def test_unknown_buffer(self):
        self.assert_syntax(mutate(IDX_A_MVIN, buf="C"))

    def test_mvin_slot_out_of_range(self):
        self.assert_syntax(mutate(IDX_A_MVIN, slot=2))

    def test_config_ld_slot_out_of_range(self):
        self.assert_syntax(mutate(2, slot=3))

    def test_non_identity_scale(self):
        self.assert_syntax(mutate(2, scale=2.0))

    def test_commands_not_a_list(self):
        report = diagnose(PLAN, {"kind": "fence"})
        self.assertEqual(report["status"], d.STATUS_SYNTAX_UNSUPPORTED)


class PlanDifferentialTests(unittest.TestCase):
    """Plan-level differential against the authoritative gemmini_check."""

    def test_accumulator_capacity(self):
        bad = dict(PLAN, accumulator_rows=31)
        accepted, reason, _ = lean_plan(bad)
        self.assertFalse(accepted)
        report = diagnose(bad, CMDS)
        self.assertEqual(report["status"], d.STATUS_PLAN_REJECTED)
        self.assertEqual(report["facts"]["plan_diagnostic"], reason)

    def test_scratchpad_capacity(self):
        bad = dict(PLAN, scratchpad_rows=31)
        accepted, reason, _ = lean_plan(bad)
        self.assertFalse(accepted)
        report = diagnose(bad, CMDS)
        self.assertEqual(report["status"], d.STATUS_PLAN_REJECTED)
        self.assertEqual(report["facts"]["plan_diagnostic"], reason)

    def test_dim_other_than_sixteen_rejected_by_instruction_model(self):
        # The plan itself is legal, but the instruction model requires DIM=16.
        bad = dict(PLAN, dim=8, scratchpad_rows=16, accumulator_rows=32)
        accepted, reason, _ = lean_plan(bad)
        self.assertTrue(accepted)  # plan checker alone accepts it
        accepted, reason, _ = lean_program(bad, CMDS)
        self.assertFalse(accepted)
        self.assertIn("DIM=16", reason)
        report = diagnose(bad, CMDS)
        self.assertEqual(report["status"], d.STATUS_PLAN_REJECTED)
        self.assertIn("DIM=16", report["summary"])

    def test_malformed_plan(self):
        bad = dict(PLAN, junk=1)
        accepted, reason, _ = lean_plan(bad)
        self.assertFalse(accepted)
        self.assertEqual(diagnose(bad, CMDS)["status"],
                         d.STATUS_PLAN_REJECTED)

    def test_noninteger_plan_field(self):
        bad = dict(PLAN, m=32.5)
        accepted, _, _ = lean_plan(bad)
        self.assertFalse(accepted)
        self.assertEqual(diagnose(bad, CMDS)["status"], d.STATUS_PLAN_REJECTED)

    def test_integral_decimal_plan_field_is_strictly_an_integer(self):
        good = dict(PLAN, m=32.0)
        accepted, reason, _ = lean_plan(good)
        self.assertTrue(accepted, reason)
        self.assertEqual(diagnose(good, CMDS)["status"], d.STATUS_NO_FAILURE)

    def test_divisibility(self):
        bad = dict(PLAN, k=33)
        accepted, reason, _ = lean_plan(bad)
        self.assertFalse(accepted)
        report = diagnose(bad, CMDS)
        self.assertEqual(report["facts"]["plan_diagnostic"], reason)


class RandomizedDifferentialTests(unittest.TestCase):
    """Seeded random single-field mutations: the advisory module and the
    authoritative Lean checker must agree at the plan/symbolic level.

    Byte-level rejections ("executable rejected") are outside the advisory
    module's scope: it sees no encoding artifact, so a stream that is
    symbolically valid but mismatched against the untouched bytes may be
    reported as no_diagnosed_failure while Lean rejects the full artifact.
    That carve-out is the only permitted disagreement.
    """

    FAILURE_STATUSES = {d.STATUS_EXECUTION_REJECTED, d.STATUS_COMPLETION_MISSING,
                        d.STATUS_OUTPUT_MISMATCH}

    def setUp(self):
        self.rng = random.Random(20260917)

    def random_commands(self):
        commands = copy.deepcopy(CMDS)
        action = self.rng.random()
        if action < 0.08:
            del commands[-1]                       # drop final fence
            return commands
        if action < 0.14:
            pos = self.rng.randrange(len(commands))
            commands.insert(pos, {"kind": "fence"})
            return commands
        if action < 0.18:
            pos = self.rng.randrange(len(commands))
            commands[pos] = {"kind": "mvini", "slot": 0, "buf": "A",
                             "offset": 0, "spad_addr": 0, "cols": 16,
                             "rows": 16}
            return commands
        pos = self.rng.randrange(len(commands) - 1)  # keep final fence
        command = commands[pos]
        field = self.rng.choice([f for f in command if f != "kind"])
        if field == "buf":
            command[field] = self.rng.choice(["A", "B", "C"])
        elif field == "slot":
            command[field] = self.rng.choice([0, 1, 2, 3, -1])
        elif field == "accumulated":
            command[field] = not command[field]
        elif field == "scale":
            command[field] = self.rng.choice([1.0, 2.0, 0.5, "1"])
        else:
            command[field] = self.rng.choice(
                [0, 1, 15, 17, 16, 2 ** 31, 2 ** 32, 10 ** 6,
                 command[field] + self.rng.choice([-40, -1, 1, 40])])
        return commands

    def test_fuzz_agreement(self):
        for _ in range(200):
            commands = self.random_commands()
            accepted, reason, _ = lean_program(PLAN, commands)
            report = diagnose(PLAN, commands)
            self.assertTrue(report["advisory_only"])
            self.assertNotIn("accepted", report)
            if accepted:
                self.assertEqual(report["status"], d.STATUS_NO_FAILURE,
                                 (report, reason, commands[pos] if False else ""))
            elif reason.startswith("executable rejected"):
                continue  # byte-level artifact check, outside advisory scope
            elif reason.startswith("program rejected:"):
                self.assertIn(report["status"], self.FAILURE_STATUSES,
                              (report, reason))
            elif "malformed commands" in reason or "unknown" in reason:
                self.assertEqual(report["status"], d.STATUS_SYNTAX_UNSUPPORTED,
                                 (report, reason))
            elif "malformed plan" in reason:
                self.assertEqual(report["status"], d.STATUS_PLAN_REJECTED,
                                 (report, reason))
            else:
                # plan legality diagnostic string
                self.assertEqual(report["status"], d.STATUS_PLAN_REJECTED,
                                 (report, reason))
                self.assertEqual(report["facts"]["plan_diagnostic"], reason)


if __name__ == "__main__":
    unittest.main()
