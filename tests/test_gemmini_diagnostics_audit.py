"""Independent checks of advisory diagnostics beyond normal template traffic."""
import unittest
from specializations.gemmini_gemm import backend as g
from specializations.gemmini_gemm.diagnostics import diagnose


class DiagnosticAuditTests(unittest.TestCase):
    def test_provenance_tracks_input_buffer_not_operand_position(self):
        plan = g.make_plan(16, 16, 16, "baseline")
        commands = g.gen_commands(plan)
        for command in commands:
            if command.kind == "mvin":
                command.buf = "B" if command.buf == "A" else "A"
        self.assertFalse(g.check_program(plan, commands)[0])
        result = diagnose(plan, commands)
        self.assertEqual(result["status"], "output_obligation_mismatch")
        self.assertTrue(result["advisory_only"])

    def test_advisory_success_does_not_imply_byte_acceptance(self):
        plan = g.make_plan(16, 16, 16, "baseline")
        commands = g.gen_commands(plan)
        for command in commands:
            if command.kind == "mvin":
                command.slot = 1 - command.slot
        # Instruction semantics allow this, but the current byte decoder fixes
        # A to slot 0 and B to slot 1. Diagnostics do not certify that boundary.
        self.assertFalse(g.check_program(plan, commands)[0])
        result = diagnose(plan, commands)
        self.assertEqual(result["status"], "no_diagnosed_failure")
        self.assertTrue(result["advisory_only"])


if __name__ == "__main__":
    unittest.main()
