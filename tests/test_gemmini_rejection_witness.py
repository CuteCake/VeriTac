import copy
import unittest
from specializations.gemmini_gemm import backend as g
from specializations.gemmini_gemm.rejection_witness import find_sparse_counterexample


class RejectionWitnessTests(unittest.TestCase):
    def test_wrong_input_provenance_has_executed_sparse_witness(self):
        plan = g.make_plan(16, 16, 16, "baseline")
        commands = g.gen_commands(plan)
        for command in commands:
            if command.kind == "mvin" and command.buf == "B":
                command.buf, command.slot = "A", 0
        self.assertFalse(g.check_program(plan, commands)[0])
        result = find_sparse_counterexample(plan, commands)
        self.assertTrue(result["found"], result)
        self.assertNotEqual(result["actual"], result["expected"])
        self.assertLessEqual(len(result["sparse_inputs"]["ones"]), 2)

    def test_correct_program_has_no_counterexample(self):
        plan = g.make_plan(16, 16, 16, "baseline")
        self.assertFalse(find_sparse_counterexample(plan, g.gen_commands(plan))["found"])

    def test_order_only_rejection_is_not_called_numerically_wrong(self):
        plan = g.make_plan(16, 16, 32, "baseline")
        original = g.gen_commands(plan)
        commands = copy.deepcopy(original[:4] + original[8:12] + original[4:8] + original[-2:])
        commands[6].out_addr = 0xA0000000
        commands[10].out_addr = 0xE0000000
        self.assertFalse(g.check_program(plan, commands)[0])
        result = find_sparse_counterexample(plan, commands)
        self.assertFalse(result["found"], result)
        self.assertIn("coefficients agree", result["reason"])
        a, b = g.deterministic_inputs(16, 16, 32, 912)
        actual, _ = g.run_commands(commands, plan, a, b)
        self.assertEqual(actual, g.scalar_matmul(a, b, 16, 16, 32))


if __name__ == "__main__":
    unittest.main()
