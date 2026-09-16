"""General validation must accept unseen shapes and programs, not a whitelist."""
import copy
import unittest
from CodeGen import gemmini as g, gemmini_encoding as e, gemmini_lowering as lowering


def check(plan, commands):
    encoded = e.emit_encoding(plan, commands)
    return g.gemmini_program_check(g.build_program_request(plan, commands, encoded))


class GeneralProgramTests(unittest.TestCase):
    def test_unseen_shapes_both_schedules(self):
        for shape in ((32, 16, 32), (16, 32, 48), (48, 32, 64), (64, 64, 64)):
            for schedule in ("baseline", "reuse_b"):
                with self.subTest(shape=shape, schedule=schedule):
                    plan = g.make_plan(*shape, schedule)
                    ok, why, _ = check(plan, g.gen_commands(plan))
                    self.assertTrue(ok, why)

    def test_independent_output_tiles_can_be_reversed(self):
        plan = g.make_plan(32, 16, 32, "baseline")
        commands = g.gen_commands(plan)
        # Two nine-command output-tile blocks, after four configurations.
        reversed_tiles = commands[:4] + commands[13:22] + commands[4:13] + commands[-1:]
        self.assertNotEqual(g.serialize_commands(commands), g.serialize_commands(reversed_tiles))
        ok, why, _ = check(plan, reversed_tiles)
        self.assertTrue(ok, why)

    def test_partial_sum_overwrite_is_rejected(self):
        plan = g.make_plan(32, 16, 32, "baseline")
        commands = copy.deepcopy(g.gen_commands(plan))
        carried = next(c for c in commands if c.kind == "preload" and c.out_addr & g.ACCUMULATE_BIT)
        carried.out_addr &= ~g.ACCUMULATE_BIT
        self.assertFalse(check(plan, commands)[0])

    def test_wrong_weight_retention_across_k_rejected(self):
        plan = g.make_plan(32, 16, 32, "reuse_b")
        commands = copy.deepcopy(g.gen_commands(plan))
        for index, c in enumerate(commands):
            if c.kind == "preload" and c.out_addr & g.ACCUMULATE_BIT:
                c.bd_spad_addr = g.GARBAGE_ADDR
                commands[index + 1].accumulated = True
                break
        self.assertFalse(check(plan, commands)[0])

    def test_batched_reuse_handles_partial_last_batch(self):
        plan = g.make_plan(48, 16, 32, "baseline")
        plan["accumulator_rows"] = 32
        commands = lowering.gen_batched_reuse_b(plan, 2)
        ok, why, _ = check(plan, commands)
        self.assertTrue(ok, why)
        self.assertEqual(g.modeled_counts(commands)["mvin_B"], 4)
        for mode in ("zero", "max", "min", "minmax", "lcg"):
            a, b = g.deterministic_inputs(48, 16, 32, 917, mode)
            result, _ = g.run_commands(commands, plan, a, b)
            self.assertEqual(result, g.scalar_matmul(a, b, 48, 16, 32))

    def test_overlarge_batch_is_rejected_by_lean(self):
        plan = g.make_plan(48, 16, 32, "baseline")
        plan["accumulator_rows"] = 32
        # The untrusted proposal is permitted; acceptance checks real accesses.
        commands = lowering.gen_batched_reuse_b(plan, 3)
        ok, reason, _ = check(plan, commands)
        self.assertFalse(ok)
        self.assertIn("execution rejected", reason)

    def test_zero_batch_rejected(self):
        with self.assertRaises(ValueError):
            lowering.gen_batched_reuse_b(g.make_plan(16, 16, 16, "baseline"), 0)


if __name__ == "__main__":
    unittest.main()
