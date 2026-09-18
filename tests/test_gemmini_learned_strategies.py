import unittest
import tempfile
from specializations.gemmini_gemm import backend as g
from specializations.gemmini_gemm.learned_strategies import gen_batched_reuse_a, gen_resident_a, certify_resident_a
from specializations.gemmini_gemm.learned_strategies import gen_resident_a_batched_rows
from specializations.gemmini_gemm.learned_strategies import gen_serpentine_b_cache
from specializations.gemmini_gemm.learned_strategies import select_learned_strategy


class LearnedScheduleTests(unittest.TestCase):
    def test_offline_selection_reuses_discoveries_without_model(self):
        for shape, spad, acc, expected in (
                ((80, 16, 32), 32, 32, 4096),
                ((16, 80, 32), 32, 32, 4096),
                ((16, 80, 32), 80, 16, 3072),
                ((32, 48, 48), 64, 32, 5376),
                ((48, 32, 32), 48, 16, 3584)):
            plan = g.make_plan(*shape, "baseline")
            plan.update(scratchpad_rows=spad, accumulator_rows=acc)
            commands, metadata = select_learned_strategy(plan)
            self.assertEqual(g.modeled_counts(commands)["dma_in_elems_int8"], expected)
            selected = next(c for c in metadata["candidates"] if c["name"] == metadata["winner"])
            self.assertTrue(selected["native_accepted"])
            self.assertEqual(metadata["global_optimality"], "not claimed")

    def test_serpentine_turns_rotate_storage_safely(self):
        for shape in ((48, 32, 32), (32, 48, 48), (48, 64, 16), (16, 48, 32), (32, 16, 32)):
            plan = g.make_plan(*shape, "baseline")
            plan.update(scratchpad_rows=shape[2] + 16, accumulator_rows=16)
            commands = gen_serpentine_b_cache(plan)
            ok, reason, _ = g.check_program(plan, commands)
            self.assertTrue(ok, reason)
            for mode in ("lcg", "minmax"):
                a, b = g.deterministic_inputs(*shape, 811, mode)
                actual, _ = g.run_commands(commands, plan, a, b)
                self.assertEqual(actual, g.scalar_matmul(a, b, *shape))
            if shape == (48, 32, 32):
                self.assertEqual(g.modeled_counts(commands)["dma_in_elems_int8"], 3584)
                self.assertEqual(len(commands), 49)

    def test_serpentine_insufficient_scratchpad_is_rejected(self):
        plan = g.make_plan(48, 32, 32, "baseline")
        plan.update(scratchpad_rows=32, accumulator_rows=16)
        self.assertFalse(g.check_program(plan, gen_serpentine_b_cache(plan))[0])

    def test_shared_slot_preserves_latched_b_after_overwrite(self):
        for shape in ((32, 48, 48), (48, 64, 32), (16, 48, 16)):
            plan = g.make_plan(*shape, "baseline")
            plan.update(scratchpad_rows=shape[2] + 16, accumulator_rows=32)
            commands = gen_resident_a_batched_rows(plan, 2)
            ok, reason, _ = g.check_program(plan, commands)
            self.assertTrue(ok, reason)
            a, b = g.deterministic_inputs(*shape, 611)
            actual, _ = g.run_commands(commands, plan, a, b)
            self.assertEqual(actual, g.scalar_matmul(a, b, *shape))
            if shape == (32, 48, 48):
                counts = g.modeled_counts(commands)
                self.assertEqual(counts["dma_in_elems_int8"], 5376)
                self.assertEqual(counts["compute_accumulated"], 9)

    def test_shared_slot_requires_both_resource_bounds(self):
        for spad, acc in ((48, 32), (64, 16)):
            plan = g.make_plan(32, 48, 48, "baseline")
            plan.update(scratchpad_rows=spad, accumulator_rows=acc)
            self.assertFalse(g.check_program(plan, gen_resident_a_batched_rows(plan, 2))[0])

    def test_existing_certificate_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "overwrite"):
                certify_resident_a(g.make_plan(16, 16, 16, "baseline"), folder)

    def test_resident_inputs_with_single_accumulator(self):
        for shape in ((16, 80, 32), (32, 48, 48), (32, 48, 16)):
            plan = g.make_plan(*shape, "baseline")
            plan.update(scratchpad_rows=shape[2] + 16, accumulator_rows=16)
            commands = gen_resident_a(plan)
            ok, reason, _ = g.check_program(plan, commands)
            self.assertTrue(ok, reason)
            counts = g.modeled_counts(commands)
            self.assertEqual(counts["mvin_A"], shape[0] * shape[2] // 256)
            a, b = g.deterministic_inputs(*shape, 711)
            actual, _ = g.run_commands(commands, plan, a, b)
            self.assertEqual(actual, g.scalar_matmul(a, b, *shape))

    def test_resident_input_capacity_is_checked(self):
        plan = g.make_plan(16, 80, 32, "baseline")
        plan.update(scratchpad_rows=32, accumulator_rows=16)
        self.assertFalse(g.check_program(plan, gen_resident_a(plan))[0])

    def test_generalized_shape_and_tail_are_checked(self):
        for shape in ((16, 80, 32), (32, 80, 48), (48, 48, 32), (16, 16, 48)):
            plan = g.make_plan(*shape, "baseline")
            plan["scratchpad_rows"] = 32
            plan["accumulator_rows"] = 32
            commands = gen_batched_reuse_a(plan, 2)
            ok, reason, _ = g.check_program(plan, commands)
            self.assertTrue(ok, reason)
            a, b = g.deterministic_inputs(*shape, 917)
            actual, _ = g.run_commands(commands, plan, a, b)
            self.assertEqual(actual, g.scalar_matmul(a, b, *shape))

    def test_original_discovery_traffic(self):
        plan = g.make_plan(16, 80, 32, "baseline")
        plan["accumulator_rows"] = 32
        counts = g.modeled_counts(gen_batched_reuse_a(plan, 2))
        self.assertEqual((counts["mvin_A"], counts["mvin_B"]), (6, 10))
        self.assertEqual(counts["dma_in_elems_int8"], 4096)

    def test_insufficient_capacity_is_not_silently_fixed(self):
        plan = g.make_plan(16, 80, 32, "baseline")
        plan["accumulator_rows"] = 16
        commands = gen_batched_reuse_a(plan, 2)
        self.assertFalse(g.check_program(plan, commands)[0])
        with self.assertRaises(ValueError):
            gen_batched_reuse_a(plan, True)


if __name__ == "__main__":
    unittest.main()
