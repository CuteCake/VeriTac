import unittest
from specializations.gemmini_gemm import backend as g
from specializations.gemmini_gemm.evaluation import compare_controls
from specializations.gemmini_gemm.lowering import gen_batched_reuse_b


class ControlsTests(unittest.TestCase):
    def test_input_identity_is_not_dma_slot(self):
        counts = g.modeled_counts([g.Mvin(1, "A", 0, 0, 16, 16),
                                   g.Mvin(0, "B", 0, 16, 16, 16),
                                   g.Mvin(0, "B", 0, 16, 16, 16)])
        self.assertEqual(counts["mvin_A"], 1)
        self.assertEqual(counts["mvin_B"], 2)
        self.assertEqual(counts["dma_in_elems_int8"], 768)

    def test_known_reuse_does_not_count_as_new(self):
        plan = g.make_plan(32, 16, 16, "baseline")
        plan["accumulator_rows"] = 32
        result = compare_controls(plan, gen_batched_reuse_b(plan, 2))
        self.assertGreater(result["input_reduction_vs_baseline"], 0)
        self.assertEqual(result["input_reduction_vs_best_existing"], 0)
        self.assertFalse(result["beats_best_existing_objective"])
        self.assertIn("existing_batched_b_2", result["exact_matches"])
        self.assertIsNone(result["hardware_latency"])

    def test_invalid_candidate_not_scored(self):
        plan = g.make_plan(16, 16, 16, "baseline")
        with self.assertRaisesRegex(ValueError, "candidate failed"):
            compare_controls(plan, [g.Fence()])


if __name__ == "__main__":
    unittest.main()
