"""Tests for CodeGen/gemmini.py (backend-owned; the Lean checker binary and
Spike trace files are covered by controller-owned tests).

Run from the project root:  PYTHONPATH=. python3 -m unittest tests.test_gemmini
"""
import os
import tempfile
import unittest
from unittest import mock

from CodeGen import gemmini as g

BIN_MISSING = "/nonexistent/veritac/gemmini_check"


def checker_available():
    path = g.checker_binary_path()
    return os.path.isfile(path) and os.access(path, os.X_OK)


def run_case(m, n, k, schedule, mode, seed):
    A, B = g.deterministic_inputs(m, n, k, seed, mode)
    plan = g.make_plan(m, n, k, schedule)
    C, stats = g.run_commands(g.gen_commands(plan), plan, A, B)
    return plan, C, stats, g.scalar_matmul(A, B, m, n, k)


class MakePlanCapacityTests(unittest.TestCase):
    def test_minimum_capacity_baseline(self):
        plan = g.make_plan(32, 32, 32, g.SCHEDULE_BASELINE)
        self.assertEqual(plan["dim"], g.DIM)
        self.assertEqual(plan["scratchpad_rows"], 2 * g.DIM)
        self.assertEqual(plan["accumulator_rows"], g.DIM)
        self.assertEqual(g.required_resources(plan),
                         (plan["scratchpad_rows"], plan["accumulator_rows"]))

    def test_minimum_capacity_reuse_b_scales_with_m(self):
        plan = g.make_plan(48, 32, 64, g.SCHEDULE_REUSE_B)
        self.assertEqual(plan["scratchpad_rows"], 2 * g.DIM)
        self.assertEqual(plan["accumulator_rows"], 48)

    def test_minimum_plans_validate(self):
        for schedule in g.SUPPORTED_SCHEDULES:
            for shape in ((16, 16, 16), (32, 32, 32), (48, 32, 64)):
                plan = g.make_plan(*shape, schedule)
                ok, reason = g.validate_plan(plan)
                self.assertTrue(ok, reason)


@unittest.skipUnless(checker_available(), "live gemmini_check binary required")
class DifferentialTests(unittest.TestCase):
    def test_square_16_both_schedules(self):
        for schedule in g.SUPPORTED_SCHEDULES:
            for mode, seed in (("lcg", 7), ("max", 0), ("minmax", 0)):
                with self.subTest(schedule=schedule, mode=mode, seed=seed):
                    _, C, stats, ref = run_case(16, 16, 16, schedule, mode, seed)
                    self.assertEqual(C, ref)
                    self.assertEqual(stats["macs"], 16 * 16 * 16)

    def test_square_32_both_schedules(self):
        for schedule in g.SUPPORTED_SCHEDULES:
            for mode, seed in (("lcg", 1), ("lcg", 987654321), ("max", 0),
                               ("minmax", 0)):
                with self.subTest(schedule=schedule, mode=mode, seed=seed):
                    _, C, stats, ref = run_case(32, 32, 32, schedule, mode, seed)
                    self.assertEqual(C, ref)
                    self.assertEqual(stats["macs"], 32 * 32 * 32)

    def test_rectangular_48x32x64_both_schedules(self):
        for schedule in g.SUPPORTED_SCHEDULES:
            for mode, seed in (("lcg", 42), ("max", 0), ("minmax", 0)):
                with self.subTest(schedule=schedule, mode=mode, seed=seed):
                    _, C, stats, ref = run_case(48, 32, 64, schedule, mode, seed)
                    self.assertEqual(C, ref)
                    self.assertEqual(stats["macs"], 48 * 32 * 64)

    def test_command_structure_counts(self):
        plan = g.make_plan(32, 32, 32, g.SCHEDULE_BASELINE)
        base = g.modeled_counts(g.gen_commands(plan))
        self.assertEqual(base["mvin_A"], 2 * 2 * 2)   # it*jt*kt A tiles
        self.assertEqual(base["mvin_B"], 2 * 2 * 2)   # baseline re-mvins B
        self.assertEqual(base["mvout"], 4)            # it*jt outputs
        self.assertEqual(base["fence"], 1)

        plan = g.make_plan(32, 32, 32, g.SCHEDULE_REUSE_B)
        reuse = g.modeled_counts(g.gen_commands(plan))
        self.assertEqual(reuse["mvin_B"], 2 * 2)      # B once per (j,k)
        self.assertEqual(reuse["mvin_A"], 2 * 2 * 2)
        self.assertEqual(reuse["mvout"], 4)
        self.assertEqual(reuse["fence"], 1)
        # modeled-objective fact only: reuse_b moves fewer B tiles.
        self.assertLess(reuse["mvin_B"], base["mvin_B"])
        self.assertLess(reuse["dma_bytes"], base["dma_bytes"])


class InputValidationTests(unittest.TestCase):
    def test_emitter_index_range_is_bounded(self):
        plan = g.make_plan(1 << 31, 16, 16, g.SCHEDULE_BASELINE)
        ok, reason = g.validate_plan(plan)
        self.assertFalse(ok)
        self.assertIn("indexing", reason)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                g.emit_c(plan, os.path.join(directory, "too_large.c"))

    def test_float_inputs_rejected(self):
        A, B = g.deterministic_inputs(16, 16, 16, 1)
        plan = g.make_plan(16, 16, 16, g.SCHEDULE_BASELINE)
        with self.assertRaises(g.GemminiError):
            g.run_commands(g.gen_commands(plan), plan, [0.5] + A[1:], B)

    def test_bool_inputs_rejected(self):
        A, B = g.deterministic_inputs(16, 16, 16, 1)
        plan = g.make_plan(16, 16, 16, g.SCHEDULE_BASELINE)
        with self.assertRaises(g.GemminiError):
            g.run_commands(g.gen_commands(plan), plan, [True] + A[1:], B)

    def test_out_of_range_inputs_rejected(self):
        A, B = g.deterministic_inputs(16, 16, 16, 1)
        plan = g.make_plan(16, 16, 16, g.SCHEDULE_BASELINE)
        for bad in (128, -129):
            with self.subTest(bad=bad):
                with self.assertRaises(g.GemminiError):
                    g.run_commands(g.gen_commands(plan), plan, [bad] + A[1:], B)

    def test_wrong_buffer_length_rejected(self):
        A, B = g.deterministic_inputs(16, 16, 16, 1)
        plan = g.make_plan(16, 16, 16, g.SCHEDULE_BASELINE)
        with self.assertRaises(g.GemminiError):
            g.run_commands(g.gen_commands(plan), plan, A[:-1], B)

    def test_deterministic_inputs_bad_mode(self):
        with self.assertRaises(ValueError):
            g.deterministic_inputs(16, 16, 16, 1, "nope")

    def test_plan_field_float_and_bool_rejected(self):
        base = g.make_plan(16, 16, 16, g.SCHEDULE_BASELINE)
        for key in ("m", "k", "scratchpad_rows"):
            for bad in (16.0, True):
                with self.subTest(key=key, bad=bad):
                    plan = dict(base)
                    plan[key] = bad
                    ok, _ = g.validate_plan(plan)
                    self.assertFalse(ok)

    def test_plan_out_of_range_rejected(self):
        base = g.make_plan(16, 16, 16, g.SCHEDULE_BASELINE)
        cases = [
            ("dim wrong", dict(dim=8)),
            ("k not divisible", dict(k=33)),
            ("below dim", dict(m=0)),
            ("sp below requirement", dict(scratchpad_rows=g.DIM)),
            ("sp above hardware", dict(scratchpad_rows=g.SCRATCHPAD_ROWS + 1)),
            ("acc above hardware", dict(accumulator_rows=g.ACC_ROWS + 1)),
        ]
        for desc, changes in cases:
            with self.subTest(desc=desc):
                plan = dict(base)
                plan.update(changes)
                ok, _ = g.validate_plan(plan)
                self.assertFalse(ok)


class MutationTests(unittest.TestCase):
    @unittest.skipUnless(checker_available(), "live gemmini_check binary required")
    def test_flipped_compute_accumulated_rejected(self):
        for schedule in g.SUPPORTED_SCHEDULES:
            with self.subTest(schedule=schedule):
                plan = g.make_plan(32, 32, 32, schedule)
                cmds = g.gen_commands(plan)
                compute = next(c for c in cmds if c.kind == "compute")
                self.assertFalse(compute.accumulated)
                compute.accumulated = True
                A, B = g.deterministic_inputs(32, 32, 32, 1)
                with self.assertRaises(g.GemminiError):
                    g.run_commands(cmds, plan, A, B)

    @unittest.skipUnless(checker_available(), "live gemmini_check binary required")
    def test_compute_accumulated_flipped_to_preloaded_rejected(self):
        plan = g.make_plan(32, 32, 32, g.SCHEDULE_REUSE_B)
        cmds = g.gen_commands(plan)
        acc_cmd = next(c for c in cmds if c.kind == "compute" and c.accumulated)
        acc_cmd.accumulated = False   # pending B is GARBAGE in WS subset
        A, B = g.deterministic_inputs(32, 32, 32, 1)
        with self.assertRaises(g.GemminiError):
            g.run_commands(cmds, plan, A, B)

    @unittest.skipUnless(checker_available(), "live gemmini_check binary required")
    def test_missing_mvout_fails_closed(self):
        for schedule in g.SUPPORTED_SCHEDULES:
            with self.subTest(schedule=schedule):
                plan = g.make_plan(32, 32, 32, schedule)
                cmds = [c for c in g.gen_commands(plan) if c.kind != "mvout"]
                A, B = g.deterministic_inputs(32, 32, 32, 1)
                with self.assertRaises(g.GemminiError):
                    g.run_commands(cmds, plan, A, B)

    @unittest.skipUnless(checker_available(), "live gemmini_check binary required")
    def test_one_missing_mvout_leaves_unwritten_outputs(self):
        plan = g.make_plan(32, 32, 32, g.SCHEDULE_BASELINE)
        cmds = g.gen_commands(plan)
        first_mvout = next(i for i, c in enumerate(cmds) if c.kind == "mvout")
        A, B = g.deterministic_inputs(32, 32, 32, 1)
        with self.assertRaises(g.GemminiError):
            g.run_commands(cmds[:first_mvout] + cmds[first_mvout + 1:],
                           plan, A, B)


class FailClosedMockTests(unittest.TestCase):
    def test_missing_checker_binary_rejects_interpretation(self):
        plan = g.make_plan(16, 16, 16, g.SCHEDULE_BASELINE)
        A, B = g.deterministic_inputs(16, 16, 16, 1)
        with mock.patch.object(g, "checker_binary_path", return_value=BIN_MISSING):
            with self.assertRaises(g.GemminiError):
                g.run_commands(g.gen_commands(plan), plan, A, B)

    def test_missing_checker_binary_optimize_has_no_winner(self):
        with mock.patch.object(g, "checker_binary_path", return_value=BIN_MISSING):
            report = g.optimize(16, 16, 16)
        self.assertIsNone(report["winner"])
        self.assertEqual(len(report["candidates"]), 2)
        for cand in report["candidates"]:
            self.assertFalse(cand["check"]["accepted"])

    def test_missing_checker_binary_blocks_emission(self):
        plan = g.make_plan(16, 16, 16, g.SCHEDULE_REUSE_B)
        with mock.patch.object(g, "checker_binary_path", return_value=BIN_MISSING):
            with self.assertRaises(ValueError):
                g.emit_c(plan, os.path.join(tempfile.gettempdir(), "vt_no_check.c"))

    def test_checker_rejection_fails_interpretation_and_emission(self):
        plan = g.make_plan(16, 16, 16, g.SCHEDULE_BASELINE)
        A, B = g.deterministic_inputs(16, 16, 16, 1)
        with mock.patch.object(g, "gemmini_check",
                               return_value=(False, "mocked rejection", None)):
            with self.assertRaises(g.GemminiError):
                g.run_commands(g.gen_commands(plan), plan, A, B)
            with self.assertRaises(ValueError):
                g.emit_c(plan, os.path.join(tempfile.gettempdir(), "vt_mock.c"))


@unittest.skipUnless(checker_available(), "live gemmini_check binary required")
class OptimizeTests(unittest.TestCase):
    def test_optimize_selects_by_modeled_objective_when_both_fit(self):
        report = g.optimize(32, 32, 32)
        self.assertIsNotNone(report["winner"])
        self.assertEqual(report["winner"]["schedule"], g.SCHEDULE_REUSE_B)
        for cand in report["candidates"]:
            self.assertTrue(cand["valid"], cand["invalid_reason"])
            self.assertTrue(cand["check"]["accepted"], cand["check"]["reason"])
            self.assertIsNotNone(cand["objective"])

    def test_optimize_capacity_fallback_to_baseline(self):
        # Accumulator capacity below reuse_b's requirement (m=32 rows) leaves
        # only baseline eligible.
        report = g.optimize(32, 32, 32, scratchpad_rows=2 * g.DIM,
                            accumulator_rows=g.DIM)
        self.assertIsNotNone(report["winner"])
        self.assertEqual(report["winner"]["schedule"], g.SCHEDULE_BASELINE)
        by_sched = {c["plan"]["schedule"]: c for c in report["candidates"]}
        self.assertFalse(by_sched[g.SCHEDULE_REUSE_B]["valid"])
        self.assertTrue(by_sched[g.SCHEDULE_BASELINE]["valid"])
        self.assertTrue(by_sched[g.SCHEDULE_BASELINE]["check"]["accepted"])

    def test_optimize_scratchpad_starvation_rejects_both(self):
        report = g.optimize(32, 32, 32, scratchpad_rows=g.DIM,
                            accumulator_rows=g.DIM)
        self.assertIsNone(report["winner"])
        for cand in report["candidates"]:
            self.assertFalse(cand["valid"])

    def test_winner_plan_round_trips(self):
        winner = g.optimize(48, 32, 64)["winner"]
        A, B = g.deterministic_inputs(48, 32, 64, 5)
        C, _ = g.run_commands(g.gen_commands(winner), winner, A, B)
        self.assertEqual(C, g.scalar_matmul(A, B, 48, 32, 64))


@unittest.skipUnless(checker_available(), "live gemmini_check binary required")
class EmitCTests(unittest.TestCase):
    def test_emission_contains_upstream_macros_and_pass_gate(self):
        for schedule, compute_macro in (
                (g.SCHEDULE_BASELINE, "gemmini_extended_compute_preloaded"),
                (g.SCHEDULE_REUSE_B, "gemmini_extended_compute_accumulated")):
            with self.subTest(schedule=schedule):
                plan = g.make_plan(32, 32, 32, schedule)
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "gemmini_%s.c" % schedule)
                    info = g.emit_c(plan, path)
                    self.assertEqual(info["path"], path)
                    self.assertEqual(info["sha256"], g.sha256_file(path))
                    with open(path) as fh:
                        text = fh.read()
                    self.assertIn("VERITAC_GEMMINI_PASS", text)
                    self.assertIn(compute_macro, text)
                    self.assertIn("gemmini_fence()", text)
                    self.assertIn("include/gemmini.h", text)
                    self.assertIn("#define VERITAC_M 32", text)

    def test_emission_rejects_invalid_plan_before_checker(self):
        plan = g.make_plan(32, 32, 32, g.SCHEDULE_BASELINE)
        plan["k"] = 33   # not divisible by DIM
        with mock.patch.object(g, "gemmini_check") as never:
            with self.assertRaises(ValueError):
                g.emit_c(plan, os.path.join(tempfile.gettempdir(), "vt_bad.c"))
            never.assert_not_called()

    def test_emission_rejects_malformed_plan_object(self):
        with self.assertRaises(ValueError):
            g.emit_c(["not", "a", "dict"],
                     os.path.join(tempfile.gettempdir(), "vt_notdict.c"))


if __name__ == "__main__":
    unittest.main()
