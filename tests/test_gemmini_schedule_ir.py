import unittest
from specializations.gemmini_gemm.schedule_ir import expand_program, expression, ScheduleError


def loop(stop, body, start=0, step=1):
    return {"for": "i", "start": start, "stop": stop, "step": step, "body": body}


class ScheduleTests(unittest.TestCase):
    def test_expanded_reduction_passes_existing_byte_checker(self):
        from specializations.gemmini_gemm import backend as g
        plan = g.make_plan(16, 16, 32, "baseline")
        baseline = g.gen_commands(plan)
        flat = g.serialize_commands(baseline)
        body = [dict(c) for c in flat[4:8]]
        body[0]["offset"] = {"expr": "i * 16"}
        body[1]["offset"] = {"expr": "i * 16 * N"}
        body[2]["out_addr"] = {"expr": "0xA0000000 if i == 0 else 0xE0000000"}
        expanded = expand_program(flat[:4] + [loop({"expr": "K // 16"}, body)] + flat[-2:],
                                  {"N": 16, "K": 32})
        self.assertEqual(expanded, flat)
        classes = {type(c).kind: type(c) for c in baseline}
        commands = [classes[c["kind"]](**{k: v for k, v in c.items() if k != "kind"})
                    for c in expanded]
        ok, why, _ = g.check_program(plan, commands)
        self.assertTrue(ok, why)
        # A compact-source arithmetic error must still fail final validation.
        body[1]["offset"] = {"expr": "i * 16"}
        bad = expand_program(flat[:4] + [loop(2, body)] + flat[-2:], {"N": 16})
        commands = [classes[c["kind"]](**{k: v for k, v in c.items() if k != "kind"}) for c in bad]
        self.assertFalse(g.check_program(plan, commands)[0])

    def test_expansion_is_concrete_and_scoped(self):
        source = [loop({"expr": "M // 16"}, [
            {"kind": "mvin", "offset": {"expr": "i * 16"}, "buf": "A"}]),
            {"kind": "fence"}]
        self.assertEqual(expand_program(source, {"M": 32}), [
            {"kind": "mvin", "offset": 0, "buf": "A"},
            {"kind": "mvin", "offset": 16, "buf": "A"}, {"kind": "fence"}])
        with self.assertRaises(ScheduleError):
            expand_program(source + [{"kind": "x", "offset": {"expr": "i"}}], {"M": 32})

    def test_negative_step_matches_python_range(self):
        for start, stop, step in ((5, -2, -2), (2, 5, -1), (-3, -6, -2), (5, 0, -3)):
            result = expand_program([loop(stop, [{"kind": "x", "i": {"expr": "i"}}], start, step)], {})
            self.assertEqual([x["i"] for x in result], list(range(start, stop, step)))

    def test_resource_bounds_include_empty_body(self):
        for source in ([loop(1000000000, [])], [loop(5, [{"kind": "fence"}])]):
            with self.assertRaises(ScheduleError):
                expand_program(source, {}, max_commands=3, max_iterations=10)

    def test_no_code_execution_or_spec_rebinding(self):
        for text in ("__import__('os').system('true')", "M.bit_length()", "2 ** 10000000", "M[0]", "1 / 2", "True + 1", "1 // 0", "1 << 99", "0 if True else int('1')", "0 if True else missing"):
            with self.subTest(text=text), self.assertRaises(ScheduleError):
                expression(text, {"M": 32})
        with self.assertRaises(ScheduleError):
            expand_program([loop(2, [])], {"i": 1})

    def test_conditionals_preserve_boolean_types(self):
        self.assertIs(expression("i > 0", {"i": 1}), True)
        self.assertEqual(expression("10 if i == 0 else 20", {"i": 0}), 10)
        with self.assertRaises(ScheduleError):
            expand_program([loop(True, [])], {})


if __name__ == "__main__":
    unittest.main()
