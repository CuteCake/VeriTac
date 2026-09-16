"""Exercise the actual Lean executable, including malformed external input."""
import json
from pathlib import Path
import subprocess
import unittest

BIN = Path(__file__).resolve().parents[1] / ".lake/build/bin/gemmini_check"


def plan(**changes):
    result = dict(m=32, n=32, k=32, dim=16, scratchpad_rows=32,
                  accumulator_rows=32, schedule="reuse_b")
    result.update(changes)
    return result


def check(value):
    proc = subprocess.run([str(BIN)], input=json.dumps(value) + "\n",
                          text=True, capture_output=True, timeout=15)
    if proc.returncode:
        raise AssertionError(proc.stderr or proc.stdout)
    return json.loads(proc.stdout)


class CheckerTests(unittest.TestCase):
    def test_capacity_boundary_and_fallback(self):
        accepted = check(plan())
        self.assertIs(accepted["accepted"], True)
        self.assertEqual(accepted["plan"], plan())
        rejected = check(plan(accumulator_rows=31))
        self.assertIs(rejected["accepted"], False)
        self.assertNotIn("plan", rejected)
        self.assertIs(check(plan(schedule="baseline", accumulator_rows=16))["accepted"], True)
        self.assertIs(check(plan(schedule="baseline", accumulator_rows=15))["accepted"], False)
        self.assertIs(check(plan(scratchpad_rows=31))["accepted"], False)

    def test_bad_shapes_and_arithmetic_bound(self):
        for changes in (dict(m=0), dict(n=0), dict(k=0), dict(dim=0),
                        dict(m=31), dict(n=17), dict(k=33), dict(k=131072),
                        dict(schedule="unknown")):
            with self.subTest(changes=changes):
                out = check(plan(**changes))
                self.assertIs(out["accepted"], False)
                self.assertNotIn("plan", out)
        self.assertIs(check(plan(k=131056))["accepted"], True)

    def test_malformed_fields_fail_closed(self):
        for name in ("m", "n", "k", "dim", "scratchpad_rows", "accumulator_rows"):
            for value in (-1, 0.5, True, None, "32", [], {}):
                with self.subTest(name=name, value=value):
                    self.assertIs(check(plan(**{name: value}))["accepted"], False)
            missing = plan()
            del missing[name]
            self.assertIs(check(missing)["accepted"], False)
        for value in (None, [], "plan", 42):
            self.assertIs(check(value)["accepted"], False)

    def test_invalid_json_rejected(self):
        proc = subprocess.run([str(BIN)], input="{broken\n", text=True,
                              capture_output=True, timeout=15)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIs(json.loads(proc.stdout)["accepted"], False)


if __name__ == "__main__":
    unittest.main()
