"""Ensure instruction traces count commands, not matrix-dump text."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "benchmarks/gemmini/run_spike.py"
SPEC = importlib.util.spec_from_file_location("gemmini_spike_runner", PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class TraceTests(unittest.TestCase):
    def test_counts_commands_and_excludes_accumulator_width_assumptions(self):
        text = """Gemmini extension configured with:
    dim = 16
GEMMINI: mvin - 0x10 cols and 0x10 rows from 0x000840f0 to addr 0x00000000
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
GEMMINI: mvin - 0x10 cols and 0x10 rows from 0x000850f0 to addr 0x00000010
GEMMINI: mvin - 0x10 cols and 0x10 rows from 0x000860f0 to addr 0x80000000
GEMMINI: preload - scratchpad output addr = 0x80000000, scratchpad preload addr = 0x00000010
GEMMINI: compute - preload = 1, scratchpad A addr = 0x00000000, scratchpad B/D addr = 0xffffffff
GEMMINI: compute - PEs after preloading:
GEMMINI: compute - PEs after matmul:
GEMMINI: compute - writing results to addr 0x80000000, :
GEMMINI: mvout - 0x10 cols and 0x10 rows from 0xa0000000 to addr 0x000870f0
VERITAC_GEMMINI_PASS
"""
        counts = runner.trace_counts(text)
        self.assertEqual(counts["operations"], {
            "mvin": 3, "scratchpad_input_bytes": 512,
            "preload": 1, "compute": 1, "mvout": 1,
        })
        self.assertEqual(counts["mvin_by_local_address"], {
            "0x00000000": 1, "0x00000010": 1, "0x80000000": 1,
        })

    def test_missing_trace_cannot_look_like_executed_compute(self):
        self.assertEqual(runner.trace_counts("VERITAC_GEMMINI_PASS\n"), {
            "operations": {}, "mvin_by_local_address": {},
        })


if __name__ == "__main__":
    unittest.main()
