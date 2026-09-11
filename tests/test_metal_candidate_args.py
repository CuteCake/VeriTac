"""Focused tests for metal_candidate.py argument validation.

Locks in the rule that the requested mappings and query tiles must leave at
least one valid (mapping, query_tile) pair: otherwise argument parsing fails
clearly instead of silently running zero configurations.  When at least one
valid combination remains, parsing succeeds and invalid pairs continue to be
skipped deterministically downstream.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks" / "attention" / "metal_candidate"))

import metal_candidate as mc


def parse(*extra):
    return mc.parse_args(list(extra))


class MetalCandidateArgsTest(unittest.TestCase):
    def test_zero_valid_pairs_fails_scalar_q4(self):
        with self.assertRaises(SystemExit):
            parse("--mappings", "scalar", "--query-tiles", "4")

    def test_zero_valid_pairs_fails_simd_q16(self):
        with self.assertRaises(SystemExit):
            parse("--mappings", "simdgroup", "--query-tiles", "16")

    def test_crossed_mappings_succeeds_with_any_valid_pair(self):
        # 16 is valid for scalar and 4 is valid for simdgroup, so at least one
        # valid combination remains; parsing must succeed, not error.
        args = parse("--mappings", "scalar,simdgroup",
                     "--query-tiles", "16,4")
        self.assertEqual(args.mappings, ["scalar", "simdgroup"])
        self.assertEqual(args.query_tiles, [16, 4])

    def test_at_least_one_valid_pair_succeeds(self):
        # scalar q8 is valid even though scalar q4 and q16-with-simd are not.
        args = parse("--mappings", "scalar", "--query-tiles", "4,8")
        self.assertEqual(args.mappings, ["scalar"])
        self.assertEqual(args.query_tiles, [4, 8])

    def test_defaults_still_valid(self):
        args = parse()
        self.assertEqual(args.mappings, ["simdgroup"])
        self.assertEqual(args.query_tiles, [4, 8])
        self.assertEqual(args.key_tiles, [8, 16])


if __name__ == "__main__":
    unittest.main()
