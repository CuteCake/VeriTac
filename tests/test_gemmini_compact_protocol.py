import json
import unittest
from specializations.gemmini_gemm import backend as g, blind_search as flat, compact_protocol as compact


class CompactProtocolTests(unittest.TestCase):
    def setUp(self):
        self.plan = g.make_plan(16, 16, 16, "baseline")

    def test_literal_program_preserves_native_acceptance(self):
        source = {"program": g.serialize_commands(g.gen_commands(self.plan)), "rationale": "synthetic test"}
        commands, _ = compact.parse_compact(json.dumps(source), self.plan)
        ok, why, _ = g.check_program(self.plan, commands)
        self.assertTrue(ok, why)

    def test_immutable_constants_and_budget(self):
        for program in (
            [{"for": "M", "start": 0, "stop": 1, "step": 1, "body": [{"kind": "fence"}]}],
            [{"for": "i", "start": 0, "stop": 100001, "step": 1, "body": []}],
            [{"kind": "config_ex", "dataflow": {"expr": "__import__('os')"}}],
            [{"kind": "config_ex", "dataflow": {"expr": "M == 16"}}],
        ):
            with self.subTest(program=program), self.assertRaises(flat.ProposalRejected):
                compact.parse_compact(json.dumps({"program": program, "rationale": "test"}), self.plan)

    def test_no_ambiguous_flat_schema_in_prompt(self):
        text = compact.build_prompt(self.plan)
        self.assertIn('"program": [<nodes>]', text)
        self.assertNotIn('"commands": [<command objects>]', text)
        self.assertIn("bit 30", text)


if __name__ == "__main__":
    unittest.main()
