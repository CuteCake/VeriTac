"""Synthetic compact-interface integration tests, not live experiment results."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from specializations.gemmini_gemm import backend as g, blind_search as b
from tests.test_gemmini_blind_search import events


class CompactSearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.plan = g.make_plan(16, 16, 32, "baseline")
        flat = g.serialize_commands(g.gen_commands(self.plan))
        body = copy.deepcopy(flat[4:8])
        body[0]["offset"] = {"expr": "q * 16"}
        body[1]["offset"] = {"expr": "q * 16 * N"}
        body[2]["out_addr"] = {"expr": "0xA0000000 if q == 0 else 0xE0000000"}
        self.source = {"program": flat[:4] + [{"for": "q", "start": 0,
                       "stop": {"expr": "K // 16"}, "step": 1, "body": body}] + flat[-2:],
                       "rationale": "synthetic fixture"}
        self.calls = 0

    def controller(self, responses, **kwargs):
        source = iter(responses)
        def runner(*args):
            self.calls += 1
            return {"returncode": 0, "stdout": events(json.dumps(next(source))), "stderr": ""}
        cfg = b.RunConfig("synthetic_compact", self.plan, max_rounds=len(responses), proposal_format="compact")
        return b.BlindSearch(cfg, Path(self.tmp.name) / "run", model_runner=runner,
                            certifier=lambda *args, **kw: {"accepted": True, "reason": "SYNTHETIC"}, **kwargs)

    @patch.object(b, "probe_isolation", return_value={"supported": True})
    def test_math_repair_keeps_original_compact_source(self, _):
        bad = copy.deepcopy(self.source)
        bad["program"][4]["body"][1]["offset"] = {"expr": "q * 16"}
        controller = self.controller([bad, self.source])
        result = controller.run()
        self.assertEqual([r["status"] for r in result["rounds"]], ["native_rejected", "native_accepted"])
        self.assertEqual(result["proposal_format"], "compact")
        self.assertEqual(result["evidence_kind"], "synthetic_test")
        self.assertIn("controller_schedule_ir", result["integrity"]["frozen"])
        self.assertIn("controller_compact_protocol", result["integrity"]["frozen"])
        saved = json.loads(Path(result["rounds"][1]["source_path"]).read_text())
        self.assertEqual(saved, self.source)
        prompt = (controller.out_dir / "prompts/round_002.prompt.txt").read_text()
        self.assertIn('"program"', prompt)
        self.assertIn('q * 16', prompt)
        self.assertIn("output_obligation_mismatch", prompt)
        self.assertEqual(result["winner"]["round"], 2)

    @patch.object(b, "probe_isolation", return_value={"supported": True})
    def test_run_configuration_change_blocks_model_call(self, _):
        controller = self.controller([self.source])
        controller.prepare()
        controller.config.model = "a-different-provider/model"
        with self.assertRaisesRegex(b.IntegrityDrift, "configuration"):
            controller._run_round(1, "synthetic prompt")
        self.assertEqual(self.calls, 0)

    @patch.object(b, "probe_isolation", return_value={"supported": True})
    def test_timeout_feedback_does_not_blame_valid_syntax(self, _):
        controller = self.controller([self.source], program_check=lambda *args: (False, "checker timed out"))
        result = controller.run()
        feedback = b.build_feedback("synthetic_compact", self.plan, result["rounds"])
        previous = feedback["previous_rounds"][0]
        self.assertTrue(previous["syntax_ok"])
        self.assertEqual(previous["status"], "checker_timeout")
        self.assertEqual(previous["own_previous_proposal"], self.source)
        self.assertIn("timed out", previous["error"])


if __name__ == "__main__":
    unittest.main()
