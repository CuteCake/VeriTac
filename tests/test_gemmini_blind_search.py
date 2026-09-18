"""Synthetic orchestration tests; model responses and certificates are mocked.

The real native byte checker is exercised unless a test explicitly injects a
failure. These tests are not evidence of autonomous model optimization.
"""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from specializations.gemmini_gemm import backend as g
from specializations.gemmini_gemm import blind_search as b


def reply(commands):
    return json.dumps({"commands": g.serialize_commands(commands), "rationale": "synthetic fixture"})


def events(text, extra=None):
    lines = [{"type": "text", "sessionID": "synthetic", "part": {"text": text}}]
    lines.extend(extra or [])
    return "\n".join(json.dumps(line) for line in lines)


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = g.make_plan(16, 16, 16, "baseline")
        self.commands = g.gen_commands(self.plan)
        self.probe = patch.object(b, "probe_isolation", return_value={"supported": True})
        self.probe.start()
        self.addCleanup(self.probe.stop)

    def controller(self, responses, **kwargs):
        outputs = iter(responses)
        def runner(argv, cwd, env, timeout):
            self.assertNotIn(str(b._default_repo_root()), cwd)
            self.assertEqual(json.loads(env["OPENCODE_CONFIG_CONTENT"])["permission"]["*"], "deny")
            value = next(outputs)
            if isinstance(value, Exception):
                raise value
            return {"returncode": 0, "stdout": value, "stderr": ""}
        config = b.RunConfig("synthetic", dict(self.plan), max_rounds=len(responses))
        return b.BlindSearch(config, self.root / "run", model_runner=runner,
                             certifier=lambda *args, **kw: {"accepted": True, "reason": "SYNTHETIC"}, **kwargs)

    def test_real_native_feedback_and_winner_binding(self):
        longer = [g.ConfigEx(1)] + self.commands
        controller = self.controller([events(reply([g.Fence()])), events(reply(longer)), events(reply(self.commands))])
        result = controller.run()
        self.assertEqual([r["status"] for r in result["rounds"]], [b.STATUS_NATIVE_REJECTED, b.STATUS_NATIVE_ACCEPTED, b.STATUS_NATIVE_ACCEPTED])
        self.assertEqual(result["winner"]["round"], 3)
        self.assertEqual(result["evidence_kind"], "synthetic_test")
        self.assertIsNone(result["rounds"][0]["usage"])
        prompt = (controller.out_dir / "prompts/round_002.prompt.txt").read_text()
        self.assertIn("own_previous_proposal", prompt)
        self.assertIn("output_obligation_mismatch", prompt)
        self.assertNotIn("gen_batched_reuse", prompt)
        with self.assertRaises(b.BlindSearchError):
            controller.prepare()

    def test_timeout_and_provider_error_are_not_semantic_rejections(self):
        controller = self.controller([
            subprocess.TimeoutExpired("synthetic", 1, output=b"partial"),
            events(reply(self.commands), [{"type": "error", "error": "provider failed"}])])
        result = controller.run()
        self.assertEqual([r["status"] for r in result["rounds"]], [b.STATUS_MODEL_TIMEOUT, b.STATUS_MODEL_PROCESS_FAILURE])
        self.assertIsNone(result["winner"])

    def test_tool_use_event_discards_even_correct_candidate(self):
        controller = self.controller([events(reply(self.commands), [
            {"type": "tool_use", "part": {"type": "tool", "tool": "read"}}])])
        with self.assertRaisesRegex(b.BlindSearchError, "tool event"):
            controller.run()
        receipt = json.loads((controller.out_dir / "receipts/round_001.receipt.json").read_text())
        self.assertEqual(receipt["status"], "isolation_violation")

    def test_failed_probe_stops_before_model(self):
        controller = self.controller([events(reply(self.commands))])
        with patch.object(b, "probe_isolation", return_value={"supported": False, "reason": "synthetic failure"}):
            with self.assertRaisesRegex(b.BlindSearchError, "isolation preflight"):
                controller.run()
        self.assertEqual(list((controller.out_dir / "model_raw").iterdir()), [])

    def test_mutated_task_stops_verification(self):
        controller = self.controller([events(reply(self.commands))])
        controller.prepare()
        controller.config.plan["k"] = 32
        with self.assertRaisesRegex(b.IntegrityDrift, "task contract"):
            controller._verify_integrity("test")

    def test_native_timeout_separate_from_rejection(self):
        controller = self.controller([events(reply(self.commands))], program_check=lambda *args: (False, "checker timed out after 60s"))
        result = controller.run()
        self.assertEqual(result["rounds"][0]["status"], "checker_timeout")
        self.assertIsNone(result["winner"])

    def test_native_acceptance_never_implies_certificate(self):
        controller = self.controller([events(reply(self.commands))])
        controller.certifier = lambda *args, **kwargs: {"accepted": False, "reason": "synthetic proof timeout"}
        result = controller.run()
        self.assertEqual(result["status"], "certification_failed")
        self.assertTrue(result["winner"]["native_accepted"])
        self.assertFalse(result["winner"]["certified"])

    def test_candidate_tamper_before_certification_detected(self):
        controller = self.controller([events(reply(self.commands))])
        controller.prepare()
        receipt = controller._run_round(1, b.build_prompt(self.plan))
        controller.receipts.append(receipt)
        Path(receipt["commands_path"]).write_text('[{"kind":"fence"}]')
        with self.assertRaisesRegex(b.IntegrityDrift, "candidate"):
            controller._finalize()


class ParserAndIntegrityTests(unittest.TestCase):
    def test_strict_response_contract(self):
        good = '{"commands":[{"kind":"fence"}],"rationale":"x"}'
        self.assertEqual(len(b.parse_proposal("```json\n" + good + "\n```", 10)[0]), 1)
        invalid = [good + good, good.replace('"x"', 'NaN'),
                   good.replace('"rationale":"x"', '"plan":{},"rationale":"x"'),
                   good.replace('"rationale":"x"', '"rationale":"x","rationale":"y"'),
                   '{"commands":[{"kind":"config_ex","dataflow":true}],"rationale":"x"}',
                   '{"commands":[{"kind":"fence","extra":1}],"rationale":"x"}']
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(b.BlindSearchError):
                b.parse_proposal(value, 10)

    def test_missing_to_present_and_content_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file"
            targets = {"x": path}
            frozen = b.freeze_hashes(targets)
            path.write_text("new")
            with self.assertRaises(b.IntegrityDrift):
                b.verify_hashes(frozen, targets)
            frozen = b.freeze_hashes(targets)
            path.write_text("edited")
            with self.assertRaises(b.IntegrityDrift):
                b.verify_hashes(frozen, targets)


if __name__ == "__main__":
    unittest.main()
