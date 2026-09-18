import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from specializations.tenstorrent_protocol import search


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.adapter = types.ModuleType('specializations.tenstorrent_protocol.adapter')
        self.adapter.validate_task = lambda t: t
        self.adapter.evaluate = lambda t, p: {'accepted': True, 'objective': [len(p), len(p), 1]}
        self.certified = []
        def certify(task, program, output, timeout):
            self.certified.append(program)
            return {'accepted': True, 'synthetic': True}
        self.adapter.certify = certify

    def invoke(self, responses):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {self.adapter.__name__: self.adapter}):
            pending = iter(responses)
            def call(*args):
                return next(pending)
            result = search.run({'id': 'synthetic'}, Path(tmp)/'run', 'TEST CONTRACT',
                                rounds=len(responses), model_call=call)
            self.assertTrue(result['config']['synthetic_model'])
            return result

    def response(self, program, extra=None):
        lines = [{'type': 'text', 'part': {'text': json.dumps({'program': program, 'rationale': ''})}}]
        if extra:
            lines.append(extra)
        return {'returncode': 0, 'timed_out': False, 'stdout': '\n'.join(map(json.dumps, lines))}

    def test_only_best_recorded_program_is_certified(self):
        first = [{'op': 'wait_reads'}, {'op': 'wait_writes'}]
        best = [{'op': 'wait_reads'}]
        r = self.invoke([self.response(first), self.response(best)])
        self.assertEqual(r['winner']['round'], 2)
        self.assertEqual(self.certified, [best])

    def test_tool_attempt_prevents_acceptance(self):
        r = self.invoke([self.response([{'op': 'wait_reads'}], {'type': 'tool_use', 'part': {'type': 'tool', 'tool': 'bash'}})])
        self.assertEqual(r['status'], 'no_certified_winner')
        self.assertEqual(self.certified, [])

    def test_timeout_with_partial_valid_reply_is_not_accepted(self):
        response = self.response([{'op': 'wait_reads'}])
        response['timed_out'] = True
        r = self.invoke([response])
        self.assertEqual(r['rounds'][0]['status'], 'model_timeout')
        self.assertEqual(self.certified, [])

    def test_native_budget_or_rejection_is_not_certified(self):
        self.adapter.evaluate = lambda t, p: {'accepted': False, 'status': 'unknown_budget'}
        r = self.invoke([self.response([{'op': 'wait_reads'}])])
        self.assertEqual(r['status'], 'no_certified_winner')
        self.assertEqual(self.certified, [])

    def test_source_drift_prevents_final_certificate(self):
        with patch.object(search, 'fingerprint', side_effect=[{'x': 'a'}, {'x': 'a'}, {'x': 'b'}]):
            with self.assertRaisesRegex(RuntimeError, 'fingerprint changed'):
                self.invoke([self.response([{'op': 'wait_reads'}])])
        self.assertEqual(self.certified, [])

    def test_exact_json_no_duplicates_or_extra_outer_fields(self):
        for raw in ['{"program":[],"program":[{}],"rationale":""}',
                    '{"program":[{}],"rationale":"","task":{}}',
                    '{"program":[{}],"rationale":NaN}']:
            with self.assertRaises(ValueError):
                search.extract_proposal(raw)


if __name__ == '__main__':
    unittest.main()
