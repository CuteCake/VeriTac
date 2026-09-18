import unittest
from specializations.tenstorrent_protocol import protocol as p
from specializations.tenstorrent_protocol.adapter import baseline, evaluate, validate_task


class AdapterTests(unittest.TestCase):
    def task(self):
        return {'id': 'test', 'input_pages': 2, 'expected': [1,0], 'slots': 2, 'page_bytes': 512}

    def test_empty_spec_is_rejected(self):
        with self.assertRaises(ValueError):
            p.parse_task({**self.task(), 'expected': []})

    def test_state_cap_is_positive_integer(self):
        t=self.task()
        for n in [0,-1,True,1.5]:
            with self.assertRaises(ValueError):
                p.check(t,p.serial_baseline(t),max_states=n)

    def test_invalid_simulation_has_no_winning_score(self):
        t=self.task()
        invalid=p.parse_proposal({'program':[], 'rationale':''})
        with self.assertRaises(ValueError):
            p.score_key(p.simulate(t,invalid))

    def test_missing_read_wait_cannot_pass(self):
        t=self.task(); prog=baseline(t)
        del prog[2]
        r=evaluate(t,prog)
        self.assertFalse(r['accepted'])
        self.assertEqual(r['status'],'unsafe')
        self.assertTrue(r['trace'])

    def test_task_copy_and_instance_bounds(self):
        t=self.task(); validated=validate_task(t)
        t['expected'][0]=0
        self.assertEqual(validated['expected'],[1,0])
        for override in [{'slots':9},{'input_pages':17},{'page_bytes':32768}]:
            with self.assertRaises(ValueError):
                validate_task({**self.task(),**override})


if __name__=='__main__':
    unittest.main()
