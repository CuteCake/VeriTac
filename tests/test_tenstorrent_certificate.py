import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from specializations.tenstorrent_protocol import protocol as p
from specializations.tenstorrent_protocol import certificate as c
from specializations.tenstorrent_protocol.adapter import baseline


class CertificateTests(unittest.TestCase):
    def task(self):
        return {'id':'proof-test','input_pages':1,'expected':[0],'slots':1,'page_bytes':32}

    def test_axiom_audit_rejects_sorry_and_missing_theorems(self):
        good='\n'.join("'SubmittedProtocol.%s' depends on axioms: [propext, Quot.sound]" % n
                       for n in ['accepted','safe','progress','outputs'])
        self.assertTrue(c.audit_axioms(good)[0])
        self.assertFalse(c.audit_axioms(good.replace('Quot.sound','sorryAx'))[0])
        self.assertFalse(c.audit_axioms(good.replace('SubmittedProtocol.outputs','Other.outputs'))[0])

    def test_graph_budget_cannot_silently_truncate(self):
        t=p.parse_task(self.task()); program=p.serial_baseline(t).program
        with self.assertRaisesRegex(ValueError,'budget'):
            c.propose_graph(t,program,max_nodes=1)

    @unittest.skipUnless(shutil.which('lake') and
        (c.ROOT/'.lake/build/lib/lean/VeriTac/Tenstorrent/Protocol.olean').exists(),
        'requires built Lean protocol library')
    def test_kernel_checks_valid_and_rejects_tampered_graphs(self):
        t=p.parse_task(self.task()); prog=p.serial_baseline(t).program
        nodes=c.propose_graph(t,prog)
        valid=c.certificate_source(t,prog,nodes)
        parts=[valid.replace('SubmittedProtocol','ValidProtocol')]
        cases=[('MissingInitial',t,prog,nodes[1:]),
               ('MissingSuccessor',t,prog,nodes[:-1]),
               ('UnsafeProgram',t,tuple(op for op in prog if not isinstance(op,p.WaitReads)),nodes),
               ('WrongSpec',p.parse_task({**self.task(),'input_pages':2,'expected':[1]}),prog,nodes)]
        for name,task,program,graph in cases:
            prefix=c.certificate_source(task,program,graph).split('theorem accepted')[0]
            prefix=prefix.replace('SubmittedProtocol',name).replace('import VeriTac.Tenstorrent.Protocol\n','')
            parts.append(prefix+'example : checkCertificate task program nodes = false := by decide +kernel\nend '+name+'\n')
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'proof_tests.lean';f.write_text('\n'.join(parts))
            run=subprocess.run(['lake','env','lean','-s','65536',str(f)],cwd=c.ROOT,
                               capture_output=True,text=True,timeout=180)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertNotIn('sorryAx',run.stdout)


if __name__=='__main__':
    unittest.main()
