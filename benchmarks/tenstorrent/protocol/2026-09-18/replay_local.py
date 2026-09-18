"""Replay certified proposals on the user's local Mac Studio Docker simulator."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
CONTAINER='veritac-tt-sim'
EXEC=['docker','exec','-w','/workspace','-e','PYTHONPATH=/workspace','-e','CXX=g++-14',
      '-e','TT_METAL_HOME=/opt/tt/tt-metal','-e','TT_METAL_RUNTIME_ROOT=/opt/tt/tt-metal',
      '-e','TT_METAL_SIMULATOR=/opt/tt/sim/libttsim_bh.so',
      '-e','TT_METAL_SLOW_DISPATCH_MODE=1','-e','TT_METAL_DISABLE_SFPLOADMACRO=1',CONTAINER]


def mapped(path): return '/workspace/'+str(Path(path).resolve().relative_to(ROOT))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def call(args,log):
    r=subprocess.run(args,capture_output=True,text=True,timeout=1200)
    Path(log).write_text(r.stdout+r.stderr)
    return r.returncode


def one(kind,name,attempt):
    if kind=='live':
        folder=HERE/'runs'/name; report=json.loads((folder/'report.json').read_text())
        cert=report['winner']['certificate'];proposal=folder/'winner.proposal.json'
    else:
        folder=HERE/'control_certificates'/name
        cert=json.loads((folder/'result.json').read_text());proposal=folder/'proposal.json'
    assert cert['accepted'] and sha(cert['source'])==cert['source_sha256']
    assert sha(cert['request'])==cert['request_sha256']
    request=json.loads(Path(cert['request']).read_text())
    assert request['program']==json.loads(proposal.read_text())['program']
    task=HERE/'tasks'/f'{name}.json'; assert json.loads(task.read_text())==request['task']
    output=HERE/'runtime'/attempt/f'{kind}_{name}'; output.mkdir(parents=True,exist_ok=False)
    remote=f'/opt/tt/artifacts/{attempt}/{kind}_{name}'
    cmd=EXEC+['python3','-m','specializations.tenstorrent_protocol.runtime.runner',
              '--task',mapped(task),'--proposal',mapped(proposal),'--out',remote+'/main',
              '--mode','package','--cmake-prefix','/opt/tt/install','--jobs','1','--timeout','240']
    rc=call(cmd,output/'driver.log')
    # Keep emitted sources, logs, complete raw outputs and verification reports.
    # Large build intermediates remain in the persistent local Docker volume.
    for item in ['results.json','run.log','output.bin','src']:
        subprocess.run(['docker','cp',f'{CONTAINER}:{remote}/main/{item}',str(output/item)],
                       capture_output=True,text=True)
    summary={'kind':kind,'task_id':name,'main_returncode':rc,
             'protocol_certificate_sha256':cert['source_sha256'],'success':False}
    if rc==0:
        main=json.loads((output/'results.json').read_text())
        assert main['status']=='verified' and main['full_output_matches_python_reference']
        extra=EXEC+['python3','-m','specializations.tenstorrent_protocol.runtime.replay_cases',
                    '--task',mapped(task),'--binary',remote+'/main/build/veritac_tt_protocol',
                    '--out',remote+'/extra']
        erc=call(extra,output/'extra-driver.log')
        subprocess.run(['docker','cp',f'{CONTAINER}:{remote}/extra',str(output/'extra')],check=True,
                       capture_output=True,text=True)
        evidence=json.loads((output/'extra/results.json').read_text())
        summary.update(extra_returncode=erc,success=erc==0 and evidence['success'],
                       host_binary_sha256=main['binary_sha256'],output_cases=1+len(evidence['cases']))
        assert main['binary_sha256']==evidence['binary_sha256']
    (output/'binding.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempt',required=True)
    p.add_argument('--kind',choices=['live','serial','both'],default='both'); args=p.parse_args()
    if not args.attempt.replace('-','').replace('_','').isalnum(): p.error('invalid attempt name')
    outcomes=[]
    for kind in (['live','serial'] if args.kind=='both' else [args.kind]):
        for name in ['pair','permutation','replication','reuse_permutation']:
            print('START',kind,name,flush=True)
            result=one(kind,name,args.attempt);outcomes.append(result);print(json.dumps(result),flush=True)
            (HERE/f'{args.attempt}_runtime_summary.json').write_text(json.dumps(outcomes,indent=2)+'\n')
            if not result['success']: raise SystemExit(1)
