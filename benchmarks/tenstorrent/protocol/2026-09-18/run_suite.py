"""Frozen first protocol optimization pilot; at most 3 live OpenCode calls."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from specializations.tenstorrent_protocol.search import run
HERE=Path(__file__).resolve().parent


def one(name):
    task=json.loads((HERE/'tasks'/f'{name}.json').read_text())
    print('START',name,flush=True)
    try:
        result=run(task,HERE/'runs'/name,(HERE/'SCOPE.md').read_text(),rounds=3,
                   model='dgxspark-glm/glm-5.3-flash',model_timeout=900,proof_timeout=600)
        summary={'task_id':name,'status':result['status'],'winner':result['winner'],
                 'round_statuses':[r['status'] for r in result['rounds']]}
    except Exception as error:
        summary={'task_id':name,'status':'aborted','error':repr(error)}
        folder=HERE/'runs'/name; folder.mkdir(parents=True,exist_ok=True)
        (folder/'abort.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('DONE',json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    outputs=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks=[pool.submit(one,n) for n in ['pair','permutation','replication','reuse_permutation']]
        for f in as_completed(tasks):
            outputs.append(f.result())
            (HERE/'suite_outcomes.json').write_text(json.dumps(outputs,indent=2)+'\n')
    raise SystemExit(0 if all(x['status']=='certified' for x in outputs) else 1)
