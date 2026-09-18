"""Repeat an already-built Metalium executable with independent full-byte checks.
Run inside the local simulator container. This is execution evidence, not proof.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from .runner import compute_reference
from .. import protocol


def replay(task, binary, output, timeout=240):
    task=protocol.parse_task(task)
    binary=Path(binary).resolve(); output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    fingerprint=hashlib.sha256(binary.read_bytes()).hexdigest()
    cases=[]
    for pattern,seed in [('zero',0),('ones',0),('alternating',0),('lcg',0),('lcg',917)]:
        case=output/f'{pattern}_{seed}';case.mkdir()
        try:
            ran=subprocess.run([str(binary),str(seed),pattern],cwd=case,
                               capture_output=True,text=True,timeout=timeout)
            log=ran.stdout+ran.stderr;code=ran.returncode
        except subprocess.TimeoutExpired as error:
            log=str(error);code=None
        (case/'run.log').write_text(log)
        reference=compute_reference(task,seed,pattern)['expected_output']
        raw=case/'output.bin'; data=raw.read_bytes() if raw.exists() else None
        ok=code==0 and data==reference and f'VERIFY OK {task.output_pages}' in log
        cases.append({'pattern':pattern,'seed':seed,'returncode':code,'all_bytes_match':ok,
                      'expected_sha256':hashlib.sha256(reference).hexdigest(),
                      'output_sha256':hashlib.sha256(data).hexdigest() if data is not None else None,
                      'output_bytes':len(data) if data is not None else None})
    intact=hashlib.sha256(binary.read_bytes()).hexdigest()==fingerprint
    result={'scope':'official ttsim execution evidence, not proof or hardware latency',
            'binary_sha256':fingerprint,'binary_unchanged':intact,'cases':cases,
            'success':intact and all(c['all_bytes_match'] for c in cases)}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,type=Path)
    parser.add_argument('--binary',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    r=replay(json.loads(args.task.read_text()),args.binary,args.out)
    print(json.dumps(r,indent=2))
    raise SystemExit(0 if r['success'] else 1)
