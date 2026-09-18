"""Bind live replies, fixed tasks, protocol certificates, C++ and simulator bytes."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
from specializations.tenstorrent_protocol import protocol as p,certificate as cert
from specializations.tenstorrent_protocol.adapter import evaluate
from specializations.tenstorrent_protocol.search import fingerprint,extract_proposal,parse_stream_events
from specializations.tenstorrent_protocol.runtime import emit_cpp,runner


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,reason):
    if not value:raise ValueError(reason)


def audit_certificate(task,program,result):
    require(result['accepted'],'certificate not accepted')
    source=Path(result['source']);request=Path(result['request'])
    require(sha(source)==result['source_sha256'],'certificate source hash changed')
    require(sha(request)==result['request_sha256'],'certificate request hash changed')
    require(read(request)=={'task':task,'program':program},'certificate task/program mismatch')
    ti=p.parse_task(task);pi=p.parse_proposal({'program':program,'rationale':''})
    expected=cert.certificate_source(ti,pi.program,cert.propose_graph(ti,pi.program))
    require(source.read_text()==expected,'certificate text does not match regenerated statement')
    ok,axs=cert.audit_axioms(Path(result['log']).read_text())
    require(ok and axs==result['axioms'],'axiom audit mismatch')
    return {'source_sha256':sha(source),'axioms':axs,'nodes':result['nodes']}


def audit_runtime(kind,name,task,proposal,certificate):
    folder=HERE/'runtime/pilot-v1'/f'{kind}_{name}';report=read(folder/'results.json')
    require(report['status']=='verified' and report['full_output_matches_python_reference'],'runtime failed')
    require(read(folder/'binding.json')['protocol_certificate_sha256']==certificate['source_sha256'],'runtime bound to different certificate')
    task_ir=p.parse_task(task);pi=p.parse_proposal(proposal)
    expected=emit_cpp.emit_bundle(task_ir,pi)
    for rel,text in expected.items():
        path=folder/'src'/rel
        require(path.read_text()==text,'generated source/metadata mismatch: '+str(path))
        require(sha(path)==report['emitted_artifacts'][rel]['sha256'],'emitted artifact hash mismatch')
    main_expected=runner.compute_reference(task_ir,report['input_seed'],report['input_pattern'])['expected_output']
    require((folder/'output.bin').read_bytes()==main_expected,'main full-output mismatch')
    extras=read(folder/'extra/results.json')
    require(extras['success'] and extras['binary_unchanged'],'extra runtime cases failed')
    require(extras['binary_sha256']==report['binary_sha256'],'extra cases used a different host executable')
    for case in extras['cases']:
        raw=folder/'extra'/f"{case['pattern']}_{case['seed']}"/'output.bin'
        expected=runner.compute_reference(task_ir,case['seed'],case['pattern'])['expected_output']
        require(raw.read_bytes()==expected,'extra full-output mismatch')
        require(sha(raw)==case['output_sha256'],'extra output hash mismatch')
    devices=read(HERE/'environment/device_code/manifest.json')
    device=next(x for x in devices if x['case']==f'{kind}_{name}')
    require(device['source_sha256']==sha(folder/'src/kernels/dm_slot_copy.cpp'),'device ELF source association mismatch')
    for rel,info in device['files'].items():
        require(sha(HERE/'environment/device_code'/f'{kind}_{name}'/rel)==info['sha256'],'device artifact hash mismatch')
    return {'case':f'{kind}_{name}','full_output_cases':1+len(extras['cases']),
            'host_binary_sha256':report['binary_sha256'],
            'device_elf_sha256':device['files']['brisc.elf']['sha256']}


def audit():
    reg=read(HERE/'preregistration.json')
    require(reg['controller_fingerprint']==fingerprint(),'formal/search fingerprint changed')
    require(sha(HERE/'run_suite.py')==reg['driver_sha256'],'registered driver changed')
    rows=[];rounds=[];runtime=[];ids=[]
    tokens={k:0 for k in ['input','output','total']};wall=0.0
    for name in reg['tasks']:
        task_path=HERE/'tasks'/f'{name}.json';task=read(task_path)
        require(sha(task_path)==reg['task_sha256'][task_path.name],'registered task changed')
        controls=HERE/'controls'/f'{name}.json'
        require(sha(controls)==reg['controls_sha256'][controls.name],'registered controls changed')
        control=read(controls)
        report=read(HERE/'runs'/name/'report.json')
        require(report['status']=='certified','task has no certificate')
        for i in range(1,reg['rounds_per_task']+1):
            f=HERE/'runs'/name/f'round_{i:03d}';receipt=read(f/'receipt.json')
            parsed=parse_stream_events((f/'stdout.jsonl').read_text())
            require(not parsed['tool_parts'] and not parsed['errors'],'model tool use or error')
            proposal=extract_proposal(parsed['text'])
            require(proposal==receipt['proposal'],'receipt differs from original model reply')
            require(parsed['usage']==receipt['usage'],'usage receipt mismatch')
            verdict=evaluate(task,proposal['program'])
            require(verdict['accepted'] and verdict['objective']==receipt['verdict']['objective'],'native receipt mismatch')
            ids.append(parsed['session_id']);rounds.append(receipt['status'])
            for key in tokens:tokens[key]+=receipt['usage'].get(key,0)
            wall+=receipt['model']['duration_seconds']
        winner=report['winner'];selected=read(HERE/'runs'/name/f"round_{winner['round']:03d}"/'receipt.json')['proposal']
        require(read(HERE/'runs'/name/'winner.proposal.json')==selected,'winner changed')
        formal=audit_certificate(task,selected['program'],winner['certificate'])
        score=evaluate(task,selected['program'])
        runtime.append(audit_runtime('live',name,task,selected,winner['certificate']))
        serial_dir=HERE/'control_certificates'/name;serial=read(serial_dir/'proposal.json');serial_cert=read(serial_dir/'result.json')
        require(serial['program']==control['controls'][0]['program'],'serial differs from registration')
        audit_certificate(task,serial['program'],serial_cert)
        runtime.append(audit_runtime('serial',name,task,serial,serial_cert))
        rows.append({'task_id':name,'model_objective':score['objective'],'model_counts':score['counts'],
                     'serial_objective':control['controls'][0]['verdict']['objective'],
                     'serial_counts':control['controls'][0]['verdict']['counts'],
                     'strong_control_objective':control['controls'][1]['verdict']['objective'],
                     'selected_round':winner['round'],'certificate':formal,
                     'kernel_check_seconds':winner['certificate']['kernel_check_seconds']})
    require(len(set(ids))==len(ids),'model sessions were reused')
    return {'success':True,'formal_and_search_fingerprint_unchanged':True,'certified_model_tasks':len(rows),
            'model_rounds':len(rounds),'round_statuses':rounds,'unique_model_sessions':len(set(ids)),
            'reported_tokens':tokens,'summed_model_call_seconds':round(wall,3),
            'unknown_usage_rounds':0,'runtime_programs':len(runtime),
            'runtime_full_output_cases':sum(x['full_output_cases'] for x in runtime),
            'rows':rows,'runtime':runtime,
            'scope':'artifact binding and replay audit; original Lean kernel checks are separate evidence; NO compiled-byte correctness theorem or hardware performance claim'}


if __name__=='__main__':
    result=audit();(HERE/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','runtime','round_statuses']},indent=2))
