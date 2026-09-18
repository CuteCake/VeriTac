"""Kernel-checked finite-state protocol certificates (not machine-code proofs)."""
from collections import deque
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from . import protocol as p

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_AXIOMS = {'propext', 'Quot.sound', 'Classical.choice'}


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def successors(task, program, state):
    """Untrusted graph proposal. Lean independently reconstructs every edge."""
    children = []
    if state.pc < len(program):
        op = program[state.pc]
        if isinstance(op, (p.WaitReads, p.WaitWrites)):
            pending = state.reads if isinstance(op, p.WaitReads) else state.writes
            if not pending:
                from dataclasses import replace
                children.append(replace(state, pc=state.pc+1))
        else:
            child, error = p._issue(state, op, task)
            if error:
                raise ValueError('unsafe issue while proposing graph: '+error)
            children.append(child)
    children.extend(p._complete_read(state, transfer) for transfer in state.reads)
    children.extend(p._complete_write(state, transfer) for transfer in state.writes)
    return children


def propose_graph(task, program, max_nodes=10000):
    if type(max_nodes) is not int or not 1 <= max_nodes <= 10000:
        raise ValueError('certificate graph cap must be 1..10000')
    seen = {p.initial_state(task)}
    queue = deque(seen)
    ordered = []
    while queue:
        state = queue.popleft()
        ordered.append(state)
        for child in successors(task, program, state):
            if child not in seen:
                if len(seen) >= max_nodes:
                    raise ValueError('certificate graph budget exhausted; proof unknown')
                seen.add(child)
                queue.append(child)
    return ordered


def _list(xs, render=str):
    return '['+', '.join(render(x) for x in xs)+']'


def _option(x):
    return 'none' if x is None else '(some '+str(x)+')'


def _state(s):
    pair=lambda x:'('+', '.join(map(str,x))+')'
    return '⟨'+', '.join([str(s.pc), _list(s.phases,lambda x:'.'+x),
                          _list(s.values,_option),_list(s.outputs,_option),
                          _list(s.reads,pair),_list(s.writes,pair)])+'⟩'


def _instruction(op):
    name={'wait_reads':'waitReads','wait_writes':'waitWrites'}.get(op.op,op.op)
    fields=p._op_fields_dict(op)
    # The native helper may include the op tag; never serialize that as an operand.
    args=[str(fields[k]) for k in ('slot','src','dst') if k in fields]
    return '.'+name+(' '+' '.join(args) if args else '')


def certificate_source(task, program, nodes):
    return '''import VeriTac.Tenstorrent.Protocol
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace SubmittedProtocol
open VeriTac.Tenstorrent
''' + f'''def task : Task := ⟨{task.input_pages}, {_list(task.expected)}, {task.slots}, {task.page_bytes}⟩
def program : List Instr := {_list(program,_instruction)}
def nodes : List State := [\n''' + ',\n'.join(_state(s) for s in nodes)+''']
theorem accepted : checkCertificate task program nodes = true := by decide +kernel
theorem safe (s : State) (h : Reachable task program s) :
    issueSafe task program s = true :=
  accepted_issueSafe task program nodes accepted s h
theorem progress (s : State) (h : Reachable task program s) :
    Acc (fun q s => q ∈ successors task program s) s :=
  accepted_accessible task program nodes accepted s h
theorem outputs {α : Type} (input : Nat → α) (s : State)
    (h : Reachable task program s) (done : successors task program s = []) :
    outputPayloads input s = task.expected.map (fun i => some (input i)) :=
  accepted_output_all_payloads task program nodes accepted s h done input
#print axioms accepted
#print axioms safe
#print axioms progress
#print axioms outputs
end SubmittedProtocol
'''


def audit_axioms(log):
    axioms={}
    for name in ('accepted','safe','progress','outputs'):
        qualified='SubmittedProtocol.'+name
        match=re.search(r"'"+re.escape(qualified)+r"' depends on axioms:\s*\[([^\]]*)\]",log)
        if match:
            axioms[name]=sorted({x.strip() for x in match.group(1).split(',') if x.strip()})
        elif "'"+qualified+"' does not depend on any axioms" in log:
            axioms[name]=[]
    return len(axioms)==4 and all(set(v)<=ALLOWED_AXIOMS for v in axioms.values()),axioms


def certify_protocol(task, program, output, timeout=600):
    from .adapter import validate_task,evaluate
    task=validate_task(task)
    program=json.loads(json.dumps(program))
    verdict=evaluate(task,program)
    if not verdict['accepted']:
        return {'accepted':False,'reason':'native rejection','verdict':verdict}
    if type(timeout) not in (int,float) or not 0<timeout<=3600:
        raise ValueError('timeout must be positive and at most 3600')
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    task_ir=p.parse_task(task)
    program_ir=p.parse_proposal({'program':program,'rationale':''}).program
    started=time.monotonic()
    try:
        nodes=propose_graph(task_ir,program_ir)
    except ValueError as error:
        result={'accepted':False,'reason':str(error),'scope':'protocol IR only'}
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    request={'task':task,'program':program}
    request_text=json.dumps(request,indent=2,sort_keys=True)+'\n'
    (output/'request.json').write_text(request_text)
    source=certificate_source(task_ir,program_ir,nodes)
    path=output/'protocol_certificate.lean'
    path.write_text(source)
    try:
        run=subprocess.run(['lake','env','lean','-s','65536',str(path)],cwd=ROOT,
                           text=True,capture_output=True,timeout=timeout)
        log=run.stdout+run.stderr
        ax_ok,axioms=audit_axioms(log)
        accepted=run.returncode==0 and ax_ok
        reason='kernel-checked protocol graph certificate' if accepted else 'Lean kernel check or axiom audit failed'
    except (OSError,subprocess.TimeoutExpired) as error:
        accepted,reason,axioms=False,str(error),{}
        log=str(error)
    if path.read_text()!=source or (output/'request.json').read_text()!=request_text:
        accepted,reason=False,'artifact changed during certification'
    (output/'protocol_certificate.log').write_text(log)
    result={'accepted':accepted,'reason':reason,
            'scope':'single-issuer asynchronous protocol IR; NOT emitted C++/machine code/hardware proof',
            'progress_assumption':'finite event semantics; physical scheduling and DMA eventual progress assumed',
            'payload_semantics':'immutable input pages and exact copies, represented by data origins',
            'nodes':len(nodes),'native_verdict':verdict,
            'source':str(path),'source_sha256':_digest(source.encode()),
            'request':str(output/'request.json'),'request_sha256':_digest(request_text.encode()),
            'log':str(output/'protocol_certificate.log'),'axioms':axioms,
            'kernel_check_seconds':round(time.monotonic()-started,3)}
    (output/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result
