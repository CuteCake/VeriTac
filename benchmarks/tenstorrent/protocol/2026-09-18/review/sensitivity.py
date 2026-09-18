"""Sensitivity: mutate protocol._issue in known ways, confirm differential
fuzz catches each mutation (oracle is not blind)."""
import sys
from dataclasses import replace

sys.path.insert(0, ".")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from specializations.tenstorrent_protocol import protocol as p
import diff_fuzz3

original_issue = p._issue


def mut_publish_no_pending_read(state, op, task):
    child, err = original_issue(state, op, task)
    if err is not None and isinstance(op, p.Publish) and "pending read" in err:
        # pretend the pending-read precondition on publish doesn't exist
        if any(r[0] == op.slot for r in state.reads):
            return replace(state, pc=state.pc + 1,
                           phases=p._set_at(state.phases, op.slot, p.PUBLISHED)), None
    return child, err


def mut_write_dst_check(state, op, task):
    child, err = original_issue(state, op, task)
    if err is not None and isinstance(op, p.Write) and "destination" in err:
        return replace(state, pc=state.pc + 1,
                       writes=tuple(sorted(state.writes + ((op.slot, op.dst, state.values[op.slot]),)))), None
    return child, err


def run_once(name, mutant):
    p._issue = mutant
    rng = __import__("random").Random(99)
    mism = 0
    trials = 0
    REF = diff_fuzz3.REF
    for trial in range(600):
        n_in = rng.choice([1, 2, 3]); n_out = rng.choice([1, 2, 3]); slots = rng.choice([1, 2, 3])
        expected = [rng.randrange(n_in) for _ in range(n_out)]
        task = {"id": "s", "input_pages": n_in, "expected": expected, "slots": slots, "page_bytes": 32}
        program = diff_fuzz3.correct_pipeline(rng, n_in, expected, slots)
        if rng.random() < 0.6:
            program = diff_fuzz3.mutate(rng, program, n_in, n_out, slots)
        if not program:
            continue
        ref = REF[p_protocol_ref_check(task, program)] if False else REF[ref_check_wrap(task, program)]
        got = p.check(task, {"program": program, "rationale": ""}, max_states=50000).status
        trials += 1
        if got != ref:
            mism += 1
    p._issue = original_issue
    print("%-28s trials=%d caught=%d %s" % (name, trials, mism,
                                            "OK" if mism else "ORACLE BLIND"))


import diff_fuzz
ref_check_wrap = diff_fuzz.ref_check
p_protocol_ref_check = None

run_once("publish ignores pending read", mut_publish_no_pending_read)
run_once("write ignores pending dst", mut_write_dst_check)

# Random near-valid generation does not necessarily activate a weakened guard.
# This directed case does: two outstanding writes to the same destination.
task = {'id':'directed','input_pages':1,'expected':[0],'slots':1,'page_bytes':32}
program = [
 {'op':'reserve','slot':0}, {'op':'read','slot':0,'src':0}, {'op':'wait_reads'},
 {'op':'publish','slot':0}, {'op':'acquire','slot':0},
 {'op':'write','slot':0,'dst':0}, {'op':'write','slot':0,'dst':0},
 {'op':'wait_writes'}, {'op':'release','slot':0}]
reference=diff_fuzz3.REF[diff_fuzz.ref_check(task,program)]
original=p.check(task,{'program':program,'rationale':''}).status
try:
 p._issue=mut_write_dst_check
 mutated=p.check(task,{'program':program,'rationale':''}).status
finally:
 p._issue=original_issue
print('directed duplicate-destination case:',{'reference':reference,'original':original,'mutant':mutated})
if original!=reference or original==mutated:
 raise SystemExit('directed mutation sensitivity failed')
