"""Small public adapter around the protocol IR, native filter and Lean path."""
from dataclasses import asdict
from . import protocol as p


def validate_task(task):
    parsed = p.parse_task(task)
    if not 1 <= parsed.input_pages <= 16 or not 1 <= len(parsed.expected) <= 16:
        raise ValueError('initial backend supports 1..16 input/output pages')
    if not 1 <= parsed.slots <= 8:
        raise ValueError('initial backend supports 1..8 CB slots')
    if not 32 <= parsed.page_bytes <= 16384:
        raise ValueError('initial backend supports 32..16384 bytes per page')
    return {**asdict(parsed), 'expected': list(parsed.expected)}


def encode_program(proposal):
    return [{'op': op.op, **asdict(op)} for op in proposal.program]


def baseline(task):
    return encode_program(p.serial_baseline(validate_task(task)))


def evaluate(task, program, max_states=200000):
    task = validate_task(task)
    if not isinstance(program, list) or not 1 <= len(program) <= 128:
        raise ValueError('initial backend supports 1..128 instructions')
    if type(max_states) is not int or not 1 <= max_states <= 200000:
        raise ValueError('state budget must be an integer in 1..200000')
    proposal = p.parse_proposal({'program': program, 'rationale': ''})
    result = p.check(task, proposal, max_states=max_states).to_dict()
    result['accepted'] = result['status'] == p.STATUS_ACCEPTED
    result['authority'] = 'native protocol filter, not a Lean certificate'
    if result['accepted']:
        counts = {op: sum(x.op == op for x in proposal.program)
                  for op in ('read', 'write', 'wait_reads', 'wait_writes')}
        live, peak = 0, 0
        for op in proposal.program:
            if op.op == 'reserve':
                live += 1
                peak = max(peak, live)
            elif op.op == 'release':
                live -= 1
        counts.update(commands=len(program), peak_live_slots=peak,
                      input_bytes=counts['read']*task['page_bytes'],
                      output_bytes=counts['write']*task['page_bytes'])
        counts['barriers'] = counts['wait_reads'] + counts['wait_writes']
        result['counts'] = counts
        result['objective'] = [counts['barriers'], len(program), peak]
    return result


def certify(task, program, output, timeout=600):
    task = validate_task(task)
    verdict = evaluate(task, program)
    if not verdict['accepted']:
        return {'accepted': False, 'reason': 'native filter did not accept', 'verdict': verdict}
    from .certificate import certify_protocol
    return certify_protocol(task, program, output, timeout=timeout)
