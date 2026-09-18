"""Experimental compact proposal protocol over the unchanged command checker.

Kept separate from the preregistered flat-command experiment. Expansion is
untrusted; only the concrete commands/bytes can obtain a correctness certificate.
"""
import json
from . import blind_search as flat
from . import schedule_ir

PROTOCOL_ID = "bounded_schedule_v1"


def build_prompt(plan, feedback=None):
    semantics = flat.instruction_text(plan)
    semantics = semantics.replace('"commands" is a list of command objects executed in order.',
                                  'Concrete command nodes below execute in order after loop expansion.')
    start = semantics.index("Response schema (exact;")
    objective = semantics[semantics.index("Objective:", start):]
    semantics = semantics[:start]
    grammar = '''Response schema: exactly {"program": [<nodes>], "rationale": "<non-empty string>"}.
No extra keys, duplicate keys, comments, or text outside the JSON object.

A node is either a concrete command with the exact fields above, or a loop:
{"for": "i", "start": 0, "stop": {"expr": "M // 16"}, "step": 1, "body": [<nodes>]}.
Loop bounds follow integer range(start, stop, step), with stop excluded and a nonzero step.
Loops nest; a variable is visible only in its body and cannot shadow another binding.
The immutable integer constants are M, N, K, DIM, SPAD_ROWS, ACC_ROWS, supplied by the fixed task.
Each numeric or Boolean command field can be a literal or {"expr": "expression"}.
The kind and input buffer fields must remain literal strings.
Expressions support integer +, -, *, //, %, unary +/-; one comparison ==, !=, <, <=, >, >=;
and conditional expressions such as "16 if i == 0 else 32". No function calls (including min/max),
attributes, indexing, bitwise operations, imports, or assignment are allowed. Arithmetic accepts
integers, not Booleans. Comparisons return Booleans. Conditional tests must be Booleans.
All integer intermediate magnitudes are at most 2^64-1. Expressions are at most 256 characters,
loop nesting is at most 12, total loop iterations at most 100000, and expanded commands at most 4096.
You may use a flat sequence of literal commands if preferred. The expansion language provides
no kernel templates or optimization strategies. Only the expanded commands are validated;
expansion success is not correctness. Rationale is not executed.
'''
    prompt = semantics + grammar + "\n" + objective
    if feedback is not None:
        prompt += "\n\nYour earlier proposals and factual validator feedback:\n" + json.dumps(feedback, sort_keys=True)
    return prompt


def parse_compact(text, plan, max_commands=4096):
    candidates = list(flat.extract_json_candidates(text))
    try:
        document = flat.strict_json_loads(candidates[0]) if len(candidates) == 1 else None
    except ValueError as error:
        raise flat.MalformedProposal(str(error)) from error
    if not isinstance(document, dict) or set(document) != {"program", "rationale"}:
        raise flat.ProposalRejected("compact reply must contain exactly program and rationale; task is immutable")
    if not isinstance(document["rationale"], str) or not document["rationale"].strip():
        raise flat.ProposalRejected("non-empty rationale required")
    constants = dict(zip(("M", "N", "K", "DIM", "SPAD_ROWS", "ACC_ROWS"),
                         (plan[k] for k in ("m", "n", "k", "dim", "scratchpad_rows", "accumulator_rows"))))
    try:
        expanded = schedule_ir.expand_program(document["program"], constants, max_commands=max_commands)
        if not expanded:
            raise ValueError("empty expanded program")
        commands = []
        for entry in expanded:
            command = flat.validate_command(entry)
            flat.validate_subset(command)
            commands.append(command)
    except (ValueError, TypeError) as error:
        raise flat.ProposalRejected("compact expansion rejected: " + str(error)) from error
    return commands, document["rationale"]
