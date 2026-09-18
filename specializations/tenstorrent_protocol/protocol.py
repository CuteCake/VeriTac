"""Restricted Tenstorrent async-copy protocol IR and exhaustive safety checker.

Scope (frozen interface v1, see .lake/overnight-2026-09-18/CONTRACT.md):
Blackhole, one Tensix core, one data-movement RISC-V kernel, one NoC,
asynchronous copies with arbitrary completion order. Each slot is a separate
circular buffer of exactly one page; the spec is an exact page permutation or
replication (output page d must equal input page expected[d] bit-for-bit).

This module is an ADVISORY native filter, not a formal authority. The trusted
formal acceptance for a restricted protocol claim is a separate Lean proof
worker; emitted C++, the Metalium implementation, ttsim, and hardware
correspondence are explicit external assumptions. This module never claims
unconditional silicon liveness: waiting without completion is stuttering
excluded from the finite event semantics, and physical progress requires DMA
completions under eventual scheduling fairness.

Failure taxonomy: unsafe / deadlock / wrong-output / incomplete / budget
(budget means the finite state cap was reached and safety is UNKNOWN — no
false acceptance and no false rejection are claimed).

Python standard library only; deterministic and reproducible.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

__all__ = [
    "ProtocolError",
    "Task",
    "parse_task",
    "Op",
    "Reserve",
    "Read",
    "WaitReads",
    "Publish",
    "Acquire",
    "Write",
    "WaitWrites",
    "Release",
    "parse_op",
    "Proposal",
    "parse_proposal",
    "State",
    "initial_state",
    "Result",
    "check",
    "SimulationResult",
    "simulate",
    "score_counts",
    "score_key",
    "serial_baseline",
    "replay_concrete",
    "MAX_INPUT_PAGES",
    "MAX_OUTPUT_PAGES",
    "MAX_SLOTS",
    "MAX_PAGE_BYTES",
    "MAX_PROGRAM_OPS",
    "DEFAULT_MAX_STATES",
    "STATUS_ACCEPTED",
    "STATUS_UNSAFE",
    "STATUS_DEADLOCK",
    "STATUS_WRONG_OUTPUT",
    "STATUS_INCOMPLETE",
    "STATUS_BUDGET",
]


# ---------------------------------------------------------------------------
# Limits (explicit, so "oversize" inputs are rejected by the exact schema)
# ---------------------------------------------------------------------------

MAX_INPUT_PAGES = 1 << 16
MAX_OUTPUT_PAGES = 1 << 16
MAX_SLOTS = 1 << 12
MAX_PAGE_BYTES = 1 << 24
MAX_PROGRAM_OPS = 1 << 16
MAX_ID_CHARS = 256
MAX_RATIONALE_CHARS = 8192
DEFAULT_MAX_STATES = 100_000
MAX_TRACE_STEPS = 1000

STATUS_ACCEPTED = "accepted"
STATUS_UNSAFE = "unsafe"
STATUS_DEADLOCK = "deadlock"
STATUS_WRONG_OUTPUT = "wrong-output"
STATUS_INCOMPLETE = "incomplete"
STATUS_BUDGET = "budget"

FREE = "free"
RESERVED = "reserved"
PUBLISHED = "published"
ACQUIRED = "acquired"

_OP_FIELDS = {
    "reserve": ("slot",),
    "read": ("slot", "src"),
    "wait_reads": (),
    "publish": ("slot",),
    "acquire": ("slot",),
    "write": ("slot", "dst"),
    "wait_writes": (),
    "release": ("slot",),
}


class ProtocolError(ValueError):
    """Raised when a task or proposal violates the exact frozen schema."""


def _require(condition, message):
    if not condition:
        raise ProtocolError(message)


def _is_int(value):
    # bool is a subclass of int in Python; the exact schema rejects it.
    return isinstance(value, int) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# Task IR (immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    id: str
    input_pages: int
    expected: tuple  # tuple[int, ...], source page index per output page
    slots: int
    page_bytes: int

    @property
    def output_pages(self):
        return len(self.expected)


def parse_task(obj):
    """Parse and freeze a task JSON object with an exact-key schema."""
    _require(isinstance(obj, dict), "task must be a JSON object")
    exact = {"id", "input_pages", "expected", "slots", "page_bytes"}
    _require(
        set(obj) == exact,
        "task fields must be exactly %s, got %s" % (sorted(exact), sorted(obj)),
    )
    task_id = obj["id"]
    _require(isinstance(task_id, str), "task id must be a string")
    _require(len(task_id) <= MAX_ID_CHARS, "task id exceeds %d chars" % MAX_ID_CHARS)
    input_pages = obj["input_pages"]
    _require(_is_int(input_pages), "input_pages must be an int (bool rejected)")
    _require(input_pages > 0, "input_pages must be positive")
    _require(
        input_pages <= MAX_INPUT_PAGES,
        "input_pages exceeds %d" % MAX_INPUT_PAGES,
    )
    expected = obj["expected"]
    _require(isinstance(expected, list), "expected must be a list")
    _require(
        1 <= len(expected) <= MAX_OUTPUT_PAGES,
        "expected length must be in 1..%d" % MAX_OUTPUT_PAGES,
    )
    for i, src in enumerate(expected):
        _require(_is_int(src), "expected[%d] must be an int (bool rejected)" % i)
        _require(
            0 <= src < input_pages,
            "expected[%d]=%r out of range [0,%d)" % (i, src, input_pages),
        )
    slots = obj["slots"]
    _require(_is_int(slots), "slots must be an int (bool rejected)")
    _require(slots > 0, "slots must be positive")
    _require(slots <= MAX_SLOTS, "slots exceeds %d" % MAX_SLOTS)
    page_bytes = obj["page_bytes"]
    _require(_is_int(page_bytes), "page_bytes must be an int (bool rejected)")
    _require(page_bytes > 0, "page_bytes must be positive")
    _require(
        page_bytes % 32 == 0,
        "page_bytes must be a multiple of 32",
    )
    _require(page_bytes <= MAX_PAGE_BYTES, "page_bytes exceeds %d" % MAX_PAGE_BYTES)
    return Task(
        id=task_id,
        input_pages=input_pages,
        expected=tuple(expected),
        slots=slots,
        page_bytes=page_bytes,
    )


# ---------------------------------------------------------------------------
# Instruction IR (immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reserve:
    slot: int
    op = "reserve"


@dataclass(frozen=True)
class Read:
    slot: int
    src: int
    op = "read"


@dataclass(frozen=True)
class WaitReads:
    op = "wait_reads"


@dataclass(frozen=True)
class Publish:
    slot: int
    op = "publish"


@dataclass(frozen=True)
class Acquire:
    slot: int
    op = "acquire"


@dataclass(frozen=True)
class Write:
    slot: int
    dst: int
    op = "write"


@dataclass(frozen=True)
class WaitWrites:
    op = "wait_writes"


@dataclass(frozen=True)
class Release:
    slot: int
    op = "release"


Op = (Reserve, Read, WaitReads, Publish, Acquire, Write, WaitWrites, Release)


def parse_op(obj):
    """Parse one instruction object with exact per-op fields."""
    _require(isinstance(obj, dict), "instruction must be a JSON object")
    name = obj.get("op")
    _require(isinstance(name, str), "instruction 'op' must be a string")
    _require(name in _OP_FIELDS, "unknown op %r" % (name,))
    fields = _OP_FIELDS[name]
    _require(
        set(obj) == set(fields) | {"op"},
        "%s fields must be exactly %s, got %s"
        % (name, sorted(fields) + ["op"], sorted(obj)),
    )
    values = []
    for field in fields:
        value = obj[field]
        _require(_is_int(value), "%s %s must be an int (bool rejected)" % (name, field))
        values.append(value)
    if name == "reserve":
        return Reserve(*values)
    if name == "read":
        return Read(*values)
    if name == "wait_reads":
        return WaitReads()
    if name == "publish":
        return Publish(*values)
    if name == "acquire":
        return Acquire(*values)
    if name == "write":
        return Write(*values)
    if name == "wait_writes":
        return WaitWrites()
    return Release(*values)


def _op_fields_dict(op):
    if isinstance(op, (WaitReads, WaitWrites)):
        return {}
    if isinstance(op, (Read,)):
        return {"slot": op.slot, "src": op.src}
    if isinstance(op, (Write,)):
        return {"slot": op.slot, "dst": op.dst}
    return {"slot": op.slot}


@dataclass(frozen=True)
class Proposal:
    program: tuple  # tuple[Op, ...]
    rationale: str


def parse_proposal(obj):
    """Parse and freeze a proposal JSON object with an exact-key schema."""
    _require(isinstance(obj, dict), "proposal must be a JSON object")
    exact = {"program", "rationale"}
    _require(
        set(obj) == exact,
        "proposal fields must be exactly %s, got %s" % (sorted(exact), sorted(obj)),
    )
    program = obj["program"]
    _require(isinstance(program, list), "program must be a list")
    _require(
        len(program) <= MAX_PROGRAM_OPS,
        "program exceeds %d instructions" % MAX_PROGRAM_OPS,
    )
    rationale = obj["rationale"]
    _require(isinstance(rationale, str), "rationale must be a string")
    _require(
        len(rationale) <= MAX_RATIONALE_CHARS,
        "rationale exceeds %d chars" % MAX_RATIONALE_CHARS,
    )
    return Proposal(program=tuple(parse_op(instr) for instr in program), rationale=rationale)


def _validate_op_ranges(op, task):
    """Return an error string if the op's indices are statically out of range."""
    if isinstance(op, Reserve) or isinstance(op, Publish) or isinstance(op, Acquire) \
            or isinstance(op, Release):
        if not (0 <= op.slot < task.slots):
            return "%s slot %d out of range [0,%d)" % (op.op, op.slot, task.slots)
    elif isinstance(op, Read):
        if not (0 <= op.slot < task.slots):
            return "read slot %d out of range [0,%d)" % (op.slot, task.slots)
        if not (0 <= op.src < task.input_pages):
            return "read src %d out of range [0,%d)" % (op.src, task.input_pages)
    elif isinstance(op, Write):
        if not (0 <= op.slot < task.slots):
            return "write slot %d out of range [0,%d)" % (op.slot, task.slots)
        if not (0 <= op.dst < task.output_pages):
            return "write dst %d out of range [0,%d)" % (op.dst, task.output_pages)
    return None


# ---------------------------------------------------------------------------
# Protocol state (immutable; pending sets canonicalized by sorting)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class State:
    pc: int
    phases: tuple          # per slot: free/reserved/published/acquired
    values: tuple          # per slot: origin page index or None
    reads: tuple           # ((slot, src), ...)
    writes: tuple          # ((slot, dst, origin), ...)
    outputs: tuple         # per output page: origin page index or None


def initial_state(task):
    return State(
        pc=0,
        phases=(FREE,) * task.slots,
        values=(None,) * task.slots,
        reads=(),
        writes=(),
        outputs=(None,) * task.output_pages,
    )


def _set_at(items, index, value):
    return items[:index] + (value,) + items[index + 1:]


def _complete_read(state, transfer):
    slot, src = transfer
    return replace(
        state,
        reads=tuple(t for t in state.reads if t != transfer),
        values=_set_at(state.values, slot, src),
    )


def _complete_write(state, transfer):
    slot, dst, origin = transfer
    return replace(
        state,
        writes=tuple(t for t in state.writes if t != transfer),
        outputs=_set_at(state.outputs, dst, origin),
    )


def _issue(state, op, task):
    """Apply a non-wait op. Returns (new_state, None) or (None, reason)."""
    phase = state.phases[op.slot]
    if isinstance(op, Reserve):
        if phase != FREE:
            return None, "reserve requires free slot %d (phase %s)" % (op.slot, phase)
        return replace(
            state,
            pc=state.pc + 1,
            phases=_set_at(state.phases, op.slot, RESERVED),
            values=_set_at(state.values, op.slot, None),
        ), None
    if isinstance(op, Read):
        if phase != RESERVED:
            return None, "read requires reserved slot %d (phase %s)" % (op.slot, phase)
        if any(r[0] == op.slot for r in state.reads):
            return None, "read requires no pending read for slot %d" % op.slot
        if any(w[0] == op.slot for w in state.writes):
            return None, "read requires no pending write for slot %d" % op.slot
        return replace(
            state,
            pc=state.pc + 1,
            values=_set_at(state.values, op.slot, None),
            reads=tuple(sorted(state.reads + ((op.slot, op.src),))),
        ), None
    if isinstance(op, Publish):
        if phase != RESERVED:
            return None, "publish requires reserved slot %d (phase %s)" % (op.slot, phase)
        if state.values[op.slot] is None:
            return None, "publish requires initialized value for slot %d" % op.slot
        if any(r[0] == op.slot for r in state.reads):
            return None, "publish requires no pending read for slot %d" % op.slot
        return replace(
            state,
            pc=state.pc + 1,
            phases=_set_at(state.phases, op.slot, PUBLISHED),
        ), None
    if isinstance(op, Acquire):
        if phase != PUBLISHED:
            return None, "acquire requires published slot %d (phase %s)" % (op.slot, phase)
        return replace(
            state,
            pc=state.pc + 1,
            phases=_set_at(state.phases, op.slot, ACQUIRED),
        ), None
    if isinstance(op, Write):
        if phase != ACQUIRED:
            return None, "write requires acquired slot %d (phase %s)" % (op.slot, phase)
        if state.values[op.slot] is None:
            return None, "write requires initialized value for slot %d" % op.slot
        if any(r[0] == op.slot for r in state.reads):
            return None, "write requires no pending read for slot %d" % op.slot
        if any(w[1] == op.dst for w in state.writes):
            return None, "write requires no pending write for destination %d" % op.dst
        return replace(
            state,
            pc=state.pc + 1,
            writes=tuple(
                sorted(state.writes + ((op.slot, op.dst, state.values[op.slot]),))
            ),
        ), None
    if isinstance(op, Release):
        if phase != ACQUIRED:
            return None, "release requires acquired slot %d (phase %s)" % (op.slot, phase)
        if any(r[0] == op.slot for r in state.reads) or any(
            w[0] == op.slot for w in state.writes
        ):
            return None, "release requires no pending transfer for slot %d" % op.slot
        return replace(
            state,
            pc=state.pc + 1,
            phases=_set_at(state.phases, op.slot, FREE),
            values=_set_at(state.values, op.slot, None),
        ), None
    return None, "unknown op %r" % (op,)


def _is_terminal(state, program_len):
    return state.pc == program_len and not state.reads and not state.writes


# ---------------------------------------------------------------------------
# Exhaustive asynchronous safety checker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Result:
    status: str
    reason: str | None
    trace: tuple            # tuple of step dicts (counterexample / terminal path)
    states_explored: int
    trace_truncated: bool

    def to_dict(self):
        return {
            "status": self.status,
            "reason": self.reason,
            "trace": list(self.trace),
            "states_explored": self.states_explored,
            "trace_truncated": self.trace_truncated,
        }


def _child_step(action, **fields):
    step = {"action": action}
    step.update(fields)
    return step


def check(task, proposal, max_states=DEFAULT_MAX_STATES):
    """Exhaustively search all reachable interleavings of issuer ops and DMA
    completions.

    Any invalid non-wait op at ANY reachable state is unsafe, even if some
    other completion ordering would have avoided it (no implicit assumption
    that DMA finishes early). Blocked waits are not errors. Acceptance requires
    every terminal execution (pc=end, no pending transfers) to have all slots
    free and every output page equal to its expected origin.
    """
    _require(type(max_states) is int and max_states > 0, "max_states must be a positive integer")
    if isinstance(task, dict):
        task = parse_task(task)
    if isinstance(proposal, dict):
        proposal = parse_proposal(proposal)
    program = proposal.program
    for op in program:
        err = _validate_op_ranges(op, task)
        if err is not None:
            return Result(
                status=STATUS_UNSAFE,
                reason="instruction 0-based index check: %s" % err,
                trace=(_child_step("invalid", reason=err),),
                states_explored=0,
                trace_truncated=False,
            )

    start = initial_state(task)
    parents = {start: (None, None)}
    queue = deque([start])
    explored = 0
    found_good_terminal = []
    first_wrong_output = None
    first_incomplete = None
    first_deadlock = None

    def rebuild(state, last_step):
        steps = []
        node = state
        while node is not None:
            parent, step = parents[node]
            if step is not None:
                steps.append(step)
            node = parent
        steps.reverse()
        if last_step is not None:
            steps.append(last_step)
        truncated = len(steps) > MAX_TRACE_STEPS
        if truncated:
            steps = steps[:MAX_TRACE_STEPS]
            steps.append({"action": "trace_truncated"})
        return tuple(steps), truncated

    def verdict(status, reason, state, last_step):
        trace, truncated = rebuild(state, last_step)
        return Result(status, reason, trace, explored, truncated)

    while queue:
        state = queue.popleft()
        explored += 1
        if explored > max_states:
            # Honest unknown: the state cap was reached before any verdict.
            return Result(
                status=STATUS_BUDGET,
                reason=(
                    "state cap %d reached after exploring %d states; "
                    "safety unknown (neither accepted nor rejected)"
                    % (max_states, explored)
                ),
                trace=(),
                states_explored=explored,
                trace_truncated=False,
            )
        transitions = []
        if state.pc < len(program):
            op = program[state.pc]
            if isinstance(op, WaitReads):
                if not state.reads:
                    transitions.append(
                        (replace(state, pc=state.pc + 1),
                         _child_step("issue", op="wait_reads"))
                    )
                # else: blocking wait; pending reads can still complete.
            elif isinstance(op, WaitWrites):
                if not state.writes:
                    transitions.append(
                        (replace(state, pc=state.pc + 1),
                         _child_step("issue", op="wait_writes"))
                    )
                # else: blocking wait; pending writes can still complete.
            else:
                child, err = _issue(state, op, task)
                if err is not None:
                    return verdict(
                        STATUS_UNSAFE,
                        "pc=%d: %s is invalid here: %s" % (state.pc, op.op, err),
                        state,
                        _child_step("invalid", pc=state.pc, op=op.op, reason=err,
                                    **_op_fields_dict(op)),
                    )
                transitions.append(
                    (child, _child_step("issue", op=op.op, **_op_fields_dict(op)))
                )
        for transfer in state.reads:
            transitions.append(
                (_complete_read(state, transfer),
                 _child_step("complete_read", slot=transfer[0], src=transfer[1]))
            )
        for transfer in state.writes:
            transitions.append(
                (_complete_write(state, transfer),
                 _child_step("complete_write", slot=transfer[0],
                             dst=transfer[1], origin=transfer[2]))
            )
        if not transitions:
            if _is_terminal(state, len(program)):
                outputs_ok = tuple(state.outputs) == task.expected
                slots_free = all(p == FREE for p in state.phases)
                if outputs_ok and slots_free:
                    found_good_terminal.append(state)
                elif not outputs_ok:
                    if first_wrong_output is None:
                        first_wrong_output = (
                            state,
                            "output origins %s do not match expected %s"
                            % (list(state.outputs), list(task.expected)),
                        )
                elif first_incomplete is None:
                    first_incomplete = (
                        state,
                        "program ended with slots not all free: %s"
                        % (list(state.phases),),
                    )
            elif first_deadlock is None:
                first_deadlock = (
                    state,
                    "pc=%d with no available transition and pending work stalled"
                    % state.pc,
                )
            continue
        for child, step in transitions:
            if child not in parents:
                parents[child] = (state, step)
                queue.append(child)

    # Queue exhausted without an unsafe issue. Acceptance requires a good
    # terminal execution AND no wrong-output / incomplete / deadlocked
    # terminal anywhere in the reachable state space.
    if first_wrong_output is not None:
        return verdict(STATUS_WRONG_OUTPUT, first_wrong_output[1],
                       first_wrong_output[0], None)
    if first_incomplete is not None:
        return verdict(STATUS_INCOMPLETE, first_incomplete[1],
                       first_incomplete[0], None)
    if first_deadlock is not None:
        return verdict(STATUS_DEADLOCK, first_deadlock[1], first_deadlock[0], None)
    if found_good_terminal:
        return verdict(STATUS_ACCEPTED, None, found_good_terminal[0], None)
    return Result(
        status=STATUS_INCOMPLETE,
        reason="no terminal execution reached",
        trace=(),
        states_explored=explored,
        trace_truncated=False,
    )


# ---------------------------------------------------------------------------
# Canonical single-schedule simulation, scoring, baseline, concrete replay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationResult:
    status: str
    reason: str | None
    counts: dict
    steps: tuple

    def to_dict(self):
        return {
            "status": self.status,
            "reason": self.reason,
            "counts": dict(self.counts),
            "steps": list(self.steps),
        }


def simulate(task, proposal):
    """Run one canonical schedule (issue eagerly; when a wait blocks, complete
    pending transfers in canonical sorted-tuple order). This is ONE concrete schedule, not a
    safety proof; use check() for the exhaustive search.
    """
    if isinstance(task, dict):
        task = parse_task(task)
    if isinstance(proposal, dict):
        proposal = parse_proposal(proposal)
    program = proposal.program
    for op in program:
        err = _validate_op_ranges(op, task)
        if err is not None:
            return SimulationResult(STATUS_UNSAFE, err, {}, ())
    state = initial_state(task)
    steps = []
    peak_live = 0
    reads_issued = writes_issued = 0

    def note_live():
        nonlocal peak_live
        live = sum(1 for p in state.phases if p != FREE)
        peak_live = max(peak_live, live)

    while state.pc < len(program):
        op = program[state.pc]
        if isinstance(op, WaitReads) and state.reads:
            transfer = state.reads[0]
            state = _complete_read(state, transfer)
            steps.append(_child_step("complete_read", slot=transfer[0], src=transfer[1]))
            continue
        if isinstance(op, WaitWrites) and state.writes:
            transfer = state.writes[0]
            state = _complete_write(state, transfer)
            steps.append(_child_step("complete_write", slot=transfer[0],
                                     dst=transfer[1], origin=transfer[2]))
            continue
        if isinstance(op, (WaitReads, WaitWrites)):
            state = replace(state, pc=state.pc + 1)
            steps.append(_child_step("issue", op=op.op))
            note_live()
            continue
        child, err = _issue(state, op, task)
        if err is not None:
            return SimulationResult(
                STATUS_UNSAFE,
                "pc=%d: %s is invalid here: %s" % (state.pc, op.op, err),
                {}, tuple(steps),
            )
        state = child
        if isinstance(op, Read):
            reads_issued += 1
        elif isinstance(op, Write):
            writes_issued += 1
        steps.append(_child_step("issue", op=op.op, **_op_fields_dict(op)))
        note_live()
    while state.reads or state.writes:
        if state.reads:
            transfer = state.reads[0]
            state = _complete_read(state, transfer)
            steps.append(_child_step("complete_read", slot=transfer[0], src=transfer[1]))
        else:
            transfer = state.writes[0]
            state = _complete_write(state, transfer)
            steps.append(_child_step("complete_write", slot=transfer[0],
                                     dst=transfer[1], origin=transfer[2]))
    if not _is_terminal(state, len(program)):
        return SimulationResult(STATUS_INCOMPLETE, "program did not drain", {}, tuple(steps))
    if tuple(state.outputs) != task.expected or not all(p == FREE for p in state.phases):
        return SimulationResult(
            STATUS_WRONG_OUTPUT,
            "terminal outputs %s vs expected %s (phases %s)"
            % (list(state.outputs), list(task.expected), list(state.phases)),
            {}, tuple(steps),
        )
    wait_reads = sum(1 for op in program if isinstance(op, WaitReads))
    wait_writes = sum(1 for op in program if isinstance(op, WaitWrites))
    counts = {
        "wait_reads": wait_reads,
        "wait_writes": wait_writes,
        "waits": wait_reads + wait_writes,
        "commands": len(program),
        "peak_live_slots": peak_live,
        "reads": reads_issued,
        "writes": writes_issued,
        "input_bytes": reads_issued * task.page_bytes,
        "output_bytes": writes_issued * task.page_bytes,
    }
    return SimulationResult(STATUS_ACCEPTED, None, counts, tuple(steps))


def score_counts(counts):
    """Lexicographic score key: waits first, then total commands, then peak
    live slots. Lower is better. Barrier counts make no performance claim;
    input/output bytes are reported separately by simulate()."""
    return (counts["waits"], counts["commands"], counts["peak_live_slots"])


def score_key(simulation):
    _require(simulation.status == STATUS_ACCEPTED, "cannot score an unsuccessful simulation")
    return score_counts(simulation.counts)


def serial_baseline(task):
    """Baseline proposal: serial copies, one page in flight at a time in slot
    0. Correct under arbitrary completion order; no optimality claim.
    """
    if isinstance(task, dict):
        task = parse_task(task)
    instructions = []
    for d, src in enumerate(task.expected):
        instructions.extend([
            Reserve(0),
            Read(0, src),
            WaitReads(),
            Publish(0),
            Acquire(0),
            Write(0, d),
            WaitWrites(),
            Release(0),
        ])
    return Proposal(
        program=tuple(instructions),
        rationale="serial baseline: one page in flight at a time using slot 0",
    )


def replay_concrete(task, proposal):
    """Replay ONE chosen schedule (the canonical one) with concrete page
    payloads and verify bit-for-bit equality of outputs against their expected
    input pages.

    This is a single-schedule, single-payload demonstration. It is NOT a
    proof for all inputs and all completion orders; use check() for the
    exhaustive protocol search and the Lean worker for formal acceptance.
    """
    if isinstance(task, dict):
        task = parse_task(task)
    if isinstance(proposal, dict):
        proposal = parse_proposal(proposal)
    sim = simulate(task, proposal)
    if sim.status != STATUS_ACCEPTED:
        return {
            "status": sim.status,
            "reason": sim.reason,
            "all_outputs_match": False,
            "outputs_match": None,
        }

    def payload(page):
        return bytes((page * 7 + j) % 256 for j in range(task.page_bytes))

    inputs = [payload(i) for i in range(task.input_pages)]
    slots = [None] * task.slots
    outputs = [None] * task.output_pages
    for step in sim.steps:
        if step["action"] == "complete_read":
            slots[step["slot"]] = inputs[step["src"]]
        elif step["action"] == "complete_write":
            outputs[step["dst"]] = slots[step["slot"]]
    outputs_match = [
        outputs[d] is not None and outputs[d] == inputs[task.expected[d]]
        for d in range(task.output_pages)
    ]
    return {
        "status": "replayed",
        "reason": None,
        "all_outputs_match": all(outputs_match),
        "outputs_match": outputs_match,
    }
