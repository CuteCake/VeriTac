"""Tenstorrent restricted async-copy protocol specialization.

Advisory Python protocol IR and exhaustive asynchronous safety checker for
the frozen overnight scope v1 (see .lake/overnight-2026-09-18/CONTRACT.md):
one Tensix core, one data-movement RISC-V kernel, one NoC, asynchronous page
copies with arbitrary completion order over per-slot single-page CBs.

This package is a native filter only. The trusted formal acceptance of a
restricted protocol claim is a separate Lean proof worker; emitted C++, the
Metalium implementation, ttsim, and hardware correspondence are explicit
external assumptions and are not certified here.

Python standard library only; deterministic and reproducible.
"""

from specializations.tenstorrent_protocol.protocol import (
    Acquire,
    MAX_OUTPUT_PAGES,
    MAX_PAGE_BYTES,
    MAX_PROGRAM_OPS,
    MAX_SLOTS,
    MAX_INPUT_PAGES,
    DEFAULT_MAX_STATES,
    Op,
    Proposal,
    ProtocolError,
    Publish,
    Read,
    Release,
    Reserve,
    Result,
    SimulationResult,
    State,
    STATUS_ACCEPTED,
    STATUS_BUDGET,
    STATUS_DEADLOCK,
    STATUS_INCOMPLETE,
    STATUS_UNSAFE,
    STATUS_WRONG_OUTPUT,
    Task,
    WaitReads,
    WaitWrites,
    Write,
    check,
    initial_state,
    parse_op,
    parse_proposal,
    parse_task,
    replay_concrete,
    score_counts,
    score_key,
    serial_baseline,
    simulate,
)

__all__ = [
    "Acquire",
    "MAX_OUTPUT_PAGES",
    "MAX_PAGE_BYTES",
    "MAX_PROGRAM_OPS",
    "MAX_SLOTS",
    "MAX_INPUT_PAGES",
    "DEFAULT_MAX_STATES",
    "Op",
    "Proposal",
    "ProtocolError",
    "Publish",
    "Read",
    "Release",
    "Reserve",
    "Result",
    "SimulationResult",
    "State",
    "STATUS_ACCEPTED",
    "STATUS_BUDGET",
    "STATUS_DEADLOCK",
    "STATUS_INCOMPLETE",
    "STATUS_UNSAFE",
    "STATUS_WRONG_OUTPUT",
    "Task",
    "WaitReads",
    "WaitWrites",
    "Write",
    "check",
    "initial_state",
    "parse_op",
    "parse_proposal",
    "parse_task",
    "replay_concrete",
    "score_counts",
    "score_key",
    "serial_baseline",
    "simulate",
]
