"""Tests for the restricted Tenstorrent protocol IR and exhaustive checker.

Advisory Python layer only; the Lean worker remains the formal authority.
Run with: PYTHONPATH=. python3 -m unittest tests.test_tenstorrent_protocol
"""

import copy
import dataclasses
import unittest

from specializations.tenstorrent_protocol import (
    Acquire,
    Proposal,
    ProtocolError,
    Publish,
    Read,
    Release,
    Reserve,
    Result,
    SimulationResult,
    STATUS_ACCEPTED,
    STATUS_BUDGET,
    STATUS_INCOMPLETE,
    STATUS_UNSAFE,
    STATUS_WRONG_OUTPUT,
    Task,
    WaitReads,
    WaitWrites,
    Write,
    check,
    parse_proposal,
    parse_task,
    replay_concrete,
    score_key,
    serial_baseline,
    simulate,
)


def make_task(**overrides):
    task = {
        "id": "tt-copy",
        "input_pages": 2,
        "expected": [1, 0],
        "slots": 2,
        "page_bytes": 1024,
    }
    task.update(overrides)
    return task


def proposal(instructions, rationale="test"):
    return Proposal(program=tuple(instructions), rationale=rationale)


BATCHED_SWAP = proposal([
    Reserve(0), Read(0, 1),
    Reserve(1), Read(1, 0),
    WaitReads(),
    Publish(0), Publish(1),
    Acquire(0), Acquire(1),
    Write(0, 0), Write(1, 1),
    WaitWrites(),
    Release(0), Release(1),
])


class TaskSchemaTests(unittest.TestCase):
    def test_parse_task_freezes_fields(self):
        task = parse_task(make_task())
        self.assertEqual(task.input_pages, 2)
        self.assertEqual(task.expected, (1, 0))
        self.assertEqual(task.output_pages, 2)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            task.slots = 3

    def test_parse_task_does_not_mutate_input(self):
        raw = make_task()
        snapshot = copy.deepcopy(raw)
        parse_task(raw)
        self.assertEqual(raw, snapshot)

    def test_missing_field_rejected(self):
        raw = make_task()
        del raw["page_bytes"]
        with self.assertRaises(ProtocolError):
            parse_task(raw)

    def test_extra_field_rejected(self):
        with self.assertRaises(ProtocolError):
            parse_task(make_task(extra=1))

    def test_bool_as_int_rejected(self):
        with self.assertRaises(ProtocolError):
            parse_task(make_task(input_pages=True))
        with self.assertRaises(ProtocolError):
            parse_task(make_task(expected=[True, 0]))

    def test_float_and_nan_rejected(self):
        with self.assertRaises(ProtocolError):
            parse_task(make_task(input_pages=2.0))
        with self.assertRaises(ProtocolError):
            parse_task(make_task(page_bytes=float("nan")))

    def test_zero_and_oversize_rejected(self):
        with self.assertRaises(ProtocolError):
            parse_task(make_task(input_pages=0))
        with self.assertRaises(ProtocolError):
            parse_task(make_task(slots=0))
        with self.assertRaises(ProtocolError):
            parse_task(make_task(slots=1 << 20))

    def test_page_bytes_multiple_of_32(self):
        with self.assertRaises(ProtocolError):
            parse_task(make_task(page_bytes=16))
        parse_task(make_task(page_bytes=32))

    def test_expected_out_of_range_rejected(self):
        with self.assertRaises(ProtocolError):
            parse_task(make_task(expected=[2, 0]))
        with self.assertRaises(ProtocolError):
            parse_task(make_task(expected=[-1, 0]))

    def test_non_string_id_rejected(self):
        with self.assertRaises(ProtocolError):
            parse_task(make_task(id=7))


class ProposalSchemaTests(unittest.TestCase):
    def test_parse_proposal_exact_fields(self):
        p = parse_proposal({"program": [{"op": "wait_reads"}], "rationale": "r"})
        self.assertEqual(len(p.program), 1)
        self.assertIsInstance(p.program[0], WaitReads)

    def test_unknown_op_rejected(self):
        with self.assertRaises(ProtocolError):
            parse_proposal({"program": [{"op": "dma"}], "rationale": "r"})

    def test_op_extra_field_rejected(self):
        with self.assertRaises(ProtocolError):
            parse_proposal({
                "program": [{"op": "read", "slot": 0, "src": 0, "extra": 1}],
                "rationale": "r",
            })

    def test_op_bool_slot_rejected(self):
        with self.assertRaises(ProtocolError):
            parse_proposal({
                "program": [{"op": "reserve", "slot": False}],
                "rationale": "r",
            })

    def test_missing_rationale_rejected(self):
        with self.assertRaises(ProtocolError):
            parse_proposal({"program": []})

    def test_proposal_is_immutable(self):
        p = parse_proposal({"program": [], "rationale": "r"})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            p.program = ()


class AcceptedScheduleTests(unittest.TestCase):
    def test_serial_baseline_accepted(self):
        task = parse_task(make_task())
        result = check(task, serial_baseline(task))
        self.assertEqual(result.status, STATUS_ACCEPTED, result.to_dict())

    def test_batched_schedule_accepted_with_completion_reordering(self):
        task = parse_task(make_task())
        result = check(task, BATCHED_SWAP)
        self.assertEqual(result.status, STATUS_ACCEPTED, result.to_dict())

    def test_replication_accepted(self):
        task = parse_task(make_task(input_pages=1, expected=[0, 0, 0], slots=3))
        program = proposal([
            Reserve(0), Read(0, 0),
            Reserve(1), Read(1, 0),
            Reserve(2), Read(2, 0),
            WaitReads(),
            Publish(0), Publish(1), Publish(2),
            Acquire(0), Acquire(1), Acquire(2),
            Write(0, 0), Write(1, 1), Write(2, 2),
            WaitWrites(),
            Release(0), Release(1), Release(2),
        ])
        result = check(task, program)
        self.assertEqual(result.status, STATUS_ACCEPTED, result.to_dict())

    def test_multiple_writes_same_slot_distinct_destinations(self):
        task = parse_task(make_task(input_pages=1, expected=[0, 0], slots=1))
        program = proposal([
            Reserve(0), Read(0, 0), WaitReads(),
            Publish(0), Acquire(0),
            Write(0, 0), Write(0, 1),
            WaitWrites(), Release(0),
        ])
        result = check(task, program)
        self.assertEqual(result.status, STATUS_ACCEPTED, result.to_dict())

    def test_wait_with_empty_queue_advances(self):
        task = parse_task(make_task(expected=[0], slots=1))
        program = proposal([
            WaitReads(), WaitWrites(),
            Reserve(0), Read(0, 0), WaitReads(),
            Publish(0), Acquire(0), Write(0, 0),
            WaitWrites(), Release(0),
        ])
        result = check(task, program)
        self.assertEqual(result.status, STATUS_ACCEPTED, result.to_dict())


class UnsafeTests(unittest.TestCase):
    def test_premature_publish_assumes_dma_finished(self):
        # publish while the read is still pending would only be valid if DMA
        # were assumed to finish early; the checker must reject it.
        task = parse_task(make_task(expected=[0], slots=1))
        program = proposal([Reserve(0), Read(0, 0), Publish(0)])
        result = check(task, program)
        self.assertEqual(result.status, STATUS_UNSAFE)

    def test_missing_read_wait_before_acquire(self):
        task = parse_task(make_task(expected=[0], slots=1))
        program = proposal([Reserve(0), Read(0, 0), Acquire(0)])
        result = check(task, program)
        self.assertEqual(result.status, STATUS_UNSAFE)

    def test_premature_release_with_pending_write(self):
        task = parse_task(make_task(expected=[0], slots=1))
        program = proposal([
            Reserve(0), Read(0, 0), WaitReads(),
            Publish(0), Acquire(0), Write(0, 0), Release(0),
        ])
        result = check(task, program)
        self.assertEqual(result.status, STATUS_UNSAFE)
        self.assertIn("pending transfer", result.reason)

    def test_publish_uninitialized_value(self):
        task = parse_task(make_task(expected=[0], slots=1))
        program = proposal([Reserve(0), Publish(0)])
        result = check(task, program)
        self.assertEqual(result.status, STATUS_UNSAFE)

    def test_read_slot_out_of_range(self):
        task = parse_task(make_task(expected=[0], slots=1))
        program = proposal([Read(1, 0)])
        result = check(task, program)
        self.assertEqual(result.status, STATUS_UNSAFE)
        self.assertIn("slot", result.reason)

    def test_read_src_out_of_range(self):
        task = parse_task(make_task(expected=[0], slots=1))
        program = proposal([Reserve(0), Read(0, 5)])
        result = check(task, program)
        self.assertEqual(result.status, STATUS_UNSAFE)
        self.assertIn("src", result.reason)

    def test_write_dst_out_of_range(self):
        task = parse_task(make_task(expected=[0], slots=1))
        program = proposal([
            Reserve(0), Read(0, 0), WaitReads(),
            Publish(0), Acquire(0), Write(0, 1),
        ])
        result = check(task, program)
        self.assertEqual(result.status, STATUS_UNSAFE)

    def test_pending_writes_to_same_destination(self):
        task = parse_task(make_task(expected=[0, 0], slots=2))
        program = proposal([
            Reserve(0), Read(0, 0),
            Reserve(1), Read(1, 0),
            WaitReads(),
            Publish(0), Publish(1),
            Acquire(0), Acquire(1),
            Write(0, 0), Write(1, 0),
        ])
        result = check(task, program)
        self.assertEqual(result.status, STATUS_UNSAFE)
        self.assertIn("destination", result.reason)

    def test_slot_reuse_before_writes_complete(self):
        # release requires the slot free path; re-reserving while the slot is
        # still acquired with a pending write must be rejected.
        task = parse_task(make_task(expected=[0], slots=1))
        program = proposal([
            Reserve(0), Read(0, 0), WaitReads(),
            Publish(0), Acquire(0), Write(0, 0),
            Reserve(0),
        ])
        result = check(task, program)
        self.assertEqual(result.status, STATUS_UNSAFE)

    def test_unsafe_found_even_when_an_ordering_would_succeed(self):
        # The canonical simulation completes DMA eagerly and would succeed;
        # the exhaustive checker must still reject because the invalid issue
        # is reachable.
        task = parse_task(make_task(expected=[0], slots=1))
        program = proposal([Reserve(0), Read(0, 0), Publish(0)])
        self.assertEqual(simulate(task, program).status, STATUS_UNSAFE)
        result = check(task, program)
        self.assertEqual(result.status, STATUS_UNSAFE)

    def test_counterexample_trace_is_concise_and_stored(self):
        task = parse_task(make_task(expected=[0], slots=1))
        program = proposal([Reserve(0), Read(0, 0), Publish(0)])
        result = check(task, program)
        self.assertEqual(result.status, STATUS_UNSAFE)
        self.assertGreater(len(result.trace), 0)
        self.assertEqual(result.trace[0]["action"], "issue")
        self.assertEqual(result.trace[-1]["action"], "invalid")
        self.assertLessEqual(len(result.trace), 8)


class OutcomeTests(unittest.TestCase):
    def test_incorrect_permutation_is_wrong_output(self):
        task = parse_task(make_task(expected=[1, 0]))
        # Copies input page 0 to both outputs; expected wants the swap.
        program = proposal([
            Reserve(0), Read(0, 0), WaitReads(),
            Publish(0), Acquire(0), Write(0, 0), Write(0, 1),
            WaitWrites(), Release(0),
        ])
        result = check(task, program)
        self.assertEqual(result.status, STATUS_WRONG_OUTPUT)

    def test_forgotten_release_is_incomplete(self):
        task = parse_task(make_task(expected=[0], slots=1))
        program = proposal([
            Reserve(0), Read(0, 0), WaitReads(),
            Publish(0), Acquire(0), Write(0, 0), WaitWrites(),
        ])
        result = check(task, program)
        self.assertEqual(result.status, STATUS_INCOMPLETE)
        self.assertIn("free", result.reason)

    def test_missing_output_write_is_wrong_output(self):
        task = parse_task(make_task(expected=[0, 1], slots=1))
        program = proposal([
            Reserve(0), Read(0, 0), WaitReads(),
            Publish(0), Acquire(0), Write(0, 0),
            WaitWrites(), Release(0),
        ])
        result = check(task, program)
        self.assertEqual(result.status, STATUS_WRONG_OUTPUT)

    def test_state_cap_is_honest_budget(self):
        task = parse_task(
            make_task(input_pages=6, expected=[0, 1, 2, 3, 4, 5], slots=6)
        )
        program = proposal([
            Reserve(0), Read(0, 0),
            Reserve(1), Read(1, 1),
            Reserve(2), Read(2, 2),
            Reserve(3), Read(3, 3),
            Reserve(4), Read(4, 4),
            Reserve(5), Read(5, 5),
            WaitReads(),
            Publish(0), Publish(1), Publish(2),
            Publish(3), Publish(4), Publish(5),
            Acquire(0), Acquire(1), Acquire(2),
            Acquire(3), Acquire(4), Acquire(5),
            Write(0, 0), Write(1, 1), Write(2, 2),
            Write(3, 3), Write(4, 4), Write(5, 5),
            WaitWrites(),
            Release(0), Release(1), Release(2),
            Release(3), Release(4), Release(5),
        ])
        result = check(task, program, max_states=50)
        self.assertEqual(result.status, STATUS_BUDGET)
        self.assertIn("unknown", result.reason)
        self.assertEqual(result.trace, ())

    def test_large_cap_still_accepts_state_heavy_program(self):
        task = parse_task(
            make_task(input_pages=6, expected=[0, 1, 2, 3, 4, 5], slots=6)
        )
        program = proposal([
            Reserve(0), Read(0, 0),
            Reserve(1), Read(1, 1),
            Reserve(2), Read(2, 2),
            Reserve(3), Read(3, 3),
            Reserve(4), Read(4, 4),
            Reserve(5), Read(5, 5),
            WaitReads(),
            Publish(0), Publish(1), Publish(2),
            Publish(3), Publish(4), Publish(5),
            Acquire(0), Acquire(1), Acquire(2),
            Acquire(3), Acquire(4), Acquire(5),
            Write(0, 0), Write(1, 1), Write(2, 2),
            Write(3, 3), Write(4, 4), Write(5, 5),
            WaitWrites(),
            Release(0), Release(1), Release(2),
            Release(3), Release(4), Release(5),
        ])
        result = check(task, program)
        self.assertEqual(result.status, STATUS_ACCEPTED, result.to_dict())

    def test_check_is_deterministic(self):
        task = parse_task(make_task())
        first = check(task, serial_baseline(task))
        second = check(task, serial_baseline(task))
        self.assertEqual(first.to_dict(), second.to_dict())


class UtilityTests(unittest.TestCase):
    def test_simulate_counts_and_bytes(self):
        task = parse_task(make_task())
        sim = simulate(task, serial_baseline(task))
        self.assertEqual(sim.status, STATUS_ACCEPTED)
        self.assertEqual(sim.counts["wait_reads"], 2)
        self.assertEqual(sim.counts["wait_writes"], 2)
        self.assertEqual(sim.counts["commands"], 16)
        self.assertEqual(sim.counts["input_bytes"], 2 * 1024)
        self.assertEqual(sim.counts["output_bytes"], 2 * 1024)
        self.assertGreaterEqual(sim.counts["peak_live_slots"], 1)

    def test_score_prefers_batched_over_serial(self):
        task = parse_task(make_task())
        serial = simulate(task, serial_baseline(task))
        batched = simulate(task, BATCHED_SWAP)
        self.assertLess(score_key(batched), score_key(serial))

    def test_replay_concrete_matches_for_accepted_program(self):
        task = parse_task(make_task())
        replay = replay_concrete(task, serial_baseline(task))
        self.assertEqual(replay["status"], "replayed")
        self.assertTrue(replay["all_outputs_match"])
        self.assertEqual(replay["outputs_match"], [True, True])

    def test_replay_concrete_rejects_wrong_payload_program(self):
        task = parse_task(make_task(expected=[0, 0]))
        program = proposal([
            Reserve(0), Read(0, 1), WaitReads(),
            Publish(0), Acquire(0), Write(0, 0), Write(0, 1),
            WaitWrites(), Release(0),
        ])
        replay = replay_concrete(task, program)
        self.assertEqual(replay["status"], "wrong-output")
        self.assertFalse(replay["all_outputs_match"])


if __name__ == "__main__":
    unittest.main()
