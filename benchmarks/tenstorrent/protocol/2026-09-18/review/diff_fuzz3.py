"""Fuzz round 3: corrected ref mapping; accepted-region stress via interleaved
correct lifecycles + mutations; adapter consistency checks on accepted results.
"""
import random
import sys

sys.path.insert(0, ".")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from specializations.tenstorrent_protocol import protocol as p
from specializations.tenstorrent_protocol import adapter
from diff_fuzz import ref_check

REF = {"good": p.STATUS_ACCEPTED, "unsafe": p.STATUS_UNSAFE,
       "wrong-output": p.STATUS_WRONG_OUTPUT, "incomplete": p.STATUS_INCOMPLETE,
       "deadlock": "deadlock"}


def correct_pipeline(rng, n_in, expected, slots):
    """Interleaved by-construction-correct program: each output page d gets
    reserve/read/wait_reads/publish/acquire/write/wait_writes/release on its
    own slot, with lifecycle steps of different pages interleaved."""
    tracks = []
    for d, src in enumerate(expected):
        s = d % slots
        tracks.append([
            {"op": "reserve", "slot": s},
            {"op": "read", "slot": s, "src": src},
            {"op": "wait_reads"},
            {"op": "publish", "slot": s},
            {"op": "acquire", "slot": s},
            {"op": "write", "slot": s, "dst": d},
            {"op": "wait_writes"},
            {"op": "release", "slot": s},
        ])
    # interleave: keep per-track index, but respect that wait_reads of a track
    # must come after its read etc. — order within track preserved by popping.
    prog = []
    idx = [0] * len(tracks)
    done = 0
    while done < len(tracks):
        t = rng.randrange(len(tracks))
        if idx[t] < len(tracks[t]):
            prog.append(tracks[t][idx[t]])
            idx[t] += 1
            if idx[t] == len(tracks[t]):
                done += 1
    return prog


def mutate(rng, prog, n_in, n_out, slots):
    prog = [dict(o) for o in prog]
    for _ in range(rng.randrange(1, 3)):
        i = rng.randrange(len(prog))
        r = rng.random()
        if r < 0.3:  # drop a wait or lifecycle op
            del prog[i]
        elif r < 0.5:  # swap two ops
            j = rng.randrange(len(prog))
            prog[i], prog[j] = prog[j], prog[i]
        elif r < 0.7:  # reindex slot/dst/src
            o = prog[i]
            if o["op"] == "read":
                o["src"] = rng.randrange(n_in)
            elif o["op"] == "write":
                o["dst"] = rng.randrange(n_out)
            elif "slot" in o:
                o["slot"] = rng.randrange(slots)
        elif r < 0.85:  # insert a wait
            prog.insert(i, {"op": rng.choice(["wait_reads", "wait_writes"])})
        else:  # insert a stray lifecycle op
            prog.insert(i, {"op": rng.choice(["reserve", "publish", "acquire", "release"]),
                            "slot": rng.randrange(slots)})
    return prog


def main():
    rng = random.Random(99)
    mism = 0
    trials = 0
    dist = {}
    accepted_seen = 0
    for trial in range(4000):
        n_in = rng.choice([1, 2, 3])
        n_out = rng.choice([1, 2, 3])
        slots = rng.choice([1, 2, 3])
        expected = [rng.randrange(n_in) for _ in range(n_out)]
        task = {"id": "f3", "input_pages": n_in, "expected": expected,
                "slots": slots, "page_bytes": 32}
        program = correct_pipeline(rng, n_in, expected, slots)
        if rng.random() < 0.6:
            program = mutate(rng, program, n_in, n_out, slots)
        if not program:
            continue
        ref = REF[ref_check(task, program)]
        got = p.check(task, {"program": program, "rationale": ""}, max_states=100000).status
        trials += 1
        dist[got] = dist.get(got, 0) + 1
        if got != ref:
            mism += 1
            if mism <= 6:
                print("MISMATCH ref=%s native=%s" % (ref, got))
                print("  task:", task)
                print("  program:", program)
        if got == p.STATUS_ACCEPTED:
            accepted_seen += 1
            prop = p.parse_proposal({"program": program, "rationale": ""})
            sim = p.simulate(task, prop)
            if sim.status != p.STATUS_ACCEPTED:
                mism += 1
                print("SIM/CHECK DISAGREE check=accepted sim=%s" % sim.status, task, program)
            rep = p.replay_concrete(task, prop)
            if not rep["all_outputs_match"]:
                mism += 1
                print("REPLAY MISMATCH on accepted program", task, program)
            ev = adapter.evaluate(dict(task), program)
            if not ev["accepted"] or ev["objective"][0] != sim.counts["waits"] \
                    or ev["objective"][1] != len(program) \
                    or ev["counts"]["input_bytes"] != sim.counts["input_bytes"]:
                mism += 1
                print("ADAPTER INCONSISTENT", ev.get("objective"), sim.counts, task, program)
    print("trials:", trials, "accepted:", accepted_seen, "mismatches:", mism)
    print("native dist:", dist)


if __name__ == "__main__":
    main()
