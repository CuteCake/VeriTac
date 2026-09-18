"""Differential fuzz: independent reference model-checker vs protocol.check.

Independent implementation choices on purpose: recursive DFS, frozenset pending
queues keyed by sequence numbers (issue order), phase stored as int, no sorted
canonicalization. Any status disagreement on small random programs is a bug.
"""
import itertools
import random
import sys

sys.path.insert(0, ".")
from specializations.tenstorrent_protocol import (
    protocol as p, STATUS_ACCEPTED, STATUS_UNSAFE, STATUS_WRONG_OUTPUT,
    STATUS_INCOMPLETE,
)

FREE, RESERVED, PUBLISHED, ACQUIRED = 0, 1, 2, 3


def ref_check(task, program):
    n_in = task["input_pages"]
    expected = tuple(task["expected"])
    n_out = len(expected)
    slots = task["slots"]

    def step(state):
        pc, ph, val, pr, pw, out = state
        # pr: set of (seq, slot, src); pw: set of (seq, slot, dst, origin)
        results = []  # (kind, payload) ; kind in {"unsafe","terminal","child"}
        if pc < len(program):
            op = program[pc]
            o = op["op"]
            if o == "wait_reads":
                if not pr:
                    results.append(("child", (pc + 1, ph, val, pr, pw, out)))
            elif o == "wait_writes":
                if not pw:
                    results.append(("child", (pc + 1, ph, val, pr, pw, out)))
            else:
                s = op["slot"]
                ok, msg = True, None
                if o == "reserve":
                    ok = ph[s] == FREE
                    nxt = (pc + 1, ph[:s] + (FREE if False else RESERVED,) + ph[s + 1:] if ok else ph,
                           val[:s] + (None,) + val[s + 1:] if ok else val, pr, pw, out)
                elif o == "read":
                    ok = ph[s] == RESERVED and not any(x[1] == s for x in pr) \
                        and not any(x[1] == s for x in pw)
                    ok = ok and 0 <= op["src"] < n_in
                    nxt = (pc + 1, ph, val[:s] + (None,) + val[s + 1:],
                           pr | {(max((x[0] for x in pr), default=-1) + 1, s, op["src"])},
                           pw, out) if ok else None
                elif o == "publish":
                    ok = ph[s] == RESERVED and val[s] is not None \
                        and not any(x[1] == s for x in pr)
                    nxt = (pc + 1, ph[:s] + (PUBLISHED,) + ph[s + 1:], val, pr, pw, out) if ok else None
                elif o == "acquire":
                    ok = ph[s] == PUBLISHED
                    nxt = (pc + 1, ph[:s] + (ACQUIRED,) + ph[s + 1:], val, pr, pw, out) if ok else None
                elif o == "write":
                    ok = ph[s] == ACQUIRED and val[s] is not None \
                        and not any(x[1] == s for x in pr) \
                        and not any(x[2] == op["dst"] for x in pw)
                    ok = ok and 0 <= op["dst"] < n_out
                    nxt = (pc + 1, ph, val, pr,
                           pw | {(max((x[0] for x in pw), default=-1) + 1, s, op["dst"], val[s])},
                           out) if ok else None
                elif o == "release":
                    ok = ph[s] == ACQUIRED and not any(x[1] == s for x in pr) \
                        and not any(x[1] == s for x in pw)
                    nxt = (pc + 1, ph[:s] + (FREE,) + ph[s + 1:], val[:s] + (None,) + val[s + 1:],
                           pr, pw, out) if ok else None
                if not ok:
                    results.append(("unsafe", None))
                else:
                    results.append(("child", nxt))
        for x in pr:
            seq, s, src = x
            npr = pr - {x}
            results.append(("child", (pc, ph, val[:s] + (src,) + val[s + 1:], npr, pw, out)))
        for x in pw:
            seq, s, d, org = x
            npw = pw - {x}
            results.append(("child", (pc, ph, val, pr, npw, out[:d] + (org,) + out[d + 1:])))
        if not results:
            if pc == len(program) and not pr and not pw:
                if tuple(out) == expected and all(f == FREE for f in ph):
                    results.append(("good", None))
                elif tuple(out) != expected:
                    results.append(("wrong", None))
                else:
                    results.append(("incomplete", None))
            else:
                results.append(("stuck", None))
        return results

    sys.setrecursionlimit(10000)

    def dfs(state, seen):
        total = 0
        for kind, nxt in step(state):
            if kind == "unsafe":
                return "unsafe", total
            if kind in ("good", "wrong", "incomplete"):
                if kind == "wrong":
                    return "wrong-output", total
                if kind == "incomplete":
                    return "incomplete", total
                total += 1
                continue
            if kind == "stuck":
                return "stuck", total
            if nxt not in seen:
                seen.add(nxt)
                st, n = dfs(nxt, seen)
                total += n
                if st != "good":
                    return st, total
        return "good", total

    start = (0, (FREE,) * slots, (None,) * slots, frozenset(), frozenset(), (None,) * n_out)
    status, _ = dfs(start, {start})
    if status == "stuck":
        return "deadlock"
    return status


def random_program(rng, n_in, n_out, slots, length):
    prog = []
    for _ in range(length):
        o = rng.choice(["reserve", "read", "wait_reads", "publish", "acquire",
                        "write", "wait_writes", "release"])
        if o in ("wait_reads", "wait_writes"):
            prog.append({"op": o})
        elif o == "read":
            prog.append({"op": "read", "slot": rng.randrange(slots), "src": rng.randrange(n_in)})
        elif o == "write":
            prog.append({"op": "write", "slot": rng.randrange(slots), "dst": rng.randrange(n_out)})
        else:
            prog.append({"op": o, "slot": rng.randrange(slots)})
    return prog


def main():
    rng = random.Random(20260918)
    mismatches = 0
    trials = 0
    status_counts = {}
    for trial in range(3000):
        n_in = rng.choice([1, 2])
        n_out = rng.choice([1, 2])
        slots = rng.choice([1, 2])
        task = {"id": "fuzz", "input_pages": n_in,
                "expected": [rng.randrange(n_in) for _ in range(n_out)],
                "slots": slots, "page_bytes": 32}
        program = random_program(rng, n_in, n_out, slots, rng.randrange(1, 9))
        try:
            ref = ref_check(task, program)
        except RecursionError:
            continue
        try:
            got = p.check(task, {"program": program, "rationale": ""}, max_states=200000).status
        except p.ProtocolError:
            got = "schema"
        trials += 1
        status_counts[got] = status_counts.get(got, 0) + 1
        # reference never says accepted when native says rejected-with-bad-status
        ref_map = {"accepted": STATUS_ACCEPTED, "unsafe": STATUS_UNSAFE,
                   "wrong-output": STATUS_WRONG_OUTPUT,
                   "incomplete": STATUS_INCOMPLETE, "deadlock": "deadlock"}
        ref_status = ref_map.get(ref, ref)
        if got != ref_status:
            mismatches += 1
            if mismatches <= 5:
                print("MISMATCH ref=%s native=%s" % (ref_status, got))
                print("  task:", task)
                print("  program:", program)
    print("trials:", trials, "mismatches:", mismatches)
    print("native status distribution:", status_counts)


if __name__ == "__main__":
    main()
