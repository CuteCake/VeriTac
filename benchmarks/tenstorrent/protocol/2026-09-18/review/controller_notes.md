# Controller review follow-up

The independent worker report is preserved as a review, not a proof.
The former missing certificate module is now implemented and kernel-tested.
The canonical simulation docstring was corrected: completions are selected in
sorted-tuple order, not FIFO. Native budget, empty-task and scoring validation
were tightened before the registered experiment.

One claim in the worker report needs qualification: its concrete payload
formula repeats every 256 input pages. It is injective only within that range,
including the adapter limit of 16 pages. The all-input guarantee instead comes
from the Lean origin theorem under the exact-copy contract, not this test
pattern. The actual ttsim runner uses separate LCG/edge patterns and compares
all output bytes independently in both C++ and Python.

The one-issuer event model cannot exhibit a blocked-barrier deadlock while
pending completions are enabled. The Lean rank certificate proves event-path
well-foundedness; physical progress still assumes eventual DMA completion and
scheduling. None of this proves multi-issuer/multi-core deadlock freedom.

Replaying the worker's random sensitivity script produced zero detections for
both sampled mutations. The publish guard is redundant as discussed; the
random workload generator also failed to activate the duplicate-destination
write mutation. Thus the report's claim that that script demonstrated
sensitivity was not reproducible from the saved script alone. A directed
same-destination, simultaneously outstanding write case was added. The original
checker agrees with the independent reference (`unsafe`); the deliberately
weakened mutant differs. Both the negative random result and the directed
result are retained in `sensitivity.log`. No production checker was modified
by this mutation test (the monkeypatch is restored in `finally`).
