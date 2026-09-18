# Tenstorrent protocol experiments

This backend targets a deliberately small Blackhole Metalium protocol: one
Tensix core, one data-movement RISC-V issuer, asynchronous NoC transfers, and
one-page circular buffers. The computation is an exact byte-preserving page
mapping, including permutations and replication. There is no floating-point
computation, SFPU execution, multi-core communication, or multi-issuer proof.

## Contract and representation

The fixed task states the input page count, the source page required at each
output position, the number of available slots, and page size. The candidate
cannot change that task. It supplies an ordered list of operations:

`reserve`, `read`, `wait_reads`, `publish`, `acquire`, `write`, `wait_writes`,
`release`.

Each slot corresponds to an independently allocated circular buffer. Issuing a
transfer and completing it are separate events. Pending transfers can complete
in any order. A read barrier blocks until all pending reads finish; the write
barrier behaves analogously. An unsafe non-barrier operation is an error even
if another completion order could have hidden the problem.

The checker must track initialized data, ownership, pending readers/writers,
destination collisions, and final output origins. A slot cannot be released
while a transfer still uses it. A terminal accepted state has no pending work,
all slots free, and exactly the requested origin at every output page.

The finite event semantics excludes an infinite sequence of idle steps.
Physical completion additionally assumes eventual DMA completion and issuer
scheduling. A proof over this model must not be called unconditional hardware
liveness, nor a proof for arbitrary Metalium programs.

## Evidence layers

1. Strict parsing and the Python checker filter proposals and provide error
   traces. These are not the formal acceptance authority.
2. A Lean certificate checks the supported protocol model. The actual theorem
   statement and its assumptions determine its scope.
3. Generated Metalium C++ is compiled and run using the official `ttsim` on a
   local Linux ARM64 container. Full outputs are compared, not just checksums.
4. Correspondence of emitted C++, compiler, firmware, Metalium primitives and
   silicon to the protocol model is a separate obligation. Recording source
   and binary hashes binds evidence to artifacts but does not prove compiler
   correctness. No final-machine-code theorem is claimed for this backend.

The optimization objective is fewer explicit read/write barriers, then fewer
commands, then lower peak slot occupancy. Actual transfer bytes are reported
separately. Neither this score nor simulator elapsed time is a hardware
performance measurement.

## Upstream reference points

- [Metalium guide](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md)
- [Circular-buffer publication](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/circular_buffers/cb_push_back.html)
- [Blackhole ISA documentation](https://github.com/tenstorrent/tt-isa-documentation/tree/main/BlackholeA0)
- [Official ttsim](https://github.com/tenstorrent/ttsim)

The initial local environment pins Metalium
`ad232e1cd799adf53841774805bc73fc5605b634` and ttsim `v1.10.8` (Blackhole ARM64).
Installation and live experiment results are recorded separately from these
requirements. See `benchmarks/tenstorrent/protocol/2026-09-18/` for the frozen
scope and task definitions. The validated CLI and local execution path are described below.

## Current result and proof boundary

The [registered pilot report](../benchmarks/tenstorrent/protocol/2026-09-18/REPORT.md)
records four certified AI proposals and eight actual Metalium programs (four
winners, four serial controls) passing 48 complete-output simulator cases.
All winners match the pre-existing batched/reuse control; no optimization
novelty or hardware speedup is claimed.

`GraphCertificate.check_sound` proves that an untrusted proposed state set,
when checked for initial membership, local safety and successor closure,
covers every reachable state. `Protocol.accepted_issueSafe` specializes that
to the explicit NoC/CB model. Each certificate also checks a strictly decreasing
rank `2 * remaining_instructions + pending_transfers` on every modeled edge;
`accepted_accessible` proves event-path well-foundedness. These events exclude
indefinite physical stalls. `accepted_output_all_payloads` interprets the proven
origins as arbitrary immutable input payloads under exact-copy semantics.

These theorems use `propext` and `Quot.sound`, without `sorryAx` or a native
computation axiom. The specific task/graph acceptance is replayed with
`decide +kernel`. The origin interpretation and atomic transfer-completion
model do not establish a refinement proof for Metalium, partial physical DMA
writes, firmware, the compiler or ELF execution. That remains future work.

## Check, certify, and search

Run from the repository root:

```bash
lake build VeriTac.Tenstorrent.Protocol
python3 -m specializations.tenstorrent_protocol check \
  --task benchmarks/tenstorrent/protocol/2026-09-18/tasks/pair.json \
  --proposal benchmarks/tenstorrent/protocol/2026-09-18/runs/pair/winner.proposal.json
python3 -m specializations.tenstorrent_protocol certify \
  --task benchmarks/tenstorrent/protocol/2026-09-18/tasks/pair.json \
  --proposal benchmarks/tenstorrent/protocol/2026-09-18/runs/pair/winner.proposal.json \
  --out .lake/tt-certificate-new
```

`check` is a native filter; `certify` performs a fresh Lean kernel check and
axiom audit. Certificate and search output directories must be new. A failed or
budget-exhausted checker never authorizes acceptance.

```bash
python3 -m specializations.tenstorrent_protocol search \
  --task benchmarks/tenstorrent/protocol/2026-09-18/tasks/permutation.json \
  --contract benchmarks/tenstorrent/protocol/2026-09-18/SCOPE.md \
  --out .lake/tt-search-new --rounds 3 --model dgxspark-glm/glm-5.3-flash \
  --model-timeout 900 --proof-timeout 600
```

The model identifier must exist in the local OpenCode configuration. The search
reuses the existing generic tool-denied OpenCode transport, fresh temporary
working directories and fresh sessions. This is not an OS-hermetic sandbox.
The fixed contract, task, own proposals and factual native feedback are exposed;
control recipes are withheld. Both source and built-protocol fingerprints are
checked through the run. Final selected programs must receive a kernel
certificate, rather than being called proved from the Python verdict alone.

The initial frontend caps input/output pages at 16, CB slots at 8, page size at
16 KiB (a positive multiple of 32), and flat programs at 128 instructions. The
current executable adapter encodes instructions in Metalium runtime arguments,
whose pinned limit is 341 words: four header words plus three words per
instruction, hence **112 instructions maximum for this emitter**. Larger IR
programs must use a different emitter; this one rejects them instead of
truncating. The first pilot uses only 11–32 instructions.

## Use the installed local simulator

The Mac Studio installation is in Docker container `veritac-tt-sim`, with
persistent volume `veritac-tt-sim-data`. Inside it:

- `/opt/tt/tt-metal`: pinned source plus recorded compatibility patches;
- `/opt/tt/install`: native ARM64 C++ SDK;
- `/opt/tt/sim/libttsim_bh.so`: official Blackhole simulator;
- `/opt/tt/artifacts`: generated programs, builds and logs;
- `/workspace`: this repository, mounted read-only.

To run a new local execution (no model call):

```bash
docker exec -w /workspace -e PYTHONPATH=/workspace -e CXX=g++-14 \
  veritac-tt-sim python3 -m specializations.tenstorrent_protocol.runtime.runner \
  --task /workspace/benchmarks/tenstorrent/protocol/2026-09-18/tasks/pair.json \
  --proposal /workspace/benchmarks/tenstorrent/protocol/2026-09-18/runs/pair/winner.proposal.json \
  --out /opt/tt/artifacts/my-new-run --jobs 1 --timeout 240 \
  --tt-metal-home /opt/tt/tt-metal --mode package --cmake-prefix /opt/tt/install \
  --env TT_METAL_SIMULATOR=/opt/tt/sim/libttsim_bh.so \
  --env TT_METAL_SLOW_DISPATCH_MODE=1 \
  --env TT_METAL_DISABLE_SFPLOADMACRO=1
```

The output directory must be new. The runner sets `TT_METAL_RUNTIME_ROOT`, runs
from a writable artifact directory, compares every output byte in C++ and
independently in Python, and retains `output.bin`. `--seed` and `--pattern`
(`lcg`, `zero`, `ones`, `alternating`) select reproducible input data. `--emit-only`
creates sources without claiming execution. This runner can also execute
native-filtered programs without a certificate; use `certify` or the recorded
pilot replay driver when a formal certificate is required.

For a fresh installation, use a new container name:

```bash
bash specializations/tenstorrent_protocol/runtime/setup_local.sh veritac-tt-sim-repro
```

The setup script preserves existing containers, uses no privileged mode or
physical-device mapping, and installs packages only inside its own container.
The commands composing the script were exercised in the recorded installation;
a second full clean installation of the consolidated script has not been run.
Apt package mirrors are not frozen; installed versions are recorded. The source
commit, submodules, base image digest and official simulator SHA-256 are pinned.
The tested container used one compile job within a 3 GiB memory limit.

The local source patches address the host profiler's AVX2 include on ARM64
(using its existing SIMDe dependency) and OpenMPI's unwanted legacy C++ bindings.
No protocol, DMA API or arithmetic implementation was patched. The streaming
profiler itself was not validated. Exact patches, installed hashes, packages,
actual device ELFs and compile logs are in the experiment's `environment/`.

## Audit the saved evidence

```bash
python3 benchmarks/tenstorrent/protocol/2026-09-18/audit_results.py
python3 -m unittest tests.test_tenstorrent_protocol tests.test_tenstorrent_certificate \
  tests.test_tenstorrent_adapters tests.test_tenstorrent_search tests.test_tenstorrent_runtime
```

The artifact audit binds the raw model response to the selected program, the
fixed task and regenerated Lean statement, the emitted C++ instruction stream,
all saved simulator output bytes and the associated JIT device ELF hashes.
Binding hashes do not replace the original kernel check or prove that C++ and
ELF refine the protocol semantics.
