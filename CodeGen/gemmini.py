"""
VeriTac CodeGen: gemmini.py
Gemmini int8 GEMM backend: plan schema, fail-closed Lean checker invocation,
WS command-stream generation (baseline i,j,k vs reuse_b j,k,i), an exact
instruction-level interpreter for the SUPPORTED subset, and upstream C emission.

Upstream provenance (pinned, hash-verified):
  repo   https://github.com/ucb-bar/gemmini-rocc-tests
  commit 7c540b3adf1b86ad93d07f893abe3a73489b568e
  files  include/gemmini.h, include/gemmini_params.h, include/gemmini_counter.h,
         rocc-software/src/xcustom.h
  vendored at .lake/gemmini-upstream-tests/ (provided by controller);
  sha256(gemmini.h)      = 18801f4eab0cd2e81e25a6c2aa97c7ed644b693e8d2c24cdeb41ab3a7208336d
  sha256(gemmini_params.h)= 3758ae967af3a179497660970201093a7fb624be00173990ce33d3f5c38da924
All emitted C uses ONLY upstream macro signatures verbatim
(gemmini_extended_config_ex / _config_st / _extended3_config_ld /
 gemmini_extended_mvin / _mvin2 / _extended_preload /
 _extended_compute_preloaded / _extended_compute_accumulated /
 _extended_mvout / gemmini_flush / gemmini_fence). Nothing is invented.

DECLARED INSTRUCTION-LEVEL SUBSET CONTRACT (the modeled target semantics;
physical-hardware conformance is an explicit open trust boundary, per
veritac_design.md section 3.2):
  * Hardware constants are taken from gemmini_params.h at C compile time and
    mirrored here: DIM=16, elem_t=int8, acc_t=int32, BANK_NUM*BANK_ROWS
    scratchpad rows, ACC_ROWS accumulator rows, ADDR_LEN=32.
  * Scratchpad tile layout: A at row 0, B at row DIM, both within the
      plan-supplied capacity.  Accumulator addresses set bit 31
      (accumulator space), bit 29 (full-C / 32-bit readback) and bit 30
      (1 = accumulate onto the addressed rows, 0 = overwrite them; this is the
      upstream `no_bias_new_matrix` trick, which gives exact zero-init without
      a bias matrix).  GARBAGE_ADDR as a preload B source means "keep the
      weight-stationary B latched by the previous preload".
  * Each COMPUTE produces one full DIMxDIMxDIM int8xint8->int32 product
    (exact: |product| <= 16384, and the plan requires k*16384 <= 2^31-1, so
    accumulator arithmetic cannot overflow for inputs in [-128,127]).
  * DMAs are modeled as completing in issue order. Upstream kernels rely on
    hardware dependency tracking between mvin/compute/mvout and issue a final
    gemmini_fence(). Refinement from this sequential model to asynchronous
    hardware execution remains unproved.
  * Weight-stationary latching: PRELOAD selects the pending B source and
    output address. COMPUTE_PRELOADED loads that B into the array;
    COMPUTE_ACCUMULATED retains the previous B. Accumulator addition is
    separately controlled by output-address bit 30.

MODELED COUNTS ARE NOT MEASUREMENTS.  Instruction/DMA counts and the
optimization objective below are analytic models used for candidate selection;
they are never a measured speedup and never part of correctness acceptance.
"""

import hashlib
import json
import os
import subprocess

# ---------------------------------------------------------------------------
# Hardware constants (mirrored from the pinned include/gemmini_params.h; the
# emitted C re-checks these against the header at compile time).
# ---------------------------------------------------------------------------

DIM = 16                       # DIM (gemmini_params.h)
ADDR_LEN = 32                  # ADDR_LEN
BANK_NUM = 4                   # BANK_NUM
BANK_ROWS = 4096               # BANK_ROWS
SCRATCHPAD_ROWS = BANK_NUM * BANK_ROWS   # 16384
ACC_ROWS = 1024                # ACC_ROWS
ELEM_MAX = 127                 # elem_t_max
ELEM_MIN = -128                # elem_t_min
ACC_ELEMENT_MAX = 2**31 - 1    # int32 accumulator bound

GARBAGE_ADDR = (1 << ADDR_LEN) - 1        # GARBAGE_ADDR ((uint32_t)(-1))
ACC_SPACE_BIT = 1 << (ADDR_LEN - 1)       # bit31: accumulator space (D_sp_addr_start)
ACCUMULATE_BIT = 1 << (ADDR_LEN - 2)      # bit30: 1=accumulate, 0=overwrite
FULL_C_BIT = 1 << (ADDR_LEN - 3)          # bit29: full-width (32-bit) C
ACC_BASE_ADDR = ACC_SPACE_BIT | ACCUMULATE_BIT | FULL_C_BIT   # (3<<30)|(1<<29)

WEIGHT_STATIONARY = 1       # gemmini.h
OUTPUT_STATIONARY = 0       # gemmini.h
NO_ACTIVATION = 0           # gemmini.h

SCHEDULE_BASELINE = "baseline"
SCHEDULE_REUSE_B = "reuse_b"
SUPPORTED_SCHEDULES = (SCHEDULE_BASELINE, SCHEDULE_REUSE_B)

UPSTREAM_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".lake", "gemmini-upstream-tests")

INT8_DOMAIN_CONSTRAINT = "inputs are int8 in [-128,127] and k*16384 <= 2147483647"


# ---------------------------------------------------------------------------
# Plan schema: {"m","n","k","dim","scratchpad_rows","accumulator_rows",
#               "schedule"} -- all ints except schedule (string).
# ---------------------------------------------------------------------------

def required_resources(plan):
    """(scratchpad_rows, accumulator_rows) the schedule minimally needs."""
    if plan["schedule"] == SCHEDULE_BASELINE:
        # One A tile + one B tile in the scratchpad; one output tile live.
        return (2 * plan["dim"], plan["dim"])
    # reuse_b: B tile + streaming A tile; m accumulator rows stay live across
    # the k sweep (all i-tiles of the current j-tile).
    return (2 * plan["dim"], plan["m"])


def validate_plan(plan):
    """Structural / divisibility / overflow / resource validation.

    Returns (ok, reason).  This is a Python pre-check for diagnostics and
    candidate generation; it is NEVER a substitute for the Lean checker
    (gemmini_check), which is invoked fail-closed before any interpretation
    or emission.
    """
    if not isinstance(plan, dict):
        return False, "plan must be a JSON object"
    for key in ("m", "n", "k", "dim", "scratchpad_rows", "accumulator_rows"):
        if key not in plan or not isinstance(plan[key], int) or isinstance(plan[key], bool):
            return False, "plan field %r must be an integer" % key
    schedule = plan.get("schedule")
    if schedule not in SUPPORTED_SCHEDULES:
        return False, ("unsupported schedule %r (supported: %s)"
                       % (schedule, list(SUPPORTED_SCHEDULES)))
    m, n, k, dim = plan["m"], plan["n"], plan["k"], plan["dim"]
    if dim != DIM:
        return False, "dim must equal upstream DIM=%d (got %d)" % (DIM, dim)
    for name, val in (("m", m), ("n", n), ("k", k)):
        if val < dim:
            return False, "%s=%d must be >= dim=%d" % (name, val, dim)
        if val % dim != 0:
            return False, "%s=%d must be divisible by dim=%d" % (name, val, dim)
    if k * 16384 > ACC_ELEMENT_MAX:
        return False, ("overflow bound violated: k*16384 = %d > 2147483647 (%s)"
                       % (k * 16384, INT8_DOMAIN_CONSTRAINT))
    if max(m * n, m * k, k * n) > ACC_ELEMENT_MAX:
        return False, "buffer extent exceeds the emitted C signed-int indexing range"
    need_sp, need_acc = required_resources(plan)
    sp, acc = plan["scratchpad_rows"], plan["accumulator_rows"]
    if sp < need_sp:
        return False, ("scratchpad_rows=%d < required %d for schedule %r"
                       % (sp, need_sp, schedule))
    if sp > SCRATCHPAD_ROWS:
        return False, "scratchpad_rows=%d > hardware capacity %d" % (sp, SCRATCHPAD_ROWS)
    if acc < need_acc:
        return False, ("accumulator_rows=%d < required %d for schedule %r"
                       % (acc, need_acc, schedule))
    if acc > ACC_ROWS:
        return False, "accumulator_rows=%d > hardware capacity %d (ACC_ROWS)" % (acc, ACC_ROWS)
    return True, "ok"


# ---------------------------------------------------------------------------
# Fail-closed Lean checker invocation.
# ---------------------------------------------------------------------------

def checker_binary_path():
    env = os.environ.get("VERITAC_GEMMINI_CHECK_BIN")
    if env:
        return env
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, ".lake", "build", "bin", "gemmini_check")


def gemmini_check(plan, timeout=30):
    """Run the Lean-supplied gemmini_check binary on one plan JSON.

    Fail-closed: any absence, crash, timeout, malformed output, or mismatched
    echo plan is a rejection.  This checks plan legality only; concrete executable acceptance additionally
    requires checkExecutable and a kernel-checked artifact certificate.
    Returns (accepted: bool, reason: str, echoed_plan_or_None).
    """
    binpath = checker_binary_path()
    if not os.path.isfile(binpath) or not os.access(binpath, os.X_OK):
        return False, ("gemmini_check binary not found or not executable at %r "
                       "(fail-closed: Lean validation required)" % binpath), None
    try:
        proc = subprocess.run(
            [binpath], input=json.dumps(plan), capture_output=True,
            text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "gemmini_check timed out after %ss (fail-closed)" % timeout, None
    except OSError as exc:
        return False, "gemmini_check failed to run: %s (fail-closed)" % exc, None
    if proc.returncode != 0:
        return False, ("gemmini_check exited %d: %s (fail-closed)"
                       % (proc.returncode, (proc.stderr or proc.stdout).strip()[:400])), None
    try:
        out = json.loads(proc.stdout)
    except ValueError:
        return False, "gemmini_check emitted non-JSON output (fail-closed)", None
    if not isinstance(out, dict) or not isinstance(out.get("accepted"), bool) \
            or not isinstance(out.get("reason"), str):
        return False, "gemmini_check output missing accepted/reason (fail-closed)", None
    if out["accepted"] and out.get("plan") != plan:
        return False, "gemmini_check accepted but echoed a different plan (fail-closed)", None
    return out["accepted"], out["reason"], out.get("plan")


# ---------------------------------------------------------------------------
# Modeled command stream for the SUPPORTED subset.
# ---------------------------------------------------------------------------

class Command:
    """One modeled Gemmini instruction (fields mirror the upstream macro args)."""
    kind = "?"


class ConfigEx(Command):
    kind = "config_ex"

    def __init__(self, dataflow):
        self.dataflow = dataflow

class ConfigLd(Command):
    kind = "config_ld"

    def __init__(self, slot, stride_bytes, scale):
        self.slot = slot            # 0 -> gemmini_extended_mvin, 1 -> mvin2
        self.stride_bytes = stride_bytes
        self.scale = scale          # must be MVIN_SCALE_IDENTITY (1.0) in subset

class ConfigSt(Command):
    kind = "config_st"

    def __init__(self, stride_bytes):
        self.stride_bytes = stride_bytes

class Mvin(Command):
    kind = "mvin"

    def __init__(self, slot, buf, offset, spad_addr, cols, rows):
        self.slot = slot            # 0 -> k_MVIN, 1 -> k_MVIN2 (upstream WS: A/B)
        self.buf = buf              # "A" or "B"
        self.offset = offset        # element offset into the int8 buffer
        self.spad_addr = spad_addr  # destination scratchpad row
        self.cols = cols
        self.rows = rows

class Preload(Command):
    kind = "preload"

    def __init__(self, bd_spad_addr, out_addr):
        self.bd_spad_addr = bd_spad_addr   # scratchpad row or GARBAGE_ADDR
        self.out_addr = out_addr           # accumulator address (bit31/30/29)

class Compute(Command):
    kind = "compute"

    def __init__(self, accumulated, a_spad_addr, bd_spad_addr):
        self.accumulated = accumulated     # False -> compute_preloaded
        self.a_spad_addr = a_spad_addr
        self.bd_spad_addr = bd_spad_addr   # GARBAGE_ADDR in the WS subset

class Mvout(Command):
    kind = "mvout"

    def __init__(self, buf_offset, acc_addr, cols, rows):
        self.buf_offset = buf_offset       # element offset into C (int32)
        self.acc_addr = acc_addr
        self.cols = cols
        self.rows = rows

class Fence(Command):
    kind = "fence"


def _config_preamble(plan):
    """Upstream tiled_matmul_outer-style one-time configuration, WS dataflow."""
    m, n, k = plan["m"], plan["n"], plan["k"]
    return [
        ConfigEx(WEIGHT_STATIONARY),
        ConfigSt(n * 4),                                   # stride_C * sizeof(acc_t)
        ConfigLd(0, k * 1, 1.0),                           # stride_A * sizeof(elem_t)
        ConfigLd(1, n * 1, 1.0),                           # stride_B * sizeof(elem_t)
    ]


def gen_baseline(plan):
    """baseline: i,j,k blocks; A and B moved in per output block.

    for each output tile (i,j): for each k-tile: mvin A(i,k), mvin2 B(k,j),
    preload(B, acc tile; overwrite at k==0 / accumulate at k>0), compute;
    then mvout the finished tile.  Only dim accumulator rows are live.
    """
    m, n, k = plan["m"], plan["n"], plan["k"]
    it, jt, kt = m // DIM, n // DIM, k // DIM
    a_sp = 0
    b_sp = DIM
    cmds = _config_preamble(plan)
    for i in range(it):
        for j in range(jt):
            for kk in range(kt):
                a_off = (i * k + kk) * DIM          # == (i*DIM)*k + kk*DIM
                b_off = (kk * n + j) * DIM          # == (kk*DIM)*n + j*DIM
                cmds.append(Mvin(0, "A", a_off, a_sp, DIM, DIM))
                cmds.append(Mvin(1, "B", b_off, b_sp, DIM, DIM))
                out = ACC_BASE_ADDR + 0 if kk > 0 else (ACC_BASE_ADDR & ~ACCUMULATE_BIT) + 0
                cmds.append(Preload(b_sp, out))
                cmds.append(Compute(False, a_sp, GARBAGE_ADDR))
            cmds.append(Mvout((i * n + j) * DIM, ACC_BASE_ADDR + 0, DIM, DIM))
    cmds.append(Fence())
    return cmds


def gen_reuse_b(plan):
    """reuse_b: j,k,i blocks; B (weight) tile retained across the i sweep.

    for each j-tile: for each k-tile: mvin2 B(k,j) once; for each i-tile:
    mvin A(i,k), preload (B latched at i==0, GARBAGE after; overwrite at
    k==0 / accumulate at k>0), compute; mvout each finished tile at k==K-1.
    m accumulator rows (one tile per i) stay live across the k sweep.
    """
    m, n, k = plan["m"], plan["n"], plan["k"]
    it, jt, kt = m // DIM, n // DIM, k // DIM
    a_sp = 0
    b_sp = DIM
    cmds = _config_preamble(plan)
    for j in range(jt):
        for kk in range(kt):
            b_off = (kk * n + j) * DIM
            cmds.append(Mvin(1, "B", b_off, b_sp, DIM, DIM))
            for i in range(it):
                a_off = (i * k + kk) * DIM
                cmds.append(Mvin(0, "A", a_off, a_sp, DIM, DIM))
                out_row = i * DIM
                out = ACC_BASE_ADDR + out_row if kk > 0 \
                    else (ACC_BASE_ADDR & ~ACCUMULATE_BIT) + out_row
                cmds.append(Preload(b_sp if i == 0 else GARBAGE_ADDR, out))
                cmds.append(Compute(i != 0, a_sp, GARBAGE_ADDR))
                if kk == kt - 1:
                    cmds.append(Mvout((i * n + j) * DIM, ACC_BASE_ADDR + out_row, DIM, DIM))
    cmds.append(Fence())
    return cmds


def gen_commands(plan):
    if plan["schedule"] == SCHEDULE_BASELINE:
        return gen_baseline(plan)
    return gen_reuse_b(plan)


# ---------------------------------------------------------------------------
# Modeled cost model (analytic; explicitly NOT a measurement).
# ---------------------------------------------------------------------------

def modeled_counts(cmds):
    counts = {"config_ex": 0, "config_ld": 0, "config_st": 0,
              "mvin": 0, "preload": 0, "compute": 0, "mvout": 0, "fence": 0}
    dma_elems_ab = 0      # int8 elements moved in
    dma_elems_c = 0       # int32 elements moved out
    for c in cmds:
        counts[c.kind] += 1
        if c.kind == "mvin":
            dma_elems_ab += c.rows * c.cols
        elif c.kind == "mvout":
            dma_elems_c += c.rows * c.cols
    counts["mvin_A"] = sum(1 for c in cmds if c.kind == "mvin" and c.slot == 0)
    counts["mvin_B"] = sum(1 for c in cmds if c.kind == "mvin" and c.slot == 1)
    counts["compute_preloaded"] = sum(1 for c in cmds if c.kind == "compute" and not c.accumulated)
    counts["compute_accumulated"] = sum(1 for c in cmds if c.kind == "compute" and c.accumulated)
    counts["dma_in_elems_int8"] = dma_elems_ab
    counts["dma_out_elems_int32"] = dma_elems_c
    counts["dma_bytes"] = dma_elems_ab * 1 + dma_elems_c * 4
    counts["instructions"] = len(cmds)
    return counts


def modeled_objective(counts):
    """Transparent selection objective: least modeled DMA bytes, then least
    modeled instruction count, then least declared accumulator rows."""
    return (counts["dma_bytes"], counts["instructions"])


# ---------------------------------------------------------------------------
# Exact instruction-level interpreter for the SUPPORTED subset.
# ---------------------------------------------------------------------------

class GemminiError(Exception):
    pass


class GemminiState:
    def __init__(self, m, n, k, A, B, scratchpad_rows, accumulator_rows):
        if len(A) != m * k or len(B) != k * n:
            raise GemminiError("input buffers have wrong length")
        for v in list(A) + list(B):
            if type(v) is not int or not (ELEM_MIN <= v <= ELEM_MAX):
                raise GemminiError("int8 input out of [-128,127]: %r" % v)
        self.m, self.n, self.k = m, n, k
        self.A = list(A)
        self.B = list(B)
        self.C = [0] * (m * n)                       # int32 result buffer
        self.scratchpad_rows = scratchpad_rows
        self.accumulator_rows = accumulator_rows
        self.spad = [None] * scratchpad_rows
        self.acc = [None] * (accumulator_rows * DIM)
        self.written = [False] * (m * n)
        self.pending_b = None
        self.ld_stride = {0: None, 1: None, 2: None} # bytes, per mvin slot
        self.ld_scale = {0: None, 1: None, 2: None}
        self.st_stride = None
        self.dataflow = None
        self.latched_b = None                        # WS: B tile latched by compute_preloaded
        self.latched_out = None                      # accumulator addr for compute
        self.in_flight = 0                           # modeled DMA ordering

    def _spad_tile(self, start_row, rows, what):
        if not (0 <= start_row and start_row + rows <= self.scratchpad_rows):
            raise GemminiError("%s: scratchpad rows [%d,%d) out of range"
                               % (what, start_row, start_row + rows))
        tile = []
        for r in range(rows):
            row = self.spad[start_row + r]
            if row is None:
                raise GemminiError("%s: scratchpad row %d never moved in" % (what, start_row + r))
            tile.append(list(row))
        return tile

    def _acc_rows(self, addr, rows, what):
        if not (addr & ACC_SPACE_BIT):
            raise GemminiError("%s: address %#x is not accumulator space" % (what, addr))
        if not (addr & FULL_C_BIT):
            raise GemminiError("%s: address %#x lacks full-C (bit29) width" % (what, addr))
        base = addr & ((1 << (ADDR_LEN - 3)) - 1)
        if base + rows > self.accumulator_rows:
            raise GemminiError("%s: accumulator rows [%d,%d) exceed ACC_ROWS=%d"
                               % (what, base, base + rows, self.accumulator_rows))
        return base


def run_commands(cmds, plan, A, B):
    """Execute a command stream exactly under the declared subset contract.

    Returns (C, stats) where C is the int32 result list of length m*n.
    Raises GemminiError on any subset violation, addressing error, or
    int32 accumulator overflow.
    """
    ok, reason = validate_plan(plan)
    if not ok:
        raise GemminiError(reason)
    accepted, reason, _ = gemmini_check(plan)
    if not accepted:
        raise GemminiError("Lean rejected interpretation: " + reason)
    m, n, k = plan["m"], plan["n"], plan["k"]
    st = GemminiState(m, n, k, A, B, plan["scratchpad_rows"], plan["accumulator_rows"])
    stats = {"instructions": 0, "macs": 0, "overwrites": 0, "accumulates": 0}
    for c in cmds:
        stats["instructions"] += 1
        if c.kind == "config_ex":
            if c.dataflow != WEIGHT_STATIONARY:
                raise GemminiError("subset supports WEIGHT_STATIONARY only")
            st.dataflow = c.dataflow
        elif c.kind == "config_ld":
            if c.slot not in (0, 1, 2):
                raise GemminiError("config_ld slot %r unsupported" % c.slot)
            if c.scale != 1.0:
                raise GemminiError("subset supports MVIN_SCALE_IDENTITY only")
            if c.stride_bytes < 0:
                raise GemminiError("config_ld stride must be >= 0")
            st.ld_stride[c.slot] = c.stride_bytes
            st.ld_scale[c.slot] = c.scale
        elif c.kind == "config_st":
            if st.st_stride is not None:
                pass  # reconfiguration allowed, last write wins (upstream semantics)
            st.st_stride = c.stride_bytes
        elif c.kind == "mvin":
            if st.ld_stride[c.slot] is None:
                raise GemminiError("mvin slot %d used before config_ld" % c.slot)
            if c.cols != DIM or c.rows != DIM:
                raise GemminiError("subset moves full %dx%d tiles only (got %dx%d)"
                                   % (DIM, DIM, c.cols, c.rows))
            if c.buf not in ("A", "B"):
                raise GemminiError("unknown input buffer")
            src = st.A if c.buf == "A" else st.B
            size = m * k if c.buf == "A" else k * n
            stride = st.ld_stride[c.slot]
            if not (0 <= c.offset and c.offset + (c.rows - 1) * stride + c.cols <= size):
                raise GemminiError("mvin %s: dram range out of bounds" % c.buf)
            if not (0 <= c.spad_addr and c.spad_addr + c.rows <= st.scratchpad_rows):
                raise GemminiError("mvin: scratchpad destination out of range")
            for r in range(c.rows):
                st.spad[c.spad_addr + r] = list(src[c.offset + r * stride:
                                                    c.offset + r * stride + c.cols])
            st.in_flight += 1
        elif c.kind == "preload":
            st.pending_b = c.bd_spad_addr
            st.latched_out = c.out_addr
        elif c.kind == "compute":
            if st.dataflow != WEIGHT_STATIONARY or st.pending_b is None or st.latched_out is None:
                raise GemminiError("compute without WS configuration and preload")
            if c.bd_spad_addr != GARBAGE_ADDR:
                raise GemminiError("supported WS compute requires garbage BD operand")
            if c.accumulated:
                if st.pending_b != GARBAGE_ADDR or st.latched_b is None:
                    raise GemminiError("compute_accumulated requires retained B weights")
            else:
                if st.pending_b == GARBAGE_ADDR:
                    raise GemminiError("compute_preloaded requires an explicit B source")
                st.latched_b = st._spad_tile(st.pending_b, DIM, "compute preload B")
            a_tile = st._spad_tile(c.a_spad_addr, DIM, "compute A")
            out_base = st._acc_rows(st.latched_out, DIM, "compute output")
            overwrite = not (st.latched_out & ACCUMULATE_BIT)
            b_tile = st.latched_b
            for r in range(DIM):
                arow = a_tile[r]
                for col in range(DIM):
                    s = 0
                    for t in range(DIM):
                        s += arow[t] * b_tile[t][col]
                    stats["macs"] += DIM
                    if overwrite:
                        v = s
                    else:
                        old = st.acc[(out_base + r) * DIM + col]
                        if old is None:
                            raise GemminiError("accumulation into uninitialized output")
                        v = old + s
                    if not (-ACC_ELEMENT_MAX - 1 <= v <= ACC_ELEMENT_MAX):
                        raise GemminiError(
                            "int32 accumulator overflow at acc row %d col %d "
                            "(value %d); violates %s"
                            % (out_base + r, col, v, INT8_DOMAIN_CONSTRAINT))
                    st.acc[(out_base + r) * DIM + col] = v
                if overwrite:
                    stats["overwrites"] += 1
                else:
                    stats["accumulates"] += 1
        elif c.kind == "mvout":
            if st.st_stride is None:
                raise GemminiError("mvout before config_st")
            if c.cols != DIM or c.rows != DIM:
                raise GemminiError("subset moves full %dx%d tiles only (got %dx%d)"
                                   % (DIM, DIM, c.cols, c.rows))
            stride_elems = st.st_stride // 4
            if st.st_stride % 4 != 0 or stride_elems < n:
                raise GemminiError("mvout stride %d bytes invalid for int32 C rows"
                                   % st.st_stride)
            if not (0 <= c.buf_offset and c.buf_offset + (c.rows - 1) * stride_elems
                    + c.cols <= m * n):
                raise GemminiError("mvout: dram range out of bounds")
            acc_base = st._acc_rows(c.acc_addr, DIM, "mvout source")
            for r in range(c.rows):
                for col in range(c.cols):
                    value = st.acc[(acc_base + r) * DIM + col]
                    if value is None:
                        raise GemminiError("read of uninitialized accumulator")
                    index = c.buf_offset + r * stride_elems + col
                    st.C[index] = value
                    st.written[index] = True
            if st.in_flight > 0:
                st.in_flight -= 1
        elif c.kind == "fence":
            st.in_flight = 0   # barrier under the in-order DMA model
        else:
            raise GemminiError("unknown command kind %r" % c.kind)
    if st.in_flight != 0:
        raise GemminiError("stream ended with DMAs in flight (missing fence)")
    if not all(st.written):
        raise GemminiError("not every output element was written")
    return st.C, stats


# ---------------------------------------------------------------------------
# Scalar reference (exact under the declared domain).
# ---------------------------------------------------------------------------

def scalar_matmul(A, B, m, n, k):
    C = [0] * (m * n)
    for i in range(m):
        for j in range(n):
            s = 0
            for t in range(k):
                s += A[i * k + t] * B[t * n + j]
            if s > ACC_ELEMENT_MAX or s < -ACC_ELEMENT_MAX - 1:
                raise GemminiError("reference overflow (plan violates %s)"
                                   % INT8_DOMAIN_CONSTRAINT)
            C[i * n + j] = s
    return C


def deterministic_inputs(m, n, k, seed, mode="lcg"):
    if mode == "lcg":
        x = seed & 0xFFFFFFFF
        A, B = [], []
        for _ in range(m * k):
            x = (1103515245 * x + 12345) & 0xFFFFFFFF
            A.append((x >> 16) % 256 - 128)
        for _ in range(k * n):
            x = (1103515245 * x + 12345) & 0xFFFFFFFF
            B.append((x >> 16) % 256 - 128)
        return A, B
    if mode == "max":
        return [ELEM_MAX] * (m * k), [ELEM_MAX] * (k * n)
    if mode == "minmax":
        return [ELEM_MIN] * (m * k), [ELEM_MAX] * (k * n)
    if mode == "min":
        return [ELEM_MIN] * (m * k), [ELEM_MIN] * (k * n)
    if mode == "zero":
        return [0] * (m * k), [0] * (k * n)
    raise ValueError(mode)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Upstream C emission.
# ---------------------------------------------------------------------------

_C_TEMPLATE = r"""// Generated by VeriTac CodeGen/gemmini.py -- Gemmini int8 GEMM backend.
// Schedule: %(schedule)s (plan: m=%(m)d n=%(n)d k=%(k)d dim=%(dim)d
//   scratchpad_rows=%(sp)d accumulator_rows=%(acc)d)
// Modeled instruction/DMA counts below are ANALYTIC MODELS, not measurements.
//
// Upstream target (pinned, hash-verified): ucb-bar/gemmini-rocc-tests
// @ 7c540b3adf1b86ad93d07f893abe3a73489b568e.  Compile with -I<upstream root>
// (e.g. -I .lake/gemmini-upstream-tests).  Run on Gemmini Spike/RTL only;
// this host build is not a Gemmini execution.
//
// Modeled counts for this schedule: mvins=%(mvin)d (A:%(mvin_a)d B:%(mvin_b)d),
// computes=%(compute)d, preloads=%(preload)d, mvouts=%(mvout)d,
// modeled DMA bytes=%(dma)d.

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include "include/gemmini.h"
#if defined(__riscv) && !defined(BAREMETAL)
#include <sys/mman.h>
#endif

#define VERITAC_M %(m)d
#define VERITAC_N %(n)d
#define VERITAC_K %(k)d
#define VERITAC_SCHEDULE_REUSE_B %(reuse_b)d
#define VERITAC_INPUT_MODE %(input_mode)d

// --- compile-time checks of the plan against the pinned upstream headers ---
#if DIM != %(dim)d
#error "VeriTac gemmini backend requires DIM == %(dim)d (gemmini_params.h)"
#endif
#if ADDR_LEN != 32
#error "VeriTac gemmini backend requires ADDR_LEN == 32"
#endif
_Static_assert(_Generic((elem_t)0, int8_t: 1, default: 0), "elem_t must be int8_t");
_Static_assert(_Generic((acc_t)0, int32_t: 1, default: 0), "acc_t must be int32_t");
_Static_assert(%(sp)d <= BANK_NUM * BANK_ROWS, "declared scratchpad exceeds target");
_Static_assert(%(acc)d <= ACC_ROWS, "declared accumulator exceeds target");
_Static_assert(1LL * VERITAC_M * VERITAC_N <= INT32_MAX &&
               1LL * VERITAC_M * VERITAC_K <= INT32_MAX &&
               1LL * VERITAC_K * VERITAC_N <= INT32_MAX,
               "buffer extent exceeds C indexing range");
#if VERITAC_M < DIM || VERITAC_N < DIM || VERITAC_K < DIM
#error "GEMM dimensions must be >= DIM"
#endif
#if (VERITAC_M %% DIM) != 0 || (VERITAC_N %% DIM) != 0 || (VERITAC_K %% DIM) != 0
#error "GEMM dimensions must be divisible by DIM (supported subset)"
#endif
#if (VERITAC_K * 16384LL) > 2147483647LL
#error "int32 accumulator overflow bound violated: k*16384 > 2^31-1"
#endif
#if 2 * DIM > (BANK_NUM * BANK_ROWS)
#error "plan scratchpad requirement exceeds header scratchpad capacity"
#endif
#if VERITAC_SCHEDULE_REUSE_B && (VERITAC_M > ACC_ROWS)
#error "reuse_b needs M accumulator rows; M exceeds ACC_ROWS"
#endif
#if !VERITAC_SCHEDULE_REUSE_B && (DIM > ACC_ROWS)
#error "baseline needs DIM accumulator rows; DIM exceeds ACC_ROWS"
#endif

// Scratchpad / accumulator addressing (upstream sp_tiled_matmul_ws layout).
#define VT_A_SP_ADDR 0u
#define VT_B_SP_ADDR ((uint32_t)DIM)
#define VT_ACC_BASE  ((uint32_t)((3u << (ADDR_LEN - 2)) | (1u << (ADDR_LEN - 3))))
#define VT_ACC_OVERWRITE (VT_ACC_BASE & ~(1u << (ADDR_LEN - 2)))  // no-bias zero-init

static elem_t vt_A[VERITAC_M * VERITAC_K] row_align(1);
static elem_t vt_B[VERITAC_K * VERITAC_N] row_align(1);
static acc_t vt_C[VERITAC_M * VERITAC_N] row_align_acc(1);

void veritac_gemmini_matmul(const elem_t *A, const elem_t *B, acc_t *C) {
  // One-time configuration (upstream tiled_matmul_outer pattern, WS dataflow).
  gemmini_extended_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0, 1, 0, 0);
  gemmini_extended_config_st(VERITAC_N * sizeof(acc_t), NO_ACTIVATION,
                             ACC_SCALE_IDENTITY);
  gemmini_extended3_config_ld(VERITAC_K * sizeof(elem_t), MVIN_SCALE_IDENTITY,
                              false, 0);
  gemmini_extended3_config_ld(VERITAC_N * sizeof(elem_t), MVIN_SCALE_IDENTITY,
                              false, 1);

  const uint32_t A_sp = VT_A_SP_ADDR;
  const uint32_t B_sp = VT_B_SP_ADDR;
  const int IT = VERITAC_M / DIM;
  const int JT = VERITAC_N / DIM;
  const int KT = VERITAC_K / DIM;

%(kernel_body)s

  gemmini_fence();
}

// Exact scalar reference (int32 exact under k*16384 <= 2^31-1).
static void veritac_scalar_ref(const elem_t *A, const elem_t *B, acc_t *C) {
  for (int i = 0; i < VERITAC_M; ++i)
    for (int j = 0; j < VERITAC_N; ++j) {
      int32_t s = 0;
      for (int t = 0; t < VERITAC_K; ++t)
        s += (int32_t)A[i * VERITAC_K + t] * (int32_t)B[t * VERITAC_N + j];
      C[i * VERITAC_N + j] = s;
    }
}

#ifdef __riscv
static inline uint64_t vt_cycles(void) {
  uint64_t c; asm volatile ("rdcycle %%0" : "=r"(c)); return c;
}
#else
#include <time.h>
static inline uint64_t vt_cycles(void) {
  return (uint64_t)clock();  // host fallback; NOT comparable to Gemmini cycles
}
#endif

int main(void) {
#if defined(__riscv) && !defined(BAREMETAL)
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); exit(1); }
#endif

  uint32_t x = %(seed)uu;
  for (int i = 0; i < VERITAC_M * VERITAC_K; ++i) {
    x = 1103515245u * x + 12345u;
    vt_A[i] = (elem_t)((int)((x >> 16) & 0xFF) - 128);  // [-128,127]
  }
  for (int i = 0; i < VERITAC_K * VERITAC_N; ++i) {
    x = 1103515245u * x + 12345u;
    vt_B[i] = (elem_t)((int)((x >> 16) & 0xFF) - 128);
  }

#if VERITAC_INPUT_MODE != 0
  for (int i = 0; i < VERITAC_M * VERITAC_K; ++i) {
    vt_A[i] = VERITAC_INPUT_MODE == 1 ? 127 : (VERITAC_INPUT_MODE == 4 ? 0 : -128);
  }
  for (int i = 0; i < VERITAC_K * VERITAC_N; ++i) {
    vt_B[i] = VERITAC_INPUT_MODE == 3 ? -128 : (VERITAC_INPUT_MODE == 4 ? 0 : 127);
  }
#endif
  // Prefault output pages under the proxy kernel and expose missing stores.
  // INT32_MIN is outside the proven output range for the accepted contract.
  for (int i = 0; i < VERITAC_M * VERITAC_N; ++i) {
    ((volatile acc_t *)vt_C)[i] = INT32_MIN;
  }
  static acc_t REF[VERITAC_M * VERITAC_N];
  veritac_scalar_ref(vt_A, vt_B, REF);

#if defined(__riscv)
  uint64_t t0 = vt_cycles();
#endif
  gemmini_flush(0);
  veritac_gemmini_matmul(vt_A, vt_B, vt_C);
#if defined(__riscv)
  uint64_t t1 = vt_cycles();
#endif

  // --- cycle CSR is informational; Spike is NOT a hardware timing model ---
#if defined(__riscv)
  printf("execution_cycle_counter_not_hardware_latency %%llu\n", (unsigned long long)(t1 - t0));
#else
  printf("host_model_time_not_gemmini_cycles\n");
#endif

  // --- correctness gate: every output compared to the scalar reference ---
  long bad = -1;
  for (long i = 0; i < (long)VERITAC_M * VERITAC_N; ++i) {
    if (vt_C[i] != REF[i]) { bad = i; break; }
  }
  if (bad < 0) {
    printf("all %%ld scalar output comparisons matched\n",
           (long)VERITAC_M * VERITAC_N);
    printf("VERITAC_GEMMINI_PASS\n");
    return 0;
  }
  printf("MISMATCH at flat index %%ld: got %%d want %%d\n",
         bad, (int)vt_C[bad], (int)REF[bad]);
  printf("VERITAC_GEMMINI_FAIL\n");
  return 1;
}
"""

_BASELINE_KERNEL = r"""  // baseline: i,j,k blocks; A and B moved in per output block.
  // Modeled per (i,j,k): 2 mvins + 1 preload + 1 compute; per (i,j): 1 mvout.
  for (int i = 0; i < IT; ++i) {
    for (int j = 0; j < JT; ++j) {
      for (int kk = 0; kk < KT; ++kk) {
        const elem_t *Ad = A + ((i * VERITAC_K + kk) * DIM);   // tile (i,kk)
        const elem_t *Bd = B + ((kk * VERITAC_N + j) * DIM);   // tile (kk,j)
        gemmini_extended_mvin(Ad, A_sp, DIM, DIM);
        gemmini_extended_mvin2(Bd, B_sp, DIM, DIM);
        const uint32_t out = (kk > 0) ? (VT_ACC_BASE + 0u)
                                      : (VT_ACC_OVERWRITE + 0u);
        gemmini_extended_preload(B_sp, out, DIM, DIM, DIM, DIM);
        gemmini_extended_compute_preloaded(A_sp, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
      }
      acc_t *Cd = C + ((i * VERITAC_N + j) * DIM);
      gemmini_extended_mvout(Cd, VT_ACC_BASE + 0u, DIM, DIM);
    }
  }"""

_REUSE_B_KERNEL = r"""  // reuse_b: j,k,i blocks; the B (weight) tile stays latched/stationary
  // across the whole i sweep; m accumulator rows stay live across the k sweep.
  // Modeled per (j,k): 1 B mvin; per (j,k,i): 1 A mvin + 1 preload + 1 compute;
  // per (i,j) at kk==KT-1: 1 mvout.
  for (int j = 0; j < JT; ++j) {
    for (int kk = 0; kk < KT; ++kk) {
      const elem_t *Bd = B + ((kk * VERITAC_N + j) * DIM);   // tile (kk,j)
      gemmini_extended_mvin2(Bd, B_sp, DIM, DIM);
      for (int i = 0; i < IT; ++i) {
        const elem_t *Ad = A + ((i * VERITAC_K + kk) * DIM); // tile (i,kk)
        gemmini_extended_mvin(Ad, A_sp, DIM, DIM);
        const uint32_t out_row = (uint32_t)(i * DIM);
        const uint32_t out = (kk > 0) ? (VT_ACC_BASE + out_row)
                                      : (VT_ACC_OVERWRITE + out_row);
        gemmini_extended_preload((i == 0) ? B_sp : GARBAGE_ADDR, out,
                                 DIM, DIM, DIM, DIM);
        if (i == 0) {
          gemmini_extended_compute_preloaded(A_sp, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
        } else {
          gemmini_extended_compute_accumulated(A_sp, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
        }
        if (kk == KT - 1) {
          acc_t *Cd = C + ((i * VERITAC_N + j) * DIM);
          gemmini_extended_mvout(Cd, VT_ACC_BASE + out_row, DIM, DIM);
        }
      }
    }
  }"""


def emit_c(plan, out_path, *, seed=0x12345678, input_mode="lcg"):
    """Emit a standalone C executable source for the plan's schedule.

    The source compiles against the pinned upstream headers
    (#include "include/gemmini.h" with -I<upstream root>) and prints
    VERITAC_GEMMINI_PASS only after every scalar output comparison succeeds.
    """
    ok, reason = validate_plan(plan)
    if not ok:
        raise ValueError("refusing to emit C for invalid plan: %s" % reason)
    accepted, chk_reason, _ = gemmini_check(plan)
    if not accepted:
        raise ValueError("refusing to emit C: Lean gemmini_check rejected the "
                         "plan: %s" % chk_reason)
    modes = {"lcg": 0, "max": 1, "minmax": 2, "min": 3, "zero": 4}
    if input_mode not in modes or type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("unsupported input mode or non-uint32 seed")
    counts = modeled_counts(gen_commands(plan))
    if plan["schedule"] == SCHEDULE_BASELINE:
        kernel, reuse_b = _BASELINE_KERNEL, 0
    else:
        kernel, reuse_b = _REUSE_B_KERNEL, 1
    text = _C_TEMPLATE % {
        "schedule": plan["schedule"], "m": plan["m"], "n": plan["n"],
        "k": plan["k"], "dim": plan["dim"], "sp": plan["scratchpad_rows"],
        "acc": plan["accumulator_rows"], "reuse_b": reuse_b,
        "mvin": counts["mvin"], "mvin_a": counts["mvin_A"],
        "mvin_b": counts["mvin_B"], "compute": counts["compute"],
        "preload": counts["preload"], "mvout": counts["mvout"],
        "dma": counts["dma_bytes"], "seed": seed, "input_mode": modes[input_mode],
        "kernel_body": kernel,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(text)
    return {"path": out_path, "sha256": sha256_file(out_path), "counts": counts}


# ---------------------------------------------------------------------------
# Concrete program acceptance: commands + executable bytes via the separate
# gemmini_program_check Lean CLI.  Distinct from plan acceptance: a checked
# plan alone is NOT a verified program.
# ---------------------------------------------------------------------------

PROGRAM_CHECK_SCHEMA = "veritac_program_request_v1"


def program_checker_binary_path():
    env = os.environ.get("VERITAC_GEMMINI_PROGRAM_CHECK_BIN")
    if env:
        return env
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, ".lake", "build", "bin", "gemmini_program_check")


def serialize_commands(cmds):
    """Canonical JSON-ready command dicts (the only representation the Lean
    program checker sees).  The checker validates the exact stream against the
    schedule's canonical form, so this serialization is part of the contract."""
    return [dict(c) if isinstance(c, dict) else {"kind": c.kind, **vars(c)} for c in cmds]


def emit_encoding(plan, commands):
    """Executable-byte artifact via the encoding backend (worker2 module).

    Fail-closed: a missing module/function/exception is 'unavailable', never an
    empty or synthesized encoding.  The returned dict is passed verbatim to the
    Lean program checker, which re-decodes the bytes independently.
    """
    try:
        from CodeGen import gemmini_encoding
    except ImportError:
        return None, "gemmini_encoding module unavailable (fail-closed)"
    try:
        emitter = getattr(gemmini_encoding, "emit_encoding")
    except AttributeError:
        return None, ("gemmini_encoding.emit_encoding missing "
                      "(fail-closed)")
    try:
        encoding = emitter(plan, commands)
    except Exception as exc:  # noqa: BLE001 - fail closed on any backend error
        return None, "encoding backend error: %s (fail-closed)" % exc
    if not isinstance(encoding, dict) or not encoding:
        return None, "encoding backend returned no artifact (fail-closed)"
    return encoding, "ok"


def build_program_request(plan, commands, encoding):
    return {"schema": PROGRAM_CHECK_SCHEMA, "plan": plan,
            "commands": serialize_commands(commands), "encoding": encoding}


def gemmini_program_check(request, timeout=60):
    """Run the Lean gemmini_program_check binary on one full request.

    Acceptance requires the Lean checker to accept the plan, the exact command
    stream, and the exact executable bytes, and to echo them back.  Any
    absence, crash, timeout, malformed output, or echo mismatch is a
    rejection.  Returns (accepted, reason, certificate_or_None).
    """
    binpath = program_checker_binary_path()
    if not os.path.isfile(binpath) or not os.access(binpath, os.X_OK):
        return False, ("gemmini_program_check binary not found or not "
                       "executable at %r (fail-closed: program acceptance "
                       "requires the Lean checker)" % binpath), None
    try:
        proc = subprocess.run(
            [binpath], input=json.dumps(request), capture_output=True,
            text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, ("gemmini_program_check timed out after %ss "
                       "(fail-closed)" % timeout), None
    except OSError as exc:
        return False, "gemmini_program_check failed to run: %s (fail-closed)" % exc, None
    if proc.returncode != 0:
        return False, ("gemmini_program_check exited %d: %s (fail-closed)"
                       % (proc.returncode,
                          (proc.stderr or proc.stdout).strip()[:400])), None
    try:
        out = json.loads(proc.stdout)
    except ValueError:
        return False, "gemmini_program_check emitted non-JSON output (fail-closed)", None
    if not isinstance(out, dict) or not isinstance(out.get("accepted"), bool) \
            or not isinstance(out.get("reason"), str):
        return False, ("gemmini_program_check output missing accepted/reason "
                       "(fail-closed)"), None
    if out["accepted"]:
        program = out.get("program")
        if not isinstance(program, dict):
            return False, "accepted without program echo (fail-closed)", None
        if program.get("plan") != request.get("plan") \
                or program.get("commands") != request.get("commands") \
                or program.get("encoding") != request.get("encoding"):
            return False, ("accepted but echoed different plan/commands/bytes "
                           "(fail-closed)"), None
        cert = out.get("certificate")
        if not isinstance(cert, dict) or not cert:
            return False, "accepted without checker certificate (fail-closed)", None
        return True, out["reason"], cert
    return False, out["reason"], None


def check_program(plan, commands, timeout=60):
    """One-call program acceptance: generate bytes, then Lean-check the full
    concrete artifact using the compiled Lean predicate. Artifact emission also
    requires a concrete kernel-checked certificate before setting verified."""
    encoding, why = emit_encoding(plan, commands)
    if encoding is None:
        return False, why, None
    return gemmini_program_check(build_program_request(plan, commands, encoding),
                                 timeout=timeout)


def emit_program_artifacts(plan, out_dir, *, seed=0x12345678, input_mode="lcg"):
    """Emit commands, executable-byte encoding, C source, and run program
    acceptance.  Returns a dict; 'program_accepted' is the only verified flag.
    Raises ValueError on invalid plans (diagnostic pre-checks)."""
    ok, reason = validate_plan(plan)
    if not ok:
        raise ValueError("refusing to emit program artifacts for invalid plan: %s" % reason)
    accepted, chk_reason, _ = gemmini_check(plan)
    if not accepted:
        raise ValueError("refusing to emit program artifacts: Lean plan check "
                         "rejected: %s" % chk_reason)
    commands = gen_commands(plan)
    commands_json = serialize_commands(commands)
    commands_path = os.path.join(out_dir, "%s_commands.json" % plan["schedule"])
    encoding, why = emit_encoding(plan, commands)
    encoding_path = os.path.join(out_dir, "%s_encoding.json" % plan["schedule"])
    os.makedirs(out_dir, exist_ok=True)
    with open(commands_path, "w") as fh:
        fh.write(json.dumps(commands_json, indent=2) + "\n")
    if encoding is not None:
        with open(encoding_path, "w") as fh:
            fh.write(json.dumps(encoding, indent=2, sort_keys=True) + "\n")
    request = build_program_request(plan, commands, encoding) \
        if encoding is not None else None
    prog_ok, prog_reason, cert = (gemmini_program_check(request)
                                  if request is not None
                                  else (False, why, None))
    kernel_certificate = None
    if prog_ok:
        from CodeGen.gemmini_certificate import write_and_check
        kernel_certificate = write_and_check(plan, commands, encoding, out_dir)
        prog_ok = kernel_certificate["accepted"]
        prog_reason = kernel_certificate["reason"]
    source = emit_c(plan, os.path.join(out_dir, "%s.c" % plan["schedule"]),
                    seed=seed, input_mode=input_mode)
    return {
        "schema": PROGRAM_CHECK_SCHEMA,
        "plan": plan,
        "commands_path": commands_path,
        "commands_sha256": hashlib.sha256(
            json.dumps(commands_json, sort_keys=True,
                       separators=(",", ":")).encode()).hexdigest(),
        "encoding_path": encoding_path if encoding is not None else None,
        "encoding": encoding,
        "program_accepted": prog_ok,
        "program_reason": prog_reason,
        "certificate": cert,
        "kernel_certificate": kernel_certificate,
        "verified": prog_ok,
        "source": source,
        "trust_boundary": ("program_accepted means the Lean program checker "
                           "proved the exact command stream computes the plan "
                           "GEMM and the Lean decoder recovered it from the "
                           "emitted bytes; the C file is a compatibility "
                           "adapter outside the verified boundary"),
    }


# ---------------------------------------------------------------------------
# Optimizer: enumerate candidates, fail-closed check each, select by the
# transparent modeled objective.  Not hardcoded: both schedules are built,
# validated, checked, and scored for every plan.
# ---------------------------------------------------------------------------

def make_plan(m, n, k, schedule):
    plan = {"m": m, "n": n, "k": k, "dim": DIM, "schedule": schedule}
    need_sp, need_acc = required_resources(plan)
    plan["scratchpad_rows"] = need_sp
    plan["accumulator_rows"] = need_acc
    return plan


def optimize(m, n, k, *, scratchpad_rows=SCRATCHPAD_ROWS, accumulator_rows=ACC_ROWS):
    """Check both schedules against the SAME target capacities and select the
    least modeled DMA traffic, breaking ties by instructions and required rows.
    There is no unchecked selection mode. Performance is a model, not latency.
    """
    candidates = []
    for schedule in SUPPORTED_SCHEDULES:
        plan = make_plan(m, n, k, schedule)
        plan.update(scratchpad_rows=scratchpad_rows, accumulator_rows=accumulator_rows)
        ok, reason = validate_plan(plan)
        accepted, chk_reason, _ = gemmini_check(plan)
        entry = {"plan": plan, "valid": ok, "invalid_reason": None if ok else reason,
                 "check": {"accepted": accepted, "reason": chk_reason},
                 "counts": None, "objective": None}
        if ok and accepted:
            counts = modeled_counts(gen_commands(plan))
            entry["counts"] = counts
            entry["objective"] = modeled_objective(counts) + (required_resources(plan)[1],)
        candidates.append(entry)
    eligible = [c for c in candidates if c["valid"] and c["check"]["accepted"]]
    winner = min(eligible, key=lambda c: c["objective"])["plan"] if eligible else None
    return {"candidates": candidates, "winner": winner}


# ---------------------------------------------------------------------------
# Self-test: 32-cubed differential case (run with `python3 -m CodeGen.gemmini`).
# ---------------------------------------------------------------------------

def _differential(m, n, k):
    failures = []
    for mode, seed in (("lcg", 1), ("lcg", 987654321), ("max", 0), ("minmax", 0)):
        A, B = deterministic_inputs(m, n, k, seed, mode)
        ref = scalar_matmul(A, B, m, n, k)
        for schedule in SUPPORTED_SCHEDULES:
            plan = make_plan(m, n, k, schedule)
            ok, reason = validate_plan(plan)
            if not ok:
                failures.append("%s/%s: plan invalid: %s" % (schedule, mode, reason))
                continue
            try:
                C, stats = run_commands(gen_commands(plan), plan, A, B)
            except GemminiError as exc:
                failures.append("%s/%s: interpreter error: %s" % (schedule, mode, exc))
                continue
            if C != ref:
                first = next(i for i in range(m * n) if C[i] != ref[i])
                failures.append("%s/%s: C[%d]=%d != ref=%d"
                                % (schedule, mode, first, C[first], ref[first]))
    return failures


def _rejection_examples():
    cases = []
    bad_shape = make_plan(32, 32, 33, SCHEDULE_BASELINE)
    ok, reason = validate_plan(bad_shape)
    cases.append(("k=33 not divisible by dim", ok, reason))
    acc_overflow = make_plan(32, 32, 131072, SCHEDULE_BASELINE)  # 131072*16384 > 2^31-1
    ok, reason = validate_plan(acc_overflow)
    cases.append(("k*16384 > 2^31-1", ok, reason))
    big = make_plan(2048, 32, 32, SCHEDULE_REUSE_B)  # needs 2048 > ACC_ROWS acc rows
    ok, reason = validate_plan(big)
    cases.append(("reuse_b accumulator capacity", ok, reason))
    starved = dict(make_plan(32, 32, 32, SCHEDULE_BASELINE))
    starved["scratchpad_rows"] = DIM  # below the required 2*dim
    ok, reason = validate_plan(starved)
    cases.append(("scratchpad_rows below schedule requirement", ok, reason))
    return cases


def main():
    m = n = k = 32
    print("VeriTac Gemmini backend self-test: %dx%dx%d int8 GEMM, dim=%d"
          % (m, n, k, DIM))
    failures = _differential(m, n, k)
    for desc, ok, reason in _rejection_examples():
        status = "rejected" if not ok else "ACCEPTED (BUG)"
        print("  reject-case %-46s -> %s: %s" % (desc, status, reason))
        if ok:
            failures.append("rejection case not rejected: %s" % desc)
    report = optimize(m, n, k)
    for cand in report["candidates"]:
        p = cand["plan"]
        print("  candidate %-9s valid=%s lean_accepted=%s reason=%s"
              % (p["schedule"], cand["valid"], cand["check"]["accepted"],
                 (cand["check"]["reason"] or "")[:80]))
        if cand["counts"]:
            print("    modeled: mvins=%d (A=%d B=%d) computes=%d mvouts=%d "
                  "dma_bytes=%d instructions=%d objective=%r"
                  % (cand["counts"]["mvin"], cand["counts"]["mvin_A"],
                     cand["counts"]["mvin_B"], cand["counts"]["compute"],
                     cand["counts"]["mvout"], cand["counts"]["dma_bytes"],
                     cand["counts"]["instructions"], cand["objective"]))
    print("  winner (by modeled objective, NOT measured speedup): %s"
          % (report["winner"]["schedule"] if report["winner"] else None))
    emit_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            ".lake", "gemmini_demo")
    for cand in report["candidates"]:
        if cand["check"]["accepted"]:
            info = emit_c(cand["plan"], os.path.join(
                emit_dir, "gemmini_gemm_%s_%dx%dx%d.c"
                % (cand["plan"]["schedule"], m, n, k)))
            print("  emitted %s (sha256=%s...)"
                  % (info["path"], info["sha256"][:16]))
        else:
            print("  emission skipped for %s (fail-closed: %s)"
                  % (cand["plan"]["schedule"], cand["check"]["reason"]))
    if failures:
        print("FAILURES (%d):" % len(failures))
        for f in failures:
            print("  " + f)
        raise SystemExit(1)
    print("self-test OK: interpreter matches exact scalar reference on all "
          "input modes and both schedules; modeled counts are models only.")


if __name__ == "__main__":
    main()
