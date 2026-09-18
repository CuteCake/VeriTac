"""Advisory structured rejection diagnostics for the full-tile Gemmini
command language.

ADVISORY ONLY.  This module NEVER authorizes acceptance: every report
carries ``advisory_only: true``, and ``no_diagnosed_failure`` explicitly
means only that this advisory model found no failure -- it is NOT
acceptance.  The existing Lean verifier (``gemmini_program_check``,
backed by ``VeriTac.Gemmini.Symbolic.check`` / ``Engine.gstep``) remains
the sole authority; this module only describes the first failure it can
reproduce with the same faithfully mirrored guards.

``diagnose(plan, commands)`` returns a JSON-serializable report with a
``status`` that strictly distinguishes:

  * ``plan_rejected``            -- the plan is malformed, outside the
                                    legality boundary of
                                    ``VeriTac.Gemmini.Plan``, or outside
                                    the DIM=16 instruction model.
  * ``syntax_unsupported``       -- a command does not parse under the
                                    strict checker parser (unknown kind,
                                    wrong field set, noninteger or
                                    negative operands, unsupported
                                    slot/buf/scale).
  * ``execution_rejected``       -- the mirrored ``Engine.gstep``
                                    transition rejects some instruction;
                                    the report gives the instruction
                                    index/kind, the failed obligation,
                                    and address/capacity/initialization
                                    facts.
  * ``completion_missing``       -- execution completed but the stream
                                    is not fence-drained.
  * ``output_obligation_mismatch`` -- execution and drain completed but
                                    some output cell is unwritten or its
                                    input-product sequence differs from
                                    the GEMM specification.  No
                                    counterexample value is invented.
  * ``no_diagnosed_failure``     -- the advisory model found no failure
                                    (NOT acceptance).
  * ``inconclusive``             -- the model cannot decide; it never
                                    guesses.

Only actual failure facts are reported.  No optimization or repair
suggestions are made.  The model tracks metadata (which input tile was
moved into which scratchpad row, accumulator initialization, retained
weights, the pending preload, and output coverage) with the same guards
as ``Engine.lean``; per-cell output obligations are compared as
ordered lists of input-element references, so no scalar or symbolic
arithmetic products are evaluated here.  If a guard depends on
information this module does not model, the report says ``inconclusive``
instead of guessing.
"""

GARBAGE_ADDR = (1 << 32) - 1
ACC_ROW_MODULUS = 1 << 29
ACC_ADDR_LIMIT = 1 << 32
DIM = 16
INT8_PRODUCT_BOUND = 16384
INT32_MAX = 2147483647

PLAN_FIELDS = ("m", "n", "k", "dim", "scratchpad_rows",
               "accumulator_rows", "schedule")
SCHEDULES = ("baseline", "reuse_b")

STATUS_PLAN_REJECTED = "plan_rejected"
STATUS_SYNTAX_UNSUPPORTED = "syntax_unsupported"
STATUS_EXECUTION_REJECTED = "execution_rejected"
STATUS_COMPLETION_MISSING = "completion_missing"
STATUS_OUTPUT_MISMATCH = "output_obligation_mismatch"
STATUS_NO_FAILURE = "no_diagnosed_failure"
STATUS_INCONCLUSIVE = "inconclusive"

AUTHORITY = ("gemmini_program_check / VeriTac.Gemmini.Symbolic.check "
             "(existing Lean verifier is authoritative)")
NOT_ACCEPTED_NOTE = ("no_diagnosed_failure is NOT acceptance; only the "
                     "authoritative Lean checker can accept")

_CMD_FIELDS = {
    "config_ex": {"kind", "dataflow"},
    "config_ld": {"kind", "slot", "stride_bytes", "scale"},
    "config_st": {"kind", "stride_bytes"},
    "mvin": {"kind", "slot", "buf", "offset", "spad_addr", "cols", "rows"},
    "preload": {"kind", "bd_spad_addr", "out_addr"},
    "compute": {"kind", "accumulated", "a_spad_addr", "bd_spad_addr"},
    "mvout": {"kind", "buf_offset", "acc_addr", "cols", "rows"},
    "fence": {"kind"},
}


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _report(status, summary, instruction=None, obligation=None, facts=None):
    report = {"advisory_only": True, "authoritative_checker": AUTHORITY,
              "status": status, "summary": summary}
    if instruction is not None:
        report["instruction"] = {"index": instruction[0],
                                 "kind": instruction[1]}
    if obligation is not None:
        report["failed_obligation"] = obligation
    if facts is not None:
        report["facts"] = facts
    if status == STATUS_NO_FAILURE:
        report["note"] = NOT_ACCEPTED_NOTE
    return report


# ---------------------------------------------------------------------------
# Plan parsing and legality (mirrors progParsePlan / planLegalProps /
# planDiagnostic in VeriTac.Gemmini.Plan)
# ---------------------------------------------------------------------------

def _strict_nat(value):
    """Mirror of progNatField? strictness: a JSON number is a strict
    nonnegative integer exactly when its decimal expansion is integral."""
    if type(value) is int:
        return value >= 0
    if type(value) is float:
        return value.is_integer() and value >= 0
    return False


def _nat(value):
    return int(value)


def _parse_plan(plan):
    """Mirror the strict plan parse; returns (fields, None) or (None, why)."""
    if not isinstance(plan, dict):
        return None, "plan must be a JSON object"
    if set(plan) != set(PLAN_FIELDS):
        return None, ("plan must have exactly the fields "
                      "m/n/k/dim/scratchpad_rows/accumulator_rows/schedule")
    values = {}
    for field in PLAN_FIELDS[:-1]:
        if not _strict_nat(plan[field]):
            return None, "plan field %r must be a nonnegative integer" % field
        values[field] = _nat(plan[field])
    if plan["schedule"] not in SCHEDULES:
        return None, "unsupported schedule"
    values["schedule"] = plan["schedule"]
    return values, None


def _required_acc_rows(p):
    return p["dim"] if p["schedule"] == "baseline" else p["m"]


def _plan_diagnostic(p):
    """Mirror of VeriTac.Gemmini.planDiagnostic (first violated clause wins)."""
    if p["m"] == 0:
        return "m must be positive"
    if p["n"] == 0:
        return "n must be positive"
    if p["k"] == 0:
        return "k must be positive"
    if p["dim"] == 0:
        return "dim must be positive"
    if p["m"] % p["dim"] != 0:
        return "m must be divisible by dim"
    if p["n"] % p["dim"] != 0:
        return "n must be divisible by dim"
    if p["k"] % p["dim"] != 0:
        return "k must be divisible by dim"
    if p["k"] * INT8_PRODUCT_BOUND > INT32_MAX:
        return "k * 16384 exceeds the int32 accumulation bound"
    if p["scratchpad_rows"] < 2 * p["dim"]:
        return "scratchpad rows below 2 * dim"
    if p["accumulator_rows"] < _required_acc_rows(p):
        return "accumulator rows below the schedule requirement"
    return "accepted"


# ---------------------------------------------------------------------------
# Command parsing (mirrors progParseCmd strictness)
# ---------------------------------------------------------------------------

def _parse_commands(commands):
    """Mirror the strict command parse; returns (list, None) or (None, (index, why))."""
    if not isinstance(commands, (list, tuple)):
        return None, (None, "commands must be a JSON array")
    parsed = []
    for index, raw in enumerate(commands):
        if not isinstance(raw, dict):
            if hasattr(raw, "kind") and hasattr(raw, "__dict__"):
                raw = {"kind": raw.kind, **vars(raw)}
            else:
                return None, (index, "command must be a JSON object")
        kind = raw.get("kind")
        if not isinstance(kind, str) or kind not in _CMD_FIELDS:
            return None, (index, "unknown command kind")
        if set(raw) != _CMD_FIELDS[kind]:
            return None, (index, "command fields do not match the kind exactly")
        if kind == "config_ex":
            if not _strict_nat(raw["dataflow"]):
                return None, (index, "dataflow must be a nonnegative integer")
            parsed.append(("config_ex", _nat(raw["dataflow"])))
        elif kind == "config_ld":
            if not _strict_nat(raw["slot"]) or raw["slot"] > 2:
                return None, (index, "config_ld slot must be 0, 1, or 2")
            if not _strict_nat(raw["stride_bytes"]):
                return None, (index, "stride_bytes must be a nonnegative integer")
            scale = raw["scale"]
            if type(scale) is bool or type(scale) not in (int, float) \
                    or scale != 1:
                return None, (index, "only identity load scale is supported")
            parsed.append(("config_ld", _nat(raw["slot"]),
                           _nat(raw["stride_bytes"])))
        elif kind == "config_st":
            if not _strict_nat(raw["stride_bytes"]):
                return None, (index, "stride_bytes must be a nonnegative integer")
            parsed.append(("config_st", _nat(raw["stride_bytes"])))
        elif kind == "mvin":
            if not _strict_nat(raw["slot"]) or raw["slot"] > 1:
                return None, (index, "mvin slot must be 0 or 1")
            if raw["buf"] not in ("A", "B"):
                return None, (index, "buf must be A or B")
            for field in ("offset", "spad_addr", "cols", "rows"):
                if not _strict_nat(raw[field]):
                    return None, (index, "%s must be a nonnegative integer" % field)
            parsed.append(("mvin", _nat(raw["slot"]), raw["buf"],
                           _nat(raw["offset"]), _nat(raw["spad_addr"]),
                           _nat(raw["cols"]), _nat(raw["rows"])))
        elif kind == "preload":
            for field in ("bd_spad_addr", "out_addr"):
                if not _strict_nat(raw[field]) or raw[field] > 0xFFFFFFFF:
                    return None, (index, "%s must be a uint32" % field)
            parsed.append(("preload", _nat(raw["bd_spad_addr"]),
                           _nat(raw["out_addr"])))
        elif kind == "compute":
            if type(raw["accumulated"]) is not bool:
                return None, (index, "accumulated must be a Boolean")
            if not _strict_nat(raw["a_spad_addr"]):
                return None, (index, "a_spad_addr must be a nonnegative integer")
            if not _strict_nat(raw["bd_spad_addr"]) or raw["bd_spad_addr"] > 0xFFFFFFFF:
                return None, (index, "bd_spad_addr must be a uint32")
            parsed.append(("compute", raw["accumulated"],
                           _nat(raw["a_spad_addr"]), _nat(raw["bd_spad_addr"])))
        elif kind == "mvout":
            for field in ("buf_offset", "cols", "rows"):
                if not _strict_nat(raw[field]):
                    return None, (index, "%s must be a nonnegative integer" % field)
            if not _strict_nat(raw["acc_addr"]) or raw["acc_addr"] > 0xFFFFFFFF:
                return None, (index, "acc_addr must be a uint32")
            parsed.append(("mvout", _nat(raw["buf_offset"]),
                           _nat(raw["acc_addr"]), _nat(raw["cols"]),
                           _nat(raw["rows"])))
        else:
            parsed.append(("fence",))
    return parsed, None


# ---------------------------------------------------------------------------
# Shared address semantics (mirrors Engine.tileFits / accAddress / accumulates)
# ---------------------------------------------------------------------------

def _tile_fits(row, capacity):
    return row % DIM == 0 and row + DIM <= capacity


def _acc_row(addr):
    return addr % ACC_ROW_MODULUS


def _acc_address(addr, capacity):
    return (addr < ACC_ADDR_LIMIT and addr >> 31 == 1
            and (addr >> 29) % 2 == 1 and _tile_fits(_acc_row(addr), capacity))


def _accumulates(addr):
    return (addr >> 30) % 2 == 1


# ---------------------------------------------------------------------------
# Symbolic metadata interpreter (mirrors Engine.gstep / Symbolic.ops)
# ---------------------------------------------------------------------------

class _State:
    __slots__ = ("ws", "ld0", "ld1", "st_stride", "spad", "acc", "pending",
                 "weights", "output", "written", "drained")

    def __init__(self):
        self.ws = False
        self.ld0 = 0
        self.ld1 = 0
        self.st_stride = 0
        self.spad = {}        # row -> (buf, offset, stride) provenance
        self.acc = {}         # accRow -> 16x16 per-cell term lists
        self.pending = None   # (bdRow, outAddr)
        self.weights = None   # (buf, offset, stride) provenance
        self.output = {}      # flat index -> term list
        self.written = set()
        self.drained = True


def _a_ref(index):
    return ("a", index)


def _b_ref(index):
    return ("b", index)


def _a_elements(prov):
    """The 16x16 A-tile input references fun r t => a (offset + r*stride + t)."""
    buf, offset, stride = prov
    return [[(buf.lower(), offset + r * stride + t) for t in range(DIM)]
            for r in range(DIM)]


def _b_elements(prov):
    """The 16x16 B-tile input references fun t c => b (offset + t*stride + c)."""
    buf, offset, stride = prov
    return [[(buf.lower(), offset + t * stride + c) for c in range(DIM)]
            for t in range(DIM)]


def _step(plan, st, instr):
    """One mirrored gstep.  Returns None on success or
    (obligation, facts) on the first failed obligation."""
    kind = instr[0]
    if kind == "config_ex":
        dataflow = instr[1]
        if dataflow != 1:
            return ("config_ex requires weight-stationary dataflow 1",
                    {"dataflow": dataflow})
        st.ws = True
        return None
    if kind == "config_ld":
        slot, stride = instr[1], instr[2]
        if slot == 0:
            st.ld0 = stride
        elif slot == 1:
            st.ld1 = stride
        else:
            return ("config_ld slot must be 0 or 1",
                    {"slot": slot})
        return None
    if kind == "config_st":
        stride = instr[1]
        if not (stride > 0 and stride % 4 == 0):
            return ("config_st requires stride_bytes > 0 and divisible by 4",
                    {"stride_bytes": stride})
        st.st_stride = stride
        return None
    if kind == "mvin":
        slot, buf, offset, spad, cols, rows = instr[1:]
        if slot == 0:
            stride = st.ld0
        elif slot == 1:
            stride = st.ld1
        else:
            return ("mvin slot must be 0 or 1", {"slot": slot})
        size = plan["m"] * plan["k"] if buf == "A" else plan["k"] * plan["n"]
        if not stride > 0:
            return ("mvin requires a positive load stride configured by "
                    "config_ld for its slot",
                    {"slot": slot, "stride": stride})
        if cols != DIM:
            return ("mvin requires full %d-column tiles" % DIM,
                    {"cols": cols})
        if rows != DIM:
            return ("mvin requires full %d-row tiles" % DIM,
                    {"rows": rows})
        extent = offset + 15 * stride + DIM
        if extent > size:
            return ("mvin host range exceeds the %s buffer extent"
                    % ("A" if buf == "A" else "B"),
                    {"buf": buf, "offset": offset, "stride": stride,
                     "required_extent": extent, "buffer_size": size})
        if not _tile_fits(spad, plan["scratchpad_rows"]):
            return ("mvin scratchpad destination requires row %% %d == 0 "
                    "and row + %d <= scratchpad_rows" % (DIM, DIM),
                    {"spad_addr": spad, "scratchpad_rows":
                     plan["scratchpad_rows"], "row_mod_16": spad % DIM})
        st.spad[spad] = (buf, offset, stride)
        st.drained = False
        return None
    if kind == "preload":
        bd, out = instr[1], instr[2]
        if not (bd == GARBAGE_ADDR or _tile_fits(bd, plan["scratchpad_rows"])):
            return ("preload requires bd_spad_addr == %d (GARBAGE_ADDR) or "
                    "tileFits(bd, scratchpad_rows)" % GARBAGE_ADDR,
                    {"bd_spad_addr": bd,
                     "scratchpad_rows": plan["scratchpad_rows"],
                     "bd_row_mod_16": bd % DIM})
        if not _acc_address(out, plan["accumulator_rows"]):
            return ("preload requires an accumulator output address with "
                    "bit31 and bit29 set and tileFits(accRow, "
                    "accumulator_rows)",
                    {"out_addr": out, "addr_below_2pow32": out < ACC_ADDR_LIMIT,
                     "bit31": out >> 31 == 1, "bit29": (out >> 29) % 2 == 1,
                     "acc_row": _acc_row(out),
                     "accumulator_rows": plan["accumulator_rows"]})
        st.pending = (bd, out)
        return None
    if kind == "compute":
        accumulated, a_row, bd = instr[1], instr[2], instr[3]
        if not st.ws:
            return ("compute requires weight-stationary dataflow configured "
                    "by config_ex 1", {"ws_configured": st.ws})
        if bd != GARBAGE_ADDR:
            return ("compute requires bd_spad_addr == %d (GARBAGE_ADDR)"
                    % GARBAGE_ADDR, {"bd_spad_addr": bd})
        if st.pending is None:
            return ("compute requires a preceding preload",
                    {"pending": None})
        b_row, out = st.pending
        a_prov = st.spad.get(a_row)
        if a_prov is None:
            return ("compute requires the A tile loaded at a_spad_addr",
                    {"a_spad_addr": a_row,
                     "a_tile_loaded": False})
        if accumulated:
            if b_row != GARBAGE_ADDR:
                return ("compute with accumulated=true requires the pending "
                        "preload's bd_spad_addr to be GARBAGE_ADDR",
                        {"pending_bd_spad_addr": b_row})
            if st.weights is None:
                return ("compute with accumulated=true requires retained B "
                        "weights from a previous preloaded compute",
                        {"retained_weights_initialized": False})
            b_prov = st.weights
        else:
            b_prov = st.spad.get(b_row)
            if b_prov is None:
                return ("compute with accumulated=false requires the B tile "
                        "loaded at the preload's bd_spad_addr",
                        {"bd_spad_addr": b_row, "b_tile_loaded": False})
        if _accumulates(out):
            old = st.acc.get(_acc_row(out))
            if old is None:
                return ("compute onto an accumulate-mode accumulator "
                        "address requires prior accumulator "
                        "initialization at accRow",
                        {"out_addr": out, "accumulates": True,
                         "acc_row": _acc_row(out),
                         "acc_row_initialized": False})
        else:
            old = [[[] for _ in range(DIM)] for _ in range(DIM)]
        a_elements = _a_elements(a_prov)
        b_elements = _b_elements(b_prov)
        result = [[old[r][c] + [[a_elements[r][t], b_elements[t][c]]
                                for t in range(DIM)]
                   for c in range(DIM)] for r in range(DIM)]
        st.acc[_acc_row(out)] = result
        st.weights = b_prov
        st.pending = None
        st.drained = False
        return None
    if kind == "mvout":
        offset, addr, cols, rows = instr[1:]
        if cols != DIM:
            return ("mvout requires full %d-column tiles" % DIM,
                    {"cols": cols})
        if rows != DIM:
            return ("mvout requires full %d-row tiles" % DIM,
                    {"rows": rows})
        if st.st_stride == 0:
            return ("mvout requires config_st with a positive stride",
                    {"st_stride": st.st_stride})
        if st.st_stride % 4 != 0:
            return ("mvout requires config_st stride divisible by 4",
                    {"st_stride": st.st_stride})
        if not _acc_address(addr, plan["accumulator_rows"]):
            return ("mvout requires an accumulator source address with "
                    "bit31 and bit29 set and tileFits(accRow, "
                    "accumulator_rows)",
                    {"acc_addr": addr,
                     "addr_below_2pow32": addr < ACC_ADDR_LIMIT,
                     "bit31": addr >> 31 == 1, "bit29": (addr >> 29) % 2 == 1,
                     "acc_row": _acc_row(addr),
                     "accumulator_rows": plan["accumulator_rows"]})
        stride = st.st_stride // 4
        if stride < DIM:
            return ("mvout stride in elements must be at least %d" % DIM,
                    {"stride_bytes": st.st_stride, "stride_elements": stride})
        extent = offset + 15 * stride + DIM
        if extent > plan["m"] * plan["n"]:
            return ("mvout host range exceeds the C buffer extent",
                    {"buf_offset": offset, "stride_elements": stride,
                     "required_extent": extent,
                     "c_elements": plan["m"] * plan["n"]})
        tile = st.acc.get(_acc_row(addr))
        if tile is None:
            return ("mvout requires the accumulator tile initialized at "
                    "accRow",
                    {"acc_addr": addr, "acc_row": _acc_row(addr),
                     "acc_row_initialized": False})
        for idx in range(plan["m"] * plan["n"]):
            if (offset <= idx and (idx - offset) // stride < DIM
                    and (idx - offset) % stride < DIM):
                st.output[idx] = tile[(idx - offset) // stride][(idx - offset) % stride]
                st.written.add(idx)
        st.drained = False
        return None
    if kind == "fence":
        st.drained = True
        return None
    return ("unsupported command kind", {"kind": kind})


def _expected_terms(p, r, c):
    """Mirrors Symbolic.expected: the GEMM product input references for one
    cell, in reduction order."""
    return [[("a", r * p["k"] + t), ("b", t * p["n"] + c)]
            for t in range(p["k"])]


def _cells_first_mismatch(p, st):
    """Mirrors Symbolic.cellsCorrect; returns the first bad cell or None."""
    for r in range(p["m"]):
        for c in range(p["n"]):
            idx = r * p["n"] + c
            if idx not in st.written:
                return {"cell_index": idx, "row": r, "col": c,
                        "written": False}
            if st.output.get(idx) != _expected_terms(p, r, c):
                return {"cell_index": idx, "row": r, "col": c,
                        "written": True,
                        "product_sequence_matches": False}
    return None


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------

def diagnose(plan, commands, *, max_work=2_000_000):
    """Advisory structured rejection diagnostics for one plan + command stream.

    Returns a JSON-serializable dict.  See the module docstring for the
    status contract.  This function never authorizes acceptance and never
    suggests repairs; the existing Lean checker is authoritative.
    """
    parsed_plan, plan_why = _parse_plan(plan)
    if parsed_plan is None:
        return _report(STATUS_PLAN_REJECTED, "malformed plan: " + plan_why)

    parsed_commands, cmd_why = _parse_commands(commands)
    if parsed_commands is None:
        index, why = cmd_why
        return _report(STATUS_SYNTAX_UNSUPPORTED,
                       "command stream does not parse under the strict "
                       "checker syntax: " + why,
                       instruction=None if index is None else (index, "parse"),
                       facts={"index": index})

    plan_reason = _plan_diagnostic(parsed_plan)
    if plan_reason != "accepted":
        return _report(STATUS_PLAN_REJECTED,
                       "plan legality failed: " + plan_reason,
                       facts={"plan_diagnostic": plan_reason})
    if parsed_plan["dim"] != DIM:
        return _report(STATUS_PLAN_REJECTED,
                       "instruction model requires DIM=16",
                       facts={"dim": parsed_plan["dim"]})

    computes = sum(c[0] == "compute" for c in parsed_commands)
    stores = sum(c[0] == "mvout" for c in parsed_commands)
    cells = parsed_plan["m"] * parsed_plan["n"]
    estimated_work = computes * DIM ** 3 + cells * (stores + parsed_plan["k"])
    if len(parsed_commands) > 4096 or computes > 256 or estimated_work > max_work:
        return _report(STATUS_INCONCLUSIVE, "advisory work budget exceeded",
                       facts={"estimated_work": estimated_work, "max_work": max_work})

    state = _State()
    for index, instr in enumerate(parsed_commands):
        failure = _step(parsed_plan, state, instr)
        if failure is not None:
            obligation, facts = failure
            return _report(STATUS_EXECUTION_REJECTED,
                           "instruction execution rejected",
                           instruction=(index, instr[0]),
                           obligation=obligation, facts=facts)

    if not state.drained:
        return _report(STATUS_COMPLETION_MISSING,
                       "completion obligation failed: missing final fence",
                       facts={"drained": False,
                              "instructions": len(parsed_commands)})

    mismatch = _cells_first_mismatch(parsed_plan, state)
    if mismatch is not None:
        return _report(STATUS_OUTPUT_MISMATCH,
                       "GEMM output obligation failed: a cell is unwritten "
                       "or its input-product sequence differs from the "
                       "specification (no counterexample value is invented)",
                       facts=mismatch)

    return _report(STATUS_NO_FAILURE,
                   "no diagnosed failure under the mirrored advisory model")
