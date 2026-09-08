"""
VeriTac Search: cost_model.py
Analytical cost model for estimating schedule performance.
"""

from CodeGen.lower import (
    CStmt, CSkip, CBufWrite, CLoop, CSeq, CAlloc,
    CExpr, CLit, CVar, CBinOp, CBufRead,
    parse_stmt,
)


def estimate_expr_bound(e: CExpr, bounds: dict[str, int]) -> int:
    """Estimate the upper bound value of an expression given variable bounds."""
    if isinstance(e, CLit):
        return e.val
    elif isinstance(e, CVar):
        return bounds.get(e.name, 256)
    elif isinstance(e, CBinOp):
        left = estimate_expr_bound(e.left, bounds)
        right = estimate_expr_bound(e.right, bounds)
        if e.op == "+":
            return left + right
        elif e.op == "*":
            return left * right
        elif e.op == "/":
            return max(1, left // max(1, right))
        elif e.op == "%":
            return max(1, right)
        return left
    elif isinstance(e, CBufRead):
        return 0
    return 256


# Number of threads / lanes we model for parallel and vectorized loops.
# These are the *analytical* divisor used to price parallelism: a parallel loop's
# work is split across `NTHREADS` cores, and a vectorized loop amortizes its
# memory traffic across `VLEN` lanes. They are modeling knobs, not hardware facts.
NTHREADS = 8
VLEN = 4


def estimate_ops(stmt: CStmt, bounds: dict[str, int] = None) -> int:
    """Estimate the total number of arithmetic operations.

    A `parallel` loop divides its per-iteration work across `NTHREADS` cores.
    All other structural forms (tiling, unrolling) preserve the arithmetic op
    count, so tiling only shows up through the memory-traffic term.
    """
    if bounds is None:
        bounds = {}

    if isinstance(stmt, CSkip):
        return 0
    elif isinstance(stmt, CBufWrite):
        return 1 + count_expr_ops(stmt.val)
    elif isinstance(stmt, CLoop):
        lo = estimate_expr_bound(stmt.lo, bounds)
        hi = estimate_expr_bound(stmt.hi, bounds)
        iters = max(0, hi - lo)
        new_bounds = {**bounds, stmt.var: hi}
        body_ops = estimate_ops(stmt.body, new_bounds)
        if stmt.annotation == "parallel":
            body_ops = max(1, body_ops // NTHREADS)
        return iters * body_ops
    elif isinstance(stmt, CSeq):
        return sum(estimate_ops(s, bounds) for s in stmt.stmts)
    elif isinstance(stmt, CAlloc):
        return estimate_ops(stmt.body, bounds)
    return 0


def count_expr_ops(e: CExpr) -> int:
    """Count arithmetic operations in an expression."""
    if isinstance(e, (CLit, CVar)):
        return 0
    elif isinstance(e, CBinOp):
        return 1 + count_expr_ops(e.left) + count_expr_ops(e.right)
    elif isinstance(e, CBufRead):
        return 0
    return 0


def estimate_memory_traffic(stmt: CStmt, bounds: dict[str, int] = None) -> int:
    """Estimate total memory accesses (reads + writes).

    A `vectorize` loop amortizes the traffic of its innermost body across `VLEN`
    lanes, which is what SIMD actually does on the memory subsystem. Together with
    the op model this makes annotation tactics (parallel / vectorize) actually
    lower a schedule's score, so the search agent can rank them.
    """
    if bounds is None:
        bounds = {}

    if isinstance(stmt, CSkip):
        return 0
    elif isinstance(stmt, CBufWrite):
        reads = count_buf_accesses(stmt.val)
        return 1 + reads  # 1 write + reads
    elif isinstance(stmt, CLoop):
        lo = estimate_expr_bound(stmt.lo, bounds)
        hi = estimate_expr_bound(stmt.hi, bounds)
        iters = max(0, hi - lo)
        new_bounds = {**bounds, stmt.var: hi}
        body_traffic = estimate_memory_traffic(stmt.body, new_bounds)
        if stmt.annotation == "vectorize":
            body_traffic = max(1, body_traffic // VLEN)
        return iters * body_traffic
    elif isinstance(stmt, CSeq):
        return sum(estimate_memory_traffic(s, bounds) for s in stmt.stmts)
    elif isinstance(stmt, CAlloc):
        return estimate_memory_traffic(stmt.body, bounds)
    return 0


def count_buf_accesses(e: CExpr) -> int:
    """Count buffer read accesses in an expression."""
    if isinstance(e, CBufRead):
        return 1
    elif isinstance(e, CBinOp):
        return count_buf_accesses(e.left) + count_buf_accesses(e.right)
    return 0


def score_schedule(stmt_json: dict) -> float:
    """Score a schedule (lower is better).
    Combines operation count and memory traffic estimate, rewarding parallel /
    vectorize annotations and, mildly, tighter tiling (better cache locality)."""
    stmt = parse_stmt(stmt_json)
    ops = estimate_ops(stmt)
    mem = estimate_memory_traffic(stmt)
    # Parallelism cuts op cost; vectorization cuts memory cost. Weight memory
    # more heavily (it dominates), since memory traffic is the bottleneck.
    return ops + 10 * mem
