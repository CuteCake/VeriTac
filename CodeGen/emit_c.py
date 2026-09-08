"""
VeriTac CodeGen: emit_c.py
Emit C code from the Python AST.
"""

from .lower import (
    CExpr, CLit, CVar, CBinOp, CBufRead,
    CStmt, CSkip, CBufWrite, CLoop, CSeq, CAlloc,
)


def emit_expr(e: CExpr) -> str:
    """Emit a C expression string."""
    if isinstance(e, CLit):
        return str(e.val)
    elif isinstance(e, CVar):
        return e.name
    elif isinstance(e, CBinOp):
        left = emit_expr(e.left)
        right = emit_expr(e.right)
        return f"({left} {e.op} {right})"
    elif isinstance(e, CBufRead):
        idx = flat_index_expr(e.buf, e.indices)
        return f"{e.buf}[{idx}]"
    else:
        raise ValueError(f"Unknown CExpr type: {type(e)}")


def flat_index_expr(buf: str, indices: list[CExpr]) -> str:
    """Generate a flat index expression for multi-dimensional buffer access.
    Uses row-major order with a stride multiplier."""
    if not indices:
        return "0"
    parts = []
    for i, idx in enumerate(indices):
        expr = emit_expr(idx)
        # Use a large stride multiplier for simplicity (matching Lean's flatIndex)
        if i == 0:
            stride = "1" + " * 1000" * (len(indices) - 1)
            parts.append(f"({expr} * {stride})" if len(indices) > 1 else expr)
        else:
            stride = "1" + " * 1000" * (len(indices) - 1 - i)
            parts.append(f"({expr} * {stride})" if i < len(indices) - 1 else expr)
    return " + ".join(parts)


def expr_mentions(e: CExpr, name: str) -> bool:
    """True if the expression references variable `name`."""
    if isinstance(e, CLit):
        return False
    if isinstance(e, CVar):
        return e.name == name
    if isinstance(e, CBinOp):
        return expr_mentions(e.left, name) or expr_mentions(e.right, name)
    if isinstance(e, CBufRead):
        return any(expr_mentions(i, name) for i in e.indices)
    return False


def writes_local_to(loop_var: str, stmt: CStmt) -> bool:
    """True iff *every* buffer write inside `stmt` depends on `loop_var`.

    This is the safe condition for both `parallel` (distinct threads touch
    distinct cells) and `vectorize` (distinct lanes touch distinct cells).
    A loop that writes to a cell *without* mentioning its own variable is a
    cross-iteration reduction/accumulation and must NOT get a pragma — that is
    exactly when OpenMP races or a SIMD lane conflict would change semantics.
    This mirrors the Lean `checkIndependence` guard and the proved
    `indexLocalP_flat_leading` criterion.
    """
    if isinstance(stmt, CBufWrite):
        return any(expr_mentions(i, loop_var) for i in stmt.indices)
    if isinstance(stmt, CLoop):
        return writes_local_to(loop_var, stmt.body)
    if isinstance(stmt, CSeq):
        return all(writes_local_to(loop_var, sub) for sub in stmt.stmts)
    if isinstance(stmt, CAlloc):
        return writes_local_to(loop_var, stmt.body)
    # CSkip and anything else: no writes, vacuously local
    return True


def emit_stmt(s: CStmt, indent: int = 1) -> str:
    """Emit a C statement string."""
    pad = "    " * indent

    if isinstance(s, CSkip):
        return f"{pad}/* skip */\n"
    elif isinstance(s, CBufWrite):
        idx = flat_index_expr(s.buf, s.indices)
        val = emit_expr(s.val)
        return f"{pad}{s.buf}[{idx}] = {val};\n"
    elif isinstance(s, CLoop):
        lo = emit_expr(s.lo)
        hi = emit_expr(s.hi)
        pragma = ""
        if s.annotation == "parallel":
            local = writes_local_to(s.var, s.body)
            if local:
                pragma = f"{pad}#pragma omp parallel for\n"
            else:
                pragma = (f"{pad}/* parallel suppressed: body writes cells "
                          f"not local to {s.var} */\n")
        elif s.annotation == "vectorize":
            if writes_local_to(s.var, s.body):
                pragma = f"{pad}#pragma omp simd\n"
            else:
                pragma = (f"{pad}/* simd suppressed: reduction/accumulation "
                          f"over {s.var} */\n")
        header = f"{pad}for (int {s.var} = {lo}; {s.var} < {hi}; {s.var}++)"
        body = emit_stmt(s.body, indent + 1)
        return f"{pragma}{header} {{\n{body}{pad}}}\n"
    elif isinstance(s, CSeq):
        return "".join(emit_stmt(sub, indent) for sub in s.stmts)
    elif isinstance(s, CAlloc):
        # Compute flat size (product of shapes)
        size_parts = [emit_expr(d) for d in s.shape]
        size = " * ".join(size_parts) if size_parts else "1"
        alloc_line = f"{pad}double {s.buf}[{size}];\n"
        body = emit_stmt(s.body, indent)
        return alloc_line + body
    else:
        raise ValueError(f"Unknown CStmt type: {type(s)}")


def emit_function(name: str, stmt: CStmt,
                  input_bufs: list[tuple[str, int]],
                  output_bufs: list[tuple[str, int]]) -> str:
    """Emit a complete C function.

    Args:
        name: Function name
        stmt: The loop nest body
        input_bufs: List of (buffer_name, flat_size)
        output_bufs: List of (buffer_name, flat_size)
    """
    params = []
    for buf, size in input_bufs:
        params.append(f"const double* {buf}")
    for buf, size in output_bufs:
        params.append(f"double* {buf}")

    param_str = ", ".join(params)
    body = emit_stmt(stmt, indent=1)

    return f"""#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void {name}({param_str}) {{
{body}}}
"""


def emit_test_harness(func_name: str,
                      input_bufs: list[tuple[str, int]],
                      output_bufs: list[tuple[str, int]],
                      init_code: str = "") -> str:
    """Emit a main() test harness that calls the function and prints output."""
    lines = [f'int main(void) {{']

    for buf, size in input_bufs:
        lines.append(f'    double {buf}[{size}];')
    for buf, size in output_bufs:
        lines.append(f'    double {buf}[{size}];')
        lines.append(f'    memset({buf}, 0, sizeof({buf}));')

    if init_code:
        lines.append(init_code)

    args = [buf for buf, _ in input_bufs] + [buf for buf, _ in output_bufs]
    lines.append(f'    {func_name}({", ".join(args)});')

    # Print first few output values
    for buf, size in output_bufs:
        lines.append(f'    printf("{buf}[0..3] = %f %f %f %f\\n", '
                     f'{buf}[0], {buf}[1], {buf}[2], {buf}[3]);')

    lines.append('    return 0;')
    lines.append('}')
    return '\n'.join(lines) + '\n'
