"""Bounded, untrusted expansion of compact schedules to concrete commands.

This adapter is not a proof and does not change the instruction language. Every
expanded command and its encoded bytes still require the existing Lean checker.
No Python source is executed: expressions are interpreted using a small AST
whitelist. The original flat-command interface remains valid.
"""
import ast
import math
import operator


class ScheduleError(ValueError):
    pass


_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
_COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
            ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
_LIMIT = (1 << 64) - 1


def _bounded(value):
    if type(value) not in (int, bool) or abs(value) > _LIMIT:
        raise ScheduleError("expression value outside bounded integer domain")
    return value


def expression(source, bindings):
    if not isinstance(source, str) or len(source) > 256:
        raise ScheduleError("expression must be a string of at most 256 characters")
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, RecursionError) as error:
        raise ScheduleError("invalid expression") from error

    def visit(node, depth=0):
        if depth > 16:
            raise ScheduleError("expression nesting limit exceeded")
        go = lambda n: visit(n, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in (int, bool):
            return _bounded(node.value)
        if isinstance(node, ast.Name) and node.id in bindings:
            return _bounded(bindings[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = go(node.operand)
            if type(value) is not int:
                raise ScheduleError("arithmetic requires integers, not booleans")
            return _bounded(value if isinstance(node.op, ast.UAdd) else -value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = go(node.left), go(node.right)
            if type(left) is not int or type(right) is not int:
                raise ScheduleError("arithmetic requires integers, not booleans")
            try:
                return _bounded(_BINARY[type(node.op)](left, right))
            except ZeroDivisionError as error:
                raise ScheduleError("division by zero") from error
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE:
            left, right = go(node.left), go(node.comparators[0])
            if type(left) is not int or type(right) is not int:
                raise ScheduleError("comparisons require integers")
            return _COMPARE[type(node.ops[0])](left, right)
        if isinstance(node, ast.IfExp):
            condition = go(node.test)
            if type(condition) is not bool:
                raise ScheduleError("conditional requires a Boolean test")
            return go(node.body if condition else node.orelse)
        raise ScheduleError("unsupported expression or unbound variable")

    return visit(tree.body)


def expand_program(program, constants, *, max_commands=4096, max_iterations=100000,
                   max_depth=12):
    """Expand loops and explicit {expr: ...} fields; enforce resource budgets.

    Loop syntax: {"for": "i", "start": 0, "stop": {"expr": "M // 16"},
                  "step": 1, "body": [ ... ]}.
    Constants and loop bounds are integers. Loops cannot redefine constants.
    Ordinary commands are dictionaries interpreted later by the strict parser.
    """
    for value in (max_commands, max_iterations, max_depth):
        if type(value) is not int or value <= 0:
            raise ScheduleError("expansion limits must be positive integers")
    if not isinstance(constants, dict) or any(
        not isinstance(k, str) or not k.isidentifier() or type(v) is not int
        or abs(v) > _LIMIT for k, v in constants.items()
    ):
        raise ScheduleError("constants must be bounded named integers")
    result = []
    iterations = 0

    def resolve(value, env):
        if isinstance(value, dict):
            if set(value) != {"expr"}:
                raise ScheduleError("field object must contain only expr")
            return expression(value["expr"], env)
        if type(value) in (int, bool):
            return _bounded(value)
        if type(value) is float and math.isfinite(value) and abs(value) <= _LIMIT:
            return value  # e.g. identity load scale; command parser checks scope
        if isinstance(value, str):
            return value
        raise ScheduleError("unsupported command field value")

    def walk(nodes, env, depth):
        nonlocal iterations
        if depth > max_depth or not isinstance(nodes, list):
            raise ScheduleError("invalid program or nesting limit exceeded")
        if len(nodes) > max_commands:
            raise ScheduleError("source node limit exceeded")
        for node in nodes:
            if not isinstance(node, dict):
                raise ScheduleError("program entries must be objects")
            if "for" in node:
                if set(node) != {"for", "start", "stop", "step", "body"}:
                    raise ScheduleError("loop fields must be for/start/stop/step/body")
                var = node["for"]
                if not isinstance(var, str) or not var.isidentifier() or var in env:
                    raise ScheduleError("loop variable invalid or shadows an existing binding")
                start, stop, step = [resolve(node[key], env) for key in ("start", "stop", "step")]
                if any(type(v) is not int for v in (start, stop, step)) or step == 0:
                    raise ScheduleError("loop bounds require integers and nonzero step")
                count = max(0, (stop - start + step - 1) // step) if step > 0 else max(
                    0, (start - stop - step - 1) // (-step))
                if count > max_iterations - iterations:
                    raise ScheduleError("iteration budget exceeded")
                if not isinstance(node["body"], list):
                    raise ScheduleError("loop body must be a list")
                for value in range(start, stop, step):
                    iterations += 1
                    if iterations > max_iterations:
                        raise ScheduleError("iteration budget exceeded")
                    walk(node["body"], {**env, var: value}, depth + 1)
            else:
                if len(result) >= max_commands:
                    raise ScheduleError("expanded command budget exceeded")
                if "kind" not in node or not isinstance(node["kind"], str):
                    raise ScheduleError("command needs a literal kind")
                result.append({key: resolve(value, env) for key, value in node.items()})

    walk(program, dict(constants), 0)
    return result
