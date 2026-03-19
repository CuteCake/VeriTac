"""
VeriTac CodeGen: lower.py
Parse LoopNest JSON from Lean CLI into Python AST.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CExpr:
    """Base class for C expressions."""
    pass

@dataclass
class CLit(CExpr):
    val: int

@dataclass
class CVar(CExpr):
    name: str

@dataclass
class CBinOp(CExpr):
    op: str  # "+", "*", "/", "%"
    left: CExpr
    right: CExpr

@dataclass
class CBufRead(CExpr):
    buf: str
    indices: list[CExpr]


@dataclass
class CStmt:
    """Base class for C statements."""
    pass

@dataclass
class CSkip(CStmt):
    pass

@dataclass
class CBufWrite(CStmt):
    buf: str
    indices: list[CExpr]
    val: CExpr

@dataclass
class CLoop(CStmt):
    var: str
    lo: CExpr
    hi: CExpr
    annotation: str  # "none", "parallel", "vectorize", "unrolled"
    body: CStmt

@dataclass
class CSeq(CStmt):
    stmts: list[CStmt]

@dataclass
class CAlloc(CStmt):
    buf: str
    shape: list[CExpr]
    body: CStmt


def parse_expr(j: dict) -> CExpr:
    """Parse a JSON expression into a CExpr."""
    tag = j["tag"]
    if tag == "lit":
        return CLit(int(j["val"]))
    elif tag == "var":
        return CVar(j["name"])
    elif tag in ("add", "mul", "div", "mod"):
        op_map = {"add": "+", "mul": "*", "div": "/", "mod": "%"}
        return CBinOp(op_map[tag], parse_expr(j["left"]), parse_expr(j["right"]))
    elif tag == "bufRead":
        return CBufRead(j["buf"], [parse_expr(i) for i in j["indices"]])
    else:
        raise ValueError(f"Unknown expr tag: {tag}")


def parse_stmt(j: dict) -> CStmt:
    """Parse a JSON statement into a CStmt."""
    tag = j["tag"]
    if tag == "skip":
        return CSkip()
    elif tag == "bufWrite":
        return CBufWrite(
            j["buf"],
            [parse_expr(i) for i in j["indices"]],
            parse_expr(j["val"])
        )
    elif tag == "loop":
        return CLoop(
            j["var"],
            parse_expr(j["lo"]),
            parse_expr(j["hi"]),
            j.get("ann", "none"),
            parse_stmt(j["body"])
        )
    elif tag == "seq":
        return CSeq([parse_stmt(j["s1"]), parse_stmt(j["s2"])])
    elif tag == "alloc":
        return CAlloc(
            j["buf"],
            [parse_expr(i) for i in j["shape"]],
            parse_stmt(j["body"])
        )
    else:
        raise ValueError(f"Unknown stmt tag: {tag}")
