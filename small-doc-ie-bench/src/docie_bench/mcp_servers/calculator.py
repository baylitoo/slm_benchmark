"""Calculator MCP server: exact arithmetic via a whitelisted AST evaluator
(never ``eval``). Run: ``python -m docie_bench.mcp_servers.calculator``.
"""

from __future__ import annotations

import ast
import math
from typing import Any

_MAX_EXPRESSION_LENGTH = 500
_MAX_POW_EXPONENT = 1000

_ALLOWED_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
}

_BINARY_OPERATORS: dict[type[ast.operator], Any] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
}


def _evaluate(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"only numbers are allowed, got {node.value!r}")
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            base, exponent = _evaluate(node.left), _evaluate(node.right)
            if abs(exponent) > _MAX_POW_EXPONENT:
                raise ValueError(f"exponent too large (max {_MAX_POW_EXPONENT})")
            return base**exponent
        operator = _BINARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise ValueError(f"operator {type(node.op).__name__} is not allowed")
        return operator(_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
            raise ValueError("only abs, round, min, max, sum, sqrt calls are allowed")
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        return _ALLOWED_FUNCTIONS[node.func.id](*[_evaluate(arg) for arg in node.args])
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_evaluate(item) for item in node.elts]
    raise ValueError(f"{type(node).__name__} is not allowed in expressions")


def calculate(expression: str) -> float:
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ValueError(f"expression too long (max {_MAX_EXPRESSION_LENGTH} chars)")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"not a valid expression: {exc.msg}") from exc
    result = _evaluate(tree)
    if isinstance(result, list):
        raise ValueError("expression must produce a number, not a list")
    return float(result)


def check_sum(values: list[float], claimed_total: float, tolerance: float = 0.01) -> dict[str, Any]:
    actual = float(sum(float(v) for v in values))
    difference = actual - float(claimed_total)
    return {
        "computed_total": round(actual, 10),
        "claimed_total": float(claimed_total),
        "difference": round(difference, 10),
        "matches": abs(difference) <= float(tolerance),
    }


def build_server() -> Any:
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("docie-calculator")

    @server.tool()
    def calc(expression: str) -> float:
        """Exactly evaluate an arithmetic expression (numbers, + - * / // % **,
        parentheses, and abs/round/min/max/sum/sqrt). Use this instead of
        computing in your head — e.g. calc("3 * 129.99 + 2 * 45.50")."""
        return calculate(expression)

    @server.tool()
    def sum_check(
        values: list[float], claimed_total: float, tolerance: float = 0.01
    ) -> dict[str, Any]:
        """Sum a list of amounts and compare with a claimed total. Returns the
        computed total, the difference, and whether they match within the
        tolerance — the one-call check for an invoice's line items vs its
        stated total."""
        return check_sum(values, claimed_total, tolerance)

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
