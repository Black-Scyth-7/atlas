"""Atlas custom tool 'math_calculator' — created via voice on 2026-06-30."""

TOOL = {
    "name": 'math_calculator',
    "description": 'Performs basic mathematical calculations such as addition, subtraction, multiplication, and division.',
    "arguments": "{'expression': '<mathematical expression>'}",
}


def _safe_eval(expr: str):
    """Safely evaluate a mathematical expression using ast.

    Supports numbers and operators: +, -, *, /, %, **, unary +/-, and parentheses.
    """
    import ast
    import operator as op

    # supported operators
    operators = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Mod: op.mod,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.UAdd: op.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](left, right)
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](operand)
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    parsed = ast.parse(expr, mode='eval')
    return _eval(parsed)


def run(args):
    try:
        expr = args.get('expression') if isinstance(args, dict) else None
        if not expr:
            return "Error: missing 'expression' argument"
        result = _safe_eval(str(expr))
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
