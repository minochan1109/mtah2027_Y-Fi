"""Small expression language shared by tools/dead_conditions.py and the situation D detector.

    and / or / not, < <= > >= == !=, + - * /, abs(x), isnan(x), numbers, column names, and
    lasting(x, s)  x has been true for the last s seconds        recent(x, s)  x was true within the last s seconds
    ahead(x, s)    x becomes true within the next s seconds
    starts(x, s)   how many times x STARTED (false -> true) within the last s seconds
Nothing else (no other Python) is allowed.
"""
import numpy as np

FPS = 25

def _runs(mask):
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def lasting(x, s):
    x = np.asarray(x, bool)
    k = int(round(s * FPS))
    out = np.zeros(len(x), bool)
    for a, b in _runs(x):
        out[min(b, a + k):b] = True
    return out


def recent(x, s):
    x = np.asarray(x, bool).astype(int)
    k = int(round(s * FPS))
    c = np.r_[0, np.cumsum(x)]
    t = np.arange(len(x))
    return (c[t + 1] - c[np.clip(t - k, 0, None)]) > 0


def ahead(x, s):
    x = np.asarray(x, bool).astype(int)
    k = int(round(s * FPS))
    c = np.r_[0, np.cumsum(x)]
    t = np.arange(len(x))
    return (c[np.clip(t + k + 1, None, len(x))] - c[t]) > 0


def starts(x, s):
    x = np.asarray(x, bool)
    on = (x & ~np.r_[False, x[:-1]]).astype(int)
    k = int(round(s * FPS))
    c = np.r_[0, np.cumsum(on)]
    t = np.arange(len(x))
    return (c[t + 1] - c[np.clip(t + 1 - k, 0, None)]).astype(float)


def evaluate_expr(expr, cols):
    """Evaluate one expression on one game's columns (dict of numpy arrays). Only the small language
    described at the top is allowed (no other Python)."""
    import ast
    import operator as op
    funcs = {"lasting": lasting, "recent": recent, "ahead": ahead, "starts": starts, "abs": np.abs, "isnan": np.isnan}
    cmp = {ast.Lt: op.lt, ast.LtE: op.le, ast.Gt: op.gt, ast.GtE: op.ge, ast.Eq: op.eq, ast.NotEq: op.ne}
    arith = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.BoolOp):
            vals = [np.asarray(ev(v), bool) for v in n.values]
            return np.logical_and.reduce(vals) if isinstance(n.op, ast.And) else np.logical_or.reduce(vals)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return ~np.asarray(ev(n.operand), bool)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.Compare):
            left, res = ev(n.left), None
            for o, c in zip(n.ops, n.comparators):
                right = ev(c)
                with np.errstate(invalid="ignore"):
                    r = cmp[type(o)](left, right)
                res = r if res is None else (res & r)
                left = right
            return res
        if isinstance(n, ast.BinOp) and type(n.op) in arith:
            return arith[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in funcs:
            return funcs[n.func.id](*[ev(x) for x in n.args])
        if isinstance(n, ast.Name):
            if n.id not in cols:
                raise ValueError(f"unknown column '{n.id}'")
            return cols[n.id]
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        raise ValueError(f"not allowed here: {ast.dump(n)[:60]}")

    v = np.asarray(ev(ast.parse(expr, mode="eval")))
    if v.dtype != bool:
        v = np.nan_to_num(v.astype(float)) != 0
    return v
