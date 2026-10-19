# -*- coding: utf-8 -*-
"""通达信公式 → Python 转译器原型（POC）。

支持语法（指标公式子集）：
    赋值语句:   名称:表达式;    （输出线）
    中间变量:   名称:=表达式;   （不输出）
    表达式:     数字 / 标识符 / 函数调用 / 四则运算 / 比较 / AND OR && ||
内置函数（numpy 序列语义）:
    MA EMA SMA REF HHV LLV CROSS COUNT EVERY EXIST BARSLAST
    SUM ABS MAX MIN

用法:
    env = {"OPEN":..., "HIGH":..., "LOW":..., "CLOSE":..., "VOL":...}
    result = eval_formula(src, env)
    result -> {"DIF": array, "DEA": array, "MACD": array, "__outputs__": [...]}
"""

from __future__ import annotations

import re

import numpy as np

# ==================== 词法分析 ====================

TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)
  | (?P<comment>\{[^}]*\})
  | (?P<str>'[^']*')
  | (?P<num>\d+\.?\d*)
  | (?P<id>[A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]*)
  | (?P<op>:=|<=|>=|<>|!=|&&|\|\||[-+*/<>=(),;:])
""", re.VERBOSE)


def tokenize(src: str):
    tokens, pos = [], 0
    while pos < len(src):
        m = TOKEN_RE.match(src, pos)
        if not m:
            raise SyntaxError(f"无法识别的字符 {src[pos]!r} @ {pos}")
        pos = m.end()
        if m.lastgroup in ("ws", "comment"):
            continue
        tokens.append((m.lastgroup, m.group()))
    tokens.append(("eof", ""))
    return tokens


# ==================== 语法分析（递归下降 → AST 元组） ====================

class Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.i = 0

    def peek(self):
        return self.toks[self.i]

    def eat(self, val=None):
        kind, v = self.toks[self.i]
        if val and v != val:
            raise SyntaxError(f"期望 {val!r}，实际 {v!r}")
        self.i += 1
        return v

    # 语句层
    def parse_program(self):
        stmts = []
        while self.peek()[0] != "eof":
            # 裸调用语句（如 DRAWLINE(...)： 没有赋值名）
            if self.peek()[0] == "id" and self.toks[self.i + 1][1] == "(":
                expr = self.parse_expr()
                self.eat(";")
                stmts.append(("hidden", f"__call{len(stmts)}__", expr))
                continue
            name = self.eat()
            kind = "output" if self.eat() in (":",) else "hidden"   # ':' or ':='
            expr = self.parse_expr()
            self.eat(";")
            stmts.append((kind, name, expr))
        return stmts

    # 表达式层（优先级从低到高）
    def parse_expr(self):
        return self.parse_or()

    def parse_or(self):
        left = self.parse_and()
        while self.peek()[1] in ("OR", "||"):
            self.eat()
            left = ("or", left, self.parse_and())
        return left

    def parse_and(self):
        left = self.parse_cmp()
        while self.peek()[1] in ("AND", "&&"):
            self.eat()
            left = ("and", left, self.parse_cmp())
        return left

    def parse_cmp(self):
        left = self.parse_add()
        while self.peek()[1] in (">", "<", ">=", "<=", "=", "<>", "!="):
            op = self.eat()
            left = ("cmp", op, left, self.parse_add())
        return left

    def parse_add(self):
        left = self.parse_mul()
        while self.peek()[1] in ("+", "-"):
            op = self.eat()
            left = ("bin", op, left, self.parse_mul())
        return left

    def parse_mul(self):
        left = self.parse_unary()
        while self.peek()[1] in ("*", "/"):
            op = self.eat()
            left = ("bin", op, left, self.parse_unary())
        return left

    def parse_unary(self):
        kind, v = self.peek()
        if v in ("+", "-"):
            self.eat()
            return ("neg" if v == "-" else "pos", self.parse_unary())
        return self.parse_atom()

    def parse_atom(self):
        kind, v = self.peek()
        if v == "(":
            self.eat("(")
            e = self.parse_expr()
            self.eat(")")
            return e
        if kind == "str":
            self.eat()
            return ("str", v.strip("'"))
        if kind == "num":
            self.eat()
            return ("const", float(v))
        if kind == "id":
            self.eat()
            if self.peek()[1] == "(":
                self.eat("(")
                args = []
                if self.peek()[1] != ")":
                    args.append(self.parse_expr())
                    while self.peek()[1] == ",":
                        self.eat(",")
                        args.append(self.parse_expr())
                self.eat(")")
                return ("call", v.upper(), args)
            return ("var", v)
        raise SyntaxError(f"意外的记号 {v!r}")


# ==================== 序列函数库（numpy 实现） ====================

def MA(x, n):
    n = int(n)
    x = _s(x)
    valid = (~np.isnan(x)).astype(float)
    cs = np.cumsum(np.insert(np.nan_to_num(x), 0, 0.0))
    cv = np.cumsum(np.insert(valid, 0, 0.0))
    out = np.full_like(x, np.nan)
    for i in range(n - 1, len(x)):
        c = cv[i + 1] - cv[i + 1 - n]
        if c == n:                              # 窗口内需满 N 个有效值
            out[i] = (cs[i + 1] - cs[i + 1 - n]) / n
    return out


def EMA(x, n):
    a = 2.0 / (n + 1)
    out = np.empty_like(np.asarray(x, float))
    prev = x[0]
    for i, v in enumerate(x):
        prev = a * v + (1 - a) * prev
        out[i] = prev
    return out


def SMA(x, n, m):
    """通达信特有：SMA(X,N,M) = (M*X + (N-M)*SMA') / N。

    种子取首个有效值（与通达信一致，从第一根就出值）。
    """
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)
    start = int(np.argmax(~np.isnan(x))) if np.isnan(x).any() else 0
    prev = x[start]
    for i in range(start, len(x)):
        prev = (m * x[i] + (n - m) * prev) / n
        out[i] = prev
    return out


def REF(x, n):
    n = int(n)
    out = np.full_like(np.asarray(x, float), np.nan)
    out[n:] = x[:-n] if n > 0 else x
    return out


def HHV(x, n):
    n = int(n)
    out = np.full_like(np.asarray(x, float), np.nan)
    for i in range(len(x)):
        lo = 0 if n == 0 else max(0, i - n + 1)     # n=0 累计至今
        out[i] = np.max(x[lo:i + 1])
    return out


def LLV(x, n):
    n = int(n)
    out = np.full_like(np.asarray(x, float), np.nan)
    for i in range(len(x)):
        lo = 0 if n == 0 else max(0, i - n + 1)
        out[i] = np.min(x[lo:i + 1])
    return out


def CROSS(a, b):
    return (a > b) & (np.roll(a, 1) <= np.roll(b, 1))


def COUNT(cond, n):
    n = int(n)
    out = np.zeros(len(cond), float)
    for i in range(n - 1, len(cond)):
        out[i] = np.sum(cond[i - n + 1:i + 1])
    return out


# ==================== 常用函数扩充（v0.2.0） ====================

def _s(x):
    """标量/列表统一转 float 数组。"""
    return np.asarray(x, float)


def SUM(x, n):
    x = _s(x)
    n = int(n)
    c = np.cumsum(np.insert(x, 0, 0.0))
    if n == 0:                                  # 0 = 累计至今
        return c[1:]
    out = (c[n:] - c[:-n]) / 1.0
    head = np.cumsum(x[:n - 1])                 # 前端增长窗口
    return np.concatenate([head, out])


def STD(x, n):
    x = _s(x)
    n = int(n)
    out = np.full_like(x, np.nan)
    for i in range(1, len(x)):
        lo = max(0, i - n + 1)
        w = x[lo:i + 1]
        if len(w) >= 2:
            out[i] = np.std(w, ddof=1)
    return out


def VAR(x, n):
    return STD(x, n) ** 2


def DMA(x, a):
    """DMA(X,A)：A 可为 0-1 标量或序列。"""
    x, a = _s(x), _s(a)
    if a.ndim == 0:
        a = np.full(len(x), float(a))
    out = np.empty_like(x)
    prev = x[0]
    for i in range(len(x)):
        prev = a[i] * x[i] + (1 - a[i]) * prev
        out[i] = prev
    return out


def WMA(x, n):
    x, n = _s(x), int(n)
    w = np.arange(1, n + 1, dtype=float)
    out = np.full_like(x, np.nan)
    for i in range(n - 1, len(x)):
        out[i] = np.dot(x[i - n + 1:i + 1], w) / w.sum()
    return out


def BACKSET(cond, n):
    n = int(n)
    out = np.array(cond, bool).copy()
    idx = np.where(out)[0]
    for i in idx:
        out[max(0, i - n + 1):i + 1] = True
    return out


def FILTER(cond, n):
    n = int(n)
    out = np.array(cond, bool).copy()
    cool = -1
    for i, c in enumerate(out):
        if cool >= i:
            out[i] = False
        elif c:
            cool = i + n
    return out


def BARSCOUNT(x):
    x = _s(x)
    valid = np.where(~np.isnan(x))[0]
    start = int(valid[0]) if len(valid) else 0
    return (np.arange(len(x), dtype=float) - start + 1.0)


def BARSSINCE(cond):
    cond = np.asarray(cond, bool)
    out = np.full(len(cond), np.nan)
    first = np.where(cond)[0]
    if len(first):
        out[:] = np.arange(len(cond)) - first[0]
    return out


def UPNDAY(x, n):
    x = _s(x)
    return EVERY(np.concatenate([[False], x[1:] > x[:-1]]), n)


def DOWNNDAY(x, n):
    x = _s(x)
    return EVERY(np.concatenate([[False], x[1:] < x[:-1]]), n)


def NDAY(a, b, n):
    return EVERY(_s(a) > _s(b), n)


def LONGCROSS(a, b, n):
    """LONGCROSS(A,B,N)：A<B 持续 N 根后上穿 B。"""
    a, b = _s(a), _s(b)
    below = EVERY(a < b, n)
    up = (a > b) & (np.roll(a, 1) <= np.roll(b, 1))
    return below & up


def IF(cond, a, b):
    return np.where(np.asarray(cond, bool), _s(a), _s(b))


def POW(x, n):
    return np.power(_s(x), float(n))


def SQRT(x):
    return np.sqrt(_s(x))


def EXP(x):
    return np.exp(_s(x))


def LN(x):
    return np.log(_s(x))


def LOG(x):
    return np.log10(_s(x))


def MOD(a, b):
    return _s(a) % _s(b)


def EVERY(cond, n):
    return COUNT(cond, n) == n


def EXIST(cond, n):
    return COUNT(cond, n) > 0


def BARSLAST(cond):
    out = np.full(len(cond), np.nan)
    last = np.nan
    for i, c in enumerate(cond):
        if c:
            last = 0
        elif not np.isnan(last):
            last += 1
        out[i] = last
    return out


# ==================== 常用函数扩充（v0.5.0） ====================

def AVEDEV(x, n):
    """AVEDEV(X,N)：平均绝对偏差（CCI 的分母）。前端增长窗口。"""
    x = _s(x)
    n = int(n)
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        lo = 0 if n == 0 else max(0, i - n + 1)
        w = x[lo:i + 1]
        out[i] = np.mean(np.abs(w - w.mean()))
    return out


def DEVSQ(x, n):
    """DEVSQ(X,N)：数据偏差平方和。"""
    x = _s(x)
    n = int(n)
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        lo = 0 if n == 0 else max(0, i - n + 1)
        w = x[lo:i + 1]
        out[i] = np.sum((w - w.mean()) ** 2)
    return out


def SLOPE(x, n):
    """SLOPE(X,N)：线性回归斜率。"""
    x, n = _s(x), int(n)
    out = np.full_like(x, np.nan)
    for i in range(n - 1, len(x)):
        w = x[i - n + 1:i + 1]
        out[i] = np.polyfit(np.arange(n), w, 1)[0]
    return out


def FORCAST(x, n):
    """FORCAST(X,N)：线性回归预测值（回归线在当前 bar 的取值）。"""
    x, n = _s(x), int(n)
    out = np.full_like(x, np.nan)
    for i in range(n - 1, len(x)):
        w = x[i - n + 1:i + 1]
        out[i] = np.polyfit(np.arange(n), w, 1)[1] + np.polyfit(np.arange(n), w, 1)[0] * (n - 1)
    return out


def HHVBARS(x, n):
    """距最近 N 周期最高值的周期数（0 = 当前就是最高）。"""
    x, n = _s(x), int(n)
    out = np.zeros(len(x), float)
    for i in range(len(x)):
        lo = 0 if n == 0 else max(0, i - n + 1)
        out[i] = i - (lo + int(np.argmax(x[lo:i + 1])))
    return out


def LLVBARS(x, n):
    x, n = _s(x), int(n)
    out = np.zeros(len(x), float)
    for i in range(len(x)):
        lo = 0 if n == 0 else max(0, i - n + 1)
        out[i] = i - (lo + int(np.argmin(x[lo:i + 1])))
    return out


def CONST(x):
    return np.full(max(1, len(np.atleast_1d(x))), float(np.asarray(x).ravel()[0]))


def ISLASTBAR(x=None):
    n = _env_len()
    out = np.zeros(max(n, 1), bool)
    if n:
        out[-1] = True
    return out


def TOTALBARSCOUNT(x=None):
    return float(_env_len())


def CURRBARSCOUNT(x=None):
    """到最后一根的倒序序号（含当前=1）。"""
    n = _env_len()
    return np.arange(n, 0, -1, dtype=float)


def REFX(x, n):
    """向前引用（引用未来第 n 根，含未来数据，仅作图用）。"""
    x, n = _s(x), int(n)
    out = np.full_like(x, np.nan)
    if n >= 0:
        out[:-n or None] = x[n:]
    return out


def _env_len():
    return int(_CURRENT_ENV.get("__len__") or 0)


def _date_part(freq):
    dates = _CURRENT_ENV.get("__dates__")
    if not dates:
        raise ValueError("日期函数需要 env 提供 __dates__（date/datetime 列表）")
    out = []
    for d in dates:
        if freq == "year":
            out.append(d.year)
        elif freq == "month":
            out.append(d.month)
        elif freq == "day":
            out.append(d.day)
        elif freq == "weekday":
            out.append(d.isoweekday())
        elif freq == "date":
            out.append(d.year * 10000 + d.month * 100 + d.day)
        elif freq == "time":
            out.append(d.hour * 60 + d.minute)
    return np.asarray(out, float)


FUNCS = {
    # 均线族
    "MA": MA, "EMA": EMA, "SMA": SMA, "DMA": DMA, "WMA": WMA,
    # 引用族
    "REF": REF, "REFX": REFX, "HHV": HHV, "LLV": LLV, "BARSLAST": BARSLAST,
    "BARSCOUNT": BARSCOUNT, "BARSSINCE": BARSSINCE,
    "HHVBARS": HHVBARS, "LLVBARS": LLVBARS,
    # 逻辑/统计族
    "CROSS": CROSS, "COUNT": COUNT, "EVERY": EVERY, "EXIST": EXIST,
    "LONGCROSS": LONGCROSS, "UPNDAY": UPNDAY, "DOWNNDAY": DOWNNDAY,
    "NDAY": NDAY, "FILTER": FILTER, "BACKSET": BACKSET,
    # 回归/统计
    "SLOPE": SLOPE, "FORCAST": FORCAST, "AVEDEV": AVEDEV, "DEVSQ": DEVSQ,
    "STD": STD, "VAR": VAR,
    # 位置/常量
    "CONST": CONST, "ISLASTBAR": ISLASTBAR,
    "TOTALBARSCOUNT": TOTALBARSCOUNT, "CURRBARSCOUNT": CURRBARSCOUNT,
    # 日期（需 env 提供 __dates__）
    "YEAR": lambda x=None: _date_part("year"),
    "MONTH": lambda x=None: _date_part("month"),
    "DAY": lambda x=None: _date_part("day"),
    "WEEKDAY": lambda x=None: _date_part("weekday"),
    "DATE": lambda x=None: _date_part("date"),
    "TIME": lambda x=None: _date_part("time"),
    # 数学族
    "SUM": SUM, "ABS": lambda x: np.abs(_s(x)),
    "MAX": lambda a, b: np.maximum(_s(a), _s(b)),
    "MIN": lambda a, b: np.minimum(_s(a), _s(b)),
    "IF": IF, "POW": POW, "SQRT": SQRT, "EXP": EXP,
    "LN": LN, "LOG": LOG, "MOD": MOD,
    "SIGN": lambda x: np.sign(_s(x)),
}

# 供 _eval / drawings 使用：当前求值环境（日期函数依赖）
_CURRENT_ENV: dict = {}


def _arity(fn) -> int:
    import inspect
    try:
        sig = inspect.signature(fn)
        required = [p for p in sig.parameters.values()
                    if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        return len(required)
    except (TypeError, ValueError):
        return 1


# ==================== 求值 ====================

def _eval(node, env):
    op = node[0]
    if op == "const":
        return float(node[1])
    if op == "var":
        if node[1] not in env:
            fn = FUNCS.get(node[1])
            if fn is not None and _arity(fn) == 0:   # 裸用的零参函数（如 MONTH）
                return fn()
            raise NameError(f"未定义变量: {node[1]}")
        return env[node[1]]
    if op == "neg":
        return -_eval(node[1], env)
    if op == "pos":
        return _eval(node[1], env)
    if op == "bin":
        a, b = _eval(node[2], env), _eval(node[3], env)
        return {"+": lambda: a + b, "-": lambda: a - b,
                "*": lambda: a * b, "/": lambda: a / b}[node[1]]()
    if op == "cmp":
        a, b = _eval(node[2], env), _eval(node[3], env)
        return {">": a > b, "<": a < b, ">=": a >= b, "<=": a <= b,
                "=": a == b, "==": a == b, "<>": a != b, "!=": a != b}[node[1]]
    if op == "and":
        return np.logical_and(_eval(node[1], env), _eval(node[2], env))
    if op == "or":
        return np.logical_or(_eval(node[1], env), _eval(node[2], env))
    if op == "str":
        return node[1]
    if op == "call":
        # 绘图指令钩子：调用方通过 __draw_hook__ 收集坐标（见 drawings.py）
        if node[1] in env.get("__draw_names__", ()) and "__draw_hook__" in env:
            args = [_eval(a, env) for a in node[2]]
            env["__draw_hook__"](node[1], args)
            return np.zeros(env.get("__len__") or 1)
        fn = FUNCS.get(node[1])
        if fn is None:
            raise ValueError(f"函数未实现: {node[1]}")
        args = [_eval(a, env) for a in node[2]]
        return fn(*args)
    raise ValueError(op)


def eval_formula(src: str, env: dict) -> dict:
    """解析并求值通达信指标公式，返回 {线名: numpy 数组}。

    env 需提供 OPEN/HIGH/LOW/CLOSE/VOL 等序列；__len__ 自动推断。
    可选 __dates__（date/datetime 列表），YEAR/MONTH/DAY 等日期函数依赖。
    """
    env = dict(env)
    env.setdefault("__len__", len(next(iter(env.values()))))
    global _CURRENT_ENV
    _CURRENT_ENV = env                      # 日期函数等依赖当前环境
    outputs, results = [], {}
    for kind, name, expr in Parser(tokenize(src)).parse_program():
        results[name] = _eval(expr, env)
        env[name] = results[name]          # 后续语句可引用前面的线
        if kind == "output":
            outputs.append(name)
    results["__outputs__"] = outputs
    return results


# ==================== 代码生成（可选：导出 Python 源码） ====================

def gen_python(src: str) -> str:
    lines = []
    for kind, name, expr in Parser(tokenize(src)).parse_program():
        lines.append(f"{name} = {_gen(expr)}")
    return "\n".join(lines)


def _gen(node):
    op = node[0]
    if op == "const":
        return repr(node[1])
    if op == "var":
        return node[1]
    if op == "neg":
        return f"(-{_gen(node[1])})"
    if op == "bin":
        return f"({_gen(node[2])} {node[1]} {_gen(node[3])})"
    if op == "cmp":
        sym = {"<>": "!=", "=": "=="}.get(node[1], node[1])
        return f"({_gen(node[2])} {sym} {_gen(node[3])})"
    if op == "and":
        return f"({_gen(node[1])} & {_gen(node[2])})"
    if op == "or":
        return f"({_gen(node[1])} | {_gen(node[2])})"
    if op == "call":
        return f"{node[1]}({', '.join(_gen(a) for a in node[2])})"
    raise ValueError(op)
