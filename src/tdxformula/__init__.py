"""tdxformula — 通达信指标公式 Python 转译器。

把通达信技术指标公式（DSL）解析并求值为 numpy 序列，语义经通达信
真实数据逐根核对（MACD/KDJ 已验证）。

    from tdxformula import eval_formula, gen_python

    src = "DIF:EMA(CLOSE,12)-EMA(CLOSE,26);DEA:EMA(DIF,9);MACD:(DIF-DEA)*2;"
    r = eval_formula(src, {"CLOSE": close_array})
    print(r["DIF"], r["__outputs__"])

    print(gen_python(src))   # -> "DIF = (EMA(CLOSE, 12.0) - EMA(CLOSE, 26.0))..."
"""

from .formula import (FUNCS, eval_formula, gen_python, tokenize)

__version__ = "0.1.0"
__all__ = ["eval_formula", "gen_python", "tokenize", "FUNCS", "__version__"]
