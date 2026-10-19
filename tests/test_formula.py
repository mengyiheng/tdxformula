import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tdxformula import eval_formula, gen_python  # noqa: E402

# ---------- 合成数据 ----------

N = 120
rng = np.random.default_rng(7)
CLOSE = 100 + np.cumsum(rng.normal(0, 1, N))


def _env():
    return {
        "CLOSE": CLOSE.copy(),
        "OPEN": CLOSE - 1, "HIGH": CLOSE + 2, "LOW": CLOSE - 2,
        "VOL": np.abs(rng.normal(1000, 100, N)), "__len__": N,
    }


# ---------- MACD：与 pandas 独立实现交叉验证 ----------

def test_macd_matches_pandas_ewm():
    r = eval_formula("DIF:EMA(CLOSE,12)-EMA(CLOSE,26);DEA:EMA(DIF,9);", _env())
    close = pd.Series(CLOSE)
    dif_ref = (close.ewm(span=12, adjust=False).mean()
               - close.ewm(span=26, adjust=False).mean()).values
    assert np.nanmax(np.abs(r["DIF"] - dif_ref)) < 1e-8
    assert r["__outputs__"] == ["DIF", "DEA"]


def test_macd_real_data_zero_error():
    """用户在通达信核对过的真实场景：DIF 与 pandas ewm 误差为 0。"""
    p = Path(r"D:\new_tdx64\vipdoc\sh\lday\sh000001.day")
    if not p.exists():
        pytest.skip("本机无通达信日线数据")
    from tdxdata import read_day
    bars = read_day(p)
    close = np.array([b.close for b in bars], float)
    r = eval_formula("DIF:EMA(CLOSE,12)-EMA(CLOSE,26);", {"CLOSE": close, "__len__": len(close)})
    s = pd.Series(close)
    ref = (s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()).values
    assert np.nanmax(np.abs(r["DIF"] - ref)) == 0.0


# ---------- KDJ：用户已在通达信逐根核对 ----------

def test_kdj_values_verified_against_tdx():
    """2026-06-08 上证指数：K=17.98 D=21.68 J=10.59，用户在通达信核对一致。"""
    p = Path(r"D:\new_tdx64\vipdoc\sh\lday\sh000001.day")
    if not p.exists():
        pytest.skip("本机无通达信日线数据")
    from tdxdata import read_day
    bars = read_day(p)
    env = {k.upper(): np.array([getattr(b, k) for b in bars], float)
           for k in ("open", "high", "low", "close", "vol")}
    r = eval_formula("""
RSV:=(CLOSE-LLV(LOW,9))/(HHV(HIGH,9)-LLV(LOW,9))*100;
K:SMA(RSV,3,1);
D:SMA(K,3,1);
J:3*K-2*D;
""", env)
    assert round(r["K"][-1], 2) == 17.98
    assert round(r["D"][-1], 2) == 21.68
    assert round(r["J"][-1], 2) == 10.59
    assert bars[-1].date == 20260608


# ---------- 语义细节 ----------

def test_hhv_llv_growing_window_at_head():
    """通达信 HHV/LLV 前几根用增长窗口（不满 N 根也出值）。"""
    r = eval_formula("H:HHV(CLOSE,5);", {"CLOSE": CLOSE, "__len__": N})
    assert not np.isnan(r["H"][0])
    assert r["H"][0] == CLOSE[0]
    assert r["H"][4] == pytest.approx(max(CLOSE[:5]))


def test_hidden_variable_not_in_outputs():
    r = eval_formula("A:=CLOSE+1;B:CLOSE*2;", _env())
    assert r["__outputs__"] == ["B"]
    assert np.allclose(r["A"], CLOSE + 1)


def test_comparison_and_logic():
    r = eval_formula("SIG:CROSS(CLOSE,REF(CLOSE,1)) AND CLOSE>0;", _env())
    sig = r["SIG"]
    assert sig.dtype == bool
    assert not sig[0]          # REF(1) 首位为 NaN，不触发


def test_gen_python_and_error_cases():
    src = "DIF:EMA(CLOSE,12)-EMA(CLOSE,26);"
    assert "EMA(CLOSE, 12.0)" in gen_python(src)
    with pytest.raises(ValueError):
        eval_formula("X:NOTEXIST(CLOSE,5);", _env())
    with pytest.raises(ValueError):
        eval_formula("X:FOO(CLOSE);", _env())
