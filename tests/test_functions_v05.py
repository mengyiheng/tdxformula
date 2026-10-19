# -*- coding: utf-8 -*-
"""v0.5.0 扩充：统计回归、位置、日期函数 + CCI/RSI 经典公式实测。"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tdxformula import eval_formula  # noqa: E402

N = 150
rng = np.random.default_rng(23)
CLOSE = 60 + np.cumsum(rng.normal(0, 1, N))
DATES = [date(2026, 1, 5) + timedelta(days=i) for i in range(N)]


def _env():
    e = {
        "CLOSE": CLOSE.copy(), "OPEN": CLOSE - 0.4,
        "HIGH": CLOSE + 1.2, "LOW": CLOSE - 1.2,
        "VOL": np.abs(rng.normal(800, 80, N)), "__len__": N, "__dates__": DATES,
    }
    return e


# ---------- 统计回归 ----------

def test_avedev_vs_pandas():
    r = eval_formula("A:AVEDEV(CLOSE,14);", _env())
    ref = pd.Series(CLOSE).rolling(14).apply(lambda w: np.abs(w - w.mean()).mean()).values
    assert np.nanmax(np.abs(r["A"][13:] - ref[13:])) < 1e-9


def test_slope_forcast_vs_polyfit():
    r = eval_formula("S:SLOPE(CLOSE,10);F:FORCAST(CLOSE,10);", _env())
    for i in range(9, N):
        w = CLOSE[i - 9:i + 1]
        k, b = np.polyfit(np.arange(10), w, 1)
        assert r["S"][i] == pytest.approx(k, abs=1e-9)
        assert r["F"][i] == pytest.approx(b + k * 9, abs=1e-9)
        if i > 30:
            break


def test_hhvbars_llvbars():
    r = eval_formula("H:HHVBARS(HIGH,10);L:LLVBARS(LOW,10);", _env())
    for i in (20, 80, 140):
        w = CLOSE + 1.2
        lo = max(0, i - 9)
        assert r["H"][i] == i - (lo + int(np.argmax(w[lo:i + 1])))


# ---------- 位置 / 常量 ----------

def test_const_islastbar_currbars():
    r = eval_formula("K:CONST(3.5);L:ISLASTBAR;C:CURRBARSCOUNT;", _env())
    assert np.all(r["K"] == 3.5)
    assert r["L"].dtype == bool and r["L"].sum() == 1 and r["L"][-1]
    assert r["C"][0] == N and r["C"][-1] == 1


# ---------- 日期函数 ----------

def test_date_functions():
    r = eval_formula("Y:YEAR;M:MONTH;D:DAY;W:WEEKDAY;DT:DATE;", _env())
    assert np.all(r["Y"] == 2026)
    assert r["M"][0] == 1
    assert r["D"][0] == 5
    assert r["W"][0] == DATES[0].isoweekday()
    assert r["DT"][0] == 20260105


# ---------- 经典公式：CCI / RSI ----------

def test_cci_vs_manual():
    src = """
TYP:=(HIGH+LOW+CLOSE)/3;
CCI:(TYP-MA(TYP,14))/(0.015*AVEDEV(TYP,14));
"""
    r = eval_formula(src, _env())
    typ = (CLOSE + 1.2 + CLOSE - 1.2 + CLOSE) / 3
    s = pd.Series(typ)
    ma = s.rolling(14).mean()
    avedev = s.rolling(14).apply(lambda w: np.abs(w - w.mean()).mean()).values
    ref = (typ - ma.values) / (0.015 * avedev)
    assert np.nanmax(np.abs(r["CCI"][13:] - ref[13:])) < 1e-9


def test_rsi_matches_wilder():
    src = """
LC:=REF(CLOSE,1);
RSI:SMA(MAX(CLOSE-LC,0),14,1)/SMA(ABS(CLOSE-LC),14,1)*100;
"""
    r = eval_formula(src, _env())
    # Wilder RSI 独立实现（种子与通达信 SMA 一致：首个有效值，即 bar=1）
    d = np.diff(CLOSE, prepend=CLOSE[0])
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    ag = np.zeros(N); al = np.zeros(N)
    ag[1] = gain[1]; al[1] = loss[1]
    for i in range(2, N):
        ag[i] = (13 * ag[i - 1] + gain[i]) / 14
        al[i] = (13 * al[i - 1] + loss[i]) / 14
    ref = np.where(al == 0, 100.0, 100 - 100 / (1 + ag / np.where(al == 0, 1, al)))
    assert np.nanmax(np.abs(r["RSI"][2:] - ref[2:])) < 1e-9
    assert 0 <= np.nanmin(r["RSI"]) and np.nanmax(r["RSI"]) <= 100.01


def test_bare_zero_arg_call():
    """通达信里日期函数不带括号裸用。"""
    r = eval_formula("M:MONTH;", _env())
    assert np.allclose(r["M"], [d.month for d in DATES])

def test_sum_nan_head_regression():
    """v0.5.1 回归：SUM 头部 NaN 曾毒化整条序列（DMI 公式触发）。"""
    from tdxformula import eval_formula
    x = np.arange(20, dtype=float)
    x[0] = np.nan
    env = {"CLOSE": x, "HIGH": x + 1, "LOW": x - 1, "VOL": np.ones(20), "__len__": 20}
    r = eval_formula("S:SUM(CLOSE,14);", env)
    assert (~np.isnan(r["S"])).sum() == 7          # 仅前 13 根无值
    assert r["S"][-1] == pytest.approx(x[6:].sum())   # 末根窗口 = x[6..19]，x[0] 记 0
    r2 = eval_formula("S0:SUM(CLOSE,0);", env)
    assert r2["S0"][-1] == pytest.approx(x[1:].sum())
