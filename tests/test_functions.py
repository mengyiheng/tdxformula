# -*- coding: utf-8 -*-
"""v0.2.0 扩充函数与经典公式实测。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tdxformula import eval_formula  # noqa: E402

N = 200
rng = np.random.default_rng(11)
CLOSE = 50 + np.cumsum(rng.normal(0, 0.8, N))


def _env():
    return {
        "CLOSE": CLOSE.copy(), "OPEN": CLOSE - 0.5,
        "HIGH": CLOSE + 1.5, "LOW": CLOSE - 1.5,
        "VOL": np.abs(rng.normal(1000, 100, N)), "__len__": N,
    }


# ---------- 新函数性质 ----------

def test_sum_rolling_and_cumulative():
    r = eval_formula("S3:SUM(CLOSE,3);S0:SUM(CLOSE,0);", _env())
    assert r["S3"][2] == pytest.approx(CLOSE[:3].sum())
    assert r["S0"][-1] == pytest.approx(CLOSE.sum())


def test_std_var_pandas():
    r = eval_formula("SD:STD(CLOSE,20);", _env())
    ref = pd.Series(CLOSE).rolling(20).std(ddof=1).values
    assert np.nanmax(np.abs(r["SD"][19:] - ref[19:])) < 1e-9


def test_dma_wma():
    r = eval_formula("M:DMA(CLOSE,0.5);W:WMA(CLOSE,5);", _env())
    # DMA 0.5 应等价 pandas ewm(alpha=0.5)
    ref = pd.Series(CLOSE).ewm(alpha=0.5, adjust=False).mean().values
    assert np.nanmax(np.abs(r["M"] - ref)) < 1e-9
    assert r["W"][4] == pytest.approx(np.dot(CLOSE[:5], np.arange(1, 6)) / 15.0)


def test_filter_backset():
    r = eval_formula("F:FILTER(CLOSE>REF(CLOSE,1),3);B:BACKSET(CLOSE>REF(CLOSE,1),2);", _env())
    f = r["F"]
    fired = np.where(f)[0]
    for i, j in zip(fired[:-1], fired[1:]):
        assert j - i > 3                      # 触发后 3 根内不再触发
    assert r["B"].dtype == bool


def test_bars_count_since():
    r = eval_formula("C1:BARSCOUNT(CLOSE);Z:BARSSINCE(CLOSE>=LOW);NN:BARSSINCE(CLOSE<0);", _env())
    assert r["C1"][0] == 1 and r["C1"][-1] == N
    # 恒真（第一根就成立）→ 距第一次成立的周期数 = 0,1,2,...
    assert np.allclose(r["Z"], np.arange(N))
    assert np.all(np.isnan(r["NN"]))    # 恒假：全 NaN


def test_if_pow_longcross():
    r = eval_formula("I:IF(CLOSE>50,1,0);P:POW(CLOSE,2);LC:LONGCROSS(CLOSE,REF(CLOSE,1),5);", _env())
    assert set(np.unique(r["I"])) <= {0.0, 1.0}
    assert np.allclose(r["P"], CLOSE ** 2)
    assert r["LC"].dtype == bool


# ---------- 经典公式实测 ----------

def test_boll_matches_pandas():
    src = """
MID:MA(CLOSE,20);
UPPER:MID+2*STD(CLOSE,20);
LOWER:MID-2*STD(CLOSE,20);
"""
    r = eval_formula(src, _env())
    s = pd.Series(CLOSE)
    mid = s.rolling(20).mean()
    upper = mid + 2 * s.rolling(20).std(ddof=1)
    assert np.nanmax(np.abs(r["MID"][19:] - mid.values[19:])) < 1e-9
    assert np.nanmax(np.abs(r["UPPER"][19:] - upper.values[19:])) < 1e-9
    # 价格应绝大多数时间处于带内
    inside = (CLOSE <= r["UPPER"]) & (CLOSE >= r["LOWER"])
    assert np.nanmean(inside[19:]) > 0.85


def test_wr_range_and_formula():
    src = """
N:=14;
WR1:100*(HHV(HIGH,N)-CLOSE)/(HHV(HIGH,N)-LLV(LOW,N));
"""
    r = eval_formula(src, _env())
    wr = r["WR1"][13:]
    assert np.nanmin(wr) >= -0.01 and np.nanmax(wr) <= 100.01


def test_bias_formula():
    src = "BIAS:(CLOSE-MA(CLOSE,12))/MA(CLOSE,12)*100;"
    r = eval_formula(src, _env())
    ref = (CLOSE / pd.Series(CLOSE).rolling(12).mean().values - 1) * 100
    assert np.nanmax(np.abs(r["BIAS"][11:] - ref[11:])) < 1e-9


def test_mtm_and_cross_signal():
    src = """
MTM:CLOSE-REF(CLOSE,12);
MTMMA:MA(MTM,6);
BUY:CROSS(MTM,MTMMA);
"""
    r = eval_formula(src, _env())
    ref = CLOSE - np.concatenate([np.full(12, np.nan), CLOSE[:-12]])
    assert np.nanmax(np.abs(r["MTM"][12:] - ref[12:])) < 1e-9
    assert r["BUY"].dtype == bool and r["BUY"].sum() > 0
