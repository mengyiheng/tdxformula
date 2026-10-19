# -*- coding: utf-8 -*-
"""绘图指令 → tdxdraw 画线桥接测试。"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

tdxdraw = pytest.importorskip("tdxdraw")
from tdxformula import eval_to_drawings  # noqa: E402

DATES = [date(2026, 1, 5) + timedelta(days=i) for i in range(30)]
CLOSE = 50 + np.sin(np.arange(30) / 3) * 5
ENV = {"CLOSE": CLOSE, "HIGH": CLOSE + 1, "LOW": CLOSE - 1,
       "OPEN": CLOSE, "VOL": np.ones(30)}


def test_drawline_creates_segment():
    src = "DRAWLINE(LOW=LLV(LOW,5),LOW,HIGH=HHV(HIGH,5),HIGH,0);T:CLOSE;"
    ds = eval_to_drawings(src, ENV, market="sh", code="600519", dates=DATES)
    assert len(ds) >= 1
    d = ds[0]
    assert d.market == "sh" and d.code == "600519"
    assert d.line_type == "segment"
    assert d.start_date in DATES and d.end_date in DATES
    assert DATES.index(d.start_date) < DATES.index(d.end_date)   # 起点在终点之前
    assert d.line_id == 1


def test_drawtext_creates_text():
    src = "DRAWTEXT(CLOSE>REF(CLOSE,1),LOW,'买');T:CLOSE;"
    ds = eval_to_drawings(src, ENV, market="sz", code="000001", dates=DATES)
    assert len(ds) >= 1
    assert all(d.line_type == "text" for d in ds)
    assert all(d.end_price == 0.0 for d in ds)
    assert [d.line_id for d in ds] == list(range(1, len(ds) + 1))


def test_polyline_creates_chain():
    src = "POLYLINE(CLOSE>0,CLOSE);T:CLOSE;"
    ds = eval_to_drawings(src, ENV, market="sh", code="600000", dates=DATES)
    assert len(ds) == 29                        # 30 个点连成 29 段
    assert all(d.line_type == "segment" for d in ds)


def test_skip_and_continued_line_ids(tmp_path):
    src = ("DRAWICON(CLOSE>REF(CLOSE,1),LOW,1);"
           "DRAWLINE(LOW=LLV(LOW,5),LOW,HIGH=HHV(HIGH,5),HIGH,0);T:CLOSE;")
    ds = eval_to_drawings(src, ENV, market="sh", code="600519", dates=DATES)
    assert all(d.line_type == "segment" for d in ds)      # ICON 被跳过


def test_write_eld_integration(tmp_path):
    src = "DRAWLINE(LOW=LLV(LOW,5),LOW,HIGH=HHV(HIGH,5),HIGH,0);T:CLOSE;"
    ds = eval_to_drawings(src, ENV, market="sh", code="600519", dates=DATES)
    out = tmp_path / "out.eld"
    assert tdxdraw.write_eld(out, ds) == len(ds)
    back = tdxdraw.read_eld(out)
    assert len(back) == len(ds)
    assert back[0].start_date == ds[0].start_date


def test_stickline_merges_runs_into_rectangles():
    # 第 3-6 根触发（连续 4 根 → 1 个矩形），第 10 根单独触发（1 个矩形）
    cond = np.zeros(30, bool)
    cond[2:6] = True
    cond[9] = True
    src = "STICKLINE(cond,HIGH,LOW,2,0);T:CLOSE;"
    env = dict(ENV, cond=cond)
    ds = eval_to_drawings(src, env, market="sh", code="600519", dates=DATES)
    rects = [d for d in ds if d.line_type == "rectangle"]
    assert len(rects) == 2
    # 合并段的价格取触发区间内 HIGH/LOW 的极值
    assert rects[0].end_price == pytest.approx(max(ENV["HIGH"][2:6]))
    assert rects[0].start_price == pytest.approx(min(ENV["LOW"][2:6]))
    assert rects[0].start_date == DATES[2] and rects[0].end_date == DATES[5]
    assert rects[1].start_date == DATES[9] and rects[1].end_date == DATES[9]


def test_drawnumber_labels_values():
    src = "DRAWNUMBER(CLOSE>REF(CLOSE,1),LOW,CLOSE);T:CLOSE;"
    ds = eval_to_drawings(src, ENV, market="sh", code="600519", dates=DATES)
    assert len(ds) >= 1
    assert all(d.line_type == "text" for d in ds)
    # 文字应为对应收盘价的数字（存于画线日期锚点上，数值经 eld 往返后校验价格字段）
    prev = np.concatenate([[np.nan], CLOSE[:-1]])
    first_up = int(np.where(CLOSE > prev)[0][0])
    assert ds[0].start_date == DATES[first_up]
    assert ds[0].start_price == pytest.approx(float(ENV["LOW"][first_up]))
