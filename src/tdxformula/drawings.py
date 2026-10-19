# -*- coding: utf-8 -*-
"""tdxformula.drawings — 把公式里的绘图指令转成通达信画线（tdxdraw.Drawing）。

打通"公式 → 画线"两件套：分析程序输出的公式（含 DRAWLINE/DRAWTEXT/POLYLINE）
经本模块转译后直接生成 .eld，可在通达信里导入成真正的画线。

    from tdxformula import eval_to_drawings

    drawings = eval_to_drawings(
        "DRAWLINE(LOW=LLV(LOW,20),LOW,HIGH=HHV(HIGH,20),HIGH,0);",
        {"CLOSE": close, "HIGH": high, "LOW": low},
        market="sh", code="600519", dates=dates)
    # drawings 为 tdxdraw.Drawing 列表，write_eld("out.eld", drawings) 即可导入
"""

from __future__ import annotations

import numpy as np

from .formula import eval_formula

# 支持的绘图指令
DRAW_NAMES = {"DRAWLINE", "DRAWTEXT", "POLYLINE", "DRAWICON", "DRAWNUMBER"}

_SKIP_NOTIFY = {"DRAWICON", "DRAWNUMBER", "STICKLINE"}


def _pairs(cond1, price1, cond2, price2):
    """DRAWLINE 语义：cond1 定起点，其后首个 cond2 定终点，循环配对。"""
    c1 = np.asarray(cond1, bool)
    c2 = np.asarray(cond2, bool)
    i, n = 0, len(c1)
    while i < n:
        if c1[i] and not np.isnan(price1[i]):
            j = i + 1
            while j < n and not (c2[j] and not np.isnan(price2[j])):
                j += 1
            if j < n:
                yield i, j
                i = j
                continue
            break
        i += 1


def _draw_hook(collector, dates):
    def hook(name, args):
        def arr(x):
            x = np.asarray(x, float)
            return x if x.ndim else np.full(len(dates), x)

        def text(x):
            return x if isinstance(x, str) else None

        if name == "DRAWLINE" and len(args) >= 4:
            c1, p1, c2, p2 = (arr(a) for a in args[:4])
            for i, j in _pairs(c1, p1, c2, p2):
                collector.append({"kind": "segment", "i1": i, "p1": float(p1[i]),
                                  "i2": j, "p2": float(p2[j])})
        elif name in ("DRAWTEXT", "DRAWICON", "DRAWNUMBER"):
            if name == "DRAWICON":
                return                      # 图标依赖客户端资源，跳过
            c, p = arr(args[0]), arr(args[1])
            label = text(args[2]) if len(args) > 2 and isinstance(args[2], str) else \
                (name.replace("DRAW", "") + str(int(args[2])) if len(args) > 2 else name)
            for i in np.where(c)[0]:
                if not np.isnan(p[i]):
                    collector.append({"kind": "text", "i": int(i),
                                      "p": float(p[i]), "text": str(label)})
        elif name == "POLYLINE":
            c, p = arr(args[0]), arr(args[1])
            idx = [int(i) for i in np.where(c)[0] if not np.isnan(p[i])]
            for a, b in zip(idx[:-1], idx[1:]):
                collector.append({"kind": "segment", "i1": a, "p1": float(p[a]),
                                  "i2": b, "p2": float(p[b])})
        elif name in _SKIP_NOTIFY:
            collector.append({"kind": "skipped", "fn": name})
    return hook


def eval_to_drawings(src: str, env: dict, *, market: str, code: str,
                     dates, period: str = "day",
                     line_id_start: int = 1, **drawing_kw):
    """求值公式并把绘图指令转成 tdxdraw.Drawing 列表。

    dates: 与序列等长的 date/datetime 可迭代对象（画线坐标的日期锚点）。
    其余关键字参数（line_width、color_bgr 等）透传给 Drawing。
    需要 `pip install tdxformula[draw]`（即 tdxdraw）。
    """
    try:
        from tdxdraw import Drawing
    except ImportError as e:
        raise ImportError("绘图导出需要 tdxdraw：pip install tdxformula[draw]") from e

    dates = list(dates)
    env = dict(env)
    env["__len__"] = len(dates)
    collector: list = []
    env["__draw_names__"] = DRAW_NAMES
    env["__draw_hook__"] = _draw_hook(collector, dates)

    eval_formula(src, env)

    drawings, lid = [], line_id_start
    for c in collector:
        if c["kind"] == "skipped" or not (0 <= c.get("i1", 0) < len(dates)):
            continue
        if c["kind"] == "segment":
            d = Drawing(market, code, "segment",
                        round(c["p1"], 4), round(c["p2"], 4),
                        dates[c["i1"]], dates[c["i2"]], line_id=lid, **drawing_kw)
        else:  # text
            d = Drawing(market, code, "text",
                        round(c["p"], 4), 0.0,
                        dates[c["i"]], dates[c["i"]], line_id=lid, **drawing_kw)
        drawings.append(d)
        lid += 1
    return drawings
