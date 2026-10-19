<p align="center"><img src="assets/cover.png" alt="tdxformula" width="720"/></p>

# tdxformula — 通达信公式 Python 转译器

把**通达信技术指标公式**（DSL）解析并转译为 numpy 序列运算，语义经通达信
真实数据**逐根核对**（MACD 与 pandas 独立实现误差为 0；KDJ 由作者在通达信
客户端逐值核对一致）。

- 📖 **解析指标公式 DSL**：赋值线 / 中间变量（`:=`）/ 四则 / 比较 / `AND OR` / 注释
- 🧮 **序列语义对齐通达信**：`HHV/LLV` 前端增长窗口、`SMA(X,N,M)` 加权递推种子等
  特殊行为全部按通达信真实行为实现（市面开源实现的重灾区）
- 🐍 **双模式**：直接求值返回 numpy 数组，或生成 Python 源码
- ✅ **可信度**：KDJ 在上证指数 2026-06-08 实测 K=17.98 / D=21.68 / J=10.59，
  与通达信客户端显示完全一致（测试内置该断言）

## 安装

```bash
pip install tdxformula numpy
```

## 快速开始

```python
import numpy as np
from tdxdata import read_day
from tdxformula import eval_formula, gen_python

bars = read_day(r"D:\new_tdx64\vipdoc\sh\lday\sh000001.day")
env = {k.upper(): np.array([getattr(b, k) for b in bars], float)
       for k in ("open", "high", "low", "close", "vol")}

macd = """
DIF:EMA(CLOSE,12)-EMA(CLOSE,26);
DEA:EMA(DIF,9);
MACD:(DIF-DEA)*2;
"""
r = eval_formula(macd, env)
print(r["DIF"][-1], r["DEA"][-1])     # numpy 序列，直接进你的回测框架
print(r["__outputs__"])               # ['DIF', 'DEA', 'MACD']

kdj = """
RSV:=(CLOSE-LLV(LOW,9))/(HHV(HIGH,9)-LLV(LOW,9))*100;
K:SMA(RSV,3,1);
D:SMA(K,3,1);
J:3*K-2*D;
"""
r2 = eval_formula(kdj, env)

print(gen_python(macd))
# DIF = (EMA(CLOSE, 12.0) - EMA(CLOSE, 26.0))
# DEA = EMA(DIF, 9.0)
# MACD = ((DIF - DEA) * 2.0)
```

## 已支持的函数

| 类别 | 函数 |
|---|---|
| 均线 | `MA` `EMA` `SMA`（通达信加权版）`DMA` `WMA` |
| 引用 | `REF` `HHV` `LLV`（支持 N=0 累计）`BARSLAST` `BARSCOUNT` `BARSSINCE` |
| 统计 | `COUNT` `EVERY` `EXIST` `SUM`（支持 N=0）`STD` `VAR` |
| 信号 | `CROSS` `LONGCROSS` `UPNDAY` `DOWNNDAY` `NDAY` `FILTER` `BACKSET` |
| 逻辑 | `IF` `AND/OR/&&/\|\|`、比较运算 |
| 算术 | `+ - * /` `ABS` `MAX` `MIN` `POW` `SQRT` `EXP` `LN` `LOG` `MOD` |

已在路线图：`DRAW*` 系列转 [tdxdraw](https://gitee.com/mengyiheng/tdxdraw) 画线、
公式参数、跨公式引用（`"MACD.DEA"#DAY`）、更多函数。

## 通达信语义对齐说明（本项目存在的理由）

| 行为 | 通达信真实语义 | 本库实现 |
|---|---|---|
| `HHV/LLV` 前 N-1 根 | 用现有数据增长窗口，照常出值 | ✅ |
| `SMA(X,N,M)` 种子 | 首个有效值，第一根即出值 | ✅ |
| `MA` 前 N-1 根 | 无值 | ✅ NaN |
| `EMA` 递推 | `(2X+(N-1)EMA')/(N+1)` | ✅ 与 pandas ewm(span) 误差 0 |


## 公式 → 画线打通（v0.3.0）

公式里的绘图指令可直接转成 [tdxdraw](https://gitee.com/mengyiheng/tdxdraw) 画线对象，
`write_eld` 生成 .eld 后在通达信里导入——**分析程序的输出变成真正的画线**：

```bash
pip install tdxformula[draw]
```

```python
from tdxformula import eval_to_drawings
from tdxdraw import write_eld

drawings = eval_to_drawings(
    "DRAWLINE(LOW=LLV(LOW,20),LOW,HIGH=HHV(HIGH,20),HIGH,0);",
    {"CLOSE": close, "HIGH": high, "LOW": low},
    market="sh", code="600519", dates=dates)
write_eld("out.eld", drawings)      # 通达信画线导入即可见
```

支持：`DRAWLINE`（配对连线）、`DRAWTEXT`（文字标注）、`POLYLINE`（折线链）；
`DRAWICON/STICKLINE` 暂跳过。
## 免责声明

本项目仅为公式文本的格式转换工具，不提供任何行情数据，不构成任何投资建议。
使用前请[阅读完整免责声明](DISCLAIMER.md)。

## License

MIT

## 量化相关定制需求

作者承接公式迁移与转译器扩展的定制开发：

| 方向 | 示例 |
|---|---|
| 📦 公式批量迁移 | 你的私藏公式库整批转译 + 与通达信逐根比对验收 |
| 🧮 函数扩充 | 按需实现未覆盖的序列函数（对齐通达信语义） |
| 🔗 深度集成 | 选股公式转盘中扫描（接 tdxalert）、绘图指令转 tdxdraw |

**联系合作**：[Gitee Issues](https://gitee.com/mengyiheng/tdxformula/issues) 或 [Gitee 私信](https://gitee.com/mengyiheng)。迁移类需求请附：公式数量、是否含绘图指令。
