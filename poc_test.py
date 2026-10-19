# -*- coding: utf-8 -*-
"""POC 验证：MACD / KDJ 公式在真实数据上运行。"""
import sys

sys.path.insert(0, r"D:\pythonProject\tdxformula")

import numpy as np
import pandas as pd

from tdx_formula import eval_formula, gen_python
from tdxdata import read_day

bars = read_day(r"D:\new_tdx64\vipdoc\sh\lday\sh000001.day")
env = {k.upper(): np.array([getattr(b, k) for b in bars], float)
       for k in ("open", "high", "low", "close", "vol")}

# ---- 公式1: MACD（通达信默认参数 12/26/9）----
macd_src = """
DIF:EMA(CLOSE,12)-EMA(CLOSE,26);
DEA:EMA(DIF,9);
MACD:(DIF-DEA)*2;
"""
r1 = eval_formula(macd_src, env)
print("=== MACD ===")
print("输出线:", r1["__outputs__"])
print("最近3根 DIF :", np.round(r1["DIF"][-3:], 3))
print("最近3根 DEA :", np.round(r1["DEA"][-3:], 3))
print("最近3根 MACD:", np.round(r1["MACD"][-3:], 3))

# ---- 公式2: KDJ（9,3,3）----
kdj_src = """
RSV:=(CLOSE-LLV(LOW,9))/(HHV(HIGH,9)-LLV(LOW,9))*100;
K:SMA(RSV,3,1);
D:SMA(K,3,1);
J:3*K-2*D;
"""
r2 = eval_formula(kdj_src, env)
print("=== KDJ ===")
print("最近3根 K:", np.round(r2["K"][-3:], 2))
print("最近3根 D:", np.round(r2["D"][-3:], 2))
print("最近3根 J:", np.round(r2["J"][-3:], 2))
print("K 值范围:", round(np.nanmin(r2["K"]), 2), "~", round(np.nanmax(r2["K"]), 2))

# ---- 与 pandas 独立实现交叉验证 ----
close = pd.Series(env["CLOSE"])
dif_ref = (close.ewm(span=12, adjust=False).mean()
           - close.ewm(span=26, adjust=False).mean()).values
print("=== 与 pandas ewm 交叉验证 ===")
print("DIF 最大绝对误差:", np.nanmax(np.abs(r1["DIF"] - dif_ref)))

# ---- 代码生成演示 ----
print("=== gen_python(MACD) ===")
print(gen_python("DIF:EMA(CLOSE,12)-EMA(CLOSE,26);DEA:EMA(DIF,9);MACD:(DIF-DEA)*2;"))
