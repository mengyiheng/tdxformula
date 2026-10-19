# -*- coding: utf-8 -*-
"""第一梯队公式（已验证可直接转译求值）。

供 s22 探针框架使用：env 提供 OPEN/HIGH/LOW/CLOSE/VOL 序列即可。
用法::

    from tdxformula import eval_formula
    r = eval_formula(DMI_ADX, env)   # r["PDI"], r["MDI"], r["ADX"], r["ADXR"]
"""

# ---------- 第一梯队 ----------

# DMI/ADX(14)：趋势强度。关注点：下跌段中 ADX 从高位回落 = 趋势衰竭
DMI_ADX = """
N:=14;
M:=6;
MTR:=SUM(MAX(MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1))),ABS(REF(CLOSE,1)-LOW)),N);
HD:=HIGH-REF(HIGH,1);
LD:=REF(LOW,1)-LOW;
DMP:=SUM(IF(HD>0 AND HD>LD,HD,0),N);
DMM:=SUM(IF(LD>0 AND LD>HD,LD,0),N);
PDI:DMP*100/MTR;
MDI:DMM*100/MTR;
ADX:MA(ABS(MDI-PDI)/(MDI+PDI)*100,M);
ADXR:(ADX+REF(ADX,M))/2;
"""

# OBV 能量潮：累计量能流，捕捉下跌段内反弹日的资金承接
OBV = """
OBV:SUM(IF(CLOSE>REF(CLOSE,1),VOL,IF(CLOSE<REF(CLOSE,1),-VOL,0)),0);
MAOBV:MA(OBV,30);
"""

# MFI(14) 资金流量：量加权的 RSI（量能 × 摆动交互项）
MFI = """
N:=14;
TYP:=(HIGH+LOW+CLOSE)/3;
MR:=SUM(IF(TYP>REF(TYP,1),TYP*VOL,0),N)/SUM(IF(TYP<REF(TYP,1),TYP*VOL,0),N);
MFI:100-(100/(1+MR));
"""

# ---------- 第二梯队 ----------

# BIAS(20)：终点乖离（超跌深度）
BIAS20 = "BIAS:(CLOSE-MA(CLOSE,20))/MA(CLOSE,20)*100;"

# BOLL 带宽：波动率挤压（带宽收窄 = 变盘前兆）
BOLL_WIDTH = """
MID:MA(CLOSE,20);
UPPER:MID+2*STD(CLOSE,20);
LOWER:MID-2*STD(CLOSE,20);
WIDTH:(UPPER-LOWER)/MID*100;
"""

TIER1 = {"DMI_ADX": DMI_ADX, "OBV": OBV, "MFI": MFI}
TIER2 = {"BIAS20": BIAS20, "BOLL_WIDTH": BOLL_WIDTH}
