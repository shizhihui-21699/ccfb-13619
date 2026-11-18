"""无线传播模型（仿真用）。

采用 COST 风格的对数距离路径损耗 + 阴影衰落 + 快衰落：
    PL(d) = PL0 + 10*n*log10(d/d0) + X_shadow
不同制式给不同的路径损耗指数与穿透余量，GSM 低频传播更远，这也是伪基站
偏好 GSM 的物理原因之一。
"""
from __future__ import annotations

import math
import random

# 参考距离 1m 处的自由空间路损（按频段近似）与典型路径损耗指数
_RAT_PARAMS = {
    "GSM":  {"pl0": 31.0, "pl_exp": 3.3, "shadow_sigma": 6.0, "fast_sigma": 2.5},
    "LTE":  {"pl0": 37.0, "pl_exp": 3.7, "shadow_sigma": 7.0, "fast_sigma": 3.0},
    "NR":   {"pl0": 39.0, "pl_exp": 3.9, "shadow_sigma": 7.5, "fast_sigma": 3.5},
}


def path_loss_db(distance_m: float, rat: str, rng: random.Random) -> float:
    p = _RAT_PARAMS[rat]
    d = max(distance_m, 1.0)
    pl = p["pl0"] + 10.0 * p["pl_exp"] * math.log10(d)
    pl += rng.gauss(0.0, p["shadow_sigma"])     # 慢衰落（建筑遮挡）
    pl += rng.gauss(0.0, p["fast_sigma"])       # 快衰落（多径）
    return pl


def rsrp_dbm(tx_power_dbm: float, distance_m: float, rat: str,
             rng: random.Random, indoor_penalty_db: float = 8.0) -> float:
    """接收功率 = 发射功率 - 路径损耗 - 室内穿透余量。"""
    return tx_power_dbm - path_loss_db(distance_m, rat, rng) - indoor_penalty_db


def rsrq_db(rsrp: float, total_rssi: float) -> float:
    """由 RSRP 与总干扰 RSSI 折算 RSRQ（dB，负值，越接近 -3 越好）。

    total_rssi 包含服务小区与所有同频邻区功率；同频污染越重 RSRQ 越差。
    """
    val = rsrp - total_rssi  # 简化关系，截断到 LTE 规范的 [-19.5, -3]
    return max(-19.5, min(-3.0, val))


def gsm_rxlev(rsrp: float) -> int:
    """GSM RxLev 0..63（每格约 1 dBm，映射 -110..-47 dBm）。"""
    return max(0, min(63, int((rsrp + 110.0))))
