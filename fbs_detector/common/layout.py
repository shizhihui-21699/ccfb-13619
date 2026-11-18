"""园区场景定义：三个园区的地理布局与参考坐标。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Park:
    park_id: str
    name: str
    center_lat: float
    center_lon: float
    radius_m: float          # 园区大致半径
    tac: int                 # 该园区合法小区使用的跟踪区码
    devices: int             # 该园区内仿真员工设备数


PARKS = [
    Park("P-A", "滨江科技园", 30.2000, 120.2000, 450.0, tac=21001, devices=26),
    Park("P-B", "云谷产业园", 30.2800, 120.2700, 400.0, tac=21002, devices=24),
    Park("P-C", "西溪创智园", 30.2400, 120.1500, 380.0, tac=21003, devices=22),
]

PARK_BY_ID = {p.park_id: p for p in PARKS}

MCC, MNC = 460, 11          # 仿真运营商（虚构 MNC，避免与真实运营商混淆）
