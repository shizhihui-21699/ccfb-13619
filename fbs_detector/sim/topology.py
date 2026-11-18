"""合法蜂窝网络拓扑仿真。

每个园区布置宏站（LTE/5G/GSM）与室内小站；生成全局唯一小区标识、
物理小区标识、频点、邻区表。另含一个「计划内扩容临时小站」（COW），
它在运营商白名单内，检测器必须能压住对它的误报。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from fbs_detector.common.geo import project_xy
from fbs_detector.common.layout import MCC, MNC, PARKS, Park


@dataclass
class Cell:
    cell_id: str                 # 内部唯一名字，如 P-A-LTE-M1
    park_id: str
    rat: str
    lat: float
    lon: float
    height_m: float
    tx_power_dbm: float
    earfcn: int
    pci: int
    tac: int
    ci: int                      # 28 位（LTE）/36 位（NR）/16 位（GSM）小区标识
    eci: int
    cipher: str
    integrity: str | None
    planned: bool = False        # 计划内（白名单）小区
    active_windows: list[tuple[float, float]] = field(default_factory=list)  # 非全时段活动
    neighbors: set[str] = field(default_factory=set)

    def active_at(self, ts: float) -> bool:
        if not self.active_windows:
            return True
        return any(a <= ts <= b for a, b in self.active_windows)


# 频段配置（仿真值）
_FREQ = {
    "GSM": (75, 43.0, 32.0),       # arfcn, 宏站功率, 小站功率
    "LTE": (1650, 46.0, 30.0),
    "NR":  (634000, 46.0, 33.0),
}
_CIPHER = {"GSM": ("A5-1", None), "LTE": ("EEA2", "EIA2"), "NR": ("NEA2", "NIA2")}


def _ring_position(park: Park, radius_frac: float, angle_deg: float) -> tuple[float, float]:
    from fbs_detector.common.geo import inverse_xy
    d = park.radius_m * radius_frac
    x = d * math.cos(math.radians(angle_deg))
    y = d * math.sin(math.radians(angle_deg))
    return inverse_xy(x, y, park.center_lat, park.center_lon)


def build_topology() -> dict[str, Cell]:
    cells: dict[str, Cell] = {}
    ci_seq = 1000

    def add(park: Park, rat: str, kind: str, idx: int, angle: float, rfrac: float) -> Cell:
        nonlocal ci_seq
        earfcn, p_macro, p_small = _FREQ[rat]
        is_macro = kind == "M"
        lat, lon = _ring_position(park, rfrac, angle)
        ci_seq += 1
        if rat == "NR":
            eci = (9000 + ci_seq) * 256 + idx
        elif rat == "LTE":
            eci = (100_000 + ci_seq) * 256 + idx
        else:
            eci = 20_000 + ci_seq          # GSM CI 16 位
        cipher, integ = _CIPHER[rat]
        c = Cell(
            cell_id=f"{park.park_id}-{rat}-{kind}{idx}",
            park_id=park.park_id, rat=rat, lat=lat, lon=lon,
            height_m=32.0 if is_macro else 8.0,
            tx_power_dbm=p_macro if is_macro else p_small,
            earfcn=earfcn, pci=(idx * 17 + (5 if rat == "NR" else 0)) % 504,
            tac=park.tac, ci=eci & 0xFFFFFFF, eci=eci,
            cipher=cipher, integrity=integ,
        )
        cells[c.cell_id] = c
        return c

    for park in PARKS:
        # 两个 LTE 宏站对打，一个 5G 宏站，一个 GSM 老站，一个室内 LTE 小站
        add(park, "LTE", "M", 1, 30, 0.55)
        add(park, "LTE", "M", 2, 210, 0.55)
        add(park, "NR", "M", 1, 120, 0.5)
        add(park, "GSM", "M", 1, 300, 0.6)
        add(park, "LTE", "S", 1, 0, 0.15)

    # 邻区：同园区 500m 内互为邻区（小站只加同制式宏站，避免乒乓名单过大）
    names = list(cells)
    for i, n1 in enumerate(names):
        c1 = cells[n1]
        p1 = project_xy(c1.lat, c1.lon, PARKS[0].center_lat, PARKS[0].center_lon)
        for n2 in names[i + 1:]:
            c2 = cells[n2]
            if c1.park_id != c2.park_id:
                continue
            p2 = project_xy(c2.lat, c2.lon, PARKS[0].center_lat, PARKS[0].center_lon)
            if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) <= 520.0:
                c1.neighbors.add(c2.cell_id)
                c2.neighbors.add(c1.cell_id)
    return cells


def add_planned_cow(cells: dict[str, Cell], start_ts: float, end_ts: float) -> Cell:
    """P-A 园区施工期间的计划内 LTE 临时小站（运营商已报备，白名单内）。"""
    park = next(p for p in PARKS if p.park_id == "P-A")
    lat, lon = _ring_position(park, 0.35, 260)
    cow = Cell(
        cell_id="P-A-LTE-COW1", park_id="P-A", rat="LTE", lat=lat, lon=lon,
        height_m=10.0, tx_power_dbm=30.0, earfcn=1650, pci=222,
        tac=park.tac, ci=77_001_288, eci=77_001 * 256 + 0,
        cipher="EEA2", integrity="EIA2", planned=True,
        active_windows=[(start_ts, end_ts)],
    )
    cells[cow.cell_id] = cow
    for c in cells.values():
        if c.park_id == "P-A" and c.rat == "LTE":
            c.neighbors.add(cow.cell_id)
            cow.neighbors.add(c.cell_id)
    return cow
