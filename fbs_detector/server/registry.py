"""运营商网络注册表（服务端可信基线）。

真实系统里由运营商 OSS/NRM 下发并每日同步：合法小区的 ECI、PCI、TAC、
频段、加密能力、计划内工程（COW/扩容/退服）窗口。检测器的一切「异常」
都相对这个基线判断，计划窗口内的变化必须被压住，这是防误报的第一道闸。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fbs_detector.sim.topology import Cell


@dataclass
class RegistryEntry:
    eci: int
    rat: str
    park_id: str
    pci: int
    tac: int
    earfcn: int
    lat: float
    lon: float
    cipher: str
    integrity: str | None
    planned: bool
    active_windows: list[tuple[float, float]] = field(default_factory=list)


class CarrierRegistry:
    def __init__(self, cells: dict[str, Cell]):
        self._by_eci: dict[int, RegistryEntry] = {}
        self._pci_earfcn: dict[tuple[int, int], list[RegistryEntry]] = {}
        for c in cells.values():
            e = RegistryEntry(
                eci=c.eci, rat=c.rat, park_id=c.park_id, pci=c.pci, tac=c.tac,
                earfcn=c.earfcn, lat=c.lat, lon=c.lon, cipher=c.cipher,
                integrity=c.integrity, planned=c.planned,
                active_windows=list(c.active_windows),
            )
            self._by_eci[e.eci] = e
            self._pci_earfcn.setdefault((c.pci, c.earfcn), []).append(e)

    def lookup_eci(self, eci: int) -> RegistryEntry | None:
        return self._by_eci.get(eci)

    def lookup_pci(self, pci: int, earfcn: int) -> list[RegistryEntry]:
        return self._pci_earfcn.get((pci, earfcn), [])

    def is_planned_active(self, eci: int, ts: float) -> bool:
        """计划内工程小区在报备窗口内活动视为合法。"""
        e = self._by_eci.get(eci)
        if not e or not e.planned:
            return False
        if not e.active_windows:
            return True
        return any(a <= ts <= b for a, b in e.active_windows)

    def all_ecis(self) -> set[int]:
        return set(self._by_eci)
