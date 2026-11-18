"""伪基站（恶意发射源）仿真模型。

覆盖三类常见手法：
1. 降级诱捕：GSM 伪基站强制回落，空口仅弱加密/不加密（仿冒银行短信的载体）；
2. 仿冒身份：盗用合法小区的 ECI 广播，但 PCI/邻区/位置对不上；
3. 隐蔽寄生：全新身份、配置尽量模仿合法网络，只在个别参数上露马脚。

伪基站身份会漂移（改 LAC/CI/PCI 试图甩掉追踪），因此检测器不能把
ECI 当稳定主键，需要按空间邻近做事件聚类。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RogueCell:
    rogue_id: str
    park_id: str
    rat: str
    lat: float
    lon: float
    tx_power_dbm: float
    earfcn: int
    # 身份漂移表：(生效起始 ts, tac, eci, pci)
    identity_schedule: list[tuple[float, int, int, int]]
    cipher: str
    integrity: str | None
    active_windows: list[tuple[float, float]]
    redirect_radius_m: float       # 强制降级/重选影响半径，0 表示不强制
    redirect_rat: str | None       # 强制指向的制式
    cgi_spoof: bool                # 是否盗用合法 ECI
    ta_bias: int = 0               # 广播 TA 的人为偏差（部分廉价设备时序粗糙）
    seed: int = 0

    def active_at(self, ts: float) -> bool:
        return any(a <= ts <= b for a, b in self.active_windows)

    def identity_at(self, ts: float) -> tuple[int, int, int]:
        """返回当前时刻广播的 (tac, eci, pci)，按漂移表取最新值。"""
        cur = self.identity_schedule[0]
        for row in self.identity_schedule:
            if ts >= row[0]:
                cur = row
            else:
                break
        return cur[1], cur[2], cur[3]


def build_rogues(ts0: float, cells: dict | None = None) -> list[RogueCell]:
    """构造四个地面真值伪基站（检测器侧不可见此定义）。

    时间均以仿真起点 ts0 为基准。cells 给定时从真实拓扑取被仿冒小区的 ECI，
    保证 R2 的「身份盗用」设定真实成立，而不是凭公式猜一个不存在的 ECI。
    """
    day = 86400.0
    rogues: list[RogueCell] = []
    # 被仿冒的 P-B LTE 宏站真实身份
    victim = cells["P-B-LTE-M1"] if cells else None
    spoof_eci = victim.eci if victim else (100_000 + 1000 + 3) * 256 + 1
    spoof_pci = victim.pci if victim else 17

    # R1：P-A 园区的 GSM 降级伪基站，工作 2 天，中途漂移两次身份
    rogues.append(RogueCell(
        rogue_id="R1-GSM-redirector", park_id="P-A", rat="GSM",
        lat=30.1985, lon=120.2022, tx_power_dbm=43.0, earfcn=75,
        identity_schedule=[
            (ts0 + 0.2 * day, 30999, 20_000 + 501, 11),
            (ts0 + 0.8 * day, 30998, 20_000 + 502, 12),
            (ts0 + 1.5 * day, 30997, 20_000 + 503, 13),
        ],
        cipher="A5-0", integrity=None,
        active_windows=[(ts0 + 0.2 * day, ts0 + 2.1 * day)],
        redirect_radius_m=260.0, redirect_rat="GSM", cgi_spoof=False,
        ta_bias=2, seed=101,
    ))

    # R2：P-B 园区的 LTE 仿冒站，盗用园区宏站 ECI，但 TAC 错误、无加密、位置偏离
    rogues.append(RogueCell(
        rogue_id="R2-LTE-spoofer", park_id="P-B", rat="LTE",
        lat=30.2788, lon=120.2718, tx_power_dbm=44.0, earfcn=1650,
        identity_schedule=[(ts0 + 1.0 * day, 21099, spoof_eci, spoof_pci)],
        cipher="EEA0", integrity="EIA0",
        active_windows=[(ts0 + 1.0 * day, ts0 + 2.6 * day)],
        redirect_radius_m=180.0, redirect_rat="LTE", cgi_spoof=True,
        ta_bias=1, seed=202,
    ))

    # R3：P-C 园区的隐蔽寄生站，参数高度模仿合法网络，仅 TAC 错位+PCI 复用+时序粗糙
    rogues.append(RogueCell(
        rogue_id="R3-LTE-stealth", park_id="P-C", rat="LTE",
        lat=30.2415, lon=120.1480, tx_power_dbm=38.0, earfcn=1650,
        identity_schedule=[
            (ts0 + 2.0 * day, 21077, 88_000 * 256 + 2, 18),
            (ts0 + 2.9 * day, 21077, 88_000 * 256 + 3, 19),
        ],
        cipher="EEA2", integrity="EIA2",   # 加密正常，最难识别的一类
        active_windows=[(ts0 + 2.0 * day, ts0 + 4.2 * day)],
        redirect_radius_m=0.0, redirect_rat=None, cgi_spoof=False,
        ta_bias=3, seed=303,
    ))

    # R4：P-B 园区边缘的短时车载 GSM 站，只出现 3 小时（考验时间窗定位）
    rogues.append(RogueCell(
        rogue_id="R4-GSM-driveby", park_id="P-B", rat="GSM",
        lat=30.2821, lon=120.2682, tx_power_dbm=40.0, earfcn=75,
        identity_schedule=[(ts0 + 3.4 * day, 30996, 20_000 + 601, 21)],
        cipher="A5-1", integrity=None,
        active_windows=[(ts0 + 3.4 * day, ts0 + 3.4 * day + 3 * 3600)],
        redirect_radius_m=200.0, redirect_rat="GSM", cgi_spoof=False,
        ta_bias=1, seed=404,
    ))
    return rogues
