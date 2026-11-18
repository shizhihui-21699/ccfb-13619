"""特征工程：把众包观测归并为候选发射源并提取异常证据。

设计原则（应对题目里说的难点）：
1. 真实网络自身有大量切换/扩容/临时站，所以每个信号都对照运营商注册表，
   计划内变化直接压制；
2. 伪基站会模仿真实参数，单特征不可信——每个信号只贡献一份对数似然比
   (LLR) 证据，权重按该信号被模仿/被良性事件触发的难度标定；
3. 伪基站会漂移身份，候选主键不用裸 ECI，而用 (park, 频点, 空间邻近簇)
   归并，ECI 只作为身份属性参与判断。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fbs_detector.common.geo import haversine_m
from fbs_detector.common.records import Observation
from fbs_detector.server.registry import CarrierRegistry, RegistryEntry

# —— 信号权重（LLR 底数为 e 的对数几率）——
# 标定思路：越难被良性事件解释、伪基站越难伪装的信号权重越高。
# 权重在检测器里集中可调，这里给经过仿真标定的默认值。
SIGNAL_WEIGHTS = {
    "S01_unknown_identity":      1.10,  # ECI 不在注册表：常见但非定论（扩容也会）
    "S02_cgi_spoof_conflict":    2.30,  # 盗用 ECI 但 PCI/位置对不上：强犯罪信号
    "S03_tac_mismatch":          1.50,  # 广播 TAC 与园区基线不符
    "S04_pci_collision":         1.20,  # 同频同 PCI 上出现两个不同 ECI
    "S05_weak_or_no_cipher":     2.00,  # EEA0/A5-0：现代网络不应出现
    "S06_integrity_disabled":    1.40,  # EIA0/NIA0
    "S07_rat_downgrade":         1.70,  # LTE/5G -> GSM 异常回落且伴随重定向半径
    "S08_geo_inconsistency":     1.60,  # 已知 ECI 的观测位置远离注册坐标
    "S09_power_too_high":        0.80,  # 信号过强/过覆盖（小站功率宏站化）
    "S10_neighbor_inconsistent": 1.00,  # 邻区表里没有任何已知邻区
    "S11_excess_reselection":    0.70,  # 异常高重选率（网络抖动也会，权重低）
    "S12_ta_rsrp_incoherent":    1.10,  # TA 指示距离与 RSRP 反演距离不符
    "S13_temporal_burst":        0.90,  # 多设备在短时间窗内同时首次发现
    "S14_identity_drift":        1.30,  # 同一空间位置的发射源频繁换 ECI/PCI/TAC
}

_WEAK_CIPHERS = {"EEA0", "A5-0", "A5/0", "NONE"}
_KNOWN_GOOD_CIPHER_PREFIX = ("EEA1", "EEA2", "EEA3", "NEA1", "NEA2", "NEA3", "A5-1", "A5-2", "A5-3")


@dataclass
class SignalHit:
    code: str
    weight: float
    count: int                     # 该信号在候选上的触发次数
    detail: str


@dataclass
class CandidateEmitter:
    """一个候选发射源（可能是合法站、计划站或伪基站）。"""
    candidate_id: str
    park_id: str
    rat: str
    earfcn: int
    eci: int
    pci: int
    tac: int
    observations: list[Observation] = field(default_factory=list)
    devices: set[str] = field(default_factory=set)
    signals: dict[str, SignalHit] = field(default_factory=dict)
    known_entry: RegistryEntry | None = None
    is_known_identity: bool = False
    # S02 识别出的「异常观测子群」（盗用 ECI 时来自伪基站的那部分观测），
    # 检测器做时空定位时优先使用，避免被同 ECI 的合法观测稀释
    anomalous_observations: list[Observation] = field(default_factory=list)
    lat: float = 0.0
    lon: float = 0.0

    def add_signal(self, code: str, detail: str) -> None:
        w = SIGNAL_WEIGHTS[code]
        if code in self.signals:
            self.signals[code].count += 1
        else:
            self.signals[code] = SignalHit(code, w, 1, detail)

    @property
    def first_ts(self) -> float:
        return min(o.ts for o in self.observations)

    @property
    def last_ts(self) -> float:
        return max(o.ts for o in self.observations)


def build_candidates(obs_list: list[Observation]) -> list[CandidateEmitter]:
    """按 (园区, 制式, 频点, ECI) 初步归并候选；身份漂移的合并在检测器聚类阶段做。"""
    groups: dict[tuple, CandidateEmitter] = {}
    for o in obs_list:
        key = (o.park_id, o.rat, o.earfcn, o.eci)
        if key not in groups:
            cid = f"C:{o.park_id}:{o.rat}:f{o.earfcn}:e{o.eci}"
            groups[key] = CandidateEmitter(
                candidate_id=cid, park_id=o.park_id, rat=o.rat,
                earfcn=o.earfcn, eci=o.eci, pci=o.pci, tac=o.tac)
        c = groups[key]
        c.observations.append(o)
        c.devices.add(o.device_id)
    # 观测质心
    for c in groups.values():
        c.lat = sum(o.lat for o in c.observations) / len(c.observations)
        c.lon = sum(o.lon for o in c.observations) / len(c.observations)
    return list(groups.values())


def _rsrp_distance_m(rsrp_dbm: float, rat: str, tx_power_dbm: float) -> float:
    """由接收功率反演距离（粗估计，仅用于 TA 一致性检查）。"""
    import math
    pl_exp = {"GSM": 3.3, "LTE": 3.7, "NR": 3.9}[rat]
    pl0 = {"GSM": 31.0, "LTE": 37.0, "NR": 39.0}[rat]
    loss = tx_power_dbm - rsrp_dbm - 8.0  # 去掉室内余量
    if loss <= pl0:
        return 1.0
    return 10.0 ** ((loss - pl0) / (10.0 * pl_exp))


def extract_signals(cand: CandidateEmitter, registry: CarrierRegistry,
                    all_candidates: list[CandidateEmitter]) -> None:
    """对一个候选提取全部可解释的异常信号。"""
    obs = cand.observations
    entry = registry.lookup_eci(cand.eci)
    cand.known_entry = entry
    cand.is_known_identity = entry is not None

    # 计划内工程窗口内：只做记录、不出信号（白名单压制）
    if entry is not None and entry.planned:
        if any(registry.is_planned_active(cand.eci, o.ts) for o in obs):
            return

    # S01 未知身份
    if entry is None:
        cand.add_signal("S01_unknown_identity",
                        f"ECI {cand.eci} 不在运营商注册表，{len(cand.devices)} 台设备观测到")

    # S02 身份盗用冲突 / S08 地理矛盾：
    # 仿冒站盗用合法 ECI 时，真假观测会被归并进同一候选。直接用整体质心会
    # 被大量合法观测稀释，因此先按 (pci, tac) 拆子群，找出与注册参数不符的
    # 异常子群，再用该子群自己的质心做地理判断。
    if entry is not None:
        subgroups: dict[tuple[int, int], list[Observation]] = {}
        for o in obs:
            subgroups.setdefault((o.pci, o.tac), []).append(o)
        bad_obs: list[Observation] = []
        for (pci, tac), grp in subgroups.items():
            if pci != entry.pci or tac != entry.tac:
                bad_obs.extend(grp)
        details = []
        if bad_obs:
            pci_bad = any(o.pci != entry.pci for o in bad_obs)
            tac_bad = any(o.tac != entry.tac for o in bad_obs)
            if pci_bad:
                details.append(f"PCI 广播 {bad_obs[0].pci}≠注册 {entry.pci}")
            if tac_bad:
                details.append(f"TAC 广播 {bad_obs[0].tac}≠注册 {entry.tac}")
            sub_lat = sum(o.lat for o in bad_obs) / len(bad_obs)
            sub_lon = sum(o.lon for o in bad_obs) / len(bad_obs)
            sub_dist = haversine_m(sub_lat, sub_lon, entry.lat, entry.lon)
            details.append(f"异常子群 {len(bad_obs)} 条观测偏离注册坐标 {sub_dist:.0f}m")
            cand.anomalous_observations = bad_obs
            cand.add_signal("S02_cgi_spoof_conflict", "；".join(details))
        else:
            # 参数都对，仍检查整体质心是否偏离（弱定位场景下的兜底）
            d = haversine_m(cand.lat, cand.lon, entry.lat, entry.lon)
            if d > 250.0:
                cand.add_signal("S08_geo_inconsistency",
                                f"观测质心距注册坐标 {d:.0f}m")

    # S03 TAC 与园区基线不符（按该园区合法 TAC 集合判断）
    park_tacs = {e.tac for e in registry._by_eci.values() if e.park_id == cand.park_id}
    if park_tacs and cand.tac not in park_tacs:
        cand.add_signal("S03_tac_mismatch",
                        f"广播 TAC {cand.tac} 不在 {cand.park_id} 合法集合 {sorted(park_tacs)}")

    # S04 同频 PCI 冲突：同 (pci, earfcn) 上出现另一个不同 ECI 的候选
    siblings = [q for q in all_candidates
                if q is not cand and q.park_id == cand.park_id
                and q.earfcn == cand.earfcn and q.pci == cand.pci and q.eci != cand.eci]
    if siblings:
        cand.add_signal("S04_pci_collision",
                        f"PCI {cand.pci}@f{cand.earfcn} 同时被 ECI {siblings[0].eci} 广播")

    # S05/S06 加密与完整性
    # A5-0/EEA0 等空口不加密在现代网络中本身就是强信号；
    # A5-1 是 GSM 老算法（可在秒级被破解），合法 GSM 宏站在注册表内不受影响，
    # 但注册表外的未知发射源使用 A5-1，视为弱加密证据。
    weak_hits = sum(1 for o in obs if o.cipher.upper() in _WEAK_CIPHERS)
    if not weak_hits and entry is None:
        weak_hits = sum(1 for o in obs if o.cipher.upper() == "A5-1")
    if weak_hits:
        ciphers = sorted({o.cipher for o in obs})
        cand.add_signal("S05_weak_or_no_cipher",
                        f"{weak_hits} 条观测为空口弱/无加密 {ciphers}")
    integ_bad = sum(1 for o in obs
                    if (o.integrity or "").upper() in ("EIA0", "NIA0"))
    if integ_bad:
        cand.add_signal("S06_integrity_disabled", f"{integ_bad} 条观测完整性保护关闭")

    # S07 制式异常回落：设备从 LTE/NR 被重选到 GSM，且该 GSM 候选身份未知
    gsm_fall = [o for o in obs if o.rat == "GSM"]
    if gsm_fall and (entry is None or entry.rat != "GSM"):
        # 设备前序为高制式的比例由观测序列推断：同设备在 ±2 个 tick 窗口存在 LTE/NR
        dev_ids = {o.device_id for o in gsm_fall}
        hi = [q for q in all_candidates if q.rat in ("LTE", "NR") and q.devices & dev_ids]
        if hi:
            cand.add_signal("S07_rat_downgrade",
                            f"{len(dev_ids)} 台设备从 LTE/5G 回落到该 GSM 小区")

    # S08 已并入上方 S02 的子群分析（参数一致但位置偏离时触发）

    # S09 过覆盖：注册表外的候选却在多个点位出现宏站级接收功率
    # （合法小站功率 30dBm 左右，不可能在园区多点产生 -62dBm 以上强信号）
    strong = sum(1 for o in obs if o.rsrp_dbm > -62.0)
    if entry is None and strong >= 3:
        cand.add_signal("S09_power_too_high",
                        f"{strong} 个点位 RSRP > -62dBm，疑似高功率非法发射")

    # S10 邻区一致性：邻区中完全没有注册表已知 ECI
    known_ecis = registry.all_ecis()
    bad_neighbor_obs = 0
    for o in obs:
        n_ecis = {n["eci"] for n in o.neighbors}
        if n_ecis and not (n_ecis & known_ecis):
            bad_neighbor_obs += 1
    if bad_neighbor_obs >= max(2, len(obs) // 5):
        cand.add_signal("S10_neighbor_inconsistent",
                        f"{bad_neighbor_obs} 条观测的邻区表与运营商邻区规划无交集")

    # S11 异常重选率
    hi_resel = sum(1 for o in obs if o.reselection_count >= 6)
    if hi_resel >= max(3, len(obs) // 6):
        cand.add_signal("S11_excess_reselection",
                        f"{hi_resel} 条观测小时重选≥6 次")

    # S12 TA 与 RSRP 距离不自洽
    tx_guess = 43.0 if cand.rat == "GSM" else 40.0
    incoherent = 0
    for o in obs:
        if o.ta is None:
            continue
        ta_dist = o.ta * 550.0 + 275.0
        rsrp_dist = _rsrp_distance_m(o.rsrp_dbm, cand.rat, tx_guess)
        if abs(ta_dist - rsrp_dist) > 600.0 and o.ta <= 2:
            incoherent += 1
    if incoherent >= max(2, len(obs) // 8):
        cand.add_signal("S12_ta_rsrp_incoherent",
                        f"{incoherent} 条观测 TA 与信号反演距离矛盾")

    # S13 时间突发：多设备在 30 分钟内首次看到该候选
    if len(cand.devices) >= 3:
        first_seen: dict[str, float] = {}
        for o in sorted(obs, key=lambda z: z.ts):
            first_seen.setdefault(o.device_id, o.ts)
        spread = max(first_seen.values()) - min(first_seen.values())
        if spread <= 1800.0 and len(first_seen) >= 3:
            cand.add_signal("S13_temporal_burst",
                            f"{len(first_seen)} 台设备在 {spread/60:.0f} 分钟内集中首次发现")

    # S14 身份漂移：同一园区/频点、质心 300m 内存在多个不同 ECI 的未知候选
    if entry is None:
        drift_peers = [q for q in all_candidates
                       if q is not cand and q.park_id == cand.park_id
                       and q.earfcn == cand.earfcn and q.eci != cand.eci
                       and registry.lookup_eci(q.eci) is None
                       and haversine_m(cand.lat, cand.lon, q.lat, q.lon) <= 300.0]
        if drift_peers:
            cand.add_signal("S14_identity_drift",
                            f"邻近 {len(drift_peers)} 个未知 ECI 候选，疑似同一发射源身份漂移")
