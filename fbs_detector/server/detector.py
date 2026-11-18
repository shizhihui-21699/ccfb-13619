"""检测引擎：证据融合 + 时空聚类 + 置信度分级。

置信度模型（可解释、可审计，而非黑盒打分）：
- 先验对数几率 logit(P_rogue)：现实中伪基站是稀有事件，先验给低；
- 每类信号贡献 LLR 权重，同类信号重复触发按 sqrt(n) 次线性累加
  （重复证据信息边际递减，防止一条规则刷分）；
- 多设备独立佐证再加一份证据（单台手机的基带异常是常见噪声，
  N 台互不相识的设备同时看到同一现象才是强证据）；
- sigmoid 转回概率，映射为 低/中/高/确凿 四级；
- 注册表计划内工程一票压制（压制事件仍留审计记录）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from fbs_detector.common.geo import dbscan_xy, haversine_m, project_xy
from fbs_detector.common.layout import PARK_BY_ID
from fbs_detector.common.records import Observation
from fbs_detector.server.features import (
    CandidateEmitter, build_candidates, extract_signals,
)
from fbs_detector.server.registry import CarrierRegistry

# 先验：千分之一量级的候选是恶意发射源（保守值，压低误报）
PRIOR_LOG_ODDS = math.log(0.001 / 0.999)

# 多设备佐证：独立设备数带来的额外 LLR（封顶，防止超大园区堆设备数）
def corroboration_llr(n_devices: int) -> float:
    if n_devices <= 1:
        return 0.0
    return min(2.2, 0.55 * math.log2(n_devices))

# 置信度分级阈值（概率）
LEVELS = [
    (0.90, "确凿", "critical"),
    (0.70, "高",   "high"),
    (0.45, "中",   "medium"),
    (0.25, "低",   "low"),
]

# 低于该概率不产生告警，只入观察名单
ALERT_FLOOR = 0.25

# 身份漂移聚类参数：同园区 300m 内的未知候选视为同一事件
DRIFT_EPS_M = 300.0
DRIFT_MIN_SAMPLES = 1


@dataclass
class Alert:
    event_id: str
    park_id: str
    rat: str
    confidence: float
    level: str
    severity: str
    lat: float
    lon: float
    radius_m: float
    time_windows: list[tuple[float, float]]
    first_seen: float
    last_seen: float
    n_devices: int
    n_observations: int
    merged_ecis: list[int]
    merged_pcis: list[int]
    evidence: list[dict]
    suppressed: bool = False
    suppress_reason: str = ""

    def top_evidence(self, k: int = 5) -> list[dict]:
        return sorted(self.evidence, key=lambda e: e["contribution"], reverse=True)[:k]


@dataclass
class DetectionResult:
    alerts: list[Alert] = field(default_factory=list)
    watchlist: list[dict] = field(default_factory=list)
    suppressed: list[Alert] = field(default_factory=list)
    candidate_count: int = 0


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


def score_candidate(cand: CandidateEmitter) -> tuple[float, list[dict]]:
    """返回 (恶意概率, 逐条证据贡献明细)。"""
    log_odds = PRIOR_LOG_ODDS
    detail = []
    for code, hit in cand.signals.items():
        contribution = hit.weight * math.sqrt(hit.count)  # 重复触发边际递减
        log_odds += contribution
        detail.append({"code": code, "weight": hit.weight, "count": hit.count,
                       "contribution": round(contribution, 3), "detail": hit.detail})
    log_odds += corroboration_llr(len(cand.devices))
    detail.append({"code": "CORROBORATION", "weight": 0.0,
                   "count": len(cand.devices),
                   "contribution": round(corroboration_llr(len(cand.devices)), 3),
                   "detail": f"{len(cand.devices)} 台独立设备佐证"})
    return _sigmoid(log_odds), detail


def _level_for(p: float) -> tuple[str, str]:
    for thr, name, sev in LEVELS:
        if p >= thr:
            return name, sev
    return "低", "low"


def _merge_time_windows(obs: list[Observation], gap_s: float = 1800.0) -> list[tuple[float, float]]:
    """把观测时间戳合并成活动时间窗（相邻间隔 >30 分钟则断开）。"""
    ts = sorted(o.ts for o in obs)
    windows = []
    start = prev = ts[0]
    for t in ts[1:]:
        if t - prev > gap_s:
            windows.append((start, prev))
            start = t
        prev = t
    windows.append((start, prev))
    return windows


def detect(observations: list[Observation], registry: CarrierRegistry) -> DetectionResult:
    result = DetectionResult()
    candidates = build_candidates(observations)
    result.candidate_count = len(candidates)
    for cand in candidates:
        extract_signals(cand, registry, candidates)

    # —— 身份漂移聚类：仅对「注册表外的未知候选」按空间邻近合并 ——
    # 已知身份（含计划内 COW）各自独立成事件，绝不与未知候选混簇，
    # 否则仿冒站会借合法小区身份洗白，计划站的压制标记也会被淹没。
    park_groups: dict[str, list[CandidateEmitter]] = {}
    for c in candidates:
        park_groups.setdefault(c.park_id, []).append(c)

    events: list[list[CandidateEmitter]] = []
    for park_id, group in park_groups.items():
        park = PARK_BY_ID[park_id]
        unknown = [c for c in group if not c.is_known_identity]
        for c in group:
            if c.is_known_identity:
                events.append([c])
        if not unknown:
            continue
        pts = [project_xy(c.lat, c.lon, park.center_lat, park.center_lon) for c in unknown]
        labels = dbscan_xy(pts, DRIFT_EPS_M, DRIFT_MIN_SAMPLES)
        clusters: dict[int, list[CandidateEmitter]] = {}
        for c, lab in zip(unknown, labels):
            clusters.setdefault(lab, []).append(c)
        for members in clusters.values():
            # 再按制式/频点细分，GSM 与 LTE 即使位置重叠也不是同一个事件
            sub: dict[tuple[str, int], list[CandidateEmitter]] = {}
            for m in members:
                sub.setdefault((m.rat, m.earfcn), []).append(m)
            events.extend(sub.values())

    # —— 每个事件打分并出告警 ——
    for idx, members in enumerate(events):
        all_obs = [o for m in members for o in m.observations]
        all_devices = {d for m in members for d in m.devices}
        # 合并证据：同信号取成员中最大贡献，避免合并后重复刷分
        merged_signals: dict[str, dict] = {}
        suppressed = False
        suppress_reason = ""
        for m in members:
            if m.known_entry is not None and m.known_entry.planned and not m.signals:
                suppressed = True
                suppress_reason = "运营商计划内工程（COW/扩容）窗口内，白名单压制"
            for code, hit in m.signals.items():
                cur = merged_signals.get(code)
                contrib = hit.weight * math.sqrt(hit.count)
                if cur is None or contrib > cur["contribution"]:
                    merged_signals[code] = {"code": code, "weight": hit.weight,
                                            "count": hit.count,
                                            "contribution": contrib, "detail": hit.detail}
        # 事件级概率：先验 + 合并后证据 + 事件级设备佐证
        log_odds = PRIOR_LOG_ODDS
        evidence = list(merged_signals.values())
        for e in evidence:
            log_odds += e["contribution"]
        log_odds += corroboration_llr(len(all_devices))
        evidence.append({"code": "CORROBORATION", "weight": 0.0, "count": len(all_devices),
                         "contribution": round(corroboration_llr(len(all_devices)), 3),
                         "detail": f"{len(all_devices)} 台独立设备佐证"})
        prob = _sigmoid(log_odds)

        # 时空定位优先使用 S02 异常子群（盗用 ECI 场景下去掉合法观测稀释）
        loc_obs = [o for m in members for o in m.anomalous_observations] or all_obs
        lat = sum(o.lat for o in loc_obs) / len(loc_obs)
        lon = sum(o.lon for o in loc_obs) / len(loc_obs)
        spread = sorted(haversine_m(lat, lon, o.lat, o.lon) for o in loc_obs)
        radius = max(100.0, spread[len(spread) // 2] * 1.25 + 120.0)
        windows = _merge_time_windows(loc_obs)
        ecis = sorted({m.eci for m in members})
        pcis = sorted({m.pci for m in members})
        primary = max(members, key=lambda m: len(m.observations))

        alert = Alert(
            event_id=f"EVT-{idx + 1:03d}", park_id=primary.park_id, rat=primary.rat,
            confidence=round(prob, 4), level=_level_for(prob)[0],
            severity=_level_for(prob)[1], lat=lat, lon=lon, radius_m=round(radius, 1),
            time_windows=windows, first_seen=min(o.ts for o in loc_obs),
            last_seen=max(o.ts for o in loc_obs), n_devices=len(all_devices),
            n_observations=len(all_obs), merged_ecis=ecis, merged_pcis=pcis,
            evidence=sorted(evidence, key=lambda e: e["contribution"], reverse=True),
            suppressed=suppressed, suppress_reason=suppress_reason,
        )
        if suppressed:
            result.suppressed.append(alert)
        elif prob >= ALERT_FLOOR:
            result.alerts.append(alert)
        else:
            result.watchlist.append({
                "event_id": alert.event_id, "park_id": alert.park_id,
                "confidence": alert.confidence, "reason": "证据不足，列入观察名单持续跟踪",
            })
    result.alerts.sort(key=lambda a: a.confidence, reverse=True)
    return result
