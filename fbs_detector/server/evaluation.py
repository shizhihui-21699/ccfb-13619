"""评估：把检测告警与仿真地面真值匹配，量化检测质量。

匹配规则（同时满足才算命中一个伪基站）：
- 园区一致；
- 告警时间窗与真值活动窗有交集（交集≥10 分钟）；
- 告警位置圆心距真值坐标 ≤ 告警不确定半径 + 200m。
未匹配的告警为误报(FP)，未被命中的真值为漏报(FN)。
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from fbs_detector.common.geo import haversine_m
from fbs_detector.server.detector import Alert


@dataclass
class Match:
    rogue_id: str
    event_id: str
    confidence: float
    spatial_error_m: float
    time_overlap_s: float
    detection_delay_s: float


def _window_overlap(w1: list[tuple[float, float]],
                    w2: list[tuple[float, float]]) -> float:
    total = 0.0
    for a1, b1 in w1:
        for a2, b2 in w2:
            total += max(0.0, min(b1, b2) - max(a1, a2))
    return total


def evaluate(alerts: list[Alert], ground_truth_path: str) -> dict:
    with open(ground_truth_path, encoding="utf-8") as f:
        gt = json.load(f)
    rogues = gt["rogues"]

    matches: list[Match] = []
    used_alerts: set[str] = set()
    for r in rogues:
        best = None
        for a in alerts:
            if a.event_id in used_alerts or a.park_id != r["park_id"]:
                continue
            overlap = _window_overlap(a.time_windows, r["active_windows"])
            if overlap < 600.0:
                continue
            err = haversine_m(a.lat, a.lon, r["lat"], r["lon"])
            if err > a.radius_m + 200.0:
                continue
            delay = max(0.0, a.first_seen - r["active_windows"][0][0])
            cand = Match(r["rogue_id"], a.event_id, a.confidence,
                         round(err, 1), overlap, round(delay, 1))
            if best is None or a.confidence > alerts_by_id(alerts, best.event_id).confidence:
                best = cand
        if best is not None:
            matches.append(best)
            used_alerts.add(best.event_id)

    tp = len(matches)
    fp = len(alerts) - tp
    fn = len(rogues) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    spatial = [m.spatial_error_m for m in matches]
    delays = [m.detection_delay_s for m in matches]

    return {
        "n_rogues_truth": len(rogues),
        "n_alerts": len(alerts),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "median_spatial_error_m": round(sorted(spatial)[len(spatial) // 2], 1) if spatial else None,
        "max_spatial_error_m": round(max(spatial), 1) if spatial else None,
        "median_detection_delay_min": round(sorted(delays)[len(delays) // 2] / 60.0, 1) if delays else None,
        "max_detection_delay_min": round(max(delays) / 60.0, 1) if delays else None,
        "matches": [m.__dict__ for m in matches],
        "false_positive_events": [a.event_id for a in alerts if a.event_id not in used_alerts],
        "missed_rogues": [r["rogue_id"] for r in rogues
                          if r["rogue_id"] not in {m.rogue_id for m in matches}],
    }


def alerts_by_id(alerts: list[Alert], event_id: str) -> Alert:
    return next(a for a in alerts if a.event_id == event_id)
