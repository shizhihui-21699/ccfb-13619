"""端到端演示入口：
仿真 → 采集端签名上报 → 服务端验签入库 → 多特征检测 → 时空定位 → 评估 → HTML 报告。

用法：
    python run_demo.py [--days 5] [--seed 42] [--out output]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fbs_detector.collector.app import LightCollector
from fbs_detector.common.records import BatchEnvelope
from fbs_detector.server.detector import detect
from fbs_detector.server.evaluation import evaluate
from fbs_detector.server.registry import CarrierRegistry
from fbs_detector.server.report import render_html
from fbs_detector.sim.world import DEVICE_SECRET, run_simulation, save_ground_truth

_CN_TZ = timezone(timedelta(hours=8))


def _fmt(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=_CN_TZ).strftime("%m-%d %H:%M")


def main() -> int:
    ap = argparse.ArgumentParser(description="伪基站发现系统端到端演示")
    ap.add_argument("--days", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=" * 68)
    print("阶段 1/5  运行网络仿真（合法基站 + 4 个隐藏伪基站 + 72 台员工设备）")
    print("=" * 68)
    result = run_simulation(days=args.days, seed=args.seed)
    n_obs = sum(len(e.observations) for e in result.envelopes)
    print(f"  仿真时长 {args.days} 天，tick=300s，共 {result.ticks} 个 tick")
    print(f"  合法小区 {len(result.cells)} 个（含 1 个计划内 COW 临时站）")
    print(f"  签名上报批次 {len(result.envelopes)} 个，观测记录 {n_obs} 条")
    gt_path = os.path.join(args.out, "ground_truth.json")
    save_ground_truth(result, gt_path)
    print(f"  地面真值已写入 {gt_path}（检测器不可读，仅评估用）")

    print("=" * 68)
    print("阶段 2/5  采集端处理 + 服务端验签入库")
    print("=" * 68)
    collectors: dict[str, LightCollector] = {}
    accepted: list = []
    tamper_rejected = 0
    for idx, env in enumerate(result.envelopes):
        wire = env.to_wire(DEVICE_SECRET)
        # 演示一次篡改攻击：挑一个批次改掉内层载荷的一个十六进制字符，
        # 外层 JSON 仍合法，HMAC 验签必须失败
        if idx == 7:
            outer = json.loads(wire.decode("utf-8"))
            payload = outer["payload"]
            flipped = "0" if payload[300] != "0" else "1"
            outer["payload"] = payload[:300] + flipped + payload[301:]
            wire = json.dumps(outer).encode("utf-8")
        try:
            recovered = BatchEnvelope.from_wire(wire, DEVICE_SECRET)
        except ValueError as exc:
            tamper_rejected += 1
            print(f"  ! 批次 {env.batch_id} 验签拒绝：{exc}")
            continue
        col = collectors.setdefault(recovered.device_id,
                                    LightCollector(recovered.device_id))
        for o in recovered.observations:
            col.ingest(o)
        accepted.extend(recovered.observations)
    total_dups = sum(c.dropped_duplicates for c in collectors.values())
    print(f"  接入设备 {len(collectors)} 台，验签通过观测 {len(accepted)} 条")
    print(f"  采集端本地去重 {total_dups} 条，篡改批次被拒 {tamper_rejected} 个")

    print("=" * 68)
    print("阶段 3/5  多特征证据提取与融合检测")
    print("=" * 68)
    registry = CarrierRegistry(result.cells)
    det = detect(accepted, registry)
    print(f"  候选发射源 {det.candidate_count} 个，产生告警 {len(det.alerts)} 起，"
          f"观察名单 {len(det.watchlist)} 个，白名单压制 {len(det.suppressed)} 起")
    for a in det.alerts:
        top = "、".join(e["code"] for e in a.top_evidence(3))
        print(f"  [{a.severity.upper():8s}] {a.event_id} {a.park_id} {a.rat:3s} "
              f"P={a.confidence:.3f} 设备{a.n_devices:2d}台 窗口 {_fmt(a.first_seen)}~{_fmt(a.last_seen)} "
              f"半径{a.radius_m:5.0f}m  证据: {top}")

    print("=" * 68)
    print("阶段 4/5  与地面真值对照评估")
    print("=" * 68)
    metrics = evaluate(det.alerts, gt_path)
    for k in ("n_rogues_truth", "n_alerts", "tp", "fp", "fn", "precision",
              "recall", "f1", "median_spatial_error_m", "max_spatial_error_m",
              "median_detection_delay_min", "max_detection_delay_min"):
        print(f"  {k:32s}: {metrics[k]}")
    if metrics["missed_rogues"]:
        print(f"  漏报真值: {metrics['missed_rogues']}")
    if metrics["false_positive_events"]:
        print(f"  误报事件: {metrics['false_positive_events']}")

    print("=" * 68)
    print("阶段 5/5  生成报告与结构化告警")
    print("=" * 68)
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)
    report_path = os.path.join(args.out, "report.html")
    render_html(det, metrics, gt["rogues"], len(accepted), len(result.envelopes), report_path)
    alerts_json = [{
        "event_id": a.event_id, "park_id": a.park_id, "rat": a.rat,
        "confidence": a.confidence, "level": a.level, "severity": a.severity,
        "lat": a.lat, "lon": a.lon, "radius_m": a.radius_m,
        "first_seen": a.first_seen, "last_seen": a.last_seen,
        "time_windows": a.time_windows, "n_devices": a.n_devices,
        "n_observations": a.n_observations,
        "merged_ecis": a.merged_ecis, "merged_pcis": a.merged_pcis,
        "evidence": a.evidence,
    } for a in det.alerts]
    alerts_path = os.path.join(args.out, "alerts.json")
    with open(alerts_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "alerts": alerts_json,
                   "watchlist": det.watchlist,
                   "suppressed": [{"event_id": a.event_id, "reason": a.suppress_reason}
                                  for a in det.suppressed]},
                  f, ensure_ascii=False, indent=2)
    print(f"  HTML 报告 : {report_path}")
    print(f"  结构化告警: {alerts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
