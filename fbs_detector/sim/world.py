"""仿真编排：驱动园区、设备、合法网络与伪基站共同运转。"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

from fbs_detector.common.layout import PARKS
from fbs_detector.common.records import BatchEnvelope
from fbs_detector.sim.device import TICK_S, DeviceSim
from fbs_detector.sim.rogue import RogueCell, build_rogues
from fbs_detector.sim.topology import Cell, add_planned_cow, build_topology

# 仿真设备与服务端之间的「应用层签名密钥」（演示用固定值，生产环境每机一密）
DEVICE_SECRET = b"sim-shared-collector-secret-v1"


@dataclass
class SimResult:
    envelopes: list[BatchEnvelope]
    cells: dict[str, Cell]
    rogues: list[RogueCell]
    gt_events: list[dict]
    ts0: float
    ts1: float
    ticks: int


def run_simulation(days: float = 5.0, seed: int = 42,
                   ts0: float = 1_762_000_000.0, batch_ticks: int = 12,
                   include_rogues: bool = True) -> SimResult:
    rng = random.Random(seed)
    ticks = int(days * 86400.0 / TICK_S)
    ts1 = ts0 + ticks * TICK_S

    cells = build_topology()
    add_planned_cow(cells, ts0 + 0.5 * 86400, ts0 + 1.6 * 86400)  # 计划内扩容站
    rogues = build_rogues(ts0, cells) if include_rogues else []

    devices: list[DeviceSim] = []
    for park in PARKS:
        for i in range(park.devices):
            devices.append(DeviceSim(
                device_id=f"{park.park_id}-D{i:03d}", park=park,
                rng=random.Random(seed * 1000 + hash(park.park_id) % 100_000 + i),
                ts0=ts0,
            ))

    pending: dict[str, list] = {d.device_id: [] for d in devices}
    envelopes: list[BatchEnvelope] = []
    gt_events: list[dict] = []
    batch_seq = 0

    for t in range(ticks):
        for d in devices:
            obs = d.step(cells, rogues)
            if obs is not None:
                pending[d.device_id].append(obs)
            gt_events.extend(d.gt_events)
            d.gt_events.clear()
        # 按批打包（模拟每小时左右的批量上报，事件驱动时凑够即发）
        if (t + 1) % batch_ticks == 0:
            for dev_id, bucket in pending.items():
                if not bucket:
                    continue
                batch_seq += 1
                env = BatchEnvelope(
                    schema_version=1, device_id=dev_id,
                    batch_id=f"B{batch_seq:06d}", issued_at=bucket[-1].ts,
                    observations=bucket,
                )
                envelopes.append(env)
            pending = {d.device_id: [] for d in devices}

    # 收尾批次
    for dev_id, bucket in pending.items():
        if bucket:
            batch_seq += 1
            envelopes.append(BatchEnvelope(
                schema_version=1, device_id=dev_id,
                batch_id=f"B{batch_seq:06d}", issued_at=bucket[-1].ts,
                observations=bucket,
            ))

    return SimResult(envelopes=envelopes, cells=cells, rogues=rogues,
                     gt_events=gt_events, ts0=ts0, ts1=ts1, ticks=ticks)


def save_ground_truth(result: SimResult, path: str) -> None:
    """导出地面真值（仅评估环节使用，检测器不可读）。"""
    gt = {
        "ts0": result.ts0, "ts1": result.ts1,
        "rogues": [
            {"rogue_id": r.rogue_id, "park_id": r.park_id, "rat": r.rat,
             "lat": r.lat, "lon": r.lon,
             "active_windows": r.active_windows,
             "identity_schedule": r.identity_schedule,
             "cipher": r.cipher, "cgi_spoof": r.cgi_spoof,
             "redirect_radius_m": r.redirect_radius_m}
            for r in result.rogues
        ],
        "camp_events": [
            {"ts": e["ts"], "d": e["device_id"], "s": e["serving"],
             "r": int(e["rogue"]), "f": int(e["forced"])}
            for e in result.gt_events
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False)
