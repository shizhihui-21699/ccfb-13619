"""员工手机仿真器：移动、测量、小区选择与采集记录。

行为模型尽量贴近真机：
- 随机游走（园区内 waypoint 模型，夜间静止）；
- 每 tick 测量所有可检测小区的 RSRP，按 RAT 优先级 + 迟滞做重选/切换；
- 进入伪基站强制影响半径时被「降级/重定向」，并记录制式切换事件；
- 仅在驻留变化或周期性心跳时上报，控制数据量。
"""
from __future__ import annotations

import math
import random

from fbs_detector.common.geo import haversine_m, inverse_xy, project_xy
from fbs_detector.common.layout import MCC, MNC, PARK_BY_ID, Park
from fbs_detector.common.records import Observation
from fbs_detector.sim.propagation import rsrp_dbm, rsrq_db
from fbs_detector.sim.rogue import RogueCell
from fbs_detector.sim.topology import Cell

TICK_S = 300.0
DETECT_RSRP_FLOOR = -118.0          # 低于此值手机检测不到
_RAT_BONUS = {"NR": 6.0, "LTE": 3.0, "GSM": 0.0}
_HYSTERESIS_DB = 3.0                # 重选迟滞，防乒乓
_HEARTBEAT_TICKS = 8                # 无变化时每 8 tick（40 分钟）心跳上报


class DeviceSim:
    def __init__(self, device_id: str, park: Park, rng: random.Random, ts0: float):
        self.device_id = device_id
        self.park = park
        self.rng = rng
        self.ts = ts0
        # 初始位置：园区内随机
        a = rng.uniform(0, 2 * math.pi)
        r = park.radius_m * math.sqrt(rng.random()) * 0.9
        self.x, self.y = r * math.cos(a), r * math.sin(a)
        self.target_x, self.target_y = self.x, self.y
        self.speed = 0.0
        self.dwell = 0
        self.serving: str | None = None          # Cell/rogue 的内部 id
        self.serving_is_rogue = False
        self.serving_since = ts0
        self.tick_idx = 0
        self.resel_this_hour = 0
        self.hour_bucket = -1
        self.airplane = False
        self.airplane_cd = rng.randint(20, 80)
        self.gt_events: list[dict] = []          # 地面真值驻留事件（评估用）

    # —— 移动 ——
    def _move(self):
        if self.dwell > 0:
            self.dwell -= 1
            return
        if math.hypot(self.target_x - self.x, self.target_y - self.y) < 15.0:
            # 选新 waypoint；夜间（本地 0-7 点）基本静止
            hour = (self.ts / 3600.0) % 24
            if hour < 7.0 and self.rng.random() < 0.85:
                self.dwell = self.rng.randint(4, 12)
                return
            a = self.rng.uniform(0, 2 * math.pi)
            r = self.park.radius_m * math.sqrt(self.rng.random()) * 0.92
            self.target_x = r * math.cos(a)
            self.target_y = r * math.sin(a)
            self.speed = self.rng.uniform(0.8, 1.8)   # 0.8~1.8 m/s
        step = self.speed * TICK_S
        dx, dy = self.target_x - self.x, self.target_y - self.y
        d = math.hypot(dx, dy) or 1.0
        go = min(step, d)
        self.x += dx / d * go
        self.y += dy / d * go

    def _latlon(self) -> tuple[float, float]:
        return inverse_xy(self.x, self.y, self.park.center_lat, self.park.center_lon)

    def _maybe_airplane(self) -> bool:
        # 低概率飞行模式切换（地铁/电梯场景的合理噪声）
        self.airplane_cd -= 1
        if self.airplane_cd <= 0:
            self.airplane = not self.airplane
            self.airplane_cd = self.rng.randint(40, 120)
            return True
        return False

    # —— 测量与选择 ——
    def _measure(self, cells: dict[str, Cell], rogues: list[RogueCell]) -> list[dict]:
        lat, lon = self._latlon()
        seen = []
        for c in cells.values():
            if c.park_id != self.park.park_id or not c.active_at(self.ts):
                continue
            dist = haversine_m(lat, lon, c.lat, c.lon)
            p = rsrp_dbm(c.tx_power_dbm, dist, c.rat, self.rng)
            if p >= DETECT_RSRP_FLOOR:
                seen.append({"id": c.cell_id, "rogue": False, "ref": c,
                             "rat": c.rat, "rsrp": p, "dist": dist})
        for rg in rogues:
            if rg.park_id != self.park.park_id or not rg.active_at(self.ts):
                continue
            dist = haversine_m(lat, lon, rg.lat, rg.lon)
            p = rsrp_dbm(rg.tx_power_dbm, dist, rg.rat, self.rng)
            if p >= DETECT_RSRP_FLOOR:
                seen.append({"id": rg.rogue_id, "rogue": True, "ref": rg,
                             "rat": rg.rat, "rsrp": p, "dist": dist})
        return seen

    def _rank(self, seen: list[dict]) -> list[dict]:
        # RAT 优先级 + 信号强度；空闲态容忍迟滞，连接态要求更强才切
        for s in seen:
            s["score"] = s["rsrp"] + _RAT_BONUS[s["rat"]]
        return sorted(seen, key=lambda s: s["score"], reverse=True)

    def _select_serving(self, seen: list[dict], connected: bool) -> tuple[str, bool, bool]:
        """返回 (serving_id, is_rogue, forced_redirect)。"""
        # 1) 伪基站强制重定向：在影响半径内且信号可检测，按概率捕获
        for s in seen:
            if not s["rogue"]:
                continue
            rg: RogueCell = s["ref"]
            if rg.redirect_radius_m > 0 and s["dist"] <= rg.redirect_radius_m:
                # 半径内捕获概率随距离边缘递减；真机并非 100% 被捕获
                p_cap = 0.9 if s["dist"] < rg.redirect_radius_m * 0.6 else 0.65
                if self.rng.random() < p_cap:
                    return rg.rogue_id, True, True
        # 2) 常规选择/重选
        ranked = self._rank(seen)
        best = ranked[0]
        if self.serving is None:
            return best["id"], best["rogue"], False
        cur = next((s for s in seen if s["id"] == self.serving), None)
        if cur is None:
            return best["id"], best["rogue"], False   # 服务小区消失
        margin = _HYSTERESIS_DB if not connected else 4.0
        if best["id"] != self.serving and best["score"] > cur["score"] + margin:
            return best["id"], best["rogue"], False
        return cur["id"], cur["rogue"], False

    # —— 主步进 ——
    def step(self, cells: dict[str, Cell], rogues: list[RogueCell]) -> Observation | None:
        self.ts += TICK_S
        self.tick_idx += 1
        hour_bucket = int(self.ts // 3600)
        if hour_bucket != self.hour_bucket:
            self.hour_bucket = hour_bucket
            self.resel_this_hour = 0
        self._move()
        airplane_toggle = self._maybe_airplane()
        if self.airplane:
            self.serving = None  # 脱网，不产生观测
            return None

        connected = self.rng.random() < 0.22     # 约 22% 时间处于连接态
        seen = self._measure(cells, rogues)
        if not seen:
            return None
        new_id, is_rogue, forced = self._select_serving(seen, connected)

        changed = new_id != self.serving
        if changed:
            self.gt_events.append({
                "ts": self.ts, "device_id": self.device_id,
                "serving": new_id, "rogue": is_rogue, "forced": forced,
            })
            self.resel_this_hour += 1
            self.serving, self.serving_is_rogue = new_id, is_rogue
            self.serving_since = self.ts
        chosen = next(s for s in seen if s["id"] == new_id)
        ref = chosen["ref"]

        # 事件驱动 + 心跳：强制重定向必报，普通变化必报，其余周期报
        emit = changed or forced or (self.tick_idx % _HEARTBEAT_TICKS == 0)
        if not emit:
            return None

        lat, lon = self._latlon()
        # 同频总干扰功率（mW 求和后转回 dBm），用于 RSRQ
        total_mw = sum(10 ** (s["rsrp"] / 10.0) for s in seen
                       if s["rat"] == chosen["rat"] and s["id"] != new_id)
        total_rssi = 10 * math.log10(total_mw + 10 ** (chosen["rsrp"] / 10.0))
        rsrq = None if chosen["rat"] == "GSM" else rsrq_db(chosen["rsrp"], total_rssi)

        if is_rogue:
            tac, eci, pci = ref.identity_at(self.ts)
            cipher, integrity, earfcn = ref.cipher, ref.integrity, ref.earfcn
            ta = max(0, int(chosen["dist"] / 550.0) + ref.ta_bias)
        else:
            c: Cell = ref
            tac, eci, pci = c.tac, c.eci, c.pci
            cipher, integrity, earfcn = c.cipher, c.integrity, c.earfcn
            ta = max(0, int(chosen["dist"] / 550.0))

        neighbors = []
        for s in sorted(seen, key=lambda z: z["rsrp"], reverse=True):
            if s["id"] == new_id:
                continue
            nref = s["ref"]
            if s["rogue"]:
                nt, neci, npci = nref.identity_at(self.ts)
            else:
                nt, neci, npci = nref.tac, nref.eci, nref.pci
            neighbors.append({"rat": s["rat"], "pci": npci, "earfcn": nref.earfcn,
                              "rsrp_dbm": round(s["rsrp"], 1), "eci": neci, "tac": nt})
            if len(neighbors) >= 6:
                break

        return Observation(
            device_id=self.device_id, ts=self.ts, park_id=self.park.park_id,
            lat=round(lat, 7), lon=round(lon, 7),
            rat=chosen["rat"], mcc=MCC, mnc=MNC, tac=tac,
            cell_id=eci & 0xFFFF, eci=eci, pci=pci, earfcn=earfcn,
            rsrp_dbm=round(chosen["rsrp"], 1), rsrq_db=None if rsrq is None else round(rsrq, 1),
            rssi_dbm=round(total_rssi, 1), ta=ta,
            cipher=cipher, integrity=integrity, connected=connected,
            serving_seconds=round(self.ts - self.serving_since, 1),
            reselection_count=self.resel_this_hour,
            neighbors=neighbors, sim_mccmnc=(MCC, MNC),
            airplane_toggle=airplane_toggle,
        )
