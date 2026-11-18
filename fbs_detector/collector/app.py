"""轻量采集端（手机侧 App 的抽象）。

真机上对应 Android TelephonyManager + 定期轮询服务；这里实现其核心逻辑：
- 接收 DeviceSim 产生的原始观测（真机里由系统回调产生）；
- 变化检测：驻留小区/制式/加密变化立即标记，无变化走周期心跳；
- 本地去重与批量缓存，凑批后签名上报，弱网时缓存重发。
"""
from __future__ import annotations

from fbs_detector.common.records import BatchEnvelope, Observation
from fbs_detector.sim.world import DEVICE_SECRET


class LightCollector:
    def __init__(self, device_id: str, secret: bytes = DEVICE_SECRET, max_batch: int = 64):
        self.device_id = device_id
        self.secret = secret
        self.max_batch = max_batch
        self._cache: list[Observation] = []
        self._last_fingerprint: tuple | None = None
        self.dropped_duplicates = 0
        self.flushed = 0

    @staticmethod
    def fingerprint(o: Observation) -> tuple:
        """用于变化检测的网络指纹。"""
        return (o.rat, o.eci, o.pci, o.tac, o.earfcn, o.cipher, o.integrity)

    def ingest(self, o: Observation) -> bool:
        """入队一条观测；与上一条指纹完全相同则视为重复，丢弃。"""
        fp = self.fingerprint(o)
        if fp == self._last_fingerprint and o.serving_seconds > 0:
            self.dropped_duplicates += 1
            return False
        self._last_fingerprint = fp
        self._cache.append(o)
        return True

    def ready(self) -> bool:
        return len(self._cache) >= self.max_batch

    def flush(self, batch_id: str) -> bytes | None:
        """签名打包为上报字节流；无数据返回 None。"""
        if not self._cache:
            return None
        env = BatchEnvelope(
            schema_version=1, device_id=self.device_id, batch_id=batch_id,
            issued_at=self._cache[-1].ts, observations=list(self._cache),
        )
        wire = env.to_wire(self.secret)
        self._cache.clear()
        self.flushed += 1
        return wire

    def pending(self) -> int:
        return len(self._cache)
