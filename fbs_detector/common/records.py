"""采集端记录的数据结构与可信上报封装。

设计要点：
- Observation 是手机轻量采集端在一个采样点上能拿到的全部信息（Android
  TelephonyManager 可获得的字段级别），不含通话内容，只含网络层参数。
- BatchEnvelope 模拟「加密传输 + 完整性保护」：生产环境用 TLS 1.3 传输，
  此处用 HMAC-SHA256 做应用层签名，防止批量数据被篡改后注入服务端。
  若环境装有 cryptography，则额外用 Fernet 做载荷加密（可选）。
"""
from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import time
from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = 1


@dataclass
class Observation:
    """单次服务小区/邻区观测（一个采样点一条）。"""
    # —— 设备与时间 ——
    device_id: str
    ts: float                       # Unix 时间戳（秒）
    park_id: str                    # 设备当前所在园区
    lat: float
    lon: float
    # —— 小区身份 ——
    rat: str                        # LTE / NR / GSM
    mcc: int
    mnc: int
    tac: int                        # LTE/5G 跟踪区码；GSM 下填 LAC
    cell_id: int                    # 小区号（CI / NCI 低 28 位）
    eci: int                        # eNodeB/gNB ID + 小区号的完整标识
    pci: int                        # 物理小区标识（PSC 混淆问题就出在这）
    earfcn: int                     # 频点
    # —— 信号质量 ——
    rsrp_dbm: float                 # GSM 下为 RxLev 折算值
    rsrq_db: float | None
    rssi_dbm: float | None
    ta: int | None                  # 时间提前量（距离粗指示）
    # —— 安全与选择行为 ——
    cipher: str                     # EEA2 / EEA1 / EEA0 / A5-1 / A5-0 ...
    integrity: str | None           # EIA2 / EIA1 / EIA0 / None(GSM)
    connected: bool                 # 连接态（业务中）还是空闲态
    serving_seconds: float          # 已驻留该小区时长
    reselection_count: int          # 本小时重选/切换次数
    # —— 环境上下文 ——
    neighbors: list[dict] = field(default_factory=list)  # [{rat,pci,earfcn,rsrp_dbm,...}]
    sim_mccmnc: tuple[int, int] | None = None            # SIM 卡归属运营商
    airplane_toggle: bool = False                        # 采样前 60s 内是否经历过飞行模式切换


@dataclass
class BatchEnvelope:
    """一批观测的可信信封。"""
    schema_version: int
    device_id: str
    batch_id: str
    issued_at: float
    observations: list[Observation]
    payload_enc: str = "gzip-json"          # gzip-json 或 fernet-gzip
    signature: str = ""

    def to_wire(self, secret: bytes) -> bytes:
        """序列化为可传输字节：gzip 压缩 JSON，并附 HMAC-SHA256 签名。"""
        body = {
            "schema_version": self.schema_version,
            "device_id": self.device_id,
            "batch_id": self.batch_id,
            "issued_at": self.issued_at,
            "payload_enc": self.payload_enc,
            "observations": [asdict(o) for o in self.observations],
        }
        raw = gzip.compress(json.dumps(body, separators=(",", ":")).encode("utf-8"))
        sig = hmac.new(secret, raw, hashlib.sha256).hexdigest()
        return json.dumps({"batch_id": self.batch_id, "sig": sig,
                           "payload": raw.hex()}).encode("utf-8")

    @staticmethod
    def from_wire(data: bytes, secret: bytes) -> "BatchEnvelope":
        """验签并解压。签名不符直接拒绝——服务端不接受不可信批量数据。"""
        outer = json.loads(data.decode("utf-8"))
        raw = bytes.fromhex(outer["payload"])
        expected = hmac.new(secret, raw, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, outer["sig"]):
            raise ValueError("HMAC 校验失败：批量数据可能被篡改或伪造")
        body = json.loads(gzip.decompress(raw))
        obs = [Observation(**o) for o in body.pop("observations")]
        env = BatchEnvelope(observations=obs, **body)
        return env


def now_ts() -> float:
    return time.time()
