"""单元测试：可信通道、信号提取、检测融合与评估匹配。

运行：python -m unittest discover -s fbs_detector/tests -v
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fbs_detector.collector.app import LightCollector
from fbs_detector.common.geo import dbscan_xy, haversine_m
from fbs_detector.common.records import BatchEnvelope, Observation
from fbs_detector.server.detector import (
    ALERT_FLOOR, PRIOR_LOG_ODDS, _sigmoid, corroboration_llr, detect,
)
from fbs_detector.server.evaluation import evaluate
from fbs_detector.server.features import (
    SIGNAL_WEIGHTS, build_candidates, extract_signals,
)
from fbs_detector.server.registry import CarrierRegistry
from fbs_detector.sim.topology import build_topology
from fbs_detector.sim.world import DEVICE_SECRET, run_simulation, save_ground_truth
import tempfile


def _obs(device="P-A-D001", ts=1000.0, park="P-A", rat="LTE", tac=21001,
         eci=999_000_001, pci=300, rsrp=-85.0, cipher="EEA2", integrity="EIA2",
         lat=30.2, lon=120.2, neighbors=None, ta=2, resel=1) -> Observation:
    return Observation(
        device_id=device, ts=ts, park_id=park, lat=lat, lon=lon,
        rat=rat, mcc=460, mnc=11, tac=tac, cell_id=eci & 0xFFFF, eci=eci,
        pci=pci, earfcn=1650 if rat == "LTE" else 75,
        rsrp_dbm=rsrp, rsrq_db=-10.0, rssi_dbm=-75.0, ta=ta,
        cipher=cipher, integrity=integrity, connected=False,
        serving_seconds=300.0, reselection_count=resel,
        neighbors=neighbors or [], sim_mccmnc=(460, 11),
    )


class TestEnvelope(unittest.TestCase):
    def test_roundtrip_and_tamper_rejection(self):
        env = BatchEnvelope(schema_version=1, device_id="d1", batch_id="b1",
                            issued_at=1.0, observations=[_obs()])
        wire = env.to_wire(DEVICE_SECRET)
        back = BatchEnvelope.from_wire(wire, DEVICE_SECRET)
        self.assertEqual(back.device_id, "d1")
        self.assertEqual(back.observations[0].eci, env.observations[0].eci)
        # 篡改载荷后必须验签失败
        outer = wire.decode("utf-8").replace('"sig":', '"sig":"00', 1)
        with self.assertRaises(ValueError):
            BatchEnvelope.from_wire(outer.encode("utf-8"), DEVICE_SECRET)
        # 错误密钥同样拒绝
        with self.assertRaises(ValueError):
            BatchEnvelope.from_wire(wire, b"wrong-secret")

    def test_collector_dedup(self):
        col = LightCollector("d1")
        o1, o2 = _obs(ts=1), _obs(ts=2)
        self.assertTrue(col.ingest(o1))
        self.assertFalse(col.ingest(o2))       # 指纹相同 -> 去重
        o3 = _obs(ts=3, eci=999_000_002)
        self.assertTrue(col.ingest(o3))


class TestGeo(unittest.TestCase):
    def test_haversine_known_distance(self):
        # 赤道上经度差 1 度约 111.19 km
        d = haversine_m(0.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(d, 111_195.0, delta=200.0)

    def test_dbscan_two_clusters_and_noise(self):
        pts = [(0.0, 0.0), (10.0, 0.0), (5.0, 5.0),
               (500.0, 500.0), (510.0, 505.0), (9000.0, 9000.0)]
        labels = dbscan_xy(pts, eps_m=50.0, min_samples=2)
        self.assertEqual(labels[-1], -1)                # 孤点为噪声
        self.assertEqual(labels[0], labels[1])          # 同簇
        self.assertEqual(labels[3], labels[4])
        self.assertNotEqual(labels[0], labels[3])


class TestScoring(unittest.TestCase):
    def test_prior_is_skeptical(self):
        # 无证据时先验概率应在千分之一量级
        self.assertAlmostEqual(_sigmoid(PRIOR_LOG_ODDS), 0.001, delta=0.0005)

    def test_corroboration_saturates(self):
        v2, v4, v32 = corroboration_llr(2), corroboration_llr(4), corroboration_llr(32)
        self.assertGreater(v4, v2)
        self.assertEqual(corroboration_llr(32), corroboration_llr(64))  # 封顶
        self.assertEqual(corroboration_llr(1), 0.0)

    def test_signal_weights_complete(self):
        # 报告里承诺的 14 类信号必须全部有权重且为正
        self.assertEqual(len(SIGNAL_WEIGHTS), 14)
        self.assertTrue(all(w > 0 for w in SIGNAL_WEIGHTS.values()))


class TestDetection(unittest.TestCase):
    def setUp(self):
        self.cells = build_topology()
        self.registry = CarrierRegistry(self.cells)

    def test_unknown_lte_with_no_cipher_triggers_strong_signals(self):
        obs = [_obs(device=f"P-A-D{i:03d}", ts=1000 + i * 300,
                    eci=888_000_000 + i, pci=401, tac=30999,
                    cipher="EEA0", integrity="EIA0", rsrp=-58.0,
                    lat=30.199 + i * 0.0002, lon=120.201)
               for i in range(6)]
        cands = build_candidates(obs)
        for c in cands:
            extract_signals(c, self.registry, cands)
        all_codes = {s for c in cands for s in c.signals}
        self.assertIn("S01_unknown_identity", all_codes)
        self.assertIn("S05_weak_or_no_cipher", all_codes)
        self.assertIn("S03_tac_mismatch", all_codes)

    def test_known_good_cell_produces_no_alert(self):
        # 取一个真实合法小区的参数构造观测，不应产生任何告警
        cell = next(c for c in self.cells.values() if c.rat == "LTE" and not c.planned)
        obs = [_obs(device=f"P-A-D{i:03d}", ts=1000 + i * 300, park=cell.park_id,
                    rat=cell.rat, tac=cell.tac, eci=cell.eci, pci=cell.pci,
                    cipher=cell.cipher, integrity=cell.integrity,
                    lat=cell.lat, lon=cell.lon) for i in range(5)]
        det = detect(obs, self.registry)
        self.assertEqual(det.alerts, [])
        self.assertEqual(det.suppressed, [])

    def test_planned_cow_is_suppressed(self):
        from fbs_detector.sim.topology import add_planned_cow
        cells = build_topology()
        cow = add_planned_cow(cells, 900.0, 2000.0)
        registry = CarrierRegistry(cells)
        obs = [_obs(device=f"P-A-D{i:03d}", ts=1000 + i * 300, park="P-A",
                    rat="LTE", tac=cow.tac, eci=cow.eci, pci=cow.pci,
                    cipher=cow.cipher, integrity=cow.integrity,
                    lat=cow.lat, lon=cow.lon) for i in range(4)]
        det = detect(obs, registry)
        self.assertEqual(det.alerts, [])
        self.assertEqual(len(det.suppressed), 1)
        self.assertIn("白名单", det.suppressed[0].suppress_reason)

    def test_identity_drift_merges_into_one_event(self):
        # 同一点位、三个漂移 ECI，应聚类合并为一个事件而非三起告警
        base = dict(lat=30.2415, lon=120.148, park="P-C", rat="LTE",
                    cipher="EEA2", integrity="EIA2", tac=21077)
        obs = []
        for k, eci in enumerate((22_528_002, 22_528_003, 22_528_004)):
            for i in range(4):
                obs.append(_obs(device=f"P-C-D{i:03d}", ts=1000 + k * 4000 + i * 300,
                                eci=eci, pci=18 + k, **base))
        det = detect(obs, self.registry)
        # 三个 ECI 同属一个空间事件
        park_c_alerts = [a for a in det.alerts if a.park_id == "P-C"]
        self.assertEqual(len(park_c_alerts), 1)
        self.assertGreaterEqual(len(park_c_alerts[0].merged_ecis), 2)


class TestEndToEnd(unittest.TestCase):
    def test_full_pipeline_detects_all_rogues_without_false_positives(self):
        result = run_simulation(days=5.0, seed=42)
        obs = [o for env in result.envelopes for o in env.observations]
        registry = CarrierRegistry(result.cells)
        det = detect(obs, registry)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as f:
            save_ground_truth(result, f.name)
            gt_path = f.name
        try:
            m = evaluate(det.alerts, gt_path)
        finally:
            os.unlink(gt_path)
        self.assertEqual(m["tp"], 4)
        self.assertEqual(m["fp"], 0)
        self.assertEqual(m["fn"], 0)
        self.assertGreaterEqual(m["f1"], 0.99)
        # 时空定位质量门槛
        self.assertLess(m["median_spatial_error_m"], 200.0)
        self.assertLess(m["median_detection_delay_min"], 15.0)
        # 计划内 COW 必须被压制
        self.assertEqual(len(det.suppressed), 1)

    def test_clean_network_produces_zero_alerts(self):
        # 无伪基站的纯净网络（仍含计划内 COW）：任何告警都是误报，必须为零
        for seed in (42, 7, 99):
            result = run_simulation(days=5.0, seed=seed, include_rogues=False)
            obs = [o for env in result.envelopes for o in env.observations]
            registry = CarrierRegistry(result.cells)
            det = detect(obs, registry)
            self.assertEqual(det.alerts, [], f"seed={seed} 纯净网络出现误报")
            self.assertEqual(len(det.suppressed), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
