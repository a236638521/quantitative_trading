import unittest
import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
from backend.app.factors.microstructure import OrderFlowAnalyzer, MICROSTRUCTURE_FACTOR_DEFINITIONS


class TestMicrostructureOrderFlow(unittest.TestCase):
    """测试高频订单流与日内微观结构 Alpha 特征工程"""

    def setUp(self):
        self.analyzer = OrderFlowAnalyzer()

    def test_ofi_mathematical_properties(self):
        """测试 Cont et al. (2014) OFI 订单流不平衡度"""
        # 模拟 3 档时序：
        # t0: bid=10.0 (100股), ask=10.2 (100股)
        # t1: bid=10.1 (150股) [买价上涨，流入 150股], ask=10.2 (100股) [卖价未变，无变化] -> OFI = +150
        # t2: bid=10.1 (80股) [买价持平，减单 70股], ask=10.2 (120股) [卖价持平，增单 20股] -> OFI = -70 - (+20) = -90
        bid_p = np.array([10.0, 10.1, 10.1])
        bid_q = np.array([100.0, 150.0, 80.0])
        ask_p = np.array([10.2, 10.2, 10.2])
        ask_q = np.array([100.0, 100.0, 120.0])

        ofi = self.analyzer.calculate_ofi(bid_p, bid_q, ask_p, ask_q)
        self.assertEqual(len(ofi), 3)
        self.assertEqual(ofi[0], 0.0)
        self.assertEqual(ofi[1], 150.0)
        self.assertEqual(ofi[2], -90.0)

    def test_micro_price_stoikov(self):
        """测试 Stoikov (2018) 微观价格反比加权与界值"""
        bid_p = np.array([10.0, 10.0])
        ask_p = np.array([10.2, 10.2])

        # 1. 买卖量相等 -> 微观价格等于中间价 10.1
        bid_q_eq = np.array([100.0, 100.0])
        ask_q_eq = np.array([100.0, 100.0])
        micro_p, bias_bp = self.analyzer.calculate_micro_price(bid_p, bid_q_eq, ask_p, ask_q_eq)
        self.assertAlmostEqual(micro_p[0], 10.1, places=3)
        self.assertAlmostEqual(bias_bp[0], 0.0, places=2)

        # 2. 买量远大于卖量 (Q_b=900, Q_a=100) -> 潜在向上突破，微观价格向卖一价 10.2 倾斜
        bid_q_heavy = np.array([900.0, 900.0])
        ask_q_light = np.array([100.0, 100.0])
        micro_p2, bias_bp2 = self.analyzer.calculate_micro_price(bid_p, bid_q_heavy, ask_p, ask_q_light)
        # P_micro = (900 * 10.2 + 100 * 10.0) / 1000 = (9180 + 1000) / 1000 = 10.18
        self.assertAlmostEqual(micro_p2[0], 10.18, places=3)
        self.assertGreater(bias_bp2[0], 0.0)
        self.assertLessEqual(micro_p2[0], ask_p[0])

    def test_depth_imbalance_bounds(self):
        """测试五档盘口深度不平衡度处于 [-1, 1]"""
        bid_sizes = np.array([[100, 100, 100, 100, 100], [500, 500, 500, 500, 500]])
        ask_sizes = np.array([[100, 100, 100, 100, 100], [0, 0, 0, 0, 0]])

        bdi = self.analyzer.calculate_depth_imbalance(bid_sizes, ask_sizes)
        self.assertEqual(len(bdi), 2)
        self.assertAlmostEqual(bdi[0], 0.0, places=3)
        self.assertAlmostEqual(bdi[1], 1.0, places=3)
        self.assertTrue(np.all(bdi >= -1.0) and np.all(bdi <= 1.0))

    def test_vpin_calculation(self):
        """测试 Easley et al. (2012) VPIN 毒性概率与数值边界"""
        np.random.seed(42)
        n = 80
        prices = 100.0 + np.cumsum(np.random.normal(0, 0.1, n))
        volumes = np.random.uniform(50, 150, n)

        vpin = self.analyzer.calculate_vpin(prices, volumes, bucket_size=100, n_buckets=10)
        self.assertEqual(len(vpin), n)
        self.assertTrue(np.all(vpin >= 0.0) and np.all(vpin <= 1.0))

    def test_roll_spread_non_negative(self):
        """测试 Roll (1984) 有效价差非负性"""
        np.random.seed(42)
        n = 50
        prices = 50.0 + np.cumsum(np.random.normal(0, 0.05, n))
        spread = self.analyzer.calculate_roll_spread(prices, window=10)
        self.assertEqual(len(spread), n)
        self.assertTrue(np.all(spread >= 0.0))

    def test_full_analysis_workflow(self):
        """测试微观结构与订单流全量分析流程"""
        result = self.analyzer.analyze(symbol="sh.600519", base_price=1800.0, n_ticks=100)

        self.assertIn("symbol", result)
        self.assertEqual(result["symbol"], "sh.600519")

        summary = result["summary"]
        self.assertIn("sentiment_score", summary)
        self.assertIn("sentiment_label", summary)
        self.assertIn("latest_price", summary)
        self.assertIn("latest_micro_price", summary)
        self.assertIn("latest_ofi", summary)
        self.assertIn("latest_vpin", summary)
        self.assertIn("vpin_level", summary)

        evals = result["factor_evaluations"]
        self.assertEqual(len(evals), len(MICROSTRUCTURE_FACTOR_DEFINITIONS))
        factor_ids = [e["id"] for e in evals]
        self.assertIn("ofi", factor_ids)
        self.assertIn("vpin", factor_ids)
        self.assertIn("micro_price_bias", factor_ids)
        self.assertIn("depth_imbalance_5", factor_ids)
        self.assertIn("active_buy_ratio", factor_ids)
        self.assertIn("roll_spread", factor_ids)

        ts = result["time_series"]
        self.assertEqual(len(ts), 100)
        first = ts[0]
        self.assertIn("time", first)
        self.assertIn("price", first)
        self.assertIn("micro_price", first)
        self.assertIn("ofi", first)
        self.assertIn("vpin", first)


if __name__ == "__main__":
    unittest.main()
