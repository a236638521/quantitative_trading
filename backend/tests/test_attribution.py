import unittest
import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
from backend.app.core.attribution import PerformanceAttributor, DEFAULT_SECTOR_MAP


class TestPerformanceAttribution(unittest.TestCase):
    """测试 Brinson 业绩归因与多因子风格暴露分析引擎"""

    def setUp(self):
        np.random.seed(42)
        self.trading_dates = pd.date_range("2023-01-01", periods=80, freq="B").strftime("%Y-%m-%d").tolist()
        self.symbols = ["sh.600519", "sz.000001", "sz.300750", "sh.518880"]

        # 模拟 4 只标的的 K 线数据
        self.data_dict = {}
        for sym in self.symbols:
            # 制造不同漂移与波动的随机行走
            mu = np.random.uniform(-0.0005, 0.0015)
            sigma = np.random.uniform(0.01, 0.02)
            rets = np.random.normal(mu, sigma, len(self.trading_dates))
            prices = 100.0 * np.exp(np.cumsum(rets))
            df = pd.DataFrame({
                "date": self.trading_dates,
                "close": prices,
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.98,
                "volume": 100000
            })
            self.data_dict[sym] = df

        # 模拟基准行情与曲线 (沪深300)
        bmk_rets = np.random.normal(0.0002, 0.012, len(self.trading_dates))
        bmk_prices = 1.0 * np.exp(np.cumsum(bmk_rets))
        self.benchmark_curve = [{"date": d, "value": float(p)} for d, p in zip(self.trading_dates, bmk_prices)]

        # 模拟策略累计净值曲线 (带有一定超额 Alpha)
        strat_rets = bmk_rets + np.random.normal(0.0004, 0.005, len(self.trading_dates))
        strat_prices = 1.0 * np.exp(np.cumsum(strat_rets))
        self.equity_curve = [{"date": d, "value": float(p)} for d, p in zip(self.trading_dates, strat_prices)]

        # 模拟持仓分配历史 (周度动态调整权重)
        self.allocation_history = []
        for i, d in enumerate(self.trading_dates):
            if i % 5 == 0:
                raw_w = np.random.uniform(0.1, 0.4, len(self.symbols))
                norm_w = raw_w / raw_w.sum()
                weights = {sym: round(float(w), 4) for sym, w in zip(self.symbols, norm_w)}
                self.allocation_history.append({"date": d, "weights": weights})

        # 模拟几笔调仓交易
        self.trades = [
            {"date": self.trading_dates[0], "symbol": "sh.600519", "action": "BUY", "shares": 300, "commission": 15.0, "stamp_tax": 0.0},
            {"date": self.trading_dates[10], "symbol": "sz.000001", "action": "SELL", "shares": 500, "commission": 12.0, "stamp_tax": 25.0}
        ]

    def test_brinson_attribution_and_identity(self):
        attributor = PerformanceAttributor()
        result = attributor.analyze(
            allocation_history=self.allocation_history,
            data_dict=self.data_dict,
            trading_dates=self.trading_dates,
            benchmark_curve=self.benchmark_curve,
            equity_curve=self.equity_curve,
            trades=self.trades
        )

        self.assertIn("summary", result)
        self.assertIn("sector_attribution", result)
        self.assertIn("cumulative_curves", result)
        self.assertIn("waterfall", result)
        self.assertIn("style_exposure", result)

        summary = result["summary"]
        tot_strat = summary["total_strategy_return"]
        tot_bmk = summary["total_benchmark_return"]
        tot_excess = summary["total_excess_return"]

        # 验证超额收益基本恒等
        self.assertAlmostEqual(tot_strat - tot_bmk, tot_excess, places=3)

        # 验证 Brinson 效应之和与总超额高度贴合
        effects_sum = summary["allocation_effect"] + summary["selection_effect"] + summary["interaction_effect"]
        self.assertAlmostEqual(effects_sum, tot_excess, delta=0.03)

        # 验证行业归因表
        sector_table = result["sector_attribution"]
        self.assertIsInstance(sector_table, list)
        self.assertGreater(len(sector_table), 0)
        for row in sector_table:
            self.assertIn("sector", row)
            self.assertIn("allocation_effect_pct", row)
            self.assertIn("selection_effect_pct", row)
            self.assertIn("interaction_effect_pct", row)
            self.assertIn("total_contribution_pct", row)

        # 验证瀑布图数据
        waterfall = result["waterfall"]
        self.assertEqual(len(waterfall), 6)
        self.assertEqual(waterfall[0]["name"], "沪深300 基准收益")
        self.assertEqual(waterfall[-1]["name"], "策略最终实现净收益")

        # 验证 Barra 风格雷达
        style = result["style_exposure"]
        self.assertEqual(len(style["indicators"]), 6)
        self.assertEqual(len(style["strategy_scores"]), 6)
        self.assertEqual(len(style["benchmark_scores"]), 6)
        self.assertIn("diagnosis", style)
        self.assertIn("style_label", style["diagnosis"])
        self.assertIn("analysis_text", style["diagnosis"])

        print("\n=== PerformanceAttributor Unit Test Output ===")
        print(f"Strategy Return: {summary['total_strategy_return']*100:.2f}% | Benchmark Return: {summary['total_benchmark_return']*100:.2f}% | Excess: {summary['total_excess_return']*100:.2f}%")
        print(f"Allocation Effect: {summary['allocation_effect']*100:.2f}%")
        print(f"Selection Effect: {summary['selection_effect']*100:.2f}%")
        print(f"Interaction Effect: {summary['interaction_effect']*100:.2f}%")
        print(f"Cost Drag: {summary['cost_drag']*100:.2f}%")
        print(f"Style Diagnosis: {style['diagnosis']['style_label']}")
        print(f"Diagnosis Detail: {style['diagnosis']['analysis_text']}")


if __name__ == "__main__":
    unittest.main()
