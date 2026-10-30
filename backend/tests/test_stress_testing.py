import unittest
import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
from backend.app.core.stress_testing import StressTestEngine, HISTORICAL_SCENARIOS


class TestStressTestingEngine(unittest.TestCase):
    """测试投资组合压力测试与黑天鹅极端情景模拟引擎"""

    def setUp(self):
        np.random.seed(42)
        self.trading_dates = pd.date_range("2023-01-01", periods=100, freq="B").strftime("%Y-%m-%d").tolist()
        self.symbols = ["sh.600519", "sz.000001", "sz.300750", "sh.518880"]

        # 构造测试行情数据
        self.data_dict = {}
        for sym in self.symbols:
            mu = 0.0002
            sigma = 0.018
            rets = np.random.normal(mu, sigma, len(self.trading_dates))
            prices = 100.0 * np.exp(np.cumsum(rets))
            df = pd.DataFrame({
                "date": self.trading_dates,
                "close": prices,
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.98,
                "volume": 200000,
                "amount": 200000 * prices
            })
            self.data_dict[sym] = df

        self.portfolio_weights = {
            "sh.600519": 0.35,
            "sz.000001": 0.25,
            "sz.300750": 0.25,
            "sh.518880": 0.15
        }
        self.engine = StressTestEngine()

    def test_var_cvar_mathematical_properties(self):
        """测试 VaR 与 CVaR 的统计公理：CVaR >= VaR, VaR(99%) >= VaR(95%)"""
        result = self.engine.run_stress_test(
            weights=self.portfolio_weights,
            data_dict=self.data_dict,
            trading_dates=self.trading_dates
        )

        var_cvar = result.get("var_cvar", {})
        self.assertIn("var_95_hist", var_cvar)
        self.assertIn("var_99_hist", var_cvar)
        self.assertIn("cvar_95_hist", var_cvar)
        self.assertIn("cvar_99_hist", var_cvar)

        hist_var_95 = var_cvar["var_95_hist"]
        hist_var_99 = var_cvar["var_99_hist"]
        hist_cvar_95 = var_cvar["cvar_95_hist"]
        hist_cvar_99 = var_cvar["cvar_99_hist"]

        # 检验历史模拟法统计单调性
        self.assertGreaterEqual(hist_var_99, hist_var_95, "VaR 99% 应大于等于 VaR 95%")
        self.assertGreaterEqual(hist_cvar_95, hist_var_95, "CVaR 95% 应大于等于 VaR 95%")
        self.assertGreaterEqual(hist_cvar_99, hist_var_99, "CVaR 99% 应大于等于 VaR 99%")

        # 参数法 (Cornish-Fisher)
        param_var_95 = var_cvar["var_95_param"]
        param_var_99 = var_cvar["var_99_param"]
        self.assertGreaterEqual(param_var_99, param_var_95, "参数法 VaR 99% 应大于等于 VaR 95%")

        # 检查核密度直方图切片
        pnl_dist = var_cvar.get("pnl_distribution", [])
        self.assertGreater(len(pnl_dist), 0)
        self.assertIn("bin_mid", pnl_dist[0])
        self.assertIn("count", pnl_dist[0])

    def test_historical_scenarios_replay(self):
        """测试五大历史极端黑天鹅危机情景重放"""
        result = self.engine.run_stress_test(
            weights=self.portfolio_weights,
            data_dict=self.data_dict,
            trading_dates=self.trading_dates
        )

        scenarios = result.get("historical_scenarios", [])
        self.assertEqual(len(scenarios), len(HISTORICAL_SCENARIOS))

        scenario_ids = [s["id"] for s in scenarios]
        self.assertIn("crash_2015", scenario_ids)
        self.assertIn("circuit_breaker_2016", scenario_ids)
        self.assertIn("trade_war_2018", scenario_ids)
        self.assertIn("covid_flash_2020", scenario_ids)
        self.assertIn("microcap_squeeze_2024", scenario_ids)

        for s in scenarios:
            self.assertIn("portfolio_loss_pct", s)
            self.assertIn("bmk_drawdown_pct", s)
            self.assertIn("relative_defensive_alpha_pct", s)
            # 历史情景冲击在组合中通常为负
            self.assertLessEqual(s["portfolio_loss_pct"], 0.0)

    def test_hypothetical_macro_shocks(self):
        """测试四大假设性宏观风险冲击"""
        result = self.engine.run_stress_test(
            weights=self.portfolio_weights,
            data_dict=self.data_dict,
            trading_dates=self.trading_dates
        )

        macro_shocks = result.get("hypothetical_shocks", [])
        self.assertEqual(len(macro_shocks), 4)

        shock_names = [s["shock_name"] for s in macro_shocks]
        self.assertTrue(any("闪崩" in name for name in shock_names))
        self.assertTrue(any("跌停" in name for name in shock_names))
        self.assertTrue(any("波动率" in name for name in shock_names))
        self.assertTrue(any("流动性" in name for name in shock_names))

        for s in macro_shocks:
            self.assertIn("est_portfolio_drop_pct", s)
            self.assertIn("severity", s)
            self.assertLessEqual(s["est_portfolio_drop_pct"], 0.0)

    def test_liquidity_and_component_breakdown(self):
        """测试标的损失贡献与流动性平仓天数预估"""
        result = self.engine.run_stress_test(
            weights=self.portfolio_weights,
            data_dict=self.data_dict,
            trading_dates=self.trading_dates
        )

        liquidity = result.get("liquidity_analysis", {})
        self.assertIn("max_liquidation_days", liquidity)
        self.assertGreaterEqual(liquidity["max_liquidation_days"], 0.1)
        self.assertIn("asset_details", liquidity)

        breakdown = result.get("asset_loss_breakdown", [])
        self.assertEqual(len(breakdown), len(self.symbols))

        total_loss_contrib = 0.0
        for comp in breakdown:
            self.assertIn("symbol", comp)
            self.assertIn("weight_pct", comp)
            self.assertIn("loss_contribution_ratio", comp)
            total_loss_contrib += comp["loss_contribution_ratio"]

        self.assertAlmostEqual(total_loss_contrib, 100.0, delta=1.5)

    def test_rating_and_diagnosis(self):
        """测试综合抗压评级与诊断建议"""
        result = self.engine.run_stress_test(
            weights=self.portfolio_weights,
            data_dict=self.data_dict,
            trading_dates=self.trading_dates
        )

        summary = result.get("summary", {})
        self.assertIn("stress_resilience_grade", summary)

        scorecard = result.get("diagnostic_scorecard", {})
        self.assertIn("grade", scorecard)
        self.assertIn("grade_color", scorecard)
        self.assertIn("diagnosis_text", scorecard)
        self.assertIn("worst_scenario_name", scorecard)
        self.assertIn("worst_scenario_loss", scorecard)



if __name__ == "__main__":
    unittest.main()
