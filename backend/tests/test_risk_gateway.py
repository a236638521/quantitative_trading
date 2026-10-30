import unittest
import sys
import os
import time
sys.path.insert(0, os.path.abspath("."))

from backend.app.core.risk_gateway import (
    PreTradeRiskGateway,
    RiskGatewayConfig,
    RiskDecision,
    RiskRuleId,
    RiskAction
)


class TestPreTradeRiskGateway(unittest.TestCase):
    """测试实盘交易事前风控网关与合规拦截器"""

    def setUp(self):
        self.config = RiskGatewayConfig(
            max_single_order_amount=500000.0,
            max_daily_turnover=2000000.0,
            max_stock_concentration=0.30,
            max_sector_exposure=0.50,
            cancel_window_sec=60,
            max_cancel_count=5,
            max_volume_ratio=0.15,
            intraday_warn_drawdown=-0.025,
            intraday_stop_drawdown=-0.040,
            blacklist_symbols={"sh.600000", "sz.000002"},
            enforce_price_limits=True,
            is_active=True
        )
        self.gateway = PreTradeRiskGateway(config=self.config)
        self.gateway.reset_day(trading_date="2023-05-15", initial_day_equity=1000000.0)

    def test_pass_normal_compliant_order(self):
        """测试正常合规订单顺利通过风控审查"""
        decision = self.gateway.validate_order(
            order_type="BUY",
            symbol="sh.600519",
            shares=100,
            price=1800.0,  # 18 万元 < 50 万元
            portfolio_total_asset=1000000.0,
            current_holdings={},
            avg_20d_volume=500000.0
        )
        self.assertTrue(decision.passed)
        self.assertEqual(decision.action, RiskAction.PASS)
        self.assertIn("通过", decision.reason)

    def test_single_order_amount_limit(self):
        """测试单笔申报金额超限拦截 (防乌龙指 / 胖手指)"""
        decision = self.gateway.validate_order(
            order_type="BUY",
            symbol="sh.600519",
            shares=300,
            price=2000.0,  # 60 万元 > 50 万元
            portfolio_total_asset=10000000.0,
            current_holdings={}
        )
        self.assertFalse(decision.passed)
        self.assertEqual(decision.action, RiskAction.REJECT)
        self.assertEqual(decision.rule_id, RiskRuleId.MAX_SINGLE_ORDER)
        self.assertIn("防乌龙指拦截", decision.reason)

    def test_daily_turnover_limit(self):
        """测试单日累计交易额上限拦截 (防死循环刷单)"""
        # 先登记已成交 180 万元
        self.gateway.record_executed_trade(action="BUY", symbol="sh.600519", shares=1000, price=1000.0)
        self.gateway.record_executed_trade(action="SELL", symbol="sz.000001", shares=80000, price=10.0)
        self.assertEqual(self.gateway.daily_buy_amount + self.gateway.daily_sell_amount, 1800000.0)

        # 尝试再买入 30 万元 (总计 210 万 > 200 万元上限)
        decision = self.gateway.validate_order(
            order_type="BUY",
            symbol="sz.300750",
            shares=1500,
            price=200.0,  # 30 万元
            portfolio_total_asset=5000000.0,
            current_holdings={}
        )
        self.assertFalse(decision.passed)
        self.assertEqual(decision.action, RiskAction.REJECT)
        self.assertEqual(decision.rule_id, RiskRuleId.MAX_DAILY_TURNOVER)
        self.assertIn("单日换手率限额", decision.reason)

    def test_stock_concentration_limit(self):
        """测试个股持仓市值占比红线拦截 (超过 30%)"""
        # 总资产 100 万元，已有持仓 20 万元，再买 15 万元 -> 35% > 30%
        current_holdings = {"sz.300750": 200000.0}
        decision = self.gateway.validate_order(
            order_type="BUY",
            symbol="sz.300750",
            shares=750,
            price=200.0,  # 15 万元
            portfolio_total_asset=1000000.0,
            current_holdings=current_holdings
        )
        self.assertFalse(decision.passed)
        self.assertEqual(decision.action, RiskAction.REJECT)
        self.assertEqual(decision.rule_id, RiskRuleId.STOCK_CONCENTRATION)
        self.assertIn("个股持仓上限", decision.reason)

    def test_sector_exposure_limit(self):
        """测试单一行业最大配置敞口拦截 (超过 50%)"""
        # 大消费行业已持仓 茅台 20万 + 五粮液 25万 = 45万
        current_holdings = {
            "sh.600519": 200000.0,
            "sz.000858": 250000.0
        }
        # 尝试再买入 泸州老窖 10 万元 -> 大消费 55% > 50%
        decision = self.gateway.validate_order(
            order_type="BUY",
            symbol="sz.000568",
            shares=500,
            price=200.0,  # 10 万元
            portfolio_total_asset=1000000.0,
            current_holdings=current_holdings
        )
        self.assertFalse(decision.passed)
        self.assertEqual(decision.action, RiskAction.REJECT)
        self.assertEqual(decision.rule_id, RiskRuleId.SECTOR_EXPOSURE)
        self.assertIn("大消费", decision.reason)
        self.assertIn("行业上限", decision.reason)

    def test_cancel_frequency_limit(self):
        """测试高频报撤单防幌骗监管拦截"""
        sym = "sz.000001"
        now = time.time()
        for _ in range(5):
            self.gateway.record_cancel_order(sym, timestamp_epoch=now)

        # 尝试买入该标的
        decision = self.gateway.validate_order(
            order_type="BUY",
            symbol=sym,
            shares=1000,
            price=15.0,
            portfolio_total_asset=1000000.0,
            current_holdings={},
            timestamp_epoch=now + 5
        )
        self.assertFalse(decision.passed)
        self.assertEqual(decision.action, RiskAction.REJECT)
        self.assertEqual(decision.rule_id, RiskRuleId.CANCEL_FREQUENCY)
        self.assertIn("防幌骗交易拦截", decision.reason)

    def test_liquidity_impact_limit(self):
        """测试单笔申报占日均成交量比例超限拦截 (超过 15%)"""
        decision = self.gateway.validate_order(
            order_type="BUY",
            symbol="sh.600519",
            shares=20000,
            price=20.0,
            portfolio_total_asset=2000000.0,
            current_holdings={},
            avg_20d_volume=100000.0  # 20,000 / 100,000 = 20% > 15%
        )
        self.assertFalse(decision.passed)
        self.assertEqual(decision.action, RiskAction.REJECT)
        self.assertEqual(decision.rule_id, RiskRuleId.LIQUIDITY_IMPACT)
        self.assertIn("流动性冲击红线", decision.reason)

    def test_intraday_circuit_breaker(self):
        """测试日内最大回撤熔断锁仓 (-4.0% 熔断禁止买入)"""
        # 开盘资产 100 万，盘中最高冲到 100 万，随后跌至 95 万 (回撤 -5% > -4%)
        circuit_decision = self.gateway.update_intraday_equity(current_equity=950000.0)
        self.assertIsNotNone(circuit_decision)
        self.assertTrue(self.gateway.is_circuit_broken)
        self.assertEqual(circuit_decision.action, RiskAction.CIRCUIT_BREAK)

        # 熔断后任何买入均被拒绝
        buy_decision = self.gateway.validate_order(
            order_type="BUY",
            symbol="sh.600519",
            shares=100,
            price=100.0,
            portfolio_total_asset=950000.0,
            current_holdings={}
        )
        self.assertFalse(buy_decision.passed)
        self.assertEqual(buy_decision.action, RiskAction.CIRCUIT_BREAK)
        self.assertIn("硬熔断状态", buy_decision.reason)

    def test_security_compliance_and_blacklist(self):
        """测试黑名单标的与涨跌停停牌拦截"""
        # 1. 黑名单标的
        decision_bl = self.gateway.validate_order(
            order_type="BUY",
            symbol="sh.600000",
            shares=1000,
            price=10.0,
            portfolio_total_asset=1000000.0,
            current_holdings={}
        )
        self.assertFalse(decision_bl.passed)
        self.assertEqual(decision_bl.rule_id, RiskRuleId.SECURITY_COMPLIANCE)
        self.assertIn("黑名单", decision_bl.reason)

        # 2. 停牌标的
        decision_susp = self.gateway.validate_order(
            order_type="BUY",
            symbol="sz.300750",
            shares=100,
            price=100.0,
            portfolio_total_asset=1000000.0,
            current_holdings={},
            market_bar={"is_suspended": True}
        )
        self.assertFalse(decision_susp.passed)
        self.assertIn("停牌中", decision_susp.reason)

        # 3. 涨停无法买入
        decision_up = self.gateway.validate_order(
            order_type="BUY",
            symbol="sz.300750",
            shares=100,
            price=100.0,
            portfolio_total_asset=1000000.0,
            current_holdings={},
            market_bar={"is_up_limit": True}
        )
        self.assertFalse(decision_up.passed)
        self.assertIn("涨停封死", decision_up.reason)

    def test_status_summary_and_audit_logs(self):
        """测试风控网关监控状态与审计日志汇总"""
        self.gateway.validate_order(
            order_type="BUY",
            symbol="sh.600519",
            shares=1000,
            price=1000.0,  # 100 万 > 50 万 -> 拦截
            portfolio_total_asset=2000000.0,
            current_holdings={}
        )
        summary = self.gateway.get_status_summary()
        self.assertIn("daily_turnover", summary)
        self.assertIn("limits", summary)
        self.assertIn("intercepted_count", summary)
        self.assertGreaterEqual(summary["intercepted_count"], 1)

        logs = self.gateway.get_audit_logs(limit=10)
        self.assertGreater(len(logs), 0)
        self.assertIn("rule_id", logs[-1])
        self.assertIn("reason", logs[-1])


if __name__ == "__main__":
    unittest.main()
