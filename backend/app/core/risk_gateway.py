import logging
import math
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# 预设大类行业映射表 (兼容股票代码与ETF)
DEFAULT_SECTOR_MAP = {
    "sh.600519": "大消费",
    "sz.000858": "大消费",
    "sz.000568": "大消费",
    "sh.600809": "大消费",
    "sz.000001": "大金融",
    "sh.600036": "大金融",
    "sh.601318": "大金融",
    "sh.600030": "大金融",
    "sz.300750": "先进制造",
    "sz.002594": "先进制造",
    "sh.688981": "硬核科创",
    "sz.300059": "科技信息",
    "sh.601012": "先进制造",
    "sh.600900": "公用红利",
    "sh.601088": "周期资源",
    "sh.510300": "大盘宽基",
    "sh.510500": "中盘宽基",
    "sz.159915": "成长创业",
    "sh.588000": "硬核科创",
    "sh.518880": "大宗商品",
    "sh.511010": "固定收益",
}

class RiskRuleId:
    MAX_SINGLE_ORDER = "RULE_01_SINGLE_ORDER_AMOUNT"       # 单笔最大申报金额限制
    MAX_DAILY_TURNOVER = "RULE_02_DAILY_TURNOVER_LIMIT"     # 单日累计交易额限制
    STOCK_CONCENTRATION = "RULE_03_STOCK_CONCENTRATION"     # 个股最大持仓占比红线
    SECTOR_EXPOSURE = "RULE_04_SECTOR_EXPOSURE_CAP"         # 单一行业最大配置敞口
    CANCEL_FREQUENCY = "RULE_05_CANCEL_ORDER_FREQUENCY"     # 盘中高频报撤单防幌骗
    LIQUIDITY_IMPACT = "RULE_06_LIQUIDITY_IMPACT_GUARD"     # 流动性穿透与单笔成交占比
    INTRADAY_DRAWDOWN = "RULE_07_INTRADAY_CIRCUIT_BREAKER"  # 日内动态最大回撤熔断
    SECURITY_COMPLIANCE = "RULE_08_SECURITY_COMPLIANCE"     # ST/退市风险与涨跌停合规


class RiskAction:
    PASS = "PASS"                     # 审查放行
    WARN = "WARN"                     # 风险提示，放行但记录预警
    REJECT = "REJECT"                 # 违规拦截，拒绝报单
    CIRCUIT_BREAK = "CIRCUIT_BREAK"   # 严重违规或触及硬回撤，全系统熔断锁仓


@dataclass
class RiskDecision:
    """风控决策结果"""
    passed: bool
    action: str
    rule_id: Optional[str] = None
    reason: str = "风控审查通过"
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "action": self.action,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "details": self.details,
            "timestamp": self.timestamp
        }


@dataclass
class RiskGatewayConfig:
    """事前风控网关配置参数"""
    # 1. 单笔申报限额 (元)
    max_single_order_amount: float = 500000.0  # 默认 50 万元
    # 2. 单日累计交易额上限 (元)
    max_daily_turnover: float = 3000000.0      # 默认 300 万元
    # 3. 单标的最大持仓权重比例 (0.0 ~ 1.0)
    max_stock_concentration: float = 0.30      # 默认 30%
    # 4. 单行业最大持仓敞口比例 (0.0 ~ 1.0)
    max_sector_exposure: float = 0.50          # 默认 50%
    # 5. 撤单频率监控窗口与上限
    cancel_window_sec: int = 60                # 监控窗口 (秒)
    max_cancel_count: int = 5                  # 窗口内最大允许撤单次数
    # 6. 流动性冲击警戒线 (单笔申报量占20日日均量的比例)
    max_volume_ratio: float = 0.15             # 默认 15%
    # 7. 日内回撤预警与熔断线 (负百分比，例如 -0.025 代表 -2.5%)
    intraday_warn_drawdown: float = -0.025     # 默认 -2.5%
    intraday_stop_drawdown: float = -0.040     # 默认 -4.0%
    # 8. 标的黑名单 (禁止买入)
    blacklist_symbols: Set[str] = field(default_factory=set)
    # 9. 涨跌停限制开关
    enforce_price_limits: bool = True
    # 10. 网关总开关
    is_active: bool = True


class PreTradeRiskGateway:
    """
    实盘量化事前风控网关与合规拦截器 (Pre-Trade Risk Control Gateway)
    在所有交易指令发送至撮合引擎或交易所前，提供毫秒级全规则穿透审查。
    """

    def __init__(self, config: Optional[RiskGatewayConfig] = None, sector_map: Optional[Dict[str, str]] = None):
        self.config = config or RiskGatewayConfig()
        self.sector_map = dict(DEFAULT_SECTOR_MAP)
        if sector_map:
            self.sector_map.update(sector_map)

        # 运行时日内动态追踪状态
        self.current_trading_date: Optional[str] = None
        self.daily_buy_amount: float = 0.0
        self.daily_sell_amount: float = 0.0
        self.intraday_high_equity: float = 0.0
        self.current_equity: float = 0.0
        self.is_circuit_broken: bool = False
        self.circuit_break_reason: Optional[str] = None

        # 撤单时间戳时序列表 {symbol: [timestamp_epoch]}
        self.cancel_history: Dict[str, List[float]] = {}

        # 风控审计与拦截日志流水
        self.audit_logs: List[Dict[str, Any]] = []

    def _resolve_sector(self, symbol: str) -> str:
        """解析标的所属行业大类"""
        if symbol in self.sector_map:
            return self.sector_map[symbol]
        clean = symbol.replace("sh.", "").replace("sz.", "")
        if clean.startswith("51") or clean.startswith("15"):
            return "大盘宽基"
        elif clean.startswith("600") or clean.startswith("601"):
            return "大消费"
        elif clean.startswith("300") or clean.startswith("688"):
            return "先进制造"
        return "综合板块"

    def reset_day(self, trading_date: str, initial_day_equity: float):
        """新交易日开盘状态重置"""
        self.current_trading_date = trading_date
        self.daily_buy_amount = 0.0
        self.daily_sell_amount = 0.0
        self.intraday_high_equity = max(100.0, initial_day_equity)
        self.current_equity = initial_day_equity
        self.is_circuit_broken = False
        self.circuit_break_reason = None
        self.cancel_history.clear()
        logger.info(f"[风控网关] 交易日 {trading_date} 开盘状态初始化，日初资产净值: ¥{initial_day_equity:,.2f}")

    def update_intraday_equity(self, current_equity: float) -> Optional[RiskDecision]:
        """盘中更新组合动态净值，检测日内最大回撤熔断"""
        self.current_equity = current_equity
        if current_equity > self.intraday_high_equity:
            self.intraday_high_equity = current_equity

        if self.intraday_high_equity <= 0:
            return None

        intraday_dd = (current_equity - self.intraday_high_equity) / self.intraday_high_equity

        # 硬熔断检查 (-4.0%)
        if intraday_dd <= self.config.intraday_stop_drawdown:
            if not self.is_circuit_broken:
                self.is_circuit_broken = True
                self.circuit_break_reason = f"日内动态回撤达到 {intraday_dd * 100:.2f}%，触及硬熔断红线 {self.config.intraday_stop_drawdown * 100:.2f}%"
                decision = RiskDecision(
                    passed=False,
                    action=RiskAction.CIRCUIT_BREAK,
                    rule_id=RiskRuleId.INTRADAY_DRAWDOWN,
                    reason=self.circuit_break_reason,
                    details={
                        "current_equity": round(current_equity, 2),
                        "high_equity": round(self.intraday_high_equity, 2),
                        "intraday_drawdown_pct": round(intraday_dd * 100, 2),
                        "threshold_pct": round(self.config.intraday_stop_drawdown * 100, 2)
                    }
                )
                self._record_log(decision, "ALL", "CIRCUIT_BREAKER", 0, 0.0)
                logger.error(f"[风控熔断] 🚨 触发硬熔断锁仓！原因: {self.circuit_break_reason}")
                return decision

        # 预警检查 (-2.5%)
        elif intraday_dd <= self.config.intraday_warn_drawdown:
            warn_msg = f"日内动态回撤达到 {intraday_dd * 100:.2f}%，触及预警线 {self.config.intraday_warn_drawdown * 100:.2f}%"
            decision = RiskDecision(
                passed=True,
                action=RiskAction.WARN,
                rule_id=RiskRuleId.INTRADAY_DRAWDOWN,
                reason=warn_msg,
                details={
                    "current_equity": round(current_equity, 2),
                    "high_equity": round(self.intraday_high_equity, 2),
                    "intraday_drawdown_pct": round(intraday_dd * 100, 2)
                }
            )
            self._record_log(decision, "ALL", "INTRADAY_WARN", 0, 0.0)
            logger.warning(f"[风控预警] ⚠️ {warn_msg}")
            return decision

        return None

    def record_cancel_order(self, symbol: str, timestamp_epoch: Optional[float] = None) -> RiskDecision:
        """登记撤单事件，实施盘中防幌骗高频监控"""
        now = timestamp_epoch if timestamp_epoch is not None else time.time()
        if symbol not in self.cancel_history:
            self.cancel_history[symbol] = []

        self.cancel_history[symbol].append(now)

        # 清除窗口外的时间戳
        cutoff = now - self.config.cancel_window_sec
        self.cancel_history[symbol] = [t for t in self.cancel_history[symbol] if t >= cutoff]

        cancel_cnt = len(self.cancel_history[symbol])
        if cancel_cnt >= self.config.max_cancel_count:
            reason = f"标的 {symbol} 在过去 {self.config.cancel_window_sec} 秒内撤单已达 {cancel_cnt} 次，超过监管上限 {self.config.max_cancel_count} 次"
            decision = RiskDecision(
                passed=False,
                action=RiskAction.REJECT,
                rule_id=RiskRuleId.CANCEL_FREQUENCY,
                reason=reason,
                details={
                    "symbol": symbol,
                    "window_sec": self.config.cancel_window_sec,
                    "cancel_count": cancel_cnt,
                    "limit_count": self.config.max_cancel_count
                }
            )
            self._record_log(decision, symbol, "CANCEL_REJECT", 0, 0.0)
            logger.warning(f"[防幌骗监控] ❌ {reason}")
            return decision

        return RiskDecision(passed=True, action=RiskAction.PASS, reason="撤单登记成功")

    def record_executed_trade(self, action: str, symbol: str, shares: int, price: float):
        """登记已撮合成交的交易，累加交易额度统计"""
        val = shares * price
        if action.upper() == "BUY":
            self.daily_buy_amount += val
        elif action.upper() == "SELL":
            self.daily_sell_amount += val

    def validate_order(
        self,
        order_type: str,            # "BUY" 或 "SELL"
        symbol: str,
        shares: int,
        price: float,
        portfolio_total_asset: float,
        current_holdings: Dict[str, float],   # {symbol: market_value}
        market_bar: Optional[Dict[str, Any]] = None,
        avg_20d_volume: Optional[float] = None,
        timestamp_epoch: Optional[float] = None
    ) -> RiskDecision:
        """
        穿透审查单笔报单，执行全套事前风控规则链检验。
        全部通过返回 PASS，任一违规返回 REJECT 或 CIRCUIT_BREAK。
        """
        if not self.config.is_active:
            return RiskDecision(passed=True, action=RiskAction.PASS, reason="风控网关已全局旁路关闭")

        order_type = order_type.upper()
        order_amount = shares * price

        # 0. 全局硬熔断检查
        if self.is_circuit_broken and order_type == "BUY":
            decision = RiskDecision(
                passed=False,
                action=RiskAction.CIRCUIT_BREAK,
                rule_id=RiskRuleId.INTRADAY_DRAWDOWN,
                reason=f"网关处于硬熔断状态，禁止任何买入开仓：{self.circuit_break_reason}",
                details={"circuit_break_reason": self.circuit_break_reason}
            )
            self._record_log(decision, symbol, order_type, shares, price)
            return decision

        # 1. 黑名单合规检查 (仅限买入)
        clean_sym = symbol.replace("sh.", "").replace("sz.", "")
        if order_type == "BUY":
            if symbol in self.config.blacklist_symbols or clean_sym in self.config.blacklist_symbols:
                decision = RiskDecision(
                    passed=False,
                    action=RiskAction.REJECT,
                    rule_id=RiskRuleId.SECURITY_COMPLIANCE,
                    reason=f"标的 {symbol} 被列入交易风控黑名单，禁止买入",
                    details={"symbol": symbol}
                )
                self._record_log(decision, symbol, order_type, shares, price)
                return decision

        # 2. 标的行情合规检查 (涨跌停 / 停牌)
        if market_bar and self.config.enforce_price_limits:
            is_suspended = bool(market_bar.get("is_suspended", False))
            if is_suspended:
                decision = RiskDecision(
                    passed=False,
                    action=RiskAction.REJECT,
                    rule_id=RiskRuleId.SECURITY_COMPLIANCE,
                    reason=f"标的 {symbol} 处于停牌中，禁止申报委托",
                    details={"symbol": symbol, "is_suspended": True}
                )
                self._record_log(decision, symbol, order_type, shares, price)
                return decision

            if order_type == "BUY" and bool(market_bar.get("is_up_limit", False)):
                decision = RiskDecision(
                    passed=False,
                    action=RiskAction.REJECT,
                    rule_id=RiskRuleId.SECURITY_COMPLIANCE,
                    reason=f"标的 {symbol} 涨停封死，申报被拒 (规避追涨挂单资金冻结)",
                    details={"symbol": symbol, "is_up_limit": True}
                )
                self._record_log(decision, symbol, order_type, shares, price)
                return decision

            if order_type == "SELL" and bool(market_bar.get("is_down_limit", False)):
                decision = RiskDecision(
                    passed=False,
                    action=RiskAction.REJECT,
                    rule_id=RiskRuleId.SECURITY_COMPLIANCE,
                    reason=f"标的 {symbol} 跌停封死，无法在市场中实现流动性成交",
                    details={"symbol": symbol, "is_down_limit": True}
                )
                self._record_log(decision, symbol, order_type, shares, price)
                return decision

        # 3. 报撤单防幌骗频率监控 (仅限买入)
        if order_type == "BUY" and symbol in self.cancel_history:
            now = timestamp_epoch if timestamp_epoch is not None else time.time()
            cutoff = now - self.config.cancel_window_sec
            recent_cancels = [t for t in self.cancel_history[symbol] if t >= cutoff]
            if len(recent_cancels) >= self.config.max_cancel_count:
                decision = RiskDecision(
                    passed=False,
                    action=RiskAction.REJECT,
                    rule_id=RiskRuleId.CANCEL_FREQUENCY,
                    reason=f"标的 {symbol} 频发撤单达到 {len(recent_cancels)} 次，触发防幌骗交易拦截保护",
                    details={"cancel_count": len(recent_cancels), "limit": self.config.max_cancel_count}
                )
                self._record_log(decision, symbol, order_type, shares, price)
                return decision

        # 4. 单笔最大申报金额限制 (防胖手指)
        if order_amount > self.config.max_single_order_amount:
            decision = RiskDecision(
                passed=False,
                action=RiskAction.REJECT,
                rule_id=RiskRuleId.MAX_SINGLE_ORDER,
                reason=f"单笔委托金额 ¥{order_amount:,.2f} 超过硬上限 ¥{self.config.max_single_order_amount:,.2f}，触发防乌龙指拦截",
                details={
                    "order_amount": round(order_amount, 2),
                    "max_limit": round(self.config.max_single_order_amount, 2),
                    "shares": shares,
                    "price": price
                }
            )
            self._record_log(decision, symbol, order_type, shares, price)
            return decision

        # 5. 单日累计交易额上限限制
        est_total_daily = (self.daily_buy_amount + self.daily_sell_amount) + order_amount
        if est_total_daily > self.config.max_daily_turnover:
            decision = RiskDecision(
                passed=False,
                action=RiskAction.REJECT,
                rule_id=RiskRuleId.MAX_DAILY_TURNOVER,
                reason=f"累计交易额将达到 ¥{est_total_daily:,.2f}，超过单日换手率限额 ¥{self.config.max_daily_turnover:,.2f}",
                details={
                    "current_daily_turnover": round(self.daily_buy_amount + self.daily_sell_amount, 2),
                    "order_amount": round(order_amount, 2),
                    "projected_daily": round(est_total_daily, 2),
                    "max_daily_turnover": round(self.config.max_daily_turnover, 2)
                }
            )
            self._record_log(decision, symbol, order_type, shares, price)
            return decision

        # 6. 流动性冲击与申报股数占比限制
        if avg_20d_volume and avg_20d_volume > 0:
            vol_ratio = shares / avg_20d_volume
            if vol_ratio > self.config.max_volume_ratio:
                decision = RiskDecision(
                    passed=False,
                    action=RiskAction.REJECT,
                    rule_id=RiskRuleId.LIQUIDITY_IMPACT,
                    reason=f"申报股数 {shares:,} 股占 20 日日均成交量比例达 {vol_ratio * 100:.1f}%，超过流动性冲击红线 {self.config.max_volume_ratio * 100:.1f}%",
                    details={
                        "shares": shares,
                        "avg_20d_volume": int(avg_20d_volume),
                        "ratio_pct": round(vol_ratio * 100, 2),
                        "limit_pct": round(self.config.max_volume_ratio * 100, 2)
                    }
                )
                self._record_log(decision, symbol, order_type, shares, price)
                return decision

        # 7. 个股最大持仓集中度红线限制 (仅限买入加仓)
        if order_type == "BUY" and portfolio_total_asset > 0:
            current_pos_val = current_holdings.get(symbol, 0.0)
            proj_pos_val = current_pos_val + order_amount
            proj_stock_weight = proj_pos_val / portfolio_total_asset

            if proj_stock_weight > self.config.max_stock_concentration:
                decision = RiskDecision(
                    passed=False,
                    action=RiskAction.REJECT,
                    rule_id=RiskRuleId.STOCK_CONCENTRATION,
                    reason=f"买入后标的 {symbol} 集中度将达到 {proj_stock_weight * 100:.1f}%，超过个股持仓上限 {self.config.max_stock_concentration * 100:.1f}%",
                    details={
                        "symbol": symbol,
                        "current_pos_val": round(current_pos_val, 2),
                        "projected_pos_val": round(proj_pos_val, 2),
                        "projected_weight_pct": round(proj_stock_weight * 100, 2),
                        "max_weight_pct": round(self.config.max_stock_concentration * 100, 2)
                    }
                )
                self._record_log(decision, symbol, order_type, shares, price)
                return decision

        # 8. 单一行业最大配置敞口红线限制 (仅限买入加仓)
        if order_type == "BUY" and portfolio_total_asset > 0:
            target_sector = self._resolve_sector(symbol)
            # 计算当前属于该行业的所有资产市值总和
            current_sector_val = 0.0
            for sym, val in current_holdings.items():
                if self._resolve_sector(sym) == target_sector:
                    current_sector_val += val

            proj_sector_val = current_sector_val + order_amount
            proj_sector_weight = proj_sector_val / portfolio_total_asset

            if proj_sector_weight > self.config.max_sector_exposure:
                decision = RiskDecision(
                    passed=False,
                    action=RiskAction.REJECT,
                    rule_id=RiskRuleId.SECTOR_EXPOSURE,
                    reason=f"买入后【{target_sector}】板块整体敞口将达到 {proj_sector_weight * 100:.1f}%，超过行业上限 {self.config.max_sector_exposure * 100:.1f}%",
                    details={
                        "sector": target_sector,
                        "projected_sector_val": round(proj_sector_val, 2),
                        "projected_sector_weight_pct": round(proj_sector_weight * 100, 2),
                        "max_sector_weight_pct": round(self.config.max_sector_exposure * 100, 2)
                    }
                )
                self._record_log(decision, symbol, order_type, shares, price)
                return decision

        # 全部通过放行
        decision = RiskDecision(
            passed=True,
            action=RiskAction.PASS,
            reason="全套事前风控合规校验通过，放行撮合",
            details={
                "symbol": symbol,
                "order_type": order_type,
                "shares": shares,
                "order_amount": round(order_amount, 2)
            }
        )
        self._record_log(decision, symbol, order_type, shares, price)
        return decision

    def _record_log(self, decision: RiskDecision, symbol: str, action: str, shares: int, price: float):
        """记录事前风控审计拦截日志"""
        log_entry = {
            "timestamp": decision.timestamp,
            "symbol": symbol,
            "action": action,
            "shares": shares,
            "price": price,
            "amount": round(shares * price, 2),
            "decision": decision.action,
            "passed": decision.passed,
            "rule_id": decision.rule_id,
            "reason": decision.reason,
            "details": decision.details
        }
        self.audit_logs.append(log_entry)
        if len(self.audit_logs) > 500:
            self.audit_logs = self.audit_logs[-500:]

    def get_audit_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取最近的风控拦截审计流水"""
        return self.audit_logs[-limit:]

    def get_status_summary(self) -> Dict[str, Any]:
        """获取当前风控网关的运行状态与红线水位监测"""
        total_turnover = self.daily_buy_amount + self.daily_sell_amount
        turnover_usage_pct = (total_turnover / self.config.max_daily_turnover) * 100 if self.config.max_daily_turnover > 0 else 0.0

        intraday_dd = 0.0
        if self.intraday_high_equity > 0:
            intraday_dd = (self.current_equity - self.intraday_high_equity) / self.intraday_high_equity

        return {
            "is_active": self.config.is_active,
            "is_circuit_broken": self.is_circuit_broken,
            "circuit_break_reason": self.circuit_break_reason,
            "current_trading_date": self.current_trading_date,
            "daily_turnover": {
                "daily_buy_amount": round(self.daily_buy_amount, 2),
                "daily_sell_amount": round(self.daily_sell_amount, 2),
                "total_turnover": round(total_turnover, 2),
                "max_limit": round(self.config.max_daily_turnover, 2),
                "usage_pct": round(turnover_usage_pct, 1)
            },
            "intraday_drawdown": {
                "current_equity": round(self.current_equity, 2),
                "high_equity": round(self.intraday_high_equity, 2),
                "drawdown_pct": round(intraday_dd * 100, 2),
                "warn_threshold_pct": round(self.config.intraday_warn_drawdown * 100, 2),
                "stop_threshold_pct": round(self.config.intraday_stop_drawdown * 100, 2)
            },
            "limits": {
                "max_single_order_amount": round(self.config.max_single_order_amount, 2),
                "max_stock_concentration_pct": round(self.config.max_stock_concentration * 100, 1),
                "max_sector_exposure_pct": round(self.config.max_sector_exposure * 100, 1),
                "cancel_frequency_limit": f"{self.config.max_cancel_count} 次 / {self.config.cancel_window_sec} 秒",
                "max_volume_ratio_pct": round(self.config.max_volume_ratio * 100, 1)
            },
            "intercepted_count": sum(1 for log in self.audit_logs if not log["passed"]),
            "total_validated_count": len(self.audit_logs)
        }
