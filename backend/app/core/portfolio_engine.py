import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd

from .portfolio import Portfolio
from .cost import AShareCostCalculator
from .metrics import MetricsCalculator
from .risk_gateway import PreTradeRiskGateway
from ..config import BacktestConfig

logger = logging.getLogger(__name__)


class PortfolioBacktestEngine:
    """
    多标的 / 投资组合截面轮动与资产配置回测引擎
    特点：
    1. 严格支持一篮子股票/ETF同时持仓与每日市值估值
    2. 定期截面调仓：先卖后买释放现金，严格遵循 T+1 与每手100股限制
    3. 涨跌停封死与停牌保护：停牌不可调仓，涨停禁买，跌停禁卖
    4. 逐日记录资产配置权重时序（供前端持仓变迁堆叠图渲染）
    5. 集成实盘交易事前风控网关与合规拦截器 (PreTradeRiskGateway)
    """

    def __init__(self, config: Optional[BacktestConfig] = None, risk_gateway: Optional[PreTradeRiskGateway] = None):
        self.config = config or BacktestConfig()
        self.cost_calc = AShareCostCalculator(self.config.cost_config)
        self.metrics_calc = MetricsCalculator()
        self.risk_gateway = risk_gateway
        self.logs: List[str] = []


    def log(self, msg: str):
        self.logs.append(msg)
        logger.info(msg)

    def run(
        self,
        strategy,
        data_dict: Dict[str, pd.DataFrame],
        trading_dates: List[str],
        benchmark_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        执行多标的组合回测
        :param strategy: 多资产轮动策略实例
        :param data_dict: 已对齐的各标的日线字典 {symbol: df}
        :param trading_dates: 完整交易日历
        :param benchmark_df: 对标基准历史行情
        """
        self.logs.clear()
        if not data_dict or not trading_dates:
            raise ValueError("回测数据或交易日历为空")

        # 1. 初始化基准行情字典
        benchmark_dict: Dict[str, float] = {}
        if benchmark_df is not None and not benchmark_df.empty:
            for _, b_row in benchmark_df.iterrows():
                benchmark_dict[str(b_row["date"])] = float(b_row["close"])

        # 2. 策略初始化与前置指标计算
        data_dict = strategy.prepare(data_dict, trading_dates)

        # 3. 初始化多标的账户
        portfolio = Portfolio(
            initial_cash=self.config.initial_cash,
            enable_t_plus_one=self.config.enable_t_plus_one
        )

        allocation_history: List[Dict[str, Any]] = []

        self.log(f"=== 开始多标的组合回测: {strategy.name} 标的池: {list(data_dict.keys())} 初始资金: CNY {self.config.initial_cash:,.2f} ===")

        # 4. 逐交易日模拟推进
        for date_idx, cur_date in enumerate(trading_dates):
            # 4.1 新交易日开盘：T+1 股份解锁为可用份额，初始化事前风控日内状态
            portfolio.on_new_day()
            if self.risk_gateway:
                self.risk_gateway.reset_day(cur_date, portfolio.total_asset)

            # 4.2 获取所有标的当天的行情快照并更新最新现价
            current_bar_map = {}
            for sym, df in data_dict.items():
                if date_idx < len(df):
                    row = df.iloc[date_idx]
                    close_p = float(row["close"])
                    current_bar_map[sym] = row
                    portfolio.update_price(sym, close_p)

            if self.risk_gateway:
                self.risk_gateway.update_intraday_equity(portfolio.total_asset)

            # 4.3 触发策略截面逻辑，判断今日是否调仓并获取目标权重 {symbol: weight}
            target_weights = strategy.on_rebalance_check(date_idx, cur_date, current_bar_map, portfolio)


            # 4.4 若今天触发了调仓，执行组合权重再平衡
            if target_weights is not None:
                self._execute_rebalance(cur_date, target_weights, current_bar_map, portfolio)

            # 4.5 收盘记录资产快照与基准价格
            bmk_close = benchmark_dict.get(cur_date, 0.0)
            portfolio.record_snapshot(cur_date, benchmark_price=bmk_close)

            # 4.6 记录资产配置时序权重（用于前端持仓变迁堆叠面积图）
            total_val = max(portfolio.total_asset, 1.0)
            daily_weights = {}
            for sym, pos in portfolio.positions.items():
                if pos.total_shares > 0:
                    daily_weights[sym] = round(pos.market_value / total_val, 4)
            daily_weights["CASH"] = round(portfolio.cash / total_val, 4)

            allocation_history.append({
                "date": cur_date,
                "weights": daily_weights
            })

        self.log(f"=== 组合回测结束，最终总资产: CNY {portfolio.total_asset:,.2f} ===")

        # 5. 计算量化绩效指标
        metrics = self.metrics_calc.calculate(portfolio, benchmark_symbol=self.config.benchmark_symbol)
        metrics["strategy_name"] = strategy.name
        metrics["symbol"] = "组合策略 (ETF/多因子)"
        metrics["initial_cash"] = self.config.initial_cash
        metrics["final_asset"] = portfolio.total_asset
        metrics["allocation_history"] = allocation_history
        metrics["trades"] = [
            {
                "timestamp": t.timestamp,
                "symbol": t.symbol,
                "action": t.action,
                "shares": t.shares,
                "price": t.price,
                "commission": t.cost.commission,
                "stamp_tax": t.cost.stamp_tax,
                "transfer_fee": t.cost.transfer_fee,
                "total_cost": t.cost.total_cost,
                "realized_pnl": t.realized_pnl,
                "cash_after": t.cash_after
            }
            for t in portfolio.trade_history
        ]
        metrics["logs"] = self.logs[-100:]
        if self.risk_gateway:
            metrics["risk_gateway_summary"] = self.risk_gateway.get_status_summary()
            metrics["risk_gateway_logs"] = self.risk_gateway.get_audit_logs(limit=50)

        return metrics

    def _execute_rebalance(
        self,
        cur_date: str,
        target_weights: Dict[str, float],
        bar_map: Dict[str, pd.Series],
        portfolio: Portfolio
    ):
        """组合调仓：先卖后买，释放资金，受实盘事前风控网关全规则穿透审计"""
        total_asset = portfolio.total_asset

        # 计算每个标的的目标总市值
        target_values = {sym: total_asset * w for sym, w in target_weights.items()}

        # 1. 【第一阶段：先卖】处理减仓或清仓的标的
        for sym, pos in list(portfolio.positions.items()):
            if pos.total_shares <= 0:
                continue

            row = bar_map.get(sym)
            if row is None:
                continue

            is_suspended = bool(row.get("is_suspended", False))
            is_down_limit = bool(row.get("is_down_limit", False))
            close_price = float(row["close"])

            if is_suspended:
                self.log(f"[{cur_date}] 调仓提示: {sym} 停牌中，无法卖出调仓！")
                continue
            if self.config.enforce_price_limits and is_down_limit:
                self.log(f"[{cur_date}] 调仓提示: {sym} 跌停封死，卖出被拒！")
                continue

            target_val = target_values.get(sym, 0.0)
            diff_val = target_val - pos.market_value

            if diff_val < -10.0:  # 需要卖出
                shares_needed = int(abs(diff_val) / close_price)
                # 若目标比例为0(彻底清仓)，则全部可用份额一次性卖出
                if target_val <= 0:
                    shares_to_sell = pos.available_shares
                else:
                    shares_to_sell = min(shares_needed, pos.available_shares)

                if shares_to_sell > 0:
                    # 事前风控网关穿透审查 (卖出)
                    if self.risk_gateway:
                        cur_holdings = {s: p.market_value for s, p in portfolio.positions.items() if p.total_shares > 0}
                        decision = self.risk_gateway.validate_order(
                            order_type="SELL",
                            symbol=sym,
                            shares=shares_to_sell,
                            price=close_price,
                            portfolio_total_asset=portfolio.total_asset,
                            current_holdings=cur_holdings,
                            market_bar=row.to_dict() if hasattr(row, "to_dict") else dict(row)
                        )
                        if not decision.passed:
                            self.log(f"[{cur_date}] ⚠️ 事前风控拦截卖出: {sym} - {decision.reason}")
                            continue

                    exec_price = self.cost_calc.get_execution_price(close_price, is_buy=False)
                    cost = self.cost_calc.calculate_cost(exec_price, shares_to_sell, is_buy=False)
                    trade = portfolio.sell(cur_date, sym, shares_to_sell, exec_price, cost)
                    if trade:
                        if self.risk_gateway:
                            self.risk_gateway.record_executed_trade("SELL", sym, shares_to_sell, exec_price)
                        self.log(f"[{cur_date}] 组合减仓: {sym} 卖出 {shares_to_sell}股 @ {exec_price:.2f}, 释放现金 CNY {trade.cash_after:.2f}")

        # 2. 【第二阶段：再买】处理需要建仓或增仓的标的
        for sym, target_val in target_values.items():
            if target_val <= 0:
                continue

            row = bar_map.get(sym)
            if row is None:
                continue

            is_suspended = bool(row.get("is_suspended", False))
            is_up_limit = bool(row.get("is_up_limit", False))
            close_price = float(row["close"])

            if is_suspended:
                self.log(f"[{cur_date}] 调仓提示: {sym} 停牌中，无法买入！")
                continue
            if self.config.enforce_price_limits and is_up_limit:
                self.log(f"[{cur_date}] 调仓提示: {sym} 涨停封死，买入被拒！")
                continue

            pos = portfolio.get_position(sym)
            diff_val = target_val - pos.market_value

            if diff_val > 100.0:  # 需要增持或新买入
                raw_shares = int(diff_val / close_price)
                # A股 100 股一手
                shares_to_buy = (raw_shares // self.config.lot_size) * self.config.lot_size

                if shares_to_buy > 0:
                    exec_price = self.cost_calc.get_execution_price(close_price, is_buy=True)
                    cost = self.cost_calc.calculate_cost(exec_price, shares_to_buy, is_buy=True)
                    total_req = exec_price * shares_to_buy + cost.total_cost

                    if total_req > portfolio.cash:
                        # 现金不够买足额，尽可能向下凑整手
                        lot_val = exec_price * self.config.lot_size * 1.001
                        affordable_lots = int(portfolio.cash / lot_val) if lot_val > 0 else 0
                        shares_to_buy = affordable_lots * self.config.lot_size
                        if shares_to_buy <= 0:
                            continue
                        cost = self.cost_calc.calculate_cost(exec_price, shares_to_buy, is_buy=True)

                    # 事前风控网关穿透审查 (买入)
                    if self.risk_gateway:
                        cur_holdings = {s: p.market_value for s, p in portfolio.positions.items() if p.total_shares > 0}
                        avg_vol = float(row.get("volume", 0.0))
                        decision = self.risk_gateway.validate_order(
                            order_type="BUY",
                            symbol=sym,
                            shares=shares_to_buy,
                            price=close_price,
                            portfolio_total_asset=portfolio.total_asset,
                            current_holdings=cur_holdings,
                            market_bar=row.to_dict() if hasattr(row, "to_dict") else dict(row),
                            avg_20d_volume=avg_vol if avg_vol > 0 else None
                        )
                        if not decision.passed:
                            self.log(f"[{cur_date}] ⚠️ 事前风控拦截买入: {sym} - {decision.reason}")
                            continue

                    trade = portfolio.buy(cur_date, sym, shares_to_buy, exec_price, cost)
                    if trade:
                        if self.risk_gateway:
                            self.risk_gateway.record_executed_trade("BUY", sym, shares_to_buy, exec_price)
                        self.log(f"[{cur_date}] 组合加仓: {sym} 买入 {shares_to_buy}股 @ {exec_price:.2f}, 费用: {cost.total_cost:.2f}")

