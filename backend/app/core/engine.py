import logging
import math
from typing import Dict, Any, Optional, List
import pandas as pd

from ..config import BacktestConfig
from .cost import AShareCostCalculator
from .portfolio import Portfolio
from .metrics import MetricsCalculator
from ..strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    A股量化回测核心撮合与执行引擎
    严格内嵌：
    1. T+1 可用持仓结算机制
    2. 买入 100 股一手整数倍限制
    3. 涨停无法买入、跌停无法卖出撮合限制
    4. 印花税 (单边万五)、佣金 (双向万二点五起五元)、过户费、滑点损耗
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.cost_calc = AShareCostCalculator(self.config.cost_config)
        self.metrics_calc = MetricsCalculator()
        self.execution_logs: List[str] = []

    def log(self, message: str):
        self.execution_logs.append(message)
        logger.info(message)

    def run(
        self,
        strategy: BaseStrategy,
        data_df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
        symbol: str = ""
    ) -> Dict[str, Any]:
        """
        启动回测
        :param strategy: 策略实例
        :param data_df: 标的行情 DataFrame
        :param benchmark_df: 基准指数 DataFrame (可选，如沪深300)
        :param symbol: 标的代码
        """
        self.execution_logs.clear()
        if data_df is None or data_df.empty:
            raise ValueError("回测行情数据为空！")

        clean_symbol = symbol or str(data_df["code"].iloc[0] if "code" in data_df.columns else "UNKNOWN")

        # 1. 策略计算指标
        df = strategy.prepare_indicators(data_df.copy())
        if "code" not in df.columns:
            df["code"] = clean_symbol

        # 2. 对齐基准行情
        benchmark_dict: Dict[str, float] = {}
        if benchmark_df is not None and not benchmark_df.empty:
            for _, b_row in benchmark_df.iterrows():
                b_date = str(b_row["date"])
                benchmark_dict[b_date] = float(b_row["close"])

        # 3. 初始化账户与资金
        portfolio = Portfolio(
            initial_cash=self.config.initial_cash,
            enable_t_plus_one=self.config.enable_t_plus_one
        )

        self.log(f"=== 开始回测: {strategy.name} 标的: {clean_symbol} 初始资金: CNY {self.config.initial_cash:,.2f} ===")

        # 4. 逐 Bar 事件驱动循环
        total_bars = len(df)
        for i in range(total_bars):
            row = df.iloc[i]
            cur_date = str(row["date"])
            close_price = float(row["close"])

            # 4.1 新交易日开盘前：重置 T+1 可用持仓
            portfolio.on_new_day()

            # 4.2 更新当前持仓现价与估值
            portfolio.update_price(clean_symbol, close_price)

            # 4.3 触发策略逻辑，获取调仓信号
            signals: List[Signal] = strategy.on_bar(i, row, portfolio)

            # 4.4 撮合执行信号
            for sig in signals:
                self._execute_signal(cur_date, sig, row, portfolio)

            # 4.5 收盘记录资产快照
            bmk_price = benchmark_dict.get(cur_date, 0.0)
            portfolio.record_snapshot(cur_date, benchmark_price=bmk_price)

        self.log(f"=== 回测结束，总历经 {total_bars} 个交易日，最终资产: CNY {portfolio.total_asset:,.2f} ===")

        # 5. 计算全套绩效指标
        metrics = self.metrics_calc.calculate(portfolio, benchmark_symbol=self.config.benchmark_symbol)
        metrics["symbol"] = clean_symbol
        metrics["strategy_name"] = strategy.name
        metrics["initial_cash"] = self.config.initial_cash
        metrics["final_asset"] = portfolio.total_asset
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
        metrics["logs"] = self.execution_logs[-100:]  # 返回最新100条回测日志

        return metrics

    def _execute_signal(self, cur_date: str, sig: Signal, row: pd.Series, portfolio: Portfolio):
        symbol = sig.symbol
        close_price = float(row["close"])
        pos = portfolio.get_position(symbol)

        is_up_limit = bool(row.get("is_up_limit", False))
        is_down_limit = bool(row.get("is_down_limit", False))

        # 计算目标股数
        shares_to_buy = 0
        shares_to_sell = 0

        if sig.target_type == "percent":
            target_asset_value = portfolio.total_asset * sig.percent
            current_pos_value = pos.market_value
            diff_value = target_asset_value - current_pos_value

            if diff_value > 0:
                # 需要买入
                raw_shares = int(diff_value / close_price)
                # A股必须是 100 股的整数倍
                shares_to_buy = (raw_shares // self.config.lot_size) * self.config.lot_size
            elif diff_value < 0:
                # 需要卖出
                raw_shares = int(abs(diff_value) / close_price)
                shares_to_sell = min(raw_shares, pos.available_shares)
                # 若目标比例为 0 (清仓)，则全部可用持仓一键卖出
                if sig.percent == 0:
                    shares_to_sell = pos.available_shares
        elif sig.target_type == "shares":
            if sig.action == "BUY":
                shares_to_buy = (sig.shares // self.config.lot_size) * self.config.lot_size
            elif sig.action == "SELL":
                shares_to_sell = min(sig.shares, pos.available_shares)

        # 1. 执行买入
        if shares_to_buy > 0:
            if self.config.enforce_price_limits and is_up_limit:
                self.log(f"[{cur_date}] 买入失败: {symbol} 触发封死涨停，无法撮合成交！原因: {sig.reason}")
                return

            exec_price = self.cost_calc.get_execution_price(close_price, is_buy=True)
            cost = self.cost_calc.calculate_cost(exec_price, shares_to_buy, is_buy=True)
            total_req = exec_price * shares_to_buy + cost.total_cost

            # 资金不足时向下压缩股数
            if total_req > portfolio.cash:
                lot_cost = exec_price * self.config.lot_size * (1.0 + self.config.cost_config.commission_rate)
                affordable_lots = int(portfolio.cash / lot_cost) if lot_cost > 0 else 0
                shares_to_buy = affordable_lots * self.config.lot_size
                if shares_to_buy <= 0:
                    self.log(f"[{cur_date}] 买入跳过: {symbol} 可用资金不足以买入一手(100股)")
                    return
                cost = self.cost_calc.calculate_cost(exec_price, shares_to_buy, is_buy=True)

            trade = portfolio.buy(cur_date, symbol, shares_to_buy, exec_price, cost)
            if trade:
                self.log(f"[{cur_date}] 买入成交: {symbol} {shares_to_buy}股 @ {exec_price:.2f}, 费用: {cost.total_cost:.2f}, 理由: {sig.reason}")

        # 2. 执行卖出
        elif shares_to_sell > 0:
            if self.config.enforce_price_limits and is_down_limit:
                self.log(f"[{cur_date}] 卖出失败: {symbol} 触发封死跌停，无法撮合成交！原因: {sig.reason}")
                return

            if pos.available_shares < shares_to_sell:
                self.log(f"[{cur_date}] 卖出受阻: {symbol} 可用股数不足({pos.available_shares}股, T+1受限或已卖出)")
                return

            exec_price = self.cost_calc.get_execution_price(close_price, is_buy=False)
            cost = self.cost_calc.calculate_cost(exec_price, shares_to_sell, is_buy=False)
            trade = portfolio.sell(cur_date, symbol, shares_to_sell, exec_price, cost)
            if trade:
                self.log(f"[{cur_date}] 卖出成交: {symbol} {shares_to_sell}股 @ {exec_price:.2f}, 费用: {cost.total_cost:.2f}, 盈亏: {trade.realized_pnl:+.2f}, 理由: {sig.reason}")
