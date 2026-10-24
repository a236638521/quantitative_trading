from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .cost import TradeCost


@dataclass
class Position:
    symbol: str
    total_shares: int = 0          # 总持仓股数
    available_shares: int = 0      # 当日可卖股数 (遵循 T+1 限制)
    avg_cost: float = 0.0          # 持仓均价 (包含建仓佣金)
    current_price: float = 0.0     # 最新现价
    realized_pnl: float = 0.0      # 累计已实现盈亏

    @property
    def market_value(self) -> float:
        """持仓市值"""
        return round(self.total_shares * self.current_price, 2)

    @property
    def unrealized_pnl(self) -> float:
        """浮动盈亏"""
        if self.total_shares == 0:
            return 0.0
        return round((self.current_price - self.avg_cost) * self.total_shares, 2)

    @property
    def unrealized_pnl_ratio(self) -> float:
        """浮动盈亏率"""
        if self.avg_cost <= 0 or self.total_shares == 0:
            return 0.0
        return round((self.current_price - self.avg_cost) / self.avg_cost, 4)


@dataclass
class TradeRecord:
    timestamp: str                 # 交易发生时间 (YYYY-MM-DD 或 YYYY-MM-DD HH:MM)
    symbol: str                    # 标的代码
    action: str                    # BUY / SELL
    shares: int                    # 成交股数
    price: float                   # 撮合价格
    cost: TradeCost                # 交易各项成本
    realized_pnl: float = 0.0      # 本笔卖出实现的盈亏 (买入为 0)
    cash_after: float = 0.0        # 交易后现金余额


@dataclass
class DailySnapshot:
    date: str                      # 日期 YYYY-MM-DD
    cash: float                    # 可用现金
    market_value: float            # 股票总市值
    total_asset: float             # 总资产 (现金 + 市值)
    daily_return: float = 0.0      # 当日收益率
    benchmark_price: float = 0.0   # 基准价格 (如沪深300)
    benchmark_return: float = 0.0  # 基准当日收益率


class Portfolio:
    """账户与持仓簿记管理，内置 A 股 T+1 制度与资金结算"""

    def __init__(self, initial_cash: float = 100000.0, enable_t_plus_one: bool = True):
        self.initial_cash: float = initial_cash
        self.cash: float = initial_cash
        self.enable_t_plus_one: bool = enable_t_plus_one
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeRecord] = []
        self.daily_snapshots: List[DailySnapshot] = []

    def on_new_day(self):
        """每个新交易日开盘前调用：T+1 规则下，当日可卖股数更新为昨日总持仓"""
        for pos in self.positions.values():
            if self.enable_t_plus_one:
                pos.available_shares = pos.total_shares
            else:
                pos.available_shares = pos.total_shares

    def get_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def update_price(self, symbol: str, price: float):
        """更新标的最新行情价格"""
        if symbol in self.positions:
            self.positions[symbol].current_price = price

    def buy(self, timestamp: str, symbol: str, shares: int, exec_price: float, cost: TradeCost) -> Optional[TradeRecord]:
        """
        买入操作
        总花费 = 股票金额 + 交易费用
        """
        total_expense = exec_price * shares + cost.total_cost
        if total_expense > self.cash:
            return None  # 资金不足

        self.cash -= total_expense
        pos = self.get_position(symbol)

        # 更新均价 (包含交易规费)
        prev_value = pos.total_shares * pos.avg_cost
        new_total_shares = pos.total_shares + shares
        pos.avg_cost = round((prev_value + total_expense) / new_total_shares, 4)
        pos.total_shares = new_total_shares
        pos.current_price = exec_price

        # T+1: 当天买入的股份不能在当天卖出
        if not self.enable_t_plus_one:
            pos.available_shares = pos.total_shares

        trade = TradeRecord(
            timestamp=timestamp,
            symbol=symbol,
            action="BUY",
            shares=shares,
            price=exec_price,
            cost=cost,
            realized_pnl=0.0,
            cash_after=round(self.cash, 2)
        )
        self.trade_history.append(trade)
        return trade

    def sell(self, timestamp: str, symbol: str, shares: int, exec_price: float, cost: TradeCost) -> Optional[TradeRecord]:
        """
        卖出操作
        净回款 = 股票金额 - 交易费用
        """
        pos = self.get_position(symbol)
        if shares > pos.available_shares:
            return None  # 可卖持仓不足 (T+1 限制或股数不足)

        revenue = exec_price * shares - cost.total_cost
        self.cash += revenue

        # 计算本笔卖出的已实现盈亏
        realized_pnl = round(shares * (exec_price - pos.avg_cost) - cost.total_cost, 2)
        pos.realized_pnl += realized_pnl

        pos.total_shares -= shares
        pos.available_shares -= shares
        pos.current_price = exec_price

        if pos.total_shares == 0:
            pos.avg_cost = 0.0

        trade = TradeRecord(
            timestamp=timestamp,
            symbol=symbol,
            action="SELL",
            shares=shares,
            price=exec_price,
            cost=cost,
            realized_pnl=realized_pnl,
            cash_after=round(self.cash, 2)
        )
        self.trade_history.append(trade)
        return trade

    @property
    def total_market_value(self) -> float:
        """当前全部持仓股票市值"""
        return round(sum(pos.market_value for pos in self.positions.values()), 2)

    @property
    def total_asset(self) -> float:
        """总资产 = 可用现金 + 持仓总市值"""
        return round(self.cash + self.total_market_value, 2)

    def record_snapshot(self, date: str, benchmark_price: float = 0.0) -> DailySnapshot:
        """记录每日盘后资产快照与收益"""
        asset = self.total_asset
        daily_return = 0.0
        benchmark_return = 0.0

        if self.daily_snapshots:
            prev_snapshot = self.daily_snapshots[-1]
            if prev_snapshot.total_asset > 0:
                daily_return = (asset - prev_snapshot.total_asset) / prev_snapshot.total_asset
            if prev_snapshot.benchmark_price > 0 and benchmark_price > 0:
                benchmark_return = (benchmark_price - prev_snapshot.benchmark_price) / prev_snapshot.benchmark_price

        snapshot = DailySnapshot(
            date=date,
            cash=round(self.cash, 2),
            market_value=self.total_market_value,
            total_asset=asset,
            daily_return=round(daily_return, 6),
            benchmark_price=benchmark_price,
            benchmark_return=round(benchmark_return, 6)
        )
        self.daily_snapshots.append(snapshot)
        return snapshot
