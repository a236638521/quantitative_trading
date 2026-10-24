from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import pandas as pd
from ..core.portfolio import Portfolio


@dataclass
class Signal:
    """策略发出的交易意图信号"""
    action: str              # "BUY" / "SELL" / "HOLD"
    symbol: str
    target_type: str = "shares"  # "shares" (具体股数), "percent" (仓位比例 0.0~1.0), "target_shares" (目标总持仓)
    shares: int = 0
    percent: float = 0.0
    reason: str = ""


class BaseStrategy(ABC):
    """量化策略抽象基类"""

    def __init__(self, name: str = "BaseStrategy", params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params: Dict[str, Any] = params or {}
        self.indicators: Dict[str, pd.Series] = {}

    @abstractmethod
    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """回测前预先计算策略所需的技术指标或因子特征"""
        pass

    @abstractmethod
    def on_bar(self, index: int, row: pd.Series, portfolio: Portfolio) -> List[Signal]:
        """
        每根 K 线触发一次策略逻辑
        :param index: 当前 Bar 的整型索引 (0, 1, 2...)
        :param row: 当前 Bar 对应的行情与指标数据
        :param portfolio: 当前账户持仓与资金状态
        :return: 交易信号列表
        """
        pass

    def buy_shares(self, symbol: str, shares: int, reason: str = "") -> Signal:
        return Signal(action="BUY", symbol=symbol, target_type="shares", shares=shares, reason=reason)

    def sell_shares(self, symbol: str, shares: int, reason: str = "") -> Signal:
        return Signal(action="SELL", symbol=symbol, target_type="shares", shares=shares, reason=reason)

    def order_target_percent(self, symbol: str, percent: float, reason: str = "") -> Signal:
        """调整持仓至目标总市值占比 (0.0 表示清仓，1.0 表示全仓)"""
        return Signal(action="BUY" if percent > 0 else "SELL", symbol=symbol, target_type="percent", percent=percent, reason=reason)

    def close_position(self, symbol: str, reason: str = "清仓") -> Signal:
        """清空该标的的全部可用持仓"""
        return Signal(action="SELL", symbol=symbol, target_type="percent", percent=0.0, reason=reason)
