import pandas as pd
from typing import List, Dict, Any, Optional
from .base import BaseStrategy, Signal
from ..core.portfolio import Portfolio


class DualMAStrategy(BaseStrategy):
    """
    经典双均线趋势跟踪策略 (Dual Moving Average)
    - 快线从下向上穿越慢线（金叉）：买入信号
    - 快线从上向下穿越慢线（死叉）：平仓卖出信号
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "fast_period": 5,      # 短周期均线 (默认5日)
            "slow_period": 20,     # 长周期均线 (默认20日)
            "position_pct": 0.95   # 每次建仓目标仓位比例 (默认95%防滑点)
        }
        if params:
            default_params.update(params)
        super().__init__(name="DualMAStrategy", params=default_params)

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        fast_p = self.params["fast_period"]
        slow_p = self.params["slow_period"]

        df["ma_fast"] = df["close"].rolling(window=fast_p).mean()
        df["ma_slow"] = df["close"].rolling(window=slow_p).mean()

        # 计算金叉与死叉
        df["ma_diff"] = df["ma_fast"] - df["ma_slow"]
        df["prev_diff"] = df["ma_diff"].shift(1)
        df["golden_cross"] = (df["prev_diff"] <= 0) & (df["ma_diff"] > 0)
        df["death_cross"] = (df["prev_diff"] >= 0) & (df["ma_diff"] < 0)

        return df

    def on_bar(self, index: int, row: pd.Series, portfolio: Portfolio) -> List[Signal]:
        signals = []
        symbol = str(row.get("code", ""))

        is_golden = bool(row.get("golden_cross", False))
        is_death = bool(row.get("death_cross", False))
        pos = portfolio.get_position(symbol)

        # 出现金叉且目前无持仓，发出建仓信号
        if is_golden and pos.total_shares == 0:
            signals.append(
                self.order_target_percent(
                    symbol=symbol,
                    percent=self.params["position_pct"],
                    reason=f"MA{self.params['fast_period']} 上穿 MA{self.params['slow_period']} 金叉建仓"
                )
            )
        # 出现死叉且目前持有股票，发出平仓信号
        elif is_death and pos.total_shares > 0:
            signals.append(
                self.close_position(
                    symbol=symbol,
                    reason=f"MA{self.params['fast_period']} 下穿 MA{self.params['slow_period']} 死叉平仓"
                )
            )

        return signals
