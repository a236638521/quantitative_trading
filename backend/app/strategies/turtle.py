import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from .base import BaseStrategy, Signal
from ..core.portfolio import Portfolio


class TurtleStrategy(BaseStrategy):
    """
    海龟交易法则 (Turtle Trading System / 唐奇安通道突破)
    - 入场：突破前 N 日最高价 (默认 20 日) 时买入
    - 出场：跌破前 M 日最低价 (默认 10 日) 时离场止损
    - 结合 ATR (真实波动幅度) 辅助动态止损
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "entry_period": 20,    # 突破周期 (上轨)
            "exit_period": 10,     # 出场周期 (下轨)
            "atr_period": 14,      # ATR 周期
            "position_pct": 0.95   # 仓位比例
        }
        if params:
            default_params.update(params)
        super().__init__(name="TurtleStrategy", params=default_params)

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        entry_p = self.params["entry_period"]
        exit_p = self.params["exit_period"]
        atr_p = self.params["atr_period"]

        # 唐奇安通道（前 N 日最高价/最低价，需 shift(1) 避免未来函数）
        df["donchian_high"] = df["high"].shift(1).rolling(window=entry_p).max()
        df["donchian_low"] = df["low"].shift(1).rolling(window=exit_p).min()

        # ATR 真实波动幅度
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=atr_p).mean()

        return df

    def on_bar(self, index: int, row: pd.Series, portfolio: Portfolio) -> List[Signal]:
        signals = []
        symbol = str(row.get("code", ""))

        close = row["close"]
        high_bound = row.get("donchian_high", np.nan)
        low_bound = row.get("donchian_low", np.nan)

        if pd.isna(high_bound) or pd.isna(low_bound):
            return signals

        pos = portfolio.get_position(symbol)

        # 向上突破 20 日高点入场
        if close > high_bound and pos.total_shares == 0:
            signals.append(
                self.order_target_percent(
                    symbol=symbol,
                    percent=self.params["position_pct"],
                    reason=f"收盘价 {close:.2f} 突破 {self.params['entry_period']}日高点 {high_bound:.2f}"
                )
            )
        # 向下跌破 10 日低点离场
        elif close < low_bound and pos.total_shares > 0:
            signals.append(
                self.close_position(
                    symbol=symbol,
                    reason=f"收盘价 {close:.2f} 跌破 {self.params['exit_period']}日低点 {low_bound:.2f}"
                )
            )

        return signals
