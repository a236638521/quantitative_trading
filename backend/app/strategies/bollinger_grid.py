import pandas as pd
from typing import List, Dict, Any, Optional
from .base import BaseStrategy, Signal
from ..core.portfolio import Portfolio


class BollingerGridStrategy(BaseStrategy):
    """
    布林带均值回归策略 (Bollinger Bands Mean-Reversion)
    - 价格触碰或跌破布林下轨时视为超跌买入
    - 价格触碰或突破布林上轨时视为超买止盈卖出
    特别适合震荡行情与宽基 ETF 网格套利。
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "period": 20,          # 均线周期
            "num_std": 2.0,        # 标准差倍数
            "position_pct": 0.95   # 仓位比例
        }
        if params:
            default_params.update(params)
        super().__init__(name="BollingerGridStrategy", params=default_params)

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.params["period"]
        num_std = self.params["num_std"]

        df["boll_mid"] = df["close"].rolling(window=period).mean()
        df["boll_std"] = df["close"].rolling(window=period).std()
        df["boll_up"] = df["boll_mid"] + num_std * df["boll_std"]
        df["boll_low"] = df["boll_mid"] - num_std * df["boll_std"]

        return df

    def on_bar(self, index: int, row: pd.Series, portfolio: Portfolio) -> List[Signal]:
        signals = []
        symbol = str(row.get("code", ""))

        close = row["close"]
        low_band = row.get("boll_low")
        up_band = row.get("boll_up")

        if pd.isna(low_band) or pd.isna(up_band):
            return signals

        pos = portfolio.get_position(symbol)

        # 跌破下轨，超跌买入
        if close <= low_band and pos.total_shares == 0:
            signals.append(
                self.order_target_percent(
                    symbol=symbol,
                    percent=self.params["position_pct"],
                    reason=f"收盘价 {close:.2f} 触及布林下轨 {low_band:.2f} 超跌建仓"
                )
            )
        # 突破上轨，超买平仓
        elif close >= up_band and pos.total_shares > 0:
            signals.append(
                self.close_position(
                    symbol=symbol,
                    reason=f"收盘价 {close:.2f} 触及布林上轨 {up_band:.2f} 超买止盈"
                )
            )

        return signals
