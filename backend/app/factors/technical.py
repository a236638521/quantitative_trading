import numpy as np
import pandas as pd


def calc_momentum(close: pd.Series, period: int = 20) -> pd.Series:
    """计算 N 日价格动量收益率 (Close_t / Close_{t-N} - 1)"""
    return (close / close.shift(period) - 1.0).fillna(0.0)


def calc_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """计算 N 日年化波动率"""
    daily_returns = close.pct_change().fillna(0.0)
    return (daily_returns.rolling(window=period).std() * np.sqrt(250)).fillna(0.0)


def calc_risk_adjusted_momentum(close: pd.Series, period: int = 20) -> pd.Series:
    """
    计算风险调整后的动量因子 (夏普式动量: 动量 / 波动率)
    避免买入虚高但剧烈波动的诱多资产
    """
    mom = calc_momentum(close, period)
    vol = calc_volatility(close, period)
    return (mom / (vol + 1e-4)).fillna(0.0)


def calc_trend_score(close: pd.Series, ma_period: int = 20) -> pd.Series:
    """计算均线偏离度 (Close_t / MA_N - 1)"""
    ma = close.rolling(window=ma_period).mean()
    return (close / ma - 1.0).fillna(0.0)
