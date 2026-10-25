import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from ..factors.technical import calc_momentum, calc_trend_score, calc_volatility
from ..factors.processor import CrossSectionProcessor


class MultiFactorStockSelectionStrategy:
    """
    成长+价值+动量综合多因子截面选股与轮动策略
    实战原理：
    1. 在候选股票池中，每月初计算各标的的截面因子特征：
       - 动量因子 (20日收益率)
       - 趋势强度因子 (收盘价相对20日均线的偏离度)
       - 低波动率因子 (20日年化波动率负向化，追求平稳上涨)
    2. 经过 MAD 去极值与 Z-Score 标准化合成综合 Alpha 打分
    3. 截面排序选出得分最高的 Top N 只股票等权重换仓持有
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.name = "MultiFactorStockSelectionStrategy"
        default_params = {
            "top_n": 3,                      # 每次持有排名前 N 只股票 (默认3只)
            "lookback_period": 20,           # 动量与波动率窗口 (默认20日)
            "weight_momentum": 0.5,          # 动量因子权重
            "weight_trend": 0.3,             # 趋势偏离因子权重
            "weight_low_vol": 0.2            # 低波动因子权重
        }
        if params:
            default_params.update(params)
        self.params = default_params

    def prepare(self, data_dict: Dict[str, pd.DataFrame], trading_dates: List[str]) -> Dict[str, pd.DataFrame]:
        lookback = self.params["lookback_period"]
        prepared_dict = {}

        for sym, df in data_dict.items():
            d = df.copy()
            close = d["close"]
            d["f_momentum"] = calc_momentum(close, period=lookback)
            d["f_trend"] = calc_trend_score(close, ma_period=lookback)
            # 低波动率取负值，波动越小得分越高
            d["f_low_vol"] = -calc_volatility(close, period=lookback)
            prepared_dict[sym] = d

        return prepared_dict

    def on_rebalance_check(
        self,
        date_idx: int,
        cur_date: str,
        bar_map: Dict[str, pd.Series],
        portfolio
    ) -> Optional[Dict[str, float]]:
        if date_idx < self.params["lookback_period"]:
            return None

        # 月初首个交易日触发换仓
        dt = pd.to_datetime(cur_date)
        is_first_day_of_month = (dt.day <= 4) and (date_idx == 0 or pd.to_datetime(cur_date).month != pd.to_datetime(portfolio.daily_snapshots[-1].date).month if portfolio.daily_snapshots else False)

        if not is_first_day_of_month:
            return None

        # 1. 截面提取各因子值
        raw_factors = {"momentum": {}, "trend": {}, "low_vol": {}}
        valid_symbols = []

        for sym, row in bar_map.items():
            is_suspended = bool(row.get("is_suspended", False))
            if not is_suspended:
                raw_factors["momentum"][sym] = float(row.get("f_momentum", 0.0))
                raw_factors["trend"][sym] = float(row.get("f_trend", 0.0))
                raw_factors["low_vol"][sym] = float(row.get("f_low_vol", 0.0))
                valid_symbols.append(sym)

        if len(valid_symbols) < self.params["top_n"]:
            return None

        # 2. 截面多因子标准化与合成打分
        s_momentum = pd.Series(raw_factors["momentum"])
        s_trend = pd.Series(raw_factors["trend"])
        s_low_vol = pd.Series(raw_factors["low_vol"])

        weights = {
            "momentum": self.params["weight_momentum"],
            "trend": self.params["weight_trend"],
            "low_vol": self.params["weight_low_vol"]
        }

        composite_scores = CrossSectionProcessor.process_and_combine(
            {"momentum": s_momentum, "trend": s_trend, "low_vol": s_low_vol},
            weights=weights
        )

        # 3. 选出得分最高的 Top N 只股票
        sorted_symbols = composite_scores.sort_values(ascending=False).index.tolist()
        top_n = min(self.params["top_n"], len(sorted_symbols))
        selected = sorted_symbols[:top_n]

        # 4. 等权重配置
        weight = round(1.0 / top_n, 4)
        return {sym: weight for sym in selected}
