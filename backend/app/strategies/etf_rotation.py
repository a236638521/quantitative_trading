import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from ..factors.technical import calc_momentum, calc_risk_adjusted_momentum


class ETFMomentumRotationStrategy:
    """
    大类资产 / 跨市场 ETF 动量轮动策略
    实战原理：
    1. 候选池涵盖：A股大盘（沪深300）、成长中小盘（中证500）、全球科技（纳指ETF）、抗通胀商品（黄金ETF）、避险资产（国债ETF）
    2. 计算各资产过去 N 日的动量得分（支持夏普式风险调整动量）
    3. 每周/每月初截面横向排序，选出最强的前 Top N 只等权持有
    4. 避险熔断机制：当进攻型资产动量全部为负时，自动转为持有国债 ETF 避险防守
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.name = "ETFMomentumRotationStrategy"
        default_params = {
            "lookback_period": 20,       # 动量回看周期 (默认20日)
            "top_n": 1,                  # 每次持有前 N 只最强资产 (默认1只)
            "rebalance_freq": "weekly",  # 调仓频率: "weekly" (每周首个交易日) / "monthly" (每月首个交易日)
            "use_risk_adjusted": True,   # 是否使用收益/波动比 (夏普式动量)
            "defense_symbol": "sh.511010" # 避险资产代码 (国债ETF)
        }
        if params:
            default_params.update(params)
        self.params = default_params

    def prepare(self, data_dict: Dict[str, pd.DataFrame], trading_dates: List[str]) -> Dict[str, pd.DataFrame]:
        """为所有标的预先计算动量指标"""
        lookback = self.params["lookback_period"]
        use_risk_adj = self.params["use_risk_adjusted"]

        prepared_dict = {}
        for sym, df in data_dict.items():
            d = df.copy()
            if use_risk_adj:
                d["score"] = calc_risk_adjusted_momentum(d["close"], period=lookback)
            else:
                d["score"] = calc_momentum(d["close"], period=lookback)
            # 记录纯价格动量收益率供避险过滤
            d["raw_momentum"] = calc_momentum(d["close"], period=lookback)
            prepared_dict[sym] = d

        return prepared_dict

    def on_rebalance_check(
        self,
        date_idx: int,
        cur_date: str,
        bar_map: Dict[str, pd.Series],
        portfolio
    ) -> Optional[Dict[str, float]]:
        """
        判断今日是否为调仓日，若是则输出目标持仓权重字典
        """
        if date_idx < self.params["lookback_period"]:
            return None  # 动量数据不足窗口期，暂不建仓

        # 1. 检查调仓频率
        freq = self.params["rebalance_freq"]
        dt = pd.to_datetime(cur_date)

        is_rebalance_day = False
        if freq == "daily":
            is_rebalance_day = True
        elif freq == "weekly":
            # 每周首个交易日 (通常是周一或节假日后的第一天)
            is_rebalance_day = (dt.weekday() == 0) or (date_idx > 0 and pd.to_datetime(cur_date).week != pd.to_datetime(portfolio.daily_snapshots[-1].date).week if portfolio.daily_snapshots else False)
        elif freq == "monthly":
            # 每月首个交易日
            is_rebalance_day = (dt.day <= 4) and (date_idx == 0 or pd.to_datetime(cur_date).month != pd.to_datetime(portfolio.daily_snapshots[-1].date).month if portfolio.daily_snapshots else False)

        if not is_rebalance_day:
            return None

        # 2. 截面提取各标的打分 (排除避险资产自身的打分)
        defense_sym = self.params["defense_symbol"]
        scores = {}
        raw_moms = {}

        for sym, row in bar_map.items():
            if sym == defense_sym:
                continue
            is_suspended = bool(row.get("is_suspended", False))
            if not is_suspended:
                scores[sym] = float(row.get("score", -999.0))
                raw_moms[sym] = float(row.get("raw_momentum", -999.0))

        if not scores:
            return {defense_sym: 1.0}

        # 3. 避险过滤：如果所有资产的 20 日原始动量全部为负（全市场普跌/熊市）
        all_negative = all(mom <= 0 for mom in raw_moms.values())
        if all_negative:
            # 避险开启：全仓配置国债ETF (若在池中) 或全仓现金
            if defense_sym in bar_map:
                return {defense_sym: 1.0}
            else:
                return {}  # 全仓现金

        # 4. 排序并选出排名前 Top N 的标的
        sorted_symbols = sorted(scores.keys(), key=lambda s: scores[s], reverse=True)
        top_n = min(self.params["top_n"], len(sorted_symbols))
        selected = sorted_symbols[:top_n]

        # 过滤掉得分为负数的标的
        valid_selected = [s for s in selected if raw_moms.get(s, -1) > 0]
        if not valid_selected:
            if defense_sym in bar_map:
                return {defense_sym: 1.0}
            return {}

        # 5. 等权重分配
        weight = round(1.0 / len(valid_selected), 4)
        target_weights = {sym: weight for sym in valid_selected}

        return target_weights
