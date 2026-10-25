import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .technical import (
    calc_momentum,
    calc_volatility,
    calc_risk_adjusted_momentum,
    calc_trend_score
)
from .processor import CrossSectionProcessor

logger = logging.getLogger(__name__)


# 可用投研因子字典定义
AVAILABLE_FACTORS = [
    {
        "id": "momentum_20",
        "name": "20日价格动量",
        "category": "动量趋势",
        "description": "过去 20 个交易日的区间累计涨跌幅，反映中期顺势动量效应。",
        "formula": "Close(t) / Close(t-20) - 1.0"
    },
    {
        "id": "momentum_60",
        "name": "60日季度动量",
        "category": "动量趋势",
        "description": "过去 60 个交易日的季度累计涨幅，反映中长期波段动量。",
        "formula": "Close(t) / Close(t-60) - 1.0"
    },
    {
        "id": "volatility_20",
        "name": "20日低波动率",
        "category": "风险控制",
        "description": "过去 20 个交易日的年化收益波动率的倒数（低波异常收益因子）。",
        "formula": "-1.0 * Std(DailyReturns, 20) * sqrt(250)"
    },
    {
        "id": "risk_adj_momentum",
        "name": "夏普动量 (风险调整动量)",
        "category": "综合质量",
        "description": "动量收益除以年化波动率，过滤虚高但伴随剧烈回撤的诱多标的。",
        "formula": "Momentum(20) / (Volatility(20) + 1e-4)"
    },
    {
        "id": "trend_score",
        "name": "均线偏离度 (MA趋势强度)",
        "category": "动量趋势",
        "description": "当前价格对 20 日移动平均线的相对偏离度，衡量趋势强弱。",
        "formula": "Close(t) / MA(20) - 1.0"
    },
    {
        "id": "rsi_14",
        "name": "14日 RSI 相对强弱",
        "category": "均值回归",
        "description": "经典 14 日相对强弱指数，反映短期超买超卖情绪与反转预期。",
        "formula": "100 - (100 / (1 + RS(14)))"
    },
    {
        "id": "reversal_5",
        "name": "5日短期反转",
        "category": "均值回归",
        "description": "短期 5 个交易日涨跌幅的相反数，捕捉 A 股特有的周度超跌反弹脉冲。",
        "formula": "-1.0 * (Close(t) / Close(t-5) - 1.0)"
    }
]


class FactorAnalyzer:
    """机构级单因子/多因子投研评价引擎 (Rank-IC、IC-IR、分层回测与多空收益)"""

    @classmethod
    def compute_single_factor(cls, factor_name: str, df: pd.DataFrame) -> pd.Series:
        """为单只标的的时间序列计算指定因子值"""
        close = df["close"].astype(float)
        
        if factor_name == "momentum_20":
            return calc_momentum(close, period=20)
        elif factor_name == "momentum_60":
            return calc_momentum(close, period=60)
        elif factor_name == "volatility_20":
            # 因子值越大越好，低波动因子取负数
            vol = calc_volatility(close, period=20)
            return -vol
        elif factor_name == "risk_adj_momentum":
            return calc_risk_adjusted_momentum(close, period=20)
        elif factor_name == "trend_score":
            return calc_trend_score(close, ma_period=20)
        elif factor_name == "rsi_14":
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-6)
            return 100.0 - (100.0 / (1.0 + rs))
        elif factor_name == "reversal_5":
            mom5 = calc_momentum(close, period=5)
            return -mom5
        else:
            # 默认返回 20 日动量
            return calc_momentum(close, period=20)

    @classmethod
    def analyze(
        cls,
        factor_name: str,
        data_dict: Dict[str, pd.DataFrame],
        trading_dates: List[str],
        forward_period: int = 5,
        quantiles: int = 5
    ) -> Dict[str, Any]:
        """
        全流程因子投研评价：
        1. 截面对齐与因子矩阵计算
        2. T+N 未来收益率矩阵计算 (无前视偏误)
        3. 逐日 Spearman Rank-IC 与累积 IC 曲线
        4. IC 统计学特征 (Mean, Std, IR, T-stat, 胜率)
        5. 分层回测 (Q1 ~ Q5) 累计净值与多空对冲 (Long - Short)
        """
        symbols = list(data_dict.keys())
        if len(symbols) < 2:
            raise ValueError("因子截面分析至少需要 2 只及以上标的")

        # 1. 构造统一时间序列面板：收盘价矩阵与因子矩阵
        # dates x symbols
        close_matrix = pd.DataFrame(index=trading_dates, columns=symbols, dtype=float)
        factor_matrix = pd.DataFrame(index=trading_dates, columns=symbols, dtype=float)

        for sym in symbols:
            df = data_dict[sym]
            df_indexed = df.set_index("date")
            common_idx = [d for d in trading_dates if d in df_indexed.index]
            close_matrix.loc[common_idx, sym] = df_indexed.loc[common_idx, "close"].astype(float)
            
            # 计算因子序列
            f_series = cls.compute_single_factor(factor_name, df)
            f_indexed = pd.Series(f_series.values, index=df["date"])
            factor_matrix.loc[common_idx, sym] = f_indexed.loc[common_idx]

        # 填充少量缺失值
        close_matrix = close_matrix.ffill().bfill()
        factor_matrix = factor_matrix.ffill().bfill()

        # 2. 计算日收益率矩阵与未来收益率矩阵
        # daily_return: t 到 t+1 的收益率
        daily_returns = close_matrix.pct_change().shift(-1)
        
        # forward_return: t 到 t+forward_period 的未来收益率
        forward_returns = (close_matrix.shift(-forward_period) / close_matrix - 1.0)

        # 3. 计算每日 Spearman Rank-IC
        ic_records = []
        cumulative_ic = 0.0

        # 由于 forward_return 需向后看 forward_period 日，最后 forward_period 日没有未来收益
        valid_ic_dates = trading_dates[:-forward_period] if len(trading_dates) > forward_period else []

        for d in valid_ic_dates:
            f_row = factor_matrix.loc[d].dropna()
            r_row = forward_returns.loc[d].dropna()
            common_syms = f_row.index.intersection(r_row.index)

            if len(common_syms) >= 3:
                # 计算 Spearman 秩相关系数
                ic, _ = spearmanr(f_row.loc[common_syms], r_row.loc[common_syms])
                if not np.isnan(ic):
                    cumulative_ic += ic
                    ic_records.append({
                        "date": str(d),
                        "ic": round(float(ic), 4),
                        "cumulative_ic": round(float(cumulative_ic), 4)
                    })

        # 计算 IC 统计量
        if ic_records:
            ic_values = [item["ic"] for item in ic_records]
            mean_ic = float(np.mean(ic_values))
            std_ic = float(np.std(ic_values))
            ic_ir = mean_ic / (std_ic + 1e-6)
            annualized_ir = ic_ir * np.sqrt(250.0 / max(1, forward_period))
            positive_ratio = float(np.mean([1.0 if x > 0 else 0.0 for x in ic_values]))
            t_stat = mean_ic / (std_ic / np.sqrt(len(ic_values)) + 1e-6)
        else:
            mean_ic = 0.0
            std_ic = 0.0
            ic_ir = 0.0
            annualized_ir = 0.0
            positive_ratio = 0.0
            t_stat = 0.0

        # 4. 动态分层回测 (Quantile Analysis Q1 ~ Q5)
        # 实际可用分层数不超过标的数量
        num_quantiles = min(quantiles, len(symbols))
        group_keys = [f"Q{i+1}" for i in range(num_quantiles)]

        # 每个分组的每日净值序列 (初始为 1.0)
        nav_curves = {g: [1.0] for g in group_keys}
        nav_dates = [trading_dates[0]]
        long_short_nav = [1.0]

        # 逐日模拟：t 日因子打分排序 -> t+1 日获取各组平均涨跌幅
        for i in range(len(trading_dates) - 1):
            curr_date = trading_dates[i]
            next_date = trading_dates[i + 1]

            f_row = factor_matrix.loc[curr_date].dropna()
            d_ret_row = daily_returns.loc[curr_date].dropna()
            valid_syms = f_row.index.intersection(d_ret_row.index)

            if len(valid_syms) < num_quantiles:
                # 标的过少，净值走平
                for g in group_keys:
                    nav_curves[g].append(nav_curves[g][-1])
                long_short_nav.append(long_short_nav[-1])
                nav_dates.append(str(next_date))
                continue

            # 按因子值从高到低排序 (Q1 为头部最好, Q_k 为尾部最差)
            sorted_syms = f_row.loc[valid_syms].sort_values(ascending=False).index.tolist()
            
            # 将标的均分切片到 num_quantiles 个组
            chunk_size = len(sorted_syms) / num_quantiles
            group_rets = {}
            for q_idx in range(num_quantiles):
                start_idx = int(q_idx * chunk_size)
                end_idx = int((q_idx + 1) * chunk_size) if q_idx < num_quantiles - 1 else len(sorted_syms)
                q_syms = sorted_syms[start_idx:end_idx]
                if q_syms:
                    q_ret = float(d_ret_row.loc[q_syms].mean())
                else:
                    q_ret = 0.0
                g_key = group_keys[q_idx]
                group_rets[g_key] = q_ret
                # 复合乘积计入净值
                nav_curves[g_key].append(round(nav_curves[g_key][-1] * (1.0 + q_ret), 5))

            # 多空对冲日收益 (Q1 头部减去 Q_tail 尾部)
            top_ret = group_rets[group_keys[0]]
            bottom_ret = group_rets[group_keys[-1]]
            ls_ret = top_ret - bottom_ret
            long_short_nav.append(round(long_short_nav[-1] * (1.0 + ls_ret), 5))
            nav_dates.append(str(next_date))

        # 5. 各组绩效统计指标
        group_stats = []
        for g in group_keys:
            navs = nav_curves[g]
            final_nav = navs[-1]
            tot_ret = (final_nav - 1.0) * 100.0
            years = max(1e-4, len(navs) / 250.0)
            ann_ret = ((final_nav ** (1.0 / years)) - 1.0) * 100.0 if final_nav > 0 else -100.0
            
            # 计算最大回撤
            cummax = pd.Series(navs).cummax()
            dd = (pd.Series(navs) - cummax) / cummax
            max_dd = float(dd.min()) * 100.0

            # 波动率与夏普
            series_rets = pd.Series(navs).pct_change().dropna()
            vol = float(series_rets.std() * np.sqrt(250.0))
            rf = 0.02
            sharpe = (ann_ret / 100.0 - rf) / (vol + 1e-6) if vol > 1e-4 else 0.0

            group_stats.append({
                "group": g,
                "total_return": round(tot_ret, 2),
                "annual_return": round(ann_ret, 2),
                "max_drawdown": round(max_dd, 2),
                "sharpe_ratio": round(sharpe, 2)
            })

        # 多空对冲绩效
        ls_final_nav = long_short_nav[-1]
        ls_tot_ret = (ls_final_nav - 1.0) * 100.0
        years = max(1e-4, len(long_short_nav) / 250.0)
        ls_ann_ret = ((ls_final_nav ** (1.0 / years)) - 1.0) * 100.0 if ls_final_nav > 0 else -100.0
        ls_cummax = pd.Series(long_short_nav).cummax()
        ls_dd = (pd.Series(long_short_nav) - ls_cummax) / ls_cummax
        ls_max_dd = float(ls_dd.min()) * 100.0

        # 单调性得分 (计算各组收益与组序号的秩相关)
        group_returns = [s["total_return"] for s in group_stats]
        group_ranks = list(range(1, len(group_returns) + 1))
        # 理想单调: group 1 > group 2 > group 3 -> 负相关 (-1.0)
        mono_corr, _ = spearmanr(group_ranks, group_returns)
        monotonicity_score = round(-float(mono_corr) if not np.isnan(mono_corr) else 0.0, 2)

        # 6. 整合返回数据结构
        return {
            "factor_name": factor_name,
            "symbols_count": len(symbols),
            "trading_days": len(trading_dates),
            "forward_period": forward_period,
            "quantiles_count": num_quantiles,
            "metrics": {
                "ic_mean": round(mean_ic, 4),
                "ic_std": round(std_ic, 4),
                "ic_ir": round(ic_ir, 4),
                "annualized_ir": round(annualized_ir, 2),
                "positive_ic_ratio": round(positive_ratio * 100.0, 1),
                "ic_t_stat": round(t_stat, 2),
                "monotonicity_score": monotonicity_score,
                "long_short_return": round(ls_tot_ret, 2),
                "long_short_annual": round(ls_ann_ret, 2),
                "long_short_max_dd": round(ls_max_dd, 2)
            },
            "ic_series": ic_records,
            "nav_dates": nav_dates,
            "quantile_navs": nav_curves,
            "long_short_nav": long_short_nav,
            "group_stats": group_stats
        }
