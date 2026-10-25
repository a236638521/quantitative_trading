import numpy as np
import pandas as pd
from typing import Dict


class CrossSectionProcessor:
    """截面因子处理引擎 (去极值、标准化与多因子合成)"""

    @staticmethod
    def winsorize_mad(series: pd.Series, n: float = 3.0) -> pd.Series:
        """MAD 绝对中位数差值去极值法 (比标准差更抗厚尾噪声)"""
        s = series.copy()
        median = s.median()
        mad = (s - median).abs().median()
        if mad < 1e-6:
            return s
        upper_limit = median + n * 1.4826 * mad
        lower_limit = median - n * 1.4826 * mad
        return s.clip(lower=lower_limit, upper=upper_limit)

    @staticmethod
    def standardize_zscore(series: pd.Series) -> pd.Series:
        """截面 Z-Score 标准化"""
        std = series.std()
        if std < 1e-6:
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std

    @staticmethod
    def rank_percentile(series: pd.Series, ascending: bool = True) -> pd.Series:
        """将因子值转换为 0.0 ~ 1.0 的百分位排名字段"""
        return series.rank(ascending=ascending, pct=True)

    @classmethod
    def process_and_combine(
        cls,
        factors_dict: Dict[str, pd.Series],
        weights: Dict[str, float]
    ) -> pd.Series:
        """
        截面多因子流水线处理与加权合成
        1. MAD 去极值
        2. Z-Score 标准化
        3. 按权重合成综合 Alpha 得分
        """
        if not factors_dict:
            return pd.Series()

        processed_dfs = []
        for name, series in factors_dict.items():
            cleaned = cls.winsorize_mad(series)
            stded = cls.standardize_zscore(cleaned)
            weight = weights.get(name, 1.0)
            processed_dfs.append(stded * weight)

        # 加权求和
        composite = pd.concat(processed_dfs, axis=1).sum(axis=1)
        return composite
