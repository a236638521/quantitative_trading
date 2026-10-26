import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

FEATURE_DEFINITIONS = [
    {
        "id": "mom_5",
        "name": "5日价格动量",
        "category": "动量趋势",
        "description": "过去 5 个交易日的区间累计涨跌幅，反映超短期资金惯性"
    },
    {
        "id": "mom_10",
        "name": "10日价格动量",
        "category": "动量趋势",
        "description": "过去 10 个交易日的区间涨跌幅，反映双周级别趋势驱动"
    },
    {
        "id": "mom_20",
        "name": "20日价格动量",
        "category": "动量趋势",
        "description": "过去 20 个交易日的区间涨跌幅，经典月度顺势动量特征"
    },
    {
        "id": "mom_60",
        "name": "60日季度动量",
        "category": "动量趋势",
        "description": "过去 60 个交易日的区间涨跌幅，反映中长期大级别主升浪/主跌浪"
    },
    {
        "id": "vol_10",
        "name": "10日年化波动率",
        "category": "波动风险",
        "description": "短期日收益率标准差年化值，度量近期的风险剧烈程度"
    },
    {
        "id": "vol_20",
        "name": "20日年化波动率",
        "category": "波动风险",
        "description": "中期日收益率标准差年化值，识别低波动稳健标的与暴涨暴跌标的"
    },
    {
        "id": "vol_60",
        "name": "60日年化波动率",
        "category": "波动风险",
        "description": "中长期历史波动率基线，提供长期风险锚点"
    },
    {
        "id": "rsi_14",
        "name": "14日相对强弱 (RSI)",
        "category": "摆动指标",
        "description": "衡量近 14 日多空力量对比，识别超买 (>70) 与超卖 (<30)"
    },
    {
        "id": "bias_20",
        "name": "20日乖离率 (BIAS)",
        "category": "摆动指标",
        "description": "收盘价距离 20 日均线的偏离幅度百分比，反映均值回归引力"
    },
    {
        "id": "boll_pct_b",
        "name": "布林带分位值 (%b)",
        "category": "摆动指标",
        "description": "价格在布林带通道内的相对位置 (0=下轨, 0.5=中轨, 1=上轨)"
    },
    {
        "id": "vol_ratio_20",
        "name": "20日成交量比",
        "category": "量价流动性",
        "description": "当日成交量与过去 20 日平均成交量的比值，度量量能爆发度"
    },
    {
        "id": "amplitude_20",
        "name": "20日振幅均值",
        "category": "量价流动性",
        "description": "过去 20 日日内高低价差平均占比，度量盘中资金博弈活跃度"
    },
    {
        "id": "sharpe_mom_20",
        "name": "20日动量夏普比",
        "category": "复合因子",
        "description": "20日动量收益率除以20日已实现波动率，性价比趋势指标"
    },
    {
        "id": "reversal_5",
        "name": "5日短期反转",
        "category": "复合因子",
        "description": "超短期反向修正因子，捕捉急涨回调与急跌超跌反弹"
    }
]


class FeatureEngineer:
    """
    量化多因子特征工程处理器
    负责：
    1. 逐标的高性能时序特征生成 (动量、波动率、摆动指标、量价分布)
    2. 严格对齐多标的日历时序面板
    3. 截面去极值 (MAD Winsorization) 与截面 Z-Score 标准化
    4. 构建未来 K 日目标超额收益率 (Target Forward Return)
    """

    def __init__(self, selected_feature_ids: Optional[List[str]] = None):
        all_ids = [f["id"] for f in FEATURE_DEFINITIONS]
        if selected_feature_ids:
            self.feature_ids = [fid for fid in selected_feature_ids if fid in all_ids]
            if not self.feature_ids:
                self.feature_ids = all_ids
        else:
            self.feature_ids = all_ids

    def compute_single_symbol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """为单个标的的行情 DataFrame 计算全部时序衍生特征"""
        d = df.copy()
        close = d["close"].astype(float)
        high = d["high"].astype(float) if "high" in d.columns else close
        low = d["low"].astype(float) if "low" in d.columns else close
        volume = d["volume"].astype(float) if "volume" in d.columns else pd.Series(1.0, index=d.index)

        daily_ret = close.pct_change().fillna(0.0)

        # 1. 动量特征
        d["mom_5"] = (close / close.shift(5) - 1.0).fillna(0.0)
        d["mom_10"] = (close / close.shift(10) - 1.0).fillna(0.0)
        d["mom_20"] = (close / close.shift(20) - 1.0).fillna(0.0)
        d["mom_60"] = (close / close.shift(60) - 1.0).fillna(0.0)

        # 2. 波动率特征 (年化)
        d["vol_10"] = (daily_ret.rolling(window=10).std() * np.sqrt(250.0)).fillna(0.0)
        d["vol_20"] = (daily_ret.rolling(window=20).std() * np.sqrt(250.0)).fillna(0.0)
        d["vol_60"] = (daily_ret.rolling(window=60).std() * np.sqrt(250.0)).fillna(0.0)

        # 3. 相对强弱 RSI (14日)
        delta = close.diff().fillna(0.0)
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        d["rsi_14"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

        # 4. 20日乖离率 BIAS
        ma20 = close.rolling(window=20).mean()
        d["bias_20"] = ((close - ma20) / (ma20 + 1e-9)).fillna(0.0)

        # 5. 布林带分位数 %b
        std20 = close.rolling(window=20).std()
        upper = ma20 + 2.0 * std20
        lower = ma20 - 2.0 * std20
        d["boll_pct_b"] = ((close - lower) / (upper - lower + 1e-9)).clip(0.0, 1.0).fillna(0.5)

        # 6. 量价比与振幅
        vol_ma20 = volume.rolling(window=20).mean()
        d["vol_ratio_20"] = (volume / (vol_ma20 + 1e-9)).clip(0.1, 10.0).fillna(1.0)
        amplitude = ((high - low) / (close + 1e-9)).clip(0.0, 0.5)
        d["amplitude_20"] = amplitude.rolling(window=20).mean().fillna(0.0)

        # 7. 复合衍生因子
        d["sharpe_mom_20"] = (d["mom_20"] / (d["vol_20"] + 1e-4)).clip(-10.0, 10.0)
        d["reversal_5"] = -d["mom_5"]

        return d

    def build_ml_dataset(
        self,
        data_dict: Dict[str, pd.DataFrame],
        trading_dates: List[str],
        forward_period: int = 5,
        relative_target: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        全量提取标的池特征，执行截面标准化并构建未来收益标签。
        
        :param data_dict: 已对齐日历的各标的日线字典 {symbol: df}
        :param trading_dates: 完整交易日历
        :param forward_period: 预测步长 (例如 5 代表未来 5 日收益)
        :param relative_target: 是否将预测目标转为截面相对超额收益 (减去截面中位数)
        :return: (train_test_df, latest_features_df)
                 - train_test_df: 带有有效未来 target 标签的历史样本 (用于训练与测试)
                 - latest_features_df: 最新交易日的特征截面 (无未来标签，用于最新实时预测)
        """
        symbol_features: Dict[str, pd.DataFrame] = {}
        for sym, df in data_dict.items():
            sym_df = self.compute_single_symbol_features(df)
            # 计算向后 forward_period 日的真实收益率
            close = sym_df["close"].astype(float)
            sym_df["target_return"] = (close.shift(-forward_period) / close - 1.0)
            symbol_features[sym] = sym_df

        # 展开为多层面板或长表格
        records = []
        latest_records = []

        total_dates = len(trading_dates)
        for date_idx, cur_date in enumerate(trading_dates):
            is_latest_window = (date_idx >= total_dates - forward_period)

            cross_section = []
            for sym, sym_df in symbol_features.items():
                if date_idx < len(sym_df):
                    row = sym_df.iloc[date_idx]
                    feat_vals = {fid: float(row.get(fid, 0.0)) for fid in self.feature_ids}
                    target_val = row.get("target_return", np.nan)
                    item = {
                        "date": str(cur_date),
                        "symbol": sym,
                        "close": float(row["close"]),
                        "target": float(target_val) if not np.isnan(target_val) else np.nan,
                        **feat_vals
                    }
                    cross_section.append(item)

            if not cross_section:
                continue

            # 对当日截面执行特征标准化 (去极值 + 截面 Z-Score)
            cross_df = pd.DataFrame(cross_section)
            for fid in self.feature_ids:
                vals = cross_df[fid].values
                # MAD 去极值
                med = np.nanmedian(vals)
                mad = np.nanmedian(np.abs(vals - med))
                high_bound = med + 3.0 * 1.4826 * (mad + 1e-6)
                low_bound = med - 3.0 * 1.4826 * (mad + 1e-6)
                clipped = np.clip(vals, low_bound, high_bound)
                # 截面标准化
                std = np.nanstd(clipped)
                if std > 1e-6:
                    cross_df[fid] = (clipped - np.nanmean(clipped)) / std
                else:
                    cross_df[fid] = 0.0

            # 处理目标值：截面超额化
            if relative_target and not is_latest_window:
                targets = cross_df["target"].values
                valid_t = targets[~np.isnan(targets)]
                if len(valid_t) > 0:
                    med_t = np.median(valid_t)
                    cross_df["target"] = cross_df["target"] - med_t

            # 分流记录
            if is_latest_window:
                latest_records.extend(cross_df.to_dict("records"))
            else:
                train_ready = cross_df.dropna(subset=["target"])
                records.extend(train_ready.to_dict("records"))

        train_test_df = pd.DataFrame(records)
        latest_df = pd.DataFrame(latest_records)

        return train_test_df, latest_df
