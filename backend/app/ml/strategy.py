import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

from .features import FeatureEngineer
from .model import AlphaPredictor

logger = logging.getLogger(__name__)


class AIMachineLearningStrategy:
    """
    AI 机器学习截面 Alpha 选股驱动策略
    实战机制：
    1. 前置预处理：利用多因子特征工程提取 12+ 维技术与动量特征
    2. 模型滚动学习：基于 LightGBM / Ridge 时序预测未来 N 日收益率
    3. 截面动态打分：在每个调仓日（周频/月频），提取模型对候选池所有标的的预测 Alpha 得分
    4. Top-K 等权持仓：做多模型预测得分最高的 Top N 标的
    5. 全市场下行防守：当池内所有标的预期得分皆为负时，自动空仓避险
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.name = "AIMachineLearningStrategy"
        default_params = {
            "model_type": "lightgbm",     # "lightgbm" 或 "ridge"
            "forward_period": 5,          # 预测未来 5 日收益
            "top_n": 2,                   # 每次精选 Top 2 只标的
            "rebalance_freq": "weekly",   # "weekly" (周度调仓) / "monthly" (月度调仓)
            "min_score": 0.0,             # 买入最低预期得分门槛
            "feature_ids": None           # 选用特征列表 (默认全量)
        }
        if params:
            default_params.update(params)
        self.params = default_params

        self.feature_engineer = FeatureEngineer(selected_feature_ids=self.params.get("feature_ids"))
        self.predictor = AlphaPredictor(model_type=self.params.get("model_type", "lightgbm"))
        self.daily_scores: Dict[Tuple[str, str], float] = {}
        self.training_summary: Dict[str, Any] = {}

    def prepare(self, data_dict: Dict[str, pd.DataFrame], trading_dates: List[str]) -> Dict[str, pd.DataFrame]:
        """
        回测前执行 AI 模型离线训练与全时序样本外打分生成
        """
        forward_period = int(self.params.get("forward_period", 5))

        # 1. 构建机器学习数据集
        train_test_df, latest_df = self.feature_engineer.build_ml_dataset(
            data_dict=data_dict,
            trading_dates=trading_dates,
            forward_period=forward_period,
            relative_target=True
        )

        # 2. 训练与时序向前滚动验证 (Walk-Forward)
        feature_ids = self.feature_engineer.feature_ids
        eval_result = self.predictor.train_walk_forward(
            dataset_df=train_test_df,
            feature_ids=feature_ids,
            forward_period=forward_period,
            test_ratio=0.4,
            n_folds=3
        )
        self.training_summary = eval_result

        # 3. 提取历史样本外预测得分并建立 (date, symbol) -> score 快速映射字典
        oos_preds = eval_result.get("oos_predictions", pd.DataFrame())
        self.daily_scores = {}
        if not oos_preds.empty:
            for _, r in oos_preds.iterrows():
                key = (str(r["date"]), str(r["symbol"]))
                self.daily_scores[key] = float(r["pred_score"])

        # 为每个标的 DataFrame 追加 score 列 (供诊断与展示)
        prepared_dict = {}
        for sym, df in data_dict.items():
            d = df.copy()
            scores = []
            for _, row in d.iterrows():
                dt_str = str(row["date"])
                sc = self.daily_scores.get((dt_str, sym), np.nan)
                scores.append(sc)
            d["ai_score"] = scores
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
        判断当前交易日是否执行调仓，并输出各标的目标持仓权重
        """
        # 前期特征窗口需要至少 30 个交易日积累
        if date_idx < 30:
            return None

        # 调仓周期判断
        rebal_freq = self.params.get("rebalance_freq", "weekly")
        cur_dt = pd.to_datetime(cur_date)

        is_rebal_day = False
        if rebal_freq == "weekly":
            # 每周调仓
            if date_idx % 5 == 0:
                is_rebal_day = True
        elif rebal_freq == "biweekly":
            # 双周调仓
            if date_idx % 10 == 0:
                is_rebal_day = True
        elif rebal_freq == "monthly":
            # 每月调仓
            if date_idx % 20 == 0:
                is_rebal_day = True
        else:
            is_rebal_day = True

        if not is_rebal_day:
            return None

        # 截面获取各标的当日的 AI 预测得分
        sym_scores = {}
        for sym in bar_map.keys():
            sc = self.daily_scores.get((cur_date, sym))
            if sc is not None and not np.isnan(sc):
                sym_scores[sym] = sc

        if not sym_scores:
            return None

        # 按得分从高到低排序
        sorted_syms = sorted(sym_scores.items(), key=lambda x: x[1], reverse=True)

        top_n = int(self.params.get("top_n", 2))
        min_score = float(self.params.get("min_score", -0.05))

        # 筛选得分超过阈值的头部标的
        selected = [s for s, sc in sorted_syms[:top_n] if sc >= min_score]

        target_weights: Dict[str, float] = {}
        if selected:
            weight_per_asset = round(0.95 / len(selected), 4)
            for sym in selected:
                target_weights[sym] = weight_per_asset
        else:
            # 市场预期全面走弱，全部清仓避险
            for sym in bar_map.keys():
                target_weights[sym] = 0.0

        return target_weights
