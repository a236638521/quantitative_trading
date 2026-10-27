import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

from .features import FeatureEngineer
from .model import AlphaPredictor
from .optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)


class AIMachineLearningStrategy:
    """
    AI 机器学习截面 Alpha 选股驱动策略
    实战机制：
    1. 前置预处理：利用多因子特征工程提取 14+ 维技术与量价特征
    2. 模型滚动学习：基于 LightGBM / Ridge 时序预测未来 N 日超额收益
    3. 截面动态打分：在每个调仓日（周频/双周/月频），提取模型对候选池所有标的的预测 Alpha 得分
    4. Top-K 资产配置：支持 等权配置 (Equal Weight)、风险平价模型 (Risk Parity)、最大夏普组合 (Max Sharpe)、全局最小方差 (GMV)
    5. 全市场下行防守：当池内所有标的预期得分皆为负时，自动空仓避险
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.name = "AIMachineLearningStrategy"
        default_params = {
            "model_type": "lightgbm",         # "lightgbm" 或 "ridge"
            "forward_period": 5,              # 预测未来 5 日收益
            "top_n": 2,                       # 每次精选 Top N 只标的
            "rebalance_freq": "weekly",       # "weekly", "biweekly", "monthly"
            "allocation_method": "equal_weight", # "equal_weight", "risk_parity", "max_sharpe", "min_variance"
            "min_score": -0.05,               # 预期得分下限门槛
            "feature_ids": None               # 选用特征列表 (默认全量)
        }
        if params:
            default_params.update(params)
        self.params = default_params

        self.feature_engineer = FeatureEngineer(selected_feature_ids=self.params.get("feature_ids"))
        self.predictor = AlphaPredictor(model_type=self.params.get("model_type", "lightgbm"))
        self.optimizer = PortfolioOptimizer()
        self.daily_scores: Dict[Tuple[str, str], float] = {}
        self.training_summary: Dict[str, Any] = {}
        self.data_dict: Dict[str, pd.DataFrame] = {}
        self.trading_dates: List[str] = []

    def prepare(self, data_dict: Dict[str, pd.DataFrame], trading_dates: List[str]) -> Dict[str, pd.DataFrame]:
        """
        回测前执行 AI 模型离线训练与全时序样本外打分生成
        """
        self.data_dict = data_dict
        self.trading_dates = trading_dates
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
            alloc_method = self.params.get("allocation_method", "equal_weight")
            if alloc_method == "equal_weight" or len(selected) == 1:
                weight_per_asset = round(0.95 / len(selected), 4)
                for sym in selected:
                    target_weights[sym] = weight_per_asset
            else:
                # 提取最近 60 个交易日的历史收益率计算协方差矩阵进行优化
                lookback_days = 60
                start_hist_idx = max(0, date_idx - lookback_days)
                hist_dates = self.trading_dates[start_hist_idx:date_idx]

                ret_dict = {}
                for sym in selected:
                    sym_df = self.data_dict.get(sym)
                    if sym_df is not None:
                        sub_df = sym_df[sym_df["date"].isin(hist_dates)]
                        if "pct_chg" in sub_df.columns:
                            ret_dict[sym] = sub_df["pct_chg"].values / 100.0
                        elif "close" in sub_df.columns:
                            ret_dict[sym] = sub_df["close"].pct_change().fillna(0.0).values

                min_len = min([len(v) for v in ret_dict.values()]) if ret_dict else 0
                if min_len >= 15:
                    ret_df = pd.DataFrame({k: v[-min_len:] for k, v in ret_dict.items()})
                    ann_rets, cov_mat, sym_order = self.optimizer.compute_returns_and_cov(ret_df)

                    opt_res = None
                    try:
                        if alloc_method == "risk_parity":
                            opt_res = self.optimizer.optimize_risk_parity(cov_mat, sym_order, ann_rets, max_weight=0.7)
                        elif alloc_method == "max_sharpe":
                            custom_r = np.array([sym_scores.get(s, ann_rets[i]) for i, s in enumerate(sym_order)])
                            opt_res = self.optimizer.optimize_max_sharpe(custom_r, cov_mat, sym_order, max_weight=0.7)
                        elif alloc_method == "min_variance":
                            opt_res = self.optimizer.optimize_min_variance(ann_rets, cov_mat, sym_order, max_weight=0.7)
                    except Exception as e:
                        logger.warning(f"调仓权重优化求解异常: {e}，降级为等权配置")
                        opt_res = None

                    if opt_res and "weights" in opt_res:
                        # 留 5% 现金作为规费与滑点缓冲
                        for sym, w in opt_res["weights"].items():
                            target_weights[sym] = round(float(w) * 0.95, 4)
                    else:
                        weight_per_asset = round(0.95 / len(selected), 4)
                        for sym in selected:
                            target_weights[sym] = weight_per_asset
                else:
                    weight_per_asset = round(0.95 / len(selected), 4)
                    for sym in selected:
                        target_weights[sym] = weight_per_asset
        else:
            # 市场预期全面走弱，全部清仓避险
            for sym in bar_map.keys():
                target_weights[sym] = 0.0

        return target_weights
