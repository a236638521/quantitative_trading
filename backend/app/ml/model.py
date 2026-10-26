import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
import lightgbm as lgb

from .features import FEATURE_DEFINITIONS

logger = logging.getLogger(__name__)


class AlphaPredictor:
    """
    AI/机器学习 Alpha 预测器与时序滚动评估引擎
    特点：
    1. 支持 LightGBM 梯度提升决策树与 Ridge 岭回归
    2. 严格时序向前滚动验证 (Walk-Forward Validation)，绝对防止时序穿越
    3. 输出样本外预测 Rank-IC、累计 IC、方向一致性胜率
    4. 提取因子特征增益重要性 (Feature Gain Importance)
    5. 生成最新截面 AI 选股建议与打分排名
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        model_params: Optional[Dict[str, Any]] = None
    ):
        self.model_type = model_type.lower()
        self.model_params = model_params or {}
        self.model = None
        self.feature_names_map = {f["id"]: f["name"] for f in FEATURE_DEFINITIONS}

    def _create_model(self):
        if self.model_type == "lightgbm":
            params = {
                "n_estimators": self.model_params.get("n_estimators", 80),
                "learning_rate": self.model_params.get("learning_rate", 0.05),
                "max_depth": self.model_params.get("max_depth", 4),
                "num_leaves": self.model_params.get("num_leaves", 15),
                "min_child_samples": self.model_params.get("min_child_samples", 5),
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "random_state": 42,
                "verbosity": -1,
                "n_jobs": 1
            }
            return lgb.LGBMRegressor(**params)
        elif self.model_type == "ridge":
            alpha = self.model_params.get("alpha", 1.0)
            return Ridge(alpha=alpha, random_state=42)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def train_walk_forward(
        self,
        dataset_df: pd.DataFrame,
        feature_ids: List[str],
        forward_period: int = 5,
        test_ratio: float = 0.35,
        n_folds: int = 3
    ) -> Dict[str, Any]:
        """
        时序向前滚动训练与样本外回测评估 (Walk-Forward Validation)
        
        :param dataset_df: 带有特征与 target 标签的数据集
        :param feature_ids: 选用的特征标识列表
        :param forward_period: 预测步长 (天)
        :param test_ratio: 样本外测试集比例 (0.2 ~ 0.5)
        :param n_folds: 滚动切片轮数
        """
        if dataset_df.empty or len(dataset_df) < 50:
            raise ValueError("数据集样本量不足，无法训练机器学习模型！")

        dates = sorted(dataset_df["date"].unique())
        total_dates = len(dates)
        if total_dates < 40:
            raise ValueError(f"可用交易日数过少 ({total_dates} 日)，需至少 40 个交易日以进行时序训练。")

        # 划分训练起始与样本外测试阶段
        split_idx = int(total_dates * (1.0 - test_ratio))
        train_init_dates = dates[:split_idx]
        test_dates = dates[split_idx:]

        # 滚动切片
        fold_size = max(1, len(test_dates) // n_folds)
        out_of_sample_preds = []

        latest_trained_model = None

        for fold in range(n_folds):
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(test_dates)
            cur_test_dates = set(test_dates[test_start:test_end])
            if not cur_test_dates:
                continue

            # 严格时序划分：训练集只能使用测试集起始日期之前的数据
            min_test_date = min(cur_test_dates)
            cur_train_dates = set([d for d in dates if d < min_test_date])

            train_mask = dataset_df["date"].isin(cur_train_dates)
            test_mask = dataset_df["date"].isin(cur_test_dates)

            df_train = dataset_df[train_mask]
            df_test = dataset_df[test_mask]

            if len(df_train) < 30 or len(df_test) == 0:
                continue

            X_train = df_train[feature_ids].values
            y_train = df_train["target"].values
            X_test = df_test[feature_ids].values

            model = self._create_model()
            model.fit(X_train, y_train)
            latest_trained_model = model

            y_pred = model.predict(X_test)

            test_records = df_test[["date", "symbol", "close", "target"]].copy()
            test_records["pred_score"] = y_pred
            out_of_sample_preds.append(test_records)

        if not out_of_sample_preds:
            raise ValueError("时序切分未能生成有效的样本外预测结果")

        oos_df = pd.concat(out_of_sample_preds, ignore_index=True)

        # 1. 逐日计算样本外 Rank-IC
        ic_series = []
        cum_ic = 0.0
        unique_test_days = sorted(oos_df["date"].unique())

        for d in unique_test_days:
            sub = oos_df[oos_df["date"] == d].dropna(subset=["target", "pred_score"])
            if len(sub) >= 2:
                r, _ = spearmanr(sub["pred_score"], sub["target"])
                if not np.isnan(r):
                    cum_ic += r
                    ic_series.append({
                        "date": str(d),
                        "ic": round(float(r), 4),
                        "cumulative_ic": round(float(cum_ic), 4)
                    })

        # 2. IC 统计学评价
        if ic_series:
            ic_vals = [x["ic"] for x in ic_series]
            ic_mean = float(np.mean(ic_vals))
            ic_std = float(np.std(ic_vals))
            ic_ir = float(ic_mean / (ic_std + 1e-6))
            ann_ir = float(ic_ir * np.sqrt(250.0 / max(1, forward_period)))
            pos_ratio = float(np.mean([1.0 if x > 0 else 0.0 for x in ic_vals])) * 100.0
            t_stat = float(ic_mean / (ic_std / np.sqrt(len(ic_vals)) + 1e-6))
        else:
            ic_mean, ic_std, ic_ir, ann_ir, pos_ratio, t_stat = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # 3. 拟合误差与方向胜率 (Directional Accuracy)
        y_true_all = oos_df["target"].values
        y_pred_all = oos_df["pred_score"].values
        mse = float(np.mean((y_true_all - y_pred_all) ** 2))
        mae = float(np.mean(np.abs(y_true_all - y_pred_all)))
        dir_correct = float(np.mean((y_true_all * y_pred_all) > 0) * 100.0)

        # 4. 全量数据重拟合，获取全局特征重要性
        final_model = self._create_model()
        final_model.fit(dataset_df[feature_ids].values, dataset_df["target"].values)
        self.model = final_model

        feature_importances = []
        if self.model_type == "lightgbm":
            raw_gains = final_model.booster_.feature_importance(importance_type="gain")
            sum_gain = np.sum(raw_gains) if np.sum(raw_gains) > 0 else 1.0
            for fid, gain in zip(feature_ids, raw_gains):
                pct = round(float(gain / sum_gain) * 100.0, 2)
                feature_importances.append({
                    "id": fid,
                    "name": self.feature_names_map.get(fid, fid),
                    "importance_pct": pct,
                    "raw_score": float(gain)
                })
        elif self.model_type == "ridge":
            raw_coef = np.abs(final_model.coef_)
            sum_coef = np.sum(raw_coef) if np.sum(raw_coef) > 0 else 1.0
            for fid, coef in zip(feature_ids, raw_coef):
                pct = round(float(coef / sum_coef) * 100.0, 2)
                feature_importances.append({
                    "id": fid,
                    "name": self.feature_names_map.get(fid, fid),
                    "importance_pct": pct,
                    "raw_score": float(coef)
                })

        feature_importances.sort(key=lambda x: x["importance_pct"], reverse=True)

        return {
            "model_type": self.model_type,
            "metrics": {
                "ic_mean": round(ic_mean, 4),
                "ic_std": round(ic_std, 4),
                "ic_ir": round(ic_ir, 2),
                "annualized_ir": round(ann_ir, 2),
                "positive_ic_ratio": round(pos_ratio, 1),
                "t_stat": round(t_stat, 2),
                "mse": round(mse, 6),
                "mae": round(mae, 4),
                "directional_accuracy": round(dir_correct, 1),
                "oos_samples_count": len(oos_df),
                "oos_days_count": len(unique_test_days)
            },
            "ic_series": ic_series,
            "feature_importances": feature_importances,
            "oos_predictions": oos_df
        }

    def predict_latest_cross_section(
        self,
        latest_features_df: pd.DataFrame,
        feature_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        根据训练好的模型，对最新截面行情输出预期收益打分与选股建议
        """
        if self.model is None or latest_features_df.empty:
            return []

        # 获取最新一天的记录
        latest_date = latest_features_df["date"].max()
        cur_day_df = latest_features_df[latest_features_df["date"] == latest_date].copy()
        if cur_day_df.empty:
            return []

        X_latest = cur_day_df[feature_ids].values
        scores = self.model.predict(X_latest)
        cur_day_df["score"] = scores

        # 排序
        cur_day_df.sort_values(by="score", ascending=False, inplace=True)

        recommendations = []
        rank = 1
        pos_count = sum(cur_day_df["score"] > 0)

        for _, row in cur_day_df.iterrows():
            sc = float(row["score"])
            # 简单映射建议仓位：Top 3 且得分为正标的平分仓位
            if rank <= 3 and sc > 0:
                suggested_weight = round(1.0 / max(1, min(3, pos_count)), 2)
                action = "BUY"
            else:
                suggested_weight = 0.0
                action = "HOLD" if sc > 0 else "AVOID"

            recommendations.append({
                "rank": rank,
                "symbol": str(row["symbol"]),
                "date": str(latest_date),
                "close": round(float(row["close"]), 3),
                "score": round(sc, 4),
                "predicted_excess_return": f"{sc*100:+.2f}%",
                "action": action,
                "suggested_weight": suggested_weight,
                "reason": f"AI预测截面综合Alpha得分: {sc:+.4f} (排名 #{rank})"
            })
            rank += 1

        return recommendations
