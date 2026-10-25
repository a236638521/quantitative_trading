import itertools
import logging
from typing import Dict, Any, List, Type
import pandas as pd

from .engine import BacktestEngine
from ..config import BacktestConfig
from ..strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class GridSearchOptimizer:
    """策略参数网格寻优与敏感性分析引擎"""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run_optimization(
        self,
        strategy_cls: Type[BaseStrategy],
        param_ranges: Dict[str, List[Any]],
        data_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
        symbol: str,
        target_metric: str = "sharpe_ratio"
    ) -> Dict[str, Any]:
        """
        执行参数网格遍历回测
        :param strategy_cls: 策略类
        :param param_ranges: 参数离散取值字典，例如 {'fast_period': [3, 5, 8, 10], 'slow_period': [15, 20, 30]}
        :param data_df: 标的行情数据 (单次预加载，内存重放)
        :param benchmark_df: 基准行情数据
        :param symbol: 标的代码
        :param target_metric: 优化目标 (sharpe_ratio, total_return, calmar_ratio, win_rate)
        """
        if not param_ranges:
            raise ValueError("参数取值范围不能为空")

        param_names = list(param_ranges.keys())
        param_value_lists = [param_ranges[k] for k in param_names]
        combinations = list(itertools.product(*param_value_lists))

        logger.info(f"开始参数寻优，总计 {len(combinations)} 组参数组合...")

        results: List[Dict[str, Any]] = []
        engine = BacktestEngine(config=self.config)

        for combo in combinations:
            params = dict(zip(param_names, combo))

            # 约束条件过滤 (例如: 双均线策略中快线周期必须严格小于慢线周期)
            if "fast_period" in params and "slow_period" in params:
                if params["fast_period"] >= params["slow_period"]:
                    continue
            if "entry_period" in params and "exit_period" in params:
                if params["exit_period"] >= params["entry_period"]:
                    continue

            try:
                strat = strategy_cls(params=params)
                res = engine.run(strat, data_df, benchmark_df, symbol)

                metric_val = res.get(target_metric, 0.0)
                results.append({
                    "params": params,
                    "target_metric_value": round(float(metric_val), 2),
                    "total_return": round(float(res.get("total_return", 0.0)), 2),
                    "annualized_return": round(float(res.get("annualized_return", 0.0)), 2),
                    "sharpe_ratio": round(float(res.get("sharpe_ratio", 0.0)), 2),
                    "max_drawdown": round(float(res.get("max_drawdown", 0.0)), 2),
                    "calmar_ratio": round(float(res.get("calmar_ratio", 0.0)), 2),
                    "win_rate": round(float(res.get("win_rate", 0.0)), 2),
                    "total_trades": int(res.get("total_trades", 0))
                })
            except Exception as e:
                logger.warning(f"参数 {params} 回测失败: {e}")

        if not results:
            return {
                "total_tested": len(combinations),
                "valid_results": 0,
                "best_params": {},
                "ranked_results": [],
                "heatmap_2d": None
            }

        # 根据目标指标降序排序 (如果是最大回撤则按绝对值升序)
        if target_metric == "max_drawdown":
            results.sort(key=lambda x: abs(x["max_drawdown"]))
        else:
            results.sort(key=lambda x: x["target_metric_value"], reverse=True)

        best_item = results[0]

        # 如果正好寻优 2 个参数，生成用于 ECharts 2D 敏感性热力图的数据结构
        heatmap_2d = None
        if len(param_names) == 2:
            p_x, p_y = param_names[0], param_names[1]
            x_vals = sorted(list(set(param_ranges[p_x])))
            y_vals = sorted(list(set(param_ranges[p_y])))

            # 构造二维点阵 [x_idx, y_idx, value]
            points = []
            param_map = {f"{r['params'][p_x]}_{r['params'][p_y]}": r for r in results}

            for x_idx, x_v in enumerate(x_vals):
                for y_idx, y_v in enumerate(y_vals):
                    key = f"{x_v}_{y_v}"
                    if key in param_map:
                        val = param_map[key].get(target_metric, 0.0)
                        points.append([x_idx, y_idx, round(float(val), 2)])
                    else:
                        points.append([x_idx, y_idx, None])

            heatmap_2d = {
                "param_x": p_x,
                "param_y": p_y,
                "x_labels": [str(x) for x in x_vals],
                "y_labels": [str(y) for y in y_vals],
                "points": points,
                "target_metric": target_metric
            }

        return {
            "total_tested": len(combinations),
            "valid_results": len(results),
            "best_params": best_item["params"],
            "best_metrics": best_item,
            "ranked_results": results[:20],  # 最佳前 20 组
            "heatmap_2d": heatmap_2d
        }
