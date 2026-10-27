"""
AlphaQuant Portfolio Optimizer & Risk Parity Engine
模块功能：
1. 均值-方差有效前沿求解 (Markowitz MVO & Efficient Frontier)
2. 最大夏普比率组合 (Maximum Sharpe Ratio Portfolio - MaxSharpe)
3. 全局最小方差组合 (Global Minimum Variance Portfolio - GMV)
4. 风险平价组合模型 (Risk Parity / Equal Risk Contribution - ERC)
5. 资产风险贡献度分解 (Marginal & Percentage Risk Contribution)
6. 投资组合蒙特卡洛模拟散点云 (Monte Carlo Simulation Cloud)
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    现代投资组合最优化与风险平价计算引擎
    """

    def __init__(self, risk_free_rate: float = 0.025, trading_days: int = 252):
        self.rf = float(risk_free_rate)
        self.annual_days = int(trading_days)

    def compute_returns_and_cov(
        self,
        returns_df: pd.DataFrame,
        shrinkage: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        计算各标的年化预期收益率与年化协方差矩阵 (支持 Ledoit-Wolf 简易收缩以提升病态矩阵条件数)
        """
        symbols = list(returns_df.columns)
        clean_df = returns_df.dropna()
        if len(clean_df) < 5:
            clean_df = returns_df.fillna(0.0)

        # 日均收益率年化
        daily_mean = clean_df.mean().values
        annual_returns = daily_mean * self.annual_days

        # 协方差矩阵年化
        sample_cov = clean_df.cov().values * self.annual_days

        # 协方差矩阵数值平滑与收缩 (提高病态矩阵求逆与二次规划稳定性)
        if shrinkage and len(symbols) > 1:
            n = len(symbols)
            var_mean = np.trace(sample_cov) / n
            target = np.eye(n) * var_mean
            shrink_factor = 0.12  # 12% 收缩向对角阵
            cov_matrix = (1.0 - shrink_factor) * sample_cov + shrink_factor * target
        else:
            cov_matrix = sample_cov

        # 保证对角线方差不为 0
        diag = np.diag(cov_matrix).copy()
        diag[diag < 1e-6] = 1e-4
        np.fill_diagonal(cov_matrix, diag)

        return annual_returns, cov_matrix, symbols

    def portfolio_performance(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        计算指定权重配置下的年化预期收益率、年化波动率与夏普比率
        """
        w = np.array(weights, dtype=float)
        port_ret = float(np.sum(w * expected_returns))
        port_var = float(w.T @ cov_matrix @ w)
        port_vol = float(np.sqrt(max(port_var, 1e-8)))
        sharpe = float((port_ret - self.rf) / port_vol) if port_vol > 0 else 0.0
        return port_ret, port_vol, sharpe

    def compute_risk_contributions(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        计算组合中各资产的边际风险贡献 (MRC)、绝对风险贡献 (RC) 与相对风险贡献百分比 (PRC)
        """
        w = np.array(weights, dtype=float)
        port_var = float(w.T @ cov_matrix @ w)
        port_vol = float(np.sqrt(max(port_var, 1e-8)))

        # 边际风险贡献 MRC = (Sigma @ w) / port_vol
        mrc = (cov_matrix @ w) / port_vol
        # 资产对总风险的贡献 RC = w * MRC (sum(RC) == port_vol)
        rc = w * mrc
        # 相对风险贡献百分比 PRC = RC / port_vol (sum(PRC) == 1.0)
        prc = rc / port_vol if port_vol > 0 else np.zeros_like(rc)

        return {
            "portfolio_volatility": float(port_vol),
            "marginal_risk_contributions": [float(x) for x in mrc],
            "risk_contributions": [float(x) for x in rc],
            "percentage_risk_contributions": [float(x) for x in prc]
        }

    def optimize_equal_weight(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        symbols: List[str]
    ) -> Dict[str, Any]:
        """
        朴素等权重基准模型 (1/N)
        """
        n = len(symbols)
        weights = np.ones(n) / n
        ret, vol, sharpe = self.portfolio_performance(weights, expected_returns, cov_matrix)
        rc_info = self.compute_risk_contributions(weights, cov_matrix)

        return {
            "model_name": "Equal Weight (1/N 等权)",
            "model_id": "equal_weight",
            "weights": {symbols[i]: round(float(weights[i]), 4) for i in range(n)},
            "expected_return": round(ret * 100, 2),
            "volatility": round(vol * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "risk_contributions": {symbols[i]: round(rc_info["percentage_risk_contributions"][i] * 100, 2) for i in range(n)}
        }

    def optimize_min_variance(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        symbols: List[str],
        max_weight: float = 0.5
    ) -> Dict[str, Any]:
        """
        全局最小方差组合 (Global Minimum Variance - GMV)
        目标：min w^T Sigma w  s.t. sum(w) = 1, 0 <= w_i <= max_weight
        """
        n = len(symbols)
        w0 = np.ones(n) / n

        def objective(w):
            return w.T @ cov_matrix @ w

        bounds = tuple((0.0, float(max_weight)) for _ in range(n))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        if not res.success or np.any(np.isnan(res.x)):
            weights = w0
        else:
            weights = np.clip(res.x, 0.0, 1.0)
            weights = weights / np.sum(weights)

        ret, vol, sharpe = self.portfolio_performance(weights, expected_returns, cov_matrix)
        rc_info = self.compute_risk_contributions(weights, cov_matrix)

        return {
            "model_name": "Global Min Variance (最小方差)",
            "model_id": "min_variance",
            "weights": {symbols[i]: round(float(weights[i]), 4) for i in range(n)},
            "expected_return": round(ret * 100, 2),
            "volatility": round(vol * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "risk_contributions": {symbols[i]: round(rc_info["percentage_risk_contributions"][i] * 100, 2) for i in range(n)}
        }

    def optimize_max_sharpe(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        symbols: List[str],
        max_weight: float = 0.5
    ) -> Dict[str, Any]:
        """
        最大夏普比率组合 (Maximum Sharpe Ratio / Tangency Portfolio)
        目标：max (w^T mu - rf) / sqrt(w^T Sigma w)
        """
        n = len(symbols)
        w0 = np.ones(n) / n

        def neg_sharpe(w):
            port_ret = np.sum(w * expected_returns)
            port_vol = np.sqrt(max(w.T @ cov_matrix @ w, 1e-8))
            return -(port_ret - self.rf) / port_vol

        bounds = tuple((0.0, float(max_weight)) for _ in range(n))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        if not res.success or np.any(np.isnan(res.x)):
            weights = w0
        else:
            weights = np.clip(res.x, 0.0, 1.0)
            weights = weights / np.sum(weights)

        ret, vol, sharpe = self.portfolio_performance(weights, expected_returns, cov_matrix)
        rc_info = self.compute_risk_contributions(weights, cov_matrix)

        return {
            "model_name": "Maximum Sharpe (最大夏普 MVO)",
            "model_id": "max_sharpe",
            "weights": {symbols[i]: round(float(weights[i]), 4) for i in range(n)},
            "expected_return": round(ret * 100, 2),
            "volatility": round(vol * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "risk_contributions": {symbols[i]: round(rc_info["percentage_risk_contributions"][i] * 100, 2) for i in range(n)}
        }

    def optimize_risk_parity(
        self,
        cov_matrix: np.ndarray,
        symbols: List[str],
        expected_returns: Optional[np.ndarray] = None,
        max_weight: float = 0.5
    ) -> Dict[str, Any]:
        """
        风险平价模型 (Risk Parity / Equal Risk Contribution - ERC)
        目标：使每个资产对投资组合总体方差的相对风险贡献相等 (RC_i = 1/N * port_vol)
        采用严格二次凸优化/对数罚项拟合，收敛稳定迅速
        """
        n = len(symbols)
        w0 = np.ones(n) / n
        target_rc = 1.0 / n

        def risk_budget_objective(w):
            port_var = w.T @ cov_matrix @ w
            if port_var <= 1e-8:
                return 1e6
            mrc = cov_matrix @ w
            rc = w * mrc
            prc = rc / port_var
            # 最小化各资产风险贡献率偏离 1/N 的残差平方和
            return np.sum((prc - target_rc) ** 2) * 10000.0

        bounds = tuple((0.01, float(max_weight)) for _ in range(n))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(
            risk_budget_objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-9}
        )

        if not res.success or np.any(np.isnan(res.x)):
            # 降级：波动率倒数加权 (Inverse Volatility)
            vols = np.sqrt(np.diag(cov_matrix))
            inv_vols = 1.0 / np.maximum(vols, 1e-4)
            weights = inv_vols / np.sum(inv_vols)
        else:
            weights = np.clip(res.x, 0.0, 1.0)
            weights = weights / np.sum(weights)

        if expected_returns is None:
            expected_returns = np.zeros(n)

        ret, vol, sharpe = self.portfolio_performance(weights, expected_returns, cov_matrix)
        rc_info = self.compute_risk_contributions(weights, cov_matrix)

        return {
            "model_name": "Risk Parity (风险平价 ERC)",
            "model_id": "risk_parity",
            "weights": {symbols[i]: round(float(weights[i]), 4) for i in range(n)},
            "expected_return": round(ret * 100, 2),
            "volatility": round(vol * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "risk_contributions": {symbols[i]: round(rc_info["percentage_risk_contributions"][i] * 100, 2) for i in range(n)}
        }

    def generate_efficient_frontier(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        symbols: List[str],
        n_points: int = 35,
        max_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        生成马科维茨有效前沿平滑曲线点集 (给定目标收益率求解最小方差)
        """
        n = len(symbols)
        min_ret = float(np.min(expected_returns))
        max_ret = float(np.max(expected_returns))

        if abs(max_ret - min_ret) < 1e-4:
            return []

        # 目标收益网格
        target_returns = np.linspace(min_ret, max_ret, n_points)
        frontier_points = []
        w0 = np.ones(n) / n
        bounds = tuple((0.0, float(max_weight)) for _ in range(n))

        for target_r in target_returns:
            def obj(w):
                return w.T @ cov_matrix @ w

            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w, tr=target_r: np.sum(w * expected_returns) - tr}
            ]

            res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)
            if res.success and not np.any(np.isnan(res.x)):
                w = np.clip(res.x, 0.0, 1.0)
                w = w / np.sum(w)
                r, v, s = self.portfolio_performance(w, expected_returns, cov_matrix)
                frontier_points.append({
                    "volatility": round(v * 100, 2),
                    "expected_return": round(r * 100, 2),
                    "sharpe_ratio": round(s, 2),
                    "weights": {symbols[i]: round(float(w[i]), 4) for i in range(n)}
                })

        # 按波动率升序排序
        frontier_points.sort(key=lambda x: x["volatility"])
        return frontier_points

    def simulate_random_portfolios(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        n_samples: int = 250,
        max_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        蒙特卡洛随机资产组合采样 (绘制经典风险-收益散点云)
        """
        n = len(expected_returns)
        samples = []

        for _ in range(n_samples):
            w = np.random.dirichlet(np.ones(n))
            w = np.minimum(w, max_weight)
            w = w / np.sum(w)

            ret, vol, sharpe = self.portfolio_performance(w, expected_returns, cov_matrix)
            samples.append({
                "volatility": round(vol * 100, 2),
                "expected_return": round(ret * 100, 2),
                "sharpe_ratio": round(sharpe, 2)
            })

        return samples

    def run_full_analysis(
        self,
        returns_df: pd.DataFrame,
        custom_expected_returns: Optional[Dict[str, float]] = None,
        max_weight: float = 0.5
    ) -> Dict[str, Any]:
        """
        一键执行多模型优化对比与有效前沿综合分析
        """
        annual_returns, cov_matrix, symbols = self.compute_returns_and_cov(returns_df)
        n = len(symbols)

        # 若提供了 AI 预测的预期超额收益，则与历史均值结合
        if custom_expected_returns:
            custom_rets = np.array([custom_expected_returns.get(s, annual_returns[i]) for i, s in enumerate(symbols)])
            # 融合收益: 60% AI 预期得分 + 40% 历史动量收益
            expected_returns = 0.6 * custom_rets + 0.4 * annual_returns
        else:
            expected_returns = annual_returns

        # 1. 各单资产指标
        asset_stats = []
        for i, sym in enumerate(symbols):
            v = np.sqrt(cov_matrix[i, i])
            r = expected_returns[i]
            s = (r - self.rf) / v if v > 0 else 0.0
            asset_stats.append({
                "symbol": sym,
                "expected_return": round(float(r) * 100, 2),
                "volatility": round(float(v) * 100, 2),
                "sharpe_ratio": round(float(s), 2)
            })

        # 2. 四大组合模型优化
        equal_w = self.optimize_equal_weight(expected_returns, cov_matrix, symbols)
        min_var = self.optimize_min_variance(expected_returns, cov_matrix, symbols, max_weight)
        max_shp = self.optimize_max_sharpe(expected_returns, cov_matrix, symbols, max_weight)
        risk_par = self.optimize_risk_parity(cov_matrix, symbols, expected_returns, max_weight)

        models = [equal_w, risk_par, max_shp, min_var]

        # 3. 有效前沿曲线与蒙特卡洛随机云
        frontier = self.generate_efficient_frontier(expected_returns, cov_matrix, symbols, n_points=35, max_weight=max_weight)
        simulations = self.simulate_random_portfolios(expected_returns, cov_matrix, n_samples=250, max_weight=max_weight)

        return {
            "symbols": symbols,
            "annual_days": self.annual_days,
            "risk_free_rate": round(self.rf * 100, 2),
            "asset_stats": asset_stats,
            "models": models,
            "efficient_frontier": frontier,
            "monte_carlo_simulations": simulations
        }
