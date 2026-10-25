import numpy as np
import pandas as pd
from typing import Dict, Any, List
from .portfolio import Portfolio, DailySnapshot, TradeRecord


class MetricsCalculator:
    """量化策略核心绩效评价指标计算引擎"""

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate  # 年化无风险利率 (默认2%)

    def calculate(self, portfolio: Portfolio, benchmark_symbol: str = "000300") -> Dict[str, Any]:
        snapshots: List[DailySnapshot] = portfolio.daily_snapshots
        if not snapshots or len(snapshots) < 2:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "benchmark_return": 0.0,
                "alpha": 0.0,
                "beta": 1.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_amount": 0.0,
                "calmar_ratio": 0.0,
                "annual_volatility": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_loss_ratio": 0.0,
                "equity_curve": [],
                "benchmark_curve": [],
                "drawdown_curve": []
            }

        # 整理每日资产时序
        df = pd.DataFrame([{
            "date": s.date,
            "total_asset": s.total_asset,
            "daily_return": s.daily_return,
            "benchmark_price": s.benchmark_price,
            "benchmark_return": s.benchmark_return
        } for s in snapshots])

        initial_asset = portfolio.initial_cash
        final_asset = df["total_asset"].iloc[-1]
        trading_days = len(df)

        # 1. 累计收益率与年化收益率
        total_return = (final_asset - initial_asset) / initial_asset
        years = max(trading_days / 250.0, 1.0 / 250.0)
        annualized_return = (1.0 + total_return) ** (1.0 / years) - 1.0

        # 2. 基准收益率
        benchmark_total_return = 0.0
        benchmark_annualized_return = 0.0
        if df["benchmark_price"].iloc[0] > 0 and df["benchmark_price"].iloc[-1] > 0:
            benchmark_total_return = (df["benchmark_price"].iloc[-1] - df["benchmark_price"].iloc[0]) / df["benchmark_price"].iloc[0]
            benchmark_annualized_return = (1.0 + benchmark_total_return) ** (1.0 / years) - 1.0

        # 3. 净值曲线与最大回撤
        df["cum_max"] = df["total_asset"].cummax()
        df["drawdown"] = (df["total_asset"] - df["cum_max"]) / df["cum_max"]
        df["drawdown_amt"] = df["cum_max"] - df["total_asset"]

        max_drawdown = float(df["drawdown"].min())  # 负数，如 -0.15
        max_drawdown_amount = float(df["drawdown_amt"].max())

        # 4. 年化波动率
        daily_returns = df["daily_return"].iloc[1:]  # 排除第一天0收益
        daily_vol = daily_returns.std() if len(daily_returns) > 1 else 0.0
        annual_volatility = float(daily_vol * np.sqrt(250))

        # 5. 夏普比率 (Sharpe Ratio)
        daily_rf = self.risk_free_rate / 250.0
        excess_daily_return = daily_returns - daily_rf
        sharpe_ratio = 0.0
        if daily_vol > 1e-6:
            sharpe_ratio = float((daily_returns.mean() - daily_rf) / daily_vol * np.sqrt(250))

        # 6. 索提诺比率 (Sortino Ratio) - 仅下行波动
        downside_returns = daily_returns[daily_returns < daily_rf]
        downside_std = downside_returns.std() if len(downside_returns) > 1 else 0.0
        sortino_ratio = 0.0
        if downside_std > 1e-6:
            sortino_ratio = float((daily_returns.mean() - daily_rf) / downside_std * np.sqrt(250))

        # 7. 卡玛比率 (Calmar Ratio)
        abs_max_dd = abs(max_drawdown)
        calmar_ratio = round(annualized_return / abs_max_dd, 2) if abs_max_dd > 1e-5 else 0.0

        # 8. Beta 与 Alpha (CAPM)
        beta = 1.0
        alpha = 0.0
        bmk_returns = df["benchmark_return"].iloc[1:]
        if len(bmk_returns) > 5 and bmk_returns.std() > 1e-6:
            cov = np.cov(daily_returns, bmk_returns)[0][1]
            var_bmk = np.var(bmk_returns)
            if var_bmk > 1e-6:
                beta = float(cov / var_bmk)
                alpha = float(annualized_return - (self.risk_free_rate + beta * (benchmark_annualized_return - self.risk_free_rate)))

        # 9. 交易统计 (仅统计已平仓卖单)
        sell_trades = [t for t in portfolio.trade_history if t.action == "SELL"]
        total_trades = len(sell_trades)
        winning_trades = [t for t in sell_trades if t.realized_pnl > 0]
        losing_trades = [t for t in sell_trades if t.realized_pnl < 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        avg_win = np.mean([t.realized_pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = abs(np.mean([t.realized_pnl for t in losing_trades])) if losing_trades else 0.0
        profit_loss_ratio = round(float(avg_win / avg_loss), 2) if avg_loss > 0 else (999.0 if avg_win > 0 else 0.0)

        # 10. 曲线序列输出 (便于 ECharts 渲染)
        equity_curve = []
        benchmark_curve = []
        drawdown_curve = []

        bmk_base = df["benchmark_price"].iloc[0] if df["benchmark_price"].iloc[0] > 0 else 1.0
        for _, row in df.iterrows():
            equity_curve.append({
                "date": row["date"],
                "value": round(row["total_asset"] / initial_asset, 4),
                "asset": round(row["total_asset"], 2)
            })
            bmk_val = round(row["benchmark_price"] / bmk_base, 4) if bmk_base > 0 else 1.0
            benchmark_curve.append({
                "date": row["date"],
                "value": bmk_val
            })
            drawdown_curve.append({
                "date": row["date"],
                "value": round(row["drawdown"] * 100, 2)  # 百分比负值
            })

        # 11. 月度收益热力矩阵计算
        monthly_returns = self._calculate_monthly_returns(df)

        return {
            "total_return": round(float(total_return) * 100, 2),                  # %
            "annualized_return": round(float(annualized_return) * 100, 2),        # %
            "benchmark_return": round(float(benchmark_total_return) * 100, 2),    # %
            "benchmark_annual_return": round(float(benchmark_annualized_return) * 100, 2),
            "alpha": round(float(alpha) * 100, 2),                                # %
            "beta": round(float(beta), 2),
            "sharpe_ratio": round(float(sharpe_ratio), 2),
            "sortino_ratio": round(float(sortino_ratio), 2),
            "max_drawdown": round(float(max_drawdown) * 100, 2),                  # % (负值)
            "max_drawdown_amount": round(float(max_drawdown_amount), 2),
            "calmar_ratio": calmar_ratio,
            "annual_volatility": round(float(annual_volatility) * 100, 2),        # %
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": round(float(win_rate) * 100, 2),                          # %
            "profit_loss_ratio": profit_loss_ratio,
            "trading_days": trading_days,
            "equity_curve": equity_curve,
            "benchmark_curve": benchmark_curve,
            "drawdown_curve": drawdown_curve,
            "monthly_returns": monthly_returns
        }

    def _calculate_monthly_returns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算月度与年度收益率矩阵 (用于热力图展示)"""
        if df.empty or "daily_return" not in df.columns:
            return {"years": [], "months": list(range(1, 13)), "data": {}}

        work_df = df.copy()
        work_df["dt"] = pd.to_datetime(work_df["date"])
        work_df["year"] = work_df["dt"].dt.year
        work_df["month"] = work_df["dt"].dt.month

        years = sorted(work_df["year"].unique().tolist())
        matrix_data = {}

        for y in years:
            y_df = work_df[work_df["year"] == y]
            matrix_data[str(y)] = {}

            # 计算各月份收益
            for m in range(1, 13):
                m_df = y_df[y_df["month"] == m]
                if not m_df.empty:
                    m_ret = float(np.prod(1.0 + m_df["daily_return"].values) - 1.0)
                    matrix_data[str(y)][str(m)] = round(m_ret * 100, 2)
                else:
                    matrix_data[str(y)][str(m)] = None

            # 计算全年累计收益
            y_ret = float(np.prod(1.0 + y_df["daily_return"].values) - 1.0)
            matrix_data[str(y)]["total"] = round(y_ret * 100, 2)

        return {
            "years": [str(y) for y in years],
            "months": list(range(1, 13)),
            "data": matrix_data
        }
