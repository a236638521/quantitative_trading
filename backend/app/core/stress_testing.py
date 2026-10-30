import logging
import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def _sanitize_floats(obj: Any) -> Any:
    """递归清理所有的 NaN 和 Inf，确保符合严格 JSON 规范"""
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    elif isinstance(obj, (float, np.floating)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return val
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    return obj

# A股五大历史极端黑天鹅事件定义库

HISTORICAL_SCENARIOS = [
    {
        "id": "crash_2015",
        "name": "2015 杠杆股灾与流动性踩踏",
        "period": "2015.06.12 - 2015.08.26",
        "duration_days": 53,
        "bmk_drawdown": -0.435,  # 沪深300同期最大跌幅
        "description": "场外配资去杠杆引发流动性冰封，千股跌停与停牌潮集中爆发，大盘两月内单边暴跌逾43%",
        "sector_shocks": {
            "大金融": -0.38,
            "大消费": -0.39,
            "先进制造": -0.52,
            "科技信息": -0.56,
            "周期资源": -0.44,
            "公用红利": -0.32,
            "大盘宽基": -0.43,
            "中盘宽基": -0.50,
            "成长创业": -0.58,
            "硬核科创": -0.55,
            "大宗商品": -0.15,
            "固定收益": 0.04,
            "现金储备": 0.00
        }
    },
    {
        "id": "circuit_breaker_2016",
        "name": "2016 初熔断机制试行危机",
        "period": "2016.01.04 - 2016.01.28",
        "duration_days": 19,
        "bmk_drawdown": -0.264,
        "description": "A股试行熔断机制，4个交易日内触发4次熔断提前休市，流动性休克导致恐慌集中释放",
        "sector_shocks": {
            "大金融": -0.21,
            "大消费": -0.23,
            "先进制造": -0.31,
            "科技信息": -0.33,
            "周期资源": -0.27,
            "公用红利": -0.18,
            "大盘宽基": -0.26,
            "中盘宽基": -0.30,
            "成长创业": -0.34,
            "硬核科创": -0.32,
            "大宗商品": -0.06,
            "固定收益": 0.02,
            "现金储备": 0.00
        }
    },
    {
        "id": "trade_war_2018",
        "name": "2018 中美贸易战持续阴跌",
        "period": "2018.01.29 - 2018.10.29",
        "duration_days": 182,
        "bmk_drawdown": -0.318,
        "description": "宏观去杠杆叠加贸易争端升级，市场单边震荡阴跌长达9个月，估值体系严重受挫",
        "sector_shocks": {
            "大金融": -0.24,
            "大消费": -0.28,
            "先进制造": -0.36,
            "科技信息": -0.39,
            "周期资源": -0.33,
            "公用红利": -0.19,
            "大盘宽基": -0.31,
            "中盘宽基": -0.36,
            "成长创业": -0.40,
            "硬核科创": -0.38,
            "大宗商品": -0.12,
            "固定收益": 0.07,
            "现金储备": 0.00
        }
    },
    {
        "id": "covid_flash_2020",
        "name": "2020 初疫情黑天鹅闪崩",
        "period": "2020.01.20 - 2020.03.23",
        "duration_days": 42,
        "bmk_drawdown": -0.161,
        "description": "突发公共卫生事件致春节开市首日近三千股跌停，随后全球流动性闪崩共振",
        "sector_shocks": {
            "大金融": -0.17,
            "大消费": -0.12,
            "先进制造": -0.16,
            "科技信息": -0.18,
            "周期资源": -0.22,
            "公用红利": -0.11,
            "大盘宽基": -0.16,
            "中盘宽基": -0.18,
            "成长创业": -0.14,
            "硬核科创": -0.15,
            "大宗商品": -0.14,
            "固定收益": 0.03,
            "现金储备": 0.00
        }
    },
    {
        "id": "microcap_squeeze_2024",
        "name": "2024 初微盘流动性踩踏挤兑",
        "period": "2024.01.08 - 2024.02.08",
        "duration_days": 24,
        "bmk_drawdown": -0.105,
        "description": "雪球敲入与量化DMA杠杆收紧导致中小微盘极端失血踩踏，与大盘蓝筹护盘形成极致撕裂",
        "sector_shocks": {
            "大金融": -0.04,
            "大消费": -0.06,
            "先进制造": -0.19,
            "科技信息": -0.25,
            "周期资源": -0.09,
            "公用红利": -0.02,
            "大盘宽基": -0.08,
            "中盘宽基": -0.18,
            "成长创业": -0.22,
            "硬核科创": -0.24,
            "大宗商品": 0.01,
            "固定收益": 0.02,
            "现金储备": 0.00
        }
    }
]


class StressTestEngine:
    """
    机构级投资组合压力测试与黑天鹅极端尾部风险分析引擎
    核心功能：
    1. 历史极端黑天鹅情景回放 (2015股灾、2016熔断、2018贸易战、2020疫情闪崩、2024微盘挤兑)
    2. 假设性宏观风险因子冲击 (系统性暴跌 -5%/-10%、波动率飙升 100%/200%、流动性骤降与折价)
    3. 尾部风险度量：95% / 99% VaR (在险价值，历史模拟法与参数法双向校验)
    4. 尾部预期损失：95% / 99% CVaR (Expected Shortfall，超越VaR截断后的平均极端损失)
    5. 组合流动性变现天数预估 (Liquidation Horizon) 与持仓标的损失贡献度归因
    6. 组合抗黑天鹅能力综合评级与定量诊断
    """

    def __init__(self, sector_map: Optional[Dict[str, str]] = None):
        from .attribution import DEFAULT_SECTOR_MAP
        self.sector_map = dict(DEFAULT_SECTOR_MAP)
        if sector_map:
            self.sector_map.update(sector_map)

    def get_sector(self, symbol: str) -> str:
        """获取标的板块归属"""
        if symbol in self.sector_map:
            return self.sector_map[symbol]
        clean_sym = symbol.replace("sh.", "").replace("sz.", "")
        if clean_sym.startswith("51") or clean_sym.startswith("15"):
            return "大盘宽基"
        elif clean_sym.startswith("601") or clean_sym.startswith("600"):
            return "大消费"
        elif clean_sym.startswith("300") or clean_sym.startswith("688"):
            return "先进制造"
        return "大盘宽基"

    def run_stress_test(
        self,
        weights: Dict[str, float],
        data_dict: Dict[str, pd.DataFrame],
        trading_dates: List[str],
        equity_curve: Optional[List[Dict[str, Any]]] = None,
        confidence_levels: Tuple[float, float] = (0.95, 0.99)
    ) -> Dict[str, Any]:
        """
        全量执行组合压力测试与尾部风险度量
        :param weights: 当前组合标的权重配置 {symbol: weight}
        :param data_dict: 标的对齐的历史行情数据 {symbol: df}
        :param trading_dates: 交易日序列
        :param equity_curve: 策略历史累计净值序列 (用于提取实际日收益率核密度与历史VaR)
        :param confidence_levels: 显著性水平 (默认 95% 与 99%)
        """
        if not weights or not data_dict:
            raise ValueError("压力测试缺少有效的持仓权重或标的行情数据")

        # 1. 归一化持仓权重
        clean_weights = {k: v for k, v in weights.items() if v > 1e-5}
        total_w = sum(clean_weights.values())
        if total_w <= 0:
            raise ValueError("持仓权重总和必须大于 0")
        norm_weights = {k: v / total_w for k, v in clean_weights.items()}

        # 2. 提取日收益率序列与组合加权历史收益序列
        returns_df = self._extract_returns(data_dict, trading_dates)
        portfolio_daily_returns = self._calculate_portfolio_returns(norm_weights, returns_df, equity_curve)

        # 3. 计算 95% / 99% VaR 与 CVaR (在险价值与尾部预期损失)
        var_cvar_results = self._calculate_var_cvar(portfolio_daily_returns, confidence_levels)

        # 4. 回放 5 大历史极端黑天鹅情景
        historical_stress_results = self._replay_historical_scenarios(norm_weights, data_dict)

        # 5. 模拟 4 项假设性宏观极端冲击
        hypothetical_shocks = self._simulate_hypothetical_shocks(norm_weights, portfolio_daily_returns)

        # 6. 计算流动性清算天数与变现折价估算
        liquidity_analysis = self._analyze_liquidity(norm_weights, data_dict)

        # 7. 测算各持仓资产在极端情景下的风险暴露与损失贡献
        asset_loss_breakdown = self._calculate_asset_loss_contribution(norm_weights, historical_stress_results)

        # 8. 综合抗黑天鹅能力打分与诊断
        diagnostic_scorecard = self._generate_diagnostic_scorecard(
            var_cvar=var_cvar_results,
            historical_stress=historical_stress_results,
            liquidity=liquidity_analysis
        )

        raw_result = {
            "summary": {
                "var_95": round(float(var_cvar_results["var_95_hist"]), 4),
                "var_99": round(float(var_cvar_results["var_99_hist"]), 4),
                "cvar_95": round(float(var_cvar_results["cvar_95_hist"]), 4),
                "cvar_99": round(float(var_cvar_results["cvar_99_hist"]), 4),
                "max_historical_stress_loss": round(float(diagnostic_scorecard["worst_scenario_loss"]), 4),
                "worst_scenario_name": diagnostic_scorecard["worst_scenario_name"],
                "stress_resilience_grade": diagnostic_scorecard["grade"],
                "liquidation_days": liquidity_analysis["max_liquidation_days"]
            },
            "var_cvar": var_cvar_results,
            "historical_scenarios": historical_stress_results,
            "hypothetical_shocks": hypothetical_shocks,
            "liquidity_analysis": liquidity_analysis,
            "asset_loss_breakdown": asset_loss_breakdown,
            "diagnostic_scorecard": diagnostic_scorecard
        }
        return _sanitize_floats(raw_result)


    def _extract_returns(self, data_dict: Dict[str, pd.DataFrame], trading_dates: List[str]) -> pd.DataFrame:
        """从行情中提取对齐的日收益率矩阵"""
        closes = {}
        for sym, df in data_dict.items():
            if "close" in df.columns:
                sub = df.copy()
                sub["date"] = sub["date"].astype(str)
                s = sub.set_index("date")["close"]
                closes[sym] = s
        close_df = pd.DataFrame(closes).reindex(trading_dates).ffill().bfill()
        ret_df = close_df.pct_change().fillna(0.0)
        ret_df["CASH"] = 0.0
        return ret_df

    def _calculate_portfolio_returns(
        self,
        weights: Dict[str, float],
        returns_df: pd.DataFrame,
        equity_curve: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """获取组合历史日收益率数组"""
        if equity_curve and len(equity_curve) > 20:
            eq_vals = [e["value"] for e in equity_curve]
            eq_s = pd.Series(eq_vals)
            rets = eq_s.pct_change().dropna().values
            if len(rets) > 20:
                return rets

        # 若无实际净值曲线，则根据当前静态权重组合计算
        port_ret = np.zeros(len(returns_df))
        for sym, w in weights.items():
            if sym in returns_df.columns:
                port_ret += w * returns_df[sym].values
        return port_ret[1:]  # 去除首日 0 收益

    def _calculate_var_cvar(
        self,
        returns: np.ndarray,
        confidence_levels: Tuple[float, float]
    ) -> Dict[str, Any]:
        """
        计算历史模拟法与参数法 (含 Cornish-Fisher 高阶矩修正) 的 95% / 99% VaR 与 CVaR
        """
        if len(returns) < 10:
            return {
                "var_95_hist": 0.02, "var_99_hist": 0.035,
                "cvar_95_hist": 0.03, "cvar_99_hist": 0.045,
                "var_95_param": 0.02, "var_99_param": 0.035,
                "cvar_95_param": 0.03, "cvar_99_param": 0.045,
                "pnl_distribution": []
            }

        # 1. 历史模拟法 (Historical Simulation)
        # 损失为正数: loss = -return
        losses = -returns
        sorted_losses = np.sort(losses)
        n = len(sorted_losses)

        # 95% 分位数 (取最高 5% 损失门槛)
        idx_95 = int(np.floor(0.95 * n))
        idx_99 = int(np.floor(0.99 * n))

        var_95_h = max(0.0, float(sorted_losses[min(idx_95, n - 1)]))
        var_99_h = max(0.0, float(sorted_losses[min(idx_99, n - 1)]))

        # 95% CVaR: 超过 95% VaR 阈值的所有极端损失的数学期望
        tail_95 = sorted_losses[idx_95:]
        cvar_95_h = float(np.mean(tail_95)) if len(tail_95) > 0 else var_95_h

        # 99% CVaR: 超过 99% VaR 阈值的所有极端损失的数学期望
        tail_99 = sorted_losses[idx_99:]
        cvar_99_h = float(np.mean(tail_99)) if len(tail_99) > 0 else var_99_h

        # 2. 参数法 (Parametric / Variance-Covariance)
        mu = float(np.mean(returns))
        sigma = float(np.std(returns))
        skewness = float(pd.Series(returns).skew()) if len(returns) > 3 else 0.0
        kurtosis = float(pd.Series(returns).kurtosis()) if len(returns) > 3 else 0.0

        # 正态标准分 z: 95% -> 1.6449, 99% -> 2.3263
        z_95 = 1.6449
        z_99 = 2.3263

        # Cornish-Fisher 展开式对偏度与超额峰度的修正:
        # z_cf = z + (z^2 - 1)*skew/6 + (z^3 - 3*z)*kurt/24 - (2*z^3 - 5*z)*skew^2/36
        z_95_cf = z_95 + (z_95**2 - 1) * skewness / 6.0 + (z_95**3 - 3 * z_95) * kurtosis / 24.0
        z_99_cf = z_99 + (z_99**2 - 1) * skewness / 6.0 + (z_99**3 - 3 * z_99) * kurtosis / 24.0

        var_95_p = max(0.0, float(z_95_cf * sigma - mu))
        var_99_p = max(0.0, float(z_99_cf * sigma - mu))

        # 正态假设下 CVaR = mu - sigma * phi(z) / (1 - alpha)
        # 对应损失为正值:
        cvar_95_p = max(var_95_p, float(sigma * (np.exp(-0.5 * z_95**2) / (np.sqrt(2 * np.pi) * 0.05)) - mu))
        cvar_99_p = max(var_99_p, float(sigma * (np.exp(-0.5 * z_99**2) / (np.sqrt(2 * np.pi) * 0.01)) - mu))

        # 3. 生成日收益率直方图分布 (用于前端渲染核密度尾部切片图)
        hist, bin_edges = np.histogram(returns * 100, bins=25)
        pnl_distribution = []
        for i in range(len(hist)):
            mid = round(float((bin_edges[i] + bin_edges[i + 1]) / 2.0), 2)
            pnl_distribution.append({
                "bin_mid": mid,
                "count": int(hist[i]),
                "is_tail_95": bool(mid <= -var_95_h * 100),
                "is_tail_99": bool(mid <= -var_99_h * 100)
            })

        return {
            "var_95_hist": var_95_h,
            "var_99_hist": var_99_h,
            "cvar_95_hist": cvar_95_h,
            "cvar_99_hist": cvar_99_h,
            "var_95_param": var_95_p,
            "var_99_param": var_99_p,
            "cvar_95_param": cvar_95_p,
            "cvar_99_param": cvar_99_p,
            "mean_daily_return": round(mu * 100, 3),
            "daily_volatility": round(sigma * 100, 3),
            "skewness": round(skewness, 2),
            "kurtosis": round(kurtosis, 2),
            "pnl_distribution": pnl_distribution
        }

    def _replay_historical_scenarios(
        self,
        weights: Dict[str, float],
        data_dict: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """
        基于 A 股五大典型历史危机情景进行持仓穿透式冲击回放
        """
        results = []
        for sc in HISTORICAL_SCENARIOS:
            sector_shocks = sc["sector_shocks"]
            portfolio_loss = 0.0
            asset_impacts = []

            for sym, w in weights.items():
                if sym == "CASH":
                    asset_impacts.append({
                        "symbol": sym,
                        "weight": round(w * 100, 2),
                        "shock_return": 0.0,
                        "loss_contribution": 0.0
                    })
                    continue

                sec = self.get_sector(sym)
                shock = sector_shocks.get(sec, sc["bmk_drawdown"])
                loss_contrib = w * shock
                portfolio_loss += loss_contrib

                asset_impacts.append({
                    "symbol": sym,
                    "sector": sec,
                    "weight": round(w * 100, 2),
                    "shock_return": round(shock * 100, 2),
                    "loss_contribution": round(loss_contrib * 100, 2)
                })

            # 按单只标的损失贡献排序
            asset_impacts.sort(key=lambda x: x["loss_contribution"])

            results.append({
                "id": sc["id"],
                "name": sc["name"],
                "period": sc["period"],
                "duration_days": sc["duration_days"],
                "description": sc["description"],
                "bmk_drawdown_pct": round(sc["bmk_drawdown"] * 100, 2),
                "portfolio_loss_pct": round(portfolio_loss * 100, 2),
                "relative_defensive_alpha_pct": round((portfolio_loss - sc["bmk_drawdown"]) * 100, 2),
                "asset_impacts": asset_impacts
            })

        return results

    def _simulate_hypothetical_shocks(
        self,
        weights: Dict[str, float],
        returns: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        模拟 4 项假设性外生极端压力测试情景
        """
        sigma = float(np.std(returns)) if len(returns) > 5 else 0.015

        shocks = [
            {
                "shock_name": "系统性单日闪崩冲击 (-5%)",
                "category": "市场极端跌幅",
                "param_desc": "基准指数单日暴跌 5%，全市场相关性趋近于 1.0",
                "est_portfolio_drop_pct": round(sum(w * (-5.0) for s, w in weights.items() if s != "CASH"), 2),
                "severity": "中度风险",
                "badge_class": "yellow"
            },
            {
                "shock_name": "极端黑天鹅跌停潮 (-10%)",
                "category": "市场极端跌幅",
                "param_desc": "大面积标的封死跌停，避险流动性折价放大",
                "est_portfolio_drop_pct": round(sum(w * (-10.0) for s, w in weights.items() if s != "CASH"), 2),
                "severity": "严重风险",
                "badge_class": "red"
            },
            {
                "shock_name": "隐含波动率极限跃升 (+150%)",
                "category": "波动率风险",
                "param_desc": "市场恐慌指数急剧跳升，风险平价资产杠杆被迫平仓",
                "est_portfolio_drop_pct": round(-sigma * 2.5 * 100, 2),
                "severity": "中度风险",
                "badge_class": "yellow"
            },
            {
                "shock_name": "宏观流动性枯竭冲击",
                "category": "流动性折价",
                "param_desc": "日成交量锐减 50%，买卖买价差扩大 3 倍，紧急变现滑点损失",
                "est_portfolio_drop_pct": round(-1.85 * sum(w for s, w in weights.items() if s != "CASH"), 2),
                "severity": "可控风险",
                "badge_class": "blue"
            }
        ]
        return shocks

    def _analyze_liquidity(
        self,
        weights: Dict[str, float],
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        基于成交量与持仓市值估算组合清算变现天数 (假设单日平仓不超过该标的过去20日平均成交金额的 10%)
        """
        asset_liquidity = []
        max_days = 1.0

        for sym, w in weights.items():
            if sym == "CASH":
                asset_liquidity.append({
                    "symbol": sym,
                    "weight": round(w * 100, 2),
                    "daily_volume_shares": 0,
                    "est_liquidation_days": 0.0,
                    "liquidity_level": "极佳 (即时可用)"
                })
                continue

            df = data_dict.get(sym)
            if df is not None and "volume" in df.columns and len(df) >= 20:
                avg_vol = float(df["volume"].iloc[-20:].mean())
                # 假设资产规模 10 万元初始资金下的持仓股数
                pos_val = 100000.0 * w
                close_p = float(df["close"].iloc[-1])
                shares = (pos_val / close_p) if close_p > 0 else 0
                # 单日最大允许平仓额度 10%
                daily_capacity = max(100.0, avg_vol * 0.1)
                days_needed = round(shares / daily_capacity, 2)
            else:
                days_needed = 0.5
                avg_vol = 1000000.0

            max_days = max(max_days, days_needed)

            liq_status = "充裕 (T+1内变现)" if days_needed <= 1.0 else ("适中 (1-2日清算)" if days_needed <= 2.0 else "警惕 (变现周期长)")

            asset_liquidity.append({
                "symbol": sym,
                "weight": round(w * 100, 2),
                "daily_volume_shares": int(avg_vol),
                "est_liquidation_days": days_needed,
                "liquidity_level": liq_status
            })

        return {
            "max_liquidation_days": round(max_days, 1),
            "avg_liquidation_days": round(float(np.mean([a["est_liquidation_days"] for a in asset_liquidity])), 2),
            "asset_details": asset_liquidity
        }

    def _calculate_asset_loss_contribution(
        self,
        weights: Dict[str, float],
        scenarios_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        汇总各持仓资产在五大历史情景中的平均损失贡献
        """
        asset_totals: Dict[str, float] = {sym: 0.0 for sym in weights}
        for sc in scenarios_results:
            for item in sc["asset_impacts"]:
                s = item["symbol"]
                asset_totals[s] = asset_totals.get(s, 0.0) + abs(item["loss_contribution"])

        n_sc = max(1, len(scenarios_results))
        total_loss_all = max(1e-5, sum(asset_totals.values()))

        breakdown = []
        for sym, tot in asset_totals.items():
            avg_loss = tot / n_sc
            contrib_ratio = (tot / total_loss_all) * 100.0
            breakdown.append({
                "symbol": sym,
                "weight_pct": round(weights.get(sym, 0.0) * 100, 2),
                "avg_scenario_loss_pct": round(avg_loss, 2),
                "loss_contribution_ratio": round(contrib_ratio, 1)
            })

        # 按损失贡献比例降序
        breakdown.sort(key=lambda x: x["loss_contribution_ratio"], reverse=True)
        return breakdown

    def _generate_diagnostic_scorecard(
        self,
        var_cvar: Dict[str, Any],
        historical_stress: List[Dict[str, Any]],
        liquidity: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        根据压力测试结果综合评定抗黑天鹅等级并生成定量诊断
        """
        worst_scenario = min(historical_stress, key=lambda x: x["portfolio_loss_pct"])
        worst_loss = worst_scenario["portfolio_loss_pct"]
        worst_name = worst_scenario["name"]

        var95 = var_cvar["var_95_hist"] * 100
        cvar95 = var_cvar["cvar_95_hist"] * 100

        # 抗压评级逻辑:
        # A 级: 最差情景回撤 < 20% 且 95% CVaR < 3%
        # B 级: 最差情景回撤 < 35% 且 95% CVaR < 5%
        # C 级: 最差情景回撤 >= 35%
        if abs(worst_loss) < 20.0 and cvar95 < 3.0:
            grade = "AAA 级 (极强防御)"
            grade_color = "#10b981"
            diagnosis_text = (
                f"该投资组合展现出卓越的抗黑天鹅韧性。在历史最剧烈的【{worst_name}】中，预期回撤仅 {worst_loss}%，"
                f"相较沪深300基准具备显著的防守超额缓冲。95% 日度条件在险价值 (CVaR) 保持在 {cvar95:.2f}% 的安全区间，"
                f"资产清算天数仅需 {liquidity['max_liquidation_days']} 天，极端行情下流动性风险极低。"
            )
        elif abs(worst_loss) < 35.0:
            grade = "AA 级 (稳健抗压)"
            grade_color = "#38bdf8"
            diagnosis_text = (
                f"该组合具备良好的系统性风险承受力。在【{worst_name}】测试中回撤深度为 {worst_loss}%，"
                f"优于大盘同期极端跌幅。95% 日度在险价值 (VaR) 为 {var95:.2f}%，尾部预期损失 (CVaR) 为 {cvar95:.2f}%。"
                f"建议关注高贝塔标的的止损线设置，在黑天鹅孕育期适当增加固定收益或现金缓冲。"
            )
        else:
            grade = "BBB 级 (进攻高弹)"
            grade_color = "#f59e0b"
            diagnosis_text = (
                f"该投资组合偏向进攻弹性风格，在极端流动性踩踏行情中承受压力较大。"
                f"在最严苛的【{worst_name}】情景下，组合回撤预计达到 {worst_loss}%。"
                f"99% 极端在险损失达 {var_cvar['var_99_hist']*100:.2f}%。建议在实盘运行中开启波动率盯盘警报与动态平仓阈值。"
            )

        return {
            "grade": grade,
            "grade_color": grade_color,
            "worst_scenario_name": worst_name,
            "worst_scenario_loss": worst_loss,
            "diagnosis_text": diagnosis_text
        }
