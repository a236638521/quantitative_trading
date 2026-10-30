import logging
import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# A股主流标的板块与行业分类映射字典 (涵盖主流宽基ETF、大类资产及典型A股股票)
DEFAULT_SECTOR_MAP = {
    # 宽基与风格类 ETF
    "sh.510300": "大盘宽基",
    "sh.510500": "中盘宽基",
    "sz.159915": "成长创业",
    "sh.588000": "硬核科创",
    # 大类资产 ETF
    "sh.518880": "大宗商品",
    "sh.511010": "固定收益",
    # 消费与医药白马
    "sh.600519": "大消费",
    "sz.000858": "大消费",
    "sz.000568": "大消费",
    "sh.600887": "大消费",
    "sz.000651": "大消费",
    "sh.603288": "大消费",
    # 大金融板块
    "sh.601318": "大金融",
    "sz.000001": "大金融",
    "sh.600036": "大金融",
    "sh.601166": "大金融",
    "sh.600030": "大金融",
    # 高端制造与新能源
    "sz.300750": "先进制造",
    "sz.002594": "先进制造",
    "sh.601012": "先进制造",
    "sh.600438": "先进制造",
    # 周期资源与红利公用
    "sh.600900": "公用红利",
    "sh.601088": "周期资源",
    "sh.601857": "周期资源",
    "sh.600028": "周期资源",
    "sh.600585": "周期资源",
    # TMT 与科技芯片
    "sz.002415": "科技信息",
    "sh.600584": "科技信息",
    "sz.000063": "科技信息",
    "sz.300059": "大金融",
    "CASH": "现金储备"
}

# 基准 (沪深300) 典型板块基准权重分布 (作为无细粒度成分时的基准权重中枢)
DEFAULT_BENCHMARK_SECTOR_WEIGHTS = {
    "大金融": 0.24,
    "大消费": 0.21,
    "先进制造": 0.20,
    "科技信息": 0.14,
    "周期资源": 0.12,
    "公用红利": 0.06,
    "大盘宽基": 0.03,
    "中盘宽基": 0.00,
    "成长创业": 0.00,
    "硬核科创": 0.00,
    "大宗商品": 0.00,
    "固定收益": 0.00,
    "现金储备": 0.00
}


class PerformanceAttributor:
    """
    机构级投资组合业绩归因与多因子风格暴露分析引擎
    核心功能：
    1. Brinson-Fachler (BF) 行业配置效应、选股效应与交互效应分解
    2. Carino 多期复利平滑连接算法 (消除多期复利几何残差，实现超额收益精确闭环)
    3. 超额收益流向瀑布图分解 (基准收益 -> 行业配置 -> 选股Alpha -> 交互协同 -> 交易费损 -> 最终组合净收益)
    4. Barra 体系 6 维多因子风格暴露 (Beta、Size、Value、Momentum、Volatility、Liquidity)
    5. 主动风格偏离度 (Active Tilts) 与组合风格定性诊断报告
    """

    def __init__(self, sector_map: Optional[Dict[str, str]] = None):
        self.sector_map = dict(DEFAULT_SECTOR_MAP)
        if sector_map:
            self.sector_map.update(sector_map)

    def get_sector(self, symbol: str) -> str:
        """获取标的所属板块/行业分类"""
        if symbol in self.sector_map:
            return self.sector_map[symbol]
        # 根据代码规则智能兜底归类
        clean_sym = symbol.replace("sh.", "").replace("sz.", "")
        if clean_sym.startswith("51") or clean_sym.startswith("15"):
            return "大盘宽基"
        elif clean_sym.startswith("601") or clean_sym.startswith("600"):
            return "大消费"
        elif clean_sym.startswith("300") or clean_sym.startswith("688"):
            return "先进制造"
        else:
            return "其他板块"

    def analyze(
        self,
        allocation_history: List[Dict[str, Any]],
        data_dict: Dict[str, pd.DataFrame],
        trading_dates: List[str],
        benchmark_curve: List[Dict[str, Any]],
        equity_curve: List[Dict[str, Any]],
        trades: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        全量执行组合业绩归因与风格暴露分析
        :param allocation_history: 逐日或各调仓日持仓权重字典 [{"date": str, "weights": {sym: weight}}]
        :param data_dict: 各标的已对齐的日线行情数据 {symbol: df}
        :param trading_dates: 完整对齐的交易日序列
        :param benchmark_curve: 基准累计净值序列 [{"date": str, "value": float}]
        :param equity_curve: 策略累计净值序列 [{"date": str, "value": float}]
        :param trades: 调仓成交明细列表 (用于精准统计佣金印花税磨损)
        """
        if not allocation_history or not benchmark_curve or not equity_curve:
            raise ValueError("归因分析缺少必要的数据源 (持仓历史/净值序列/基准序列)")

        # 1. 计算单期收益与基准各板块收益
        returns_df = self._calculate_asset_returns(data_dict, trading_dates)
        bmk_series = pd.Series(
            [b["value"] for b in benchmark_curve],
            index=[b["date"] for b in benchmark_curve]
        )
        bmk_ret_series = bmk_series.pct_change().fillna(0.0)

        strat_series = pd.Series(
            [e["value"] for e in equity_curve],
            index=[e["date"] for e in equity_curve]
        )
        strat_ret_series = strat_series.pct_change().fillna(0.0)

        # 2. 运行多期 Brinson-Fachler 归因 (含 Carino 多期平滑调整)
        brinson_result = self._calculate_brinson_bf(
            allocation_history=allocation_history,
            returns_df=returns_df,
            bmk_ret_series=bmk_ret_series,
            strat_ret_series=strat_ret_series
        )

        # 3. 统计交易摩擦费用对超额收益的扣减
        total_costs = 0.0
        if trades:
            for t in trades:
                comm = t.get("commission", 0.0) or 0.0
                tax = t.get("stamp_tax", 0.0) or t.get("tax", 0.0) or 0.0
                trans = t.get("transfer_fee", 0.0) or 0.0
                total_costs += (comm + tax + trans)
        
        init_asset = equity_curve[0]["value"] if equity_curve else 1.0
        # 规费率拖累 (相对于初始本金)
        cost_drag = total_costs / init_asset if init_asset > 100.0 else (total_costs / 100000.0)

        # 4. 构建超额收益流向瀑布图 (Waterfall Breakdown)
        total_strat_return = (strat_series.iloc[-1] - strat_series.iloc[0]) / strat_series.iloc[0] if len(strat_series) > 1 else 0.0
        total_bmk_return = (bmk_series.iloc[-1] - bmk_series.iloc[0]) / bmk_series.iloc[0] if len(bmk_series) > 1 else 0.0
        total_excess_return = total_strat_return - total_bmk_return

        waterfall_data = self._build_waterfall(
            total_bmk_return=total_bmk_return,
            brinson_summary=brinson_result["summary"],
            cost_drag=cost_drag,
            total_strat_return=total_strat_return
        )

        # 5. Barra 体系 6 维多因子风格暴露分析
        style_exposure = self._calculate_style_exposure(
            allocation_history=allocation_history,
            data_dict=data_dict,
            trading_dates=trading_dates,
            bmk_ret_series=bmk_ret_series,
            strat_ret_series=strat_ret_series
        )

        return {
            "summary": {
                "total_strategy_return": round(float(total_strat_return), 4),
                "total_benchmark_return": round(float(total_bmk_return), 4),
                "total_excess_return": round(float(total_excess_return), 4),
                "allocation_effect": round(float(brinson_result["summary"]["allocation"]), 4),
                "selection_effect": round(float(brinson_result["summary"]["selection"]), 4),
                "interaction_effect": round(float(brinson_result["summary"]["interaction"]), 4),
                "cost_drag": round(float(cost_drag), 4)
            },
            "sector_attribution": brinson_result["sector_table"],
            "cumulative_curves": brinson_result["cumulative_series"],
            "waterfall": waterfall_data,
            "style_exposure": style_exposure
        }

    def _calculate_asset_returns(
        self,
        data_dict: Dict[str, pd.DataFrame],
        trading_dates: List[str]
    ) -> pd.DataFrame:
        """从各标的收盘价中提取逐日收益率矩阵"""
        closes = {}
        for sym, df in data_dict.items():
            if "close" in df.columns:
                sub = df.copy()
                sub["date"] = sub["date"].astype(str)
                s = sub.set_index("date")["close"]
                closes[sym] = s

        close_df = pd.DataFrame(closes).reindex(trading_dates).ffill().bfill()
        ret_df = close_df.pct_change().fillna(0.0)
        # 现金收益率为 0 (或活期微利)
        ret_df["CASH"] = 0.0
        return ret_df

    def _calculate_brinson_bf(
        self,
        allocation_history: List[Dict[str, Any]],
        returns_df: pd.DataFrame,
        bmk_ret_series: pd.Series,
        strat_ret_series: pd.Series
    ) -> Dict[str, Any]:
        """
        执行 Brinson-Fachler (BF) 行业多期业绩归因分解
        BF 分解公式：
        - 行业配置效应: (wp_i - wb_i) * (Rb_i - Rb)
        - 个股选择效应: wb_i * (Rp_i - Rb_i)
        - 交互协同效应: (wp_i - wb_i) * (Rp_i - Rb_i)
        """
        # 将调仓历史转为字典索引，方便按交易日提取持仓权重
        alloc_dict = {}
        last_weights = {}
        for item in allocation_history:
            d = str(item["date"])
            w = item.get("weights", {})
            if w:
                last_weights = w
            alloc_dict[d] = dict(last_weights)

        dates = returns_df.index.tolist()
        n_days = len(dates)

        # 识别全部涉及的板块
        all_symbols = [col for col in returns_df.columns if col != "CASH"]
        all_sectors = sorted(list(set(self.get_sector(s) for s in all_symbols) | set(DEFAULT_BENCHMARK_SECTOR_WEIGHTS.keys())))

        # 记录每期的各项效应
        daily_alloc_eff = []
        daily_selec_eff = []
        daily_inter_eff = []

        # 行业累计累加器 {sector: {"alloc": 0, "select": 0, "inter": 0, "wp_sum": 0, "wb_sum": 0, "count": 0}}
        sector_accum = {
            sec: {"alloc": 0.0, "select": 0.0, "inter": 0.0, "wp_sum": 0.0, "wb_sum": 0.0, "rp_sum": 0.0, "rb_sum": 0.0, "count": 0}
            for sec in all_sectors
        }

        # 计算整个回测期复利收益率，用于 Carino 多期平滑连接因子
        strat_cum = (1.0 + strat_ret_series).prod() - 1.0
        bmk_cum = (1.0 + bmk_ret_series).prod() - 1.0
        cum_excess = strat_cum - bmk_cum

        # Carino 全局常数因子
        log_diff_total = math.log(max(1e-6, 1.0 + strat_cum)) - math.log(max(1e-6, 1.0 + bmk_cum))
        k_global = (cum_excess / log_diff_total) if abs(log_diff_total) > 1e-7 else 1.0

        cum_alloc = 0.0
        cum_select = 0.0
        cum_inter = 0.0

        cum_series = []

        for i in range(1, n_days):
            cur_date = dates[i]
            prev_date = dates[i - 1]

            # 获取前一交易日收盘后的持仓权重 (决策权重)
            weights = alloc_dict.get(prev_date, alloc_dict.get(cur_date, {}))
            if not weights and last_weights:
                weights = last_weights

            # 获取当日各标的实际收益
            day_rets = returns_df.loc[cur_date]
            # 1. 按板块聚合策略权重与策略收益
            wp_sec = {sec: 0.0 for sec in all_sectors}
            rp_sec_weighted = {sec: 0.0 for sec in all_sectors}

            for sym, w in weights.items():
                if w <= 0:
                    continue
                sec = self.get_sector(sym)
                ret_sym = float(day_rets.get(sym, 0.0))
                wp_sec[sec] = wp_sec.get(sec, 0.0) + w
                rp_sec_weighted[sec] = rp_sec_weighted.get(sec, 0.0) + w * ret_sym

            wp_total = sum(wp_sec.values())
            if wp_total > 0:
                wp_sec = {s: v / wp_total for s, v in wp_sec.items()}

            # 2. 获取基准在各板块的权重 wb_i 与基准板块收益 Rb_i
            wb_sec = {}
            rb_sec = {}
            for sec in all_sectors:
                wb_sec[sec] = DEFAULT_BENCHMARK_SECTOR_WEIGHTS.get(sec, 0.02)
                sec_syms = [s for s in all_symbols if self.get_sector(s) == sec]
                if sec_syms:
                    sec_day_rets = [float(day_rets.get(s, 0.0)) for s in sec_syms]
                    rb_sec[sec] = sum(sec_day_rets) / len(sec_day_rets)
                else:
                    rb_sec[sec] = float(bmk_ret_series.get(cur_date, 0.0))

            wb_total = sum(wb_sec.values())
            if wb_total > 0:
                wb_sec = {s: v / wb_total for s, v in wb_sec.items()}

            # 校准板块基准收益使其加权和与外部实际基准收益一致
            raw_rb_m = sum(wb_sec[s] * rb_sec[s] for s in all_sectors)
            ext_rb_m = float(bmk_ret_series.get(cur_date, raw_rb_m))
            rb_diff = ext_rb_m - raw_rb_m
            for s in all_sectors:
                rb_sec[s] += rb_diff
            rb_market = ext_rb_m

            # 各板块策略内平均收益 Rp_i
            # 若该板块未持仓，设 Rp_i = Rb_i，代表无选股Alpha，选股效应与交互效应为0
            rp_sec = {}
            for sec in all_sectors:
                if wp_sec[sec] > 1e-5:
                    rp_sec[sec] = rp_sec_weighted[sec] / (wp_sec[sec] * wp_total)
                else:
                    rp_sec[sec] = rb_sec[sec]

            # 实际当日策略组合收益
            raw_rp_m = sum(wp_sec[s] * rp_sec[s] for s in all_sectors)
            ext_rp_m = float(strat_ret_series.get(cur_date, raw_rp_m))
            rp_diff = ext_rp_m - raw_rp_m
            for s in all_sectors:
                if wp_sec[s] > 1e-5:
                    rp_sec[s] += rp_diff
            rp_market = ext_rp_m

            # 单期 Carino 因子 kt:
            log_diff_t = math.log(max(1e-6, 1.0 + rp_market)) - math.log(max(1e-6, 1.0 + rb_market))
            excess_t = rp_market - rb_market
            kt_raw = (log_diff_t / excess_t) if abs(excess_t) > 1e-7 else 1.0
            kt = kt_raw * k_global

            # 计算当日单期 Brinson-Fachler 效应
            day_alloc = 0.0
            day_select = 0.0
            day_inter = 0.0

            for sec in all_sectors:
                wp = wp_sec.get(sec, 0.0)
                wb = wb_sec.get(sec, 0.0)
                rp = rp_sec.get(sec, rb_market)
                rb = rb_sec.get(sec, rb_market)

                # BF 经典公式并乘以 Carino 因子连接
                # 1. 行业配置效应: (wp - wb) * (rb - rb_market)
                q_alloc = (wp - wb) * (rb - rb_market) * kt
                # 2. 个股选择效应: wb * (rp - rb)
                q_select = wb * (rp - rb) * kt
                # 3. 交互协同效应: (wp - wb) * (rp - rb)
                q_inter = (wp - wb) * (rp - rb) * kt

                day_alloc += q_alloc
                day_select += q_select
                day_inter += q_inter

                # 累加到行业
                sector_accum[sec]["alloc"] += q_alloc
                sector_accum[sec]["select"] += q_select
                sector_accum[sec]["inter"] += q_inter
                sector_accum[sec]["wp_sum"] += wp
                sector_accum[sec]["wb_sum"] += wb
                sector_accum[sec]["rp_sum"] += rp
                sector_accum[sec]["rb_sum"] += rb
                sector_accum[sec]["count"] += 1

            cum_alloc += day_alloc
            cum_select += day_select
            cum_inter += day_inter

            daily_alloc_eff.append(day_alloc)
            daily_selec_eff.append(day_select)
            daily_inter_eff.append(day_inter)

            cum_series.append({
                "date": cur_date,
                "cum_excess": round(float(cum_alloc + cum_select + cum_inter) * 100, 3),
                "cum_allocation": round(float(cum_alloc) * 100, 3),
                "cum_selection": round(float(cum_select) * 100, 3),
                "cum_interaction": round(float(cum_inter) * 100, 3)
            })

        # 整理行业明细表
        sector_rows = []
        for sec, data in sector_accum.items():
            cnt = max(data["count"], 1)
            avg_wp = data["wp_sum"] / cnt
            avg_wb = data["wb_sum"] / cnt
            alloc_pct = data["alloc"] * 100
            select_pct = data["select"] * 100
            inter_pct = data["inter"] * 100
            total_pct = alloc_pct + select_pct + inter_pct

            # 仅展示策略持仓过或主要活跃的板块
            if avg_wp > 0.005 or abs(total_pct) > 0.05:
                sector_rows.append({
                    "sector": sec,
                    "avg_weight_strategy": round(avg_wp * 100, 2),
                    "avg_weight_benchmark": round(avg_wb * 100, 2),
                    "active_weight": round((avg_wp - avg_wb) * 100, 2),
                    "allocation_effect_pct": round(alloc_pct, 2),
                    "selection_effect_pct": round(select_pct, 2),
                    "interaction_effect_pct": round(inter_pct, 2),
                    "total_contribution_pct": round(total_pct, 2)
                })

        # 按总贡献降序排序
        sector_rows.sort(key=lambda r: r["total_contribution_pct"], reverse=True)

        return {
            "summary": {
                "allocation": cum_alloc,
                "selection": cum_select,
                "interaction": cum_inter,
                "total_excess": cum_alloc + cum_select + cum_inter
            },
            "sector_table": sector_rows,
            "cumulative_series": cum_series
        }

    def _build_waterfall(
        self,
        total_bmk_return: float,
        brinson_summary: Dict[str, float],
        cost_drag: float,
        total_strat_return: float
    ) -> List[Dict[str, Any]]:
        """
        构建超额收益流向瀑布图数据序列
        基准收益 -> 行业配置效应 -> 个股选择效应 -> 交互协同效应 -> 规费损耗 -> 最终策略收益
        """
        bmk_pct = round(total_bmk_return * 100, 2)
        alloc_pct = round(brinson_summary.get("allocation", 0.0) * 100, 2)
        select_pct = round(brinson_summary.get("selection", 0.0) * 100, 2)
        inter_pct = round(brinson_summary.get("interaction", 0.0) * 100, 2)
        cost_pct = round(-cost_drag * 100, 2)
        strat_pct = round(total_strat_return * 100, 2)

        return [
            {
                "name": "沪深300 基准收益",
                "value": bmk_pct,
                "type": "base",
                "desc": "回测区间大盘核心指数基础收益"
            },
            {
                "name": "行业配置超额 (Allocation)",
                "value": alloc_pct,
                "type": "increment" if alloc_pct >= 0 else "decrement",
                "desc": "在领涨板块的超额配置与落后板块的低配贡献"
            },
            {
                "name": "个股选择超额 (Selection)",
                "value": select_pct,
                "type": "increment" if select_pct >= 0 else "decrement",
                "desc": "在各行业内部由 AI 打分精选龙头产生的个股 Alpha"
            },
            {
                "name": "配置与选股交互 (Interaction)",
                "value": inter_pct,
                "type": "increment" if inter_pct >= 0 else "decrement",
                "desc": "超配强势个股与行业共振产生的协同复利效应"
            },
            {
                "name": "交易规费与滑点磨损",
                "value": cost_pct,
                "type": "drag",
                "desc": "双向佣金、卖出印花税及冲击滑点成本扣减"
            },
            {
                "name": "策略最终实现净收益",
                "value": strat_pct,
                "type": "total",
                "desc": "扣除全成本后的投资组合真实到手收益"
            }
        ]

    def _calculate_style_exposure(
        self,
        allocation_history: List[Dict[str, Any]],
        data_dict: Dict[str, pd.DataFrame],
        trading_dates: List[str],
        bmk_ret_series: pd.Series,
        strat_ret_series: pd.Series
    ) -> Dict[str, Any]:
        """
        计算组合相对于沪深300基准在 Barra 6 大核心风格维度的暴露度与雷达图得分
        1. Beta (市场敏感度 / 进攻防守弹性)
        2. Size (规模风格 / 大盘蓝筹 vs 中小微盘)
        3. Value (估值风格 / 深度价值 vs 成长高估)
        4. Momentum (动量风格 / 趋势追强 vs 超跌反转)
        5. Volatility (波动率风格 / 高弹性博弈 vs 低波防守)
        6. Liquidity (换手流动性 / 热门高换手 vs 低流动性冷门)
        """
        # 1. Beta 计算 (协方差 / 基准方差)
        cov = np.cov(strat_ret_series.values, bmk_ret_series.values)
        bmk_var = np.var(bmk_ret_series.values)
        beta_val = float(cov[0, 1] / bmk_var) if bmk_var > 1e-7 else 1.0

        # 2. 统计持仓平均特征
        last_weights = allocation_history[-1].get("weights", {}) if allocation_history else {}
        if not last_weights and len(allocation_history) > 1:
            last_weights = allocation_history[-2].get("weights", {})

        # 提取标的池各标的历史收益与波动
        sym_vols = {}
        sym_moms = {}
        for sym, df in data_dict.items():
            if "close" in df.columns and len(df) > 20:
                closes = df["close"].values
                # 动量: 近20日收益率
                mom20 = (closes[-1] - closes[-20]) / closes[-20] if closes[-20] > 0 else 0.0
                sym_moms[sym] = mom20
                # 波动: 近20日收益率标准差
                rets20 = np.diff(closes[-21:]) / closes[-21:-1]
                vol20 = float(np.std(rets20) * np.sqrt(250))
                sym_vols[sym] = vol20
            else:
                sym_moms[sym] = 0.0
                sym_vols[sym] = 0.2

        # 加权组合动量与组合波动率
        strat_mom = sum(sym_moms.get(s, 0.0) * w for s, w in last_weights.items() if s != "CASH")
        strat_vol = sum(sym_vols.get(s, 0.2) * w for s, w in last_weights.items() if s != "CASH")

        # 基准年化波动率与动量
        bmk_vol = float(np.std(bmk_ret_series.values[-60:]) * np.sqrt(250)) if len(bmk_ret_series) > 60 else 0.18
        bmk_mom = float((bmk_ret_series.values[-20:] + 1).prod() - 1.0) if len(bmk_ret_series) > 20 else 0.02

        # 标的规模与估值先验得分表 (在标准化 0~100 刻度内)
        # 沪深300作为基准在各大维度默认值为 50 分标准中枢
        size_scores = {
            "sh.600519": 98, "sh.601318": 95, "sh.600036": 90, "sh.510300": 85,
            "sz.000858": 88, "sz.300750": 92, "sz.002594": 86, "sz.000001": 80,
            "sh.510500": 55, "sz.159915": 60, "sh.588000": 58, "sh.518880": 70, "sh.511010": 75
        }
        value_scores = {
            "sh.600036": 92, "sh.601318": 88, "sz.000001": 90, "sh.600900": 85,
            "sh.510300": 65, "sh.600519": 45, "sz.000858": 48, "sz.300750": 30,
            "sz.002594": 35, "sz.159915": 25, "sh.588000": 20, "sh.511010": 95
        }

        # 组合加权风格得分计算 (0~100)
        tot_w = max(1e-5, sum(w for s, w in last_weights.items() if s != "CASH"))
        strat_size = sum(size_scores.get(s, 60) * w for s, w in last_weights.items() if s != "CASH") / tot_w if tot_w > 0 else 60.0
        strat_val_score = sum(value_scores.get(s, 50) * w for s, w in last_weights.items() if s != "CASH") / tot_w if tot_w > 0 else 50.0

        # Beta 映射到 0~100 (Beta=1 对应 50, Beta=1.5 对应 75, Beta=0.5 对应 25)
        strat_beta_score = min(100.0, max(0.0, beta_val * 50.0))

        # 动量映射 (相对基准动量)
        strat_mom_score = min(100.0, max(0.0, 50.0 + (strat_mom - bmk_mom) * 200.0))

        # 波动率映射 (相对基准波动率)
        strat_vol_score = min(100.0, max(0.0, 50.0 + (strat_vol - bmk_vol) * 200.0))

        # 换手流动性得分 (调仓越频繁得分越高，默认中等活跃度 55)
        strat_liq_score = min(100.0, max(20.0, 55.0 + (len(allocation_history) / max(len(trading_dates), 1)) * 100.0))

        radar_indicators = [
            {"name": "市场敏感度 (Beta)", "max": 100},
            {"name": "市值规模 (Size)", "max": 100},
            {"name": "价值风格 (Value)", "max": 100},
            {"name": "动量趋势 (Momentum)", "max": 100},
            {"name": "波动弹性 (Volatility)", "max": 100},
            {"name": "流动性换手 (Liquidity)", "max": 100}
        ]

        strategy_values = [
            round(strat_beta_score, 1),
            round(strat_size, 1),
            round(strat_val_score, 1),
            round(strat_mom_score, 1),
            round(strat_vol_score, 1),
            round(strat_liq_score, 1)
        ]

        benchmark_values = [50.0, 70.0, 55.0, 50.0, 50.0, 50.0]

        # 主动风格偏离度 (Active Tilts)
        active_tilts = [
            {
                "factor": radar_indicators[i]["name"].split(" ")[0],
                "tilt": round(strategy_values[i] - benchmark_values[i], 1),
                "strategy_score": strategy_values[i],
                "benchmark_score": benchmark_values[i]
            }
            for i in range(len(radar_indicators))
        ]

        # 组合风格定性诊断生成
        diagnosis = self._generate_style_diagnosis(strategy_values, active_tilts)

        return {
            "indicators": radar_indicators,
            "strategy_scores": strategy_values,
            "benchmark_scores": benchmark_values,
            "active_tilts": active_tilts,
            "diagnosis": diagnosis,
            "beta": round(beta_val, 2)
        }

    def _generate_style_diagnosis(
        self,
        scores: List[float],
        tilts: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """根据主动偏离度生成定性分析诊断"""
        beta_score, size_score, val_score, mom_score, vol_score, liq_score = scores

        # 核心主风格判断
        size_type = "大盘蓝筹" if size_score >= 65 else ("中小盘成长" if size_score <= 45 else "中大盘均衡")
        val_type = "深度价值" if val_score >= 65 else ("高景气成长" if val_score <= 40 else "价值成长平衡")
        risk_type = "进攻弹性型" if beta_score >= 60 or vol_score >= 60 else "防守稳健型"

        label = f"{size_type} · {val_type} ({risk_type})"

        # 深入评语
        pos_tilts = [t["factor"] for t in tilts if t["tilt"] >= 8]
        neg_tilts = [t["factor"] for t in tilts if t["tilt"] <= -8]

        pos_str = "、".join(pos_tilts) if pos_tilts else "各项均衡"
        neg_str = "、".join(neg_tilts) if neg_tilts else "无明显低配"

        summary = (
            f"当前策略呈现显著的【{label}】特征。在【{pos_str}】上具有明显的主动超额暴露，"
            f"而在【{neg_str}】上相对低配规避风险。"
            f"当市场处于价值蓝筹防御周期或龙头集聚行情时，该风格具备更强的安全垫与 Alpha 捕捉能力。"
        )

        return {
            "style_label": label,
            "analysis_text": summary,
            "primary_tilt": pos_tilts[0] if pos_tilts else "均衡配置"
        }
