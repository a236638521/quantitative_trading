import logging
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import baostock as bs

from .storage import DataStorage

logger = logging.getLogger(__name__)


def normalize_symbol(symbol: str) -> str:
    """标准化 A 股代码格式，例如 600519 -> sh.600519, 000001 -> sz.000001"""
    s = symbol.strip().lower()
    if s.startswith("sh.") or s.startswith("sz."):
        return s
    if s.startswith("sh") or s.startswith("sz"):
        return f"{s[:2]}.{s[2:]}"

    # 根据代码特征判断交易所
    if s.startswith(("60", "68", "51", "11", "000300", "000905", "000852", "000001")):
        if s == "000001":
            # 000001 既是平安银行也是上证指数，若纯输入 000001 优先判定为平安银行 sz.000001
            return "sz.000001"
        if s in ("000300", "000905", "000852"):
            return f"sh.{s}"
        return f"sh.{s}"
    else:
        return f"sz.{s}"


def get_price_limit_ratio(symbol: str) -> float:
    """根据标的代码判断涨跌停幅度限制"""
    code = symbol.split(".")[-1]
    if code.startswith(("300", "301", "688")):
        return 0.20  # 创业板、科创板 20%
    return 0.10      # 主板与主流ETF 10%


class MarketDataProvider:
    """A股市场数据获取器 (基于 Baostock + 本地 Parquet 缓存双保险)"""

    def __init__(self, storage: Optional[DataStorage] = None):
        self.storage = storage or DataStorage()

    def fetch_daily_bars(
        self,
        symbol: str,
        start_date: str = "2023-01-01",
        end_date: str = "2024-12-31",
        adjust: str = "qfq",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        获取日线行情数据（含前复权、涨跌停线计算）
        :param symbol: 股票/ETF代码，如 600519 或 sh.600519
        :param start_date: 开始日期 YYYY-MM-DD
        :param end_date: 结束日期 YYYY-MM-DD
        :param adjust: 复权类型 (qfq: 前复权, hfq: 后复权, none: 不复权)
        :param use_cache: 是否优先读取本地缓存
        """
        norm_symbol = normalize_symbol(symbol)
        clean_code = norm_symbol.split(".")[-1]

        # 1. 尝试从本地缓存读取
        if use_cache and self.storage.exists(norm_symbol, period="daily", adjust=adjust):
            cached_df = self.storage.load_bars(norm_symbol, period="daily", adjust=adjust)
            if cached_df is not None and not cached_df.empty:
                # 检查日期覆盖范围
                min_date = cached_df["date"].min()
                max_date = cached_df["date"].max()
                if min_date <= start_date and max_date >= end_date:
                    mask = (cached_df["date"] >= start_date) & (cached_df["date"] <= end_date)
                    return cached_df.loc[mask].reset_index(drop=True)

        # 2. 从 Baostock 实时获取
        adjust_flag_map = {"qfq": "2", "hfq": "1", "none": "3"}
        adj_flag = adjust_flag_map.get(adjust, "2")

        # 登录 Baostock
        bs.login()
        try:
            fields = "date,code,open,high,low,close,preclose,volume,amount,pctChg"
            rs = bs.query_history_k_data_plus(
                norm_symbol,
                fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag=adj_flag
            )

            if rs.error_code != "0":
                logger.error(f"Baostock 获取失败: {rs.error_msg}")
                return pd.DataFrame()

            data_list = []
            while (rs.error_code == "0") and rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                return pd.DataFrame()

            df = pd.DataFrame(data_list, columns=rs.fields)

            # 数据类型清洗与转换
            float_cols = ["open", "high", "low", "close", "preclose", "volume", "amount", "pctChg"]
            for col in float_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

            df = df.rename(columns={"pctChg": "pct_chg"})

            # 计算涨跌停限制价 (A股四舍五入到分)
            limit_ratio = get_price_limit_ratio(norm_symbol)
            # 若 preclose 存在则按前收价算，否则用当日前一交易日的 close
            if "preclose" in df.columns:
                df["up_limit"] = (df["preclose"] * (1.0 + limit_ratio)).round(2)
                df["down_limit"] = (df["preclose"] * (1.0 - limit_ratio)).round(2)
            else:
                df["up_limit"] = (df["close"].shift(1) * (1.0 + limit_ratio)).round(2)
                df["down_limit"] = (df["close"].shift(1) * (1.0 - limit_ratio)).round(2)

            # 判断是否封死涨停或跌停 (无法成交)
            df["is_up_limit"] = (df["close"] >= df["up_limit"]) & (df["open"] >= df["up_limit"])
            df["is_down_limit"] = (df["close"] <= df["down_limit"]) & (df["open"] <= df["down_limit"])

            # 3. 写入本地缓存以备下次秒级回测
            self.storage.save_bars(norm_symbol, df, period="daily", adjust=adjust)

            return df

        finally:
            bs.logout()

    def fetch_benchmark_bars(
        self,
        benchmark_symbol: str = "000300",
        start_date: str = "2023-01-01",
        end_date: str = "2024-12-31"
    ) -> pd.DataFrame:
        """获取对标基准指数的历史数据 (默认沪深300)"""
        norm_bmk = normalize_symbol(benchmark_symbol)
        return self.fetch_daily_bars(norm_bmk, start_date=start_date, end_date=end_date, adjust="none")
