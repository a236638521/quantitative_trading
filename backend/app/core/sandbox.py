import sys
import traceback
import logging
from typing import Dict, Any, Type, Optional
import pandas as pd
import numpy as np

from .engine import BacktestEngine
from .portfolio import Portfolio
from ..config import BacktestConfig
from ..strategies.base import BaseStrategy, Signal
from ..factors.technical import (
    calc_momentum,
    calc_volatility,
    calc_risk_adjusted_momentum,
    calc_trend_score
)
from ..factors.processor import CrossSectionProcessor

logger = logging.getLogger(__name__)


# 预置三套开箱即用的经典策略模板代码
SAMPLE_TEMPLATES = {
    "rsi_reversal": {
        "name": "RSI 超跌反弹策略",
        "description": "基于 14 日相对强弱指标，当 RSI < 25 超跌时建仓买入，RSI > 75 超买时平仓止盈。",
        "code": """class CustomRSIStrategy(BaseStrategy):
    \"\"\"
    自定义 RSI 均值回归反弹策略
    \"\"\"
    def __init__(self, params=None):
        default_params = {
            "rsi_period": 14,
            "oversold": 25.0,     # 超跌阈值买入
            "overbought": 75.0,   # 超买阈值卖出
            "position_pct": 0.95
        }
        if params:
            default_params.update(params)
        super().__init__(name="CustomRSIStrategy", params=default_params)

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.params["rsi_period"]
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-6)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    def on_bar(self, index: int, row: pd.Series, portfolio: Portfolio):
        signals = []
        symbol = str(row.get("code", ""))
        rsi = row.get("rsi", 50.0)
        pos = portfolio.get_position(symbol)

        # 跌破超跌线，买入
        if rsi < self.params["oversold"] and pos.total_shares == 0:
            signals.append(
                self.order_target_percent(symbol, self.params["position_pct"], f"RSI={rsi:.1f} 超跌建仓")
            )
        # 突破超买线，卖出
        elif rsi > self.params["overbought"] and pos.total_shares > 0:
            signals.append(
                self.close_position(symbol, f"RSI={rsi:.1f} 超买止盈")
            )
        return signals
"""
    },
    "kdj_cross": {
        "name": "KDJ 随机指标金叉死叉策略",
        "description": "基于 9 日 KDJ 随机指标，在超卖区金叉（K 上穿 D 且 K < 30）买入，超买区死叉卖出。",
        "code": """class CustomKDJStrategy(BaseStrategy):
    \"\"\"
    自定义 KDJ 随机指标金叉策略
    \"\"\"
    def __init__(self, params=None):
        default_params = {
            "n": 9,
            "buy_threshold": 30.0,
            "sell_threshold": 70.0,
            "position_pct": 0.95
        }
        if params:
            default_params.update(params)
        super().__init__(name="CustomKDJStrategy", params=default_params)

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = self.params["n"]
        low_list = df["low"].rolling(window=n, min_periods=n).min()
        high_list = df["high"].rolling(window=n, min_periods=n).max()
        rsv = ((df["close"] - low_list) / (high_list - low_list + 1e-6)) * 100
        
        # 计算 K, D, J
        k_values = [50.0]
        d_values = [50.0]
        for val in rsv.fillna(50.0).iloc[1:]:
            k = 2/3 * k_values[-1] + 1/3 * val
            d = 2/3 * d_values[-1] + 1/3 * k
            k_values.append(k)
            d_values.append(d)
            
        df["k"] = k_values
        df["d"] = d_values
        df["j"] = 3 * df["k"] - 2 * df["d"]
        
        # 金叉与死叉
        df["kd_diff"] = df["k"] - df["d"]
        df["prev_diff"] = df["kd_diff"].shift(1)
        df["golden_cross"] = (df["prev_diff"] <= 0) & (df["kd_diff"] > 0)
        df["death_cross"] = (df["prev_diff"] >= 0) & (df["kd_diff"] < 0)
        return df

    def on_bar(self, index: int, row: pd.Series, portfolio: Portfolio):
        signals = []
        symbol = str(row.get("code", ""))
        pos = portfolio.get_position(symbol)
        k = row.get("k", 50)
        
        if row.get("golden_cross") and k < self.params["buy_threshold"] and pos.total_shares == 0:
            signals.append(self.order_target_percent(symbol, self.params["position_pct"], f"KDJ低位金叉(K={k:.1f})建仓"))
        elif row.get("death_cross") and k > self.params["sell_threshold"] and pos.total_shares > 0:
            signals.append(self.close_position(symbol, f"KDJ高位死叉(K={k:.1f})平仓"))
        return signals
"""
    },
    "grid_trading": {
        "name": "动态等比网格套利策略",
        "description": "适合宽基 ETF。以初始价为基准，每下跌 3% 补仓一手，每上涨 3% 卖出一手，震荡高抛低吸。",
        "code": """class CustomGridStrategy(BaseStrategy):
    \"\"\"
    自定义动态等比网格交易策略
    \"\"\"
    def __init__(self, params=None):
        default_params = {
            "grid_step_pct": 0.03,  # 网格步长 (3%)
            "trade_lots": 10        # 每次买卖手数量 (10手=1000股)
        }
        if params:
            default_params.update(params)
        super().__init__(name="CustomGridStrategy", params=default_params)
        self.last_trade_price = 0.0

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def on_bar(self, index: int, row: pd.Series, portfolio: Portfolio):
        signals = []
        symbol = str(row.get("code", ""))
        close = float(row["close"])
        step = self.params["grid_step_pct"]
        lot_shares = self.params["trade_lots"] * 100

        pos = portfolio.get_position(symbol)

        # 首次入场打底仓
        if self.last_trade_price <= 0:
            self.last_trade_price = close
            signals.append(self.buy_shares(symbol, lot_shares * 3, f"网格初始化底仓 @ {close:.2f}"))
            return signals

        # 价格相比上次交易价格下跌达到步长 -> 触发逢低买入补仓
        if close <= self.last_trade_price * (1.0 - step):
            signals.append(self.buy_shares(symbol, lot_shares, f"网格触碰下轨 -{step*100:.1f}% 补仓"))
            self.last_trade_price = close
        # 价格相比上次交易价格上涨达到步长 -> 触发止盈卖出一档
        elif close >= self.last_trade_price * (1.0 + step) and pos.available_shares >= lot_shares:
            signals.append(self.sell_shares(symbol, lot_shares, f"网格触碰上轨 +{step*100:.1f}% 止盈"))
            self.last_trade_price = close

        return signals
"""
    }
}


class StrategySandbox:
    """动态策略代码安全解析与执行沙箱"""

    @classmethod
    def execute_custom_strategy(
        cls,
        code_str: str,
        data_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
        symbol: str,
        config: BacktestConfig
    ) -> Dict[str, Any]:
        """
        动态编译并执行用户编写的策略脚本
        """
        # 1. 构建安全的执行上下文沙箱环境
        sandbox_globals: Dict[str, Any] = {
            "__builtins__": __builtins__,
            "pd": pd,
            "np": np,
            "BaseStrategy": BaseStrategy,
            "Signal": Signal,
            "Portfolio": Portfolio,
            "calc_momentum": calc_momentum,
            "calc_volatility": calc_volatility,
            "calc_risk_adjusted_momentum": calc_risk_adjusted_momentum,
            "calc_trend_score": calc_trend_score,
            "CrossSectionProcessor": CrossSectionProcessor
        }
        sandbox_locals: Dict[str, Any] = {}

        # 2. 编译并执行用户代码
        try:
            compiled = compile(code_str, "<custom_strategy>", "exec")
            exec(compiled, sandbox_globals, sandbox_locals)
        except SyntaxError as se:
            line_no = se.lineno
            return {
                "success": False,
                "error_type": "SyntaxError",
                "message": f"语法错误 (第 {line_no} 行): {se.msg}",
                "detail": str(se)
            }
        except Exception as e:
            return {
                "success": False,
                "error_type": type(e).__name__,
                "message": f"代码初始化异常: {str(e)}",
                "detail": traceback.format_exc()
            }

        # 3. 在 locals 中自动寻找继承 BaseStrategy 的类
        target_cls = None
        for name, obj in sandbox_locals.items():
            if isinstance(obj, type) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                target_cls = obj
                break

        if target_cls is None:
            return {
                "success": False,
                "error_type": "StrategyNotFound",
                "message": "未在代码中找到继承自 BaseStrategy 的策略类，请确保策略类声明如 `class MyStrategy(BaseStrategy):`",
                "detail": "Missing BaseStrategy subclass"
            }

        # 4. 实例化策略并调用回测引擎
        try:
            strategy_instance = target_cls()
            engine = BacktestEngine(config=config)
            metrics = engine.run(
                strategy=strategy_instance,
                data_df=data_df,
                benchmark_df=benchmark_df,
                symbol=symbol
            )
            return {
                "success": True,
                "strategy_name": target_cls.__name__,
                "metrics": metrics
            }
        except Exception as e:
            logger.exception("自定义策略回测运行异常")
            return {
                "success": False,
                "error_type": "RuntimeExecutionError",
                "message": f"策略回测运行异常: {str(e)}",
                "detail": traceback.format_exc()
            }
