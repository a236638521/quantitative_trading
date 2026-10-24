from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config import BacktestConfig, TradingCostConfig
from ..core.engine import BacktestEngine
from ..data.provider import MarketDataProvider, normalize_symbol
from ..strategies.dual_ma import DualMAStrategy
from ..strategies.turtle import TurtleStrategy
from ..strategies.bollinger_grid import BollingerGridStrategy

router = APIRouter(prefix="/api")
data_provider = MarketDataProvider()

STRATEGY_REGISTRY = {
    "DualMAStrategy": {
        "cls": DualMAStrategy,
        "name": "双均线交叉策略",
        "description": "快线金叉慢线买入，死叉平仓。经典的顺势跟踪模型。",
        "default_params": {
            "fast_period": 5,
            "slow_period": 20,
            "position_pct": 0.95
        },
        "param_schema": [
            {"key": "fast_period", "label": "快线周期 (日)", "type": "int", "default": 5, "min": 2, "max": 60},
            {"key": "slow_period", "label": "慢线周期 (日)", "type": "int", "default": 20, "min": 5, "max": 250},
            {"key": "position_pct", "label": "仓位比例", "type": "float", "default": 0.95, "min": 0.1, "max": 1.0}
        ]
    },
    "TurtleStrategy": {
        "cls": TurtleStrategy,
        "name": "海龟唐奇安突破策略",
        "description": "突破20日通道高点建仓，跌破10日低点止损。顺应中长期单边趋势。",
        "default_params": {
            "entry_period": 20,
            "exit_period": 10,
            "atr_period": 14,
            "position_pct": 0.95
        },
        "param_schema": [
            {"key": "entry_period", "label": "入场突破周期 (日)", "type": "int", "default": 20, "min": 5, "max": 120},
            {"key": "exit_period", "label": "出场止损周期 (日)", "type": "int", "default": 10, "min": 3, "max": 60},
            {"key": "atr_period", "label": "ATR波动率周期", "type": "int", "default": 14, "min": 5, "max": 30},
            {"key": "position_pct", "label": "仓位比例", "type": "float", "default": 0.95, "min": 0.1, "max": 1.0}
        ]
    },
    "BollingerGridStrategy": {
        "cls": BollingerGridStrategy,
        "name": "布林带均值回归策略",
        "description": "股价触碰下轨超跌建仓，触碰上轨超买止盈。适合宽基 ETF 或震荡箱体。",
        "default_params": {
            "period": 20,
            "num_std": 2.0,
            "position_pct": 0.95
        },
        "param_schema": [
            {"key": "period", "label": "均线周期 (日)", "type": "int", "default": 20, "min": 5, "max": 60},
            {"key": "num_std", "label": "标准差倍数", "type": "float", "default": 2.0, "min": 1.0, "max": 3.5},
            {"key": "position_pct", "label": "仓位比例", "type": "float", "default": 0.95, "min": 0.1, "max": 1.0}
        ]
    }
}


class BacktestRequest(BaseModel):
    symbol: str = Field(default="510300", description="标的代码 (如 510300 或 600519)")
    start_date: str = Field(default="2023-01-01", description="回测开始日期")
    end_date: str = Field(default="2024-01-01", description="回测结束日期")
    strategy_name: str = Field(default="DualMAStrategy", description="策略标识")
    strategy_params: Dict[str, Any] = Field(default_factory=dict, description="策略自定义参数")
    initial_cash: float = Field(default=100000.0, description="初始本金 (元)")
    commission_rate: float = Field(default=0.00025, description="佣金费率")
    min_commission: float = Field(default=5.0, description="最低佣金")
    stamp_tax_rate: float = Field(default=0.0005, description="卖出印花税率")
    transfer_fee_rate: float = Field(default=0.00001, description="过户费率")
    slippage_rate: float = Field(default=0.0005, description="滑点")
    enable_t_plus_one: bool = Field(default=True, description="是否启用 T+1")
    enforce_price_limits: bool = Field(default=True, description="是否涨跌停无法买卖")
    benchmark_symbol: str = Field(default="000300", description="对比基准代码")


@router.get("/health")
def health_check():
    return {"status": "ok", "system": "A-Share Quantitative Platform"}


@router.get("/strategies")
def list_strategies():
    """获取所有可用策略及参数定义"""
    result = []
    for key, item in STRATEGY_REGISTRY.items():
        result.append({
            "id": key,
            "name": item["name"],
            "description": item["description"],
            "default_params": item["default_params"],
            "param_schema": item["param_schema"]
        })
    return {"strategies": result}


@router.post("/backtest/run")
def run_backtest(req: BacktestRequest):
    """运行量化策略回测"""
    if req.strategy_name not in STRATEGY_REGISTRY:
        raise HTTPException(status_code=400, detail=f"未知策略: {req.strategy_name}")

    # 1. 获取标的历史行情
    df = data_provider.fetch_daily_bars(
        symbol=req.symbol,
        start_date=req.start_date,
        end_date=req.end_date,
        adjust="qfq"
    )
    if df.empty:
        raise HTTPException(status_code=400, detail=f"未获取到标的 {req.symbol} 在指定时间段内的行情数据")

    # 2. 获取基准数据
    bmk_df = data_provider.fetch_benchmark_bars(
        benchmark_symbol=req.benchmark_symbol,
        start_date=req.start_date,
        end_date=req.end_date
    )

    # 3. 构造费用与回测配置
    cost_cfg = TradingCostConfig(
        commission_rate=req.commission_rate,
        min_commission=req.min_commission,
        stamp_tax_rate=req.stamp_tax_rate,
        transfer_fee_rate=req.transfer_fee_rate,
        slippage_rate=req.slippage_rate
    )
    backtest_cfg = BacktestConfig(
        initial_cash=req.initial_cash,
        cost_config=cost_cfg,
        benchmark_symbol=req.benchmark_symbol,
        enable_t_plus_one=req.enable_t_plus_one,
        enforce_price_limits=req.enforce_price_limits
    )

    # 4. 实例化策略
    strat_item = STRATEGY_REGISTRY[req.strategy_name]
    merged_params = strat_item["default_params"].copy()
    merged_params.update(req.strategy_params)
    strategy_instance = strat_item["cls"](params=merged_params)

    # 5. 执行回测
    engine = BacktestEngine(config=backtest_cfg)
    try:
        results = engine.run(
            strategy=strategy_instance,
            data_df=df,
            benchmark_df=bmk_df,
            symbol=req.symbol
        )
        return {"code": 200, "message": "回测成功", "data": results}
    except Exception as e:
        logger.exception("回测运行异常")
        raise HTTPException(status_code=500, detail=f"回测运行异常: {str(e)}")


@router.get("/market/kline")
def get_kline(symbol: str = "510300", start_date: str = "2023-01-01", end_date: str = "2024-01-01"):
    """获取 K 线基础数据 (支持图表渲染)"""
    df = data_provider.fetch_daily_bars(symbol=symbol, start_date=start_date, end_date=end_date)
    if df.empty:
        raise HTTPException(status_code=404, detail="行情数据为空")

    records = df[["date", "open", "close", "low", "high", "volume", "amount"]].to_dict(orient="records")
    return {"symbol": symbol, "count": len(records), "data": records}
