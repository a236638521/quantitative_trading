import time
from typing import Dict, Any, Optional, List
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config import BacktestConfig, TradingCostConfig
from ..core.engine import BacktestEngine
from ..core.portfolio_engine import PortfolioBacktestEngine
from ..core.optimizer import GridSearchOptimizer
from ..core.sandbox import StrategySandbox, SAMPLE_TEMPLATES
from ..core.watchdog import SignalWatchdog, WebhookNotifier
from ..factors.analysis import FactorAnalyzer, AVAILABLE_FACTORS
from ..data.provider import MarketDataProvider, normalize_symbol
from ..strategies.dual_ma import DualMAStrategy
from ..strategies.turtle import TurtleStrategy
from ..strategies.bollinger_grid import BollingerGridStrategy
from ..strategies.etf_rotation import ETFMomentumRotationStrategy
from ..strategies.multi_factor import MultiFactorStockSelectionStrategy
from ..ml.features import FeatureEngineer, FEATURE_DEFINITIONS
from ..ml.model import AlphaPredictor
from ..ml.strategy import AIMachineLearningStrategy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")
data_provider = MarketDataProvider()

STRATEGY_REGISTRY = {
    "DualMAStrategy": {
        "cls": DualMAStrategy,
        "is_portfolio": False,
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
        ],
        "optimize_presets": [
            {"param": "fast_period", "label": "快线周期", "default_range": [3, 5, 8, 10, 15]},
            {"param": "slow_period", "label": "慢线周期", "default_range": [15, 20, 30, 45, 60]}
        ]
    },
    "TurtleStrategy": {
        "cls": TurtleStrategy,
        "is_portfolio": False,
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
        ],
        "optimize_presets": [
            {"param": "entry_period", "label": "入场周期", "default_range": [15, 20, 30, 45]},
            {"param": "exit_period", "label": "出场周期", "default_range": [5, 10, 15, 20]}
        ]
    },
    "BollingerGridStrategy": {
        "cls": BollingerGridStrategy,
        "is_portfolio": False,
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
        ],
        "optimize_presets": [
            {"param": "period", "label": "均线周期", "default_range": [10, 15, 20, 30]},
            {"param": "num_std", "label": "标准差倍数", "default_range": [1.5, 2.0, 2.5, 3.0]}
        ]
    },
    "ETFMomentumRotationStrategy": {
        "cls": ETFMomentumRotationStrategy,
        "is_portfolio": True,
        "name": "大类资产 ETF 动量轮动策略",
        "description": "涵盖 A股大盘/中小盘/纳指/黄金/国债ETF，动态持有动量最强者，熊市普跌自动避险防守。",
        "default_pool": ["510300", "510500", "513100", "518880", "511010"],
        "default_params": {
            "lookback_period": 20,
            "top_n": 1,
            "rebalance_freq": "weekly",
            "use_risk_adjusted": True
        },
        "param_schema": [
            {"key": "lookback_period", "label": "动量窗口 (日)", "type": "int", "default": 20, "min": 5, "max": 60},
            {"key": "top_n", "label": "持有资产数量", "type": "int", "default": 1, "min": 1, "max": 3}
        ],
        "optimize_presets": [
            {"param": "lookback_period", "label": "动量窗口", "default_range": [10, 15, 20, 25, 30]},
            {"param": "top_n", "label": "持有数量", "default_range": [1, 2]}
        ]
    },
    "MultiFactorStockSelectionStrategy": {
        "cls": MultiFactorStockSelectionStrategy,
        "is_portfolio": True,
        "name": "核心蓝筹多因子截面选股策略",
        "description": "在核心蓝筹池中，月初按动量、均线趋势与低波动三因子打分，精选 Top 3 标的等权换仓。",
        "default_pool": ["600519", "000858", "300750", "002594", "601318", "600036", "000333", "600900", "601899", "300059"],
        "default_params": {
            "top_n": 3,
            "lookback_period": 20,
            "weight_momentum": 0.5,
            "weight_trend": 0.3,
            "weight_low_vol": 0.2
        },
        "param_schema": [
            {"key": "top_n", "label": "持股数量 (只)", "type": "int", "default": 3, "min": 2, "max": 5},
            {"key": "lookback_period", "label": "特征窗口 (日)", "type": "int", "default": 20, "min": 10, "max": 60}
        ],
        "optimize_presets": [
            {"param": "lookback_period", "label": "特征窗口", "default_range": [10, 15, 20, 30]},
            {"param": "top_n", "label": "持股数量", "default_range": [2, 3, 5]}
        ]
    },
    "AIMachineLearningStrategy": {
        "cls": AIMachineLearningStrategy,
        "is_portfolio": True,
        "name": "AI 机器学习 Alpha 预测驱动策略",
        "description": "基于 LightGBM/Ridge 多维量价时序滚动学习，预测未来超额收益，动态精选截面最强标的。",
        "default_pool": ["510300", "510500", "513100", "518880", "511010"],
        "default_params": {
            "model_type": "lightgbm",
            "forward_period": 5,
            "top_n": 2,
            "rebalance_freq": "weekly"
        },
        "param_schema": [
            {"key": "forward_period", "label": "预测期限 (日)", "type": "int", "default": 5, "min": 1, "max": 20},
            {"key": "top_n", "label": "持仓数量 (只)", "type": "int", "default": 2, "min": 1, "max": 4}
        ]
    }
}


class BacktestRequest(BaseModel):
    symbol: str = Field(default="510300", description="标的代码 (单只代码或多标的逗号分隔池)")
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


class OptimizationRequest(BaseModel):
    symbol: str = Field(default="510300", description="标的代码")
    start_date: str = Field(default="2023-01-01", description="开始日期")
    end_date: str = Field(default="2024-01-01", description="结束日期")
    strategy_name: str = Field(default="DualMAStrategy", description="策略名称")
    param_ranges: Dict[str, List[Any]] = Field(default_factory=dict, description="各参数离散取值列表")
    target_metric: str = Field(default="sharpe_ratio", description="优化目标: sharpe_ratio / total_return / calmar_ratio / win_rate")
    initial_cash: float = Field(default=100000.0, description="初始本金")
    commission_rate: float = Field(default=0.00025, description="佣金费率")
    min_commission: float = Field(default=5.0, description="最低佣金")
    stamp_tax_rate: float = Field(default=0.0005, description="印花税率")
    transfer_fee_rate: float = Field(default=0.00001, description="过户费率")
    slippage_rate: float = Field(default=0.0005, description="滑点")
    enable_t_plus_one: bool = Field(default=True, description="是否启用 T+1")
    enforce_price_limits: bool = Field(default=True, description="是否涨跌停无法买卖")
    benchmark_symbol: str = Field(default="000300", description="基准代码")


class SandboxRunRequest(BaseModel):
    code: str = Field(..., description="Python 策略代码字符串")
    symbol: str = Field(default="510300", description="标的代码")
    start_date: str = Field(default="2023-01-01", description="开始日期")
    end_date: str = Field(default="2024-01-01", description="结束日期")
    initial_cash: float = Field(default=100000.0, description="初始本金")
    commission_rate: float = Field(default=0.00025, description="佣金费率")
    min_commission: float = Field(default=5.0, description="最低佣金")
    stamp_tax_rate: float = Field(default=0.0005, description="印花税率")
    transfer_fee_rate: float = Field(default=0.00001, description="过户费率")
    slippage_rate: float = Field(default=0.0005, description="滑点")
    enable_t_plus_one: bool = Field(default=True, description="严格 T+1")
    enforce_price_limits: bool = Field(default=True, description="涨跌停限制")
    benchmark_symbol: str = Field(default="000300", description="基准代码")


class FactorAnalyzeRequest(BaseModel):
    factor_name: str = Field(default="momentum_20", description="因子名称")
    symbols: List[str] = Field(default_factory=lambda: ["510300", "510500", "513100", "518880", "511010"], description="股票/ETF标的池")
    start_date: str = Field(default="2023-01-01", description="开始日期")
    end_date: str = Field(default="2024-01-01", description="结束日期")
    forward_period: int = Field(default=5, description="未来收益预测周期 (日)")
    quantiles: int = Field(default=5, description="分层组数")


class FactorCorrelationRequest(BaseModel):
    symbols: List[str] = Field(default_factory=lambda: ["510300", "510500", "513100", "518880", "511010"], description="股票/ETF标的池")
    start_date: str = Field(default="2023-01-01", description="开始日期")
    end_date: str = Field(default="2024-01-01", description="结束日期")
    factor_names: Optional[List[str]] = Field(default=None, description="参与分析的因子标识列表")


class WatchdogConfigUpdateRequest(BaseModel):
    webhook_platform: str = Field(default="feishu", description="平台: feishu / dingtalk / wecom")
    webhook_url: str = Field(default="", description="机器人 Webhook 地址")
    webhook_secret: Optional[str] = Field(default="", description="加签密钥 (钉钉可选)")
    auto_alert_enabled: bool = Field(default=False, description="是否开启自动预警")
    monitored_tasks: Optional[List[Dict[str, Any]]] = Field(default=None, description="监控任务列表")


class WatchdogTestAlertRequest(BaseModel):
    webhook_platform: str = Field(default="feishu")
    webhook_url: str = Field(...)
    webhook_secret: Optional[str] = Field(default="")


class WatchdogSendSignalRequest(BaseModel):
    signal: Dict[str, Any] = Field(...)
    webhook_platform: Optional[str] = Field(default=None)
    webhook_url: Optional[str] = Field(default=None)
    webhook_secret: Optional[str] = Field(default=None)


class MLTrainRequest(BaseModel):
    symbols: List[str] = Field(default_factory=lambda: ["510300", "510500", "513100", "518880", "511010"], description="候选标的池")
    start_date: str = Field(default="2023-01-01", description="训练开始日期")
    end_date: str = Field(default="2024-01-01", description="评估结束日期")
    model_type: str = Field(default="lightgbm", description="算法模型 (lightgbm / ridge)")
    forward_period: int = Field(default=5, description="未来预测步长 (日)")
    test_ratio: float = Field(default=0.35, description="时序滚动样本外测试集比例")
    feature_ids: Optional[List[str]] = Field(default=None, description="选用的特征标识列表")
    top_n: int = Field(default=2, description="选股数量")


class MLBacktestRequest(BaseModel):
    symbols: List[str] = Field(default_factory=lambda: ["510300", "510500", "513100", "518880", "511010"])
    start_date: str = Field(default="2023-01-01")
    end_date: str = Field(default="2024-01-01")
    model_type: str = Field(default="lightgbm")
    forward_period: int = Field(default=5)
    top_n: int = Field(default=2)
    rebalance_freq: str = Field(default="weekly")
    feature_ids: Optional[List[str]] = Field(default=None)
    initial_cash: float = Field(default=100000.0)
    commission_rate: float = Field(default=0.00025)
    stamp_tax_rate: float = Field(default=0.0005)
    slippage_rate: float = Field(default=0.0005)


@router.get("/health")
def health_check():
    return {"status": "ok", "system": "A-Share Quantitative Platform"}


@router.get("/strategies")
def list_strategies():
    """获取所有可用策略及参数定义、寻优预设"""
    result = []
    for key, item in STRATEGY_REGISTRY.items():
        result.append({
            "id": key,
            "name": item["name"],
            "description": item["description"],
            "is_portfolio": item.get("is_portfolio", False),
            "default_pool": item.get("default_pool", []),
            "default_params": item["default_params"],
            "param_schema": item["param_schema"],
            "optimize_presets": item.get("optimize_presets", [])
        })
    return {"strategies": result}


@router.post("/backtest/run")
def run_backtest(req: BacktestRequest):
    """运行量化策略回测 (支持单标的技术面策略与多资产/ETF组合轮动策略)"""
    if req.strategy_name not in STRATEGY_REGISTRY:
        raise HTTPException(status_code=400, detail=f"未知策略: {req.strategy_name}")

    strat_item = STRATEGY_REGISTRY[req.strategy_name]
    is_portfolio = strat_item.get("is_portfolio", False)

    # 1. 获取基准数据
    bmk_df = data_provider.fetch_benchmark_bars(
        benchmark_symbol=req.benchmark_symbol,
        start_date=req.start_date,
        end_date=req.end_date
    )

    # 2. 构造费用与回测配置
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

    # 3. 实例化策略
    merged_params = strat_item["default_params"].copy()
    merged_params.update(req.strategy_params)
    strategy_instance = strat_item["cls"](params=merged_params)

    # 4. 执行回测分支：多资产组合 vs 单标的
    if is_portfolio:
        # 解析标的池
        default_pool = strat_item.get("default_pool", ["510300", "510500", "518880", "511010"])
        input_sym = req.symbol.strip()
        if "," in input_sym or " " in input_sym or ";" in input_sym:
            pool = [s.strip() for s in input_sym.replace(";", ",").replace(" ", ",").split(",") if s.strip()]
        else:
            pool = default_pool

        # 批量获取多标的行情并对齐日历
        data_dict = data_provider.fetch_multi_daily_bars(
            symbols=pool,
            start_date=req.start_date,
            end_date=req.end_date,
            adjust="qfq"
        )
        if not data_dict:
            raise HTTPException(status_code=400, detail="组合标的行情数据获取失败")

        trading_dates, aligned_dict = data_provider.align_multi_bars(data_dict, bmk_df)

        port_engine = PortfolioBacktestEngine(config=backtest_cfg)
        try:
            results = port_engine.run(
                strategy=strategy_instance,
                data_dict=aligned_dict,
                trading_dates=trading_dates,
                benchmark_df=bmk_df
            )
            return {"code": 200, "message": "组合回测成功", "data": results}
        except Exception as e:
            logger.exception("组合回测运行异常")
            raise HTTPException(status_code=500, detail=f"组合回测运行异常: {str(e)}")
    else:
        # 单标的回测
        df = data_provider.fetch_daily_bars(
            symbol=req.symbol,
            start_date=req.start_date,
            end_date=req.end_date,
            adjust="qfq"
        )
        if df.empty:
            raise HTTPException(status_code=400, detail=f"未获取到标的 {req.symbol} 的行情数据")

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


@router.post("/backtest/optimize")
def run_optimization(req: OptimizationRequest):
    """运行策略参数网格寻优"""
    if req.strategy_name not in STRATEGY_REGISTRY:
        raise HTTPException(status_code=400, detail=f"未知策略: {req.strategy_name}")

    strat_item = STRATEGY_REGISTRY[req.strategy_name]
    strategy_cls = strat_item["cls"]

    # 1. 预加载行情与基准
    df = data_provider.fetch_daily_bars(
        symbol=req.symbol,
        start_date=req.start_date,
        end_date=req.end_date,
        adjust="qfq"
    )
    if df.empty:
        raise HTTPException(status_code=400, detail=f"未获取到标的 {req.symbol} 的行情数据")

    bmk_df = data_provider.fetch_benchmark_bars(
        benchmark_symbol=req.benchmark_symbol,
        start_date=req.start_date,
        end_date=req.end_date
    )

    # 2. 构造配置
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

    # 3. 准备参数范围
    param_ranges = req.param_ranges
    if not param_ranges:
        param_ranges = {p["param"]: p["default_range"] for p in strat_item.get("optimize_presets", [])}

    optimizer = GridSearchOptimizer(config=backtest_cfg)
    try:
        opt_result = optimizer.run_optimization(
            strategy_cls=strategy_cls,
            param_ranges=param_ranges,
            data_df=df,
            benchmark_df=bmk_df,
            symbol=req.symbol,
            target_metric=req.target_metric
        )
        return {"code": 200, "message": "网格寻优完成", "data": opt_result}
    except Exception as e:
        logger.exception("参数寻优异常")
        raise HTTPException(status_code=500, detail=f"参数寻优异常: {str(e)}")


@router.get("/market/kline")
def get_kline(symbol: str = "510300", start_date: str = "2023-01-01", end_date: str = "2024-01-01"):
    """获取 K 线基础数据 (支持图表渲染)"""
    df = data_provider.fetch_daily_bars(symbol=symbol, start_date=start_date, end_date=end_date)
    if df.empty:
        raise HTTPException(status_code=404, detail="行情数据为空")

    records = df[["date", "open", "close", "low", "high", "volume", "amount"]].to_dict(orient="records")
    return {"symbol": symbol, "count": len(records), "data": records}


@router.get("/sandbox/templates")
def get_sandbox_templates():
    """获取沙箱预置策略模板代码与说明"""
    return {"templates": SAMPLE_TEMPLATES}


@router.post("/sandbox/run")
def run_sandbox_code(req: SandboxRunRequest):
    """在线编译执行自定义策略并运行回测"""
    if not req.code or not req.code.strip():
        raise HTTPException(status_code=400, detail="策略代码不能为空")

    # 1. 加载标的数据与基准数据
    df = data_provider.fetch_daily_bars(
        symbol=req.symbol,
        start_date=req.start_date,
        end_date=req.end_date,
        adjust="qfq"
    )
    if df.empty:
        raise HTTPException(status_code=400, detail=f"标的 {req.symbol} 行情数据获取失败")

    bmk_df = data_provider.fetch_benchmark_bars(
        benchmark_symbol=req.benchmark_symbol,
        start_date=req.start_date,
        end_date=req.end_date
    )

    # 2. 构造回测配置
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

    # 3. 在沙箱中编译执行
    exec_res = StrategySandbox.execute_custom_strategy(
        code_str=req.code,
        data_df=df,
        benchmark_df=bmk_df,
        symbol=req.symbol,
        config=backtest_cfg
    )

    if not exec_res.get("success", False):
        return {
            "code": 400,
            "success": False,
            "error_type": exec_res.get("error_type", "ExecutionError"),
            "message": exec_res.get("message", "回测执行失败"),
            "detail": exec_res.get("detail", "")
        }

    return {
        "code": 200,
        "success": True,
        "message": f"自定义策略 {exec_res['strategy_name']} 沙箱回测完成",
        "data": exec_res["metrics"]
    }


@router.get("/factors/available")
def get_available_factors():
    """获取系统支持的投研因子列表"""
    return {"factors": AVAILABLE_FACTORS}


@router.post("/factors/analyze")
def run_factor_analysis(req: FactorAnalyzeRequest):
    """运行因子有效性分析 (Rank-IC、分层回测与多空对冲)"""
    if not req.symbols or len(req.symbols) < 2:
        raise HTTPException(status_code=400, detail="因子分析至少需要提供 2 只标的")

    # 1. 批量获取标的行情
    data_dict = data_provider.fetch_multi_daily_bars(
        symbols=req.symbols,
        start_date=req.start_date,
        end_date=req.end_date,
        adjust="qfq"
    )
    if not data_dict or len(data_dict) < 2:
        raise HTTPException(status_code=400, detail="标的池数据不足以开展因子分析")

    # 2. 对齐交易日历
    bmk_df = data_provider.fetch_benchmark_bars(
        benchmark_symbol="000300",
        start_date=req.start_date,
        end_date=req.end_date
    )
    trading_dates, aligned_dict = data_provider.align_multi_bars(data_dict, bmk_df)

    # 3. 调用 FactorAnalyzer 执行分析
    try:
        analysis_result = FactorAnalyzer.analyze(
            factor_name=req.factor_name,
            data_dict=aligned_dict,
            trading_dates=trading_dates,
            forward_period=req.forward_period,
            quantiles=req.quantiles
        )
        return {"code": 200, "message": "因子投研分析完成", "data": analysis_result}
    except Exception as e:
        logger.exception("因子投研分析异常")
        raise HTTPException(status_code=500, detail=f"因子投研分析异常: {str(e)}")


@router.post("/factors/correlation")
def run_factor_correlation(req: FactorCorrelationRequest):
    """计算多因子之间的截面 Rank 相关性矩阵与共线性诊断"""
    if not req.symbols or len(req.symbols) < 2:
        raise HTTPException(status_code=400, detail="多因子相关性分析至少需要提供 2 只标的")

    data_dict = data_provider.fetch_multi_daily_bars(
        symbols=req.symbols,
        start_date=req.start_date,
        end_date=req.end_date,
        adjust="qfq"
    )
    if not data_dict or len(data_dict) < 2:
        raise HTTPException(status_code=400, detail="标的池数据不足以开展多因子相关性分析")

    bmk_df = data_provider.fetch_benchmark_bars(
        benchmark_symbol="000300",
        start_date=req.start_date,
        end_date=req.end_date
    )
    trading_dates, aligned_dict = data_provider.align_multi_bars(data_dict, bmk_df)

    try:
        matrix_result = FactorAnalyzer.compute_correlation_matrix(
            data_dict=aligned_dict,
            trading_dates=trading_dates,
            factor_ids=req.factor_names
        )
        return {"code": 200, "message": "多因子相关性矩阵计算完成", "data": matrix_result}
    except Exception as e:
        logger.exception("多因子相关性计算异常")
        raise HTTPException(status_code=500, detail=f"多因子相关性计算异常: {str(e)}")


@router.get("/watchdog/config")
def get_watchdog_config():
    """获取实盘模拟盯盘配置"""
    wd = SignalWatchdog()
    return {"code": 200, "data": wd.load_config()}


@router.post("/watchdog/config")
def update_watchdog_config(req: WatchdogConfigUpdateRequest):
    """更新实盘模拟盯盘配置"""
    wd = SignalWatchdog()
    curr_cfg = wd.load_config()
    curr_cfg["webhook_platform"] = req.webhook_platform
    curr_cfg["webhook_url"] = req.webhook_url
    curr_cfg["webhook_secret"] = req.webhook_secret or ""
    curr_cfg["auto_alert_enabled"] = req.auto_alert_enabled
    if req.monitored_tasks is not None:
        curr_cfg["monitored_tasks"] = req.monitored_tasks
    wd.save_config(curr_cfg)
    return {"code": 200, "message": "盯盘配置更新成功", "data": curr_cfg}


@router.post("/watchdog/scan")
def scan_watchdog_signals():
    """全量扫描监控策略，侦测最新交易日实时产生的操作信号"""
    wd = SignalWatchdog()
    signals = wd.scan_live_signals()
    return {
        "code": 200,
        "message": f"扫描完成，侦测到 {len(signals)} 条最新交易信号",
        "data": {
            "signals": signals,
            "scan_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }


@router.post("/watchdog/test-webhook")
def test_webhook_connection(req: WatchdogTestAlertRequest):
    """向指定机器人 Webhook 发送测试卡片"""
    test_title = "AlphaQuant 机器人连通性测试"
    test_content = (
        f"**恭喜！机器人 Webhook 配置成功**\n"
        f"- **目标平台**: {req.webhook_platform.upper()}\n"
        f"- **测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- **状态**: 正常在线，已接入实时量化信号盯盘通道！"
    )
    result = WebhookNotifier.send_notification(
        platform=req.webhook_platform,
        webhook_url=req.webhook_url,
        secret=req.webhook_secret,
        title=test_title,
        content_markdown=test_content,
        action_type="BUY"
    )
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("message", "推送失败"))
    return {"code": 200, "message": "测试卡片推送成功", "data": result}


@router.post("/watchdog/send-signal")
def send_signal_alert(req: WatchdogSendSignalRequest):
    """推送指定的实时信号到机器人 Webhook"""
    wd = SignalWatchdog()
    cfg = wd.load_config()
    platform = req.webhook_platform or cfg.get("webhook_platform", "feishu")
    webhook_url = req.webhook_url or cfg.get("webhook_url", "")
    secret = req.webhook_secret or cfg.get("webhook_secret", "")

    if not webhook_url:
        raise HTTPException(status_code=400, detail="未配置机器人 Webhook 地址")

    sig = req.signal
    action_label = "【买入建仓】" if sig.get("action") == "BUY" else ("【卖出平仓】" if sig.get("action") == "SELL" else "【动态调仓】")
    title = f"{sig.get('strategy_label')} - {action_label} {sig.get('symbol')}"
    content = (
        f"**策略**: {sig.get('strategy_label')}\n"
        f"**标的**: {sig.get('symbol')} ({sig.get('symbol_name', '')})\n"
        f"**操作**: **{sig.get('action')}**\n"
        f"**参考价格**: ¥{sig.get('price')}\n"
        f"**触发理由**: {sig.get('reason')}\n"
        f"**信号日期**: {sig.get('date')}"
    )

    result = WebhookNotifier.send_notification(
        platform=platform,
        webhook_url=webhook_url,
        secret=secret,
        title=title,
        content_markdown=content,
        action_type=sig.get("action", "BUY")
    )
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("message", "推送失败"))
    return {"code": 200, "message": "预警卡片推送成功", "data": result}


# =========================================================================
# Phase 3: AI / 机器学习 Alpha 预测模型与特征工程接口
# =========================================================================

@router.get("/ml/features")
def get_ml_features():
    """获取所有可用特征库及算法模型元数据"""
    return {
        "code": 200,
        "data": {
            "features": FEATURE_DEFINITIONS,
            "models": [
                {
                    "id": "lightgbm",
                    "name": "LightGBM 梯度提升决策树 (推荐)",
                    "description": "非线性多决策树集成，支持特征增益提取与高维度非线性关系挖掘"
                },
                {
                    "id": "ridge",
                    "name": "Ridge 岭回归 (L2 正则化线性基准)",
                    "description": "经典多因子线性加权基准，具有抗共线性过拟合与系数权重解释能力"
                }
            ]
        }
    }


@router.post("/ml/train")
def train_and_evaluate_ml(req: MLTrainRequest):
    """
    触发 AI/机器学习特征构建、时序向前滚动验证 (Walk-Forward) 并输出特征重要性与最新截面预测
    """
    bmk_df = data_provider.fetch_daily_bars(symbol="000300", start_date=req.start_date, end_date=req.end_date, adjust="qfq")
    data_dict = data_provider.fetch_multi_daily_bars(symbols=req.symbols, start_date=req.start_date, end_date=req.end_date, adjust="qfq")
    if not data_dict:
        raise HTTPException(status_code=400, detail="获取标的池行情失败")

    trading_dates, aligned_dict = data_provider.align_multi_bars(data_dict, bmk_df)
    if len(trading_dates) < 40:
        raise HTTPException(status_code=400, detail=f"有效对齐交易日仅 {len(trading_dates)} 天，建议拉长回测时间范围至 1 年以上")

    try:
        fe = FeatureEngineer(selected_feature_ids=req.feature_ids)
        train_test_df, latest_df = fe.build_ml_dataset(
            data_dict=aligned_dict,
            trading_dates=trading_dates,
            forward_period=req.forward_period,
            relative_target=True
        )

        predictor = AlphaPredictor(model_type=req.model_type)
        eval_result = predictor.train_walk_forward(
            dataset_df=train_test_df,
            feature_ids=fe.feature_ids,
            forward_period=req.forward_period,
            test_ratio=req.test_ratio,
            n_folds=3
        )

        # 最新截面预测建议
        latest_recs = predictor.predict_latest_cross_section(latest_df, fe.feature_ids)

        return {
            "code": 200,
            "message": "AI 模型训练与时序滚动评估完成",
            "data": {
                "model_type": req.model_type,
                "forward_period": req.forward_period,
                "metrics": eval_result["metrics"],
                "feature_importances": eval_result["feature_importances"],
                "ic_series": eval_result["ic_series"],
                "latest_predictions": latest_recs,
                "symbols_count": len(req.symbols),
                "trading_days": len(trading_dates)
            }
        }
    except Exception as e:
        logger.exception("AI 机器学习模型训练异常")
        raise HTTPException(status_code=500, detail=f"AI 模型训练异常: {str(e)}")


@router.post("/ml/backtest")
def run_ml_backtest(req: MLBacktestRequest):
    """
    运行 AI 机器学习策略驱动的完整历史回测 (基于 A 股真实撮合引擎与费用模型)
    """
    bmk_df = data_provider.fetch_daily_bars(symbol="000300", start_date=req.start_date, end_date=req.end_date, adjust="qfq")
    data_dict = data_provider.fetch_multi_daily_bars(symbols=req.symbols, start_date=req.start_date, end_date=req.end_date, adjust="qfq")
    if not data_dict:
        raise HTTPException(status_code=400, detail="获取标的池行情失败")

    trading_dates, aligned_dict = data_provider.align_multi_bars(data_dict, bmk_df)

    cost_cfg = TradingCostConfig(
        commission_rate=req.commission_rate,
        min_commission=5.0,
        stamp_tax_rate=req.stamp_tax_rate,
        transfer_fee_rate=0.00001,
        slippage_rate=req.slippage_rate
    )
    backtest_cfg = BacktestConfig(
        initial_cash=req.initial_cash,
        cost_config=cost_cfg,
        enable_t_plus_one=True,
        enforce_price_limits=True
    )

    strategy_params = {
        "model_type": req.model_type,
        "forward_period": req.forward_period,
        "top_n": req.top_n,
        "rebalance_freq": req.rebalance_freq,
        "feature_ids": req.feature_ids
    }
    strategy_instance = AIMachineLearningStrategy(params=strategy_params)
    port_engine = PortfolioBacktestEngine(config=backtest_cfg)

    try:
        results = port_engine.run(
            strategy=strategy_instance,
            data_dict=aligned_dict,
            trading_dates=trading_dates,
            benchmark_df=bmk_df
        )

        # 附加上模型自身的训练指标与特征重要性供看板展示
        summary = getattr(strategy_instance, "training_summary", {})
        results["ai_metrics"] = summary.get("metrics", {})
        results["feature_importances"] = summary.get("feature_importances", [])
        results["ic_series"] = summary.get("ic_series", [])

        return {
            "code": 200,
            "message": "AI 策略回测运行成功",
            "data": results
        }
    except Exception as e:
        logger.exception("AI 策略回测异常")
        raise HTTPException(status_code=500, detail=f"AI 策略回测异常: {str(e)}")



