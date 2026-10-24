import os
from pathlib import Path
from pydantic import BaseModel, Field

# 基础目录定位
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"

os.makedirs(CACHE_DIR, exist_ok=True)


class TradingCostConfig(BaseModel):
    """A股交易费用配置"""
    # 买入/卖出佣金比例（默认万分之2.5）
    commission_rate: float = Field(default=0.00025, description="佣金费率")
    # 单笔佣金最低收费（元，默认 5 元）
    min_commission: float = Field(default=5.0, description="最低佣金")
    # 卖出印花税率（默认万分之5，仅卖出收取）
    stamp_tax_rate: float = Field(default=0.0005, description="卖出印花税率")
    # 过户费率（买卖双向，万分之0.1）
    transfer_fee_rate: float = Field(default=0.00001, description="过户费率")
    # 滑点（比例滑点，例如 0.05% 表示买高0.05%，卖低0.05%）
    slippage_rate: float = Field(default=0.0005, description="比例滑点")


class BacktestConfig(BaseModel):
    """回测默认参数配置"""
    initial_cash: float = Field(default=100000.0, description="初始资金(元)")
    cost_config: TradingCostConfig = Field(default_factory=TradingCostConfig)
    benchmark_symbol: str = Field(default="000300", description="基准代码(默认沪深300)")
    lot_size: int = Field(default=100, description="每手股数(A股为100股)")
    enable_t_plus_one: bool = Field(default=True, description="是否启用T+1交易规则")
    enforce_price_limits: bool = Field(default=True, description="是否校验涨跌停禁止买卖")
