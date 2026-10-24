from dataclasses import dataclass
from ..config import TradingCostConfig


@dataclass
class TradeCost:
    commission: float = 0.0      # 佣金
    stamp_tax: float = 0.0       # 印花税
    transfer_fee: float = 0.0    # 过户费
    slippage: float = 0.0        # 滑点导致的价格变动金额
    total_cost: float = 0.0      # 手续费总计 (佣金 + 印花税 + 过户费)


class AShareCostCalculator:
    """A股交易费用计算器"""

    def __init__(self, config: TradingCostConfig = None):
        self.config = config or TradingCostConfig()

    def get_execution_price(self, price: float, is_buy: bool) -> float:
        """计算考虑滑点后的实际成交价"""
        if is_buy:
            return round(price * (1.0 + self.config.slippage_rate), 4)
        else:
            return round(price * (1.0 - self.config.slippage_rate), 4)

    def calculate_cost(self, price: float, shares: int, is_buy: bool) -> TradeCost:
        """
        计算单笔交易的总费用
        :param price: 撮合成交价 (未含滑点)
        :param shares: 股数
        :param is_buy: 是否买入
        """
        nominal_amount = price * shares

        # 1. 佣金 (双向，有最低收费)
        commission = nominal_amount * self.config.commission_rate
        if commission < self.config.min_commission:
            commission = self.config.min_commission

        # 2. 印花税 (仅卖出单边收取)
        stamp_tax = (nominal_amount * self.config.stamp_tax_rate) if not is_buy else 0.0

        # 3. 过户费 (双向)
        transfer_fee = nominal_amount * self.config.transfer_fee_rate

        # 4. 滑点成本 (滑点每股差价 * 股数)
        slippage_cost = price * self.config.slippage_rate * shares

        total_fee = round(commission + stamp_tax + transfer_fee, 2)

        return TradeCost(
            commission=round(commission, 2),
            stamp_tax=round(stamp_tax, 2),
            transfer_fee=round(transfer_fee, 2),
            slippage=round(slippage_cost, 2),
            total_cost=total_fee
        )
