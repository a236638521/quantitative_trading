"""
AlphaQuant AI/Machine Learning Quantitative Module
支持：多因子特征工程、LightGBM/Ridge预测器、时序Walk-Forward交叉验证与AI选股策略
"""

from .features import FeatureEngineer, FEATURE_DEFINITIONS
from .model import AlphaPredictor
from .strategy import AIMachineLearningStrategy
from .optimizer import PortfolioOptimizer

__all__ = [
    "FeatureEngineer",
    "FEATURE_DEFINITIONS",
    "AlphaPredictor",
    "AIMachineLearningStrategy",
    "PortfolioOptimizer",
]
