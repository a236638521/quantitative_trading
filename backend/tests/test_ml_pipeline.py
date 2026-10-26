import os
import sys
sys.path.insert(0, ".")
for proxy_key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ[proxy_key] = ""

import numpy as np
import pandas as pd
from backend.app.ml.features import FeatureEngineer, FEATURE_DEFINITIONS
from backend.app.ml.model import AlphaPredictor
from backend.app.data.provider import MarketDataProvider

def test_ml_flow():
    print("=== Testing Feature Engineering & LightGBM Training ===")
    provider = MarketDataProvider()
    symbols = ["510300", "510500", "518880", "511010"]
    data_dict = provider.fetch_multi_daily_bars(symbols, start_date="2023-01-01", end_date="2024-01-01", adjust="qfq")
    bmk_df = provider.fetch_daily_bars("000300", start_date="2023-01-01", end_date="2024-01-01", adjust="qfq")
    trading_dates, aligned_dict = provider.align_multi_bars(data_dict, bmk_df)

    fe = FeatureEngineer()
    print(f"Loaded {len(aligned_dict)} symbols, {len(trading_dates)} trading dates.")
    train_test_df, latest_df = fe.build_ml_dataset(aligned_dict, trading_dates, forward_period=5)
    print(f"Dataset shape: {train_test_df.shape}, Latest shape: {latest_df.shape}")

    predictor = AlphaPredictor(model_type="lightgbm")
    result = predictor.train_walk_forward(train_test_df, fe.feature_ids, forward_period=5, test_ratio=0.35, n_folds=3)
    print("METRICS:", result["metrics"])
    print("TOP 5 IMPORTANT FEATURES:")
    for f in result["feature_importances"][:5]:
        print(f" - {f['name']} ({f['id']}): {f['importance_pct']}%")

    recs = predictor.predict_latest_cross_section(latest_df, fe.feature_ids)
    print(f"LATEST PREDICTIONS ({len(recs)} symbols):")
    for r in recs:
        print(f" - Rank #{r['rank']} {r['symbol']}: Score {r['score']:+.4f}, Action {r['action']}, Weight {r['suggested_weight']}")

from backend.app.ml.strategy import AIMachineLearningStrategy
from backend.app.core.portfolio_engine import PortfolioBacktestEngine
from backend.app.config import BacktestConfig

def test_ai_backtest():
    print("\n=== Testing AIMachineLearningStrategy Backtest ===")
    provider = MarketDataProvider()
    symbols = ["510300", "510500", "518880", "511010"]
    data_dict = provider.fetch_multi_daily_bars(symbols, start_date="2023-01-01", end_date="2024-01-01", adjust="qfq")
    bmk_df = provider.fetch_daily_bars("000300", start_date="2023-01-01", end_date="2024-01-01", adjust="qfq")
    trading_dates, aligned_dict = provider.align_multi_bars(data_dict, bmk_df)

    strat = AIMachineLearningStrategy(params={"model_type": "lightgbm", "forward_period": 5, "top_n": 2, "rebalance_freq": "weekly"})
    engine = PortfolioBacktestEngine(config=BacktestConfig(initial_cash=100000.0))

    results = engine.run(
        strategy=strat,
        data_dict=aligned_dict,
        trading_dates=trading_dates,
        benchmark_df=bmk_df
    )
    print("BACKTEST RESULTS:")
    print(" - Total Return:", results.get("metrics", {}).get("total_return"), "%")
    print(" - Annualized Return:", results.get("metrics", {}).get("annualized_return"), "%")
    print(" - Sharpe Ratio:", results.get("metrics", {}).get("sharpe_ratio"))
    print(" - Max Drawdown:", results.get("metrics", {}).get("max_drawdown"), "%")
    print(" - Total Trades:", len(results.get("trades", [])))
    print(" - Final Asset: CNY", results.get("final_asset"))

if __name__ == "__main__":
    test_ml_flow()
    test_ai_backtest()

