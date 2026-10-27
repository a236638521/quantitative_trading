import os
import sys
sys.path.insert(0, ".")
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ[k] = ""

import numpy as np
import pandas as pd
from backend.app.ml.optimizer import PortfolioOptimizer

def test_optimizer_synthetic():
    print("=== Testing PortfolioOptimizer with synthetic asset returns ===")
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=200, freq="B")
    
    # 4 assets with different expected returns and volatilities
    # Asset A: Low risk, low return (Bond-like)
    # Asset B: Moderate risk, moderate return (Large cap)
    # Asset C: High risk, high return (Tech/Growth)
    # Asset D: Uncorrelated/Negative corr asset (Gold-like)
    r_a = np.random.normal(0.0002, 0.005, size=200)
    r_b = np.random.normal(0.0005, 0.012, size=200)
    r_c = np.random.normal(0.0008, 0.022, size=200)
    r_d = np.random.normal(0.0003, 0.010, size=200)

    returns_df = pd.DataFrame({
        "Bond_ETF": r_a,
        "Bluechip_ETF": r_b,
        "Growth_ETF": r_c,
        "Gold_ETF": r_d
    }, index=dates)

    optimizer = PortfolioOptimizer(risk_free_rate=0.025)
    res = optimizer.run_full_analysis(returns_df, max_weight=0.6)

    print("Assets:")
    for a in res["asset_stats"]:
        print(f" - {a['symbol']}: Return={a['expected_return']}%, Vol={a['volatility']}%, Sharpe={a['sharpe_ratio']}")

    print("\nModels:")
    for m in res["models"]:
        print(f"\nModel: {m['model_name']}")
        print(f"  Return: {m['expected_return']}%, Vol: {m['volatility']}%, Sharpe: {m['sharpe_ratio']}")
        print(f"  Weights: {m['weights']}")
        print(f"  Risk Contributions (%): {m['risk_contributions']}")

    # Verify Risk Parity properties: in Risk Parity, risk contributions should be close to 25% each
    rp_m = next(m for m in res["models"] if m["model_id"] == "risk_parity")
    rp_rcs = list(rp_m["risk_contributions"].values())
    print("\nRisk Parity RC std deviation:", np.std(rp_rcs))
    assert np.std(rp_rcs) < 10.0, "Risk parity should balance risk contributions!"

    print("\nEfficient Frontier points generated:", len(res["efficient_frontier"]))
    assert len(res["efficient_frontier"]) > 5, "Should generate efficient frontier points!"

    print("Monte Carlo simulation points:", len(res["monte_carlo_simulations"]))
    assert len(res["monte_carlo_simulations"]) == 250, "Should generate 250 random portfolios!"

    print("\nALL OPTIMIZER TESTS PASSED!")

if __name__ == "__main__":
    test_optimizer_synthetic()
