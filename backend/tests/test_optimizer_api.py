import os
import sys
sys.path.insert(0, ".")
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ[k] = ""

import requests

def test_api():
    session = requests.Session()
    session.trust_env = False

    print("Testing POST /api/optimizer/analyze...")
    payload = {
        "symbols": ["510300", "510500", "518880", "511010"],
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "risk_free_rate": 0.025,
        "max_weight": 0.6
    }
    res = session.post("http://127.0.0.1:8000/api/optimizer/analyze", json=payload).json()
    print("OPTIMIZER CODE:", res.get("code"), res.get("message"))
    data = res.get("data", {})
    models = data.get("models", [])
    print(f"Models count: {len(models)}")
    for m in models:
        print(f" - {m['model_name']}: Return={m['expected_return']}%, Vol={m['volatility']}%, Sharpe={m['sharpe_ratio']}, Weights={m['weights']}")

    print("Efficient Frontier points:", len(data.get("efficient_frontier", [])))
    print("Monte Carlo points:", len(data.get("monte_carlo_simulations", [])))

    print("\nTesting POST /api/ml/backtest with allocation_method='risk_parity'...")
    bt_payload = {
        "symbols": ["510300", "510500", "518880", "511010"],
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "model_type": "lightgbm",
        "forward_period": 5,
        "top_n": 2,
        "rebalance_freq": "weekly",
        "allocation_method": "risk_parity",
        "initial_cash": 100000.0
    }
    bt_res = session.post("http://127.0.0.1:8000/api/ml/backtest", json=bt_payload).json()
    print("BACKTEST (Risk Parity) CODE:", bt_res.get("code"), bt_res.get("message"))
    data = bt_res.get("data", {})
    metrics = data.get("metrics", {})
    tr = metrics.get('total_return') if metrics.get('total_return') is not None else data.get('total_return', 0.0)
    sr = metrics.get('sharpe_ratio') if metrics.get('sharpe_ratio') is not None else data.get('sharpe_ratio', 0.0)
    dd = metrics.get('max_drawdown') if metrics.get('max_drawdown') is not None else data.get('max_drawdown', 0.0)
    trades = len(data.get('trades', []))
    final_asset = data.get('final_asset')
    print(f"BACKTEST SUMMARY: Return={tr}%, Sharpe={sr}, MaxDD={dd}%, Trades={trades}, FinalAsset={final_asset}")

if __name__ == "__main__":
    test_api()
