import os
import sys
sys.path.insert(0, ".")
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ[k] = ""

import requests

def test_api():
    session = requests.Session()
    session.trust_env = False

    print("Testing GET /api/ml/features...")
    res = session.get("http://127.0.0.1:8000/api/ml/features").json()
    print("FEATURES CODE:", res.get("code"), "count:", len(res.get("data", {}).get("features", [])))

    print("Testing POST /api/ml/train...")
    payload = {
        "symbols": ["510300", "510500", "518880", "511010"],
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "model_type": "lightgbm",
        "forward_period": 5,
        "test_ratio": 0.35,
        "top_n": 2
    }
    train_res = session.post("http://127.0.0.1:8000/api/ml/train", json=payload).json()
    print("TRAIN CODE:", train_res.get("code"), train_res.get("message"))
    data = train_res.get("data", {})
    metrics = data.get("metrics", {})
    print(f"METRICS: IC Mean={metrics.get('ic_mean')}, IC-IR={metrics.get('ic_ir')}, Acc={metrics.get('directional_accuracy')}%")
    print("TOP 3 FEATURES:", [f"{f['name']}({f['importance_pct']}%)" for f in data.get("feature_importances", [])[:3]])
    print("PREDICTIONS COUNT:", len(data.get("latest_predictions", [])))

    print("\nTesting POST /api/ml/backtest...")
    bt_payload = {
        "symbols": ["510300", "510500", "518880", "511010"],
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "model_type": "lightgbm",
        "forward_period": 5,
        "top_n": 2,
        "rebalance_freq": "weekly",
        "initial_cash": 100000.0
    }
    bt_res = session.post("http://127.0.0.1:8000/api/ml/backtest", json=bt_payload).json()
    print("BACKTEST CODE:", bt_res.get("code"), bt_res.get("message"))
    bt_data = bt_res.get("data", {})
    print(f"BACKTEST: Return={bt_data.get('total_return')}%, Sharpe={bt_data.get('sharpe_ratio')}, MaxDD={bt_data.get('max_drawdown')}%, FinalAsset={bt_data.get('final_asset')}")

if __name__ == "__main__":
    test_api()
