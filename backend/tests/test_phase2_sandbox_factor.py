import os
for proxy_key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ[proxy_key] = ""

import requests

def test_sandbox():
    print("Testing Sandbox...")
    t_res = requests.get("http://127.0.0.1:8000/api/sandbox/templates").json()
    rsi_code = t_res["templates"]["rsi_reversal"]["code"]
    payload = {
        "code": rsi_code,
        "symbol": "510300",
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "initial_cash": 100000.0
    }
    run_res = requests.post("http://127.0.0.1:8000/api/sandbox/run", json=payload).json()
    print("SANDBOX SUCCESS:", run_res.get("success"))
    print("DATA KEYS:", list(run_res.get("data", {}).keys()))
    metrics_obj = run_res.get("data", {}).get("metrics", {})
    print("METRICS OBJ:", metrics_obj)
    print("TRADES COUNT:", len(run_res.get("data", {}).get("trades", [])))

def test_factor():
    print("\nTesting Factor Research...")
    payload = {
        "factor_name": "momentum_20",
        "symbols": ["510300", "510500", "518880", "511010"],
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "forward_period": 5,
        "quantiles": 4
    }
    f_res = requests.post("http://127.0.0.1:8000/api/factors/analyze", json=payload).json()
    print("FACTOR CODE:", f_res.get("code"))
    metrics = f_res.get("data", {}).get("metrics", {})
    print("RANK-IC MEAN:", metrics.get("ic_mean"))
    print("IC IR:", metrics.get("ic_ir"))
    print("LONG-SHORT RETURN:", metrics.get("long_short_return"))
    print("MONOTONICITY SCORE:", metrics.get("monotonicity_score"))
    print("QUANTILE NAV KEYS:", list(f_res.get("data", {}).get("quantile_navs", {}).keys()))

if __name__ == "__main__":
    test_sandbox()
    test_factor()
