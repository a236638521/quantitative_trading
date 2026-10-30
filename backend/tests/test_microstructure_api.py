import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath("."))
import requests


def test_microstructure_api():
    session = requests.Session()
    session.trust_env = False  # 绕过 Windows 本地代理 10808 端口

    base_url = "http://127.0.0.1:8000"

    print("1. Testing GET /api/microstructure/factors...")
    resp = session.get(f"{base_url}/api/microstructure/factors")
    print("STATUS:", resp.status_code)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    factors = resp.json()["data"]
    print(f"Total Factors: {len(factors)}")
    factor_ids = [f["id"] for f in factors]
    print("Factor IDs:", factor_ids)
    assert "ofi" in factor_ids
    assert "vpin" in factor_ids
    assert "micro_price_bias" in factor_ids
    assert "depth_imbalance_5" in factor_ids
    assert "active_buy_ratio" in factor_ids
    assert "roll_spread" in factor_ids

    print("\n2. Testing POST /api/microstructure/analyze...")
    payload = {
        "symbol": "sh.600519",
        "base_price": 1800.0,
        "n_ticks": 100,
        "vpin_bucket_size": 40,
        "vpin_n_buckets": 15
    }
    resp = session.post(f"{base_url}/api/microstructure/analyze", json=payload)
    print("STATUS:", resp.status_code)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()["data"]
    
    print("Analysis Symbol:", data.get("symbol"))
    assert data.get("symbol") == "sh.600519"

    summary = data.get("summary", {})
    print("Summary KPIs:", summary)
    assert "latest_price" in summary
    assert "ofi_current" in summary
    assert "vpin_current" in summary
    assert "micro_bias_bp" in summary
    assert "depth_imbalance" in summary
    assert "active_buy_pct" in summary
    assert "toxicity_level" in summary

    time_series = data.get("time_series", [])
    print(f"Time Series Points: {len(time_series)}")
    assert len(time_series) == 100
    first_pt = time_series[0]
    for key in ["time", "price", "mid_price", "micro_price", "ofi", "vpin", "depth_imbalance"]:
        assert key in first_pt, f"Key {key} missing in time series point"

    evals = data.get("factor_evaluations", [])
    print(f"Factor Evaluations Count: {len(evals)}")
    assert len(evals) == 6
    for ev in evals:
        print(f" - [{ev['id']}] IC={ev['rank_ic']}, Signal={ev['signal']}")
        assert "rank_ic" in ev
        assert "signal" in ev
        assert "current_value" in ev

    diagnostics = data.get("order_flow_diagnostics", {})
    print("Order Flow Diagnostics:", diagnostics)
    assert "ofi_bias_direction" in diagnostics
    assert "informed_trading_risk" in diagnostics

    print("\nAll Microstructure API tests PASSED successfully!")


if __name__ == "__main__":
    test_microstructure_api()
