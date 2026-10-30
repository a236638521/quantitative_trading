import os
import sys
sys.path.insert(0, os.path.abspath("."))
import requests

def test_attribution_api():
    session = requests.Session()
    session.trust_env = False  # 绕过 Windows 本地代理 10808 端口

    base_url = "http://127.0.0.1:8000"

    print("1. Testing POST /api/attribution/analyze...")
    payload = {
        "symbols": ["510300", "510500", "518880", "511010"],
        "start_date": "2023-01-01",
        "end_date": "2024-01-01"
    }
    resp = session.post(f"{base_url}/api/attribution/analyze", json=payload)
    print("ATTRIBUTION STATUS:", resp.status_code)
    data = resp.json()
    assert resp.status_code == 200, f"Error: {data}"
    attrib = data["data"]

    summary = attrib["summary"]
    print("Summary:", summary)
    assert "total_strategy_return" in summary
    assert "allocation_effect" in summary
    assert "selection_effect" in summary

    sectors = attrib["sector_attribution"]
    print(f"Sector attribution rows count: {len(sectors)}")
    for s in sectors[:3]:
        print(f" - {s['sector']}: StrategyW={s['avg_weight_strategy']}%, AllocEff={s['allocation_effect_pct']}%, TotalContrib={s['total_contribution_pct']}%")

    waterfall = attrib["waterfall"]
    print(f"Waterfall steps count: {len(waterfall)}")
    for w in waterfall:
        print(f" - {w['name']}: {w['value']}% ({w['type']})")

    style = attrib["style_exposure"]
    print("Style Diagnosis:", style["diagnosis"]["style_label"])
    print("Primary tilt:", style["diagnosis"]["primary_tilt"])

    print("\n2. Testing POST /api/ml/backtest with auto attribution...")
    ml_payload = {
        "symbols": ["510300", "510500", "518880", "511010"],
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "top_n": 2,
        "allocation_method": "risk_parity"
    }
    ml_resp = session.post(f"{base_url}/api/ml/backtest", json=ml_payload)
    print("ML BACKTEST STATUS:", ml_resp.status_code)
    ml_data = ml_resp.json()
    assert ml_resp.status_code == 200, f"Error: {ml_data}"
    ml_res = ml_data["data"]
    assert "attribution" in ml_res
    assert ml_res["attribution"] is not None
    print("Auto attribution summary in ML backtest:", ml_res["attribution"]["summary"])

    print("\nALL ATTRIBUTION API TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_attribution_api()
