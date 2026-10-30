import os
import sys
sys.path.insert(0, os.path.abspath("."))
import requests

def test_stress_testing_api():
    session = requests.Session()
    session.trust_env = False  # 绕过 Windows 本地代理 10808 端口

    base_url = "http://127.0.0.1:8000"

    print("1. Testing POST /api/stress_test/run...")
    payload = {
        "symbols": ["510300", "510500", "518880", "511010"],
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "weights": {
            "510300": 0.3,
            "510500": 0.25,
            "518880": 0.25,
            "511010": 0.2
        }
    }
    resp = session.post(f"{base_url}/api/stress_test/run", json=payload)
    print("STRESS TEST STATUS:", resp.status_code)
    data = resp.json()
    assert resp.status_code == 200, f"Error: {data}"
    stress_data = data["data"]

    summary = stress_data["summary"]
    print("Summary:", summary)
    assert "var_95" in summary
    assert "cvar_95" in summary
    assert "stress_resilience_grade" in summary
    assert "liquidation_days" in summary

    scenarios = stress_data["historical_scenarios"]
    print(f"Historical scenarios count: {len(scenarios)}")
    for sc in scenarios:
        print(f" - {sc['name']}: PortfolioLoss={sc['portfolio_loss_pct']}%, Bmk={sc['bmk_drawdown_pct']}%, RelAlpha={sc['relative_defensive_alpha_pct']}%")

    shocks = stress_data["hypothetical_shocks"]
    print(f"Hypothetical shocks count: {len(shocks)}")
    for sh in shocks:
        print(f" - {sh['shock_name']}: Drop={sh['est_portfolio_drop_pct']}%, Severity={sh['severity']}")

    liq = stress_data["liquidity_analysis"]
    print("Liquidity Analysis: Max Days =", liq["max_liquidation_days"], "Avg Days =", liq["avg_liquidation_days"])

    breakdown = stress_data["asset_loss_breakdown"]
    print(f"Asset loss breakdown count: {len(breakdown)}")
    for b in breakdown:
        print(f" - {b['symbol']}: Weight={b['weight_pct']}%, LossContribRatio={b['loss_contribution_ratio']}%")

    scorecard = stress_data["diagnostic_scorecard"]
    print("Scorecard Grade:", scorecard["grade"], "Diagnosis:", scorecard["diagnosis_text"])

    print("\n2. Testing POST /api/ml/backtest with auto stress_test mount...")
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
    assert "stress_test" in ml_res
    assert ml_res["stress_test"] is not None
    print("Auto stress_test summary in ML backtest:", ml_res["stress_test"]["summary"])

    print("\nALL STRESS TESTING API TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_stress_testing_api()
