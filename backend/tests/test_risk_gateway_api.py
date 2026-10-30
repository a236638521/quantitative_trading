import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath("."))
import requests


def test_risk_gateway_api():
    session = requests.Session()
    session.trust_env = False  # 绕过 Windows 本地代理 10808 端口

    base_url = "http://127.0.0.1:8000"

    print("1. Testing GET /api/risk_gateway/config...")
    resp = session.get(f"{base_url}/api/risk_gateway/config")
    print("STATUS:", resp.status_code)
    assert resp.status_code == 200
    cfg = resp.json()["data"]
    print("Initial Config:", cfg)
    assert "max_single_order_amount" in cfg
    assert "max_stock_concentration" in cfg

    print("\n2. Testing POST /api/risk_gateway/config (dynamic update)...")
    update_payload = {
        "max_single_order_amount": 400000.0,
        "max_stock_concentration": 0.25,
        "blacklist_symbols": ["sh.600000", "sz.000002"]
    }
    update_resp = session.post(f"{base_url}/api/risk_gateway/config", json=update_payload)
    assert update_resp.status_code == 200
    new_cfg = update_resp.json()["data"]
    print("Updated Config:", new_cfg)
    assert new_cfg["max_single_order_amount"] == 400000.0
    assert new_cfg["max_stock_concentration"] == 0.25

    print("\n3. Testing POST /api/risk_gateway/validate (Compliant Order)...")
    compliant_order = {
        "order_type": "BUY",
        "symbol": "sh.600519",
        "shares": 100,
        "price": 1800.0,  # 18 万元 < 40 万元
        "portfolio_total_asset": 1000000.0,
        "current_holdings": {}
    }
    comp_resp = session.post(f"{base_url}/api/risk_gateway/validate", json=compliant_order)
    assert comp_resp.status_code == 200
    comp_res = comp_resp.json()["data"]
    print("Compliant Result:", comp_res)
    assert comp_res["passed"] is True
    assert comp_res["action"] == "PASS"

    print("\n4. Testing POST /api/risk_gateway/validate (Violating Max Single Order)...")
    violation_order = {
        "order_type": "BUY",
        "symbol": "sh.600519",
        "shares": 300,
        "price": 1800.0,  # 54 万元 > 40 万元
        "portfolio_total_asset": 2000000.0,
        "current_holdings": {}
    }
    viol_resp = session.post(f"{base_url}/api/risk_gateway/validate", json=violation_order)
    assert viol_resp.status_code == 200
    viol_res = viol_resp.json()["data"]
    print("Violation Result:", viol_res)
    assert viol_res["passed"] is False
    assert viol_res["action"] == "REJECT"
    assert "RULE_01_SINGLE_ORDER_AMOUNT" in viol_res["rule_id"]

    print("\n5. Testing POST /api/risk_gateway/validate (Blacklist Violation)...")
    bl_order = {
        "order_type": "BUY",
        "symbol": "sh.600000",
        "shares": 1000,
        "price": 10.0,
        "portfolio_total_asset": 1000000.0,
        "current_holdings": {}
    }
    bl_resp = session.post(f"{base_url}/api/risk_gateway/validate", json=bl_order)
    assert bl_resp.status_code == 200
    bl_res = bl_resp.json()["data"]
    print("Blacklist Result:", bl_res)
    assert bl_res["passed"] is False
    assert bl_res["rule_id"] == "RULE_08_SECURITY_COMPLIANCE"

    print("\n6. Testing GET /api/risk_gateway/status...")
    stat_resp = session.get(f"{base_url}/api/risk_gateway/status")
    assert stat_resp.status_code == 200
    status_data = stat_resp.json()["data"]
    print("Risk Status Summary:", status_data)
    assert "daily_turnover" in status_data
    assert "intraday_drawdown" in status_data
    assert status_data["intercepted_count"] >= 2

    print("\n7. Testing GET /api/risk_gateway/logs...")
    logs_resp = session.get(f"{base_url}/api/risk_gateway/logs?limit=5")
    assert logs_resp.status_code == 200
    logs = logs_resp.json()["data"]
    print(f"Audit Logs Count: {len(logs)}")
    for l in logs[-3:]:
        print(f" - [{l['decision']}] {l['symbol']} {l['action']} {l['shares']}股 @ ¥{l['price']}: {l['reason']}")

    print("\n8. Testing POST /api/risk_gateway/reset...")
    reset_resp = session.post(f"{base_url}/api/risk_gateway/reset")
    assert reset_resp.status_code == 200
    reset_data = reset_resp.json()["data"]
    print("Reset Status:", reset_data["is_circuit_broken"])
    assert reset_data["is_circuit_broken"] is False

    print("\nALL PRE-TRADE RISK GATEWAY API TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_risk_gateway_api()
