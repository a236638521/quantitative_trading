import os
for proxy_key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ[proxy_key] = ""

import requests

def test_factor_correlation():
    print("=== Testing Factor Correlation Matrix ===")
    payload = {
        "symbols": ["510300", "510500", "518880", "511010"],
        "start_date": "2023-01-01",
        "end_date": "2024-01-01"
    }
    res = requests.post("http://127.0.0.1:8000/api/factors/correlation", json=payload).json()
    print("CODE:", res.get("code"))
    data = res.get("data", {})
    print("FACTOR LABELS:", data.get("factor_labels"))
    print("MATRIX SIZE:", len(data.get("matrix", [])), "x", len(data.get("matrix", [[]])[0]))
    print("REDUNDANT PAIRS:", data.get("redundant_pairs"))
    print("ORTHOGONAL PAIRS:", data.get("orthogonal_pairs"))
    print("HEATMAP DATA COUNT:", len(data.get("heatmap_data", [])))

def test_watchdog_scan():
    print("\n=== Testing Signal Watchdog Scan ===")
    res = requests.post("http://127.0.0.1:8000/api/watchdog/scan").json()
    print("CODE:", res.get("code"))
    print("MESSAGE:", res.get("message"))
    signals = res.get("data", {}).get("signals", [])
    print(f"FOUND {len(signals)} SIGNALS:")
    for sig in signals[:3]:
        print(f" - [{sig['action']}] {sig['symbol']} ({sig['symbol_name']}): {sig['reason']} @ {sig['price']}")

def test_watchdog_config():
    print("\n=== Testing Watchdog Config ===")
    cfg_res = requests.get("http://127.0.0.1:8000/api/watchdog/config").json()
    print("CONFIG CODE:", cfg_res.get("code"))
    print("PLATFORM:", cfg_res.get("data", {}).get("webhook_platform"))
    print("TASKS COUNT:", len(cfg_res.get("data", {}).get("monitored_tasks", [])))

if __name__ == "__main__":
    test_factor_correlation()
    test_watchdog_scan()
    test_watchdog_config()
