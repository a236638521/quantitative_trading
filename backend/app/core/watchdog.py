import os
import time
import json
import hmac
import hashlib
import base64
import urllib.parse
import logging
from typing import Dict, List, Any, Optional
import requests
import pandas as pd

from ..data.provider import MarketDataProvider
from ..strategies.dual_ma import DualMAStrategy
from ..strategies.turtle import TurtleStrategy
from ..strategies.bollinger_grid import BollingerGridStrategy
from ..strategies.etf_rotation import ETFMomentumRotationStrategy
from ..strategies.multi_factor import MultiFactorStockSelectionStrategy
from ..core.portfolio import Portfolio

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "watchdog_config.json"))

DEFAULT_MONITOR_TASKS = [
    {
        "id": "task_dual_ma_300",
        "strategy_name": "DualMAStrategy",
        "strategy_label": "双均线交叉策略",
        "symbol": "510300",
        "symbol_name": "沪深300ETF",
        "params": {"fast_period": 5, "slow_period": 20, "position_pct": 0.95},
        "enabled": True
    },
    {
        "id": "task_turtle_500",
        "strategy_name": "TurtleStrategy",
        "strategy_label": "海龟唐奇安突破策略",
        "symbol": "510500",
        "symbol_name": "中证500ETF",
        "params": {"entry_period": 20, "exit_period": 10, "atr_period": 14, "position_pct": 0.95},
        "enabled": True
    },
    {
        "id": "task_etf_rotation",
        "strategy_name": "ETFMomentumRotationStrategy",
        "strategy_label": "大类资产 ETF 动量轮动",
        "symbol": "510300,510500,513100,518880,511010",
        "symbol_name": "ETF 全天候资产池",
        "params": {"lookback_period": 20, "top_n": 1, "use_risk_adjusted": True},
        "enabled": True
    },
    {
        "id": "task_multi_factor",
        "strategy_name": "MultiFactorStockSelectionStrategy",
        "strategy_label": "核心蓝筹多因子截面选股",
        "symbol": "600519,000858,300750,002594,601318,600036",
        "symbol_name": "核心蓝筹选股池",
        "params": {"top_n": 3, "lookback_period": 20},
        "enabled": True
    }
]


class WebhookNotifier:
    """多渠道机器人预警推送器 (飞书 / 钉钉 / 企业微信)"""

    @staticmethod
    def _sign_dingtalk(secret: str) -> tuple:
        """钉钉机器人加签计算 (HMAC-SHA256)"""
        timestamp = str(round(time.time() * 1000))
        secret_enc = secret.encode("utf-8")
        string_to_sign = f"{timestamp}\n{secret}".encode("utf-8")
        hmac_code = hmac.new(secret_enc, string_to_sign, digestmod=hashlib.sha256).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        return timestamp, sign

    @classmethod
    def send_notification(
        cls,
        platform: str,
        webhook_url: str,
        secret: Optional[str],
        title: str,
        content_markdown: str,
        action_type: str = "BUY"
    ) -> Dict[str, Any]:
        """
        向指定平台发送格式化通知卡片
        """
        if not webhook_url or not webhook_url.startswith("http"):
            return {"success": False, "message": "Webhook URL 格式不合法"}

        headers = {"Content-Type": "application/json"}
        platform = platform.lower()

        # 临时清空可能导致国内机器人接口卡死的代理
        req_proxies = {"http": None, "https": None}

        try:
            if platform == "feishu" or platform == "lark":
                # 飞书富文本互动卡片
                header_color = "green" if action_type == "BUY" else ("red" if action_type == "SELL" else "blue")
                payload = {
                    "msg_type": "interactive",
                    "card": {
                        "header": {
                            "title": {"tag": "plain_text", "content": f"🚨 {title}"},
                            "template": header_color
                        },
                        "elements": [
                            {
                                "tag": "div",
                                "text": {
                                    "tag": "lark_md",
                                    "content": content_markdown
                                }
                            },
                            {
                                "tag": "hr"
                            },
                            {
                                "tag": "note",
                                "elements": [
                                    {
                                        "tag": "plain_text",
                                        "content": f"来源: AlphaQuant 量化盯盘引擎 | 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                                    }
                                ]
                            }
                        ]
                    }
                }
                res = requests.post(webhook_url, json=payload, headers=headers, timeout=8, proxies=req_proxies)
                res_data = res.json()
                if res_data.get("code") == 0 or res_data.get("StatusCode") == 0:
                    return {"success": True, "message": "飞书卡片推送成功", "raw": res_data}
                return {"success": False, "message": f"飞书返回错误: {res_data.get('msg', str(res_data))}"}

            elif platform == "dingtalk":
                # 钉钉机器人 (支持加签安全设置)
                target_url = webhook_url
                if secret and secret.strip():
                    timestamp, sign = cls._sign_dingtalk(secret.strip())
                    connector = "&" if "?" in target_url else "?"
                    target_url = f"{target_url}{connector}timestamp={timestamp}&sign={sign}"

                action_color = "#10b981" if action_type == "BUY" else ("#ef4444" if action_type == "SELL" else "#38bdf8")
                markdown_text = f"### 🚨 {title}\n\n{content_markdown}\n\n---\n> <font color='{action_color}'>**AlphaQuant 盘中量化预警**</font> | {time.strftime('%H:%M:%S')}"

                payload = {
                    "msgtype": "markdown",
                    "markdown": {
                        "title": title,
                        "text": markdown_text
                    }
                }
                res = requests.post(target_url, json=payload, headers=headers, timeout=8, proxies=req_proxies)
                res_data = res.json()
                if res_data.get("errcode") == 0:
                    return {"success": True, "message": "钉钉消息推送成功", "raw": res_data}
                return {"success": False, "message": f"钉钉返回错误: {res_data.get('errmsg', str(res_data))}"}

            elif platform == "wecom":
                # 企业微信机器人
                action_color = "info" if action_type == "BUY" else ("warning" if action_type == "SELL" else "comment")
                markdown_text = f"### 🚨 {title}\n{content_markdown}\n\n> <font color='{action_color}'>AlphaQuant 盘中量化预警</font>"
                payload = {
                    "msgtype": "markdown",
                    "markdown": {
                        "content": markdown_text
                    }
                }
                res = requests.post(webhook_url, json=payload, headers=headers, timeout=8, proxies=req_proxies)
                res_data = res.json()
                if res_data.get("errcode") == 0:
                    return {"success": True, "message": "企业微信推送成功", "raw": res_data}
                return {"success": False, "message": f"企业微信返回错误: {res_data.get('errmsg', str(res_data))}"}

            else:
                # 通用 Webhook POST
                payload = {
                    "title": title,
                    "content": content_markdown,
                    "action": action_type,
                    "timestamp": time.time()
                }
                res = requests.post(webhook_url, json=payload, headers=headers, timeout=8, proxies=req_proxies)
                return {"success": res.status_code == 200, "message": f"通用 Webhook 响应 HTTP {res.status_code}"}

        except Exception as e:
            logger.exception("Webhook 发送异常")
            return {"success": False, "message": f"推送异常: {str(e)}"}


class SignalWatchdog:
    """实盘模拟盯盘与实时信号侦测引擎"""

    def __init__(self):
        self.data_provider = MarketDataProvider()
        self._ensure_config()

    def _ensure_config(self):
        """确保配置持久化文件存在"""
        if not os.path.exists(CONFIG_PATH):
            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            default_data = {
                "webhook_platform": "feishu",
                "webhook_url": "",
                "webhook_secret": "",
                "auto_alert_enabled": False,
                "monitored_tasks": DEFAULT_MONITOR_TASKS
            }
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(default_data, f, ensure_ascii=False, indent=2)

    def load_config(self) -> Dict[str, Any]:
        """读取持久化盯盘配置"""
        self._ensure_config()
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {
                "webhook_platform": "feishu",
                "webhook_url": "",
                "webhook_secret": "",
                "auto_alert_enabled": False,
                "monitored_tasks": DEFAULT_MONITOR_TASKS
            }

    def save_config(self, cfg: Dict[str, Any]):
        """保存盯盘配置"""
        self._ensure_config()
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    def scan_live_signals(self) -> List[Dict[str, Any]]:
        """
        全量扫描已启用的监控策略，侦测最新交易日触发的实时操作信号
        """
        cfg = self.load_config()
        tasks = [t for t in cfg.get("monitored_tasks", []) if t.get("enabled", True)]
        signals_found = []

        end_date = time.strftime("%Y-%m-%d")
        # 抓取最近 180 天的数据以满足均线与唐奇安指标的 lookback 计算
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=180)).strftime("%Y-%m-%d")

        for task in tasks:
            strat_name = task["strategy_name"]
            sym = task["symbol"]
            params = task.get("params", {})

            try:
                if strat_name == "DualMAStrategy":
                    df = self.data_provider.fetch_daily_bars(sym, start_date=start_date, end_date=end_date)
                    if df.empty or len(df) < 25:
                        continue
                    strat = DualMAStrategy(params=params)
                    df_ind = strat.prepare_indicators(df)
                    last_row = df_ind.iloc[-1]
                    prev_row = df_ind.iloc[-2]

                    # 检查最新交易日是否产生交叉
                    is_golden = (prev_row.get("ma_diff", 0) <= 0) and (last_row.get("ma_diff", 0) > 0)
                    is_death = (prev_row.get("ma_diff", 0) >= 0) and (last_row.get("ma_diff", 0) < 0)

                    if is_golden:
                        signals_found.append({
                            "id": f"sig_{sym}_{last_row['date']}_buy",
                            "strategy_id": strat_name,
                            "strategy_label": task["strategy_label"],
                            "symbol": sym,
                            "symbol_name": task.get("symbol_name", sym),
                            "action": "BUY",
                            "price": round(float(last_row["close"]), 3),
                            "date": str(last_row["date"]),
                            "reason": f"MA{params.get('fast_period', 5)} 快线上穿 MA{params.get('slow_period', 20)} 慢线形成低位金叉",
                            "status": "READY"
                        })
                    elif is_death:
                        signals_found.append({
                            "id": f"sig_{sym}_{last_row['date']}_sell",
                            "strategy_id": strat_name,
                            "strategy_label": task["strategy_label"],
                            "symbol": sym,
                            "symbol_name": task.get("symbol_name", sym),
                            "action": "SELL",
                            "price": round(float(last_row["close"]), 3),
                            "date": str(last_row["date"]),
                            "reason": f"MA{params.get('fast_period', 5)} 快线下穿 MA{params.get('slow_period', 20)} 慢线形成死叉",
                            "status": "READY"
                        })

                elif strat_name == "TurtleStrategy":
                    df = self.data_provider.fetch_daily_bars(sym, start_date=start_date, end_date=end_date)
                    if df.empty or len(df) < 25:
                        continue
                    strat = TurtleStrategy(params=params)
                    df_ind = strat.prepare_indicators(df)
                    last_row = df_ind.iloc[-1]
                    high_bound = last_row.get("donchian_high", 999999)
                    low_bound = last_row.get("donchian_low", 0)
                    close = float(last_row["close"])

                    if close > high_bound:
                        signals_found.append({
                            "id": f"sig_{sym}_{last_row['date']}_buy",
                            "strategy_id": strat_name,
                            "strategy_label": task["strategy_label"],
                            "symbol": sym,
                            "symbol_name": task.get("symbol_name", sym),
                            "action": "BUY",
                            "price": round(close, 3),
                            "date": str(last_row["date"]),
                            "reason": f"收盘价突破 {params.get('entry_period', 20)} 日唐奇安通道高点 ({high_bound:.2f})",
                            "status": "READY"
                        })
                    elif close < low_bound:
                        signals_found.append({
                            "id": f"sig_{sym}_{last_row['date']}_sell",
                            "strategy_id": strat_name,
                            "strategy_label": task["strategy_label"],
                            "symbol": sym,
                            "symbol_name": task.get("symbol_name", sym),
                            "action": "SELL",
                            "price": round(close, 3),
                            "date": str(last_row["date"]),
                            "reason": f"收盘价跌破 {params.get('exit_period', 10)} 日唐奇安通道低点 ({low_bound:.2f})",
                            "status": "READY"
                        })

                elif strat_name == "ETFMomentumRotationStrategy":
                    pool_syms = [s.strip() for s in sym.split(",") if s.strip()]
                    multi_data = self.data_provider.fetch_multi_daily_bars(pool_syms, start_date=start_date, end_date=end_date)
                    if multi_data and len(multi_data) >= 2:
                        strat = ETFMomentumRotationStrategy(params=params)
                        lookback = params.get("lookback_period", 20)
                        # 计算当前各标的动量得分
                        mom_scores = {}
                        for s_code, s_df in multi_data.items():
                            if len(s_df) >= lookback:
                                c = s_df["close"].values
                                m_val = (c[-1] / c[-lookback] - 1.0)
                                mom_scores[s_code] = m_val

                        if mom_scores:
                            top_sym = max(mom_scores, key=mom_scores.get)
                            top_val = mom_scores[top_sym]
                            last_date = str(multi_data[top_sym]["date"].iloc[-1])
                            last_price = float(multi_data[top_sym]["close"].iloc[-1])

                            if top_val > 0:
                                signals_found.append({
                                    "id": f"sig_etf_{top_sym}_{last_date}_rebalance",
                                    "strategy_id": strat_name,
                                    "strategy_label": task["strategy_label"],
                                    "symbol": top_sym,
                                    "symbol_name": f"轮动动量领涨标的 ({top_sym})",
                                    "action": "REBALANCE",
                                    "price": round(last_price, 3),
                                    "date": last_date,
                                    "reason": f"大类资产动量排名第一 (20日动量={top_val*100:+.2f}%)，建议调仓配置持有",
                                    "status": "READY"
                                })
                            else:
                                signals_found.append({
                                    "id": f"sig_etf_defense_{last_date}",
                                    "strategy_id": strat_name,
                                    "strategy_label": task["strategy_label"],
                                    "symbol": "CASH",
                                    "symbol_name": "全市场避险空仓",
                                    "action": "SELL",
                                    "price": 1.0,
                                    "date": last_date,
                                    "reason": f"大类资产池所有标的动量全部为负 (领涨者仍下跌 {top_val*100:.2f}%)，建议空仓避险防守",
                                    "status": "READY"
                                })

            except Exception as e:
                logger.warning(f"扫描策略 {strat_name} 标的 {sym} 异常: {e}")

        return signals_found
