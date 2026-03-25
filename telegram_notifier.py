"""Optional Telegram push notifications for the trading agent.

If TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID are not set, all methods
silently no-op — the agent never crashes due to Telegram issues.

Setup:
  1. Message @BotFather on Telegram → /newbot → copy the token
  2. Start a chat with your bot
  3. Visit https://api.telegram.org/bot<TOKEN>/getUpdates → find "chat":{"id":...}
  4. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to your .env file
"""

import logging

import requests

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, token: str | None, chat_id: str | None):
        self._enabled = bool(token and chat_id)
        self._url = f"https://api.telegram.org/bot{token}/sendMessage"
        self._chat_id = chat_id

    def send(self, text: str) -> None:
        if not self._enabled:
            return
        try:
            requests.post(
                self._url,
                json={"chat_id": self._chat_id, "text": text},
                timeout=10,
            )
        except Exception as e:
            logger.warning("Telegram send failed: %s", e)

    def notify_startup(self, watchlist: list[str], mode_label: str, interval_sec: int) -> None:
        self.send(
            f"🚀 Trading Agent started\n"
            f"Mode: {mode_label}\n"
            f"Watchlist: {', '.join(watchlist)}\n"
            f"Interval: {interval_sec // 60} min"
        )

    def notify_trade(
        self,
        action: str,
        symbol: str,
        quantity: int,
        order_type: str,
        price: float | None,
        status: str,
        paper: bool,
    ) -> None:
        icon = "📈" if action == "BUY" else "📉"
        price_str = f"@ ${price:.2f}" if price is not None else "(market)"
        mode_tag = "[PAPER]" if paper else "[LIVE]"
        self.send(
            f"{icon} {action} order placed\n"
            f"{symbol} × {quantity} {price_str} ({order_type})\n"
            f"Status: {status} | {mode_tag}"
        )

    def notify_cycle_complete(self, trade_count: int, net_liq: float | None) -> None:
        from datetime import datetime, timezone
        time_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
        nlv_str = f" | NLV: ${net_liq:,.0f}" if net_liq is not None else ""
        trades_str = f"{trade_count} trade(s) placed" if trade_count else "No trades placed"
        self.send(f"✅ Cycle complete — {time_str}\n{trades_str}{nlv_str}")

    def notify_error(self, error_msg: str) -> None:
        self.send(f"❌ Cycle failed\n{error_msg}")
