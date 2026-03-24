"""Trade journal and P&L / performance tracking."""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)

JOURNAL_FILE = "trade_journal.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fetch_current_price(symbol: str) -> Optional[float]:
    """Fetch the latest close price via yfinance. Returns None on failure."""
    try:
        hist = yf.Ticker(symbol).history(period="1d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2)
    except Exception as e:
        logger.warning("Could not fetch price for %s: %s", symbol, e)
    return None


def _load_journal() -> dict:
    if os.path.exists(JOURNAL_FILE):
        try:
            with open(JOURNAL_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Could not read %s: %s. Starting fresh.", JOURNAL_FILE, e)
    return {"trades": [], "snapshots": []}


def _save_journal(data: dict) -> None:
    tmp = JOURNAL_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, JOURNAL_FILE)
    except OSError as e:
        logger.error("Failed to save journal: %s", e)


class PerformanceTracker:
    """Records every placed order and portfolio snapshots; computes P&L metrics."""

    def __init__(self):
        self._data = _load_journal()

    # ------------------------------------------------------------------
    # Write methods — called by the agent after each relevant tool call
    # ------------------------------------------------------------------

    def record_trade(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str,
        limit_price: Optional[float],
        order_id: int,
    ) -> None:
        """
        Record a placed order with an estimated fill price.
        LMT orders use limit_price; MKT orders fetch the current yfinance price.
        """
        if order_type == "LMT" and limit_price is not None:
            estimated_fill = limit_price
        else:
            estimated_fill = _fetch_current_price(symbol)
            if estimated_fill is None:
                logger.warning(
                    "Could not estimate fill price for MKT %s %s — "
                    "trade recorded but excluded from P&L.", action, symbol
                )

        self._data["trades"].append({
            "order_id": order_id,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "order_type": order_type,
            "limit_price": limit_price,
            "estimated_fill": estimated_fill,
            "timestamp": _now_iso(),
        })
        _save_journal(self._data)
        price_str = "N/A" if estimated_fill is None else f"{estimated_fill:.2f}"
        logger.info("[Tracker] Recorded %s %dx%s @ %s", action, quantity, symbol, price_str)

    def record_snapshot(self, portfolio: dict) -> None:
        """Append a net-liquidation snapshot whenever get_portfolio is called."""
        net_liq = portfolio.get("net_liquidation")
        if net_liq is None:
            return
        self._data["snapshots"].append({
            "timestamp": _now_iso(),
            "net_liquidation": net_liq,
            "cash": portfolio.get("cash"),
        })
        _save_journal(self._data)

    # ------------------------------------------------------------------
    # P&L computation (pure, in-memory)
    # ------------------------------------------------------------------

    def _compute_pnl(self) -> dict:
        """
        FIFO P&L across all trades that have an estimated fill price.
        Returns a dict keyed by symbol with realized_pnl, open_lots, closed_trades.
        """
        trades = [t for t in self._data["trades"] if t.get("estimated_fill") is not None]

        buy_queues: dict[str, list] = defaultdict(list)
        realized: dict[str, float] = defaultdict(float)
        closed_trades: dict[str, list] = defaultdict(list)

        for trade in trades:
            sym = trade["symbol"]
            qty = trade["quantity"]
            price = trade["estimated_fill"]

            if trade["action"] == "BUY":
                buy_queues[sym].append({"qty": qty, "price": price})

            elif trade["action"] == "SELL":
                remaining = qty
                while remaining > 0 and buy_queues[sym]:
                    lot = buy_queues[sym][0]
                    matched = min(lot["qty"], remaining)
                    pnl = (price - lot["price"]) * matched
                    realized[sym] += pnl
                    closed_trades[sym].append({
                        "symbol": sym,
                        "buy_price": lot["price"],
                        "sell_price": price,
                        "qty": matched,
                        "pnl": round(pnl, 2),
                    })
                    lot["qty"] -= matched
                    remaining -= matched
                    if lot["qty"] == 0:
                        buy_queues[sym].pop(0)

        result = {}
        for sym in set(buy_queues) | set(realized):
            result[sym] = {
                "realized_pnl": round(realized[sym], 2),
                "open_lots": buy_queues[sym],
                "closed_trades": closed_trades[sym],
            }
        return result

    # ------------------------------------------------------------------
    # Public report
    # ------------------------------------------------------------------

    def get_performance_report(self, portfolio: Optional[dict] = None) -> dict:
        """
        Full performance report: P&L, win rate, return %, best/worst trade.

        Pass portfolio=None (recommended for paper trading) to use yfinance
        prices for unrealized P&L. The broker's market_value field reflects
        cost basis rather than live price, so the yfinance path is more accurate.
        """
        pnl_by_sym = self._compute_pnl()
        trades = self._data["trades"]
        snapshots = self._data["snapshots"]

        priced_trades = [t for t in trades if t.get("estimated_fill") is not None]
        all_closed = [ct for v in pnl_by_sym.values() for ct in v["closed_trades"]]

        wins = [ct for ct in all_closed if ct["pnl"] > 0]
        losses = [ct for ct in all_closed if ct["pnl"] < 0]
        win_rate_pct = round(len(wins) / len(all_closed) * 100, 1) if all_closed else None

        total_realized = round(sum(v["realized_pnl"] for v in pnl_by_sym.values()), 2)

        # Unrealized P&L
        unrealized_by_sym: dict[str, float] = {}
        if portfolio:
            for pos in portfolio.get("positions", []):
                sym = pos["symbol"]
                unrealized_by_sym[sym] = round(
                    pos["market_value"] - pos["avg_cost"] * pos["shares"], 2
                )
        else:
            for sym, data in pnl_by_sym.items():
                open_lots = data["open_lots"]
                if not open_lots:
                    continue
                total_qty = sum(lot["qty"] for lot in open_lots)
                if total_qty == 0:
                    continue
                current_price = _fetch_current_price(sym)
                if current_price is None:
                    continue
                avg_cost = sum(lot["qty"] * lot["price"] for lot in open_lots) / total_qty
                unrealized_by_sym[sym] = round((current_price - avg_cost) * total_qty, 2)

        total_unrealized = round(sum(unrealized_by_sym.values()), 2)

        # Total return % from first to latest snapshot
        total_return_pct = None
        if len(snapshots) >= 2:
            initial = snapshots[0]["net_liquidation"]
            latest = snapshots[-1]["net_liquidation"]
            if initial:
                total_return_pct = round((latest - initial) / initial * 100, 2)

        # Per-symbol summary
        all_syms = set(pnl_by_sym) | set(unrealized_by_sym)
        symbol_summary = {
            sym: {
                "realized_pnl": pnl_by_sym.get(sym, {}).get("realized_pnl", 0.0),
                "unrealized_pnl": unrealized_by_sym.get(sym, 0.0),
                "closed_trades": len(pnl_by_sym.get(sym, {}).get("closed_trades", [])),
                "open_lots": len(pnl_by_sym.get(sym, {}).get("open_lots", [])),
            }
            for sym in sorted(all_syms)
        }

        best = max(all_closed, key=lambda x: x["pnl"]) if all_closed else None
        worst = min(all_closed, key=lambda x: x["pnl"]) if all_closed else None

        return {
            "total_trades_recorded": len(priced_trades),
            "closed_trade_count": len(all_closed),
            "win_rate_pct": win_rate_pct,
            "wins": len(wins),
            "losses": len(losses),
            "total_realized_pnl": total_realized,
            "total_unrealized_pnl": total_unrealized,
            "total_pnl": round(total_realized + total_unrealized, 2),
            "total_return_pct": total_return_pct,
            "best_trade": best,
            "worst_trade": worst,
            "pnl_by_symbol": symbol_summary,
            "snapshot_count": len(snapshots),
            "first_snapshot": snapshots[0] if snapshots else None,
            "latest_snapshot": snapshots[-1] if snapshots else None,
            "as_of": _now_iso(),
        }

    def log_startup_summary(self) -> None:
        """Log a concise P&L banner at the start of each cycle. No-ops if no trades yet."""
        priced = [t for t in self._data["trades"] if t.get("estimated_fill") is not None]
        if not priced:
            logger.info("[Tracker] No trades recorded yet.")
            return

        r = self.get_performance_report()
        lines = [
            "=" * 55,
            "  PERFORMANCE SUMMARY",
            f"  Trades recorded  : {r['total_trades_recorded']}",
            f"  Closed trades    : {r['closed_trade_count']}",
            f"  Win rate         : {r['win_rate_pct']}%" if r["win_rate_pct"] is not None else "  Win rate         : N/A",
            f"  Realized P&L     : ${r['total_realized_pnl']:+.2f}",
            f"  Unrealized P&L   : ${r['total_unrealized_pnl']:+.2f}",
            f"  Total P&L        : ${r['total_pnl']:+.2f}",
        ]
        if r["total_return_pct"] is not None:
            lines.append(f"  Total return     : {r['total_return_pct']:+.2f}%")
        if r["best_trade"]:
            bt = r["best_trade"]
            lines.append(f"  Best trade       : +${bt['pnl']:.2f}  ({bt['symbol']} x{bt['qty']})")
        if r["worst_trade"]:
            wt = r["worst_trade"]
            lines.append(f"  Worst trade      : ${wt['pnl']:.2f}  ({wt['symbol']} x{wt['qty']})")
        lines.append("=" * 55)
        for line in lines:
            logger.info(line)
