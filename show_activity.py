#!/usr/bin/env python3
"""
show_activity.py — Human-readable daily digest of what the bot did.

Reads daily_log.json (written by the agent after every cycle) and shows
exactly what Claude decided, what orders were placed, and how the portfolio
moved — with no IB connection needed.

Usage:
    python show_activity.py           # today's activity
    python show_activity.py --days 7  # last 7 days
    python show_activity.py --all     # everything since launch
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta

DAILY_LOG_FILE = "daily_log.json"
JOURNAL_FILE = "trade_journal.json"
W = 66


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _parse_iso(ts: str) -> datetime:
    """Parse ISO timestamp, return UTC-aware datetime."""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)


def _fmt_local(ts: str) -> str:
    """Format ISO timestamp as local time string."""
    try:
        dt = _parse_iso(ts).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


def _rule(c="─", width=W):
    print(c * width)


def main():
    parser = argparse.ArgumentParser(description="Show daily bot activity digest.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--days", type=int, default=1,
                       help="Show activity for the last N days (default: 1 = today)")
    group.add_argument("--all", action="store_true",
                       help="Show all activity since launch")
    args = parser.parse_args()

    log_data = _load_json(DAILY_LOG_FILE)
    cycles = log_data.get("cycles", [])

    if not cycles:
        print("\n  No activity recorded yet.")
        print(f"  (Looking for {DAILY_LOG_FILE})")
        print("  The bot writes this file after every cycle.\n")
        sys.exit(0)

    # Filter by date range
    if args.all:
        visible = cycles
        period_label = "ALL TIME"
    else:
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
        visible = [c for c in cycles if _parse_iso(c.get("timestamp", "")) >= cutoff]
        if args.days == 1:
            period_label = f"TODAY  ({datetime.now().strftime('%Y-%m-%d')})"
        else:
            period_label = f"LAST {args.days} DAYS"

    print()
    _rule("═")
    print(f"{'BOT ACTIVITY DIGEST':^{W}}")
    print(f"{period_label:^{W}}")
    _rule("═")
    print(f"  Total cycles on record : {len(cycles)}")
    print(f"  Cycles shown           : {len(visible)}")

    if not visible:
        print(f"\n  No activity in the last {args.days} day(s).")
        print(f"  Try: python show_activity.py --days 30  or  --all\n")
        return

    prev_nlv = None

    for i, cycle in enumerate(visible, 1):
        ts = cycle.get("timestamp", "")
        decisions = cycle.get("claude_reasoning", [])
        orders = cycle.get("orders_placed", [])
        snap = cycle.get("portfolio_snapshot")

        # Cycle header
        print()
        _rule("─")
        cycle_index = cycles.index(cycle) + 1
        print(f"  Cycle #{cycle_index:<4}  {_fmt_local(ts)}")
        _rule("─")

        # Claude's reasoning
        if decisions:
            print("  Claude's reasoning:")
            for text in decisions:
                # Indent each paragraph
                for line in text.strip().splitlines():
                    line = line.strip()
                    if line:
                        # Word-wrap at W-6 chars
                        while len(line) > W - 6:
                            cut = line[:W - 6].rfind(" ")
                            cut = cut if cut > 0 else W - 6
                            print(f"    {line[:cut]}")
                            line = line[cut:].lstrip()
                        if line:
                            print(f"    {line}")
            print()
        else:
            print("  Claude's reasoning:  (no text output this cycle)")
            print()

        # Orders
        if orders:
            print("  Orders placed:")
            for o in orders:
                sym = o.get("symbol", "?")
                action = o.get("action", "?")
                qty = o.get("quantity", "?")
                otype = o.get("order_type", "?")
                lp = o.get("limit_price")
                status = o.get("status", "?")
                actual = o.get("actual_fill_price")

                price_str = f"@ ${lp:.2f} {otype}" if lp else otype
                fill_str = f"  [FILLED @ ${actual:.2f}]" if actual else f"  [{status}]"
                confirmed = "[CONFIRMED]" if actual else "[unconfirmed]"
                print(f"    {action:<4} {sym:<6} x{qty:<5} {price_str:<20}{fill_str}  {confirmed}")
        else:
            print("  Orders placed:  none")

        # Portfolio snapshot
        if snap:
            nlv = snap.get("net_liquidation")
            cash = snap.get("cash")
            nlv_str = f"${nlv:>12,.2f}" if nlv is not None else "N/A"
            cash_str = f"${cash:>12,.2f}" if cash is not None else "N/A"

            delta_str = ""
            if prev_nlv is not None and nlv is not None:
                delta = nlv - prev_nlv
                delta_str = f"  ({delta:>+,.2f} this cycle)"
            prev_nlv = nlv

            print()
            print(f"  Portfolio:  NLV {nlv_str}  Cash {cash_str}{delta_str}")
        else:
            print()
            print("  Portfolio:  no snapshot this cycle (get_portfolio not called)")

    # Summary footer
    print()
    _rule("═")
    total_orders = sum(len(c.get("orders_placed", [])) for c in visible)
    confirmed_orders = sum(
        1 for c in visible
        for o in c.get("orders_placed", [])
        if o.get("actual_fill_price") is not None
    )
    cycles_with_trades = sum(1 for c in visible if c.get("orders_placed"))

    print(f"  SUMMARY  ({len(visible)} cycles shown)")
    _rule()
    dots = lambda l, v: f"  {l}{'.' * max(1, W - 4 - len(l) - len(str(v)))}{v}"
    print(dots("Cycles with trades", cycles_with_trades))
    print(dots("Total orders placed", total_orders))
    print(dots("Confirmed fills (IB returned fill price)", confirmed_orders))
    print(dots("Unconfirmed (submitted but fill price unknown)",
               total_orders - confirmed_orders))

    if total_orders > 0 and confirmed_orders < total_orders:
        print()
        print("  TIP: Run  python verify.py  to cross-check unconfirmed orders")
        print("       against IB's own execution records.")

    print()
    _rule("═")
    print(f"  Files:  {DAILY_LOG_FILE}  |  {JOURNAL_FILE}")
    print(f"  To see all history:   python show_activity.py --all")
    print(f"  To verify with IB:    python verify.py")
    _rule("═")
    print()


if __name__ == "__main__":
    main()
