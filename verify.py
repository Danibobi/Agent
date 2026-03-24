#!/usr/bin/env python3
"""
verify.py — Independent ground-truth check against Interactive Brokers.

Cross-references trade_journal.json against IB's own execution records.
Claude cannot fake this: IB's execution history is completely outside
the agent's control. Run this any time you want proof of what actually happened.

Usage:
    python verify.py
    python verify.py --days 30   # look back 30 days of IB executions
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

from ib_insync import IB, ExecutionFilter

import config

JOURNAL_FILE = "trade_journal.json"
PRICE_TOLERANCE = 0.10   # $0.10 tolerance for "prices match"
W = 68


def _load_journal() -> dict:
    if not os.path.exists(JOURNAL_FILE):
        print(f"  trade_journal.json not found — no local trades recorded yet.")
        return {"trades": [], "snapshots": []}
    with open(JOURNAL_FILE) as f:
        return json.load(f)


def _rule(c="─"):
    print(c * W)


def _row(label, value):
    dots = W - 4 - len(label) - len(str(value))
    print(f"  {label}{'.' * max(1, dots)}{value}")


def _section(title):
    print()
    print(f"  {title}")
    _rule()


def _fifo_pnl(fills: list[dict]) -> float:
    """Compute FIFO realised P&L from a list of {symbol, action, qty, price} dicts."""
    from collections import defaultdict
    buy_queues: dict[str, list] = defaultdict(list)
    realized = 0.0
    for f in sorted(fills, key=lambda x: x.get("time", "")):
        sym = f["symbol"]
        qty = f["qty"]
        price = f["price"]
        if f["action"] in ("BOT", "BUY"):
            buy_queues[sym].append({"qty": qty, "price": price})
        elif f["action"] in ("SLD", "SELL"):
            rem = qty
            while rem > 0 and buy_queues[sym]:
                lot = buy_queues[sym][0]
                matched = min(lot["qty"], rem)
                realized += (price - lot["price"]) * matched
                lot["qty"] -= matched
                rem -= matched
                if lot["qty"] == 0:
                    buy_queues[sym].pop(0)
    return round(realized, 2)


def main():
    parser = argparse.ArgumentParser(description="Verify trade_journal.json against IB execution history.")
    parser.add_argument("--days", type=int, default=90,
                        help="How many days of IB execution history to fetch (default: 90)")
    args = parser.parse_args()

    print()
    _rule("═")
    print(f"{'IB EXECUTION VERIFICATION REPORT':^{W}}")
    print(f"{'Ground truth: IB records vs trade_journal.json':^{W}}")
    _rule("═")

    # ── Step 1: Load local journal ────────────────────────────────────────────
    journal = _load_journal()
    journal_trades = journal.get("trades", [])
    # Build lookup: order_id → journal entry
    journal_by_order: dict[int, dict] = {
        t["order_id"]: t for t in journal_trades if t.get("order_id") is not None
    }
    print(f"\n  Journal loaded: {len(journal_trades)} trades recorded locally")

    # ── Step 2: Connect to IB and fetch real executions ───────────────────────
    print(f"\n  Connecting to IB ({config.IB_HOST}:{config.IB_PORT}) ...")
    ib = IB()
    try:
        ib.connect(config.IB_HOST, config.IB_PORT, clientId=config.IB_CLIENT_ID + 99)
    except Exception as e:
        print(f"\n  ERROR: Could not connect to IB: {e}")
        print("  Make sure TWS or IB Gateway is running.")
        sys.exit(1)

    try:
        ef = ExecutionFilter()
        fills = ib.reqExecutions(ef)
        ib.sleep(2)
    finally:
        ib.disconnect()

    print(f"  IB executions fetched: {len(fills)} fills")

    if not fills and not journal_trades:
        print("\n  Nothing to compare. No trades in journal, no fills in IB.")
        return

    # ── Step 3: Build IB fill lookup by orderId ───────────────────────────────
    # Each fill: fill.execution.orderId, fill.contract.symbol,
    #            fill.execution.side ("BOT"/"SLD"), fill.execution.shares,
    #            fill.execution.price, fill.execution.time
    ib_by_order: dict[int, dict] = {}
    ib_fills_list: list[dict] = []
    for fill in fills:
        ex = fill.execution
        order_id = ex.orderId
        entry = {
            "order_id": order_id,
            "symbol": fill.contract.symbol,
            "action": ex.side,           # "BOT" or "SLD"
            "qty": ex.shares,
            "price": ex.price,
            "time": str(ex.time),
            "exec_id": ex.execId,
        }
        ib_fills_list.append(entry)
        # If multiple partial fills for same order, keep the last (or aggregate)
        if order_id not in ib_by_order:
            ib_by_order[order_id] = entry
        else:
            # Accumulate partial fills
            ib_by_order[order_id]["qty"] += ex.shares
            # Weighted average price
            existing = ib_by_order[order_id]
            total_qty = existing["qty"]
            existing["price"] = round(
                (existing["price"] * (total_qty - ex.shares) + ex.price * ex.shares) / total_qty, 4
            )

    # ── Step 4: Reconcile ─────────────────────────────────────────────────────
    confirmed = []
    price_diff = []
    not_filled = []
    untracked = []

    journal_order_ids = set(journal_by_order.keys())
    ib_order_ids = set(ib_by_order.keys())

    for order_id, jt in journal_by_order.items():
        if order_id in ib_by_order:
            ib_fill = ib_by_order[order_id]
            j_price = jt.get("actual_fill_price") or jt.get("estimated_fill")
            ib_price = ib_fill["price"]
            diff = abs((j_price or 0) - ib_price)
            if diff <= PRICE_TOLERANCE:
                confirmed.append({"journal": jt, "ib": ib_fill, "price_diff": diff})
            else:
                price_diff.append({"journal": jt, "ib": ib_fill, "price_diff": round(diff, 4)})
        else:
            not_filled.append(jt)

    for order_id, ib_fill in ib_by_order.items():
        if order_id not in journal_order_ids:
            untracked.append(ib_fill)

    # ── Step 5: Print results ─────────────────────────────────────────────────
    _section(f"CONFIRMED  ({len(confirmed)} trades — journal matches IB)")
    if confirmed:
        print(f"  {'OrderID':>8}  {'Symbol':<6}  {'Action':<4}  {'Qty':>5}  {'Journal$':>9}  {'IB$':>9}  {'Diff':>6}")
        _rule()
        for r in confirmed:
            j, ib_f = r["journal"], r["ib"]
            j_price = j.get("actual_fill_price") or j.get("estimated_fill") or 0
            action_label = "BUY" if ib_f["action"] == "BOT" else "SELL"
            print(f"  {j['order_id']:>8}  {j['symbol']:<6}  {action_label:<4}  "
                  f"{int(j['quantity']):>5}  ${j_price:>8.2f}  ${ib_f['price']:>8.2f}  "
                  f"${r['price_diff']:>5.3f}")
    else:
        print("  None.")

    _section(f"PRICE DISCREPANCY  ({len(price_diff)} trades — in both but prices differ > ${PRICE_TOLERANCE})")
    if price_diff:
        print(f"  {'OrderID':>8}  {'Symbol':<6}  {'Journal$':>9}  {'IB$':>9}  {'Diff':>8}")
        _rule()
        for r in price_diff:
            j, ib_f = r["journal"], r["ib"]
            j_price = j.get("actual_fill_price") or j.get("estimated_fill") or 0
            print(f"  {j['order_id']:>8}  {j['symbol']:<6}  ${j_price:>8.2f}  "
                  f"${ib_f['price']:>8.2f}  ${r['price_diff']:>7.3f}")
    else:
        print("  None — all prices are accurate.")

    _section(f"NOT FILLED  ({len(not_filled)} journal entries with NO matching IB execution)")
    if not_filled:
        print("  These orders were placed (submitted to IB) but never actually filled.")
        print("  Their P&L contribution in the journal is based on estimated prices only.")
        print()
        for t in not_filled:
            fill_price = t.get("actual_fill_price") or t.get("estimated_fill")
            fill_str = f"${fill_price:.2f}" if fill_price else "no price"
            confirmed_str = "fill_confirmed=true" if t.get("fill_confirmed") else "UNCONFIRMED"
            print(f"  OrderID {t['order_id']:>6}  {t['action']:<4} {t['symbol']:<6} "
                  f"x{t['quantity']}  {fill_str}  [{confirmed_str}]  {t.get('timestamp', '')}")
    else:
        print("  None — every journal entry has a matching IB execution. Good.")

    _section(f"UNTRACKED IB FILLS  ({len(untracked)} IB fills NOT in journal)")
    if untracked:
        print("  These trades happened in IB but were not recorded by the agent.")
        print("  Possible cause: manual trades placed directly in TWS.")
        print()
        for f in untracked:
            action_label = "BUY" if f["action"] == "BOT" else "SELL"
            print(f"  OrderID {f['order_id']:>6}  {action_label:<4} {f['symbol']:<6} "
                  f"x{int(f['qty'])}  @ ${f['price']:.2f}  {f['time']}")
    else:
        print("  None — IB and journal are in sync.")

    # ── Step 6: P&L comparison ────────────────────────────────────────────────
    _section("P&L COMPARISON")

    # Journal P&L (from estimated/actual fill prices)
    journal_fills_for_pnl = [
        {
            "symbol": t["symbol"],
            "action": t["action"],
            "qty": t["quantity"],
            "price": t.get("actual_fill_price") or t.get("estimated_fill") or 0,
            "time": t.get("timestamp", ""),
        }
        for t in journal_trades
        if (t.get("actual_fill_price") or t.get("estimated_fill"))
    ]
    journal_pnl = _fifo_pnl(journal_fills_for_pnl)

    # IB-verified P&L (only confirmed fills with real IB prices)
    confirmed_order_ids = {r["journal"]["order_id"] for r in confirmed}
    confirmed_journal_trades = [t for t in journal_trades if t.get("order_id") in confirmed_order_ids]
    ib_fills_for_pnl = [
        {
            "symbol": ib_by_order[t["order_id"]]["symbol"],
            "action": ib_by_order[t["order_id"]]["action"],
            "qty": ib_by_order[t["order_id"]]["qty"],
            "price": ib_by_order[t["order_id"]]["price"],
            "time": ib_by_order[t["order_id"]]["time"],
        }
        for t in confirmed_journal_trades
    ]
    ib_pnl = _fifo_pnl(ib_fills_for_pnl)

    delta = round(ib_pnl - journal_pnl, 2)
    _row("Journal P&L (estimated prices)", f"${journal_pnl:>+,.2f}")
    _row("IB-verified P&L (real fill prices, confirmed only)", f"${ib_pnl:>+,.2f}")
    _row("Delta (IB - Journal)", f"${delta:>+,.2f}")

    if not_filled:
        unconfirmed_count = len(not_filled)
        print(f"\n  NOTE: {unconfirmed_count} journal entr{'y' if unconfirmed_count == 1 else 'ies'} "
              f"have no IB match and are excluded from IB-verified P&L.")

    # ── Summary verdict ───────────────────────────────────────────────────────
    _section("VERDICT")
    issues = len(price_diff) + len(not_filled) + len(untracked)
    if issues == 0 and confirmed:
        print("  ALL CLEAR. Every trade in the journal is confirmed by IB.")
        print("  The agent's reported P&L is accurate.")
    elif len(not_filled) > 0 and len(confirmed) == 0:
        print("  WARNING: No confirmed fills found. Either:")
        print("  - The agent has not completed any trades yet, or")
        print("  - All submitted orders expired/were cancelled without filling.")
        print("  The journal's P&L figures are entirely unconfirmed estimates.")
    else:
        if len(not_filled) > 0:
            print(f"  {len(not_filled)} journal entr{'y' if len(not_filled)==1 else 'ies'} "
                  f"never filled — remove from P&L calculations.")
        if len(price_diff) > 0:
            print(f"  {len(price_diff)} price discrepanc{'y' if len(price_diff)==1 else 'ies'} "
                  f"— IB prices differ from journal estimates.")
        if len(untracked) > 0:
            print(f"  {len(untracked)} untracked IB fill(s) — likely manual TWS trades.")
        print(f"\n  Trust the IB-verified P&L (${ib_pnl:+,.2f}) over the journal P&L (${journal_pnl:+,.2f}).")

    print()
    _rule("═")
    print(f"{'Report generated: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^{W}}")
    _rule("═")
    print()


if __name__ == "__main__":
    main()
