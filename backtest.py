#!/usr/bin/env python3
"""
backtest.py — Offline backtester for the Claude trading agent strategy.

Runs the same RSI/MACD/SMA signals the agent uses against real historical data
from yfinance. No IB connection, no Claude API needed.

Usage:
    python backtest.py
    python backtest.py --symbols AAPL MSFT NVDA
    python backtest.py --period 2y --capital 50000
    python backtest.py --symbols GOOGL AMZN --period 5y
"""

import argparse
import math
import sys
from datetime import datetime

import pandas as pd
import yfinance as yf

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
DEFAULT_PERIOD = "2y"
DEFAULT_CAPITAL = 100_000.0
MAX_POS_PCT = 0.10          # mirrors config.MAX_POSITION_SIZE_PCT
RISK_FREE_RATE = 0.04       # annualised, for Sharpe calculation
TRADING_DAYS_PER_YEAR = 252

RSI_BUY_THRESHOLD = 35      # RSI crosses below this → potential buy
RSI_SELL_THRESHOLD = 65     # RSI crosses above this → potential sell


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest the Claude trading agent strategy.")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                   metavar="SYM", help="Space-separated ticker symbols (default: AAPL MSFT GOOGL AMZN NVDA)")
    p.add_argument("--period", default=DEFAULT_PERIOD,
                   choices=["1y", "2y", "5y", "max"],
                   help="Historical period to fetch (default: 2y)")
    p.add_argument("--capital", type=float, default=DEFAULT_CAPITAL,
                   help="Starting cash in USD (default: 100000)")
    return p.parse_args()


# ── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, period: str) -> pd.DataFrame:
    """Returns OHLCV DataFrame sorted ascending with timezone stripped."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    if df.empty:
        raise ValueError(f"No data returned for {symbol!r} with period={period!r}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.sort_index(inplace=True)
    return df


# ── Indicator Computation ─────────────────────────────────────────────────────
# Mirrors data.py formulas exactly (same EWM spans, same RSI formula).

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Identical to data.py _compute_rsi."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds SMA_20, SMA_50, RSI, MACD, MACD_signal, MACD_hist columns."""
    close = df["Close"]
    df = df.copy()
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["RSI"] = _compute_rsi(close)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df


# ── Signal Generation ─────────────────────────────────────────────────────────
# Translates the trading rules from agent.py's SYSTEM_PROMPT into boolean signals.
#
# BUY : RSI crosses below RSI_BUY_THRESHOLD  AND  MACD > signal  AND  SMA20 > SMA50
# SELL: RSI crosses above RSI_SELL_THRESHOLD  OR   MACD crosses below signal
#
# Cross-detection (condition flips False→True) produces one signal per entry
# into the zone rather than re-triggering on every bar inside it.
# News sentiment is omitted — no historical API exists; technicals are the
# dominant driver per the SYSTEM_PROMPT.

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rsi = df["RSI"]
    macd = df["MACD"]
    sig = df["MACD_signal"]
    sma20 = df["SMA_20"]
    sma50 = df["SMA_50"]

    rsi_oversold = rsi < RSI_BUY_THRESHOLD
    rsi_overbought = rsi > RSI_SELL_THRESHOLD
    macd_bullish = macd > sig
    macd_bearish = macd < sig

    # Condition flips to True today from False yesterday
    rsi_cross_down = rsi_oversold & ~rsi_oversold.shift(1).fillna(False)
    rsi_cross_up = rsi_overbought & ~rsi_overbought.shift(1).fillna(False)
    macd_cross_down = macd_bearish & ~macd_bearish.shift(1).fillna(False)

    # Uptrend filter: mirrors SYSTEM_PROMPT's SMA crossover rule
    uptrend = sma20 > sma50

    buy_signal = rsi_cross_down & macd_bullish & uptrend
    sell_signal = rsi_cross_up | macd_cross_down

    df["signal"] = 0
    df.loc[buy_signal, "signal"] = 1
    df.loc[sell_signal, "signal"] = -1

    return df


# ── Trade Simulation ──────────────────────────────────────────────────────────

def simulate(symbols: list[str], period: str, starting_capital: float) -> dict:
    """
    Walk-forward simulation across all symbols with a single shared cash pool.
    Returns a results dict consumed by compute_metrics() and print_report().
    """
    print(f"\nFetching data for {len(symbols)} symbol(s) (period={period}) ...")

    symbol_dfs: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []

    for sym in symbols:
        try:
            df = fetch_ohlcv(sym, period)
            df = add_indicators(df)
            df = generate_signals(df)
            symbol_dfs[sym] = df
            print(f"  {sym}: {len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})")
        except Exception as e:
            print(f"  {sym}: SKIPPED — {e}")
            skipped.append(sym)

    if not symbol_dfs:
        print("No data could be fetched. Exiting.")
        sys.exit(1)

    all_dates = sorted(set().union(*[set(df.index) for df in symbol_dfs.values()]))

    cash = starting_capital
    positions: dict[str, dict] = {}   # sym → {qty, entry_price, entry_date}
    trades: list[dict] = []            # closed trades
    equity_curve: list[tuple] = []    # (datetime, portfolio_value)

    for dt in all_dates:
        for sym, df in symbol_dfs.items():
            if dt not in df.index:
                continue
            row = df.loc[dt]
            signal = row["signal"]
            price = row["Close"]

            if pd.isna(price) or price <= 0:
                continue

            # BUY
            if signal == 1 and sym not in positions:
                # Compute current total equity to size the position
                current_equity = cash + sum(
                    positions[s]["qty"] * (
                        symbol_dfs[s].loc[dt, "Close"]
                        if dt in symbol_dfs[s].index
                        else positions[s]["entry_price"]
                    )
                    for s in positions
                )
                allocation = current_equity * MAX_POS_PCT
                qty = int(allocation // price)
                cost = qty * price
                if qty > 0 and cost <= cash:
                    cash -= cost
                    positions[sym] = {
                        "qty": qty,
                        "entry_price": price,
                        "entry_date": dt,
                    }

            # SELL
            elif signal == -1 and sym in positions:
                pos = positions.pop(sym)
                qty = pos["qty"]
                pnl = (price - pos["entry_price"]) * qty
                hold_days = (dt - pos["entry_date"]).days
                cash += qty * price
                trades.append({
                    "symbol": sym,
                    "entry_date": pos["entry_date"],
                    "exit_date": dt,
                    "entry_price": pos["entry_price"],
                    "exit_price": price,
                    "qty": qty,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl / (pos["entry_price"] * qty) * 100, 2),
                    "hold_days": hold_days,
                })

        # End-of-day portfolio snapshot
        open_value = sum(
            pos["qty"] * (
                symbol_dfs[sym].loc[dt, "Close"]
                if dt in symbol_dfs[sym].index
                else pos["entry_price"]
            )
            for sym, pos in positions.items()
        )
        equity_curve.append((dt, cash + open_value))

    # Mark remaining open positions to last available price
    open_trades: list[dict] = []
    for sym, pos in positions.items():
        df = symbol_dfs[sym]
        last_price = float(df["Close"].iloc[-1])
        last_date = df.index[-1]
        pnl = (last_price - pos["entry_price"]) * pos["qty"]
        open_trades.append({
            "symbol": sym,
            "entry_date": pos["entry_date"],
            "exit_date": last_date,
            "entry_price": pos["entry_price"],
            "exit_price": last_price,
            "qty": pos["qty"],
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / (pos["entry_price"] * pos["qty"]) * 100, 2),
            "hold_days": (last_date - pos["entry_date"]).days,
            "open_at_end": True,
        })

    return {
        "symbols": list(symbol_dfs.keys()),
        "skipped": skipped,
        "starting_capital": starting_capital,
        "trades": trades,
        "open_trades": open_trades,
        "equity_curve": equity_curve,
        "period": period,
        "start_date": all_dates[0],
        "end_date": all_dates[-1],
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: dict) -> dict:
    trades = results["trades"]              # closed (hit exit signal)
    open_trades = results["open_trades"]    # still open at end
    equity_curve = results["equity_curve"]
    start_capital = results["starting_capital"]
    all_trades = trades + open_trades

    # Total / annualised return
    if equity_curve:
        final_value = equity_curve[-1][1]
        total_return = (final_value - start_capital) / start_capital
        n_years = (results["end_date"] - results["start_date"]).days / 365.25
        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    else:
        final_value = start_capital
        total_return = ann_return = 0.0

    # Max drawdown
    max_dd = 0.0
    if equity_curve:
        peak = equity_curve[0][1]
        for _, v in equity_curve:
            peak = max(peak, v)
            max_dd = max(max_dd, (peak - v) / peak)

    # Sharpe ratio (annualised)
    sharpe = 0.0
    if len(equity_curve) >= 2:
        eq = pd.Series([v for _, v in equity_curve])
        daily_ret = eq.pct_change().dropna()
        daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
        excess = daily_ret - daily_rf
        if excess.std() > 0:
            sharpe = excess.mean() / excess.std() * math.sqrt(TRADING_DAYS_PER_YEAR)

    # Trade stats (closed trades only for win rate / avg hold)
    n_closed = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(wins) / n_closed * 100 if n_closed else 0.0
    avg_hold = sum(t["hold_days"] for t in trades) / n_closed if n_closed else 0.0
    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0.0
    gross_wins = sum(t["pnl"] for t in wins)
    gross_losses = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")
    total_pnl = sum(t["pnl"] for t in all_trades)

    # Per-symbol breakdown
    sym_stats: dict[str, dict] = {}
    for t in all_trades:
        s = t["symbol"]
        if s not in sym_stats:
            sym_stats[s] = {"pnl": 0.0, "trades": 0, "wins": 0}
        sym_stats[s]["pnl"] += t["pnl"]
        sym_stats[s]["trades"] += 1
        if t["pnl"] > 0:
            sym_stats[s]["wins"] += 1

    return {
        "final_value": round(final_value, 2),
        "total_return": round(total_return * 100, 2),
        "ann_return": round(ann_return * 100, 2),
        "max_drawdown": round(max_dd * 100, 2),
        "sharpe": round(sharpe, 3),
        "win_rate": round(win_rate, 1),
        "n_closed": n_closed,
        "n_open_at_end": len(open_trades),
        "n_total": n_closed + len(open_trades),
        "avg_hold_days": round(avg_hold, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "total_pnl": round(total_pnl, 2),
        "sym_stats": sym_stats,
        "best_trade": max(trades, key=lambda t: t["pnl"]) if trades else None,
        "worst_trade": min(trades, key=lambda t: t["pnl"]) if trades else None,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(results: dict, metrics: dict) -> None:
    W = 65

    def rule(c="─"):
        print(c * W)

    def row(label, value):
        dots = W - 4 - len(label) - len(str(value))
        print(f"  {label}{'.' * max(1, dots)}{value}")

    def section(title):
        print()
        print(f"  {title}")
        rule()

    m = metrics

    print()
    rule("═")
    print(f"{'BACKTEST REPORT':^{W}}")
    print(f"{'Strategy: RSI + MACD Mean-Reversion + SMA Trend Filter':^{W}}")
    rule("═")

    section("RUN PARAMETERS")
    row("Symbols", ", ".join(results["symbols"]))
    if results["skipped"]:
        row("Skipped", ", ".join(results["skipped"]))
    row("Period", results["period"])
    row("Start date", str(results["start_date"].date()))
    row("End date", str(results["end_date"].date()))
    row("Starting capital", f"${results['starting_capital']:,.0f}")
    row("Max position size", f"{int(MAX_POS_PCT * 100)}% of portfolio per symbol")
    row("Buy signal", f"RSI < {RSI_BUY_THRESHOLD} + MACD bullish + SMA20 > SMA50")
    row("Sell signal", f"RSI > {RSI_SELL_THRESHOLD} OR MACD bearish crossover")

    section("PORTFOLIO SUMMARY")
    row("Final portfolio value", f"${m['final_value']:>13,.2f}")
    row("Total P&L", f"${m['total_pnl']:>+13,.2f}")
    row("Total return", f"{m['total_return']:>+.2f}%")
    row("Annualised return (CAGR)", f"{m['ann_return']:>+.2f}%")
    row("Max drawdown", f"{m['max_drawdown']:.2f}%")
    row("Sharpe ratio (annualised)", f"{m['sharpe']:.3f}")

    section("TRADE STATISTICS")
    row("Total positions opened", str(m["n_total"]))
    row("  Closed (hit exit signal)", str(m["n_closed"]))
    row("  Open at period end", str(m["n_open_at_end"]))
    row("Win rate (closed trades)", f"{m['win_rate']:.1f}%")
    row("Profit factor", f"{m['profit_factor']:.2f}x")
    row("Avg hold period", f"{m['avg_hold_days']:.1f} days")
    row("Avg winning trade", f"${m['avg_win']:>+,.2f}")
    row("Avg losing trade", f"${m['avg_loss']:>+,.2f}")
    if m["best_trade"]:
        t = m["best_trade"]
        row("Best trade", f"{t['symbol']}  +${t['pnl']:,.2f}  ({t['pnl_pct']:+.1f}%)")
    if m["worst_trade"]:
        t = m["worst_trade"]
        row("Worst trade", f"{t['symbol']}  ${t['pnl']:,.2f}  ({t['pnl_pct']:+.1f}%)")

    section("PER-SYMBOL BREAKDOWN")
    print(f"  {'Symbol':<8}  {'P&L':>12}  {'Trades':>7}  {'Win%':>6}")
    rule()
    for sym, s in sorted(m["sym_stats"].items(), key=lambda x: -x[1]["pnl"]):
        n = s["trades"]
        wr = f"{s['wins'] / n * 100:.0f}%" if n else "N/A"
        print(f"  {sym:<8}  ${s['pnl']:>+11,.2f}  {n:>7}  {wr:>6}")

    section("ALL CLOSED TRADES")
    if results["trades"]:
        hdr = f"  {'#':>3}  {'Sym':<6}  {'Entry':>10}  {'Exit':>10}  {'Qty':>5}  {'Buy@':>8}  {'Sell@':>8}  {'P&L':>10}  {'Days':>5}"
        print(hdr)
        rule()
        for i, t in enumerate(sorted(results["trades"], key=lambda x: x["entry_date"]), 1):
            print(
                f"  {i:>3}  {t['symbol']:<6}  "
                f"{str(t['entry_date'].date()):>10}  "
                f"{str(t['exit_date'].date()):>10}  "
                f"{t['qty']:>5}  "
                f"${t['entry_price']:>7.2f}  "
                f"${t['exit_price']:>7.2f}  "
                f"${t['pnl']:>+9,.2f}  "
                f"{t['hold_days']:>5}"
            )
    else:
        print("  No closed trades in this period.")

    if results["open_trades"]:
        section("OPEN POSITIONS AT PERIOD END  (marked to last price)")
        for t in results["open_trades"]:
            print(
                f"  {t['symbol']:<6}  entered {str(t['entry_date'].date())}  "
                f"{t['qty']} shares @ ${t['entry_price']:.2f}  "
                f"last ${t['exit_price']:.2f}  "
                f"unrealised ${t['pnl']:>+,.2f}"
            )

    section("HOW TO READ THESE RESULTS")
    print(f"""
  RETURN BENCHMARKS (S&P 500 long-run avg ~10-11% annualised):
    > 15% annualised  — Strong. Beats buy-and-hold by a clear margin.
    10-15%            — Competitive with a simple index fund.
    5-10%             — Underwhelming. Not worth the operational complexity.
    < 5%              — Poor. Just buy SPY.

  RISK METRICS:
    Max drawdown < 20%    — Acceptable for an active strategy.
    Max drawdown > 30%    — High risk. Psychologically very hard to hold live.
    Sharpe >= 1.0         — Good risk-adjusted return. Deploy with care.
    Sharpe >= 1.5         — Excellent. High confidence to go live.
    Sharpe < 0.5          — Poor. Randomness explains the result better than skill.

  TRADE QUALITY:
    Profit factor > 1.5   — $1.50 gained per $1.00 lost. Healthy edge.
    Profit factor < 1.0   — Losing money on average. Do NOT deploy.
    Win rate > 50%        — More wins than losses (helpful psychologically).
    Avg hold 5-30 days    — Consistent with swing-trading intent.

  MINIMUM BARS TO TRUST (statistical validity):
    >= 30 closed trades   — Bare minimum for any confidence.
    >= 50 closed trades   — Start to trust the metrics.
    < 20 closed trades    — Results are noise. Run --period 5y instead.

  RED FLAGS — do NOT give the bot a live account if ANY of these are true:
    Profit factor < 1.0            (losing strategy)
    < 20 closed trades             (no statistical basis)
    1-2 trades = > 80% of profit   (luck, not edge)
    Max drawdown > total return    (one bad month erases all gains)
    Sharpe < 0.3                   (random noise)

  SURVIVORSHIP BIAS NOTE:
    This watchlist (AAPL, MSFT, GOOGL, AMZN, NVDA) are today's largest winners.
    Live trading on a broader universe will perform worse.
    Treat these numbers as an optimistic upper bound, not a realistic forecast.
    To sanity-check: also run --period 1y to see recent-only performance.
""")
    rule("═")
    print(f"{'Report generated: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^{W}}")
    rule("═")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    symbols = [s.upper() for s in args.symbols]
    results = simulate(symbols, args.period, args.capital)
    metrics = compute_metrics(results)
    print_report(results, metrics)


if __name__ == "__main__":
    main()
