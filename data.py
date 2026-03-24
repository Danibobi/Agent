"""Market data and technical indicators via yfinance."""

import yfinance as yf
import pandas as pd
from datetime import datetime


def get_market_data(symbol: str, period: str = "1mo") -> dict:
    """Fetch OHLCV data and compute technical indicators for a symbol."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)

    if hist.empty:
        return {"error": f"No data found for {symbol}"}

    # Compute technical indicators
    close = hist["Close"]
    hist["SMA_20"] = close.rolling(20).mean()
    hist["SMA_50"] = close.rolling(50).mean()
    hist["RSI"] = _compute_rsi(close)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    hist["MACD"] = ema12 - ema26
    hist["MACD_signal"] = hist["MACD"].ewm(span=9, adjust=False).mean()

    latest = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) >= 2 else latest

    return {
        "symbol": symbol,
        "current_price": round(float(latest["Close"]), 2),
        "open": round(float(latest["Open"]), 2),
        "high": round(float(latest["High"]), 2),
        "low": round(float(latest["Low"]), 2),
        "volume": int(latest["Volume"]),
        "prev_close": round(float(prev["Close"]), 2),
        "change_pct": round((float(latest["Close"]) - float(prev["Close"])) / float(prev["Close"]) * 100, 2),
        "sma_20": round(float(latest["SMA_20"]), 2) if pd.notna(latest["SMA_20"]) else None,
        "sma_50": round(float(latest["SMA_50"]), 2) if pd.notna(latest["SMA_50"]) else None,
        "rsi": round(float(latest["RSI"]), 2) if pd.notna(latest["RSI"]) else None,
        "macd": round(float(latest["MACD"]), 4) if pd.notna(latest["MACD"]) else None,
        "macd_signal": round(float(latest["MACD_signal"]), 4) if pd.notna(latest["MACD_signal"]) else None,
        "period": period,
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def get_news(symbol: str) -> dict:
    """Fetch recent news headlines for a symbol via yfinance."""
    ticker = yf.Ticker(symbol)
    try:
        news_items = ticker.news or []
    except Exception:
        news_items = []

    headlines = []
    for item in news_items[:10]:
        content = item.get("content", {})
        title = content.get("title", item.get("title", ""))
        pub_date = content.get("pubDate", item.get("providerPublishTime", ""))
        if title:
            headlines.append({"title": title, "published": str(pub_date)})

    return {"symbol": symbol, "headlines": headlines}


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))
