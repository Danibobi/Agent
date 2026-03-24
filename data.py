"""Market data and technical indicators via yfinance."""

import yfinance as yf
import pandas as pd
from datetime import datetime

_BULLISH_KEYWORDS = [
    "beat", "beats", "record", "surge", "surges", "rally", "rallies",
    "upgrade", "upgraded", "outperform", "buy", "strong", "strength",
    "growth", "grew", "profit", "profitable", "revenue beat", "earnings beat",
    "raised guidance", "raises guidance", "dividend increase", "buyback",
    "acquisition", "partnership", "breakthrough", "launch", "launches",
    "above expectations", "better than expected", "all-time high",
]

_BEARISH_KEYWORDS = [
    "miss", "misses", "missed", "decline", "declines", "fell", "fall",
    "downgrade", "downgraded", "underperform", "sell", "weak", "weakness",
    "loss", "losses", "revenue miss", "earnings miss", "lowered guidance",
    "lowers guidance", "lawsuit", "investigation", "recall",
    "below expectations", "worse than expected", "layoffs", "layoff",
    "bankruptcy", "debt", "default", "warning", "concern",
]


def _score_sentiment(title: str) -> tuple[str, float]:
    """Score a headline using keyword matching. Returns (label, score) where
    label is 'bullish'/'bearish'/'neutral' and score is in [-1.0, +1.0]."""
    lower = title.lower()
    bullish_hits = sum(1 for kw in _BULLISH_KEYWORDS if kw in lower)
    bearish_hits = sum(1 for kw in _BEARISH_KEYWORDS if kw in lower)
    raw = bullish_hits - bearish_hits
    score = max(-1.0, min(1.0, raw / 3.0))
    if score > 0.1:
        label = "bullish"
    elif score < -0.1:
        label = "bearish"
    else:
        label = "neutral"
    return label, round(score, 3)


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
    """Fetch recent news headlines for a symbol with keyword-based sentiment scoring."""
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
            sentiment_label, sentiment_score = _score_sentiment(title)
            headlines.append({
                "title": title,
                "published": str(pub_date),
                "sentiment": sentiment_label,
                "sentiment_score": sentiment_score,
            })

    sentiments = [h["sentiment"] for h in headlines]
    overall_score = round(sum(h["sentiment_score"] for h in headlines) / len(headlines), 3) if headlines else 0.0
    overall_label = "bullish" if overall_score > 0.1 else ("bearish" if overall_score < -0.1 else "neutral")

    return {
        "symbol": symbol,
        "headlines": headlines,
        "sentiment_summary": {
            "bullish": sentiments.count("bullish"),
            "bearish": sentiments.count("bearish"),
            "neutral": sentiments.count("neutral"),
            "overall_score": overall_score,
            "overall_label": overall_label,
        },
    }


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))
