"""
🦇 BATMAN AI TRADING SYSTEM — SENTIMENT ENGINE
=================================================
Proxy sentiment from technicals (always available for walk-forward)
+ Live news API sentiment (Alpha Vantage / NewsAPI) for live trading.
"""

import os
import logging
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] 🦇 %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BatmanSentiment")


from data_pipeline import load_config  # one loader: config file + env-var secrets


# ═══════════════════════════════════════════════════════════
# PROXY SENTIMENT (Computed from technicals — always available)
# ═══════════════════════════════════════════════════════════

def compute_proxy_sentiment(df):
    """
    Compute proxy sentiment features from price/volume data.
    Always available for historical walk-forward (no API needed).

    Features added:
        Sentiment_RSI:       RSI-based fear/greed [-1, 1]
        Vol_Anomaly:         Volume anomaly score
        Momentum_Sentiment:  Price momentum sentiment
        Metal_Sentiment:     Gold+Silver momentum as sector sentiment
        Vol_Regime:          Volatility regime (negative = fearful)
        Proxy_Sentiment:     Weighted composite of above
    """
    df = df.copy()

    # 1. RSI-based Fear/Greed
    if "RSI_14" in df.columns:
        df["Sentiment_RSI"] = (df["RSI_14"] - 50) / 50  # [-1, 1]
    else:
        df["Sentiment_RSI"] = 0.0

    # 2. Volume Anomaly (high volume = conviction)
    if "Volume_Delta" in df.columns:
        df["Vol_Anomaly"] = (df["Volume_Delta"] - 1.0).clip(-2, 2)
    else:
        df["Vol_Anomaly"] = 0.0

    # 3. Momentum Sentiment (5d return normalized)
    if "Return_5d" in df.columns:
        df["Momentum_Sentiment"] = df["Return_5d"].clip(-0.15, 0.15) / 0.15
    else:
        df["Momentum_Sentiment"] = 0.0

    # 4. Metal Sentiment (combined gold+silver momentum — bullish for miners)
    gc_pct = df.get("GC=F_Pct_5d", pd.Series(0, index=df.index))
    si_pct = df.get("SI=F_Pct_5d", pd.Series(0, index=df.index))
    raw_metal = (gc_pct.fillna(0) + si_pct.fillna(0)) / 2
    df["Metal_Sentiment"] = raw_metal.clip(-0.1, 0.1) / 0.1

    # 5. Volatility Regime (high ATR/price ratio = fear = negative)
    if "ATR_20" in df.columns and "Close" in df.columns:
        atr_ratio = df["ATR_20"] / df["Close"]
        rolling_mean = atr_ratio.rolling(60, min_periods=10).mean()
        rolling_std = atr_ratio.rolling(60, min_periods=10).std().replace(0, 1)
        df["Vol_Regime"] = (-(atr_ratio - rolling_mean) / rolling_std).clip(-2, 2).fillna(0)
    else:
        df["Vol_Regime"] = 0.0

    # 6. Composite Proxy Sentiment
    df["Proxy_Sentiment"] = (
        df["Sentiment_RSI"] * 0.25
        + df["Momentum_Sentiment"] * 0.30
        + df["Metal_Sentiment"] * 0.25
        + df["Vol_Regime"] * 0.10
        + df["Vol_Anomaly"] * 0.10
    )

    logger.info(
        f"✅ Proxy sentiment computed. "
        f"Range: [{df['Proxy_Sentiment'].min():.2f}, {df['Proxy_Sentiment'].max():.2f}]"
    )
    return df


# ═══════════════════════════════════════════════════════════
# LIVE NEWS SENTIMENT (API-based — for live trading only)
# ═══════════════════════════════════════════════════════════

def fetch_alpha_vantage_sentiment(ticker="USAS", api_key=None):
    """Fetch news sentiment from Alpha Vantage News & Sentiment API."""
    if not api_key:
        return None
    try:
        import requests
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": api_key,
            "limit": 50,
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if "feed" not in data:
            logger.warning(f"Alpha Vantage: No feed data for {ticker}")
            return None

        scores = [float(a.get("overall_sentiment_score", 0)) for a in data["feed"]]
        if not scores:
            return None

        avg = np.mean(scores)
        logger.info(f"📰 Alpha Vantage: {ticker} sentiment={avg:.3f} ({len(scores)} articles)")
        return {"sentiment_score": avg, "article_count": len(scores), "source": "alpha_vantage"}

    except Exception as e:
        logger.warning(f"Alpha Vantage API error: {e}")
        return None


def fetch_newsapi_sentiment(query="USAS mining", api_key=None):
    """Fetch news from NewsAPI.org and score with VADER."""
    if not api_key:
        return None
    try:
        import requests
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query, "apiKey": api_key,
            "language": "en", "sortBy": "publishedAt", "pageSize": 50,
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data.get("status") != "ok" or not data.get("articles"):
            return None

        scores = []
        for article in data["articles"]:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            if text.strip():
                scores.append(analyzer.polarity_scores(text)["compound"])

        if not scores:
            return None

        avg = np.mean(scores)
        logger.info(f"📰 NewsAPI: '{query}' sentiment={avg:.3f} ({len(scores)} articles)")
        return {"sentiment_score": avg, "article_count": len(scores), "source": "newsapi"}

    except ImportError:
        logger.warning("vaderSentiment not installed. pip install vaderSentiment")
        return None
    except Exception as e:
        logger.warning(f"NewsAPI error: {e}")
        return None


def get_live_sentiment(ticker="USAS", config=None):
    """
    Get real-time news sentiment. Tries Alpha Vantage → NewsAPI → fallback to proxy.
    """
    if config is None:
        config = load_config()

    # Try Alpha Vantage
    av_key = config.get("alpha_vantage_key", "")
    if av_key:
        result = fetch_alpha_vantage_sentiment(ticker, av_key)
        if result:
            return result

    # Try NewsAPI
    na_key = config.get("newsapi_key", "")
    if na_key:
        result = fetch_newsapi_sentiment(f"{ticker} mining gold silver", na_key)
        if result:
            return result

    logger.info("📰 No news API configured. Using proxy sentiment only.")
    return {"sentiment_score": 0.0, "article_count": 0, "source": "none"}
