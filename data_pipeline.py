"""
🦇 BATMAN AI TRADING SYSTEM — DATA PIPELINE
==============================================
Fetches OHLCV data via yfinance, engineers deadly technical indicators,
scales features, creates sequences for LSTM, and splits chronologically.
No leakage. No mercy.
"""

import os
import sys
import time
import logging
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import ta as ta_lib
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import yaml

# ─── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] 🦇 %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BatmanDataPipeline")


# Secrets come from environment variables, never from files in the repo.
_ENV_OVERRIDES = {
    "alpha_vantage_key": "ALPHA_VANTAGE_KEY",
    "newsapi_key": "NEWSAPI_KEY",
    "email_from": "BATMAN_EMAIL_FROM",
    "email_password": "BATMAN_EMAIL_PASSWORD",
    "email_to": "BATMAN_EMAIL_TO",
}


def load_config(path="config.yaml"):
    """Load settings from config.yaml (falls back to config.example.yaml),
    then overlay secrets from environment variables."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, path)
    if not os.path.exists(config_path):
        config_path = os.path.join(base_dir, "config.example.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    for key, env in _ENV_OVERRIDES.items():
        if os.environ.get(env):
            config[key] = os.environ[env]
    return config


def fetch_data(ticker, start=None, end=None, max_retries=3):
    """
    Fetch daily OHLCV data for main ticker + Aux metals (Gold/Silver).
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    config = load_config()
    if start is None:
        start = config.get("data_fetch_start", "2025-01-01")

    # Fetch Main Ticker
    df_main = _fetch_single(ticker, start, end, max_retries)
    
    # Fetch Aux Tickers (Metals)
    aux_tickers = config.get("aux_tickers", ["GC=F", "SI=F"])
    
    for aux in aux_tickers:
        try:
            df_aux = _fetch_single(aux, start, end, max_retries)
            # Merge Close price as feature, e.g., 'GC=F_Close'
            # We assume same trading days, or ffill
            series = df_aux["Close"].rename(f"{aux}_Close")
            df_main = df_main.join(series, how="left")
        except Exception as e:
            logger.warning(f"Failed to fetch aux ticker {aux}: {e}")
            
    df_main = df_main.ffill() # Fill missing metal data
    return df_main


def _fetch_single(ticker, start, end, max_retries):
    """Helper to fetch single ticker."""
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Fetching {ticker} data from {start} to {end}...")
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty: raise ValueError(f"No data for {ticker}")
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # Basic validation
            if "Close" not in df.columns: raise ValueError("Missing Close")
             # Keep OHLCV for main ticker, just Close for aux usually, but here we return full first
            return df[["Open", "High", "Low", "Close", "Volume"]].copy() if "Volume" in df.columns else df[["Open", "High", "Low", "Close"]].copy()

        except Exception as e:
            logger.warning(f"Attempt {attempt} failed for {ticker}: {e}")
            time.sleep(2**attempt)
            if attempt == max_retries: raise


def engineer_features(df, config=None):
    """
    Engineer technical indicators from OHLCV data.

    Features: SMA50, SMA200, RSI14, MACD(12,26,9), BBands(20,2), Volume % Change
    All computed via the `ta` library (technical analysis).

    Args:
        df: DataFrame with OHLCV columns
        config: Optional config dict for indicator params

    Returns:
        DataFrame with original + engineered feature columns
    """
    if config is None:
        config = load_config()

    df = df.copy()

    logger.info("Engineering features...")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Simple Moving Averages
    sma_short = config.get("sma_short", 50)
    sma_long = config.get("sma_long", 200)
    df["SMA_50"] = ta_lib.trend.SMAIndicator(close=close, window=sma_short).sma_indicator()
    df["SMA_200"] = ta_lib.trend.SMAIndicator(close=close, window=sma_long).sma_indicator()

    # RSI
    rsi_period = config.get("rsi_period", 14)
    df["RSI_14"] = ta_lib.momentum.RSIIndicator(close=close, window=rsi_period).rsi()
    df["RSI_7"] = ta_lib.momentum.RSIIndicator(close=close, window=7).rsi()

    # Money Flow Index (MFI)
    df["MFI_14"] = ta_lib.volume.MFIIndicator(high=high, low=low, close=close, volume=df["Volume"], window=14).money_flow_index()

    # Rate of Change (ROC)
    df["ROC_10"] = ta_lib.momentum.ROCIndicator(close=close, window=10).roc()

    # Rolling Z-Score (Mean Reversion)
    roll_mean_20 = close.rolling(20).mean()
    df["Z_Score_20"] = (close - roll_mean_20) / (df["Close"].rolling(20).std() + 1e-6)

    # MACD
    macd_fast = config.get("macd_fast", 12)
    macd_slow = config.get("macd_slow", 26)
    macd_signal = config.get("macd_signal", 9)
    macd = ta_lib.trend.MACD(close=close, window_slow=macd_slow,
                              window_fast=macd_fast, window_sign=macd_signal)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # Bollinger Bands
    bb_period = config.get("bb_period", 20)
    bb_std = config.get("bb_std", 2)
    bbands = ta_lib.volatility.BollingerBands(close=close, window=bb_period, window_dev=bb_std)
    df["BB_Lower"] = bbands.bollinger_lband()
    df["BB_Mid"] = bbands.bollinger_mavg()
    df["BB_Upper"] = bbands.bollinger_hband()
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]  # Band width ratio

    # Volume Features
    df["Volume_Pct"] = df["Volume"].pct_change() * 100
    df["Volume_Delta"] = df["Volume"] / df["Volume"].rolling(20).mean()  # Volume ratio vs 20d avg

    # Momentum (Lethal Predictors)
    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_5d"] = df["Close"].pct_change(5)

    # Volatility
    df["Vol_20"] = df["Close"].rolling(20).std()

    # ATR (20-day Average True Range)
    df["ATR_20"] = ta_lib.volatility.AverageTrueRange(high=high, low=low, close=close, window=20).average_true_range()

    # ADX (Trend Strength)
    adx_ind = ta_lib.trend.ADXIndicator(high=high, low=low, close=close, window=14)
    df["ADX_14"] = adx_ind.adx()
    df["ADX_Pos"] = adx_ind.adx_pos()
    df["ADX_Neg"] = adx_ind.adx_neg()

    # CCI (Commodity Channel Index)
    df["CCI_14"] = ta_lib.trend.CCIIndicator(high=high, low=low, close=close, window=14).cci()

    # Williams %R
    df["Williams_R"] = ta_lib.momentum.WilliamsRIndicator(high=high, low=low, close=close, lbp=14).williams_r()

    # Stochastic Oscillator
    stoch = ta_lib.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    # Autoregressive Lags (Price & Volume Action History)
    for lag in [1, 2, 3]:
        df[f"Lag_Return_{lag}"] = df["Return_1d"].shift(lag)
    
    for lag in [1, 2]:
        df[f"Lag_Vol_{lag}"] = df["Volume_Delta"].shift(lag)

    # Metal Features (SMA Delta)
    metal_period = config.get("metal_sma_period", 5)
    aux_tickers = config.get("aux_tickers", ["GC=F", "SI=F"])
    
    for aux in aux_tickers:
        col = f"{aux}_Close"
        if col in df.columns:
            sma_col = f"{aux}_SMA_{metal_period}"
            df[sma_col] = df[col].rolling(metal_period).mean()
            df[f"{aux}_SMA_Delta"] = df[sma_col].diff()
            df[f"{aux}_Trend_Up"] = (df[f"{aux}_SMA_Delta"] > 0).astype(int)
            # 5-day % change in metal close
            df[f"{aux}_Pct_5d"] = df[col].pct_change(5)

    # Proxy Sentiment Features (computed from technicals — always available)
    from sentiment import compute_proxy_sentiment
    df = compute_proxy_sentiment(df)

    # BINARY TARGET: 1 if Close(t+1) > Close(t)
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    
    logger.info(f"✅ Engineered features + Sentiment + Binary Target. Columns: {len(df.columns)}")
    return df


def handle_nans(df):
    """
    Handle NaN values: forward-fill then drop leading NaNs.

    Args:
        df: DataFrame with potential NaN values

    Returns:
        Clean DataFrame with no NaN values
    """
    before = len(df)
    df = df.ffill()
    df = df.dropna()
    after = len(df)
    dropped = before - after
    if dropped > 0:
        logger.info(f"Dropped {dropped} leading NaN rows (from indicator warm-up)")
    return df


def get_feature_columns():
    """Return the list of feature columns used for model input (45 total)."""
    return [
        "Close", "SMA_50", "SMA_200", "RSI_14",
        "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Lower", "BB_Mid", "BB_Upper", "BB_Width",
        "Volume_Pct", "Volume_Delta",
        "Return_1d", "Return_5d",
        "Vol_20", "ATR_20",
        "GC=F_Close", "GC=F_SMA_Delta", "GC=F_Pct_5d",
        "SI=F_Close", "SI=F_SMA_Delta", "SI=F_Pct_5d",
        # New Phase 2 Features (10+)
        "ADX_14", "ADX_Pos", "ADX_Neg",
        "CCI_14", "Williams_R",
        "Stoch_K", "Stoch_D",
        "Lag_Return_1", "Lag_Return_2", "Lag_Return_3",
        "Lag_Vol_1", "Lag_Vol_2",
        # New Alpha Features
        "RSI_7", "MFI_14", "ROC_10", "Z_Score_20",
        # Proxy Sentiment (6 features)
        "Sentiment_RSI", "Vol_Anomaly", "Momentum_Sentiment",
        "Metal_Sentiment", "Vol_Regime", "Proxy_Sentiment",
    ]


def scale_data(df, feature_cols=None):
    """
    Scale features with StandardScaler and target (Close) with MinMaxScaler(0,1).

    Args:
        df: DataFrame with engineered features
        feature_cols: List of feature column names

    Returns:
        Tuple: (scaled_features_array, scaled_target_array, feature_scaler, target_scaler)
    """
    if feature_cols is None:
        # Dynamically select numeric columns that are not 'Target' or 'Trend_Up' flags
        # Or better, iterate valid known features
        known = get_feature_columns()
        feature_cols = [c for c in known if c in df.columns]

    feature_scaler = StandardScaler()
    # No scaler for Binary Target
    target_scaler = None 

    features_data = df[feature_cols].values
    target_data = df["Target"].values.reshape(-1, 1) # This is 0/1

    scaled_features = feature_scaler.fit_transform(features_data)
    scaled_target = target_data # No scaling for binary
    
    # We still return target_scaler as None to keep signature similar or handle downstream
    logger.info(f"✅ Scaled {len(feature_cols)} features. Target is Binary (No Scale).")
    return scaled_features, scaled_target, feature_scaler, target_scaler


def create_sequences(features, target, lookback=60):
    """
    Create sliding window sequences for LSTM training.

    Args:
        features: Scaled feature array (n_samples, n_features)
        target: Scaled target array (n_samples, 1)
        lookback: Number of timesteps to look back

    Returns:
        Tuple: (X, y) where X is (n_sequences, lookback, n_features) and y is (n_sequences,)
    """
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback:i])
        y.append(target[i, 0])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    logger.info(f"✅ Created {len(X)} sequences (lookback={lookback}, features={X.shape[2]})")
    return X, y


def split_data(X, y, train_pct=0.80, val_pct=0.10):
    """
    Chronological split — no shuffle, no leakage.

    Args:
        X: Input sequences
        y: Target values
        train_pct: Training set proportion
        val_pct: Validation set proportion

    Returns:
        Tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    n = len(X)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    logger.info(f"✅ Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def run_pipeline(ticker=None, config=None):
    """
    Full data pipeline orchestrator.

    Fetch → Engineer → Clean → Scale → Sequence → Split → Save

    Args:
        ticker: Ticker symbol override
        config: Config dict override

    Returns:
        Dict with all pipeline outputs
    """
    if config is None:
        config = load_config()
    if ticker is None:
        ticker = config.get("ticker", "AAPL")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    logger.info(f"{'='*60}")
    logger.info(f"🦇 BATMAN DATA PIPELINE — {ticker}")
    logger.info(f"{'='*60}")

    # 1. Fetch
    df = fetch_data(ticker, start=config.get("data_start", "2010-01-01"))

    # 2. Engineer features
    df = engineer_features(df, config)

    # 3. Handle NaNs
    df = handle_nans(df)

    # 4. Save raw data
    data_path = os.path.join(base_dir, config.get("data_csv_path", "data.csv"))
    df.to_csv(data_path)
    logger.info(f"💾 Saved data to {data_path} ({len(df)} rows)")

    # 5. Scale
    feature_cols = get_feature_columns()
    scaled_features, scaled_target, feature_scaler, target_scaler = scale_data(df, feature_cols)

    # 6. Save scalers
    scalers_path = os.path.join(base_dir, config.get("scalers_path", "scalers.pkl"))
    with open(scalers_path, "wb") as f:
        pickle.dump({
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "feature_columns": feature_cols,
        }, f)
    logger.info(f"💾 Saved scalers to {scalers_path}")

    # 7. Create sequences
    lookback = config.get("lookback", 60)
    X, y = create_sequences(scaled_features, scaled_target, lookback)

    # 8. Split
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X, y,
        train_pct=config.get("train_pct", 0.80),
        val_pct=config.get("val_pct", 0.10),
    )

    # 9. Save sequences
    seq_path = os.path.join(base_dir, config.get("sequences_path", "sequences.npz"))
    np.savez_compressed(
        seq_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )
    logger.info(f"💾 Saved sequences to {seq_path}")

    logger.info(f"{'='*60}")
    logger.info("🦇 DATA PIPELINE COMPLETE")
    logger.info(f"{'='*60}")

    return {
        "df": df,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_columns": feature_cols,
    }


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(ticker=ticker)
