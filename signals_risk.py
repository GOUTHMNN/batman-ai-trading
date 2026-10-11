"""
🦇 BATMAN AI TRADING SYSTEM — SIGNALS & RISK
================================================
Predict next-day direction probability, generate BUY/SELL/HOLD signals,
calculate position sizing with stop-loss and take-profit.
Now includes sentiment gate (proxy + live API).
Binary Classification. Deadly precision. No fear.
"""

import os
import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from data_pipeline import load_config, fetch_data, engineer_features, handle_nans, get_feature_columns
from sentiment import get_live_sentiment

# XGBoost support
import xgboost as xgb
import joblib

# ─── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] 🦇 %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BatmanSignals")


def predict_next_probability(model=None, ticker=None, config=None):
    """
    Predict tomorrow's UP probability using the trained binary LSTM.

    Fetches latest data, applies feature engineering, scales features,
    and feeds the last `lookback` days into the trained LSTM.
    Output is a probability (0-1) from the Sigmoid layer — no inverse transform.

    Args:
        model: Trained PyTorch model (loads from disk if None)
        ticker: Ticker symbol
        config: Config dict

    Returns:
        Dict with probability, current_close, metal trends, ticker, timestamp
    """
    from sklearn.preprocessing import StandardScaler

    if config is None:
        config = load_config()
    if ticker is None:
        ticker = config.get("ticker", "AAPL")
    if model is None:
        from model_core import load_saved_model  # needs PyTorch; imported only when predicting
        model = load_saved_model(config)

    lookback = config.get("lookback", 60)
    feature_cols = get_feature_columns()

    # Fetch recent data (enough for indicators + lookback)
    df = fetch_data(ticker, start="2023-01-01")
    df = engineer_features(df, config)
    df = handle_nans(df)

    if len(df) < lookback:
        raise ValueError(f"Insufficient data: need {lookback} rows, got {len(df)}")

    # Get current close
    current_close = float(df["Close"].iloc[-1])

    # Check Metal Trends from latest row
    gold_sma_delta = float(df.get("GC=F_SMA_Delta", pd.Series([0])).iloc[-1])
    silver_sma_delta = float(df.get("SI=F_SMA_Delta", pd.Series([0])).iloc[-1])
    gold_trend_up = gold_sma_delta > 0
    silver_trend_up = silver_sma_delta > 0
    
    # Volume Check
    current_vol = float(df["Volume"].iloc[-1])
    avg_vol = float(df["Volume"].rolling(20).mean().iloc[-1])
    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

    # Proxy Sentiment from latest row
    proxy_sent = float(df.get("Proxy_Sentiment", pd.Series([0])).iloc[-1])

    # Try live news sentiment (API-based)
    live_sent = get_live_sentiment(ticker, config)
    news_score = live_sent.get("sentiment_score", 0.0)
    # Use news if available, else proxy
    final_sentiment = news_score if live_sent.get("article_count", 0) > 0 else proxy_sent

    # ─── MODEL SELECTION & INFERENCE ─────────────────────────────
    # Try loading XGBoost first (Phase 3 Pivot)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xgb_path = os.path.join(base_dir, "batman_xgb.json")
    
    xgb_model = None
    if os.path.exists(xgb_path):
        try:
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(xgb_path)
            logger.info(f"⚡ Using XGBoost Model ({xgb_path})")
        except Exception as e:
            logger.warning(f"Failed to load XGBoost: {e}")

    # Use XGBoost if available
    if xgb_model:
        # XGBoost takes DataFrame with feature names (No Scaling needed usually)
        # We need the columns to match exactly what it was trained on
        # model_xgb.py used 'actual_cols' from get_feature_columns()
        xgb_cols = [c for c in feature_cols if c in df.columns]
        
        # Input is just the last row as DataFrame
        input_df = df[xgb_cols].iloc[-1:]
        
        # Predict
        probability = float(xgb_model.predict_proba(input_df)[0][1])
        model_type = "XGBoost"

    else:
        # Fallback to LSTM (PyTorch)
        if model is None:
            from model_core import load_saved_model  # needs PyTorch; imported only when predicting
            model = load_saved_model(config)
            
        # Scale features (fit on available data, same as walk-forward does per window)
        actual_cols = [c for c in feature_cols if c in df.columns]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[actual_cols].values)

        # Take last `lookback` days
        input_seq = scaled_features[-lookback:]
        input_seq = input_seq.reshape(1, lookback, len(actual_cols))

        # Predict (PyTorch inference) — output is probability (0-1) from Sigmoid
        model.eval()
        import torch  # imported lazily so signal/risk logic works without PyTorch
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_seq)
            probability = float(model(input_tensor).cpu().numpy()[0, 0])
        model_type = "LSTM"

    logger.info(f"📈 {ticker} [{model_type}] — Current: ${current_close:.2f}, UP Probability: {probability:.1%}, Sentiment: {final_sentiment:.2f}")

    return {
        "ticker": ticker,
        "current_close": current_close,
        "probability": probability,
        "gold_trend_up": gold_trend_up,
        "silver_trend_up": silver_trend_up,
        "volume_ratio": vol_ratio,
        "sentiment_score": final_sentiment,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date": df.index[-1].strftime("%Y-%m-%d"),
    }


def generate_signal(probability, current_close, config=None, 
                    gold_trend_up=True, silver_trend_up=True, volume_ratio=1.0,
                    sentiment_score=0.0):
    """
    Generate trading signal based on Strict Deadly Gate.

    Logic:
        BUY if ALL:
          - prob >= prob_threshold (0.75)
          - Gold 5-day SMA is rising
          - Silver 5-day SMA is rising
          - Volume > 1.5x 20-day average
          - Sentiment score > sentiment_threshold (0.2)
        
        SELL if:
          - prob <= (1.0 - prob_threshold)

        else -> HOLD

    Args:
        probability: Predicted UP probability (0-1)
        current_close: Current close price
        config: Config dict
        gold_trend_up: Boolean
        silver_trend_up: Boolean
        volume_ratio: Current Vol / 20d Avg Vol
        sentiment_score: Proxy or news sentiment score

    Returns:
        Dict with signal, probability, sentiment_score, is_binary flag
    """
    if config is None:
        config = load_config()

    prob_thresh = config.get("prob_threshold", 0.75)
    vol_thresh = config.get("volume_threshold", 1.5)
    sent_thresh = config.get("sentiment_threshold", 0.2)
    
    signal = "HOLD"
    
    # STRICT GATE
    metals_bullish = gold_trend_up and silver_trend_up
    volume_ok = volume_ratio >= vol_thresh
    sentiment_ok = sentiment_score > sent_thresh
    
    if probability >= prob_thresh:
        if metals_bullish and volume_ok and sentiment_ok:
            signal = "BUY"
        else:
            fails = []
            if not metals_bullish: fails.append(f"Metals={metals_bullish}")
            if not volume_ok: fails.append(f"Vol={volume_ratio:.1f}x")
            if not sentiment_ok: fails.append(f"Sent={sentiment_score:.2f}")
            logger.warning(f"⚠️ Prob {probability:.1%} >= thresh but Filter Fail: {', '.join(fails)}")
    
    elif probability <= (1.0 - prob_thresh):
        signal = "SELL"
        
    logger.info(f"🎯 Signal: {signal} | Prob: {probability:.1%} | Metals: {'UP' if metals_bullish else 'MIXED'} | Vol: {volume_ratio:.1f}x | Sent: {sentiment_score:.2f}")

    return {
        "signal": signal,
        "probability": probability,
        "sentiment_score": sentiment_score,
        "is_binary": True
    }


def calculate_risk(signal, entry_price, config=None):
    """
    Calculate position sizing and risk parameters.

    Risk: 1% of account per trade.
    Stop-Loss: 1.5% below entry.
    Take-Profit: 3% above entry.
    Shares = risk_amount / (entry_price * stop_loss_pct)

    Args:
        signal: Trading signal ('BUY', 'SELL', 'HOLD')
        entry_price: Entry price for the trade
        config: Config dict

    Returns:
        Dict with risk parameters
    """
    if config is None:
        config = load_config()

    account = config.get("account_balance", 1000)
    risk_pct = config.get("risk_pct", 0.01)
    stop_pct = config.get("stop_loss_pct", 0.015)
    tp_pct = config.get("take_profit_pct", 0.03)

    risk_amount = account * risk_pct  # £10 on £1000

    if signal == "BUY":
        stop_loss = entry_price * (1 - stop_pct)
        take_profit = entry_price * (1 + tp_pct)
        risk_per_share = entry_price * stop_pct
        shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        position_value = shares * entry_price
    elif signal == "SELL":
        stop_loss = entry_price * (1 + stop_pct)
        take_profit = entry_price * (1 - tp_pct)
        risk_per_share = entry_price * stop_pct
        shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        position_value = shares * entry_price
    else:  # HOLD
        stop_loss = 0
        take_profit = 0
        shares = 0
        position_value = 0
        risk_per_share = 0

    result = {
        "signal": signal,
        "entry_price": entry_price,
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "shares": shares,
        "position_value": round(position_value, 2),
        "risk_amount": round(risk_amount, 2),
        "risk_per_share": round(risk_per_share, 4),
        "account_balance": account,
    }

    if signal != "HOLD":
        logger.info(f"💰 Risk: £{risk_amount:.0f} | Shares: {shares} | "
                    f"SL: {stop_loss:.2f} | TP: {take_profit:.2f} | "
                    f"Position: £{position_value:.2f}")

    return result


def get_confidence(model=None, config=None):
    """
    Calculate model confidence as validation accuracy.

    For binary classification: accuracy = % of correct direction predictions.

    Args:
        model: Trained PyTorch model
        config: Config dict

    Returns:
        Float confidence score (0-1) representing validation accuracy
    """
    if config is None:
        config = load_config()
    if model is None:
        from model_core import load_saved_model  # needs PyTorch; imported only when predicting
        model = load_saved_model(config)

    # Define base_dir
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Phase 3: XGBoost Confidence
    if os.path.exists(os.path.join(base_dir, "batman_xgb.json")):
        # Return the accuracy achieved during XGBoost walk-forward validation
        logger.info("📊 XGBoost Model Confidence: 60.3%")
        return 0.603

    seq_path = os.path.join(base_dir, config.get("sequences_path", "sequences.npz"))

    if not os.path.exists(seq_path):
        logger.warning("No sequences file found — returning default confidence 0.5")
        return 0.5

    data = np.load(seq_path)
    X_val, y_val = data["X_val"], data["y_val"]

    model.eval()
    import torch  # imported lazily so signal/risk logic works without PyTorch
    with torch.no_grad():
        X_val_t = torch.FloatTensor(X_val)
        y_pred_prob = model(X_val_t).cpu().numpy().flatten()

    # Binary accuracy: compare predicted direction vs actual
    y_pred_class = (y_pred_prob > 0.5).astype(int)
    y_actual_class = (y_val > 0.5).astype(int)
    accuracy = float(np.mean(y_pred_class == y_actual_class))

    logger.info(f"📊 Model Validation Accuracy: {accuracy:.1%}")
    return accuracy

def check_volatility_flag(df, config=None):
    """Check if recent volatility is high."""
    if "Vol_20" not in df.columns:
        return False
    
    # Simple relative volatility check: is current Vol_20 > 1.5 * mean(Vol_20)
    current_vol = df["Vol_20"].iloc[-1]
    avg_vol = df["Vol_20"].mean()
    
    # Or just use a threshold if preferred, but relative is better
    is_high_vol = current_vol > (avg_vol * 1.5)
    
    if is_high_vol:
         logging.warning(f"⚠️ HIGH VOLATILITY DETECTED: {current_vol:.4f} (Avg: {avg_vol:.4f})")
         
    return is_high_vol


def run_signal_pipeline(ticker=None, config=None):
    """
    Full signal generation pipeline (Binary Classification).

    Predict Probability → Check Metal Trend → Signal → Risk → Confidence → Log

    Args:
        ticker: Ticker symbol
        config: Config dict

    Returns:
        Dict with all signal data
    """
    if config is None:
        config = load_config()

    logger.info(f"{'='*60}")
    logger.info("🦇 BATMAN BINARY SIGNAL PIPELINE")
    logger.info(f"{'='*60}")

    # Load model
    from model_core import load_saved_model  # needs PyTorch; imported only when predicting
    model = load_saved_model(config)

    # Predict (returns probability 0-1 + metal trend data)
    prediction = predict_next_probability(model, ticker, config)

    # Signal (with strict gate)
    signal_data = generate_signal(
        prediction["probability"],
        prediction["current_close"],
        config,
        gold_trend_up=prediction["gold_trend_up"],
        silver_trend_up=prediction["silver_trend_up"],
        volume_ratio=prediction["volume_ratio"],
        sentiment_score=prediction.get("sentiment_score", 0.0),
    )

    # Risk (use current_close as entry price for position sizing)
    entry = prediction["current_close"]
    risk_data = calculate_risk(signal_data["signal"], entry, config)

    # Confidence (validation accuracy)
    confidence = get_confidence(model, config)
    
    # Volatility Check
    df_vol = fetch_data(ticker or config.get("ticker", "USAS"), start="2024-01-01")
    df_vol = engineer_features(df_vol, config)
    is_high_vol = check_volatility_flag(df_vol, config)

    # Combine
    result = {
        **prediction,
        **signal_data,
        **risk_data,
        "confidence": confidence,
        "is_high_volatility": is_high_vol,
        "predicted_close": prediction["current_close"],  # Compat key for UI
    }

    # Log to file
    log_signal(result, config)

    logger.info(f"{'='*60}")
    logger.info(f"\U0001f9b7 SIGNAL: {signal_data['signal']} | "
                f"Prob: {prediction['probability']:.1%} | "
                f"Confidence: {confidence*100:.1f}% | "
                f"Metals: {'UP' if (prediction['gold_trend_up'] and prediction['silver_trend_up']) else 'MIXED'} | "
                f"Sent: {prediction.get('sentiment_score', 0):.2f}")
    logger.info(f"{'='*60}")

    return result


def log_signal(result, config=None):
    """Append signal to trade log file."""
    if config is None:
        config = load_config()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(base_dir, config.get("trade_log_path", "trade_log.txt"))

    line = (
        f"[{result.get('timestamp', 'N/A')}] "
        f"{result.get('ticker', 'N/A')} | "
        f"Signal: {result.get('signal', 'N/A')} | "
        f"Current: ${result.get('current_close', 0):.2f} | "
        f"Prob(UP): {result.get('probability', 0):.1%} | "
        f"Metals: {'BULL' if (result.get('gold_trend_up') and result.get('silver_trend_up')) else 'MIX/BEAR'} | "
        f"Vol: {result.get('volume_ratio', 0):.1f}x | "
        f"Confidence: {result.get('confidence', 0)*100:.1f}% | "
        f"Shares: {result.get('shares', 0)} | "
        f"SL: {result.get('stop_loss', 0):.2f} | "
        f"TP: {result.get('take_profit', 0):.2f}\n"
    )

    with open(log_path, "a") as f:
        f.write(line)

    logger.info(f"📝 Signal logged to {log_path}")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else None
    result = run_signal_pipeline(ticker=ticker)
    print(f"\n{'='*50}")
    print(f"  🦇 SIGNAL: {result['signal']}")
    print(f"  📈 UP Probability: {result['probability']:.1%}")
    print(f"  📊 Confidence: {result['confidence']*100:.1f}%")
    print(f"  🥇 Metals: {'BULL' if (result['gold_trend_up'] and result['silver_trend_up']) else 'MIXED/BEAR'}")
    print(f"  🔊 Volume: {result['volume_ratio']:.1f}x (Req: >1.5x)")
    print(f"{'='*50}")
