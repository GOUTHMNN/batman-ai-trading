"""
🦇 BATMAN AI TRADING SYSTEM — XGBOOST CORE
==============================================
Gradient Boosting Pivot for maximum tabular accuracy.
Strict Walk-Forward Validation.
No Mercy.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from sklearn.model_selection import ParameterGrid
import joblib

from data_pipeline import fetch_data, engineer_features, handle_nans, get_feature_columns

# ─── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] 🦇 %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BatmanXGB")


from data_pipeline import load_config  # one loader: config file + env-var secrets


def train_xgb_walk_forward(ticker, config=None, min_accuracy=0.60):
    """
    XGBoost Walk-Forward Validation.
    Train on rolling window, predict next step.
    Grid Search optimization included.
    """
    if config is None: config = load_config()
    
    start_date = config.get("data_fetch_start", "2020-01-01")
    logger.info(f"🦇 XGBOOST WALK-FORWARD for {ticker} (Start: {start_date})")

    # 1. Pipeline
    df = fetch_data(ticker, start=start_date)
    df = engineer_features(df, config)
    df = handle_nans(df)

    feature_cols = get_feature_columns()
    actual_cols = [c for c in feature_cols if c in df.columns]
    
    logger.info(f"   Features: {len(actual_cols)} | Rows: {len(df)}")

    # 2. Walk-Forward Setup
    train_window = config.get("train_window_days", 252)
    retrain_every = config.get("retrain_step_days", 30)
    
    results = []
    model = None
    best_params = None
    
    # Initial Sweep to find best params
    initial_train = df.iloc[:train_window]
    
    # Load params if exist to stabilise results
    if os.path.exists("xgb_params.pkl"):
        logger.info("Loading saved hyperparameters from xgb_params.pkl")
        best_params = joblib.load("xgb_params.pkl")
    else:
        logger.info("🔥 Fresh Hyperparameter Sweep...")
        best_params = grid_search_xgb(initial_train, actual_cols)
        
    logger.info(f"   Best Params: {best_params}")

    # Walk-Forward Loop
    for i in range(train_window, len(df)):
        # Retrain check
        if (i - train_window) % retrain_every == 0:
            train_subset = df.iloc[max(0, i - train_window):i]
            X_train = train_subset[actual_cols]
            y_train = train_subset["Target"]
            
            # Up days are only ~40% of the data, so weight them up; otherwise the model
            # learns to say "down" every day and looks 60% accurate while knowing nothing.
            pos = int(y_train.sum())
            spw = (len(y_train) - pos) / max(pos, 1)
            model = xgb.XGBClassifier(**best_params, scale_pos_weight=spw, eval_metric="logloss")
            model.fit(X_train, y_train)
        
        # Predict
        row = df.iloc[i:i+1]
        X_test = row[actual_cols]
        target = row["Target"].values[0]
        
        prob = model.predict_proba(X_test)[0][1]
        pred = 1 if prob >= 0.5 else 0
        
        results.append({
            "Date": row.index[0],
            "Target": target,
            "Predicted": pred,
            "Probability": prob
        })

    results_df = pd.DataFrame(results).set_index("Date")
    results_df = pd.DataFrame(results).set_index("Date")
    results_df["Hit"] = (results_df["Target"] == results_df["Predicted"]).astype(int)
    results_df["RollingAcc"] = results_df["Hit"].rolling(window=50).mean()
    results_df["Close"] = [df.loc[d, "Close"] for d in results_df.index]
    
    acc = accuracy_score(results_df["Target"], results_df["Predicted"])
    f1 = f1_score(results_df["Target"], results_df["Predicted"], zero_division=0)
    bal = balanced_accuracy_score(results_df["Target"], results_df["Predicted"])
    baseline = max(results_df["Target"].mean(), 1 - results_df["Target"].mean())

    logger.info(f"{'='*60}")
    logger.info(f"XGB RESULTS: Acc={acc:.1%} (always-majority baseline {baseline:.1%}) | "
                f"Balanced acc={bal:.1%} | F1={f1:.3f} | Preds={len(results_df)}")
    if acc <= baseline:
        logger.warning("No edge over the always-majority baseline. Treat signals as noise.")
    logger.info(f"{'='*60}")
    
    # Save if good
    if acc >= min_accuracy:
        save_path = "batman_xgb.json"
        model.save_model(save_path)
        logger.info(f"💾 Model Saved: {save_path}")
        
        # Save params
        joblib.dump(best_params, "xgb_params.pkl")
    else:
        logger.warning(f"⚠️ Accuracy {acc:.1%} < {min_accuracy:.0%}. Not saved.")

    return results_df, acc, best_params


def grid_search_xgb(df, feature_cols):
    """Simple Grid Search for XGBoost."""
    X = df[feature_cols]
    y = df["Target"]
    
    # Time-series split (last 20% validation)
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]
    
    grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [50, 100, 200],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
    
    best_score = 0
    best_p = None
    
    # Sample grid to save time (Random Search effectively)
    param_list = list(ParameterGrid(grid))
    import random
    # Use full grid search (108 combinations is manageable)
    # if len(param_list) > 20:
    #     param_list = random.sample(param_list, 20)
        
    for params in param_list:
        m = xgb.XGBClassifier(**params, eval_metric="logloss", n_jobs=1)
        m.fit(X_train, y_train)
        preds = m.predict(X_val)
        acc = balanced_accuracy_score(y_val, preds)  # raw accuracy rewards always saying "down"
        
        if acc > best_score:
            best_score = acc
            best_p = params
            
    return best_p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="USAS")
    parser.add_argument("--target", type=float, default=0.60)
    args = parser.parse_args()
    
    train_xgb_walk_forward(args.ticker, min_accuracy=args.target)
