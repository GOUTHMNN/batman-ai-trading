"""
Pure logic behind the Signal Lab dashboard: data loading, walk-forward
evaluation against baselines, today's signal and a cost-aware backtest.
No Streamlit imports here, so everything is testable on its own.
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from data_pipeline import engineer_features, fetch_data, get_feature_columns, handle_nans, load_config

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(HERE, "sample_data", "usas_features.csv")

# Hyperparameters chosen by the original sweep (see xgb_params.pkl history).
XGB_PARAMS = dict(max_depth=5, learning_rate=0.01, n_estimators=50, subsample=0.8, colsample_bytree=0.8)


# ─────────────────────────────── data ───────────────────────────────

def load_market_data(ticker: str, start: str = "2020-01-01"):
    """Fetch prices from Yahoo Finance and engineer features.
    Falls back to the bundled USAS snapshot if the live fetch fails.
    Returns (df, source) where source is 'live' or 'sample'."""
    config = load_config()
    try:
        raw = fetch_data(ticker, start=start)
        df = handle_nans(engineer_features(raw, config))
        if len(df) < 400:
            raise ValueError(f"Only {len(df)} usable rows for {ticker}; need at least 400.")
        return df, "live"
    except Exception as live_error:
        if ticker.upper() == "USAS" and os.path.exists(SAMPLE_PATH):
            df = pd.read_csv(SAMPLE_PATH, parse_dates=["Date"]).set_index("Date")
            return df, "sample"
        raise RuntimeError(f"Couldn't load data for {ticker}: {live_error}") from live_error


def feature_columns(df: pd.DataFrame):
    return [c for c in get_feature_columns() if c in df.columns]


def market_snapshot(df: pd.DataFrame) -> dict:
    last, prev = df.iloc[-1], df.iloc[-2]
    week_ago = df.iloc[-6] if len(df) > 6 else df.iloc[0]
    daily = df["Close"].pct_change()
    return {
        "date": df.index[-1],
        "close": float(last["Close"]),
        "chg_1d": float(last["Close"] / prev["Close"] - 1),
        "chg_5d": float(last["Close"] / week_ago["Close"] - 1),
        "vol_20d": float(daily.tail(20).std() * np.sqrt(252)),
        "rsi": float(last.get("RSI_14", np.nan)),
        "above_sma200": bool(last["Close"] > last.get("SMA_200", np.inf)),
        "up_share": float(df["Target"].iloc[:-1].mean()),
    }


# ─────────────────────────── walk-forward ───────────────────────────

def _fit(train: pd.DataFrame, cols, balanced: bool, seed: int):
    extra = {}
    if balanced:
        pos = int(train["Target"].sum())
        extra["scale_pos_weight"] = (len(train) - pos) / max(pos, 1)
    model = xgb.XGBClassifier(**XGB_PARAMS, **extra, eval_metric="logloss", random_state=seed, n_jobs=2)
    model.fit(train[cols], train["Target"])
    return model


def walk_forward(df: pd.DataFrame, train_window=252, retrain_every=30, balanced=True, seed=42) -> pd.DataFrame:
    """Train on the last `train_window` days, predict the next day, retrain every
    `retrain_every` days. The final row's label is unknown, so it is excluded."""
    cols = feature_columns(df)
    labelled = df.iloc[:-1]
    probs, model = [], None
    for i in range(train_window, len(labelled)):
        if (i - train_window) % retrain_every == 0:
            model = _fit(labelled.iloc[i - train_window:i], cols, balanced, seed)
        probs.append(model.predict_proba(labelled.iloc[i:i + 1][cols])[0, 1])
    out = labelled.iloc[train_window:][["Close", "Return_1d", "Target"]].copy()
    out["prob_up"] = probs
    out["next_return"] = labelled["Close"].shift(-1).div(labelled["Close"]).sub(1).iloc[train_window:]
    return out


@dataclass
class Evaluation:
    table: pd.DataFrame
    rolling: pd.DataFrame
    p_value: float
    edge: float          # model balanced accuracy minus 0.5
    verdict: str
    verdict_level: str   # 'good' | 'neutral' | 'bad'


def evaluate(wf: pd.DataFrame, threshold=0.5, window=120) -> Evaluation:
    y = wf["Target"].values
    pred = (wf["prob_up"].values >= threshold).astype(int)
    majority = int(y.mean() >= 0.5)
    always = np.full_like(y, majority)
    momentum = (wf["Return_1d"].values > 0).astype(int)

    def row(name, p, prob=None):
        return {
            "Model": name,
            "Accuracy": accuracy_score(y, p),
            "Balanced accuracy": balanced_accuracy_score(y, p),
            "F1 (up days)": f1_score(y, p, zero_division=0),
            "ROC AUC": roc_auc_score(y, prob) if prob is not None and len(set(y)) > 1 else np.nan,
            "Predicts up": p.mean(),
        }

    table = pd.DataFrame([
        row(f"Always predict {'up' if majority else 'down'}", always),
        row("Repeat yesterday", momentum),
        row("XGBoost (walk-forward)", pred, wf["prob_up"].values),
    ]).set_index("Model")

    hits = int((pred == y).sum())
    base = float((always == y).mean())
    p_value = float(binomtest(hits, len(y), base, alternative="greater").pvalue)
    bal = float(table.loc["XGBoost (walk-forward)", "Balanced accuracy"])
    edge = bal - 0.5

    rolling = pd.DataFrame({
        "XGBoost": pd.Series((pred == y).astype(float), index=wf.index).rolling(window).mean(),
        "Baseline": pd.Series((always == y).astype(float), index=wf.index).rolling(window).mean(),
    }).dropna()

    if p_value < 0.05 and edge > 0.02:
        verdict, level = (f"The model beats the always-{'up' if majority else 'down'} baseline "
                          f"(balanced accuracy {bal:.1%}, p = {p_value:.3f}). Worth a closer look.", "good")
    elif edge > 0.02:
        verdict, level = (f"Some signal (balanced accuracy {bal:.1%}), but not statistically distinguishable "
                          f"from the baseline (p = {p_value:.2f}).", "neutral")
    else:
        verdict, level = (f"No measurable edge. Balanced accuracy is {bal:.1%}, a coin flip, and the model "
                          f"does not beat always predicting {'up' if majority else 'down'} (p = {p_value:.2f}).", "bad")
    return Evaluation(table, rolling, p_value, edge, verdict, level)


# ─────────────────────────── today's signal ─────────────────────────

def todays_signal(df: pd.DataFrame, config: dict, train_window=252, balanced=True, seed=42) -> dict:
    """Train on the most recent labelled window and score the latest day."""
    from signals_risk import calculate_risk, generate_signal

    cols = feature_columns(df)
    labelled = df.iloc[:-1]
    model = _fit(labelled.iloc[-train_window:], cols, balanced, seed)
    latest = df.iloc[-1]
    prob = float(model.predict_proba(df.iloc[[-1]][cols])[0, 1])

    gold_up = bool(latest.get("GC=F_SMA_Delta", 0) > 0)
    silver_up = bool(latest.get("SI=F_SMA_Delta", 0) > 0)
    vol_ratio = float(latest.get("Volume_Delta", 1.0))
    sentiment = float(latest.get("Proxy_Sentiment", 0.0))

    sig = generate_signal(prob, float(latest["Close"]), config, gold_up, silver_up, vol_ratio, sentiment)
    risk = calculate_risk(sig["signal"], float(latest["Close"]), config)

    importances = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False).head(8)
    gates = [
        ("Model probability of an up day", prob, config.get("prob_threshold", 0.75),
         prob >= config.get("prob_threshold", 0.75), f"{prob:.0%} vs {config.get('prob_threshold', 0.75):.0%} needed"),
        ("Gold trend (5-day SMA rising)", None, None, gold_up, "rising" if gold_up else "falling"),
        ("Silver trend (5-day SMA rising)", None, None, silver_up, "rising" if silver_up else "falling"),
        ("Volume vs 20-day average", vol_ratio, config.get("volume_threshold", 1.5),
         vol_ratio >= config.get("volume_threshold", 1.5), f"{vol_ratio:.2f}× vs {config.get('volume_threshold', 1.5):.1f}× needed"),
        ("Price-based sentiment", sentiment, config.get("sentiment_threshold", 0.2),
         sentiment > config.get("sentiment_threshold", 0.2), f"{sentiment:+.2f} vs {config.get('sentiment_threshold', 0.2):+.2f} needed"),
    ]
    return {"date": df.index[-1], "prob": prob, "signal": sig["signal"], "risk": risk,
            "gates": gates, "importances": importances}


# ───────────────────────────── backtest ─────────────────────────────

def backtest(wf: pd.DataFrame, threshold=0.55, cost_bps=15.0, strict_gate=None) -> dict:
    """Long-or-cash strategy: hold tomorrow if P(up) >= threshold. Costs are charged on
    every position change (commission + slippage, in basis points)."""
    position = (wf["prob_up"] >= threshold).astype(float)
    if strict_gate is not None:
        position = position * strict_gate.reindex(wf.index).fillna(0).astype(float)
    r = wf["next_return"].fillna(0.0)
    trades = position.diff().abs().fillna(position.iloc[0])
    strat = position * r - trades * cost_bps / 10_000
    equity = (1 + strat).cumprod()
    hold = (1 + r).cumprod()

    def stats(rets, eq):
        days = len(rets)
        total = float(eq.iloc[-1] - 1)
        ann = float((eq.iloc[-1]) ** (252 / max(days, 1)) - 1)
        sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
        dd = float((eq / eq.cummax() - 1).min())
        return total, ann, sharpe, dd

    s_total, s_ann, s_sharpe, s_dd = stats(strat, equity)
    h_total, h_ann, h_sharpe, h_dd = stats(r, hold)
    in_mkt = position > 0
    win_rate = float((r[in_mkt] > 0).mean()) if in_mkt.any() else float("nan")
    return {
        "curve": pd.DataFrame({"Strategy": equity, "Buy & hold": hold}),
        "metrics": pd.DataFrame({
            "Strategy": [s_total, s_ann, s_sharpe, s_dd],
            "Buy & hold": [h_total, h_ann, h_sharpe, h_dd],
        }, index=["Total return", "Annualised return", "Sharpe ratio", "Max drawdown"]),
        "exposure": float(in_mkt.mean()),
        "trades": int((position.diff() > 0).sum() + (position.iloc[0] > 0)),
        "win_rate": win_rate,
    }


def strict_gate_series(df: pd.DataFrame, config: dict) -> pd.Series:
    """Daily mask of the original rule-based filters (metals up, volume, sentiment)."""
    return (
        (df.get("GC=F_SMA_Delta", 0) > 0)
        & (df.get("SI=F_SMA_Delta", 0) > 0)
        & (df.get("Volume_Delta", 0) >= config.get("volume_threshold", 1.5))
        & (df.get("Proxy_Sentiment", 0) > config.get("sentiment_threshold", 0.2))
    )
