"""
Honest walk-forward evaluation.

Answers one question: does the model beat simple baselines on days it has never seen?

It replays the same walk-forward scheme the live system uses (train on the last
252 trading days, retrain every 30, predict the next day) on the saved data.csv,
so it runs offline and is reproducible. Results go to results/.

    python evaluate.py              # uses data.csv
    python evaluate.py --seed 7     # different XGBoost seed
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from scipy.stats import binomtest

from data_pipeline import get_feature_columns

HERE = os.path.dirname(os.path.abspath(__file__))

# The hyperparameters the original sweep picked. They were chosen on raw accuracy,
# which on a 60/40 label rewards predicting "down" almost every day.
ORIGINAL_PARAMS = dict(max_depth=5, learning_rate=0.01, n_estimators=50, subsample=0.8, colsample_bytree=0.8)


def walk_forward(df, cols, params, balanced, train_window=252, retrain_every=30, seed=42):
    probs, model = [], None
    for i in range(train_window, len(df)):
        if (i - train_window) % retrain_every == 0:
            tr = df.iloc[i - train_window:i]
            extra = {}
            if balanced:
                pos = tr["Target"].sum()
                extra["scale_pos_weight"] = (len(tr) - pos) / max(pos, 1)
            model = xgb.XGBClassifier(**params, **extra, eval_metric="logloss", random_state=seed, n_jobs=1)
            model.fit(tr[cols], tr["Target"])
        probs.append(model.predict_proba(df.iloc[i:i + 1][cols])[0, 1])
    return np.array(probs)


def score(name, y, pred, prob=None):
    out = {
        "model": name,
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "f1_up": round(float(f1_score(y, pred, zero_division=0)), 4),
        "share_predicted_up": round(float(np.mean(pred)), 4),
    }
    out["roc_auc"] = round(float(roc_auc_score(y, prob)), 4) if prob is not None else None
    return out


def _plot(dates, y, preds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    colours = {"Always down": "#9aa1ad", "XGBoost, original": "#e4572e", "XGBoost, class-balanced": "#2e86ab"}
    for name, p in preds.items():
        rolling = pd.Series((p == y).astype(float), index=dates).rolling(120).mean()
        ax.plot(rolling.index, rolling.values, label=name, color=colours.get(name), lw=2 if "Always" not in name else 1.5,
                ls="--" if "Always" in name else "-")
    ax.axhline(0.5, color="#cccccc", lw=1)
    ax.set_ylabel("Accuracy, rolling 120 days")
    ax.set_title("USAS next-day direction: walk-forward accuracy vs the always-down baseline", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data, parse_dates=["Date"]).set_index("Date")
    cols = [c for c in get_feature_columns() if c in df.columns]
    test = df.iloc[252:]
    y = test["Target"].values

    rows = []
    # Baselines
    majority = int(df.iloc[:252]["Target"].mean() >= 0.5)
    rows.append(score("Always predict the majority class (down)", y, np.full_like(y, majority)))
    rows.append(score("Repeat yesterday's direction", y, (test["Return_1d"].values > 0).astype(int)))

    # Original setup vs class-balanced
    p_orig = walk_forward(df, cols, ORIGINAL_PARAMS, balanced=False, seed=args.seed)
    rows.append(score("XGBoost, original settings", y, (p_orig >= 0.5).astype(int), p_orig))
    p_bal = walk_forward(df, cols, ORIGINAL_PARAMS, balanced=True, seed=args.seed)
    rows.append(score("XGBoost, class-balanced", y, (p_bal >= 0.5).astype(int), p_bal))

    # Is the best model's hit rate distinguishable from the majority baseline?
    hits = int(((p_bal >= 0.5).astype(int) == y).sum())
    base_rate = float((y == majority).mean())
    p_value = binomtest(hits, len(y), base_rate, alternative="greater").pvalue

    summary = {
        "ticker": "USAS",
        "test_days": int(len(y)),
        "test_period": [str(test.index.min().date()), str(test.index.max().date())],
        "share_of_up_days": round(float(y.mean()), 4),
        "results": rows,
        "balanced_model_vs_majority_p_value": round(float(p_value), 4),
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    _plot(test.index, y, {
        "Always down": np.full_like(y, majority),
        "XGBoost, original": (p_orig >= 0.5).astype(int),
        "XGBoost, class-balanced": (p_bal >= 0.5).astype(int),
    }, os.path.join(HERE, "results", "walk_forward_vs_baseline.png"))
    with open(os.path.join(HERE, "results", "evaluation.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTest days: {len(y)}  ({summary['test_period'][0]} to {summary['test_period'][1]})")
    print(f"Up days in test period: {y.mean():.1%}\n")
    print(f"{'Model':<44}{'Acc':>7}{'BalAcc':>8}{'F1 up':>7}{'AUC':>7}{'% up':>7}")
    for r in rows:
        auc = f"{r['roc_auc']:.3f}" if r["roc_auc"] is not None else "  -  "
        print(f"{r['model']:<44}{r['accuracy']:>7.1%}{r['balanced_accuracy']:>8.1%}{r['f1_up']:>7.3f}{auc:>7}{r['share_predicted_up']:>7.0%}")
    print(f"\nBalanced model vs always-down baseline, one-sided binomial p = {p_value:.3f}")


if __name__ == "__main__":
    main()
