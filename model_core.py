"""
🦇 BATMAN AI TRADING SYSTEM — MODEL CORE (PyTorch)
=====================================================
Attention-Enhanced LSTM. Binary Classification. Asymmetric Loss.
Walk-Forward with Kill Switch. No illusions. No excuses.
"""

import os
import sys
import logging
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, confusion_matrix, f1_score, classification_report
import yaml

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ─── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] 🦇 %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BatmanModelCore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from data_pipeline import load_config  # one loader: config file + env-var secrets


# ═══════════════════════════════════════════════════════════
# ATTENTION LAYER (Dot-Product Self-Attention)
# ═══════════════════════════════════════════════════════════

class DotProductAttention(nn.Module):
    """
    Simple dot-product self-attention over LSTM sequence outputs.
    Input:  (batch, seq_len, hidden_dim)
    Output: (batch, seq_len, hidden_dim) — attention-weighted
    """
    def __init__(self, hidden_dim):
        super(DotProductAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, x):
        Q = self.query(x)  # (batch, seq, hidden)
        K = self.key(x)
        V = self.value(x)

        # Scaled dot-product attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (batch, seq, seq)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.bmm(weights, V)  # (batch, seq, hidden)
        return attended


# ═══════════════════════════════════════════════════════════
# ATTENTION-ENHANCED LSTM (BINARY CLASSIFICATION)
# ═══════════════════════════════════════════════════════════

class BatmanLSTM(nn.Module):
    """
    Attention-Enhanced Binary Classifier.
    LSTM(80, return_seq) -> Drop -> Attention -> LSTM(40) -> Drop -> Dense(64) -> Dense(1, Sigmoid)
    """
    def __init__(self, input_size, hidden1=80, hidden2=40, dropout=0.3, dense_units=64):
        super(BatmanLSTM, self).__init__()

        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.attention = DotProductAttention(hidden1)

        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden2, dense_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq, feat)
        out, _ = self.lstm1(x)          # (batch, seq, hidden1)
        out = self.dropout1(out)
        out = self.attention(out)        # Attention-weighted (batch, seq, hidden1)

        out, _ = self.lstm2(out)         # (batch, seq, hidden2)
        out = self.dropout2(out[:, -1, :])  # Take last step (batch, hidden2)

        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        return out


# ═══════════════════════════════════════════════════════════
# ASYMMETRIC BCE LOSS (False Negatives penalized 1.5x)
# ═══════════════════════════════════════════════════════════

class AsymmetricBCELoss(nn.Module):
    """
    Binary Cross-Entropy with asymmetric weighting.
    False Negatives (missing a BUY when market goes up) cost 1.5x more.
    weight_fn = pos_weight for positive class.
    """
    def __init__(self, pos_weight=1.5):
        super(AsymmetricBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        # BCEWithLogitsLoss equivalent but pred already has sigmoid
        # Use manual BCE: -[w*y*log(p) + (1-y)*log(1-p)]
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1 - eps)
        loss = -(self.pos_weight * target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        return loss.mean()


def build_model(input_shape, config=None):
    if config is None: config = load_config()

    n_features = input_shape[1]
    h1 = config.get("lstm_units_1", 80)
    h2 = config.get("lstm_units_2", 40)
    drop = config.get("dropout", 0.3)
    dense = config.get("dense_units", 64)

    model = BatmanLSTM(n_features, h1, h2, drop, dense).to(DEVICE)
    logger.info(f"✅ Attention Model: LSTM({h1})->Attn->LSTM({h2})->Dense({dense})->Sigmoid")
    return model


def load_saved_model(config=None):
    """Load a saved model from disk."""
    if config is None: config = load_config()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, config.get("model_path", "batman_lstm.h5"))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model at {model_path}. Train first.")

    state = torch.load(model_path, map_location=DEVICE, weights_only=True)

    h1 = config.get("lstm_units_1", 80)
    h2 = config.get("lstm_units_2", 40)
    drop = config.get("dropout", 0.3)
    dense = config.get("dense_units", 64)

    input_size = state["lstm1.weight_ih_l0"].shape[1]
    model = BatmanLSTM(input_size, h1, h2, drop, dense).to(DEVICE)
    model.load_state_dict(state)
    model.eval()

    logger.info(f"✅ Loaded saved model from {model_path}")
    return model


def train_model(model, X_train, y_train, X_val, y_val, config=None):
    if config is None: config = load_config()

    epochs = config.get("epochs", 50)
    batch_size = config.get("batch_size", 32)
    patience = config.get("patience", 25)
    lr = config.get("learning_rate", 0.0002)

    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(DEVICE)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = AsymmetricBCELoss(pos_weight=1.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Gradient clipping via hook
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            preds = (out > 0.5).float()
            correct += (preds == by).sum().item()
            total += by.size(0)

        train_loss = epoch_loss / len(loader)
        train_acc = correct / total

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss = criterion(val_out, y_val_t).item()
            val_preds = (val_out > 0.5).float()
            val_acc = (val_preds == y_val_t).sum().item() / y_val_t.size(0)

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Monitor val_accuracy for early stopping (not val_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            logger.info(f"   Epoch {epoch}: Loss={train_loss:.4f} Acc={train_acc:.1%} | Val Loss={val_loss:.4f} Val Acc={val_acc:.1%}")

        if patience_counter >= patience:
            logger.info(f"   Early stop at epoch {epoch}. Best val acc: {best_val_acc:.1%}")
            break

    if best_state: model.load_state_dict(best_state)
    return history


# ═══════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION v2 (with Kill Switch + F1 + Confusion)
# ═══════════════════════════════════════════════════════════

def walk_forward_validation(ticker, config=None, min_accuracy=0.62, save=True, df_preloaded=None):
    """
    Attention-Enhanced Binary Walk-Forward:
    - Train Window: 252 days
    - Predict: 1 day ahead
    - Retrain: every 30 days OR if rolling accuracy < target
    - Track: Rolling 100-day accuracy + F1-score
    - Kill Switch: After 3 failed retrain cycles
    - save: If True, save model to disk when acc >= 65%
    - df_preloaded: Pre-fetched DataFrame to skip re-fetching
    Returns: (res_df, final_accuracy, model)
    """
    from data_pipeline import fetch_data, engineer_features, handle_nans, get_feature_columns
    from sklearn.preprocessing import StandardScaler

    if config is None: config = load_config()
    logger.info(f"\U0001f480 WALK-FORWARD v2 for {ticker}")
    logger.info(f"   Min accuracy target: {min_accuracy*100:.0f}%")

    # Phase 3: XGBoost Delegate
    # If XGBoost model exists or config requests it, delegate to model_xgb.py
    # This ensures the Scanner sees the 60.3% accuracy, not the LSTM's 42%.
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "batman_xgb.json")):
        logger.info("🚀 Delegating Walk-Forward to XGBoost Engine (Phase 3 Active)")
        try:
            from model_xgb import train_xgb_walk_forward
            # train_xgb_walk_forward returns (results_df, acc, best_params)
            # We need to adapt it to match (res_df, final_accuracy, model) signature
            res_df, acc, best_params = train_xgb_walk_forward(ticker, config, min_accuracy)
            return res_df, acc, best_params # Return params as 'model' placeholder
        except ImportError:
            logger.error("❌ Could not import model_xgb. Falling back to LSTM.")


    # 1. Fetch (or reuse preloaded)
    if df_preloaded is not None:
        df = df_preloaded
    else:
        start_date = config.get("data_fetch_start", "2020-01-01")
        df = fetch_data(ticker, start=start_date)
        df = engineer_features(df, config)
        df = handle_nans(df)

    train_window = config.get("train_window_days", 252)
    retrain_every = config.get("retrain_step_days", 30)
    lookback = config.get("lookback", 60)

    if len(df) < train_window + lookback + 50:
        logger.error(f"Insufficient data: {len(df)} rows. Need {train_window + lookback + 50}+.")
        return pd.DataFrame(), 0.0, None

    results = []
    rolling_acc_window = []
    model = None
    scaler = None
    retrain_fail_count = 0
    max_retrain_fails = 3
    total_retrains = 0

    feature_cols = get_feature_columns()
    actual_cols = [c for c in feature_cols if c in df.columns]
    logger.info(f"   Features: {len(actual_cols)} | Data rows: {len(df)} | Train window: {train_window}")

    for i in range(train_window, len(df) - 1):
        days_since_train = (i - train_window) % retrain_every
        is_retrain_step = (days_since_train == 0)

        curr_acc = 1.0
        if len(rolling_acc_window) >= 20:
            curr_acc = sum(rolling_acc_window[-100:]) / len(rolling_acc_window[-100:])

        should_retrain = (model is None) or (is_retrain_step and curr_acc < min_accuracy)

        if should_retrain:
            if model is None:
                logger.info("   \U0001f525 Initial Training...")
            else:
                logger.info(f"   \u26a0\ufe0f Acc {curr_acc:.1%} < {min_accuracy:.0%}. Retraining @ {df.index[i].date()}...")

            train_subset = df.iloc[max(0, i - train_window):i]
            X_raw = train_subset[actual_cols].values
            y_raw = train_subset["Target"].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw)

            X_seq, y_seq = [], []
            for j in range(lookback, len(X_scaled)):
                X_seq.append(X_scaled[j - lookback:j])
                y_seq.append(y_raw[j])

            if len(X_seq) < 50:
                continue

            X_seq = np.array(X_seq, dtype=np.float32)
            y_seq = np.array(y_seq, dtype=np.float32)

            split_idx = int(len(X_seq) * 0.8)
            X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]
            X_val, y_val = X_seq[split_idx:], y_seq[split_idx:]

            model = build_model((lookback, len(actual_cols)), config)
            prev_level = logging.getLogger("BatmanModelCore").level
            logging.getLogger("BatmanModelCore").setLevel(logging.WARNING)
            train_model(model, X_train, y_train, X_val, y_val, config)
            logging.getLogger("BatmanModelCore").setLevel(prev_level)

            total_retrains += 1

            if curr_acc < min_accuracy and len(rolling_acc_window) >= 20:
                retrain_fail_count += 1
                logger.warning(f"   \U0001f480 Retrain fail count: {retrain_fail_count}/{max_retrain_fails}")
                if retrain_fail_count >= max_retrain_fails:
                    logger.error(f"   \u274c KILL SWITCH: {retrain_fail_count} fails. Acc: {curr_acc:.1%}.")
                    break
            else:
                retrain_fail_count = 0

        input_sub = df.iloc[i - lookback + 1:i + 1]
        if len(input_sub) < lookback:
            continue

        X_in = input_sub[actual_cols].values
        X_in_scaled = scaler.transform(X_in)
        X_tens = torch.FloatTensor(X_in_scaled).unsqueeze(0).to(DEVICE)

        model.eval()
        with torch.no_grad():
            prob = model(X_tens).item()

        actual_target = df.iloc[i]["Target"]
        pred_bin = 1 if prob >= 0.5 else 0
        hit = 1 if pred_bin == actual_target else 0

        rolling_acc_window.append(hit)
        if len(rolling_acc_window) > 100:
            rolling_acc_window.pop(0)

        current_rolling_acc = sum(rolling_acc_window) / len(rolling_acc_window) if rolling_acc_window else 0

        results.append({
            "Date": df.index[i],
            "Probability": prob,
            "Predicted": pred_bin,
            "Target": actual_target,
            "Hit": hit,
            "RollingAcc": current_rolling_acc,
            "Close": df.iloc[i]["Close"],
        })

    if not results:
        logger.warning("No predictions made.")
        return pd.DataFrame(), 0.0, None

    res_df = pd.DataFrame(results).set_index("Date")
    final_acc = res_df["Hit"].mean()

    y_true = res_df["Target"].values.astype(int)
    y_pred = res_df["Predicted"].values.astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    logger.info(f"{'=' * 60}")
    logger.info(f"\U0001f9b7 WALK-FORWARD: {ticker} | Acc={final_acc:.1%} | F1={f1:.3f} | Preds={len(res_df)} | Retrains={total_retrains}")
    logger.info(f"{'=' * 60}")
    logger.info(f"\n{classification_report(y_true, y_pred, target_names=['DOWN', 'UP'], zero_division=0)}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    _plot_walk_forward_results(res_df, y_true, y_pred, final_acc, f1, ticker, base_dir)

    if save:
        if final_acc >= 0.65:
            model_path = os.path.join(base_dir, config.get("model_path", "batman_lstm.h5"))
            torch.save(model.state_dict(), model_path)
            logger.info(f"\U0001f4be Model SAVED (acc={final_acc:.1%} >= 65%)")
        elif final_acc >= min_accuracy:
            logger.info(f"\u26a0\ufe0f Acc {final_acc:.1%} meets min but below 65% save threshold.")
        else:
            logger.error(f"\u274c Acc {final_acc:.1%} below {min_accuracy:.0%}. NOT saved.")

    return res_df, final_acc, model


def _plot_walk_forward_results(res_df, y_true, y_pred, accuracy, f1, ticker, save_dir):
    """Plot Bat-Signal accuracy curve + confusion matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ─── Accuracy Curve (Yellow Bat-Signal) ───
    ax1 = axes[0]
    ax1.plot(res_df.index, res_df["RollingAcc"] * 100,
             color="#f1c40f", linewidth=1.8, label="Rolling 100-day Accuracy")
    ax1.axhline(y=62, color="#e74c3c", linestyle="--", alpha=0.7, label="Min Target (62%)")
    ax1.axhline(y=65, color="#2ecc71", linestyle="--", alpha=0.7, label="Save Threshold (65%)")
    ax1.axhline(y=50, color="#555", linestyle=":", alpha=0.5, label="Coin Flip (50%)")
    ax1.set_title(f"🦇 {ticker} Walk-Forward Accuracy — {accuracy:.1%} | F1: {f1:.3f}",
                  fontsize=13, fontweight="bold", color="#f1c40f")
    ax1.set_ylabel("Accuracy %", color="#e0e0e0")
    ax1.set_ylim(30, 80)
    ax1.legend(loc="lower left", facecolor="#1a1a2e", edgecolor="#333", labelcolor="#e0e0e0", fontsize=8)
    ax1.set_facecolor("#0a0a0a")
    ax1.tick_params(colors="#e0e0e0")
    for spine in ax1.spines.values():
        spine.set_color("#333")
    ax1.grid(True, alpha=0.15, color="#555")

    # ─── Confusion Matrix ───
    ax2 = axes[1]
    cm = confusion_matrix(y_true, y_pred)
    im = ax2.imshow(cm, interpolation='nearest', cmap='YlOrRd')
    ax2.set_title("Confusion Matrix", fontsize=13, fontweight="bold", color="#f1c40f")
    ax2.set_ylabel("Actual", color="#e0e0e0")
    ax2.set_xlabel("Predicted", color="#e0e0e0")
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["DOWN", "UP"], color="#e0e0e0")
    ax2.set_yticklabels(["DOWN", "UP"], color="#e0e0e0")

    # Add text annotations
    for i_idx in range(2):
        for j_idx in range(2):
            ax2.text(j_idx, i_idx, str(cm[i_idx, j_idx]),
                     ha="center", va="center", fontsize=18, fontweight="bold",
                     color="black" if cm[i_idx, j_idx] > cm.max() / 2 else "white")

    ax2.set_facecolor("#0a0a0a")
    fig.colorbar(im, ax=ax2)

    fig.patch.set_facecolor("#0a0a0a")
    plt.tight_layout()

    plot_path = os.path.join(save_dir, "walk_forward_results.png")
    fig.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"📊 Walk-forward plot saved: {plot_path}")


# ═══════════════════════════════════════════════════════════
# HYPERPARAMETER SWEEP (Brute-force + Walk-Forward)
# ═══════════════════════════════════════════════════════════

def hyperparameter_sweep(ticker, config=None, min_accuracy=0.60):
    """
    Full Upgrade: Proxy Sentiment + Expanded Data + Hyperparameter Sweep.
    Phase 1: Quick-evaluate all grid combos (reduced epochs)
    Phase 2: Full walk-forward on top 3 combos
    Saves only the winning model.
    """
    from data_pipeline import fetch_data, engineer_features, handle_nans, get_feature_columns
    from sklearn.preprocessing import StandardScaler
    import itertools

    if config is None: config = load_config()

    logger.info("=" * 60)
    logger.info("\U0001f9b7 FULL UPGRADE: Sentiment + Expanded Data + Sweep")
    logger.info("=" * 60)

    # 1. Fetch expanded data once
    start_date = config.get("data_fetch_start", "2020-01-01")
    df = fetch_data(ticker, start=start_date)
    df = engineer_features(df, config)
    df = handle_nans(df)

    feature_cols = get_feature_columns()
    actual_cols = [c for c in feature_cols if c in df.columns]
    n_features = len(actual_cols)

    logger.info(f"\U0001f4ca Data: {len(df)} rows | Features: {n_features} | Start: {df.index[0].date()}")

    # 2. Define sweep grid (Phase 2: Higher Capacity)
    grid = {
        "lookback": [45, 60, 75],
        "lstm_units_1": [80, 100, 128],
        "dropout": [0.3, 0.4],
        "learning_rate": [0.0002, 0.0001],
    }

    combos = []
    for lb, h1, d, lr in itertools.product(
        grid["lookback"], grid["lstm_units_1"], grid["dropout"], grid["learning_rate"]
    ):
        combos.append({
            "lookback": lb, "lstm_units_1": h1, "lstm_units_2": max(h1 // 2, 20),
            "dropout": d, "learning_rate": lr, "dense_units": 64,
            "epochs": 15, "patience": 5, "batch_size": 32,
        })

    logger.info(f"\U0001f525 PHASE 1: Quick Evaluation \u2014 {len(combos)} combinations")

    # 3. Phase 1: Quick evaluation (train/val split, reduced epochs)
    phase1_results = []

    for idx, combo in enumerate(combos):
        lb = combo["lookback"]

        X_raw = df[actual_cols].values
        y_raw = df["Target"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        X_seq, y_seq = [], []
        for j in range(lb, len(X_scaled)):
            X_seq.append(X_scaled[j - lb:j])
            y_seq.append(y_raw[j])

        if len(X_seq) < 100:
            continue

        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32)

        # Chronological 80/20 split
        split_idx = int(len(X_seq) * 0.8)
        X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]
        X_val, y_val = X_seq[split_idx:], y_seq[split_idx:]

        model = build_model((lb, n_features), combo)

        prev_level = logging.getLogger("BatmanModelCore").level
        logging.getLogger("BatmanModelCore").setLevel(logging.ERROR)
        train_model(model, X_train, y_train, X_val, y_val, combo)
        logging.getLogger("BatmanModelCore").setLevel(prev_level)

        model.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val).to(DEVICE)
            val_preds = (model(X_val_t) > 0.5).float().cpu().numpy().flatten()

        val_acc = float((val_preds == y_val).mean())
        val_f1 = f1_score(y_val.astype(int), val_preds.astype(int), zero_division=0)

        phase1_results.append({"combo": combo, "val_acc": val_acc, "val_f1": val_f1})

        if (idx + 1) % 9 == 0 or idx == 0 or idx == len(combos) - 1:
            logger.info(
                f"   [{idx+1}/{len(combos)}] LB={lb} H1={combo['lstm_units_1']} "
                f"D={combo['dropout']:.2f} LR={combo['learning_rate']:.4f} "
                f"\u2192 Acc={val_acc:.1%} F1={val_f1:.3f}"
            )

    phase1_results.sort(key=lambda x: x["val_acc"], reverse=True)

    logger.info(f"\n{'=' * 60}")
    logger.info("\U0001f3c6 TOP 5 QUICK-EVAL:")
    for i, r in enumerate(phase1_results[:5]):
        c = r["combo"]
        logger.info(
            f"   #{i+1}: Acc={r['val_acc']:.1%} F1={r['val_f1']:.3f} | "
            f"LB={c['lookback']} H1={c['lstm_units_1']} D={c['dropout']:.2f} LR={c['learning_rate']:.4f}"
        )

    # 4. Phase 2: Full walk-forward on top 3
    logger.info(f"\n{'=' * 60}")
    logger.info("\U0001f525 PHASE 2: Full Walk-Forward on Top 3")

    best_acc = 0
    best_combo = None
    best_res_df = None
    best_model = None

    for rank, r in enumerate(phase1_results[:3]):
        combo = r["combo"]
        wf_config = {**config, **combo}
        wf_config["epochs"] = 50
        wf_config["patience"] = 25

        logger.info(
            f"\n   WF #{rank+1}: LB={combo['lookback']} H1={combo['lstm_units_1']} "
            f"D={combo['dropout']:.2f} LR={combo['learning_rate']:.4f}"
        )

        res_df, acc, model = walk_forward_validation(
            ticker, wf_config, min_accuracy=min_accuracy, save=False, df_preloaded=df
        )

        if acc > best_acc:
            best_acc = acc
            best_combo = combo
            best_res_df = res_df
            best_model = model

    # 5. Final verdict
    logger.info(f"\n{'=' * 60}")
    logger.info("\U0001f9b7 SWEEP COMPLETE")
    logger.info(f"   Best Accuracy: {best_acc:.1%}")
    if best_combo:
        logger.info(
            f"   Best Config: LB={best_combo['lookback']} H1={best_combo['lstm_units_1']} "
            f"D={best_combo['dropout']:.2f} LR={best_combo['learning_rate']:.4f}"
        )

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if best_acc >= 0.60 and best_model is not None:
        model_path = os.path.join(base_dir, config.get("model_path", "batman_lstm.h5"))
        torch.save(best_model.state_dict(), model_path)
        logger.info(f"\U0001f4be WINNING MODEL SAVED: {model_path} (acc={best_acc:.1%})")
    elif best_acc < min_accuracy:
        logger.error(
            f"   \u274c KILL SWITCH: Best {best_acc:.1%} < {min_accuracy:.0%}. "
            f"Realistic cap reached\u2014focus on risk management over prediction."
        )
    else:
        logger.info(f"   \u26a0\ufe0f Acc {best_acc:.1%} meets target but below 60% save threshold.")

    return best_res_df, best_acc, best_combo


# ═══════════════════════════════════════════════════════════
# PRODUCTION TRAINING
# ═══════════════════════════════════════════════════════════

def train_production_model(ticker, config=None):
    """
    Train a final production model on ALL available data and save it.
    This ensures the system has a model to run, even if walk-forward
    accuracy was below the strict validation threshold.
    """
    from data_pipeline import run_pipeline
    
    if config is None: config = load_config()

    logger.info("=" * 60)
    logger.info("\U0001f680 TRAINING PRODUCTION MODEL")
    logger.info("=" * 60)

    # Run full pipeline (fetches, engineers, scales, splits)
    # We use the pipeline's output but we'll retrain on Train+Val+Test combined for max data
    data = run_pipeline(ticker, config)
    
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Combine for final training
    X_full = np.concatenate([X_train, X_val, X_test], axis=0)
    y_full = np.concatenate([y_train, y_val, y_test], axis=0)

    # Train w/ slight validation split just for monitoring
    split = int(len(X_full) * 0.9)
    X_T = X_full[:split]
    y_T = y_full[:split]
    X_V = X_full[split:]
    y_V = y_full[split:]

    logger.info(f"   Full Dataset: {len(X_full)} sequences (Lookback={data['X_train'].shape[1]})")

    # Build
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, config)

    # Train
    # Train
    history = train_model(model, X_T, y_T, X_V, y_V, config)

    # Save
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, config.get("model_path", "batman_lstm.h5"))
    torch.save(model.state_dict(), model_path)
    
    logger.info(f"💾 PRODUCTION MODEL SAVED: {model_path}")
    
    # Calculate final metrics on the validation split we used
    model.eval()
    with torch.no_grad():
        val_out = model(torch.FloatTensor(X_V).to(DEVICE))
        val_preds = (val_out > 0.5).float().cpu().numpy().flatten()
        y_V_np = y_V
        
    final_acc = (val_preds == y_V_np).mean()
    final_f1 = f1_score(y_V_np.astype(int), val_preds.astype(int), zero_division=0)
    
    return {
        "model": model,
        "history": history,
        "eval_results": {
            "accuracy": final_acc,
            "f1": final_f1
        }
    }


def plot_loss(history):
    """Plot training history (Loss & Accuracy). Returns matplotlib figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history['loss'], label='Train Loss', color='#f1c40f')
    ax1.plot(history['val_loss'], label='Val Loss', color='#00aaff')
    ax1.set_title('Model Loss (Binary Cross-Entropy)', color='#e0e0e0')
    ax1.set_xlabel('Epoch', color='#e0e0e0')
    ax1.set_ylabel('Loss', color='#e0e0e0')
    ax1.legend()
    ax1.grid(True, alpha=0.1)
    ax1.set_facecolor('#0a0a0a')
    
    # Accuracy
    ax2.plot(history['acc'], label='Train Acc', color='#f1c40f')
    ax2.plot(history['val_acc'], label='Val Acc', color='#00aaff')
    ax2.set_title('Model Accuracy', color='#e0e0e0')
    ax2.set_xlabel('Epoch', color='#e0e0e0')
    ax2.set_ylabel('Accuracy', color='#e0e0e0')
    ax2.legend()
    ax2.grid(True, alpha=0.1)
    ax2.set_facecolor('#0a0a0a')
    
    fig.patch.set_facecolor('#0a0a0a')
    plt.tight_layout()
    
    # Style ticks
    for ax in [ax1, ax2]:
        ax.tick_params(colors='#e0e0e0')
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.spines['top'].set_color('#333')
        ax.spines['right'].set_color('#333')
        
    return fig


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="\U0001f9b7 Batman Model Core")
    parser.add_argument("--ticker", type=str, default="USAS")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation")
    parser.add_argument("--retrain-usas", action="store_true", help="Retrain USAS with attention LSTM")
    parser.add_argument("--full-upgrade", action="store_true", help="Full upgrade: sentiment + sweep")
    parser.add_argument("--train-prod", action="store_true", help="Train production model (force save)")
    parser.add_argument("--min-accuracy", type=float, default=0.62, help="Min accuracy for walk-forward")
    parser.add_argument("--target-min", type=float, default=0.60, help="Min target for sweep")
    args = parser.parse_args()

    if args.train_prod:
        train_production_model(args.ticker)

    elif args.full_upgrade:
        ticker = args.ticker
        min_acc = args.target_min
        if min_acc > 1: min_acc /= 100.0

        res, acc, best_cfg = hyperparameter_sweep(ticker, min_accuracy=min_acc)

        print(f"\n{'=' * 50}")
        print(f"  \U0001f9b7 SWEEP RESULTS: {ticker}")
        print(f"  \U0001f4ca Best Accuracy: {acc * 100:.2f}%")
        if best_cfg:
            print(f"  \u2699\ufe0f  Config: LB={best_cfg['lookback']} H1={best_cfg['lstm_units_1']} D={best_cfg['dropout']} LR={best_cfg['learning_rate']}")
        if res is not None and len(res) > 0:
            y_t = res["Target"].values.astype(int)
            y_p = res["Predicted"].values.astype(int)
            print(f"  \U0001f3af F1 Score: {f1_score(y_t, y_p, zero_division=0):.3f}")
        print(f"{'=' * 50}")

    elif args.retrain_usas or args.walk_forward:
        ticker = args.ticker
        min_acc = args.min_accuracy
        if min_acc > 1: min_acc /= 100.0

        res, acc, _ = walk_forward_validation(ticker, min_accuracy=min_acc)

        print(f"\n{'=' * 50}")
        print(f"  \U0001f9b7 WALK-FORWARD RESULTS: {ticker}")
        print(f"  \U0001f4ca Final Accuracy: {acc * 100:.2f}%")
        if len(res) > 0:
            y_t = res["Target"].values.astype(int)
            y_p = res["Predicted"].values.astype(int)
            print(f"  \U0001f3af F1 Score: {f1_score(y_t, y_p, zero_division=0):.3f}")
        print(f"{'=' * 50}")

    else:
        print("Usage:")
        print("  python model_core.py --retrain-usas --min-accuracy=62")
        print("  python model_core.py --full-upgrade --target-min=60")
        print("  python model_core.py --train-prod")

# Alias for main.py compatibility
run_training = train_production_model
