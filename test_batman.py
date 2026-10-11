"""
🦇 BATMAN AI TRADING SYSTEM — UNIT TESTS
============================================
Basic assertions to verify data shapes, prediction ranges,
signal logic, and risk calculations. Run: pytest   (or python test_batman.py)
"""

import os
import sys
import numpy as np

try:
    import pytest
except ImportError:  # still runnable with plain `python test_batman.py`
    pytest = None


def _need_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        if pytest:
            pytest.skip("PyTorch not installed")
        raise RuntimeError("SKIPPED: PyTorch not installed")

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_config_loads():
    """Test that config.yaml loads correctly."""
    from data_pipeline import load_config
    config = load_config()
    assert isinstance(config, dict), "Config should be a dict"
    assert "ticker" in config, "Config should have 'ticker' key"
    assert "lookback" in config, "Config should have 'lookback' key"
    assert config["lookback"] == 60, f"Lookback should be 60, got {config['lookback']}"
    assert config["risk_pct"] == 0.01, "Risk percent should be 0.01"
    print("✅ test_config_loads PASSED")


def test_feature_columns():
    """Test that feature columns list is correct."""
    from data_pipeline import get_feature_columns
    cols = get_feature_columns()
    assert isinstance(cols, list), "Feature columns should be a list"
    assert len(cols) == 45, f"Should have 45 features, got {len(cols)}"
    assert "Close" in cols, "Close should be in features"
    assert "ADX_14" in cols, "ADX_14 should be in features"
    assert "CCI_14" in cols, "CCI_14 should be in features"
    assert "Williams_R" in cols, "Williams_R should be in features"
    assert "Lag_Return_1" in cols, "Lag_Return_1 should be in features"
    assert "Proxy_Sentiment" in cols, "Proxy_Sentiment should be in features"
    print("\u2705 test_feature_columns PASSED")


def test_signal_logic():
    """Test BUY/SELL/HOLD signal generation logic (with sentiment gate)."""
    from signals_risk import generate_signal

    config = {
        "prob_threshold": 0.75,
        "volume_threshold": 1.5,
        "sentiment_threshold": 0.2,
    }

    # BUY: all gates pass (prob + metals + volume + sentiment)
    result = generate_signal(0.85, 100.0, config, 
                             gold_trend_up=True, silver_trend_up=True, 
                             volume_ratio=2.0, sentiment_score=0.5)
    assert result["signal"] == "BUY", f"Expected BUY, got {result['signal']}"
    assert result["is_binary"] == True, "Should be binary"

    # HOLD: High prob but Low Sentiment
    result = generate_signal(0.85, 100.0, config, 
                             gold_trend_up=True, silver_trend_up=True, 
                             volume_ratio=2.0, sentiment_score=0.1)
    assert result["signal"] == "HOLD", f"Expected HOLD (low sentiment), got {result['signal']}"

    # HOLD: High prob but Low Volume
    result = generate_signal(0.85, 100.0, config, 
                             gold_trend_up=True, silver_trend_up=True, 
                             volume_ratio=1.0, sentiment_score=0.5)
    assert result["signal"] == "HOLD", f"Expected HOLD (low vol), got {result['signal']}"

    # HOLD: High prob but Mixed Metals (Silver down)
    result = generate_signal(0.85, 100.0, config, 
                             gold_trend_up=True, silver_trend_up=False, 
                             volume_ratio=2.0, sentiment_score=0.5)
    assert result["signal"] == "HOLD", f"Expected HOLD (silver down), got {result['signal']}"

    # SELL: probability <= 0.25 (1 - 0.75)
    result = generate_signal(0.10, 100.0, config, 
                             gold_trend_up=True, silver_trend_up=True, 
                             volume_ratio=2.0, sentiment_score=0.5)
    assert result["signal"] == "SELL", f"Expected SELL, got {result['signal']}"

    # HOLD: middle probability
    result = generate_signal(0.50, 100.0, config, 
                             gold_trend_up=True, silver_trend_up=True, 
                             volume_ratio=2.0, sentiment_score=0.5)
    assert result["signal"] == "HOLD", f"Expected HOLD, got {result['signal']}"

    print("\u2705 test_signal_logic PASSED")


def test_risk_calculation():
    """Test position sizing and risk parameters."""
    from signals_risk import calculate_risk

    config = {
        "account_balance": 1000,
        "risk_pct": 0.01,
        "stop_loss_pct": 0.015,
        "take_profit_pct": 0.03,
    }

    # BUY signal
    result = calculate_risk("BUY", 100.0, config)
    assert result["risk_amount"] == 10.0, f"Risk amount should be £10, got {result['risk_amount']}"
    assert result["stop_loss"] == 98.5, f"Stop loss should be 98.50, got {result['stop_loss']}"
    assert result["take_profit"] == 103.0, f"Take profit should be 103.00, got {result['take_profit']}"
    assert result["shares"] > 0, "Should have at least 1 share"
    assert result["shares"] == 6, f"Shares should be 6 (10 / 1.5), got {result['shares']}"

    # HOLD signal — no position
    result = calculate_risk("HOLD", 100.0, config)
    assert result["shares"] == 0, "HOLD should have 0 shares"
    assert result["stop_loss"] == 0, "HOLD should have 0 stop loss"

    # SELL signal
    result = calculate_risk("SELL", 100.0, config)
    assert result["stop_loss"] == 101.5, f"SELL stop loss should be 101.50, got {result['stop_loss']}"
    assert result["take_profit"] == 97.0, f"SELL take profit should be 97.00, got {result['take_profit']}"

    print("✅ test_risk_calculation PASSED")


def test_sequence_creation():
    """Test sliding window sequence creation."""
    from data_pipeline import create_sequences

    n_samples = 200
    n_features = 11
    lookback = 60

    features = np.random.randn(n_samples, n_features).astype(np.float32)
    target = np.random.rand(n_samples, 1).astype(np.float32)

    X, y = create_sequences(features, target, lookback)

    expected_len = n_samples - lookback
    assert X.shape == (expected_len, lookback, n_features), \
        f"X shape should be ({expected_len}, {lookback}, {n_features}), got {X.shape}"
    assert y.shape == (expected_len,), \
        f"y shape should be ({expected_len},), got {y.shape}"
    assert X.dtype == np.float32, f"X dtype should be float32, got {X.dtype}"

    print("✅ test_sequence_creation PASSED")


def test_split_no_leakage():
    """Test that chronological split has no data leakage."""
    from data_pipeline import split_data

    n = 1000
    lookback = 60
    n_features = 11

    X = np.arange(n).reshape(-1, 1, 1).repeat(lookback, axis=1).repeat(n_features, axis=2).astype(np.float32)
    y = np.arange(n, dtype=np.float32)

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    # Ensure chronological order
    assert y_train[-1] < y_val[0], "Train should end before val starts"
    assert y_val[-1] < y_test[0], "Val should end before test starts"

    # Ensure no overlap
    train_set = set(y_train.tolist())
    val_set = set(y_val.tolist())
    test_set = set(y_test.tolist())
    assert len(train_set & val_set) == 0, "Train and val should not overlap"
    assert len(val_set & test_set) == 0, "Val and test should not overlap"
    assert len(train_set & test_set) == 0, "Train and test should not overlap"

    print("✅ test_split_no_leakage PASSED")


def test_model_builds():
    """Test that LSTM model builds with correct binary architecture."""
    _need_torch()
    from model_core import build_model

    config = {
        "lstm_units_1": 80,
        "lstm_units_2": 40,
        "dropout": 0.3,
        "dense_units": 64,
        "learning_rate": 0.0003,
    }

    input_shape = (60, 41)  # 41 features
    model = build_model(input_shape, config)

    assert model is not None, "Model should not be None"

    # Verify it's a BatmanLSTM
    from model_core import BatmanLSTM
    assert isinstance(model, BatmanLSTM), "Model should be BatmanLSTM instance"

    # Verify Attention Layer existence
    assert hasattr(model, 'attention'), "Model should have Attention layer"
    
    # Verify dual LSTM layers
    assert hasattr(model, 'lstm1'), "Model should have lstm1 layer"
    assert hasattr(model, 'lstm2'), "Model should have lstm2 layer"
    assert model.lstm1.input_size == 41, f"LSTM1 input size should be 41, got {model.lstm1.input_size}"
    assert model.lstm1.hidden_size == 80, f"LSTM1 hidden size should be 80, got {model.lstm1.hidden_size}"
    assert model.lstm2.hidden_size == 40, f"LSTM2 hidden size should be 40, got {model.lstm2.hidden_size}"

    # Verify Sigmoid output
    assert hasattr(model, 'sigmoid'), "Model should have sigmoid activation"

    print("\u2705 test_model_builds PASSED")


def test_prediction_range():
    """Test that model predictions are probabilities in [0, 1] range."""
    _need_torch()
    import torch
    from model_core import build_model

    config = {
        "lstm_units_1": 80,
        "lstm_units_2": 40,
        "dropout": 0.3,
        "dense_units": 64,
        "learning_rate": 0.0003,
    }

    model = build_model((60, 41), config)
    model.eval()

    # Random input
    X = np.random.randn(5, 60, 41).astype(np.float32)
    with torch.no_grad():
        preds = model(torch.FloatTensor(X)).cpu().numpy()

    assert preds.shape == (5, 1), f"Prediction shape should be (5, 1), got {preds.shape}"
    assert np.all(np.isfinite(preds)), "All predictions should be finite"
    assert np.all(preds >= 0) and np.all(preds <= 1), \
        f"Predictions should be in [0, 1], got min={preds.min():.4f} max={preds.max():.4f}"

    print("\u2705 test_prediction_range PASSED")


def test_data_csv_structure():
    """Test data.csv has expected structure (if it exists)."""
    import pandas as pd

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data.csv")

    if not os.path.exists(data_path):
        print("⏭️  test_data_csv_structure SKIPPED (data.csv not found — run pipeline first)")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    assert len(df) > 200, f"Data should have >200 rows, got {len(df)}"

    required_cols = ["Close", "SMA_50", "SMA_200", "RSI_14", "MACD", "Volume_Pct"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    assert df.isnull().sum().sum() == 0, "Data should have no NaN values"

    print("\u2705 test_data_csv_structure PASSED")


def test_proxy_sentiment():
    """Test proxy sentiment features are computed correctly."""
    import pandas as pd
    from sentiment import compute_proxy_sentiment

    # Create mock DataFrame with needed columns
    n = 100
    df = pd.DataFrame({
        "RSI_14": np.linspace(30, 70, n),
        "Volume_Delta": np.random.uniform(0.5, 2.0, n),
        "Return_5d": np.random.uniform(-0.05, 0.05, n),
        "GC=F_Pct_5d": np.random.uniform(-0.03, 0.03, n),
        "SI=F_Pct_5d": np.random.uniform(-0.03, 0.03, n),
        "ATR_20": np.random.uniform(0.1, 0.5, n),
        "Close": np.linspace(5, 10, n),
    })

    result = compute_proxy_sentiment(df)

    expected_cols = ["Sentiment_RSI", "Vol_Anomaly", "Momentum_Sentiment",
                    "Metal_Sentiment", "Vol_Regime", "Proxy_Sentiment"]
    for col in expected_cols:
        assert col in result.columns, f"Missing proxy sentiment column: {col}"

    # Composite should be bounded
    assert result["Proxy_Sentiment"].notna().all(), "Proxy_Sentiment should have no NaN"
    assert result["Sentiment_RSI"].min() >= -1 and result["Sentiment_RSI"].max() <= 1, \
        "Sentiment_RSI should be in [-1, 1]"

    print("\u2705 test_proxy_sentiment PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("\U0001f9b7 BATMAN AI TRADING SYSTEM — TEST SUITE")
    print("=" * 60)

    tests = [
        test_config_loads,
        test_feature_columns,
        test_signal_logic,
        test_risk_calculation,
        test_sequence_creation,
        test_split_no_leakage,
        test_model_builds,
        test_prediction_range,
        test_data_csv_structure,
        test_proxy_sentiment,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"\u274c {test_fn.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            if "SKIPPED" in str(e):
                skipped += 1
            else:
                print(f"\u274c {test_fn.__name__} ERROR: {e}")
                failed += 1

    print("\n" + "=" * 60)
    print(f"\U0001f9b7 RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
