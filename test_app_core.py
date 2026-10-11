"""Tests for the dashboard logic, run on the bundled USAS snapshot (no network)."""
import numpy as np
import pandas as pd

import app_core as core


def _df():
    return pd.read_csv(core.SAMPLE_PATH, parse_dates=["Date"]).set_index("Date").tail(420)


def test_walk_forward_is_out_of_sample():
    df = _df()
    wf = core.walk_forward(df, train_window=252, retrain_every=30)
    # one prediction per labelled day after the first training window; the last (unlabelled) day is excluded
    assert len(wf) == len(df) - 1 - 252
    assert wf.index[0] == df.index[252]
    assert wf["prob_up"].between(0, 1).all()


def test_evaluation_includes_baselines():
    wf = core.walk_forward(_df(), 252, 30)
    ev = core.evaluate(wf)
    assert {"Repeat yesterday", "XGBoost (walk-forward)"} <= set(ev.table.index)
    assert ev.table.iloc[0]["Balanced accuracy"] == 0.5  # always-majority baseline has no skill by definition
    assert ev.verdict_level in {"good", "neutral", "bad"}


def test_backtest_costs_reduce_returns():
    wf = core.walk_forward(_df(), 252, 30)
    free = core.backtest(wf, threshold=0.5, cost_bps=0)
    costly = core.backtest(wf, threshold=0.5, cost_bps=50)
    assert costly["curve"]["Strategy"].iloc[-1] <= free["curve"]["Strategy"].iloc[-1]
    assert np.isclose(free["curve"]["Buy & hold"].iloc[-1], costly["curve"]["Buy & hold"].iloc[-1])


def test_todays_signal_shape():
    from data_pipeline import load_config
    s = core.todays_signal(_df(), load_config())
    assert s["signal"] in {"BUY", "SELL", "HOLD"}
    assert len(s["gates"]) == 5
