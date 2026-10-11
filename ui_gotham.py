"""
Batman AI · Signal Lab
A research dashboard: walk-forward ML signals tested against simple baselines.
Run locally:  streamlit run ui_gotham.py
"""

import html
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import app_core as core
from data_pipeline import load_config

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Batman AI · Signal Lab", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

# ─────────────────────────── design tokens ───────────────────────────
INK, MUTE, LINE, PANEL, BG = "#eeebe4", "#9aa1ad", "#262b36", "#12151b", "#0a0c10"
PURPLE, GREEN, RED, AMBER = "#b18cff", "#3ddc84", "#ff5d5d", "#ffc53d"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@700;800;900&family=JetBrains+Mono:wght@400;500&family=Schibsted+Grotesk:wght@400;500;600;700&display=swap');
:root {{ --ink:{INK}; --mute:{MUTE}; --line:{LINE}; --panel:{PANEL}; --bg:{BG}; --purple:{PURPLE}; --green:{GREEN}; --red:{RED}; --amber:{AMBER}; }}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {{ font-family: "Schibsted Grotesk", system-ui, sans-serif; }}
.stApp {{ background: radial-gradient(1200px 600px at 85% -10%, rgba(177,140,255,.08), transparent 60%), var(--bg); color: var(--ink); }}
#MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1240px; }}
section[data-testid="stSidebar"] {{ background: #0d0f14; border-right: 1px solid var(--line); }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}
h1, h2, h3 {{ font-family: "Big Shoulders Display", "Arial Narrow", sans-serif !important; text-transform: uppercase; letter-spacing: .01em; color: var(--ink); }}
.mono, .eyebrow {{ font-family: "JetBrains Mono", ui-monospace, monospace; }}
.eyebrow, [data-testid="stMarkdownContainer"] p.eyebrow {{ font-family:"JetBrains Mono", ui-monospace, monospace !important; font-size: .72rem !important; letter-spacing: .1em; text-transform: uppercase; color: var(--mute) !important; margin: 0 !important; line-height:1.5 !important; }}
[data-testid="stMarkdownContainer"] p.note, .note {{ color: var(--mute); font-size:.85rem !important; line-height:1.55; }}

/* header */
.top {{ display:flex; align-items:flex-end; justify-content:space-between; gap:24px; flex-wrap:wrap; margin-bottom: 1.2rem; }}
.word, [data-testid="stMarkdownContainer"] p.word {{ font-family:"Big Shoulders Display","Arial Narrow",sans-serif !important; font-weight:900 !important; font-size: clamp(2.8rem, 1.8rem + 3.4vw, 4.8rem) !important; line-height:.85 !important; text-transform:uppercase; margin:8px 0 0 !important; letter-spacing:.005em; }}
.word span {{ color: transparent; -webkit-text-stroke: 1.5px var(--ink); }}
.top-links {{ display:flex; gap:8px; flex-wrap:wrap; }}
.pill {{ display:inline-flex; align-items:center; gap:8px; padding:6px 12px; border-radius:999px; border:1px solid var(--line); background: rgba(18,21,27,.8); color: var(--ink) !important; text-decoration:none !important; font-size:.85rem; }}
.pill:hover {{ border-color: var(--mute); }}
.pill.warn {{ border-color: rgba(255,197,61,.45); color: var(--amber) !important; }}
.dot {{ width:8px; height:8px; border-radius:50%; background: var(--green); box-shadow: 0 0 10px var(--green); }}
.dot.sample {{ background: var(--amber); box-shadow: 0 0 10px var(--amber); }}

/* kpis */
.kpis {{ display:grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap:12px; margin: 4px 0 18px; }}
@media (max-width: 900px) {{ .kpis {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
.kpi {{ background: var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px 16px; }}
.kpi .v {{ font-family:"Big Shoulders Display",sans-serif; font-weight:800; font-size:2rem; line-height:1.05; margin-top:6px; font-variant-numeric: tabular-nums; }}
.kpi .s {{ font-family:"JetBrains Mono",monospace; font-size:.75rem; color: var(--mute); margin-top:2px; }}
.up {{ color: var(--green); }} .down {{ color: var(--red); }} .purple {{ color: var(--purple); }}

/* cards & verdict */
.card {{ background: var(--panel); border:1px solid var(--line); border-radius:16px; padding:18px 20px; }}
.verdict {{ border-radius:16px; padding:18px 20px; border:1px solid var(--line); display:flex; gap:14px; align-items:flex-start; margin-bottom: 14px; }}
.verdict b {{ font-family:"Big Shoulders Display",sans-serif; font-size:1.5rem; text-transform:uppercase; letter-spacing:.01em; display:block; margin-bottom:4px; }}
.verdict.good {{ background: rgba(61,220,132,.07); border-color: rgba(61,220,132,.4); }}
.verdict.neutral {{ background: rgba(255,197,61,.07); border-color: rgba(255,197,61,.4); }}
.verdict.bad {{ background: rgba(255,93,93,.06); border-color: rgba(255,93,93,.35); }}
.signal {{ text-align:center; padding: 22px 16px; }}
.signal .big {{ font-family:"Big Shoulders Display",sans-serif; font-weight:900; font-size:5rem; line-height:.9; }}
.signal.BUY .big {{ color: var(--green); }} .signal.SELL .big {{ color: var(--red); }} .signal.HOLD .big {{ color: var(--ink); }}
.gate {{ display:flex; justify-content:space-between; gap:12px; padding:11px 0; border-top:1px solid var(--line); font-size:.95rem; }}
.gate:first-child {{ border-top:0; }}
.gate .r {{ font-family:"JetBrains Mono",monospace; font-size:.8rem; color: var(--mute); text-align:right; }}
.ok {{ color: var(--green); font-weight:700; }} .no {{ color: var(--red); font-weight:700; }}
.rows {{ display:grid; gap:0; }}
.rows div {{ display:flex; justify-content:space-between; padding:9px 0; border-top:1px solid var(--line); font-size:.95rem; }}
.rows div:first-child {{ border-top:0; }}
.rows span:last-child {{ font-family:"JetBrains Mono",monospace; }}
.note {{ color: var(--mute); font-size:.85rem; }}

/* streamlit widgets */
[data-testid="stTab"] {{ padding: 10px 14px !important; }}
[data-testid="stTab"] p {{ font-family:"JetBrains Mono",monospace !important; text-transform:uppercase; letter-spacing:.08em; font-size:.76rem !important; color: var(--mute); }}
[data-testid="stTab"][aria-selected="true"] p, [data-testid="stTab"]:hover p {{ color: var(--ink) !important; }}
.stButton > button, .stFormSubmitButton > button {{ border-radius: 999px; font-weight:600; border:1px solid var(--line); }}
.stFormSubmitButton > button {{ background: var(--ink); color: var(--bg); border-color: var(--ink); width:100%; }}
.stFormSubmitButton > button:hover {{ background:#fff; color: var(--bg); }}
[data-testid="stDataFrame"] {{ border:1px solid var(--line); border-radius: 12px; }}
</style>
""", unsafe_allow_html=True)


def plot_layout(fig, height=380, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=28, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Schibsted Grotesk, sans-serif", color=MUTE, size=12), hovermode="x unified",
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE, font=dict(color=INK, family="JetBrains Mono")),
    )
    fig.update_xaxes(gridcolor="#1b1f28", zeroline=False, showline=False)
    fig.update_yaxes(gridcolor="#1b1f28", zeroline=False, showline=False)
    return fig


def pct(x, d=1, sign=True):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{d}%}" if sign else f"{x:.{d}%}"


# ─────────────────────────── cached work ───────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_data(ticker, start):
    return core.load_market_data(ticker, start)


@st.cache_data(ttl=3600, show_spinner=False)
def get_walk_forward(ticker, start, train_window, retrain_every, balanced):
    df, _ = get_data(ticker, start)
    return core.walk_forward(df, train_window, retrain_every, balanced)


# ───────────────────────────── sidebar ─────────────────────────────
base_config = load_config()
with st.sidebar:
    st.markdown('<p class="eyebrow">Controls</p>', unsafe_allow_html=True)
    with st.form("controls"):
        ticker = st.text_input("Ticker", value=st.session_state.get("ticker", "USAS"),
                               help="Any Yahoo Finance symbol, e.g. USAS, AAPL, NVDA, GLD, BTC-USD").strip().upper()
        start = st.selectbox("History from", ["2018-01-01", "2020-01-01", "2022-01-01"], index=1)
        st.markdown('<p class="eyebrow" style="margin-top:8px">Model</p>', unsafe_allow_html=True)
        train_window = st.slider("Training window (trading days)", 126, 504, 252, 21)
        retrain_every = st.slider("Retrain every (days)", 10, 60, 30, 5)
        balanced = st.toggle("Balance up and down days", value=True,
                             help="Without this, the model can look accurate by always predicting the more common direction.")
        st.markdown('<p class="eyebrow" style="margin-top:8px">Strategy</p>', unsafe_allow_html=True)
        threshold = st.slider("Go long when P(up) is at least", 0.50, 0.80, 0.55, 0.01)
        strict = st.toggle("Also require metals, volume and sentiment filters", value=False)
        cost_bps = st.slider("Trading cost per change (bps)", 0, 50, 15, 5)
        st.markdown('<p class="eyebrow" style="margin-top:8px">Risk</p>', unsafe_allow_html=True)
        account = st.number_input("Account size (£)", 100, 1_000_000, int(base_config.get("account_balance", 1000)), 100)
        risk_pct = st.slider("Risk per trade", 0.005, 0.03, float(base_config.get("risk_pct", 0.01)), 0.005, format="%.3f")
        run = st.form_submit_button("Run analysis")
    st.session_state["ticker"] = ticker
    st.markdown(
        '<p class="note" style="margin-top:14px">Research demo, not financial advice. Models are retrained on the fly '
        'using only data available at each point in time.</p>', unsafe_allow_html=True)

config = {**base_config, "account_balance": account, "risk_pct": risk_pct}

# ───────────────────────────── data ─────────────────────────────
try:
    with st.spinner(f"Loading {ticker} prices and building 45 features…"):
        df, source = get_data(ticker, start)
except Exception as e:
    st.error(f"{e}\n\nCheck the ticker symbol and try again. USAS always works thanks to a bundled snapshot.")
    st.stop()

snap = core.market_snapshot(df)

# ───────────────────────────── header ─────────────────────────────
src_pill = ('<span class="pill"><i class="dot"></i>Live Yahoo Finance data</span>' if source == "live" else
            '<span class="pill warn"><i class="dot sample"></i>Offline snapshot (live data unavailable)</span>')
st.markdown(f"""
<div class="top">
  <div>
    <p class="eyebrow">Batman AI · Signal Lab · {html.escape(ticker)} · data to {snap['date']:%d %b %Y}</p>
    <p class="word">Signal <span>Lab</span></p>
  </div>
  <div class="top-links">
    {src_pill}
    <a class="pill" href="https://github.com/GOUTHMNN/batman-ai-trading" target="_blank">Code ↗</a>
    <a class="pill" href="https://goutham.work" target="_blank">goutham.work ↗</a>
  </div>
</div>
""", unsafe_allow_html=True)


def kpi(label, value, sub="", cls=""):
    return f'<div class="kpi"><p class="eyebrow">{label}</p><div class="v {cls}">{value}</div><div class="s">{sub}</div></div>'


st.markdown('<div class="kpis">' + "".join([
    kpi("Last close", f"{snap['close']:,.2f}", f"{snap['date']:%a %d %b}"),
    kpi("1 day", pct(snap["chg_1d"]), "close to close", "up" if snap["chg_1d"] >= 0 else "down"),
    kpi("5 days", pct(snap["chg_5d"]), "week to date", "up" if snap["chg_5d"] >= 0 else "down"),
    kpi("Volatility", pct(snap["vol_20d"], 0, False), "20-day, annualised"),
    kpi("RSI (14)", f"{snap['rsi']:.0f}", "above 200-day SMA" if snap["above_sma200"] else "below 200-day SMA",
        "purple"),
]) + "</div>", unsafe_allow_html=True)

with st.spinner("Running walk-forward evaluation (training a model every few weeks of history)…"):
    wf = get_walk_forward(ticker, start, train_window, retrain_every, balanced)
ev = core.evaluate(wf)

tab_market, tab_model, tab_signal, tab_bt, tab_method = st.tabs(
    ["Market", "Model vs baseline", "Today's signal", "Backtest", "How it works"])

# ───────────────────────────── market ─────────────────────────────
with tab_market:
    view = df.tail(504)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=view.index, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"],
                                 name="Price", increasing_line_color=GREEN, decreasing_line_color=RED,
                                 increasing_fillcolor=GREEN, decreasing_fillcolor=RED), row=1, col=1)
    for col, colour in (("SMA_50", PURPLE), ("SMA_200", AMBER)):
        if col in view:
            fig.add_trace(go.Scatter(x=view.index, y=view[col], name=col.replace("_", " "), line=dict(color=colour, width=1.6)), row=1, col=1)
    fig.add_trace(go.Bar(x=view.index, y=view["Volume"], name="Volume", marker_color="#2a3040", showlegend=False), row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(plot_layout(fig, 520), width="stretch", config={"displayModeBar": False})
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"""<div class="card"><p class="eyebrow">Why this is hard</p>
        <p style="margin:.4rem 0 0">Over this history <b>{ticker}</b> closed higher on only <b>{snap['up_share']:.0%}</b> of days.
        A model that always says "{'up' if snap['up_share'] >= .5 else 'down'}" is right {max(snap['up_share'], 1 - snap['up_share']):.0%} of the time
        without learning anything, so accuracy alone is a trap. The next tab compares the model with that baseline.</p></div>""",
                    unsafe_allow_html=True)
    with c2:
        last = df.iloc[-1]
        metal = []
        if "GC=F_Close" in df:
            metal.append(f"<div><span>Gold (GC=F)</span><span>{last['GC=F_Close']:,.0f} · {'rising' if last.get('GC=F_SMA_Delta', 0) > 0 else 'falling'}</span></div>")
        if "SI=F_Close" in df:
            metal.append(f"<div><span>Silver (SI=F)</span><span>{last['SI=F_Close']:,.2f} · {'rising' if last.get('SI=F_SMA_Delta', 0) > 0 else 'falling'}</span></div>")
        st.markdown(f"""<div class="card"><p class="eyebrow">Context</p><div class="rows">
        <div><span>Rows of history</span><span>{len(df):,}</span></div>
        <div><span>Features per day</span><span>{len(core.feature_columns(df))}</span></div>
        {''.join(metal)}
        <div><span>Volume vs 20-day average</span><span>{last.get('Volume_Delta', float('nan')):.2f}×</span></div>
        </div></div>""", unsafe_allow_html=True)

# ─────────────────────────── model vs baseline ───────────────────────────
with tab_model:
    icon = {"good": "✅", "neutral": "⚖️", "bad": "🧭"}[ev.verdict_level]
    title = {"good": "Edge detected", "neutral": "Inconclusive", "bad": "No edge"}[ev.verdict_level]
    st.markdown(f'<div class="verdict {ev.verdict_level}"><div style="font-size:1.6rem">{icon}</div><div><b>{title}</b>{html.escape(ev.verdict)}</div></div>',
                unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ev.rolling.index, y=ev.rolling["Baseline"], name="Always-majority baseline",
                             line=dict(color=MUTE, width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=ev.rolling.index, y=ev.rolling["XGBoost"], name="XGBoost walk-forward",
                             line=dict(color=PURPLE, width=2.4)))
    fig.add_hline(y=0.5, line=dict(color="#333a48", width=1))
    fig.update_yaxes(tickformat=".0%", title="Accuracy, rolling 120 days")
    st.plotly_chart(plot_layout(fig, 360), width="stretch", config={"displayModeBar": False})
    tbl = pd.DataFrame({
        "Accuracy": ev.table["Accuracy"].map("{:.1%}".format),
        "Balanced accuracy": ev.table["Balanced accuracy"].map("{:.1%}".format),
        "F1 (up days)": ev.table["F1 (up days)"].map("{:.3f}".format),
        "ROC AUC": ev.table["ROC AUC"].map(lambda v: "–" if pd.isna(v) else f"{v:.3f}"),
        "Predicts up": ev.table["Predicts up"].map("{:.0%}".format),
    })
    st.dataframe(tbl, width="stretch")
    st.markdown(f'<p class="note">{len(wf):,} out-of-sample days, {wf.index.min():%b %Y} to {wf.index.max():%b %Y}. '
                f'Balanced accuracy is the fairest single number here: 50% means no skill whatever the up/down mix. '
                f'One-sided binomial test against the baseline: p = {ev.p_value:.3f}.</p>', unsafe_allow_html=True)

# ─────────────────────────── today's signal ───────────────────────────
with tab_signal:
    sig = core.todays_signal(df, config, train_window, balanced)
    c1, c2 = st.columns([0.9, 1.4])
    with c1:
        st.markdown(f"""<div class="card signal {sig['signal']}">
          <p class="eyebrow">Signal for the next session · {sig['date']:%d %b %Y}</p>
          <div class="big">{sig['signal']}</div>
          <p class="mono" style="color:var(--mute);margin:.6rem 0 0">P(up) = {sig['prob']:.1%}</p>
        </div>""", unsafe_allow_html=True)
        r = sig["risk"]
        if sig["signal"] == "HOLD":
            st.markdown('<div class="card" style="margin-top:12px"><p class="eyebrow">Position sizing</p>'
                        '<p class="note" style="margin:.4rem 0 0">No trade. Every gate has to pass before the system takes a position.</p></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="card" style="margin-top:12px"><p class="eyebrow">Position sizing</p><div class="rows">
              <div><span>Entry</span><span>{r['entry_price']:,.2f}</span></div>
              <div><span>Stop-loss</span><span>{r['stop_loss']:,.2f}</span></div>
              <div><span>Take-profit</span><span>{r['take_profit']:,.2f}</span></div>
              <div><span>Shares</span><span>{r['shares']:,}</span></div>
              <div><span>Capital at risk</span><span>£{r['risk_amount']:,.2f}</span></div>
            </div></div>""", unsafe_allow_html=True)
    with c2:
        gates = "".join(
            f'<div class="gate"><span>{"<span class=ok>✓</span>" if g[3] else "<span class=no>✕</span>"}&nbsp; {g[0]}</span><span class="r">{g[4]}</span></div>'
            for g in sig["gates"])
        st.markdown(f'<div class="card"><p class="eyebrow">Gates (BUY needs all five)</p>{gates}</div>', unsafe_allow_html=True)
        imp = sig["importances"][::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=PURPLE,
                               text=[f"{v:.1%}" for v in imp.values], textposition="outside",
                               textfont=dict(family="JetBrains Mono", color=MUTE, size=11)))
        fig.update_xaxes(showticklabels=False, range=[0, float(imp.max()) * 1.25])
        fig.update_layout(title=dict(text="What the latest model leans on", font=dict(size=13, color=MUTE)))
        st.plotly_chart(plot_layout(fig, 300, legend=False), width="stretch", config={"displayModeBar": False})

# ───────────────────────────── backtest ─────────────────────────────
with tab_bt:
    gate = core.strict_gate_series(df, config) if strict else None
    bt = core.backtest(wf, threshold, cost_bps, gate)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt["curve"].index, y=bt["curve"]["Buy & hold"], name="Buy & hold", line=dict(color=MUTE, width=1.5)))
    fig.add_trace(go.Scatter(x=bt["curve"].index, y=bt["curve"]["Strategy"], name="Strategy", line=dict(color=PURPLE, width=2.4),
                             fill="tozeroy", fillcolor="rgba(177,140,255,.06)"))
    fig.update_yaxes(title="Growth of £1", tickprefix="£")
    st.plotly_chart(plot_layout(fig, 380), width="stretch", config={"displayModeBar": False})
    m = bt["metrics"]
    cols = st.columns(4)
    for col, (label, key, fmt) in zip(cols, [("Total return", "Total return", "pct"), ("Annualised", "Annualised return", "pct"),
                                            ("Sharpe", "Sharpe ratio", "num"), ("Max drawdown", "Max drawdown", "pct")]):
        s, h = m.loc[key, "Strategy"], m.loc[key, "Buy & hold"]
        val = pct(s) if fmt == "pct" else f"{s:.2f}"
        ref = pct(h) if fmt == "pct" else f"{h:.2f}"
        col.markdown(kpi(label, val, f"buy & hold {ref}", "up" if s >= h else "down"), unsafe_allow_html=True)
    st.markdown(f'<p class="note">Long-or-cash: hold the next session when P(up) ≥ {threshold:.2f}'
                f'{" and all filters pass" if strict else ""}. In the market {bt["exposure"]:.0%} of days, '
                f'{bt["trades"]} entries, hit rate {pct(bt["win_rate"], 0, False)}. Costs of {cost_bps} bps are charged on every switch. '
                f'Past performance in a backtest says little about the future.</p>', unsafe_allow_html=True)

# ───────────────────────────── method ─────────────────────────────
with tab_method:
    st.markdown("""
<div class="card">
<p class="eyebrow">Pipeline</p>
<div class="rows">
<div><span>1 · Data</span><span>Daily prices from Yahoo Finance, plus gold and silver futures</span></div>
<div><span>2 · Features</span><span>45 per day: trend, momentum, volatility, volume, lags, metals, price-based sentiment</span></div>
<div><span>3 · Target</span><span>Does tomorrow close above today?</span></div>
<div><span>4 · Model</span><span>XGBoost, retrained on a rolling window (walk-forward)</span></div>
<div><span>5 · Test</span><span>Every prediction is made on a day the model has never seen</span></div>
<div><span>6 · Judge</span><span>Against always-majority and repeat-yesterday baselines</span></div>
</div></div>
""", unsafe_allow_html=True)
    st.markdown("""
#### Why the baselines matter
The first version of this project reported 60% accuracy on USAS and looked like a success. It wasn't: only 40% of days
closed up, so always predicting "down" also scores 60%. The model had learned to say "down" almost every day.
Class balancing and balanced accuracy fixed the measurement, and the honest result was no edge. This dashboard shows that
test live for any ticker, so you can see for yourself whether the signal is real.

#### What's in the code
An LSTM with attention (PyTorch), the XGBoost model, a cost-aware backtester and a scheduled overnight run that emails the
day's signal. The full write-up and tests are on [GitHub](https://github.com/GOUTHMNN/batman-ai-trading).
""")
