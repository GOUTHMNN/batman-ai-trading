"""
🦇 BATMAN AI TRADING SYSTEM — GOTHAM CITY DASHBOARD
======================================================
Streamlit app with full Batman/Gotham dark theme.
Replaced ALL icons with lethal custom SVG vectors.
NO MERCY.
"""

import os
import sys
import io
import time
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import yaml

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Page Config (MUST be first Streamlit command) ───
st.set_page_config(
    page_title="🦇 Batman AI Trading",
    page_icon="🦇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
# LETHAL SVG ASSETS
# ═══════════════════════════════════════════════════════════

# Color Palette: #111111 (stroke), #f1c40f (accent)
# viewBox="0 0 24 24"

BAT_LOGO_SVG = """<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M2 12C2 12 5 4 12 4C19 4 22 12 22 12" stroke="#f1c40f" stroke-width="1.5" stroke-linecap="round"/>
<path d="M12 18L10 14H14L12 18Z" fill="#f1c40f"/>
<path d="M2 12C4 16 8 18 12 18C16 18 20 16 22 12" stroke="#f1c40f" stroke-width="1.5" stroke-linecap="round"/>
<path d="M7 8L8 10M17 8L16 10" stroke="#f1c40f" stroke-width="1.5" stroke-linecap="round"/>
</svg>"""

BAT_SIGNAL_SVG = """<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<circle cx="12" cy="12" r="9" stroke="#f1c40f" stroke-width="1.5"/>
<path d="M12 7V3M3 12H7M21 12H17M12 21V17" stroke="#f1c40f" stroke-width="1.5" stroke-linecap="round"/>
<path d="M7 12C7 9.23858 9.23858 7 12 7C14.7614 7 17 9.23858 17 12" stroke="#f1c40f" stroke-width="1.5" stroke-dasharray="2 2"/>
</svg>"""

BAT_GEAR_SVG = """<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 15C13.6569 15 15 13.6569 15 12C15 10.3431 13.6569 9 12 9C10.3431 9 9 10.3431 9 12C9 13.6569 10.3431 15 12 15Z" stroke="#f1c40f" stroke-width="1.5"/>
<path d="M19.4 15L20.4 17.5L18 19.5L15.6 18.5C15 19 14.4 19.4 13.8 19.8L13.5 22.5H10.5L10.2 19.8C9.6 19.4 9 19 8.4 18.5L6 19.5L3.6 17.5L4.6 15C4.2 14.4 3.9 13.8 3.6 13.2L1 12.5V9.5L3.6 8.8C3.9 8.2 4.2 7.6 4.6 7L3.6 4.5L6 2.5L8.4 3.5C9 3 9.6 2.6 10.2 2.2L10.5 -0.5H13.5L13.8 2.2C14.4 2.6 15 3 15.6 3.5L18 2.5L20.4 4.5L19.4 7C19.8 7.6 20.1 8.2 20.4 8.8L23 9.5V12.5L20.4 13.2C20.1 13.8 19.8 14.4 19.4 15Z" stroke="#f1c40f" stroke-width="1.5" stroke-linejoin="round"/>
</svg>"""

BAT_HEAD_SVG = """<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M4 14C4 18 8 21 12 21C16 21 20 18 20 14V8L17 5L12 9L7 5L4 8V14Z" stroke="#f1c40f" stroke-width="1.5"/>
<path d="M8 13L10 15L16 9" stroke="#f1c40f" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

BAT_CHART_SVG = """<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M3 21H21" stroke="#f1c40f" stroke-width="1.5" stroke-linecap="round"/>
<path d="M7 21V14" stroke="#f1c40f" stroke-width="1.5" stroke-linecap="round"/>
<path d="M12 21V10" stroke="#f1c40f" stroke-width="1.5" stroke-linecap="round"/>
<path d="M17 21V6" stroke="#f1c40f" stroke-width="1.5" stroke-linecap="round"/>
<path d="M7 10L12 6L17 2" stroke="#f1c40f" stroke-width="1.5" stroke-linecap="round" stroke-dasharray="2 2"/>
</svg>"""

BAT_CLOCK_SVG = """<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<circle cx="12" cy="12" r="9" stroke="#f1c40f" stroke-width="1.5"/>
<path d="M12 7V12L15 15" stroke="#f1c40f" stroke-width="1.5" stroke-linecap="round"/>
<path d="M12 3V5" stroke="#f1c40f" stroke-width="1.5" stroke-linecap="round"/>
</svg>"""

BAT_MAIL_SVG = """<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect x="3" y="5" width="18" height="14" rx="2" stroke="#f1c40f" stroke-width="1.5"/>
<path d="M3 7L12 13L21 7" stroke="#f1c40f" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

SIGNAL_BUY_SVG = """<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 19V5" stroke="#2ecc71" stroke-width="2" stroke-linecap="round"/>
<path d="M5 12L12 5L19 12" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

SIGNAL_SELL_SVG = """<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 5V19" stroke="#c0392b" stroke-width="2" stroke-linecap="round"/>
<path d="M19 12L12 19L5 12" stroke="#c0392b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

SIGNAL_HOLD_SVG = """<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M5 12H19" stroke="#f1c40f" stroke-width="2" stroke-linecap="round"/>
</svg>"""


# ═══════════════════════════════════════════════════════════
# GOTHAM CSS — The Dark Knight Theme
# ═══════════════════════════════════════════════════════════

GOTHAM_CSS = """
<style>
    /* ─── Google Fonts ─── */
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@300;400;500;600;700&family=Roboto+Condensed:wght@300;400;700&display=swap');

    /* ─── Root Variables ─── */
    :root {
        --bg-primary: #0a0a0a;
        --bg-secondary: #1a1a2e;
        --bg-card: #16162a;
        --bg-card-hover: #1e1e3a;
        --shadow-gray: #2d2d44;
        --accent-yellow: #f1c40f;
        --accent-yellow-glow: rgba(241, 196, 15, 0.3);
        --accent-blue: #00aaff;
        --accent-blue-glow: rgba(0, 170, 255, 0.3);
        --accent-red: #c0392b;
        --accent-green: #2ecc71;
        --text-primary: #e0e0e0;
        --text-secondary: #888888;
        --border-dark: #333333;
    }

    /* ─── Global ─── */
    .stApp {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: 'Roboto Condensed', sans-serif !important;
    }

    /* ─── Hide Streamlit Defaults ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* ─── Sidebar ─── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 50%, #0d0d1a 100%) !important;
        border-right: 1px solid var(--border-dark) !important;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary) !important;
    }

    /* ─── Tabs ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: var(--bg-secondary) !important;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: var(--text-secondary) !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px 20px !important;
        font-family: 'Oswald', sans-serif !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--accent-yellow) !important;
        background-color: rgba(241, 196, 15, 0.1) !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(241, 196, 15, 0.15) !important;
        color: var(--accent-yellow) !important;
        box-shadow: 0 0 15px var(--accent-yellow-glow) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--accent-yellow) !important;
    }

    /* ─── Metric Cards & SVG Icons ─── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, var(--bg-card) 0%, var(--shadow-gray) 100%) !important;
        border: 1px solid var(--border-dark) !important;
        border-radius: 10px !important;
        padding: 15px 20px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4) !important;
        transition: all 0.3s ease !important;
        position: relative;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 0 20px var(--accent-blue-glow) !important;
        border-color: var(--accent-blue) !important;
        transform: translateY(-2px);
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-family: 'Roboto Condensed', sans-serif !important;
        text-transform: uppercase !important;
        font-size: 12px !important;
        letter-spacing: 1px !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--accent-yellow) !important;
        font-family: 'Oswald', sans-serif !important;
        font-weight: 600 !important;
    }

    /* ─── Buttons with SVG Shim ─── */
    .stButton > button {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--shadow-gray) 100%) !important;
        color: var(--accent-yellow) !important;
        border: 1px solid var(--accent-yellow) !important;
        border-radius: 8px !important;
        font-family: 'Oswald', sans-serif !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        padding: 10px 25px !important;
        transition: all 0.3s ease !important;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--accent-yellow) 0%, #e6b30e 100%) !important;
        color: #000 !important;
        box-shadow: 0 0 25px var(--accent-yellow-glow), 0 0 50px rgba(241, 196, 15, 0.15) !important;
        transform: translateY(-1px) !important;
    }

    /* ─── Bat-Signal Pulse Button ─── */
    .bat-signal-btn .stButton > button {
        animation: batPulse 2s ease-in-out infinite !important;
    }
    @keyframes batPulse {
        0%, 100% { box-shadow: 0 0 10px var(--accent-yellow-glow); }
        50% { box-shadow: 0 0 30px var(--accent-yellow-glow), 0 0 60px rgba(241, 196, 15, 0.2); }
    }

    /* ──– SVG Utilities ─── */
    .bat-icon {
        width: 24px;
        height: 24px;
        display: inline-block;
        vertical-align: middle;
        margin-right: 8px;
        filter: drop-shadow(0 0 2px var(--accent-yellow));
    }
    .bat-icon:hover {
        filter: drop-shadow(0 0 8px var(--accent-yellow));
        transform: scale(1.1);
        transition: transform 0.2s;
    }
    .bat-icon-lg {
        width: 64px;
        height: 64px;
        display: block;
        margin: 0 auto 10px;
        filter: drop-shadow(0 0 10px var(--accent-blue));
    }

    /* ─── Headers ─── */
    h1, h2, h3 {
        font-family: 'Oswald', sans-serif !important;
        color: var(--accent-yellow) !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
    }
    h1 { font-size: 2.2rem !important; }
    h2 { font-size: 1.6rem !important; color: var(--accent-blue) !important; }
    h3 { font-size: 1.2rem !important; }

    /* ─── Gotham Watermark ─── */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(ellipse at 50% 30%, rgba(26, 26, 46, 0.4) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
    }

    /* ─── Rain Overlay ─── */
    .stApp::after {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: repeating-linear-gradient(
            transparent,
            transparent 2px,
            rgba(0, 170, 255, 0.015) 2px,
            rgba(0, 170, 255, 0.015) 4px
        );
        animation: rainFall 0.8s linear infinite;
        pointer-events: none;
        z-index: 0;
    }
    @keyframes rainFall {
        0% { transform: translateY(0); }
        100% { transform: translateY(4px); }
    }
    
    /* ─── Gothic Card Container ─── */
    .gothic-card {
        background: linear-gradient(135deg, var(--bg-card) 0%, #12122a 100%);
        border: 1px solid var(--border-dark);
        border-radius: 12px;
        padding: 25px;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        transition: all 0.3s ease;
    }
    .gothic-card:hover {
        box-shadow: 0 0 25px var(--accent-blue-glow);
        border-color: var(--accent-blue);
    }
</style>
"""

st.markdown(GOTHAM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def load_config():
    """Load settings via the shared loader (config file + env-var secrets)."""
    try:
        from data_pipeline import load_config as _load
        return _load()
    except Exception:
        return {}

def fig_to_streamlit(fig):
    """Convert matplotlib figure to streamlit-compatible bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def create_price_chart(df, title="Price Chart"):
    """Create a styled price chart."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df["Close"], color="#00aaff", linewidth=1.2, label="Close")
    if "SMA_50" in df.columns:
        ax.plot(df.index, df["SMA_50"], color="#f1c40f", linewidth=0.8, alpha=0.7, label="SMA 50")
    if "SMA_200" in df.columns:
        ax.plot(df.index, df["SMA_200"], color="#c0392b", linewidth=0.8, alpha=0.7, label="SMA 200")
    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        ax.fill_between(df.index, df["BB_Lower"], df["BB_Upper"], alpha=0.08, color="#00aaff", label="BB")

    ax.set_title(title, fontsize=14, fontweight="bold", color="#f1c40f", fontfamily="sans-serif")
    ax.set_ylabel("Price", color="#e0e0e0")
    ax.legend(loc="upper left", facecolor="#1a1a2e", edgecolor="#333", labelcolor="#e0e0e0", fontsize=9)
    ax.set_facecolor("#0a0a0a")
    fig.patch.set_facecolor("#0a0a0a")
    ax.tick_params(colors="#e0e0e0")
    ax.xaxis.label.set_color("#e0e0e0")
    ax.yaxis.label.set_color("#e0e0e0")
    for spine in ax.spines.values():
        spine.set_color("#333333")
    plt.tight_layout()
    return fig

def render_svg(svg_string, width=24, height=24, class_name="bat-icon"):
    """Inject SVG directly via HTML."""
    return f'<div class="{class_name}" style="width:{width}px; height:{height}px;">{svg_string}</div>'

# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding:20px 0;">
        <div style="width:80px; height:80px; margin:0 auto; filter:drop-shadow(0 0 10px #f1c40f);">
            {BAT_LOGO_SVG}
        </div>
        <h1 style="font-size:1.8rem; margin:10px 0 5px; letter-spacing:4px;
                   color:#f1c40f; font-family:'Oswald',sans-serif;">
            BATMAN AI
        </h1>
        <p style="color:#888; font-size:13px; letter-spacing:2px;
                  font-family:'Roboto Condensed',sans-serif; text-transform:uppercase;">
            Trading System
        </p>
        <div style="width:60%; height:1px; background:linear-gradient(90deg,transparent,#f1c40f,transparent);
                    margin:15px auto;"></div>
    </div>
    """, unsafe_allow_html=True)

    config = load_config()
    _EMAIL_READY = all(config.get(k) for k in ("email_from", "email_to", "email_password"))

    ticker = st.text_input("🎯 Ticker Symbol", value=config.get("ticker", "AAPL"), help="e.g., AAPL, TSLA, BTC-USD")
    st.markdown("---")

    st.markdown(f"""
    <div style="background:#16162a; border-radius:8px; padding:15px; margin:10px 0; border:1px solid #333;">
        <div style="display:flex; align-items:center; margin-bottom:10px;">
            <div style="width:20px; margin-right:8px;">{BAT_GEAR_SVG}</div>
            <p style="color:#888; font-size:12px; text-transform:uppercase; margin:0;">System Config</p>
        </div>
        <p style="color:#e0e0e0; font-size:13px; margin:4px 0;">
            📊 Lookback: <span style="color:#00aaff;">{config.get("lookback", 60)}</span> days
        </p>
        <p style="color:#e0e0e0; font-size:13px; margin:4px 0;">
            💰 Account: <span style="color:#2ecc71;">£{config.get("account_balance", 1000)}</span>
        </p>
        <p style="color:#e0e0e0; font-size:13px; margin:4px 0;">
            ⚡ Risk: <span style="color:#f1c40f;">{config.get("risk_pct", 0.01)*100:.0f}%</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <p style="color:#555; font-size:11px; text-align:center; font-style:italic;">
        "It's not who I am underneath,<br>but what I do that defines me."
    </p>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# MAIN CONTENT — TABS
# ═══════════════════════════════════════════════════════════

st.warning(
    "Research demo, not financial advice. In honest walk-forward testing this model showed no "
    "edge over simply predicting \"down\" every day (balanced accuracy 50.4%). "
    "Details: github.com/GOUTHMNN/batman-ai-trading",
    icon="⚠️",
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "OVERVIEW", "TRAIN", "SIGNALS", "BACKTEST", "PATROL", "SCANNER"
])

# ... (Tabs 1-5 unmodified, skipped in this replacement block) ...
# Actually, I cannot easily skip intermediate content efficiently with replace_file_content if chunks are far apart.
# I will make TWO SEPARATE replace_file_content calls or one huge one.
# Given the tool supports multiple chunks, I will use that feature.


# ───────────────────────────────────────────────────────────
# TAB 1: GOTHAM OVERVIEW
# ───────────────────────────────────────────────────────────
with tab1:
    st.markdown(f"""
    <div style="text-align:center; padding-bottom:20px;">
        <div class="bat-icon-lg">{BAT_SIGNAL_SVG}</div>
        <h1 style="margin-bottom:5px;">Gotham Overview</h1>
        <p style="color:#888; letter-spacing:2px; text-transform:uppercase;">
            Market Intelligence • {ticker} • {datetime.now().strftime("%Y-%m-%d")}
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_fetch, col_status = st.columns([1, 3])
    with col_fetch:
        fetch_btn = st.button("⚡ SCAN MARKET DATA", use_container_width=True)

    if fetch_btn or st.session_state.get("data_fetched", False):
        try:
            from data_pipeline import fetch_data, engineer_features, handle_nans
            with st.spinner("🦇 Scanning Gotham markets..."):
                df = fetch_data(ticker, start=config.get("data_start", "2010-01-01"))
                df = engineer_features(df, config)
                df = handle_nans(df)
                st.session_state["overview_df"] = df
                st.session_state["data_fetched"] = True

            latest = df.iloc[-1]
            prev = df.iloc[-2]
            change = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Close", f"${latest['Close']:.2f}", f"{change:+.2f}%")
            with c2: st.metric("RSI 14", f"{latest['RSI_14']:.1f}")
            with c3: st.metric("MACD", f"{latest['MACD']:.3f}")
            with c4: st.metric("Volume", f"{latest['Volume']:,.0f}")

            st.markdown("---")
            chart_df = df.tail(504)
            fig = create_price_chart(chart_df, f"{ticker} — Last 2 Years")
            st.image(fig_to_streamlit(fig), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Failed to fetch data: {e}")

# ───────────────────────────────────────────────────────────
# TAB 2: TRAIN BATMAN
# ───────────────────────────────────────────────────────────
with tab2:
    st.markdown(f"""
    <div style="text-align:center; padding-bottom:20px;">
        <div class="bat-icon-lg" style="filter:drop-shadow(0 0 10px #00aaff);">{BAT_GEAR_SVG}</div>
        <h1 style="margin-bottom:5px;">Train Batman</h1>
        <p style="color:#888; letter-spacing:2px; text-transform:uppercase;">
            LSTM Neural Network Training
        </p>
    </div>
    """, unsafe_allow_html=True)

    model_path = os.path.join(BASE_DIR, config.get("model_path", "batman_lstm.h5"))
    model_exists = os.path.exists(model_path)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="gothic-card">
            <h3 style="color:#00aaff; margin-top:0;">Active Engine</h3>
            <p style="color:#e0e0e0;">Type: <span style="color:#f1c40f;">XGBoost (Gradient Boosting)</span></p>
            <p style="color:#888; font-size:12px;">Benchmark Acc: <span style="color:#2ecc71;">60.3%</span> (Target)</p>
            <p style="color:#555; font-size:10px;">*Training below creates a NEW LSTM model to challenge this benchmark.*</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="gothic-card">
            <h3 style="color:#00aaff; margin-top:0;">Config</h3>
            <p style="color:#e0e0e0;">Epochs: {config.get('epochs', 50)} | Batch: {config.get('batch_size', 32)}</p>
            <p style="color:#e0e0e0;">LR: {config.get('learning_rate', 0.0008)}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="bat-signal-btn">', unsafe_allow_html=True)
    train_btn = st.button("🦇 EXECUTE TRAINING PROTOCOL", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if train_btn:
        try:
            from model_core import run_training, plot_loss
            with st.spinner("🏋️ The Dark Knight is training..."):
                result = run_training(ticker=ticker, config=config)
                st.session_state["train_history"] = result["history"]
                st.session_state["eval_results"] = result["eval_results"]

            eval_r = result["eval_results"]
            c1, c2 = st.columns(2)
            c1.metric("Validation Accuracy", f"{eval_r['accuracy']:.1%}")
            c2.metric("F1 Score", f"{eval_r['f1']:.3f}")

            st.markdown("### 📉 Training Loss")
            st.image(fig_to_streamlit(plot_loss(result["history"])), use_container_width=True)
            st.success("🦇 The Dark Knight is trained and ready for battle!")

        except Exception as e:
            st.error(f"❌ Training failed: {e}")

# ───────────────────────────────────────────────────────────
# TAB 3: SIGNALS & ALERTS
# ───────────────────────────────────────────────────────────
with tab3:
    st.markdown(f"""
    <div style="text-align:center; padding-bottom:20px;">
        <div class="bat-icon-lg">{BAT_HEAD_SVG}</div>
        <h1 style="margin-bottom:5px;">Signals & Alerts</h1>
        <p style="color:#888; letter-spacing:2px; text-transform:uppercase;">
            AI-Powered Trading Signals
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="bat-signal-btn">', unsafe_allow_html=True)
    predict_btn = st.button("🦇 ACTIVATE BAT-SIGNAL", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn:
        try:
            from signals_risk import run_signal_pipeline
            with st.spinner("🦇 Scanning Gotham for signals..."):
                st.session_state["signal_result"] = run_signal_pipeline(ticker=ticker, config=config)
        except Exception as e:
            st.error(f"❌ Signal failed: {e}")

    if "signal_result" in st.session_state:
        res = st.session_state["signal_result"]
        sig = res["signal"]
        
        # Determine SVG
        sig_svg = SIGNAL_HOLD_SVG
        color = "#f1c40f"
        if sig == "BUY":
            sig_svg = SIGNAL_BUY_SVG
            color = "#2ecc71"
        elif sig == "SELL":
            sig_svg = SIGNAL_SELL_SVG
            color = "#c0392b"

        st.markdown(f"""
        <div style="border:2px solid {color}; border-radius:12px; padding:20px; text-align:center;
                    background:linear-gradient(135deg, {color}22 0%, {color}05 100%);
                    box-shadow:0 0 30px {color}44;">
            <div style="width:80px; height:80px; margin:0 auto;">{sig_svg}</div>
            <h1 style="color:{color}; font-size:3.5rem; margin:10px 0;">{sig}</h1>
            <p style="color:#888;">{res.get('ticker', ticker)} • Confidence: {res.get('confidence', 0)*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Current", f"${res['current_close']:.2f}")
        c2.metric("Predicted", f"${res['predicted_close']:.2f}", f"{res.get('pct_change', 0):+.2f}%")
        c3.metric("Position Size", f"£{res.get('position_value', 0):.2f}")

# ───────────────────────────────────────────────────────────
# TAB 4: BACKTEST BATCAVE
# ───────────────────────────────────────────────────────────
with tab4:
    st.markdown(f"""
    <div style="text-align:center; padding-bottom:20px;">
        <div class="bat-icon-lg">{BAT_CHART_SVG}</div>
        <h1 style="margin-bottom:5px;">Backtest Batcave</h1>
    </div>
    """, unsafe_allow_html=True)

    backtest_btn = st.button("🦇 RUN SIMULATION", use_container_width=True)

    if backtest_btn:
        try:
            from backtest_engine import run_full_backtest
            with st.spinner("🦇 Simulating trades..."):
                bt_r, metrics, bt_fig = run_full_backtest(config=config)
                st.session_state["backtest_metrics"] = metrics
                st.session_state["backtest_result"] = bt_r
        except Exception as e:
            st.error(f"❌ Backtest failed: {e}")

    if "backtest_metrics" in st.session_state:
        m = st.session_state["backtest_metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.3f}")
        c2.metric("Total Return", f"{m['total_return_pct']:+.2f}%")
        c3.metric("Max Drawdown", f"{m['max_drawdown_pct']:.2f}%")
        c4.metric("Win Rate", f"{m['win_rate_pct']:.1f}%")

        path = os.path.join(BASE_DIR, config.get("backtest_plot_path", "backtest_plot.png"))
        if os.path.exists(path):
            st.image(path, use_container_width=True)

# ───────────────────────────────────────────────────────────
# TAB 5: NIGHT PATROL
# ───────────────────────────────────────────────────────────
with tab5:
    st.markdown(f"""
    <div style="text-align:center; padding-bottom:20px;">
        <div class="bat-icon-lg">{BAT_CLOCK_SVG}</div>
        <h1 style="margin-bottom:5px;">Night Patrol</h1>
        <p style="color:#888;">Automated Execution</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="gothic-card">
            <div style="width:24px; float:left; margin-right:10px;">{BAT_CLOCK_SVG}</div>
            <h3 style="color:#00aaff; margin-top:0;">Scheduler</h3>
            <p style="color:#e0e0e0;">Execute at <span style="color:#f1c40f;">{config.get("schedule_time", "03:00")} GMT</span></p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="gothic-card">
             <div style="width:24px; float:left; margin-right:10px;">{BAT_MAIL_SVG}</div>
            <h3 style="color:#00aaff; margin-top:0;">Alerts</h3>
            <p style="color:#e0e0e0;">Email alerts: {"configured" if _EMAIL_READY else "not configured on this server"}</p>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🦇 RUN MANUAL PATROL (Execute Pipeline)", disabled=not _EMAIL_READY,
                 help=None if _EMAIL_READY else "Needs BATMAN_EMAIL_* environment variables; disabled on the public demo."):
         try:
             from deploy_night_patrol import daily_patrol
             with st.spinner("🌙 Executing patrol..."):
                 daily_patrol(config=config)
             st.success("✅ Patrol complete. Check email.")
         except Exception as e:
             st.error(f"❌ Failed: {e}")

# ───────────────────────────────────────────────────────────
# TAB 6: USAS SCANNER
# ───────────────────────────────────────────────────────────
with tab6:
    st.markdown(f"""
    <div style="text-align:center; padding-bottom:20px;">
        <div class="bat-icon-lg">{BAT_SIGNAL_SVG}</div>
        <h1 style="margin-bottom:5px;">USAS Scanner</h1>
        <p style="color:#888;">Walk-Forward Validation (1 Year) • High Precision</p>
    </div>
    """, unsafe_allow_html=True)

    c_sel, c_btn = st.columns([1, 2])
    with c_sel:
        scan_ticker = st.selectbox("Ticker", ["USAS", "AAPL", "TSLA", "BTC-USD"], index=0)
    
    with c_btn:
        st.write("") # spacer
        st.write("")
        scan_btn = st.button("🦇 SCAN LAST YEAR (WALK-FORWARD)", use_container_width=True)

    if scan_btn:
        try:
            from model_core import walk_forward_validation
            # from backtest_engine import run_walk_forward_backtest # Binary backtest not strictly same
            
            with st.spinner(f"🦇 Running Binary Walk-Forward on {scan_ticker} (2025-Now)..."):
                config = load_config()
                # Run WF
                res_df, acc, _ = walk_forward_validation(ticker=scan_ticker, config=config)
                st.session_state["wf_results"] = res_df
                st.session_state["wf_acc"] = acc
                
        except Exception as e:
            st.error(f"❌ Scan Failed: {e}")

    if "wf_results" in st.session_state:
        res_df = st.session_state["wf_results"]
        acc = st.session_state["wf_acc"]
        
        target_acc = config.get("min_accuracy_target", 0.60)
        
        # Metrics
        m1, m2 = st.columns(2)
        m1.metric("Directional Accuracy", f"{acc:.1%}", 
                 delta="TARGET DESTROYED" if acc >= target_acc else "BELOW TARGET",
                 delta_color="normal" if acc >= target_acc else "inverse")
        m2.metric("Total Predictions", len(res_df))

        # Alert
        if acc < target_acc:
            st.error(f"⚠️ Accuracy {acc:.1%} < {target_acc:.1%}: Failed. Shadows win.")
        else:
            st.success("✅ Model is LETHAL. Market is clear.")

        # Plot Rolling Accuracy (The Yellow Line)
        st.markdown("### 📈 Rolling 50-Day Accuracy")
        if "RollingAcc" in res_df.columns:
            roll_acc = res_df["RollingAcc"]
            
            fig_acc, ax_acc = plt.subplots(figsize=(10, 4))
            ax_acc.plot(res_df.index, roll_acc, color="#f1c40f", linewidth=2.5, label="Bat-Signal (50d)")
            ax_acc.axhline(y=target_acc, color="#c0392b", linestyle="--", alpha=0.8, label="Kill Switch (60%)")
            
            # Styling
            ax_acc.set_title("Rolling 50-Day Accuracy Curve", color="#e0e0e0")
            ax_acc.set_facecolor("#0a0a0a")
            fig_acc.patch.set_facecolor("#0a0a0a")
            ax_acc.tick_params(colors="#e0e0e0")
            for spine in ax_acc.spines.values(): spine.set_color("#333")
            plt.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="#e0e0e0")
            st.pyplot(fig_acc)

        # Data Table
        st.markdown("### 📋 Prediction Log")
        # Show Prob, Target, Hit
        st.dataframe(res_df[["Close", "Probability", "Target", "Hit", "RollingAcc"]].style.format({
            "Close": "${:.2f}", 
            "Probability": "{:.1%}",
            "RollingAcc": "{:.1%}"
        }).background_gradient(subset=["Probability"], cmap="gray"))

st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:30px 0;">
    <p style="color:#444; font-size:10px; font-style:italic;">
        "The night is darkest just before the dawn."
    </p>
</div>
""", unsafe_allow_html=True)
