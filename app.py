"""
app.py
------
PE + Delivery Valuation Signal Framework -- Streamlit app.
"""

import pandas as pd
import streamlit as st
import requests
import io
import time

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from data_pipeline import extract_annual_fundamentals, build_step_function_pe, apply_corporate_action_exclusions, fetch_price_volume, fetch_eod2_delivery_data
from technical_indicators import add_all_technical_indicators
from pe_signal import pe_percentile_rank, delivery_percentile_rank
from vadm import delivery_flow_strength, compute_vadm, classify_quadrant, generate_quadrant_signals, generate_quadrant_ablation_signals, signal_onsets
from backtest import run_backtest, run_ablation_backtest, test_h3_interaction


APP_NAME = "VADM TERMINAL"

# Color palette -- deliberately softer than pure neon-on-black. Same dark,
# sharp, monospace terminal FEEL, but desaturated enough to not strain the
# eyes on a long session. Used consistently across CSS and every chart.
BG = "#0B0E14"
PANEL_BG = "#11151C"
BORDER = "#242B38"
GRID = "#1C212B"
TEXT = "#E6EDF3"
TEXT_MUTED = "#8B949E"
ACCENT = "#D4A017"
GREEN = "#3FB950"
RED = "#F85149"
CAUTION = "#D29922"
MUTED_GRAY = "#6E7681"

# Must be the first Streamlit command
st.set_page_config(page_title=f"{APP_NAME}", layout="wide", page_icon=":chart_with_upwards_trend:")

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');
    p, div, h1, h2, h3, h4, h5, h6, label, input, button, li {{
        font-family: 'IBM Plex Mono', 'Consolas', 'Courier New', monospace !important;
    }}
    span.material-symbols-rounded, span.material-icons, .stIcon {{
        font-family: 'Material Symbols Rounded' !important;
    }}
    .stApp {{
        background-color: {BG};
        color: {TEXT};
    }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .block-container {{
        padding-top: 1.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }}
    div[data-testid="metric-container"] {{
        background-color: {PANEL_BG};
        border: 1px solid {BORDER};
        border-radius: 3px;
        padding: 10px 15px;
        border-top: 2px solid {ACCENT};
    }}
    div[data-testid="stMetricValue"] > div {{
        color: {TEXT} !important;
        font-size: 1.6rem !important;
    }}
    div[data-testid="stMetricLabel"] > div > div > p {{
        color: {ACCENT} !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.78rem !important;
        letter-spacing: 0.04em;
    }}
    div.stSelectbox > div > div, input {{
        background-color: {PANEL_BG} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 3px !important;
    }}
    div.stButton > button[kind="primary"] {{
        background-color: {PANEL_BG};
        color: {GREEN};
        border: 1px solid {GREEN};
        border-radius: 3px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        height: 100%;
        margin-top: 15px;
        padding: 18px 0px;
        transition: background-color 0.15s ease;
    }}
    div.stButton > button[kind="primary"]:hover {{
        background-color: {GREEN};
        color: {BG};
    }}
    .st-expander, .stPopover {{
        border-color: {BORDER} !important;
        border-radius: 3px !important;
        background-color: {PANEL_BG} !important;
    }}
    h1, h2, h3, h4, h5, h6, p {{
        color: {TEXT};
    }}
    hr {{
        border-color: {BORDER};
    }}
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: {BG}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}
    </style>
""", unsafe_allow_html=True)




def parse_exclusions(text):
    if not text or not text.strip():
        return []
    windows = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            start, end = chunk.split(":")
            windows.append((start.strip(), end.strip()))
        except ValueError:
            st.warning(f"Could not parse exclusion window '{chunk}'")
    return windows


@st.cache_data(ttl=86400, show_spinner=False)
def get_all_nse_stocks():
    try:
        url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        res = requests.get(url, headers=headers, timeout=10)
        df = pd.read_csv(io.StringIO(res.text))
        mapping = {}
        for _, row in df.iterrows():
            mapping[f"{row['NAME OF COMPANY']} ({row['SYMBOL']})"] = f"{row['SYMBOL']}.NS"
        return mapping
    except Exception:
        return {
            "HDFC Bank Limited (HDFCBANK)": "HDFCBANK.NS",
            "Reliance Industries (RELIANCE)": "RELIANCE.NS",
            "TCS Limited (TCS)": "TCS.NS"
        }


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def cached_price_volume(ticker, start, end):
    return fetch_price_volume(ticker, start, end)


def run_single_stock(ticker, xlsx_file, exclusions_text, params):
    annual = extract_annual_fundamentals(xlsx_file)

    # --- Single data fetch: EOD2 gives price + volume + delivery together ---
    # (yfinance removed from the primary path -- it was hitting
    # YFRateLimitError on Streamlit Cloud's shared IPs, a known problem with
    # Yahoo Finance blocking cloud-hosted scrapers. EOD2's data is static
    # files served over GitHub's CDN, no rate limiting like this.)
    nse_symbol = ticker.replace(".NS", "").replace(".BO", "")
    eod2_df, eod2_err = fetch_eod2_delivery_data(nse_symbol)
    if eod2_err:
        return None, f"Price/delivery data unavailable for {nse_symbol}: {eod2_err}"

    price_start = params["price_start"]
    price_df = eod2_df.loc[eod2_df.index >= price_start, ["price", "volume"]]
    if price_df.empty:
        return None, f"No price data for {nse_symbol} from {price_start} onward."

    pe_df = build_step_function_pe(annual, price_df["price"], filing_lag_days=params["filing_lag_days"])
    exclusions = parse_exclusions(exclusions_text)
    pe_df = apply_corporate_action_exclusions(pe_df, exclusions)

    merged = pe_df.join(price_df[["volume"]], how="left")
    merged = add_all_technical_indicators(
        merged,
        volume_window=params["volume_window"],
        momentum_window=params["momentum_window"],
        rsi_window=params["rsi_window"],
        dma_short=params["dma_short"],
        dma_long=params["dma_long"],
    )
    merged["pe_percentile"] = pe_percentile_rank(merged["pe"], min_periods=params["pe_min_periods"])

    # --- Delivery / VADM / Quadrant layer (sir's redesigned framework) ---
    # Same eod2_df already fetched above -- just join the delivery_pct column,
    # no second network call needed.
    merged = merged.join(eod2_df[["delivery_pct"]], how="left")
    merged["delivery_percentile"] = delivery_percentile_rank(merged["delivery_pct"], min_periods=params["pe_min_periods"])
    merged["quadrant"] = classify_quadrant(merged["pe_percentile"], merged["delivery_percentile"])
    merged["delivery_flow"] = delivery_flow_strength(merged["delivery_percentile"])
    merged["vadm"] = compute_vadm(merged["pe_percentile"], merged["delivery_flow"])
    n_bad_delivery = eod2_df.attrs.get("n_impossible_delivery_pct_rows_removed", 0)

    # VADM IS the operational signal now (quadrant is the visual 4-state view
    # of the same underlying PE-percentile + delivery-flow combination)
    merged = generate_quadrant_signals(
        merged,
        vadm_buy_pctile=params["vadm_buy_pctile"],
        vadm_sell_pctile=params["vadm_sell_pctile"],
    )
    merged = generate_quadrant_ablation_signals(
        merged,
        cheap_pctile=params["cheap_pctile"],
        delivery_pctile_midpoint=0.5,
    )

    # --- Statistical evaluation uses EPISODE ONSETS, not raw signal-days ---
    # Confirmed on real data: a persistent regime can hold for weeks (one
    # sell episode ran 29 consecutive days), and counting every day as an
    # independent observation inflates apparent significance. The CHART
    # still shows every day (merged, unchanged); backtest/ablation/stats
    # use eval_df, where each signal column is collapsed to onset-only.
    eval_df = merged.copy()
    for col in ["heavy_buying", "heavy_selling", "buy_pe_only", "buy_delivery_only",
                "buy_combined", "sell_pe_only", "sell_delivery_only", "sell_combined"]:
        eval_df[col] = signal_onsets(merged[col])

    n_buy_episodes = int(eval_df["heavy_buying"].sum())
    n_sell_episodes = int(eval_df["heavy_selling"].sum())
    n_buy_raw_days = int(merged["heavy_buying"].sum())
    n_sell_raw_days = int(merged["heavy_selling"].sum())

    results = run_backtest(eval_df, holding_periods=params["holding_periods"])
    ablation_results = run_ablation_backtest(eval_df, holding_periods=params["holding_periods"])

    h3_model, h3_err = test_h3_interaction(merged, holding_period=params.get("h3_holding_period", 63))
    return {
        "annual": annual, "merged": merged, "results": results,
        "ablation_results": ablation_results,
        "h3_model": h3_model, "h3_err": h3_err,
        "n_bad_delivery": n_bad_delivery,
        "n_buy_episodes": n_buy_episodes, "n_sell_episodes": n_sell_episodes,
        "n_buy_raw_days": n_buy_raw_days, "n_sell_raw_days": n_sell_raw_days,
    }, None

def render_stock_chart(ticker, merged):
    if go is None: return
    fig = go.Figure()

    # 1. PROFESSIONAL CANDLESTICK CHART
    # Check if OHLC data exists, fallback to line if it doesn't
    if "open" in merged.columns:
        fig.add_trace(go.Candlestick(
            x=merged.index,
            open=merged["open"], high=merged["high"],
            low=merged["low"], close=merged["price"],
            name="OHLC",
            increasing_line_color="#3FB950", decreasing_line_color="#F85149",
            increasing_fillcolor="#3FB950", decreasing_fillcolor="#F85149"
        ))
    else:
        fig.add_trace(go.Scatter(x=merged.index, y=merged["price"], name="PRICE", line=dict(color="#E6EDF3")))

    # 2. EXACT ENTRY & EXIT LOGIC (Edge Detection / First Day Only)
    is_buy_zone = merged["quadrant"] == "Value Confirmation"
    is_sell_zone = merged["quadrant"] == "Reversal/De-rating"
    is_value_trap = merged["quadrant"] == "Value Trap Warning"
    
    # Fresh Entries (Kal nahi tha, aaj hai)
    entry_long = is_buy_zone & (~is_buy_zone.shift(1).fillna(False))
    entry_short = is_sell_zone & (~is_sell_zone.shift(1).fillna(False))
    
    # Exits (Long exit kab karein? Jab stock Value Trap ya Reversal mein gir jaye)
    exit_zone = is_sell_zone | is_value_trap
    exit_long = exit_zone & (~exit_zone.shift(1).fillna(False)) & is_buy_zone.shift(1).fillna(False).rolling(30).max().astype(bool) # Exit if we were recently long

    df_entry_long = merged[entry_long]
    df_entry_short = merged[entry_short]
    df_exit_long = merged[exit_long]

    # 3. PLOT ENTRY / EXIT MARKERS WITH TEXT ON EXACT CANDLES
    # Entry Long (Below Candle)
    if not df_entry_long.empty:
        y_pos = df_entry_long["low"] * 0.95 if "low" in df_entry_long else df_entry_long["price"] * 0.95
        fig.add_trace(go.Scatter(
            x=df_entry_long.index, y=y_pos, mode="markers+text", name="ENTRY LONG",
            text="⬆ ENTRY LONG", textposition="bottom center",
            marker=dict(color="#3FB950", size=12, symbol="triangle-up", line=dict(color="#FFFFFF", width=1)),
            textfont=dict(color="#3FB950", size=11, family="IBM Plex Mono", weight="bold")
        ))

    # Entry Short (Above Candle)
    if not df_entry_short.empty:
        y_pos = df_entry_short["high"] * 1.05 if "high" in df_entry_short else df_entry_short["price"] * 1.05
        fig.add_trace(go.Scatter(
            x=df_entry_short.index, y=y_pos, mode="markers+text", name="ENTRY SHORT",
            text="⬇ ENTRY SHORT", textposition="top center",
            marker=dict(color="#F85149", size=12, symbol="triangle-down", line=dict(color="#FFFFFF", width=1)),
            textfont=dict(color="#F85149", size=11, family="IBM Plex Mono", weight="bold")
        ))
        
    # Exit Long (Above Candle)
    if not df_exit_long.empty:
        y_pos = df_exit_long["high"] * 1.05 if "high" in df_exit_long else df_exit_long["price"] * 1.05
        fig.add_trace(go.Scatter(
            x=df_exit_long.index, y=y_pos, mode="markers+text", name="EXIT LONG",
            text="✖ EXIT LONG", textposition="top center",
            marker=dict(color="#D4A017", size=10, symbol="x", line=dict(color="#FFFFFF", width=1)),
            textfont=dict(color="#D4A017", size=10, family="IBM Plex Mono")
        ))

    # Remove Range Slider from bottom (Makes chart look much cleaner)
    fig.update_layout(xaxis_rangeslider_visible=False)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0B0E14',
        plot_bgcolor='#0B0E14',
        title=dict(text=f"> {ticker} : CANDLESTICK SIGNAL ANALYSIS", font=dict(size=14, color="#D4A017", family="IBM Plex Mono, monospace")),
        height=650, # Increased height for better candle visibility
        yaxis=dict(title="PRICE (INR)", showgrid=True, gridcolor="#1C212B", zeroline=False, tickfont=dict(color="#E6EDF3", family="IBM Plex Mono, monospace")),
        xaxis=dict(tickfont=dict(color="#E6EDF3", family="IBM Plex Mono, monospace"), gridcolor="#1C212B"),
        legend=dict(orientation="h", y=1.05, x=0, bgcolor='#0B0E14', font=dict(size=11, color="#E6EDF3", family="IBM Plex Mono, monospace")),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

def render_ablation_table(ablation_results, side):
    labels = {
        f"{side}_pe_only": "PE ONLY (H1)",
        f"{side}_delivery_only": "DELIVERY ONLY (H2)",
        f"{side}_combined": "COMBINED / VADM (H3)",
    }
    rows = []
    for key, label in labels.items():
        df = ablation_results.get(key)
        if df is None or df.empty: continue
        row = df[df["holding_period"] == "fwd_ret_63d"]
        if row.empty: continue
        row = row.iloc[0]

        pval = row['p_value']
        if pd.notnull(pval):
            pval_str = "<0.001" if pval < 0.001 else f"{pval:.4f}"
        else:
            pval_str = "-"

        rows.append({
            "TYPE": label,
            "SIG": row["n_signals"],
            "RET(63D)": f"{row['avg_signal_return']:.2%}" if pd.notnull(row['avg_signal_return']) else "-",
            "WIN(%)": f"{row['win_rate']:.2%}" if pd.notnull(row['win_rate']) else "-",
            "P-VAL": pval_str,
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


QUADRANT_COLORS = {
    "Value Confirmation": "#3FB950",
    "Reversal/De-rating": "#F85149",
    "Momentum without Margin of Safety": "#D4A017",
    "Value Trap Warning": "#6E7681",
}


def render_quadrant_chart(merged, quadrant_col="quadrant"):
    """
    Scatter of PE percentile (x) vs Delivery percentile (y), colored by the
    4-state quadrant classification, with the 50/50 dividing lines drawn in.
    Requires merged to already have 'pe_percentile', 'delivery_percentile',
    and quadrant_col columns (from vadm.classify_quadrant).
    """
    if go is None:
        return
    plot_df = merged.dropna(subset=["pe_percentile", "delivery_percentile", quadrant_col])
    if plot_df.empty:
        st.info("[SYSTEM] No valid quadrant data yet -- check PE/delivery percentile warm-up period.")
        return

    fig = go.Figure()
    for label, color in QUADRANT_COLORS.items():
        subset = plot_df[plot_df[quadrant_col] == label]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["pe_percentile"] * 100, y=subset["delivery_percentile"] * 100,
            mode="markers", name=label.upper(),
            marker=dict(color=color, size=5, opacity=0.55),
        ))

    # Most recent day, highlighted
    last = plot_df.iloc[[-1]]
    fig.add_trace(go.Scatter(
        x=last["pe_percentile"] * 100, y=last["delivery_percentile"] * 100,
        mode="markers", name="LATEST",
        marker=dict(color="#E6EDF3", size=14, symbol="star", line=dict(color="#0B0E14", width=1)),
    ))

    fig.add_vline(x=50, line_dash="dot", line_color="#30363D")
    fig.add_hline(y=50, line_dash="dot", line_color="#30363D")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0E14", plot_bgcolor="#0B0E14",
        title=dict(text="> VALUATION x DELIVERY QUADRANT", font=dict(size=14, color="#D4A017", family="IBM Plex Mono, monospace")),
        height=480,
        xaxis=dict(title="PE PERCENTILE (own history) -->", range=[0, 100], gridcolor="#1C212B", tickfont=dict(color="#E6EDF3", family="IBM Plex Mono, monospace")),
        yaxis=dict(title="DELIVERY PERCENTILE (own history) -->", range=[0, 100], gridcolor="#1C212B", tickfont=dict(color="#E6EDF3", family="IBM Plex Mono, monospace")),
        legend=dict(orientation="h", y=-0.15, font=dict(size=10, color="#E6EDF3", family="IBM Plex Mono, monospace")),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    dist = plot_df[quadrant_col].value_counts()
    cols = st.columns(4)
    for i, label in enumerate(QUADRANT_COLORS.keys()):
        cols[i].metric(label.upper()[:18], int(dist.get(label, 0)))


def render_hypothesis_panel():
    """Static display of H1-H4 so they're visible to the user/report reviewer,
    not just buried in code comments."""
    st.markdown("<span style='color:#D4A017; font-family:monospace; border-bottom: 1px solid #D4A017;'>[ RESEARCH HYPOTHESES ]</span>", unsafe_allow_html=True)
    hyps = [
        ("H1", "Main effect - Valuation (P/E)",
         "Low P/E is associated with higher average forward return than high P/E."),
        ("H2", "Main effect - Flow (Deliverables)",
         "Rising delivery-based buying pressure is associated with higher average "
         "forward return than falling delivery-based buying pressure."),
        ("H3", "Interaction - CORE HYPOTHESIS",
         "The effect of delivery-based buying pressure on forward return is NOT "
         "uniform across the P/E valuation regime (i.e. flow's predictive power "
         "differs between cheap and expensive stocks)."),
        ("H4", "Composite indicator",
         "VADM (Valuation Adjusted Delivery Momentum) = f(PE relative percentile, "
         "Delivery flow) -- see vadm.py; formula is a proposed implementation, "
         "not yet confirmed as final."),
    ]
    for tag, title, text in hyps:
        st.markdown(
            f"<div style='margin-bottom:10px; font-family:monospace; font-size:13px;'>"
            f"<span style='color:#3FB950; font-weight:700;'>{tag}</span> "
            f"<span style='color:#D4A017;'>[{title}]</span><br>"
            f"<span style='color:#ADB7C4;'>{text}</span></div>",
            unsafe_allow_html=True,
        )
    st.caption(
        "H3's formal test is the interaction regression below (forward_return ~ "
        "pe_percentile + delivery_flow + pe_percentile*delivery_flow). The "
        "quadrant chart above is a visual/exploratory view of the same idea, "
        "not a substitute for the regression."
    )


def render_h3_results(h3_model, h3_err, holding_period):
    st.markdown("<span style='color:#D4A017; font-family:monospace; border-bottom: 1px solid #D4A017;'>[ H3 INTERACTION REGRESSION ]</span>", unsafe_allow_html=True)
    st.caption(f"forward_return ({holding_period}d) ~ pe_percentile + delivery_flow + pe_percentile*delivery_flow")

    if h3_err:
        st.warning(f"[H3] {h3_err}")
        return

    params_ = h3_model.params
    pvalues = h3_model.pvalues
    bse = h3_model.bse

    rows = []
    for name in ["const", "pe_percentile", "delivery_flow", "interaction"]:
        if name not in params_.index:
            continue
        p = pvalues[name]
        rows.append({
            "TERM": name.upper(),
            "COEF": f"{params_[name]:+.4f}",
            "STD ERR": f"{bse[name]:.4f}",
            "P-VALUE": "<0.001" if p < 0.001 else f"{p:.4f}",
            "SIGNIFICANT (5%)": "YES" if p < 0.05 else "no",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    interaction_p = pvalues.get("interaction", None)
    n_obs = int(h3_model.nobs)
    r2 = h3_model.rsquared
    if interaction_p is not None:
        if interaction_p < 0.05:
            verdict = "SUPPORTS H3 -- interaction term IS statistically significant. Delivery flow's predictive effect on forward return genuinely differs by PE regime."
            color = "#3FB950"
        else:
            verdict = "DOES NOT SUPPORT H3 at 5% level -- interaction term is not statistically significant with this sample. H1/H2 may still hold individually; the interaction itself isn't confirmed here."
            color = "#F85149"
        st.markdown(
            f"<div style='font-family:monospace; font-size:13px; color:{color}; margin-top:8px;'>{verdict}</div>",
            unsafe_allow_html=True,
        )
    st.caption(f"n = {n_obs} observations, R-squared = {r2:.4f}. Note: this is a single-stock regression -- for a real conclusion on H3, run this across your full stock universe and look at how consistently the interaction term comes out significant, not just one stock.")


def format_terminal_df(df):
    """Formats raw backtest DataFrame into a clean Terminal-style display."""
    if df is None or df.empty: return df
    out = df.copy()
    out = out.rename(columns={
        "holding_period": "PERIOD",
        "n_signals": "SIG_CNT",
        "avg_signal_return": "AVG_RET",
        "avg_baseline_return": "BASE_RET",
        "win_rate": "WIN_RATE",
        "t_stat": "T_STAT",
        "p_value": "P_VAL"
    })
    for col in ["AVG_RET", "BASE_RET", "WIN_RATE"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "-")
    if "T_STAT" in out.columns:
        out["T_STAT"] = out["T_STAT"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
    if "P_VAL" in out.columns:
        out["P_VAL"] = out["P_VAL"].apply(lambda x: "<0.001" if pd.notnull(x) and x < 0.001 else (f"{x:.4f}" if pd.notnull(x) else "-"))
    if "note" in out.columns:
        out = out.drop(columns=["note"])
    return out

ACCESS_CODE = "Navi@123"

BOOT_SEQUENCE = [
    "AUTHENTICATING OPERATOR...",
    "CALIBRATING VALUATION ENGINE...",
    "SYNCING DELIVERY FLOW DATA...",
    "COMPILING QUADRANT MATRIX...",
    "WELCOME BACK, NAVEEN.",
]


def check_access() -> bool:
    """Simple session-gated password check with a short boot-sequence
    animation on success. Not real security (client-side, single shared
    password) -- just keeps the terminal from being casually stumbled into."""
    if st.session_state.get("authenticated"):
        return True

    st.markdown(
        f"<div style='text-align:center; margin-top:12vh;'>"
        f"<span style='color:{ACCENT}; font-family:monospace; font-size:1.4rem; letter-spacing:0.15em;'>[ {APP_NAME} ]</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        code = st.text_input("ACCESS CODE", type="password", key="access_code_input", label_visibility="collapsed", placeholder="ACCESS CODE")
        submit = st.button("AUTHENTICATE", use_container_width=True)

        if submit:
            if code == ACCESS_CODE:
                st.session_state["authenticated"] = True
                placeholder = st.empty()
                for line in BOOT_SEQUENCE:
                    placeholder.markdown(
                        f"<div style='font-family:monospace; color:{GREEN}; font-size:0.95rem; text-align:center;'>{line}</div>",
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.45)
                st.rerun()
            else:
                st.markdown(
                    f"<div style='color:{RED}; font-family:monospace; text-align:center; font-size:0.85rem;'>[ACCESS DENIED] Incorrect code.</div>",
                    unsafe_allow_html=True,
                )
    return False


def main():
    st.markdown(f"<h3 style='color: {ACCENT}; font-family: monospace; font-size: 1.2rem; letter-spacing:0.08em;'>[ {APP_NAME} ] VALUATION x DELIVERY SIGNAL ENGINE</h3>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([3, 3, 2, 2])

    with c1:
        TICKER_MAPPING = get_all_nse_stocks()
        company_name = st.selectbox("TICKER", options=list(TICKER_MAPPING.keys()), index=None, placeholder="<SEARCH TICKER>")
        ticker = TICKER_MAPPING.get(company_name) if company_name else ""

    with c2:
        xlsx_file = st.file_uploader("DATA (.XLSX)", type=["xlsx"])

    with c3:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        with st.popover("[ STRAT PARAMS ]", use_container_width=True):
            st.markdown("<span style='color:#D4A017'>VALUATION BOUNDS</span>", unsafe_allow_html=True)
            cheap_pctile = st.slider("CHEAP PE %", 0.05, 0.50, 0.20, 0.05)
            expensive_pctile = st.slider("EXPENSIVE PE %", 0.50, 0.95, 0.80, 0.05)

            st.markdown("<span style='color:#D4A017'>VADM SIGNAL (operational trigger)</span>", unsafe_allow_html=True)
            vadm_buy_pctile = st.slider("VADM BUY -- TOP % OF OWN HISTORY", 0.70, 0.99, 0.90, 0.01,
                                         help="Heavy buying = VADM in the top X% of this stock's own VADM history. NOT a fixed number -- a fixed threshold was tested and found to fire on 55%+ of days, which isn't 'heavy' at all.")
            vadm_sell_pctile = st.slider("VADM SELL -- BOTTOM % OF OWN HISTORY", 0.01, 0.30, 0.10, 0.01,
                                          help="Heavy selling = VADM in the bottom X% of its own history.")
            h3_holding_period = st.selectbox("H3 REGRESSION HOLDING (DAYS)", [21, 63, 126, 252], index=1)

            st.markdown("<span style='color:#D4A017'>ADVANCED ENGINE</span>", unsafe_allow_html=True)
            volume_window = st.number_input("VOL WINDOW", value=20)
            momentum_window = st.number_input("MOM WINDOW", value=10)
            rsi_window = st.number_input("RSI WINDOW", value=14)
            dma_short = st.number_input("SHORT DMA", value=50)
            dma_long = st.number_input("LONG DMA", value=200)
            filing_lag_days = st.number_input("FILING LAG", value=60)
            pe_min_periods = st.number_input("MIN PE HIST.", value=252)
            price_start = st.text_input("START DATE", value="2016-01-01")
            holding_periods_text = st.text_input("HOLDING (DAYS)", value="21,63,126,252")
            exclusions_text = st.text_input("EXCLUSIONS", help="YYYY-MM-DD:YYYY-MM-DD")

    with c4:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        run_btn = st.button("< EXECUTE >", type="primary", use_container_width=True)

    holding_periods = tuple(int(x.strip()) for x in holding_periods_text.split(",") if x.strip())

    params = dict(
        cheap_pctile=cheap_pctile, expensive_pctile=expensive_pctile,
        vadm_buy_pctile=vadm_buy_pctile, vadm_sell_pctile=vadm_sell_pctile,
        h3_holding_period=h3_holding_period,
        volume_window=volume_window,
        filing_lag_days=filing_lag_days, pe_min_periods=pe_min_periods,
        momentum_window=momentum_window, rsi_window=rsi_window,
        dma_short=dma_short, dma_long=dma_long,
        price_start=price_start, holding_periods=holding_periods,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    if not ticker or xlsx_file is None:
        st.markdown("<span style='color:#3FB950; font-family:monospace;'>[SYSTEM] AWAITING INPUTS...</span>", unsafe_allow_html=True)
        return

    if run_btn:
        with st.spinner(f"[SYSTEM] AGGREGATING DATA ARRAYS FOR {ticker}..."):
            result, err = run_single_stock(ticker, xlsx_file, exclusions_text, params)

        if err:
            st.error(f"[ERROR] {err}")
            return

        merged = result["merged"]

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("TRADING DAYS", len(merged))
        d2.metric("VALID PE ARRAY", int(merged["pe"].notna().sum()))
        d3.metric("HEAVY BUY SIG", f"{result['n_buy_raw_days']} days / {result['n_buy_episodes']} episodes")
        d4.metric("HEAVY SELL SIG", f"{result['n_sell_raw_days']} days / {result['n_sell_episodes']} episodes")
        st.caption("Stats/backtest below use EPISODES (independent onsets) -- a persistent multi-day regime is one episode, not one observation per day. Chart below still marks every day.")

        st.markdown("<br>", unsafe_allow_html=True)
        render_stock_chart(ticker, merged)
        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<span style='color:#3FB950; font-family:monospace; border-bottom: 1px solid #3FB950;'>[ BUY ZONE ] FORWARD RETURNS</span>", unsafe_allow_html=True)
            st.dataframe(format_terminal_df(result["results"]["buy_signal_eval"]), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("<span style='color:#F85149; font-family:monospace; border-bottom: 1px solid #F85149;'>[ SELL ZONE ] FORWARD RETURNS</span>", unsafe_allow_html=True)
            st.dataframe(format_terminal_df(result["results"]["sell_signal_eval"]), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        a1, a2 = st.columns(2)
        with a1:
            st.markdown("<span style='color:#D4A017; font-family:monospace;'>BUY ALPHA ATTRIBUTION (63D)</span>", unsafe_allow_html=True)
            render_ablation_table(result["ablation_results"], "buy")
        with a2:
            st.markdown("<span style='color:#D4A017; font-family:monospace;'>SELL ALPHA ATTRIBUTION (63D)</span>", unsafe_allow_html=True)
            render_ablation_table(result["ablation_results"], "sell")

        st.markdown("<br><hr>", unsafe_allow_html=True)
        render_quadrant_chart(merged)
        st.markdown("<br>", unsafe_allow_html=True)
        render_h3_results(result["h3_model"], result["h3_err"], params["h3_holding_period"])
        st.markdown("<br>", unsafe_allow_html=True)
        render_hypothesis_panel()


if __name__ == "__main__":
    if check_access():
        main()
