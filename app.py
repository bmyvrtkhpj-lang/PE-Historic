"""
app.py
------
PE + Technical Entry/Exit Signal Framework -- Streamlit app.
"""

import pandas as pd
import streamlit as st
import requests
import io

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from data_pipeline import extract_annual_fundamentals, build_step_function_pe, apply_corporate_action_exclusions, fetch_price_volume
from technical_indicators import add_all_technical_indicators
from pe_signal import pe_percentile_rank, generate_signals, generate_ablation_signals
from backtest import run_backtest, run_ablation_backtest


# Must be the first Streamlit command
st.set_page_config(page_title="BLOOMBERG TERMINAL - QUANT FRAMEWORK", layout="wide")

# --- BLOOMBERG TERMINAL EXACT CSS INJECTION ---
st.markdown("""
    <style>
    /* Global Monospace & Pure Black Background */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
    
    /* Apply monospace everywhere */
    p, div, h1, h2, h3, h4, h5, h6, label, input, button, li {
        font-family: 'Space Mono', 'Consolas', 'Courier New', monospace !important;
    }
    
    /* FIX: Protect Streamlit's internal icons from becoming text */
    span.material-symbols-rounded, span.material-icons, .stIcon {
        font-family: 'Material Symbols Rounded' !important;
    }
    
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Control Bar & Padding Alignment */
    .block-container {
        padding-top: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }
    
    /* Bloomberg Sharp Metric Cards (Zero Border Radius, Amber Accents) */
    div[data-testid="metric-container"] {
        background-color: #000000;
        border: 1px solid #333333;
        border-radius: 0px;
        padding: 10px 15px;
        border-top: 2px solid #FFB000;
    }
    
    div[data-testid="stMetricValue"] > div {
        color: #FFFFFF !important;
        font-size: 1.8rem !important;
    }
    div[data-testid="stMetricLabel"] > div > div > p {
        color: #FFB000 !important;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    /* Input Fields & Select Boxes */
    div.stSelectbox > div > div, input {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
        border-radius: 0px !important;
    }
    
    /* The RUN button - Bloomberg style 'Execute' key */
    div.stButton > button[kind="primary"] {
        background-color: #000000;
        color: #00FF00;
        border: 1px solid #00FF00;
        border-radius: 0px;
        font-weight: 700;
        text-transform: uppercase;
        height: 100%;
        margin-top: 15px;
        padding: 20px 0px;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #00FF00;
        color: #000000;
    }
    
    /* Expander / Popovers */
    .st-expander, .stPopover {
        border-color: #333333 !important;
        border-radius: 0px !important;
        background-color: #000000 !important;
    }
    
    /* Headers & Markdown */
    h1, h2, h3, h4, h5, h6, p {
        color: #FFFFFF;
    }
    hr {
        border-color: #333333;
    }
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
    price_start = params["price_start"]
    price_end = pd.Timestamp.today().strftime("%Y-%m-%d")
    price_df = cached_price_volume(ticker, price_start, price_end)
    if price_df.empty:
        return None, f"No data returned for {ticker}."

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
    merged = generate_signals(
        merged,
        cheap_pctile=params["cheap_pctile"],
        expensive_pctile=params["expensive_pctile"],
        volume_z_threshold=params["volume_z_threshold"],
        require_momentum_confirmation=params["require_momentum_confirmation"],
    )
    merged = generate_ablation_signals(
        merged,
        cheap_pctile=params["cheap_pctile"],
        expensive_pctile=params["expensive_pctile"],
        volume_z_threshold=params["volume_z_threshold"],
        require_momentum_confirmation=params["require_momentum_confirmation"],
    )

    results = run_backtest(merged, holding_periods=params["holding_periods"])
    ablation_results = run_ablation_backtest(merged, holding_periods=params["holding_periods"])
    return {"annual": annual, "merged": merged, "results": results, "ablation_results": ablation_results}, None


def render_stock_chart(ticker, merged):
    if go is None: return
    fig = go.Figure()
    
    # Bloomberg Base Price Line (Yellowish/White)
    fig.add_trace(go.Scatter(x=merged.index, y=merged["price"], name="PRICE", yaxis="y", 
                             line=dict(width=1.5, color="#FFFFFF")))
    
    buys = merged[merged["heavy_buying"]]
    sells = merged[merged["heavy_selling"]]
    
    # Bloomberg Markers (Pure Green & Pure Red)
    fig.add_trace(go.Scatter(x=buys.index, y=buys["price"], mode="markers", name="HEAVY BUY",
                              marker=dict(color="#00FF00", size=8, symbol="triangle-up", 
                                          line=dict(color="#000000", width=1))))
    fig.add_trace(go.Scatter(x=sells.index, y=sells["price"], mode="markers", name="HEAVY SELL",
                              marker=dict(color="#FF0000", size=8, symbol="triangle-down", 
                                          line=dict(color="#000000", width=1))))
                                          
    # PE Percentile (Amber Dashed)
    fig.add_trace(go.Scatter(x=merged.index, y=merged["pe_percentile"] * 100, name="PE %RANK",
                              yaxis="y2", line=dict(width=1, dash="dot", color="#FFB000")))
                              
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        title=dict(text=f"> {ticker} : SIGNAL ANALYSIS", font=dict(size=14, color="#FFB000", family="Space Mono, monospace")),
        height=500,
        yaxis=dict(title="PRICE (INR)", showgrid=True, gridcolor="#222222", zeroline=False, tickfont=dict(color="#FFFFFF", family="Space Mono, monospace")),
        yaxis2=dict(title="PE PERCENTILE", overlaying="y", side="right", range=[0, 100], showgrid=False, tickfont=dict(color="#FFB000", family="Space Mono, monospace")),
        xaxis=dict(tickfont=dict(color="#FFFFFF", family="Space Mono, monospace"), gridcolor="#222222"),
        legend=dict(orientation="h", y=1.05, x=0, bgcolor='#000000', font=dict(size=11, color="#FFFFFF", family="Space Mono, monospace")),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_ablation_table(ablation_results, side):
    labels = {
        f"{side}_pe_only": "VAL ONLY",
        f"{side}_technical_only": "TECH ONLY",
        f"{side}_combined": "COMBINED",
    }
    rows = []
    for key, label in labels.items():
        df = ablation_results.get(key)
        if df is None or df.empty: continue
        row = df[df["holding_period"] == "fwd_ret_63d"]
        if row.empty: continue
        row = row.iloc[0]
        
        # Format P-Value strictly
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

def format_terminal_df(df):
    """Formats raw backtest DataFrame into a clean Terminal-style display."""
    if df is None or df.empty: return df
    out = df.copy()
    
    # Rename columns to Bloomberg style shorthand
    out = out.rename(columns={
        "holding_period": "PERIOD",
        "n_signals": "SIG_CNT",
        "avg_signal_return": "AVG_RET",
        "avg_baseline_return": "BASE_RET",
        "win_rate": "WIN_RATE",
        "t_stat": "T_STAT",
        "p_value": "P_VAL"
    })
    
    # Format percentages
    for col in ["AVG_RET", "BASE_RET", "WIN_RATE"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "-")
            
    # Format Stats
    if "T_STAT" in out.columns:
        out["T_STAT"] = out["T_STAT"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
    if "P_VAL" in out.columns:
        out["P_VAL"] = out["P_VAL"].apply(lambda x: "<0.001" if pd.notnull(x) and x < 0.001 else (f"{x:.4f}" if pd.notnull(x) else "-"))
        
    # Drop messy raw notes column if it exists
    if "note" in out.columns:
        out = out.drop(columns=["note"])
        
    return out

def main():
    st.markdown("<h3 style='color: #FFB000; font-family: monospace; font-size: 1.2rem;'>[ BLOOMBERG ] SYSTEMATIC QUANT FRAMEWORK</h3>", unsafe_allow_html=True)
    
    # --- COMMAND BAR (Overlap Fixed) ---
    # Adjusted ratios so file uploader gets more space, removed vertical_alignment to prevent squishing
    c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
    
    with c1:
        TICKER_MAPPING = get_all_nse_stocks()
        company_name = st.selectbox("TICKER", options=list(TICKER_MAPPING.keys()), index=None, placeholder="<SEARCH TICKER>")
        ticker = TICKER_MAPPING.get(company_name) if company_name else ""
        
    with c2:
        xlsx_file = st.file_uploader("DATA (.XLSX)", type=["xlsx"])
        
    with c3:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True) # Aligns popover with inputs
        with st.popover("[ STRAT PARAMS ]", use_container_width=True):
            st.markdown("<span style='color:#FFB000'>VALUATION BOUNDS</span>", unsafe_allow_html=True)
            cheap_pctile = st.slider("CHEAP PE %", 0.05, 0.50, 0.20, 0.05)
            expensive_pctile = st.slider("EXPENSIVE PE %", 0.50, 0.95, 0.80, 0.05)
            
            st.markdown("<span style='color:#FFB000'>TECHNICAL TRIGGERS</span>", unsafe_allow_html=True)
            volume_z_threshold = st.slider("VOL SPIKE (Z)", 0.5, 4.0, 1.5, 0.1)
            require_momentum_confirmation = st.checkbox("REQUIRE MOMENTUM", value=True)
            
            st.markdown("<span style='color:#FFB000'>ADVANCED ENGINE</span>", unsafe_allow_html=True)
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
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True) # Aligns button with inputs
        run_btn = st.button("< EXECUTE >", type="primary", use_container_width=True)

    holding_periods = tuple(int(x.strip()) for x in holding_periods_text.split(",") if x.strip())

    params = dict(
        cheap_pctile=cheap_pctile, expensive_pctile=expensive_pctile,
        volume_z_threshold=volume_z_threshold, volume_window=volume_window,
        require_momentum_confirmation=require_momentum_confirmation,
        filing_lag_days=filing_lag_days, pe_min_periods=pe_min_periods,
        momentum_window=momentum_window, rsi_window=rsi_window,
        dma_short=dma_short, dma_long=dma_long,
        price_start=price_start, holding_periods=holding_periods,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- MAIN TERMINAL DISPLAY ---
    if not ticker or xlsx_file is None:
        st.markdown("<span style='color:#00FF00; font-family:monospace;'>[SYSTEM] AWAITING INPUTS...</span>", unsafe_allow_html=True)
        return

    if run_btn:
        with st.spinner(f"[SYSTEM] AGGREGATING DATA ARRAYS FOR {ticker}..."):
            result, err = run_single_stock(ticker, xlsx_file, exclusions_text, params)
        
        if err:
            st.error(f"[ERROR] {err}")
            return

        merged = result["merged"]
        
        # Metrics Row
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("TRADING DAYS", len(merged))
        d2.metric("VALID PE ARRAY", int(merged["pe"].notna().sum()))
        d3.metric("HEAVY BUY SIG", int(merged['heavy_buying'].sum()))
        d4.metric("HEAVY SELL SIG", int(merged['heavy_selling'].sum()))

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Chart
        render_stock_chart(ticker, merged)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Data Tables - Now Formatted!
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<span style='color:#00FF00; font-family:monospace; border-bottom: 1px solid #00FF00;'>[ BUY ZONE ] FORWARD RETURNS</span>", unsafe_allow_html=True)
            st.dataframe(format_terminal_df(result["results"]["buy_signal_eval"]), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("<span style='color:#FF0000; font-family:monospace; border-bottom: 1px solid #FF0000;'>[ SELL ZONE ] FORWARD RETURNS</span>", unsafe_allow_html=True)
            st.dataframe(format_terminal_df(result["results"]["sell_signal_eval"]), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("<span style='color:#FFB000; font-family:monospace;'>BUY ALPHA ATTRIBUTION (63D)</span>", unsafe_allow_html=True)
            render_ablation_table(result["ablation_results"], "buy")
        with a2:
            st.markdown("<span style='color:#FFB000; font-family:monospace;'>SELL ALPHA ATTRIBUTION (63D)</span>", unsafe_allow_html=True)
            render_ablation_table(result["ablation_results"], "sell")


if __name__ == "__main__":
    main()
