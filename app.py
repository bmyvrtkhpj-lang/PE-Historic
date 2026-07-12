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
st.set_page_config(page_title="Quant Signal Framework", layout="wide")

# --- INSTITUTIONAL UI CSS INJECTION ---
st.markdown("""
    <style>
    /* Stealth Dark Mode Background */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Professional Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 4px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Clean up expanders and tabs */
    .st-expander {
        background-color: #161b22;
        border-color: #30363d;
    }
    
    /* Accent color overrides for sliders/buttons */
    div.stButton > button:first-child {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        margin-top: 15px;
        padding: 20px 0px;
    }
    div.stButton > button:hover {
        background-color: #2ea043;
        border: none;
    }
    
    /* Top padding reduction for clean look */
    .block-container {
        padding-top: 2rem;
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
            st.warning(f"Could not parse exclusion window '{chunk}' -- expected format YYYY-MM-DD:YYYY-MM-DD")
    return windows


@st.cache_data(ttl=86400, show_spinner=False)
def get_all_nse_stocks():
    try:
        url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        res = requests.get(url, headers=headers, timeout=10)
        df = pd.read_csv(io.StringIO(res.text))

        mapping = {}
        for _, row in df.iterrows():
            display_name = f"{row['NAME OF COMPANY']} ({row['SYMBOL']})"
            mapping[display_name] = f"{row['SYMBOL']}.NS"
        return mapping
    except Exception:
        return {
            "HDFC Bank Limited (HDFCBANK)": "HDFCBANK.NS",
            "Reliance Industries Limited (RELIANCE)": "RELIANCE.NS",
            "Tata Consultancy Services Limited (TCS)": "TCS.NS"
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
        return None, f"No price/volume data returned for {ticker}."

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
    if go is None:
        st.info("plotly not installed -- skipping chart.")
        return
    fig = go.Figure()
    
    # Base Price Line (Subtle)
    fig.add_trace(go.Scatter(x=merged.index, y=merged["price"], name="Price", yaxis="y", 
                             line=dict(width=1.5, color="#8b949e")))
    
    buys = merged[merged["heavy_buying"]]
    sells = merged[merged["heavy_selling"]]
    
    # High-Vis Neon Markers for Institutional Look
    fig.add_trace(go.Scatter(x=buys.index, y=buys["price"], mode="markers", name="Heavy Buying",
                              marker=dict(color="#00FFCC", size=10, symbol="triangle-up", 
                                          line=dict(color="black", width=1))))
    fig.add_trace(go.Scatter(x=sells.index, y=sells["price"], mode="markers", name="Heavy Selling",
                              marker=dict(color="#FF007F", size=10, symbol="triangle-down", 
                                          line=dict(color="black", width=1))))
                                          
    # PE Percentile (Orange Dashed)
    fig.add_trace(go.Scatter(x=merged.index, y=merged["pe_percentile"] * 100, name="PE Percentile",
                              yaxis="y2", line=dict(width=1, dash="dot", color="#f78166")))
                              
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text=f"{ticker} | Technical & Valuation Signal Analysis", font=dict(size=18, color="#e6edf3")),
        height=550,
        yaxis=dict(title="Price (INR)", showgrid=True, gridcolor="#30363d"),
        yaxis2=dict(title="PE Percentile (%)", overlaying="y", side="right", range=[0, 100], showgrid=False),
        legend=dict(orientation="h", y=1.05, x=0, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_ablation_table(ablation_results, side):
    labels = {
        f"{side}_pe_only": "Valuation ONLY",
        f"{side}_technical_only": "Technical ONLY",
        f"{side}_combined": "COMBINED SIGNAL",
    }
    rows = []
    for key, label in labels.items():
        df = ablation_results.get(key)
        if df is None or df.empty:
            continue
        row = df[df["holding_period"] == "fwd_ret_63d"]
        if row.empty:
            continue
        row = row.iloc[0]
        rows.append({
            "Variant": label,
            "Signals (n)": row["n_signals"],
            "Avg Return (63d)": f"{row['avg_signal_return']:.2%}" if pd.notnull(row['avg_signal_return']) else "N/A",
            "Win Rate": f"{row['win_rate']:.2%}" if pd.notnull(row['win_rate']) else "N/A",
            "p-value": f"{row['p_value']:.4f}" if pd.notnull(row['p_value']) else "N/A",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main():
    st.title("Systematic Entry/Exit Framework")
    st.markdown("*Valuation (PE own-history) confirmed by technical triggers (Volume Z-Score + Momentum).*")
    st.markdown("---")
    
    # --- TOP COMMAND CENTER (No Sidebar) ---
    st.markdown("### 🎛️ Command Center")
    
    # Row 1: Data Inputs
    c1, c2, c3 = st.columns([1.5, 1.5, 1])
    with c1:
        TICKER_MAPPING = get_all_nse_stocks()
        company_name = st.selectbox("Search Indian Stock", options=list(TICKER_MAPPING.keys()), index=None, placeholder="e.g. HDFC Bank")
        ticker = TICKER_MAPPING.get(company_name) if company_name else ""
    with c2:
        xlsx_file = st.file_uploader("Screener.in Export (.xlsx)", type=["xlsx"])
    with c3:
        exclusions_text = st.text_input("Corp Action Exclusions", help="YYYY-MM-DD:YYYY-MM-DD")

    # Row 2: Signal Parameters
    p1, p2, p3, p4 = st.columns([1, 1, 1, 1])
    with p1:
        cheap_pctile = st.slider("Cheap PE Percentile", 0.05, 0.50, 0.20, 0.05)
    with p2:
        expensive_pctile = st.slider("Expensive PE Percentile", 0.50, 0.95, 0.80, 0.05)
    with p3:
        volume_z_threshold = st.slider("Volume Spike (z-score)", 0.5, 4.0, 1.5, 0.1)
    with p4:
        require_momentum_confirmation = st.checkbox("Require Momentum", value=True)
        run_btn = st.button("RUN BACKTEST", type="primary", use_container_width=True)
        
    # Row 3: Advanced Settings (Collapsible)
    with st.expander("⚙️ Advanced Settings"):
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            volume_window = st.number_input("Vol Rolling Win (days)", value=20)
            momentum_window = st.number_input("Momentum Win (days)", value=10)
        with a2:
            rsi_window = st.number_input("RSI Win (days)", value=14)
            dma_short = st.number_input("Short DMA", value=50)
        with a3:
            dma_long = st.number_input("Long DMA", value=200)
            filing_lag_days = st.number_input("Filing Lag (days)", value=60)
        with a4:
            pe_min_periods = st.number_input("Min PE History (days)", value=252)
            price_start = st.text_input("Start Date", value="2016-01-01")
            
        holding_periods_text = st.text_input("Holding Periods", value="21,63,126,252")

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

    st.markdown("---")

    # --- MAIN STAGE: DASHBOARD ---
    if not ticker or xlsx_file is None:
        st.info("👆 Please select a stock and upload its Screener.in Excel export above to begin.")
        return

    if run_btn:
        with st.spinner(f"Aggregating data for {ticker}..."):
            result, err = run_single_stock(ticker, xlsx_file, exclusions_text, params)
        
        if err:
            st.error(err)
            return

        merged = result["merged"]
        
        # Top level KPIs
        st.markdown("### Risk & Exposure Overview")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Trading Days", len(merged))
        d2.metric("Valid PE Days", int(merged["pe"].notna().sum()))
        d3.metric("Heavy Buy Triggers", int(merged['heavy_buying'].sum()))
        d4.metric("Heavy Sell Triggers", int(merged['heavy_selling'].sum()))

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charting
        render_stock_chart(ticker, merged)
        
        st.markdown("---")
        
        # Evaluation Tables
        st.markdown("### Forward Return Evaluation")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🟢 Heavy Buying Zones**")
            st.dataframe(result["results"]["buy_signal_eval"], use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**🔴 Heavy Selling Zones**")
            st.dataframe(result["results"]["sell_signal_eval"], use_container_width=True, hide_index=True)

        st.markdown("---")
        
        # Ablation Module
        st.markdown("### Alpha Source Attribution (Ablation Check)")
        st.caption("Isolating whether the edge comes from the valuation zone, the technical trigger, or the combination.")
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**Buy Side Edge (63d)**")
            render_ablation_table(result["ablation_results"], "buy")
        with a2:
            st.markdown("**Sell Side Edge (63d)**")
            render_ablation_table(result["ablation_results"], "sell")


if __name__ == "__main__":
    main()
