"""
app.py
------
PE + Delivery Valuation Signal Framework -- Streamlit app (Cyber-Institutional UI).
"""

import pandas as pd
import streamlit as st
import requests
import io
import time

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None

from data_pipeline import extract_annual_fundamentals, build_step_function_pe, apply_corporate_action_exclusions, fetch_price_volume, fetch_eod2_delivery_data
from technical_indicators import add_all_technical_indicators
from pe_signal import pe_percentile_rank, delivery_percentile_rank
from vadm import delivery_flow_strength, compute_vadm, classify_quadrant, generate_quadrant_signals, generate_quadrant_ablation_signals, signal_onsets
from backtest import run_backtest, run_ablation_backtest, test_h3_interaction


APP_NAME = "VADM TERMINAL"
# --- TRADINGVIEW LIGHT THEME COLOR PALETTE ---
BG = "#FFFFFF"          # Pure White background
PANEL_BG = "#FFFFFF"    # White panels
BORDER = "#E0E3EB"      # Subtle light grey borders
GRID = "#F0F3FA"        # Very faint gridlines
TEXT = "#131722"        # Dark grey/black for primary text
TEXT_MUTED = "#787B86"  # Medium grey for secondary text
ACCENT = "#2962FF"      # TradingView Primary Blue
GREEN = "#089981"       # TradingView Positive Green
RED = "#F23645"         # TradingView Negative Red
CAUTION = "#FF9800"     # Warning Orange

st.set_page_config(page_title=f"{APP_NAME}", layout="wide", page_icon="📈")

# --- TRADINGVIEW 100% REPLICA CSS INJECTION ---
st.markdown(f"""
    <style>
    /* Clean Sans-Serif font just like TradingView */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    p, div, h1, h2, h3, h4, h5, h6, label, input, button, li, th, td {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Trebuchet MS', Roboto, Ubuntu, sans-serif !important;
    }}
    
    .stApp {{ background-color: {BG}; color: {TEXT}; }}
    #MainMenu, footer {{ visibility: hidden; }}
    .block-container {{ padding: 1.5rem 2rem !important; max-width: 100% !important; }}
    
    /* Sleek metric panels with Rounded Corners (TradingView style) */
    .metric-card {{
        background-color: {PANEL_BG};
        border: 1px solid {BORDER};
        border-radius: 8px; /* Smooth rounded corners */
        padding: 16px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05); /* Very subtle shadow */
    }}
    .metric-title {{ color: {TEXT_MUTED}; font-size: 0.85rem; font-weight: 500; }}
    .metric-value {{ color: {TEXT}; font-size: 1.7rem; font-weight: 600; margin-top: 4px; }}
    .metric-sub {{ color: {TEXT_MUTED}; font-size: 0.75rem; margin-top: 2px; }}

    /* Borderless & Styled DataFrames */
    .stDataFrame {{ border: none !important; }}
    [data-testid="stTable"] {{ border: none !important; background-color: transparent !important; }}
    th {{ 
        background-color: {BG} !important; 
        color: {TEXT_MUTED} !important; 
        font-size: 0.8rem !important; 
        font-weight: 500 !important;
        border-bottom: 1px solid {BORDER} !important; 
    }}
    td {{ 
        border-bottom: 1px solid {GRID} !important; 
        font-size: 0.85rem !important; 
        color: {TEXT} !important;
    }}

    /* Pill-style Checkboxes for Chart Controls */
    div[data-testid="stCheckbox"] {{
        background-color: {BG};
        border: 1px solid {BORDER};
        padding: 6px 14px;
        border-radius: 16px; /* Pill shape */
        transition: all 0.2s ease;
        color: {TEXT};
        font-size: 0.85rem;
    }}
    div[data-testid="stCheckbox"]:hover {{ border-color: {ACCENT}; }}
    
    /* TradingView Blue Primary Button */
    div.stButton > button[kind="primary"] {{
        background-color: {ACCENT}; 
        color: #FFFFFF; 
        border: none;
        border-radius: 8px; /* Rounded corners */
        font-weight: 600; 
        height: 100%;
        padding: 14px 0px; 
        transition: 0.2s ease;
    }}
    div.stButton > button[kind="primary"]:hover {{ background-color: #1E53E5; color: #FFFFFF; }}
    
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {BG}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
    </style>
""", unsafe_allow_html=True)


# ==========================================
# ENGINE LOGIC (UNCHANGED)
# ==========================================

def parse_exclusions(text):
    if not text or not text.strip(): return []
    windows = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk: continue
        try:
            start, end = chunk.split(":")
            windows.append((start.strip(), end.strip()))
        except ValueError:
            st.warning(f"Could not parse exclusion window '{chunk}'")
    return windows

@st.cache_data(ttl=86400, show_spinner=False)
def get_all_nse_stocks():
    try:
        res = requests.get("https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv", headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        df = pd.read_csv(io.StringIO(res.text))
        return {f"{row['NAME OF COMPANY']} ({row['SYMBOL']})": f"{row['SYMBOL']}.NS" for _, row in df.iterrows()}
    except Exception:
        return {"HDFC Bank Limited (HDFCBANK)": "HDFCBANK.NS", "Reliance Industries (RELIANCE)": "RELIANCE.NS"}

@st.cache_data(show_spinner=False, ttl=6 * 3600)
def cached_price_volume(ticker, start, end):
    return fetch_price_volume(ticker, start, end)

def run_single_stock(ticker, xlsx_file, exclusions_text, params):
    annual = extract_annual_fundamentals(xlsx_file)
    nse_symbol = ticker.replace(".NS", "").replace(".BO", "")
    eod2_df, eod2_err = fetch_eod2_delivery_data(nse_symbol)
    if eod2_err: return None, f"Price/delivery data unavailable: {eod2_err}"

    price_start = params["price_start"]
    price_df = eod2_df.loc[eod2_df.index >= price_start, ["price", "volume", "open", "high", "low"]]
    if price_df.empty: return None, f"No price data from {price_start} onward."

    pe_df = build_step_function_pe(annual, price_df["price"], filing_lag_days=params["filing_lag_days"])
    pe_df = apply_corporate_action_exclusions(pe_df, parse_exclusions(exclusions_text))

    merged = pe_df.join(price_df[["volume", "open", "high", "low"]], how="left")
    merged = add_all_technical_indicators(merged, volume_window=params["volume_window"], momentum_window=params["momentum_window"], rsi_window=params["rsi_window"], dma_short=params["dma_short"], dma_long=params["dma_long"])
    merged["pe_percentile"] = pe_percentile_rank(merged["pe"], min_periods=params["pe_min_periods"])

    merged = merged.join(eod2_df[["delivery_pct"]], how="left")
    merged["delivery_percentile"] = delivery_percentile_rank(merged["delivery_pct"], min_periods=params["pe_min_periods"])
    merged["quadrant"] = classify_quadrant(merged["pe_percentile"], merged["delivery_percentile"])
    merged["delivery_flow"] = delivery_flow_strength(merged["delivery_percentile"])
    merged["vadm"] = compute_vadm(merged["pe_percentile"], merged["delivery_flow"])

    merged = generate_quadrant_signals(merged, vadm_buy_pctile=params["vadm_buy_pctile"], vadm_sell_pctile=params["vadm_sell_pctile"])
    merged = generate_quadrant_ablation_signals(merged, cheap_pctile=params["cheap_pctile"], delivery_pctile_midpoint=0.5)

    eval_df = merged.copy()
    for col in ["heavy_buying", "heavy_selling", "buy_pe_only", "buy_delivery_only", "buy_combined", "sell_pe_only", "sell_delivery_only", "sell_combined"]:
        eval_df[col] = signal_onsets(merged[col])

    results = run_backtest(eval_df, holding_periods=params["holding_periods"])
    ablation_results = run_ablation_backtest(eval_df, holding_periods=params["holding_periods"])
    h3_model, h3_err = test_h3_interaction(merged, holding_period=params.get("h3_holding_period", 63))
    
    return {
        "annual": annual, "merged": merged, "results": results, "ablation_results": ablation_results,
        "h3_model": h3_model, "h3_err": h3_err,
        "n_buy_episodes": int(eval_df["heavy_buying"].sum()), "n_sell_episodes": int(eval_df["heavy_selling"].sum()),
        "n_buy_raw_days": int(merged["heavy_buying"].sum()), "n_sell_raw_days": int(merged["heavy_selling"].sum()),
    }, None


# ==========================================
# UI / UX LAYER (REWRITTEN & UPGRADED)
# ==========================================

def color_returns(val):
    """Pandas Styler function to color negative red, positive green."""
    if isinstance(val, str) and '%' in val:
        try:
            num = float(val.replace('%', '').strip())
            if num > 0: return f'color: {GREEN}; font-weight: 500;'
            elif num < 0: return f'color: {RED}; font-weight: 500;'
        except: pass
    return ''

def render_advanced_chart(ticker, merged, show_pe=True, show_delivery=True, show_rsi=False, show_obv=False, show_momentum=False):
    """Multi-Pane TradingView Style Chart."""
    if go is None: return
    
    plot_df = merged.dropna(subset=["price"]).copy()
    if plot_df.empty: return

    has_ohlc = all(c in plot_df.columns for c in ["open", "high", "low"])
    
    # 2-Pane Base Setup (Row 1: Price 75%, Row 2: Oscillators/Volume 25%)
    extra_rows = [r for r, show in zip(["RSI", "OBV", "MOM"], [show_rsi, show_obv, show_momentum]) if show]
    n_rows = 2 + len(extra_rows)
    row_heights = [0.70, 0.30] if not extra_rows else [0.55, 0.15] + [(0.30/len(extra_rows))]*len(extra_rows)
    
    specs = [[{"secondary_y": False}]] + [[{"secondary_y": True}]] + [[{"secondary_y": False}]]*len(extra_rows)
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights, specs=specs)

    # --- PANE 1: Candlesticks ---
    if has_ohlc:
        fig.add_trace(go.Candlestick(
            x=plot_df.index, open=plot_df["open"], high=plot_df["high"], low=plot_df["low"], close=plot_df["price"],
            increasing_line_color="#089981", decreasing_line_color="#F23645",
            increasing_fillcolor="#089981", decreasing_fillcolor="#F23645", name="OHLC"
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["price"], name="PRICE", line=dict(color=TEXT)), row=1, col=1)

    # --- PANE 2: Volume & Oscillator Lines ---
    vol_colors = ["#089981" if c >= o else "#F23645" for c, o in zip(plot_df["price"], plot_df["open"])] if has_ohlc else [TEXT_MUTED]*len(plot_df)
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["volume"], marker_color=vol_colors, opacity=0.3, name="VOL"), row=2, col=1, secondary_y=False)
    
    if show_pe and "pe_percentile" in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["pe_percentile"]*100, line=dict(width=1, dash="dot", color=ACCENT), opacity=0.7, name="PE %RANK"), row=2, col=1, secondary_y=True)
    if show_delivery and "delivery_percentile" in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["delivery_percentile"]*100, line=dict(width=1, dash="dot", color=CAUTION), opacity=0.7, name="DELIV %RANK"), row=2, col=1, secondary_y=True)

    # --- TRADINGVIEW ANNOTATIONS (Edge Detection Logic) ---
    is_buy_zone = plot_df["quadrant"] == "Value Confirmation"
    is_sell_zone = plot_df["quadrant"] == "Reversal/De-rating"
    is_bad_zone = plot_df["quadrant"].isin(["Value Trap Warning", "Reversal/De-rating"])
    
    df_entry_long = plot_df[signal_onsets(is_buy_zone)]
    df_entry_short = plot_df[signal_onsets(is_sell_zone)]
    df_exit_long = plot_df[signal_onsets(is_bad_zone)]

    # Entry Long (CE)
    for idx, row in df_entry_long.iterrows():
        y_val = row["low"] if "low" in row and pd.notnull(row["low"]) else row["price"]
        fig.add_annotation(
            x=idx, y=y_val, text="<b>CE</b><br>Long (CE)<br>+8", showarrow=True, arrowhead=1, arrowsize=1.5,
            arrowcolor="#2962FF", ax=0, ay=45, font=dict(size=9, color="#FFF", family="IBM Plex Mono"), align="center",
            bgcolor="#089981", bordercolor="#089981", borderpad=2, row=1, col=1
        )
    # Entry Short (PE)
    for idx, row in df_entry_short.iterrows():
        y_val = row["high"] if "high" in row and pd.notnull(row["high"]) else row["price"]
        fig.add_annotation(
            x=idx, y=y_val, text="<b>PE</b><br>Short (PE)<br>-8", showarrow=True, arrowhead=1, arrowsize=1.5,
            arrowcolor="#FF9800", ax=0, ay=-45, font=dict(size=9, color="#FFF", family="IBM Plex Mono"), align="center",
            bgcolor="#F23645", bordercolor="#F23645", borderpad=2, row=1, col=1
        )
    # Exit Long
    for idx, row in df_exit_long.iterrows():
        y_val = row["high"] if "high" in row and pd.notnull(row["high"]) else row["price"]
        fig.add_annotation(
            x=idx, y=y_val, text="-8<br>Exit Long", showarrow=True, arrowhead=1, arrowsize=1,
            arrowcolor="#9C27B0", ax=0, ay=-35, font=dict(size=9, color=TEXT, family="IBM Plex Mono"), row=1, col=1
        )

    # --- Extra Panes (RSI/OBV) ---
    cur = 3
    if show_rsi and "rsi" in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["rsi"], line=dict(color=ACCENT, width=1)), row=cur, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color=RED, row=cur, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color=GREEN, row=cur, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=cur, col=1)
        cur+=1
    
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        title=dict(text=f"> {ticker} : MASTER OHLC VIEW", font=dict(size=14, color=ACCENT)),
        height=650 + (120 * len(extra_rows)), margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=1.04, x=0, bgcolor=BG), hovermode="x unified",
        xaxis_rangeslider_visible=False, dragmode="pan"
    )
    fig.update_xaxes(gridcolor=GRID, showgrid=True)
    fig.update_yaxes(gridcolor=GRID, showgrid=True)
    fig.update_yaxes(title_text="VOL", showgrid=False, row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="RANK", range=[0, 100], showgrid=False, row=2, col=1, secondary_y=True)

    st.plotly_chart(fig, width='stretch', config={"scrollZoom": True, "modeBarButtonsToAdd": ["drawline", "drawrect"], "displaylogo": False})

def format_terminal_df(df):
    """Outputs a Styled Pandas Dataframe with Color Coding."""
    if df is None or df.empty: return df
    out = df.rename(columns={"holding_period": "PERIOD", "n_signals": "SIG_CNT", "avg_signal_return": "AVG_RET", "avg_baseline_return": "BASE_RET", "win_rate": "WIN_RATE", "t_stat": "T_STAT", "p_value": "P_VAL"})
    for col in ["AVG_RET", "BASE_RET", "WIN_RATE"]:
        if col in out.columns: out[col] = out[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "-")
    if "T_STAT" in out.columns: out["T_STAT"] = out["T_STAT"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
    if "P_VAL" in out.columns: out["P_VAL"] = out["P_VAL"].apply(lambda x: "<0.001" if pd.notnull(x) and x < 0.001 else (f"{x:.4f}" if pd.notnull(x) else "-"))
    if "note" in out.columns: out = out.drop(columns=["note"])
    # Return Styled Dataframe
    return out.style.applymap(color_returns).set_properties(**{'text-align': 'right'})

def render_ablation_table(ablation_results, side):
    labels = {f"{side}_pe_only": "PE ONLY (H1)", f"{side}_delivery_only": "DELIVERY ONLY (H2)", f"{side}_combined": "COMBINED / VADM (H3)"}
    rows = []
    for key, label in labels.items():
        df = ablation_results.get(key)
        if df is None or df.empty: continue
        row = df[df["holding_period"] == "fwd_ret_63d"]
        if row.empty: continue
        row = row.iloc[0]
        pval = row['p_value']
        rows.append({
            "TYPE": label, "SIG": row["n_signals"],
            "RET(63D)": f"{row['avg_signal_return']:.2%}" if pd.notnull(row['avg_signal_return']) else "-",
            "WIN(%)": f"{row['win_rate']:.2%}" if pd.notnull(row['win_rate']) else "-",
            "P-VAL": "<0.001" if pd.notnull(pval) and pval < 0.001 else (f"{pval:.4f}" if pd.notnull(pval) else "-"),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows).style.applymap(color_returns).set_properties(**{'text-align': 'right'}), width='stretch', hide_index=True)


def main():
    st.markdown(f"<h3 style='color: {ACCENT}; font-size: 1.2rem; margin-bottom: 0px;'>[ {APP_NAME} ]</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
    with c1:
        TICKER_MAPPING = get_all_nse_stocks()
        company_name = st.selectbox("TICKER", options=list(TICKER_MAPPING.keys()), index=None, placeholder="<SEARCH TICKER>", label_visibility="collapsed")
        ticker = TICKER_MAPPING.get(company_name) if company_name else ""
    with c2:
        xlsx_file = st.file_uploader("DATA", type=["xlsx"], label_visibility="collapsed")
    with c3:
        with st.popover("[ STRAT PARAMS ]", width='stretch'):
            st.markdown("<span style='color:#D4A017'>VALUATION BOUNDS</span>", unsafe_allow_html=True)
            cheap_pctile = st.slider("CHEAP PE %", 0.05, 0.50, 0.20, 0.05)
            expensive_pctile = st.slider("EXPENSIVE PE %", 0.50, 0.95, 0.80, 0.05)
            st.markdown("<span style='color:#D4A017'>VADM SIGNAL</span>", unsafe_allow_html=True)
            vadm_buy_pctile = st.slider("VADM BUY (TOP %)", 0.70, 0.99, 0.90, 0.01)
            vadm_sell_pctile = st.slider("VADM SELL (BOTTOM %)", 0.01, 0.30, 0.10, 0.01)
            h3_holding_period = st.selectbox("H3 REGRESSION HOLDING", [21, 63, 126, 252], index=1)
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
        run_btn = st.button("< EXECUTE >", type="primary", width='stretch')

    if not ticker or xlsx_file is None: return

    if run_btn:
        with st.spinner(f"[SYSTEM] AGGREGATING ARRAYS..."):
            params = dict(cheap_pctile=cheap_pctile, expensive_pctile=expensive_pctile, vadm_buy_pctile=vadm_buy_pctile, vadm_sell_pctile=vadm_sell_pctile, h3_holding_period=h3_holding_period, volume_window=volume_window, filing_lag_days=filing_lag_days, pe_min_periods=pe_min_periods, momentum_window=momentum_window, rsi_window=rsi_window, dma_short=dma_short, dma_long=dma_long, price_start=price_start, holding_periods=tuple(int(x.strip()) for x in holding_periods_text.split(",") if x.strip()))
            result, err = run_single_stock(ticker, xlsx_file, exclusions_text, params)
        if err: st.error(err); return

        merged = result["merged"]

        # --- CUSTOM METRIC CARDS ---
        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        def custom_metric(col, title, val, sub):
            col.markdown(f"<div class='metric-card'><div class='metric-title'>{title}</div><div class='metric-value'>{val}</div><div class='metric-sub'>{sub}</div></div>", unsafe_allow_html=True)
        
        custom_metric(m1, "TRADING DAYS", len(merged), "Total History")
        custom_metric(m2, "VALID PE ARRAY", int(merged["pe"].notna().sum()), "Calculated Periods")
        custom_metric(m3, "HEAVY BUY SIG", result['n_buy_raw_days'], f"[ {result['n_buy_episodes']} EPISODES ]")
        custom_metric(m4, "HEAVY SELL SIG", result['n_sell_raw_days'], f"[ {result['n_sell_episodes']} EPISODES ]")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- PILL CONTROLS ---
        st.markdown("<span style='color:#D4A017; font-size:0.8rem;'>[ CHART OVERLAYS ]</span>", unsafe_allow_html=True)
        t1, t2, t3, t4, t5 = st.columns(5)
        show_pe = t1.checkbox("PE RANK", value=True)
        show_delivery = t2.checkbox("DELIVERY RANK", value=True)
        show_rsi = t3.checkbox("RSI", value=False)
        show_obv = t4.checkbox("OBV", value=False)
        show_momentum = t5.checkbox("MOMENTUM", value=False)

        render_advanced_chart(ticker, merged, show_pe=show_pe, show_delivery=show_delivery, show_rsi=show_rsi, show_obv=show_obv, show_momentum=show_momentum)
        
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<span style='color:#3FB950; font-family:monospace; border-bottom: 1px solid #3FB950;'>[ BUY ZONE ] FORWARD RETURNS</span>", unsafe_allow_html=True)
            st.dataframe(format_terminal_df(result["results"]["buy_signal_eval"]), width='stretch', hide_index=True)
            st.markdown("<br><span style='color:#D4A017; font-family:monospace;'>BUY ALPHA ATTRIBUTION (63D)</span>", unsafe_allow_html=True)
            render_ablation_table(result["ablation_results"], "buy")
        with c2:
            st.markdown("<span style='color:#F85149; font-family:monospace; border-bottom: 1px solid #F85149;'>[ SELL ZONE ] FORWARD RETURNS</span>", unsafe_allow_html=True)
            st.dataframe(format_terminal_df(result["results"]["sell_signal_eval"]), width='stretch', hide_index=True)
            st.markdown("<br><span style='color:#D4A017; font-family:monospace;'>SELL ALPHA ATTRIBUTION (63D)</span>", unsafe_allow_html=True)
            render_ablation_table(result["ablation_results"], "sell")

if __name__ == "__main__":
    main()
