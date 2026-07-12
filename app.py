"""
app.py
------
PE + Technical Entry/Exit Signal Framework -- Streamlit app.

Layers (see individual files for details):
  data_pipeline.py         -> Screener export parsing + step-function daily PE
  technical_indicators.py  -> volume z-score, momentum, RSI, DMA regime, OBV
  pe_signal.py             -> PE percentile banding + heavy buying/selling signal
  backtest.py              -> forward returns + statistical evaluation

Deploy on Streamlit Community Cloud:
  1. Push this whole folder to a GitHub repo (public, for the free tier).
  2. On share.streamlit.io, "New app" -> pick the repo -> main file: app.py.
  3. That's it -- requirements.txt is auto-installed by Streamlit Cloud.

IMPORTANT LIMITATION carried over from design discussion, restated here so
it isn't lost: PE resolution is ANNUAL (step function), not quarterly, since
this assumes the free Screener.in export (no Prime). Corporate actions
(mergers, demergers) must be entered manually per stock below -- they do NOT
reliably show up as automatic EPS outliers (confirmed on real HDFC Bank data,
where the 2023 merger looked unremarkable in the EPS series itself).
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
from pe_signal import pe_percentile_rank, generate_signals
from backtest import run_backtest


st.set_page_config(page_title="PE + Technical Signal Framework", layout="wide")


def parse_exclusions(text):
    """Parses 'YYYY-MM-DD:YYYY-MM-DD, YYYY-MM-DD:YYYY-MM-DD' into a list of tuples."""
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
            st.warning(f"Could not parse exclusion window '{chunk}' -- expected format YYYY-MM-DD:YYYY-MM-DD, skipping it.")
    return windows

@st.cache_data(ttl=86400, show_spinner=False) # 24 hours cache
def get_all_nse_stocks():
    """Fetches the live list of all NSE equity stocks dynamically."""
    try:
        # NSE official list link
        url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        res = requests.get(url, headers=headers, timeout=10)
        df = pd.read_csv(io.StringIO(res.text))
        
        # Map create karo: "Company Name (SYMBOL)" -> "SYMBOL.NS"
        mapping = {}
        for _, row in df.iterrows():
            display_name = f"{row['NAME OF COMPANY']} ({row['SYMBOL']})"
            mapping[display_name] = f"{row['SYMBOL']}.NS"
            
        return mapping
    except Exception as e:
        # Agar NSE ki website error de, toh ye default backup chalega
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
        return None, f"No price/volume data returned for {ticker}. Check the ticker format (e.g. HDFCBANK.NS)."

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

    results = run_backtest(merged, holding_periods=params["holding_periods"])
    return {"annual": annual, "merged": merged, "results": results}, None


def render_stock_chart(ticker, merged):
    if go is None:
        st.info("plotly not installed -- skipping chart.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged.index, y=merged["price"], name="Price", yaxis="y", line=dict(width=1.6)))
    buys = merged[merged["heavy_buying"]]
    sells = merged[merged["heavy_selling"]]
    fig.add_trace(go.Scatter(x=buys.index, y=buys["price"], mode="markers", name="Heavy Buying",
                              marker=dict(color="green", size=9, symbol="triangle-up")))
    fig.add_trace(go.Scatter(x=sells.index, y=sells["price"], mode="markers", name="Heavy Selling",
                              marker=dict(color="red", size=9, symbol="triangle-down")))
    fig.add_trace(go.Scatter(x=merged.index, y=merged["pe_percentile"] * 100, name="PE Percentile",
                              yaxis="y2", line=dict(width=1.2, dash="dot", color="orange")))
    fig.update_layout(
        title=f"{ticker} -- Price with Heavy Buying/Selling Signals + PE Percentile",
        height=440,
        yaxis=dict(title="Price"),
        yaxis2=dict(title="PE Percentile (%)", overlaying="y", side="right", range=[0, 100]),
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("PE + Technical Entry/Exit Signal Framework")
    st.caption(
        "Multi-stock framework: valuation (PE percentile, own history) confirmed by a "
        "technical trigger (volume z-score + momentum). Annual-EPS step function -- "
        "see module docstrings for the reasoning and known limitations."
    )

    st.header("1. Universe setup")
    n_stocks = st.number_input("Number of stocks", min_value=1, max_value=100, value=1, step=1)

    stock_configs = []
    for i in range(int(n_stocks)):
        with st.expander(f"Stock {i + 1}", expanded=(i == 0)):
            c1, c2 = st.columns(2)
            with c1:
                # 2000+ Indian stocks ka data load karo
                TICKER_MAPPING = get_all_nse_stocks()
                
                # Dropdown mein ab kisi bhi company ka naam search kar sakte ho
                company_name = st.selectbox(
                    f"Search Indian Stock", 
                    options=list(TICKER_MAPPING.keys()),
                    index=None,
                    placeholder="Type company name (e.g. Zomato, Tata Motors)",
                    key=f"name_{i}"
                )
                
                # Background mein code automatically usko '.NS' wala ticker bana lega
                ticker = TICKER_MAPPING.get(company_name) if company_name else ""
                )
            with c2:
                xlsx_file = st.file_uploader(f"Screener.in Excel export", type=["xlsx"], key=f"file_{i}")
            exclusions_text = st.text_input(
                "Corporate action exclusion windows (optional) -- format YYYY-MM-DD:YYYY-MM-DD, comma-separated for multiple",
                key=f"exclusions_{i}",
                help="e.g. 2023-05-01:2024-08-01 for a merger year. These are NOT auto-detected -- see module notes on why.",
            )
            stock_configs.append({"ticker": ticker, "xlsx_file": xlsx_file, "exclusions_text": exclusions_text})

    st.header("2. Backtest parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        cheap_pctile = st.slider("Cheap PE percentile threshold", 0.05, 0.50, 0.20, 0.05)
        expensive_pctile = st.slider("Expensive PE percentile threshold", 0.50, 0.95, 0.80, 0.05)
        filing_lag_days = st.number_input("Annual filing lag (days)", min_value=0, max_value=180, value=60, step=5)
    with c2:
        volume_z_threshold = st.slider("Volume z-score threshold ('heavy')", 0.5, 4.0, 1.5, 0.1)
        require_momentum_confirmation = st.checkbox("Require momentum confirmation", value=True)
        pe_min_periods = st.number_input("Min days before PE percentile starts", min_value=30, max_value=1000, value=252, step=10)
    with c3:
        momentum_window = st.number_input("Momentum window (days)", min_value=2, max_value=60, value=10)
        rsi_window = st.number_input("RSI window (days)", min_value=2, max_value=60, value=14)
        price_start = st.text_input("Price history start date", value="2016-01-01")

    dma_short = st.number_input("Short DMA (days)", min_value=5, max_value=100, value=50)
    dma_long = st.number_input("Long DMA (days)", min_value=50, max_value=400, value=200)
    holding_periods_text = st.text_input("Forward holding periods (trading days, comma-separated)", value="21,63,126,252")
    holding_periods = tuple(int(x.strip()) for x in holding_periods_text.split(",") if x.strip())

    params = dict(
        cheap_pctile=cheap_pctile, expensive_pctile=expensive_pctile,
        volume_z_threshold=volume_z_threshold, require_momentum_confirmation=require_momentum_confirmation,
        filing_lag_days=filing_lag_days, pe_min_periods=pe_min_periods,
        momentum_window=momentum_window, rsi_window=rsi_window,
        dma_short=dma_short, dma_long=dma_long,
        price_start=price_start, holding_periods=holding_periods,
    )

    if st.button("Run Backtest", type="primary", use_container_width=True):
        valid_configs = [s for s in stock_configs if s["ticker"] and s["xlsx_file"] is not None]
        if not valid_configs:
            st.warning("Add at least one stock with both a ticker and a Screener export before running.")
            return

        summary_rows = []
        for cfg in valid_configs:
            with st.spinner(f"Running {cfg['ticker']}..."):
                result, err = run_single_stock(cfg["ticker"], cfg["xlsx_file"], cfg["exclusions_text"], params)
            if err:
                st.error(f"{cfg['ticker']}: {err}")
                continue

            st.subheader(cfg["ticker"])
            render_stock_chart(cfg["ticker"], result["merged"])

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Heavy Buying -- forward return evaluation**")
                st.dataframe(result["results"]["buy_signal_eval"], use_container_width=True, hide_index=True)
            with c2:
                st.markdown("**Heavy Selling -- forward return evaluation**")
                st.dataframe(result["results"]["sell_signal_eval"], use_container_width=True, hide_index=True)

            buy_eval = result["results"]["buy_signal_eval"]
            sell_eval = result["results"]["sell_signal_eval"]
            summary_rows.append({
                "Ticker": cfg["ticker"],
                "Heavy Buying Days": int(result["merged"]["heavy_buying"].sum()),
                "Heavy Selling Days": int(result["merged"]["heavy_selling"].sum()),
                "Buy Avg Return (63d)": buy_eval.loc[buy_eval["holding_period"] == "fwd_ret_63d", "avg_signal_return"].values[0] if "fwd_ret_63d" in buy_eval["holding_period"].values else None,
                "Sell Avg Return (63d)": sell_eval.loc[sell_eval["holding_period"] == "fwd_ret_63d", "avg_signal_return"].values[0] if "fwd_ret_63d" in sell_eval["holding_period"].values else None,
            })

        if summary_rows:
            st.header("Multi-stock summary")
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
