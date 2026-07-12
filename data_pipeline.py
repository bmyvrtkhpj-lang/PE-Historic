"""
data_pipeline.py
----------------
Layer 1: DATA

Turns a Screener.in free "Export to Excel" file into an annual EPS series,
then combines that with a daily price series (from yfinance, fetched in your
own environment -- this sandbox has no internet access to Yahoo Finance) to
build a continuous DAILY PE time series using a step-function EPS assumption.

Why step-function EPS: Screener's free export gives 10 years of ANNUAL
Profit & Loss data, but only ~9-10 recent QUARTERS (no Screener Prime).
So instead of a quarterly PE series (which we can't build reliably for the
full history), we treat EPS as updating once a year, ~60 days after the
fiscal year end (approximate SEBI LODR annual-filing lag -- verify the
current exact deadline before relying on this number), and held flat until
the next annual result. Daily price still gives fine resolution; only the
EPS denominator is coarse. This was verified end-to-end on the real
HDFC Bank export earlier -- the merge_asof step-function mechanic works
correctly (confirmed against known filing dates).

IMPORTANT, confirmed on real data: a merger/corporate-action year can look
completely unremarkable in the EPS series (net profit and share count can
jump together and roughly cancel out), so an automatic "flag big EPS jumps"
rule is NOT reliable. Corporate actions must be supplied manually per stock
via `corporate_action_exclusions`.
"""

import pandas as pd
from openpyxl import load_workbook


def find_section_row(ws, label, max_row=250):
    """Find the row number where column A matches `label` exactly (case-insensitive)."""
    for r in range(1, max_row + 1):
        val = ws.cell(row=r, column=1).value
        if val is not None and str(val).strip().lower() == label.strip().lower():
            return r
    return None


def _row_values(ws, row):
    """Read values across a row (columns B onward) until a blank cell is hit."""
    vals = []
    c = 2
    while True:
        v = ws.cell(row=row, column=c).value
        if v is None:
            break
        vals.append(v)
        c += 1
    return vals


def extract_annual_fundamentals(xlsx_path):
    """
    Pulls annual Report Date, Net Profit, and Adjusted Equity Shares from the
    'Data Sheet' tab of a Screener.in free export, and derives Annual EPS.

    Label-based lookup (not hardcoded row numbers) so this can be reused
    across different company templates -- banks, NBFCs, and manufacturers
    have different P&L layouts on Screener. STILL TEST ON EACH NEW STOCK:
    a template variant this hasn't seen could break the section-boundary
    assumptions below.

    Returns a DataFrame: fiscal_year_end, net_profit_cr, shares_cr,
    annual_eps, yoy_shares_change_pct, company.
    """
    wb = load_workbook(xlsx_path, data_only=True)
    ws = wb["Data Sheet"]

    company_name = ws["B1"].value

    pl_header_row = find_section_row(ws, "PROFIT & LOSS")
    quarters_header_row = find_section_row(ws, "Quarters")
    if pl_header_row is None or quarters_header_row is None:
        raise ValueError(
            "Could not locate 'PROFIT & LOSS' / 'Quarters' section headers -- "
            "sheet layout differs for this stock. Inspect the file manually."
        )

    report_date_row = None
    net_profit_row = None
    for r in range(pl_header_row, quarters_header_row):
        label = ws.cell(row=r, column=1).value
        if label is None:
            continue
        label = str(label).strip()
        if label == "Report Date" and report_date_row is None:
            report_date_row = r
        if label == "Net profit" and net_profit_row is None:
            net_profit_row = r

    if report_date_row is None or net_profit_row is None:
        raise ValueError(
            "Could not find annual 'Report Date' / 'Net profit' rows in the "
            "PROFIT & LOSS block."
        )

    derived_row = find_section_row(ws, "DERIVED:")
    if derived_row is None:
        raise ValueError("Could not find 'DERIVED:' section -- shares data location differs.")

    shares_row = None
    for r in range(derived_row, derived_row + 10):
        label = ws.cell(row=r, column=1).value
        if label and "Adjusted Equity Shares" in str(label):
            shares_row = r
            break
    if shares_row is None:
        raise ValueError("Could not find 'Adjusted Equity Shares in Cr' row.")

    dates = _row_values(ws, report_date_row)
    net_profit = _row_values(ws, net_profit_row)
    shares = _row_values(ws, shares_row)

    n = min(len(dates), len(net_profit), len(shares))
    df = pd.DataFrame({
        "fiscal_year_end": pd.to_datetime(dates[:n]),
        "net_profit_cr": net_profit[:n],
        "shares_cr": shares[:n],
    })
    df["annual_eps"] = df["net_profit_cr"] / df["shares_cr"]
    df["yoy_shares_change_pct"] = df["shares_cr"].pct_change() * 100
    df["company"] = company_name
    return df


def build_step_function_pe(annual_df, daily_price, filing_lag_days=60):
    """
    Combine annual EPS (step function, becomes "known to the market"
    `filing_lag_days` after fiscal year end) with a daily price series to
    build a continuous daily PE series across the full annual history.

    daily_price: pandas Series indexed by date (DatetimeIndex), values = close price.
    Returns a DataFrame indexed by date with columns: price, annual_eps, pe.
    """
    eps_df = annual_df.copy()
    eps_df["effective_date"] = eps_df["fiscal_year_end"] + pd.Timedelta(days=filing_lag_days)
    eps_df = eps_df.sort_values("effective_date")

    price_df = daily_price.rename("price").to_frame().reset_index()
    price_df.columns = ["date", "price"]

    merged = pd.merge_asof(
        price_df.sort_values("date"),
        eps_df[["effective_date", "annual_eps"]].rename(columns={"effective_date": "date"}),
        on="date",
        direction="backward",
    )
    merged["pe"] = merged["price"] / merged["annual_eps"]
    return merged.set_index("date")


def apply_corporate_action_exclusions(pe_df, exclusions):
    """
    exclusions: list of (start_date_str, end_date_str) tuples. PE is set to
    NaN in these windows -- e.g. merger/demerger/large one-off periods that
    won't show up as an automatic EPS outlier (confirmed on HDFC Bank: net
    profit and share count moved together and roughly cancelled out).
    """
    out = pe_df.copy()
    for start, end in exclusions or []:
        out.loc[start:end, "pe"] = None
    return out


def fetch_price_volume(ticker, start, end):
    """
    Fetches daily Close price + Volume via yfinance.
    Using Ticker.history() avoids the MultiIndex column format issues.
    """
    import yfinance as yf
    
    # Use Ticker.history to avoid MultiIndex structure issues
    stock = yf.Ticker(ticker)
    raw = stock.history(start=start, end=end)
    
    # Return empty DataFrame if no data is found
    if raw.empty:
        return pd.DataFrame(columns=["price", "volume"])
        
    out = pd.DataFrame({
        "price": pd.to_numeric(raw["Close"], errors="coerce"),
        "volume": pd.to_numeric(raw["Volume"], errors="coerce"),
    }).dropna()
    
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out
