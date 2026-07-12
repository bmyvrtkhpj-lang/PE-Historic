# PE + Technical Entry/Exit Signal Framework

A multi-stock framework combining **valuation (PE, own-history percentile)**
with a **technical trigger (volume z-score + price momentum)** to flag
historical "heavy buying" / "heavy selling" zones per stock, then backtests
forward returns from those zones with a statistical (t-test) check against
each stock's own baseline.

## Folder structure

```
.
├── app.py                     # Streamlit app (entry point for deployment)
├── data_pipeline.py            # Layer 1: Screener.in parser + step-function daily PE
├── technical_indicators.py     # Layer 2: volume z-score, momentum, RSI, DMA regime, OBV
├── pe_signal.py                 # Layer 3: PE percentile banding + signal generation
├── backtest.py                  # Layer 4: forward returns + t-test evaluation
└── requirements.txt
```

## Getting the input data per stock

1. Go to the stock's page on Screener.in (needs a free login).
2. Click **Export to Excel** — this downloads the file the app expects.
3. Note the exact Yahoo Finance ticker for the same stock (e.g. `HDFCBANK.NS`,
   `TCS.NS`) — needed for the price/volume side.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a **public** GitHub repo (Community Cloud free tier needs public repos)
   and push all the files above (same folder level, no subfolder).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pick the repo, branch, and set **Main file path** to `app.py`.
4. Deploy — Streamlit Cloud installs `requirements.txt` automatically.

## Known limitations (stated explicitly, not hidden)

- **PE resolution is ANNUAL, not quarterly.** Screener's free export (no
  Prime) only gives ~9-10 recent quarters but 10 years of annual P&L. So EPS
  updates once a year (step function, ~60 days after fiscal year-end — this
  lag is an approximation of the SEBI LODR filing deadline; verify the
  current exact number before relying on it for a formal report). Daily
  price still gives fine resolution; only the EPS denominator is coarse.
- **Corporate actions are NOT auto-detected.** Confirmed on real HDFC Bank
  data: the 2023 HDFC Ltd merger does not show up as an EPS outlier (net
  profit and share count moved together and roughly cancelled out). You
  must supply exclusion windows manually per stock in the app.
- **Signal thresholds (cheap/expensive percentile, volume z-score) are
  starting defaults**, not validated constants. Tune them on an initial
  in-sample period and confirm on a later held-out period, rather than
  tuning on the full history used for evaluation.
- **Multiple comparisons**: testing several holding periods × several
  stocks produces several p-values. Treat an individual p < 0.05 with
  caution until you've corrected for multiple testing (e.g. Bonferroni)
  across the full run.
- **Survivorship bias**: the app only works on stocks you feed it (their
  current listed history). Delisted/merged-away names from your chosen
  universe won't appear unless added explicitly — state this as a limitation
  in the report rather than treating the sample as complete.
- **RSI** implemented here is the simple-moving-average version, not
  Wilder's smoothed/exponential variant — the two differ slightly; this is
  flagged so it isn't assumed to be the "standard" TradingView RSI without
  checking.
