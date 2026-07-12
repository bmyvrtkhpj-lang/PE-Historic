"""
pe_signal.py
------------
Layer 3: PE BANDING + SIGNAL GENERATION

Turns the raw daily PE series into a PERCENTILE RANK relative to the
stock's OWN history (not a fixed absolute PE number -- raw PE isn't
comparable across stocks or across time for the same stock's changing
growth regime, as discussed).

CRITICAL: uses an EXPANDING window rank, not a full-sample rank. A
full-sample percentile would let PE extremes from LATER years influence
today's percentile -- that is lookahead bias hiding inside a "simple"
statistic. Expanding window means each day's percentile only ever uses
data available up to and including that day.
"""

import numpy as np
import pandas as pd


def pe_percentile_rank(pe: pd.Series, min_periods: int = 252) -> pd.Series:
    """
    Expanding-window percentile rank of PE vs. its own history up to that
    date. min_periods=252 (~1 trading year) means no percentile is produced
    until at least a year of PE history exists, to avoid noisy early-sample
    percentiles from 5-10 data points.

    NOTE: this loop-based expanding().apply() is O(n^2) -- for ~2500 trading
    days (10yr) per stock this is fine (a few seconds), but if you later
    extend the backtest window a lot further, this will need a faster
    implementation (e.g. a running sorted structure). Flagging this now
    rather than after it becomes a real slowdown at 15-20 stocks.
    """
    def _pctrank(x):
        return x.rank(pct=True).iloc[-1]
    return pe.expanding(min_periods=min_periods).apply(_pctrank, raw=False)


def generate_signals(df: pd.DataFrame,
                      cheap_pctile: float = 0.20,
                      expensive_pctile: float = 0.80,
                      volume_z_threshold: float = 1.5,
                      require_momentum_confirmation: bool = True) -> pd.DataFrame:
    """
    df must already contain: 'pe_percentile', 'volume_zscore', 'momentum'
    (from technical_indicators.add_all_technical_indicators + pe_percentile_rank).

    heavy_buying  = cheap valuation zone + volume spike + (optional) positive momentum
    heavy_selling = expensive valuation zone + volume spike + (optional) negative momentum

    These thresholds (0.20 / 0.80 / 1.5) are starting defaults, not
    validated constants -- they should be tuned/checked against your actual
    Nifty 100 sample once real data is in, ideally on an initial in-sample
    period, then confirmed on a held-out later period rather than tuned on
    the full history (to keep the backtest honest).
    """
    out = df.copy()
    momentum_buy_ok = (out["momentum"] > 0) if require_momentum_confirmation else True
    momentum_sell_ok = (out["momentum"] < 0) if require_momentum_confirmation else True

    out["heavy_buying"] = (
        (out["pe_percentile"] <= cheap_pctile)
        & (out["volume_zscore"] >= volume_z_threshold)
        & momentum_buy_ok
    ).fillna(False)

    out["heavy_selling"] = (
        (out["pe_percentile"] >= expensive_pctile)
        & (out["volume_zscore"] >= volume_z_threshold)
        & momentum_sell_ok
    ).fillna(False)

    return out
