import numpy as np
import pandas as pd


def expanding_percentile_rank(series: pd.Series, min_periods: int = 252) -> pd.Series:
    """
    Generic expanding-window percentile rank of a series vs its OWN history
    up to and including that day (avoids lookahead bias -- a full-sample
    rank would let future extremes influence today's percentile).
    Used for both PE (own-history valuation regime) and now delivery
    percentage (own-history relative accumulation/distribution threshold),
    so both axes of the quadrant use the identical, consistent methodology.
    """
    def _pctrank(x):
        return x.rank(pct=True).iloc[-1]
    return series.expanding(min_periods=min_periods).apply(_pctrank, raw=False)


def pe_percentile_rank(pe: pd.Series, min_periods: int = 252) -> pd.Series:
    return expanding_percentile_rank(pe, min_periods)


def delivery_percentile_rank(delivery_pct: pd.Series, min_periods: int = 252) -> pd.Series:
    """
    Percentile rank of delivery% vs the STOCK'S OWN history (not an
    absolute 50% threshold) -- per the refinement that different stocks
    have structurally different typical delivery% baselines (some are
    naturally high-delivery, some naturally more speculative/liquid), so a
    flat 50% isn't a fair comparison across stocks. Using the stock's own
    50th percentile as the accumulation/distribution split keeps the "50%"
    framing your sir wanted, but makes it relative and fair per-stock.
    """
    return expanding_percentile_rank(delivery_pct, min_periods)


def generate_signals(df, cheap_pctile, expensive_pctile, volume_z_threshold, require_momentum_confirmation, momentum_threshold=0.0):
    """
    momentum_threshold: minimum ABSOLUTE momentum required to count as
    confirmation (e.g. 0.02 = momentum must be beyond +/-2%, not just any
    tiny positive/negative wiggle). Default 0.0 preserves the old
    "any sign counts" behaviour, for backward compatibility.
    """
    out = df.copy()
    momentum_buy_ok = (out["momentum"] > momentum_threshold) if require_momentum_confirmation else True
    momentum_sell_ok = (out["momentum"] < -momentum_threshold) if require_momentum_confirmation else True

    out["heavy_buying"] = ((out["pe_percentile"] <= cheap_pctile) & (out["volume_zscore"] >= volume_z_threshold) & momentum_buy_ok).fillna(False)
    out["heavy_selling"] = ((out["pe_percentile"] >= expensive_pctile) & (out["volume_zscore"] >= volume_z_threshold) & momentum_sell_ok).fillna(False)
    return out


def generate_ablation_signals(df, cheap_pctile, expensive_pctile, volume_z_threshold, require_momentum_confirmation, momentum_threshold=0.0):
    out = df.copy()
    momentum_buy_ok = (out["momentum"] > momentum_threshold) if require_momentum_confirmation else True
    momentum_sell_ok = (out["momentum"] < -momentum_threshold) if require_momentum_confirmation else True
    out["buy_pe_only"] = (out["pe_percentile"] <= cheap_pctile).fillna(False)
    out["buy_technical_only"] = ((out["volume_zscore"] >= volume_z_threshold) & momentum_buy_ok).fillna(False)
    out["buy_combined"] = (out["buy_pe_only"] & out["buy_technical_only"])
    out["sell_pe_only"] = (out["pe_percentile"] >= expensive_pctile).fillna(False)
    out["sell_technical_only"] = ((out["volume_zscore"] >= volume_z_threshold) & momentum_sell_ok).fillna(False)
    out["sell_combined"] = (out["sell_pe_only"] & out["sell_technical_only"])
    return out
