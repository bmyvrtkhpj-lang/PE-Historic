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
    def _pctrank(x):
        return x.rank(pct=True).iloc[-1]
    return pe.expanding(min_periods=min_periods).apply(_pctrank, raw=False)

def generate_signals(df, cheap_pctile, expensive_pctile, volume_z_threshold, require_momentum_confirmation):
    out = df.copy()
    momentum_buy_ok = (out["momentum"] > 0) if require_momentum_confirmation else True
    momentum_sell_ok = (out["momentum"] < 0) if require_momentum_confirmation else True
    
    out["heavy_buying"] = ((out["pe_percentile"] <= cheap_pctile) & (out["volume_zscore"] >= volume_z_threshold) & momentum_buy_ok).fillna(False)
    out["heavy_selling"] = ((out["pe_percentile"] >= expensive_pctile) & (out["volume_zscore"] >= volume_z_threshold) & momentum_sell_ok).fillna(False)
    return out

def generate_ablation_signals(df, cheap_pctile, expensive_pctile, volume_z_threshold, require_momentum_confirmation):
    out = df.copy()
    momentum_buy_ok = (out["momentum"] > 0) if require_momentum_confirmation else True
    momentum_sell_ok = (out["momentum"] < 0) if require_momentum_confirmation else True
    out["buy_pe_only"] = (out["pe_percentile"] <= cheap_pctile).fillna(False)
    out["buy_technical_only"] = ((out["volume_zscore"] >= volume_z_threshold) & momentum_buy_ok).fillna(False)
    out["buy_combined"] = (out["buy_pe_only"] & out["buy_technical_only"])
    out["sell_pe_only"] = (out["pe_percentile"] >= expensive_pctile).fillna(False)
    out["sell_technical_only"] = ((out["volume_zscore"] >= volume_z_threshold) & momentum_sell_ok).fillna(False)
    out["sell_combined"] = (out["sell_pe_only"] & out["sell_technical_only"])
    return out
