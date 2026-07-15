"""
vadm.py
-------
Layer 3 (redesigned per sir's framework, replacing the old single boolean
heavy_buying/heavy_selling signal in pe_signal.py):

  Regime layer  = PE percentile vs OWN history (slow, quarterly)
                  -- pe_signal.pe_percentile_rank()
  Trigger layer = Delivery percentage percentile vs OWN history (fast, daily)
                  -- pe_signal.delivery_percentile_rank()

Both axes now use the IDENTICAL expanding-percentile-rank methodology, split
at each stock's own 50th percentile -- this keeps the "50%" framing your sir
wanted, while being fair across stocks with different typical delivery%
baselines (a flat absolute 50% delivery threshold would not be, since some
stocks are structurally more liquid/speculative than others).

Produces:
  - A 4-state named quadrant classification (Value Confirmation / Value Trap
    Warning / Momentum without Margin of Safety / Reversal-De-rating)
  - VADM, a continuous composite score

H3 (core hypothesis): the effect of delivery-based buying pressure on
forward returns is NOT uniform across PE valuation regime -- this is an
INTERACTION effect. classify_quadrant() below is a simple/visual way to see
this; the formal statistical test of H3 needs a regression with an explicit
interaction term (forward_return ~ pe_percentile + delivery_flow +
pe_percentile*delivery_flow, testing significance of that third
coefficient) -- that regression is NOT implemented here yet, this module
only covers the composite score + visual classification.
"""

import numpy as np
import pandas as pd

from pe_signal import expanding_percentile_rank


def delivery_flow_strength(delivery_percentile: pd.Series, midpoint: float = 0.5) -> pd.Series:
    """
    Normalized distance of delivery PERCENTILE (own-history-relative, 0-1)
    from its own 50th percentile midpoint. Positive = today's delivery% is
    in the upper half of this stock's own history (net accumulation),
    negative = lower half (net distribution).

    Returns a value in [-1, 1]: magnitude = how far from the stock's own
    50/50 split.
    """
    return (delivery_percentile - midpoint) / midpoint


def compute_vadm(pe_percentile: pd.Series, delivery_flow: pd.Series) -> pd.Series:
    """
    VADM (Valuation Adjusted Delivery Momentum).

    IMPORTANT: the source notes define VADM only as
    "f(PE relative percentile, Delivery flow Z-score)" -- no explicit
    mathematical form was given on paper. This is a PROPOSED implementation,
    not a transcription of a given formula -- confirm with your sir before
    treating this as final.

    delivery_flow here is delivery_flow_strength() (own-history relative
    50th-percentile threshold), not an absolute value or rolling z-score.

    Give FULL strength when valuation and flow AGREE (the two decisive
    quadrants), and DAMPEN when they DISAGREE (the two caution quadrants) --
    matching the quadrant names themselves:
      - Value Confirmation (cheap + buying)      -> agreement    -> full strength
      - Reversal/De-rating (expensive + selling) -> agreement    -> full strength
      - Momentum w/o Margin of Safety (expensive + buying) -> disagreement -> dampened
      - Value Trap Warning (cheap + selling)     -> disagreement -> dampened

        multiplier_t = (1 - pe_percentile_t)  if delivery_flow_t >= 0
                     = pe_percentile_t         if delivery_flow_t <  0
        VADM_t = delivery_flow_t * multiplier_t
    """
    multiplier = np.where(delivery_flow >= 0, 1 - pe_percentile, pe_percentile)
    return delivery_flow * multiplier


def classify_quadrant(pe_percentile: pd.Series,
                       delivery_percentile: pd.Series,
                       cheap_pctile: float = 0.5,
                       delivery_pctile_midpoint: float = 0.5) -> pd.Series:
    """
    Classifies each day into one of the 4 named states:
      Cheap + Accumulation      -> "Value Confirmation"
      Cheap + Distribution      -> "Value Trap Warning"
      Expensive + Accumulation  -> "Momentum without Margin of Safety"
      Expensive + Distribution  -> "Reversal/De-rating"

    Both thresholds are the stock's OWN 50th percentile (relative, not
    absolute) -- pe_percentile and delivery_percentile should both already
    be 0-1 expanding-percentile-rank series (from pe_signal.py).
    """
    is_cheap = pe_percentile <= cheap_pctile
    is_accumulating = delivery_percentile > delivery_pctile_midpoint

    conditions = [
        is_cheap & is_accumulating,
        is_cheap & ~is_accumulating,
        (~is_cheap) & is_accumulating,
        (~is_cheap) & (~is_accumulating),
    ]
    labels = ["Value Confirmation", "Value Trap Warning",
              "Momentum without Margin of Safety", "Reversal/De-rating"]

    result = pd.Series(np.select(conditions, labels, default=None), index=pe_percentile.index)
    result[pe_percentile.isna() | delivery_percentile.isna()] = None
    return result


def generate_quadrant_signals(df: pd.DataFrame, vadm_buy_pctile: float = 0.90, vadm_sell_pctile: float = 0.10) -> pd.DataFrame:
    """
    Replaces the old volume+momentum based generate_signals() AND the old
    fixed-threshold VADM version. VADM is THE operational signal, but
    thresholded by its OWN PERCENTILE RANK, not a fixed absolute number.

    WHY: tested on real HDFC Bank data and found VADM's own distribution is
    skewed (median ~0.22, not centered on 0) -- a fixed threshold like 0.15
    sat close to the MEDIAN, so "heavy_buying" fired on >55% of days. That
    defeats the entire point of "heavy" meaning rare/extreme. Using VADM's
    own expanding percentile rank (same methodology as PE and delivery
    percentile elsewhere in this project) makes "heavy" genuinely mean
    "top/bottom X% of this stock's own VADM history" -- adapts correctly
    regardless of a stock's own VADM distribution shape.

    vadm_buy_pctile=0.90: buy signal = VADM in the top 10% of its own history.
    vadm_sell_pctile=0.10: sell signal = VADM in the bottom 10% of its own history.

    df must already contain 'vadm' (from compute_vadm).
    Adds: vadm_percentile, heavy_buying, heavy_selling.
    """
    out = df.copy()
    out["vadm_percentile"] = expanding_percentile_rank(out["vadm"])
    out["heavy_buying"] = (out["vadm_percentile"] >= vadm_buy_pctile).fillna(False)
    out["heavy_selling"] = (out["vadm_percentile"] <= vadm_sell_pctile).fillna(False)
    return out


def generate_quadrant_ablation_signals(df: pd.DataFrame, cheap_pctile: float = 0.5, delivery_pctile_midpoint: float = 0.5) -> pd.DataFrame:
    """
    Replaces the old volume+momentum based generate_ablation_signals().
    Now tests H1 (PE main effect alone) vs H2 (delivery main effect alone)
    vs H3-manifestation (VADM/quadrant combined) directly -- this ablation
    IS the practical test of your three hypotheses side by side (the formal
    statistical H3 test is test_h3_interaction() in backtest.py; this is
    the simpler subgroup-comparison view of the same question).

    df must already contain 'pe_percentile', 'delivery_percentile',
    'heavy_buying', 'heavy_selling' (from generate_quadrant_signals).
    """
    out = df.copy()
    out["buy_pe_only"] = (out["pe_percentile"] <= cheap_pctile).fillna(False)
    out["buy_delivery_only"] = (out["delivery_percentile"] > delivery_pctile_midpoint).fillna(False)
    out["buy_combined"] = out["heavy_buying"]

    out["sell_pe_only"] = (out["pe_percentile"] >= (1 - cheap_pctile)).fillna(False)
    out["sell_delivery_only"] = (out["delivery_percentile"] <= delivery_pctile_midpoint).fillna(False)
    out["sell_combined"] = out["heavy_selling"]
    return out


def signal_onsets(bool_series: pd.Series) -> pd.Series:
    """
    True only on the FIRST day of each consecutive run of True values.

    WHY THIS MATTERS: confirmed on real HDFC Bank data that heavy_selling
    (75 raw signal-days) was really only 21 distinct episodes -- one episode
    ran 29 CONSECUTIVE days. Evaluating every day of a persistent regime as
    an independent observation in a t-test violates the independence
    assumption and inflates apparent statistical significance (forward
    returns on consecutive days are highly overlapping/correlated, not
    independent draws). Use this for statistical evaluation; keep the full
    day-level signal for chart display, where showing every day within an
    episode is still the correct visual.
    """
    s = bool_series.fillna(False)
    return s & ~s.shift(1, fill_value=False)
