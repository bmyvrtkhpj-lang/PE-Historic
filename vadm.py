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
