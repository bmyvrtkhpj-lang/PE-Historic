"""
backtest.py
-----------
Layer 4: BACKTEST + EVALUATION

For each signal day (heavy_buying / heavy_selling), measures what actually
happened to the price over several forward holding periods, and compares
that against the stock's overall baseline forward-return distribution using
a two-sample t-test -- so the report can defend "this zone shows heavy
buying" statistically, not just visually.
"""

import pandas as pd
from scipy import stats


def forward_returns(price: pd.Series, holding_periods=(21, 63, 126, 252)) -> pd.DataFrame:
    """
    holding_periods are in TRADING days by default:
    21 ~ 1 month, 63 ~ 3 months, 126 ~ 6 months, 252 ~ 1 year.
    Uses price.shift(-h) deliberately -- this IS forward-looking by design,
    because we're measuring what happens AFTER a signal, which is exactly
    what a backtest needs (not a bug, unlike the lookahead risk in PE ranking).
    """
    fwd = pd.DataFrame(index=price.index)
    for h in holding_periods:
        fwd[f"fwd_ret_{h}d"] = price.shift(-h) / price - 1
    return fwd


def evaluate_signal(signal_mask: pd.Series, fwd_returns: pd.DataFrame, min_signal_obs: int = 5) -> pd.DataFrame:
    """
    Returns a tidy DataFrame, one row per holding period, with:
    n_signals, avg_signal_return, avg_baseline_return, win_rate, t_stat, p_value.

    p_value < 0.05 is a common convention for "statistically distinguishable
    from baseline" -- but with multiple holding periods and multiple stocks
    tested, correcting for multiple comparisons (e.g. Bonferroni) is worth
    doing before making strong claims in the report, rather than treating
    each p-value in isolation.
    """
    rows = []
    for col in fwd_returns.columns:
        signal_rets = fwd_returns.loc[signal_mask.reindex(fwd_returns.index, fill_value=False), col].dropna()
        baseline_rets = fwd_returns[col].dropna()

        if len(signal_rets) < min_signal_obs:
            rows.append({
                "holding_period": col,
                "n_signals": len(signal_rets),
                "avg_signal_return": signal_rets.mean() if len(signal_rets) else None,
                "avg_baseline_return": baseline_rets.mean(),
                "win_rate": (signal_rets > 0).mean() if len(signal_rets) else None,
                "t_stat": None,
                "p_value": None,
                "note": "too few signal observations for a reliable t-test",
            })
            continue

        t_stat, p_val = stats.ttest_ind(signal_rets, baseline_rets, equal_var=False)
        rows.append({
            "holding_period": col,
            "n_signals": len(signal_rets),
            "avg_signal_return": signal_rets.mean(),
            "avg_baseline_return": baseline_rets.mean(),
            "win_rate": (signal_rets > 0).mean(),
            "t_stat": t_stat,
            "p_value": p_val,
            "note": "",
        })
    return pd.DataFrame(rows)


def run_backtest(df: pd.DataFrame, holding_periods=(21, 63, 126, 252)) -> dict:
    """
    df must contain: 'price', 'heavy_buying', 'heavy_selling' columns.
    Returns {'buy_signal_eval': DataFrame, 'sell_signal_eval': DataFrame}.
    """
    fwd = forward_returns(df["price"], holding_periods)
    return {
        "buy_signal_eval": evaluate_signal(df["heavy_buying"], fwd),
        "sell_signal_eval": evaluate_signal(df["heavy_selling"], fwd),
    }


def run_ablation_backtest(df: pd.DataFrame, holding_periods=(21, 63, 126, 252)) -> dict:
    """
    df must contain the ablation columns (buy_pe_only, buy_delivery_only,
    buy_combined, and the sell_* equivalents) -- these now test H1 (PE main
    effect) vs H2 (delivery main effect) vs H3-manifestation (VADM/quadrant
    combined), replacing the old volume+momentum-based ablation.

    Returns a dict of 6 DataFrames, one per variant, so you can compare
    avg_signal_return and win_rate side by side and see whether combining
    PE + delivery is actually beating either one alone.
    """
    fwd = forward_returns(df["price"], holding_periods)
    variants = ["buy_pe_only", "buy_delivery_only", "buy_combined",
                "sell_pe_only", "sell_delivery_only", "sell_combined"]
    return {v: evaluate_signal(df[v], fwd) for v in variants if v in df.columns}


def test_h3_interaction(df, holding_period=63):
    """
    Formal statistical test of H3 (core hypothesis): the effect of
    delivery-based buying pressure on forward return is NOT uniform across
    the PE valuation regime -- i.e. an INTERACTION effect, not two
    independent (additive) main effects.

    Regression:
        forward_return ~ pe_percentile + delivery_flow + (pe_percentile * delivery_flow)

    The INTERACTION term's coefficient is what actually tests H3. If it's
    statistically significant (p < 0.05, conventionally), that supports H3:
    delivery flow's predictive power genuinely differs depending on whether
    the stock is cheap or expensive. If NOT significant, H1/H2 (the two
    main effects) might still hold individually, but there's no regression
    evidence the two combine non-additively -- i.e. VADM's multiplicative
    design wouldn't be statistically supported by this specific test, even
    if individual examples look reasonable.

    df must contain: 'price', 'pe_percentile', 'delivery_flow'.
    Returns (statsmodels RegressionResults, None) or (None, error_message).
    """
    import statsmodels.api as sm

    data = df[["price", "pe_percentile", "delivery_flow"]].copy()
    data["fwd_return"] = data["price"].shift(-holding_period) / data["price"] - 1
    data = data.dropna()

    if len(data) < 30:
        return None, f"Only {len(data)} usable observations -- not enough for a reliable regression (need 30+)."

    X = pd.DataFrame({
        "pe_percentile": data["pe_percentile"].values,
        "delivery_flow": data["delivery_flow"].values,
    }, index=data.index)
    X["interaction"] = X["pe_percentile"] * X["delivery_flow"]
    X = sm.add_constant(X)
    y = data["fwd_return"]

    model = sm.OLS(y, X).fit()
    return model, None
