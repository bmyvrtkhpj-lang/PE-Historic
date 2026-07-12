"""
technical_indicators.py
------------------------
Layer 2: TECHNICAL SIGNAL

These define "heavy buying/selling" quantitatively -- both volume-based and
price-based components combined, per your requirement.

All functions take/return pandas Series indexed by date so they compose
directly onto the price/volume DataFrame from data_pipeline.py.
"""

import numpy as np
import pandas as pd


def rolling_volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    How many standard deviations today's volume is above/below its own
    recent rolling average. This is the core "heavy" definition -- a
    z-score of 2 means volume is a genuine spike, not routine noise.
    """
    roll_mean = volume.rolling(window, min_periods=window).mean()
    roll_std = volume.rolling(window, min_periods=window).std(ddof=1)
    return (volume - roll_mean) / roll_std.replace(0, np.nan)


def price_momentum(price: pd.Series, window: int = 10) -> pd.Series:
    """Simple rate-of-change momentum over `window` trading days."""
    return price.pct_change(window)


def rsi(price: pd.Series, window: int = 14) -> pd.Series:
    """
    Simple-moving-average RSI (NOT Wilder's smoothed/exponential variant --
    flagging this explicitly since the two give slightly different values
    and are often confused; SMA-based is simpler to reason about and audit,
    which matters for an academic report).
    """
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    # where avg_loss is 0 and avg_gain > 0, RSI should be 100 (all gains, no losses)
    result = result.where(~((avg_loss == 0) & (avg_gain > 0)), 100)
    return result


def dma_regime(price: pd.Series, short_window: int = 50, long_window: int = 200) -> pd.Series:
    """Bull/Bear regime filter -- same logic as the Bull & Bear tab in your CAPM dashboard."""
    short_dma = price.rolling(short_window, min_periods=short_window).mean()
    long_dma = price.rolling(long_window, min_periods=long_window).mean()
    regime = pd.Series(np.where(short_dma >= long_dma, "Bull", "Bear"), index=price.index)
    regime[short_dma.isna() | long_dma.isna()] = None
    return regime


def obv(price: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume -- distinguishes a SUSTAINED buying/selling phase from
    a single-day spike, which matters specifically for "heavy buying/selling"
    rather than routine volatility.
    """
    direction = np.sign(price.diff()).fillna(0)
    return (direction * volume).cumsum()


def add_all_technical_indicators(df: pd.DataFrame,
                                  volume_window: int = 20,
                                  momentum_window: int = 10,
                                  rsi_window: int = 14,
                                  dma_short: int = 50,
                                  dma_long: int = 200) -> pd.DataFrame:
    """
    df must contain 'price' and 'volume' columns (from data_pipeline output).
    Returns df with added columns: volume_zscore, momentum, rsi, regime, obv.
    """
    out = df.copy()
    out["volume_zscore"] = rolling_volume_zscore(out["volume"], volume_window)
    out["momentum"] = price_momentum(out["price"], momentum_window)
    out["rsi"] = rsi(out["price"], rsi_window)
    out["regime"] = dma_regime(out["price"], dma_short, dma_long)
    out["obv"] = obv(out["price"], out["volume"])
    return out
