from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange


FEATURE_COLUMNS: List[str] = [
    "return_1d",
    "return_5d",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_high_band_dist",
    "bb_low_band_dist",
    "atr_14",
]


def build_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical feature set for model and heuristics.

    Returns a new DataFrame containing the original OHLCV columns plus feature columns.
    """
    df = price_df.copy()

    # Returns
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_5d"] = df["Close"].pct_change(5)

    # RSI
    rsi = RSIIndicator(close=df["Close"], window=14)
    df["rsi_14"] = rsi.rsi()

    # MACD
    macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2.0)
    df["bb_high_band"] = bb.bollinger_hband()
    df["bb_low_band"] = bb.bollinger_lband()
    df["bb_high_band_dist"] = (df["bb_high_band"] - df["Close"]) / df["Close"]
    df["bb_low_band_dist"] = (df["Close"] - df["bb_low_band"]) / df["Close"]

    # ATR
    atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["atr_14"] = atr.average_true_range()

    return df


def build_forward_returns(df: pd.DataFrame, horizon_days: int = 5) -> pd.Series:
    """Compute forward return label used for training.

    Positive if future price is higher than today.
    """
    return df["Close"].shift(-horizon_days).pct_change(horizon_days)

