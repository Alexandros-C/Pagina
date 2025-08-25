from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class PatternSignal:
    name: str
    direction: str  # "bullish" | "bearish" | "neutral"
    confidence: float  # 0..1
    description: str


def _find_local_extrema(series: pd.Series, order: int = 3) -> tuple[List[int], List[int]]:
    """Return indices of local minima and maxima using a simple window rule.

    A point is a local max if it is the maximum within +/- order window.
    """
    highs: List[int] = []
    lows: List[int] = []
    values = series.values
    for i in range(order, len(values) - order):
        window = values[i - order : i + order + 1]
        if values[i] == np.max(window):
            highs.append(i)
        if values[i] == np.min(window):
            lows.append(i)
    return lows, highs


def detect_double_top_bottom(df: pd.DataFrame, tolerance: float = 0.01) -> Optional[PatternSignal]:
    close = df["Close"]
    lows, highs = _find_local_extrema(close, order=3)
    if len(highs) >= 2:
        last_two = highs[-2:]
        a, b = close.iloc[last_two[0]], close.iloc[last_two[1]]
        if abs(a - b) / ((a + b) / 2) <= tolerance and last_two[1] > last_two[0]:
            return PatternSignal(
                name="Double Top",
                direction="bearish",
                confidence=0.6,
                description="Two recent swing highs at similar levels suggest resistance and potential pullback.",
            )
    if len(lows) >= 2:
        last_two = lows[-2:]
        a, b = close.iloc[last_two[0]], close.iloc[last_two[1]]
        if abs(a - b) / ((a + b) / 2) <= tolerance and last_two[1] > last_two[0]:
            return PatternSignal(
                name="Double Bottom",
                direction="bullish",
                confidence=0.6,
                description="Two recent swing lows at similar levels suggest support and potential bounce.",
            )
    return None


def detect_head_and_shoulders(df: pd.DataFrame, tolerance: float = 0.02) -> Optional[PatternSignal]:
    close = df["Close"].values
    idx = np.arange(len(close))
    if len(close) < 30:
        return None
    # crude approach: look at last ~30 bars
    window = close[-30:]
    wi = idx[-30:]
    peak = np.argmax(window)
    if 5 < peak < len(window) - 5:
        left = window[:peak]
        right = window[peak + 1 :]
        if len(left) > 3 and len(right) > 3:
            left_peak = np.max(left)
            right_peak = np.max(right)
            if abs(left_peak - right_peak) / ((left_peak + right_peak) / 2) <= tolerance and window[peak] > left_peak and window[peak] > right_peak:
                return PatternSignal(
                    name="Head and Shoulders",
                    direction="bearish",
                    confidence=0.55,
                    description="Central higher peak with two similar shoulders suggests topping formation.",
                )
    # Inverse
    trough = np.argmin(window)
    if 5 < trough < len(window) - 5:
        left = window[:trough]
        right = window[trough + 1 :]
        if len(left) > 3 and len(right) > 3:
            left_trough = np.min(left)
            right_trough = np.min(right)
            if abs(left_trough - right_trough) / ((left_trough + right_trough) / 2) <= tolerance and window[trough] < left_trough and window[trough] < right_trough:
                return PatternSignal(
                    name="Inverse Head and Shoulders",
                    direction="bullish",
                    confidence=0.55,
                    description="Central lower trough with two similar shoulders suggests bottoming formation.",
                )
    return None


def detect_triangle(df: pd.DataFrame, lookback: int = 40, tolerance: float = 0.02) -> Optional[PatternSignal]:
    if len(df) < lookback:
        return None
    window = df.iloc[-lookback:]
    highs = window["High"].values
    lows = window["Low"].values
    x = np.arange(lookback)
    # Fit simple linear trends for highs and lows
    coef_high = np.polyfit(x, highs, 1)
    coef_low = np.polyfit(x, lows, 1)
    slope_high = coef_high[0]
    slope_low = coef_low[0]
    narrowing = (highs.max() - lows.min()) > 0 and (highs[-1] - lows[-1]) / (highs.max() - lows.min()) < 0.6
    if narrowing and slope_high < 0 and slope_low > 0:
        return PatternSignal(
            name="Symmetric Triangle",
            direction="neutral",
            confidence=0.5,
            description="Converging highs and lows indicate consolidation; watch for breakout.",
        )
    if narrowing and slope_high < 0 and slope_low >= -tolerance:
        return PatternSignal(
            name="Descending Triangle",
            direction="bearish",
            confidence=0.52,
            description="Lower highs against flat to slightly rising lows suggest bearish pressure.",
        )
    if narrowing and slope_low > 0 and slope_high <= tolerance:
        return PatternSignal(
            name="Ascending Triangle",
            direction="bullish",
            confidence=0.52,
            description="Higher lows against flat to slightly falling highs suggest bullish pressure.",
        )
    return None


def detect_patterns(df: pd.DataFrame) -> List[PatternSignal]:
    signals: List[PatternSignal] = []
    for detector in (detect_double_top_bottom, detect_head_and_shoulders, detect_triangle):
        try:
            sig = detector(df)
            if sig is not None:
                signals.append(sig)
        except Exception:
            continue
    return signals

