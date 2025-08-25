from __future__ import annotations

from typing import Optional

import pandas as pd
import yfinance as yf


def fetch_ohlcv(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV price data using yfinance.

    Args:
        symbol: The market symbol, e.g., "AAPL", "BTC-USD".
        period: History period, e.g., "1y", "6mo", "5y".
        interval: Bar interval, e.g., "1d", "1h", "5m".

    Returns:
        DataFrame indexed by datetime with columns: Open, High, Low, Close, Volume.
    """
    data = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError(f"No data returned for {symbol} with period={period}, interval={interval}")

    # Normalize columns across yfinance versions and MultiIndex cases
    if isinstance(data.columns, pd.MultiIndex):
        # Typical shape: levels ["Price", "Ticker"]. Try to select the symbol from the last level.
        try:
            data = data.xs(symbol, axis=1, level=-1)
        except Exception:
            # If there's only one ticker in the last level, drop it
            try:
                last_level = data.columns.get_level_values(-1)
                if len(set(last_level)) == 1:
                    data.columns = data.columns.get_level_values(0)
                else:
                    # Fallback: take the first level strings
                    data.columns = [str(t[0]) for t in data.columns]
            except Exception:
                # Last resort: flatten by joining levels
                data.columns = ["_".join([str(p) for p in col if p]) for col in data.columns]
    else:
        # Ensure all names are strings
        data.columns = [str(c) for c in data.columns]

    # Some versions return lowercase; standardize title case for OHLCV
    rename_map = {}
    for c in list(data.columns):
        key = c.strip().lower()
        if key in {"open", "high", "low", "close", "volume", "adj close", "adj_close"}:
            if key in {"adj close", "adj_close"}:
                std = "Adj Close"
            else:
                std = key.title()
            rename_map[c] = std
    if rename_map:
        data = data.rename(columns=rename_map)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Data missing required columns: {missing}")

    data = data.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    return data

