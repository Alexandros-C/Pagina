from __future__ import annotations

import sys
from typing import Optional

import click

from .data import fetch_ohlcv
from .features import build_features
from .patterns import detect_patterns
from .model import train_and_infer


@click.group()
def cli() -> None:
    """AI Chart Pattern Bot CLI"""


@cli.command()
@click.argument("symbol", type=str)
@click.option("--period", default="1y", show_default=True, help="Data lookback period")
@click.option("--interval", default="1d", show_default=True, help="Bar interval")
@click.option("--horizon", default=5, show_default=True, help="Forecast horizon in days")
def scan(symbol: str, period: str, interval: str, horizon: int) -> None:
    """Scan SYMBOL and output suggested action with rationale."""
    try:
        price_df = fetch_ohlcv(symbol, period=period, interval=interval)
    except Exception as exc:
        click.echo(f"Error fetching data: {exc}")
        sys.exit(1)

    # Feature build for consistency (even if model does it internally)
    _ = build_features(price_df)

    # Pattern detectors
    pattern_signals = detect_patterns(price_df)

    # Model inference
    model_result = train_and_infer(price_df, horizon_days=horizon)

    # Aggregate rationale
    lines = [
        f"Symbol: {symbol}",
        f"Suggested Action: {model_result.action}",
        f"Model Prob Up: {model_result.proba_up:.2%} | Down: {model_result.proba_down:.2%}",
    ]
    for sig in pattern_signals:
        lines.append(f"Pattern: {sig.name} ({sig.direction}, conf={sig.confidence:.0%}) - {sig.description}")
    for r in model_result.rationale:
        lines.append(f"Reason: {r}")

    click.echo("\n".join(lines))


def main() -> None:
    cli(standalone_mode=True)


if __name__ == "__main__":
    main()

