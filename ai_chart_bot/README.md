AI Chart Pattern Bot

Quickstart

1) Create venv and install deps

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r /workspace/ai_chart_bot/requirements.txt
```

2) Run a scan

```bash
python -m ai_chart_bot.cli scan AAPL --period 1y --interval 1d
```

Notes

- Fetches OHLCV (yfinance), builds RSI/MACD/Bollinger/ATR, detects simple patterns, trains a small classifier, and suggests Buy/Sell/Wait with rationale. Educational use only.
