# Stock Backtester

A small Streamlit app for downloading price data, showing charts, and backtesting a few common trading strategies.

## Quick start

1. Clone the repository and go to the project folder:
   ```powershell
   git clone <repository-url>
   cd stock-backtester
   ```
2. Install the required packages:
   ```powershell
   python -m pip install -r requirements.txt
   ```
3. Launch the dashboard:
   ```powershell
   streamlit run market_dashboard/dashboard.py
   ```

The app will open in your browser at `http://localhost:8501`.

## What this repo contains

- `market_dashboard/dashboard.py` — the main Streamlit app
- `market_dashboard/modules/` — helper modules for data, indicators, backtests, and portfolio metrics
- `requirements.txt` — the Python dependencies
- `README.md` — this guide
- `LICENSE` — the project license
- `test_integration.py` — a basic integration test that checks the main modules together

## Features

- Download market data from Yahoo Finance
- Calculate common indicators like moving averages, RSI, MACD, and Bollinger Bands
- Run simple backtests for MA crossover, RSI, and Bollinger Band strategies
- See performance metrics such as total return, Sharpe ratio, drawdown, and win rate

## Notes

This repo is kept intentionally small and easy to understand. The main app lives in `market_dashboard/dashboard.py`, and the rest of the code is split into a few straightforward modules.

Old documentation and legacy test scripts have been moved to the `archive/` folder to keep the repository root clean.
