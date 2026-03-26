# Stock Backtester - Quant Market Analytics Dashboard

A comprehensive Streamlit-based quantitative trading platform with backtesting engine and interactive trading simulator. Analyze market data, backtest strategies, and practice manual trading with real-time feedback.

## 🆕 Latest Update (March 27, 2026)
- Repository successfully uploaded to GitHub
- Ready for community contributions and feature enhancements

## ✨ New Features (v1.1.0)

### 🔍 Enhanced Stock Search & Discovery
- **Smart Search**: Find stocks by company name or ticker symbol
- **Popular Stocks**: Categorized collections (Tech, Financial, Healthcare, etc.)
- **Quick Selection**: One-click stock selection with auto-fill
- **Stock Info**: Detailed company information and market data

### 🎮 Interactive Trading Simulator
- **Manual Trading**: Buy/sell with real-time P&L tracking
- **Time Controls**: Navigate through historical data day-by-day
- **Live Metrics**: Cash, positions, equity, unrealized P&L
- **Trade History**: Complete log with CSV export
- **Performance Analysis**: Sharpe ratio, win rate, drawdown tracking
- **Visual Feedback**: Trade markers on interactive charts

### 🎯 Dual Mode Interface
- **Backtesting Mode**: Automated strategy testing (existing)
- **Simulator Mode**: Manual trading practice (new)
- **Unified Data**: Same market data for both modes
- **Mode Switching**: Seamless transition between analysis and practice

## Features

### 📊 Dashboard Features
- **Multi-ticker data analysis** with yfinance integration
- **Advanced charting** with candlesticks, volume, and technical indicators
- **Technical indicators**: Moving Averages (MA50/MA200), RSI, MACD, Bollinger Bands
- **Market correlation analysis** with interactive heatmaps
- **Drawdown tracking** across multiple assets
- **Real-time metrics** for market analysis

### 🤖 Backtesting Engine
- **Multiple trading strategies**:
  - MA Crossover (MA50 > MA200)
  - RSI Threshold (Buy RSI<30, Sell RSI>70)
  - RSI Mean-Reversion (Buy/Sell crossing RSI 50)
  - Bollinger Bands (Buy lower band, Sell upper band)
  
- **Flexible trading modes**:
  - Day Trading (holding_period = 0)
  - Swing Trading (holding_period = 1-5 days)
  - Position Trading (holding_period = 20+ days)
  
- **Position sizing**:
  - Fixed: All-in/out (0 or 1 position)
  - Dynamic: Continuous 0-1 position sizing
  
- **Transaction costs**: Configurable fee percentage per trade

- **Performance metrics**:
  - Total Return (%)
  - Sharpe Ratio (interval-aware annualization)
  - Max Drawdown (%)
  - Win Rate (% profitable days)
  - Buy-and-Hold comparison

- **Signal visualization**: Entry/exit markers on candlestick charts
  - 🟢 Green diamonds = Buy signals
  - 🔴 Red X = Sell signals

### 📈 Interval Support
- **Data intervals**: 1m, 5m, 15m, 1h, 1d
- **Sharpe ratio annualization** adjusted per interval:
  - 1d: 252 trading days/year
  - 1h: 252 × 6.5 hours/day
  - 5m: 252 × 6.5 × 12 bars/hour
  - 1m: 252 × 6.5 × 60 bars/hour

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd stock-backtester
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the dashboard**
```bash
cd market_dashboard
streamlit run market_dashboard/dashboard.py
```

The app will open at `http://localhost:8501`

## Project Structure

```
stock-backtester/
├── market_dashboard/
│   ├── dashboard.py              # Main Streamlit app
│   ├── modules/
│   │   ├── data.py              # Data download (yfinance)
│   │   ├── indicators.py        # Technical indicators (MA, RSI, MACD, BB)
│   │   ├── strategies.py        # Backtesting strategies (440+ lines)
│   │   ├── portfolio.py         # Portfolio metrics & calculations
│   │   ├── utils.py             # Utility functions (returns, correlation)
│   │   └── __pycache__/
├── requirements.txt
└── README.md
```

## Usage Guide

### 1. Data Selection
- **Tickers**: Enter comma-separated symbols (e.g., `AAPL,MSFT,NVDA`)
- **Interval**: Select candle frequency (1d, 1h, 5m, etc.)
- **Date Range**: Set start and end dates for analysis

### 2. Dashboard Views
Toggle to display/hide:
- **Advanced Chart**: Candlesticks + indicators + volume + MACD + RSI
- **Drawdown**: Historical drawdown comparison across tickers
- **Correlation Heatmap**: Asset correlation matrix

### 3. Backtesting Setup

#### Strategy Selection
1. Choose strategy from dropdown:
   - "None" (no backtest)
   - "MA Crossover"
   - "RSI (Threshold)"
   - "RSI (Mean-Reversion)"
   - "Bollinger Bands"

#### Backtest Period
- Set independent date range (separate from dashboard dates)
- Useful for testing on specific sub-periods

#### Position Configuration
- **Position Sizing**: Fixed (all-in/out) or Dynamic (0-1 scaling)
- **Hold Days**: 0 (day trade) to 252 (position trade)
  - 0: Exit same bar
  - 1-5: Swing trading (multi-day holds)
  - 20+: Position trading (extended trends)

#### Cost Parameters
- **Transaction Fee**: 0.0% - 1.0% per trade
  - Example: 0.1% = $1 fee on $1,000 entry
- **Sharpe Annualization**: Select calculation method
  - "1d (252 days/yr)": Fixed annualization
  - "1h", "5m", "1m": Interval-specific calculation

### 4. Results Interpretation

#### Metrics Panel (4 columns)
- **Total Return**: Strategy profit/loss in percent
- **Sharpe Ratio**: Risk-adjusted return (higher = better)
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: % of days with positive returns

#### Chart Visualization
- **Row 1**: Price with MA50, MA200, Bollinger Bands + entry/exit markers
- **Row 2**: Trading volume
- **Row 3**: MACD line and signal line
- **Row 4**: RSI (0-100 scale)
- **Row 5** (backtest only): Strategy equity vs. buy-and-hold

## Strategy Details

### MA Crossover
**Signal Logic**: Buy when MA50 > MA200, Sell when MA50 ≤ MA200
```
Use Case: Trend-following, position trading
Best For: Strong trending markets
Parameters: holding_period ≥ 5 (let trends develop)
```

### RSI Threshold
**Signal Logic**: Buy when RSI < 30, Sell when RSI > 70
```
Use Case: Mean-reversion, swing trading
Best For: Range-bound markets
Parameters: holding_period = 1-3 (short-term bounces)
```

### RSI Mean-Reversion
**Signal Logic**: Buy when RSI crosses above 50, Sell when crosses below 50
```
Use Case: Momentum mean-reversion
Best For: Medium-term swings
Parameters: holding_period = 2-5 (capture reversal)
```

### Bollinger Bands
**Signal Logic**: Buy at lower band, Sell at upper band
```
Use Case: Volatility-based mean-reversion
Best For: High-volatility assets
Parameters: holding_period = 1-2 (quick reversions)
```

## Module Documentation

### `modules/strategies.py` (440+ lines)

#### Base Strategy Class
```python
class Strategy(ABC):
    def __init__(self, holding_period=0, position_type="fixed", fee_pct=0.0)
    def generate_signals(price, indicators_dict) -> pd.Series
    def compute_positions_and_equity(signals, close, initial_equity=100) -> dict
    def compute_metrics(equity_series, daily_returns, interval="1d") -> dict
```

#### Strategy Implementations
- `MovingAverageCrossover(Strategy)`
- `RSIStrategy(Strategy, mode="threshold"|"mean_reversion")`
- `BollingerBandsStrategy(Strategy)`

#### Utilities
- `buy_hold_equity(close, initial_equity=100)`: Reference equity curve

### `modules/portfolio.py`

```python
sharpe_ratio(returns, interval="1d", risk_free_rate=0.02) -> float
win_rate(daily_returns) -> float  # 0-100%
max_drawdown(price) -> float
portfolio_returns(returns, weights) -> pd.Series
```

### `modules/indicators.py`

```python
moving_averages(price) -> (ma50, ma200)
rsi(close, period=14) -> pd.Series
macd(close) -> (macd_line, signal)
bollinger(close) -> (upper, lower)
```

### `modules/data.py`

```python
download_data(tickers, start, end, interval) -> pd.DataFrame
# Returns: MultiIndex for multiple tickers, DataFrame for single ticker
```

## Example Workflows

### Example 1: Test a Day-Trading RSI Strategy
1. Select tickers: `AAPL`
2. Interval: `1h` (hourly candles)
3. Strategy: `RSI (Threshold)`
4. Backtest dates: Last 30 days
5. **Hold Days**: `0` (day trade)
6. **Transaction Fee**: `0.1%`
7. **Sharpe**: `1h` (hourly-adjusted)
8. **Result**: See intraday signals and metrics

### Example 2: Compare Swing Trading Strategies
1. Select ticker: `MSFT`
2. Interval: `1d` (daily candles)
3. Backtest dates: Full year (e.g., 2023-2024)
4. Run multiple backtests with:
   - MA Crossover (Hold=5 days)
   - RSI Threshold (Hold=3 days)
   - Bollinger Bands (Hold=2 days)
5. Compare equity curves and Sharpe ratios

### Example 3: Portfolio Analysis
1. Select tickers: `AAPL,MSFT,NVDA,TSLA,SPY`
2. View drawdown across all tickers
3. Check correlation heatmap (mid/long-term relationships)
4. Select one ticker for individual strategy backtest

## Performance Tips

- **Faster backtests**: Use daily interval (`1d`) instead of intraday
- **Reduce data**: Backtest on shorter date ranges first (e.g., 6 months)
- **Check signals**: Visually verify entry/exit markers match strategy logic
- **Compare fees**: Test with 0.0% fee first, then add realistic costs

## Data Source & API

- **Data Provider**: Yahoo Finance (via `yfinance`)
- **Auto-adjustment**: Dividend & split adjusted prices
- **Update Frequency**: Live (fetched on app startup)

## Limitations & Considerations

⚠️ **Important Notes**
- **Backtests are historical**: Past performance ≠ future results
- **No slippage modeling**: Assumes entry at close price
- **No partial fills**: Position size is 0 or 1 (fixed mode)
- **Limited order types**: Market orders only (no limit/stop)
- **Timezone**: Uses market timezone of asset
- **Indicator lag**: First 200 bars (MA200) produce NaN values

## Troubleshooting

### App won't start
```bash
# Check dependencies
pip list | grep -E "streamlit|pandas|plotly|yfinance"

# Reinstall if needed
pip install -r requirements.txt --upgrade
```

### Data download errors
- Check internet connection
- Verify ticker symbols are valid (e.g., AAPL not Apple)
- Ensure date range includes trading days (weekends excluded)

### Chart not displaying
- Toggle "Advanced Chart" OFF then ON
- Check browser cache (Ctrl+Shift+Delete)
- Try different ticker with more data

### Backtest shows no signals
- Verify date range has sufficient data (first 200 bars may be NaN for indicators)
- Check if selected strategy requires specific market conditions
- Try changing holding period or fees to see if it helps debug

## Future Enhancements

- [ ] Multi-position portfolio backtesting
- [ ] Parameter optimization (grid search over holding_period, MA periods)
- [ ] Trade logging export (CSV with entry/exit details)
- [ ] More strategies: MACD crosses, Stochastic, ATR breakouts
- [ ] Risk management: Stop-loss, take-profit automation
- [ ] Walk-forward analysis for parameter robustness
- [ ] Monte Carlo simulation for drawdown estimation
- [ ] Live trading integration (paper trading via API)

## Contributing

Contributions are welcome! To add a new strategy:

1. Extend `Strategy` base class in `modules/strategies.py`
2. Implement `generate_signals()` method
3. Test with sample data
4. Add to sidebar dropdown in `dashboard.py`

Example:
```python
class MyStrategy(Strategy):
    def generate_signals(self, price, indicators_dict):
        # Your logic here
        return signal_series
```

## License

MIT License - See LICENSE file for details

## Support

For issues, questions, or feature requests, please create an issue on GitHub or contact the maintainer.

---

**Dashboard Version**: 1.0.0  
**Last Updated**: March 2026  
**Python**: 3.8+  
**Streamlit**: 1.25+
