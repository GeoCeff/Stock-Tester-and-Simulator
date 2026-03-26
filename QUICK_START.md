# Stock Backtester - Quick Start Guide

## 🚀 Get Started in 2 Minutes

### Installation
```bash
cd stock-backtester
pip install -r requirements.txt
```

### Run Application
```bash
streamlit run market_dashboard/dashboard.py
```

### Run Tests
```bash
python test_integration.py
```

---

## 📊 UI Sections at a Glance

### LEFT SIDEBAR

#### 📈 Data Selection
```
Ticker Symbols: [AAPL,MSFT,NVDA]    ← Comma-separated list
Interval:       [1d ▼]              ← 1m, 5m, 15m, 1h, 1d
Start Date:     [2023-01-01]
End Date:       [Today]
```

#### 👁️ Display Options
```
📊 Chart        ☑       Show candlesticks & indicators
📉 Drawdown     ☑       Show equity drawdown curve
🔗 Correlation  ☑       Show ticker correlation heatmap
```

#### 🤖 Backtesting
```
Select Strategy:    [MA Crossover ▼]  ← 3 strategies available
📋 Quick Presets:   [Swing (2-Day) ▼] ← Instant configuration
Backtest Period:    [2023-01-01 to 2023-12-31]
Position Sizing:    [Fixed ⊙ Dynamic] ← All-in or fractional
Hold Days:          [2]                ← 0=day, 1-5=swing, 20+=position
⚙️ Advanced Options  [▼]               ← Fees, Sharpe mode
```

---

## 📈 Main Content Area

### Backtest Results (if strategy selected)

```
┌─────────────────────────────────────────────────────────┐
│ 💰 Total Return     │ 📊 Sharpe Ratio  │ 📉 Max Drawdown │
│ +12.45%            │ 0.89             │ -8.23%          │
│                    │                  │                 │
│ 🎯 Win Rate        │ 📅 Period        │                 │
│ 54.9%              │ 252 days         │                 │
└─────────────────────────────────────────────────────────┘

┌─ Trade Log ────────────────────────────────────────────┐
│ Entry Date    │ Entry  │ Exit Date    │ Exit   │ Return │
├──────────────┼────────┼──────────────┼────────┼────────┤
│ 2023-01-15   │ $150.2 │ 2023-01-17   │ $152.1 │ +1.26% │
│ 2023-01-18   │ $151.8 │ 2023-01-20   │ $149.5 │ -1.52% │
│ ...          │ ...    │ ...          │ ...    │ ...    │
│                                      │ [📥 Export CSV] │
└────────────────────────────────────────────────────────┘

┌─ Advanced Chart (5 rows) ───────────────────────────────┐
│ Row 1: Candlestick + MA50/200 + Bollinger + Signals    │
│ Row 2: Volume                                           │
│ Row 3: MACD                                             │
│ Row 4: RSI                                              │
│ Row 5: Equity Curve (Strategy vs Buy-Hold)             │
└────────────────────────────────────────────────────────┘

┌─ Optional: Drawdown & Correlation ─────────────────────┐
│ (toggle on/off in Display Options)                      │
└────────────────────────────────────────────────────────┘
```

---

## ⚡ Key Features

| Feature | Benefit |
|---------|---------|
| **Session Caching** | Change chart toggle? Instant. No recalculation |
| **Quick Presets** | Day Trading / Swing / Position - one click |
| **Trade Export** | Download trades as CSV for analysis |
| **Entry/Exit Markers** | Green diamonds = buy, Red X = sell |
| **Buy-Hold Comparison** | See strategy vs benchmark equity curve |
| **Error Messages** | User-friendly with hints for fixes |

---

## 🎯 Use Cases

### Day Trading Analysis
1. Select quick preset **"Day Trading"** (0-day holding)
2. Set interval to **1h** (hourly candles)
3. Choose strategy (e.g., RSI)
4. Review trade log for daily scalps
5. Export trades to CSV

### Swing Trading Backtest
1. Select preset **"Swing (2-Day)"**
2. Use default interval **1d** (daily candles)
3. Test MA Crossover strategy
4. Check Sharpe ratio and win rate
5. Compare equity curve to buy-hold

### Position Trading Study
1. Select preset **"Position Trading"** (20+ days)
2. Use **1d** interval
3. Backtest Bollinger Bands strategy
4. Focus on max drawdown metric
5. Analyze correlation of holdings

---

## 💡 Pro Tips

1. **Faster Backtests**: Use shorter date ranges for quick iterations
2. **Fair Comparison**: Enable Buy-Hold to benchmark strategy
3. **Transaction Costs**: Add 0.05-0.1% fees to simulate real slippage
4. **Multiple Backtests**: Cache remembers results (fast on repeat)
5. **Different Time Frames**: Test same strategy on 1h vs 1d
6. **CSV Analysis**: Export trades to Excel for deeper analysis

---

## 🔍 Troubleshooting

### "No data for ticker in range"
- ✓ Check ticker symbol (AAPL not Apple)
- ✓ Verify date range has market activity
- ✓ Try well-known tickers (AAPL, MSFT, NVDA)

### Backtest shows "NaN"
- ✓ Strategy may not generate signals in period
- ✓ Try longer date range (more opportunities)
- ✓ Adjust holding period or position sizing

### App runs slow
- ✓ Reduce number of tickers
- ✓ Shorter date range
- ✓ Disable correlation heatmap toggle

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| README.md | Overview & feature descriptions |
| DEVELOPMENT.md | Architecture & adding new strategies |
| CODE_REFACTORING.md | Code style & patterns |
| ROADMAP.md | Planned features (Tier 1-3) |
| COMPLETION_SUMMARY.md | Full project summary |
| requirements.txt | Python dependencies |

---

## 🎮 Strategy Reference

### Moving Average Crossover
- **Signal**: MA50 crosses above MA200 = buy, below = sell
- **Best For**: Trend following
- **Holding**: Usually multi-day

### RSI Strategy (Threshold Mode)
- **Signal**: RSI < 30 = buy, RSI > 70 = sell
- **Best For**: Overbought/oversold detection
- **Holding**: Short-term reversals

### RSI Strategy (Mean-Reversion Mode)
- **Signal**: RSI crosses 50 line
- **Best For**: Range-bound markets
- **Holding**: Intraday to swing

### Bollinger Bands
- **Signal**: Price touches lower band = buy, upper band = sell
- **Best For**: Volatility-based entries
- **Holding**: Variable based on volatility

---

## ✨ Release Notes (v1.1.0)

### New Features ✨
- Session caching for instant parameter changes
- Quick presets (Day Trading, Swing, Position)
- Trade log with statistics and CSV export
- 5-row advanced chart with equity curve
- Organized sidebar with emojis
- Collapsible advanced options

### Improvements 🚀
- Modular code with helper functions
- Better error messages
- Faster backtest execution
- Cleaner UI layout
- Comprehensive testing suite

### Bug Fixes 🐛
- Fixed metric display alignment
- Proper date filtering
- Correct fee calculation
- Signal marker placement

---

**Ready to backtest? Run:** 
```bash
streamlit run market_dashboard/dashboard.py
```

**Questions?** See DEVELOPMENT.md for code architecture.
