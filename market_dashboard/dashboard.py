"""
Stock Backtester - Quant Market Analytics Dashboard
A Streamlit-based dashboard for analyzing market data and backtesting trading strategies.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
from datetime import datetime, timedelta

from modules.data import download_data
from modules.indicators import moving_averages, rsi, macd, bollinger
from modules.utils import compute_returns, correlation_matrix
from modules.strategies import (
    MovingAverageCrossover, RSIStrategy, BollingerBandsStrategy, 
    buy_hold_equity
)
from modules.portfolio import sharpe_ratio, max_drawdown, win_rate
from modules.stock_search import (
    search_stocks, get_stock_info, get_popular_stocks, 
    get_stock_categories, format_market_cap, format_price
)
from modules.simulator import (
    TradingSimulator, create_simulator_session, get_simulator_engine, 
    reset_simulator
)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

st.set_page_config(layout="wide", page_title="Quant Market Analytics", page_icon="📊")

TRADING_DAYS_PER_YEAR = 252
HOURS_PER_TRADING_DAY = 6.5

DEFAULT_TICKERS = "AAPL,MSFT,NVDA,TSLA,SPY"
DEFAULT_START = pd.to_datetime("2023-01-01")
DEFAULT_END = pd.to_datetime("today")

INTERVALS = ["1m", "5m", "15m", "1h", "1d"]
DEFAULT_INTERVAL = "1d"

STRATEGY_OPTIONS = [
    "None",
    "MA Crossover",
    "RSI (Threshold)",
    "RSI (Mean-Reversion)",
    "Bollinger Bands"
]

TRADING_PRESETS = {
    "Day Trading": {"holding_period": 0, "position_type": "Fixed", "transaction_fee": 0.001},
    "Swing (2-Day)": {"holding_period": 2, "position_type": "Fixed", "transaction_fee": 0.001},
    "Swing (5-Day)": {"holding_period": 5, "position_type": "Fixed", "transaction_fee": 0.001},
    "Position Trading": {"holding_period": 20, "position_type": "Fixed", "transaction_fee": 0.0005},
}

SHARPE_MODES = {
    "Daily (252 days/yr)": "1d",
    "Hourly": "1h",
    "5-Minute": "5m",
    "1-Minute": "1m",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_ticker_data(data, ticker, start_date, end_date):
    """Extract single ticker data for backtest period."""
    if len(data.columns.names) > 1:  # MultiIndex (multiple tickers)
        ticker_data = data.xs(ticker, level=1, axis=1)
    else:
        ticker_data = data
    
    mask = (ticker_data.index.date >= start_date) & (ticker_data.index.date <= end_date)
    return ticker_data[mask]


def compute_all_indicators(close):
    """Pre-compute all indicators needed by strategies."""
    ma50, ma200 = moving_averages(close)
    rsi_vals = rsi(close)
    bb_upper, bb_lower = bollinger(close)
    
    return {
        'ma50': ma50,
        'ma200': ma200,
        'rsi': rsi_vals,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'close': close
    }


def get_strategy_instance(name, config):
    """Create strategy instance based on name and config."""
    strategies = {
        "MA Crossover": MovingAverageCrossover,
        "RSI (Threshold)": lambda **kw: RSIStrategy(mode="threshold", **kw),
        "RSI (Mean-Reversion)": lambda **kw: RSIStrategy(mode="mean_reversion", **kw),
        "Bollinger Bands": BollingerBandsStrategy,
    }
    
    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}")
    
    strategy_class = strategies[name]
    return strategy_class(
        holding_period=config['holding_period'],
        position_type="fixed" if config['position_type'] == "Fixed" else "dynamic",
        fee_pct=config['fee_pct']
    )


def run_single_backtest(strategy_name, price, indicators, config):
    """Execute backtest for single strategy."""
    strategy = get_strategy_instance(strategy_name, config)
    signals = strategy.generate_signals(price, indicators)
    results = strategy.compute_positions_and_equity(signals, price, initial_equity=100)
    metrics = strategy.compute_metrics(
        results['equity'],
        results['daily_return'],
        interval=config['interval'],
        risk_free_rate=0.02
    )
    return {**results, **metrics}


def create_backtest_key(strategy, start, end, ticker, holding_days, fee):
    """Create hashable key for caching backtest results."""
    return (strategy, str(start), str(end), ticker, holding_days, fee)


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_metrics_panel(metrics):
    """Show 4-column metrics dashboard."""
    metric_cols = st.columns(4)
    
    specs = [
        ("📈 Total Return", f"{metrics['total_return']:.2f}%", "good" if metrics['total_return'] > 0 else "bad"),
        ("⚖️ Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}", "good" if metrics['sharpe_ratio'] > 1 else "neutral"),
        ("📉 Max Drawdown", f"{metrics['max_drawdown']:.2f}%", "bad" if metrics['max_drawdown'] < -10 else "neutral"),
        ("🎯 Win Rate", f"{metrics['win_rate']:.1f}%", "good" if metrics['win_rate'] > 50 else "neutral"),
    ]
    
    for col, (label, value, delta_color) in zip(metric_cols, specs):
        col.metric(label, value)


def display_trade_log(backtest_data, strategy_name):
    """Show trade details and export option."""
    if not backtest_data or 'trades' not in backtest_data or len(backtest_data['trades']) == 0:
        st.info("ℹ️ No trades executed in backtest period.")
        return
    
    st.subheader("📋 Trade Log")
    
    trades = backtest_data['trades']
    trades_df = pd.DataFrame(trades)
    
    # Format columns for display
    display_df = trades_df.copy()
    display_df['entry_date'] = pd.to_datetime(display_df['entry_date']).dt.strftime('%Y-%m-%d')
    display_df['exit_date'] = pd.to_datetime(display_df['exit_date']).dt.strftime('%Y-%m-%d')
    display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
    display_df['exit_price'] = display_df['exit_price'].apply(lambda x: f"${x:.2f}")
    display_df['return_pct'] = display_df['return_pct'].apply(lambda x: f"{x:.2f}%")
    
    display_df = display_df[['entry_date', 'entry_price', 'exit_date', 'exit_price', 'return_pct']]
    display_df.columns = ['Entry Date', 'Entry Price', 'Exit Date', 'Exit Price', 'Return %']
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Trade statistics
    col1, col2, col3 = st.columns(3)
    
    total_trades = len(trades)
    winning_trades = [t for t in trades if t['return_pct'] > 0]
    losing_trades = [t for t in trades if t['return_pct'] < 0]
    
    with col1:
        st.metric("Total Trades", total_trades)
    with col2:
        st.metric("Winning Trades", len(winning_trades))
    with col3:
        st.metric("Losing Trades", len(losing_trades))
    
    if winning_trades:
        avg_win = np.mean([t['return_pct'] for t in winning_trades])
        max_win = max([t['return_pct'] for t in winning_trades])
        st.write(f"**📈 Wins:** Avg {avg_win:.2f}% | Max {max_win:.2f}%")
    
    if losing_trades:
        avg_loss = np.mean([t['return_pct'] for t in losing_trades])
        max_loss = min([t['return_pct'] for t in losing_trades])
        st.write(f"**📉 Losses:** Avg {avg_loss:.2f}% | Max {max_loss:.2f}%")
    
    # Export button
    csv_data = trades_df.to_csv(index=False)
    st.download_button(
        "📥 Download Trades (CSV)",
        csv_data,
        f"trades_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv",
        help="Export trade log for external analysis"
    )


def display_advanced_chart(data, selected_ticker, backtest_data=None, backtest_signals=None):
    """Show candlestick chart with indicators and equity curve."""
    df = data.xs(selected_ticker, level=1, axis=1) if len(data.columns.names) > 1 else data
    price = df["Close"]
    
    # Compute indicators
    ma50, ma200 = moving_averages(price)
    rsi_values = rsi(price)
    macd_line, signal = macd(price)
    upper, lower = bollinger(price)
    
    # Determine chart rows
    num_rows = 5 if backtest_data is not None else 4
    row_heights = [0.4, 0.12, 0.15, 0.1, 0.23] if num_rows == 5 else [0.5, 0.15, 0.2, 0.15]
    
    fig = sp.make_subplots(
        rows=num_rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.02,
        subplot_titles=("Price Action", "Volume", "MACD", "RSI", "Equity Curve") if num_rows == 5 
                      else ("Price Action", "Volume", "MACD", "RSI")
    )
    
    # Row 1: Candlestick + MA + BB
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price"
        ),
        row=1, col=1
    )
    
    fig.add_trace(go.Scatter(x=df.index, y=ma50, name="MA50", line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ma200, name="MA200", line=dict(color="red")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=upper, name="BB Upper", line=dict(color="gray", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=lower, name="BB Lower", line=dict(color="gray", dash="dash")), row=1, col=1)
    
    # Add entry/exit signals if backtest active
    if backtest_data is not None and backtest_signals is not None:
        entries_idx = backtest_signals['entries'][backtest_signals['entries'] > 0].index
        exits_idx = backtest_signals['exits'][backtest_signals['exits'] > 0].index
        
        if len(entries_idx) > 0:
            entry_lows = df.loc[entries_idx, "Low"]
            fig.add_trace(
                go.Scatter(x=entries_idx, y=entry_lows, mode="markers",
                          marker=dict(size=10, color="green", symbol="diamond"),
                          name="Buy", showlegend=True),
                row=1, col=1
            )
        
        if len(exits_idx) > 0:
            exit_highs = df.loc[exits_idx, "High"]
            fig.add_trace(
                go.Scatter(x=exits_idx, y=exit_highs, mode="markers",
                          marker=dict(size=10, color="red", symbol="x"),
                          name="Sell", showlegend=True),
                row=1, col=1
            )
    
    # Row 2: Volume
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="lightblue"), row=2, col=1)
    
    # Row 3: MACD
    fig.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD", line=dict(color="blue")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=signal, name="Signal", line=dict(color="red")), row=3, col=1)
    
    # Row 4: RSI
    fig.add_trace(go.Scatter(x=df.index, y=rsi_values, name="RSI", line=dict(color="purple")), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1, annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1, annotation_text="Oversold")
    
    # Row 5: Equity curve (if backtest active)
    if backtest_data is not None:
        equity = backtest_data['equity']
        bh_equity = buy_hold_equity(price[equity.index], initial_equity=100)
        
        fig.add_trace(
            go.Scatter(x=equity.index, y=equity.values, name="Strategy", line=dict(color="blue", width=2)),
            row=num_rows, col=1
        )
        fig.add_trace(
            go.Scatter(x=bh_equity.index, y=bh_equity.values, name="Buy & Hold", 
                      line=dict(color="gray", width=2, dash="dash")),
            row=num_rows, col=1
        )
    
    # Layout
    fig.update_layout(
        height=900 if num_rows == 4 else 1100,
        hovermode='x unified',
        template='plotly_dark'
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=4, col=1)
    if num_rows == 5:
        fig.update_yaxes(title_text="Equity ($)", row=5, col=1)
    
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Initialize session state
    if 'backtest_cache' not in st.session_state:
        st.session_state.backtest_cache = {}
    
    # Initialize welcome and preferences
    if 'show_welcome' not in st.session_state:
        st.session_state.show_welcome = True
    if 'ui_mode' not in st.session_state:
        st.session_state.ui_mode = 'simple'
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'
    if 'mode' not in st.session_state:
        st.session_state.mode = 'backtesting'
    
    # Show welcome dashboard if needed
    if st.session_state.show_welcome:
        show_welcome_dashboard()
        return  # Exit early to show only welcome screen
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    
    st.sidebar.title("📊 Quant Market Analytics")
    
    # Theme and mode controls at top
    with st.sidebar:
        col1, col2 = st.columns(2)
        with col1:
            current_theme = st.session_state.get('theme', 'dark')
            theme_options = ["🌙 Dark", "☀️ Light"]
            theme_index = 0 if current_theme == 'dark' else 1
            selected_theme = st.selectbox(
                "Theme",
                theme_options,
                index=theme_index,
                key="theme_selector",
                help="Switch between dark and light themes"
            )
            st.session_state.theme = 'dark' if 'Dark' in selected_theme else 'light'
        
        with col2:
            current_mode = st.session_state.get('ui_mode', 'simple')
            mode_options = ["🎯 Simple", "⚙️ Expert"]
            mode_index = 0 if current_mode == 'simple' else 1
            selected_mode = st.selectbox(
                "Mode",
                mode_options,
                index=mode_index,
                key="mode_selector",
                help="Simple: Guided | Expert: Full controls"
            )
            st.session_state.ui_mode = 'simple' if 'Simple' in selected_mode else 'expert'
        
        # Welcome screen toggle
        if st.button("🏠 Welcome Screen", help="Return to welcome dashboard"):
            st.session_state.show_welcome = True
            st.rerun()
        
        st.sidebar.divider()
    
    with st.sidebar:
        # Stock search section
        st.subheader("🔍 Stock Search")
        
        search_query = st.text_input(
            "Search stocks",
            placeholder="e.g., Apple, AAPL, TSLA",
            help="Search by company name or ticker symbol"
        )
        
        if search_query:
            with st.spinner("🔍 Searching..."):
                search_results = search_stocks(search_query, limit=5)
            
            if search_results:
                st.write("**Search Results:**")
                for stock in search_results:
                    if st.button(
                        f"{stock['symbol']} - {stock['name'][:30]}",
                        key=f"search_{stock['symbol']}",
                        help=f"Sector: {stock['sector']}"
                    ):
                        # Update ticker input with selected stock
                        st.session_state.ticker_input = stock['symbol']
                        st.rerun()
            else:
                st.info("No stocks found. Try a different search term.")
        
        # Popular stocks
        st.markdown("**Popular Stocks:**")
        category = st.selectbox(
            "Category",
            get_stock_categories(),
            index=0,
            key="popular_category"
        )
        
        popular_stocks = get_popular_stocks(category)
        cols = st.columns(3)
        for i, symbol in enumerate(popular_stocks[:9]):  # Show first 9
            with cols[i % 3]:
                if st.button(symbol, key=f"popular_{symbol}"):
                    st.session_state.ticker_input = symbol
                    st.rerun()
        
        st.sidebar.divider()
        
        # Data section
        st.subheader("📈 Data Selection")
        
        # Use session state for ticker input to persist selections
        if 'ticker_input' not in st.session_state:
            st.session_state.ticker_input = DEFAULT_TICKERS
            
        tickers_input = st.text_input(
            "Ticker Symbols",
            st.session_state.ticker_input,
            help="Comma-separated list (e.g., AAPL,MSFT,NVDA)"
        )
        
        # Update session state when user types
        st.session_state.ticker_input = tickers_input
        
        interval = st.selectbox(
            "Interval",
            INTERVALS,
            index=4,
            help="Candle frequency for analysis"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input("Start Date", value=DEFAULT_START, key="data_start")
        with col2:
            end = st.date_input("End Date", value=DEFAULT_END, key="data_end")
        
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        
        st.sidebar.divider()
        
        # Display options
        st.subheader("👁️ Display Options")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            show_price = st.toggle("📊 Chart", value=True)
        with col2:
            show_drawdown = st.toggle("📉 Drawdown", value=True)
        with col3:
            show_corr = st.toggle("🔗 Correlation", value=True)
        
        st.sidebar.divider()
        
        # Mode selection
        st.subheader("🎮 Mode Selection")
        
        mode = st.radio(
            "Select Mode",
            ["📊 Backtesting", "🎮 Simulator"],
            index=0,
            help="Backtesting: Test strategies automatically | Simulator: Trade manually"
        )
        
        is_simulator_mode = "Simulator" in mode
        
        st.sidebar.divider()
        
        # Backtesting section
        if not is_simulator_mode:
            st.subheader("🤖 Backtesting")
            
            strategy_name = st.selectbox(
                "Select Strategy",
                STRATEGY_OPTIONS,
                help="Choose trading strategy to backtest"
            )
        
        backtest_start = None
        backtest_end = None
        config = None
        
        if strategy_name != "None":
            # Quick presets
            preset = st.selectbox(
                "📋 Quick Presets",
                ["Custom"] + list(TRADING_PRESETS.keys()),
                help="Pre-configured trading styles"
            )
            
            # Initialize config from preset or custom
            if preset != "Custom" and preset in TRADING_PRESETS:
                preset_config = TRADING_PRESETS[preset]
                default_hold = preset_config['holding_period']
                default_pos = preset_config['position_type']
                default_fee = preset_config['transaction_fee']
            else:
                default_hold, default_pos, default_fee = 0, "Fixed", 0.0
            
            # Backtesting date range
            st.markdown("**Backtest Period**")
            col1, col2 = st.columns(2)
            with col1:
                backtest_start = st.date_input(
                    "From",
                    value=DEFAULT_START,
                    key="backtest_start",
                    help="Start date for backtest"
                )
            with col2:
                backtest_end = st.date_input(
                    "To",
                    value=DEFAULT_END,
                    key="backtest_end"
                )
            
            # Position & fees
            st.markdown("**Position Configuration**")
            col1, col2 = st.columns(2)
            
            with col1:
                position_type = st.radio(
                    "Position Sizing",
                    ["Fixed", "Dynamic"],
                    index=0 if default_pos == "Fixed" else 1,
                    horizontal=True,
                    help="Fixed=all-in | Dynamic=0-1"
                )
            
            with col2:
                holding_period = st.number_input(
                    "Hold Days",
                    value=default_hold,
                    min_value=0,
                    max_value=252,
                    help="0=day, 1-5=swing, 20+=position"
                )
            
            # Advanced options (only in expert mode)
            if st.session_state.ui_mode == 'expert':
                with st.expander("⚙️ Advanced Options", expanded=False):
                    transaction_fee = st.slider(
                        "Transaction Fee (%)",
                        min_value=0.0,
                        max_value=1.0,
                        value=default_fee * 100,
                        step=0.01,
                        help="Per-trade cost"
                    ) / 100
                    
                    sharpe_mode = st.selectbox(
                        "Sharpe Annualization",
                        list(SHARPE_MODES.keys()),
                    )
                    sharpe_interval = SHARPE_MODES[sharpe_mode]
            else:
                # Simple mode defaults
                transaction_fee = default_fee
                sharpe_interval = "1d"
            
            config = {
                'position_type': position_type,
                'holding_period': int(holding_period),
                'fee_pct': transaction_fee,
                'interval': interval,
            }
        
        # Simulator section
        else:
            st.subheader("🎮 Trading Simulator")
            
            # Initialize simulator
            create_simulator_session()
            simulator = get_simulator_engine()
            
            # Simulator controls
            simulator_active = st.toggle(
                "🎮 Activate Simulator", 
                value=st.session_state.simulator.get('active', False),
                help="Enable manual trading simulation"
            )
            
            if simulator_active:
                st.session_state.simulator['active'] = True
                
                # Simulator settings
                st.markdown("**Simulator Settings**")
                col1, col2 = st.columns(2)
                with col1:
                    sim_start = st.date_input(
                        "Start Date",
                        value=DEFAULT_START,
                        key="sim_start"
                    )
                with col2:
                    sim_end = st.date_input(
                        "End Date", 
                        value=DEFAULT_END,
                        key="sim_end"
                    )
                
                initial_equity = st.number_input(
                    "Initial Equity ($)",
                    value=10000,
                    min_value=1000,
                    max_value=1000000,
                    step=1000,
                    help="Starting capital for simulation"
                )
                
                sim_fee = st.slider(
                    "Transaction Fee (%)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.01,
                    help="Per-trade cost"
                ) / 100
                
                # Initialize simulator with settings
                if not hasattr(simulator, 'sim_data') or simulator.sim_data is None:
                    try:
                        # Get data for the selected ticker
                        selected_ticker = tickers[0] if tickers else "AAPL"
                        sim_data = download_data([selected_ticker], sim_start, sim_end, interval)
                        
                        if len(sim_data) > 0:
                            simulator.reset()
                            simulator.transaction_fee = sim_fee
                            simulator.initial_equity = initial_equity
                            simulator.set_timeframe(sim_data, sim_start, sim_end)
                            st.success(f"✅ Simulator ready! Trading {selected_ticker} from {sim_start} to {sim_end}")
                        else:
                            st.error("❌ No data available for the selected period")
                            simulator_active = False
                            
                    except Exception as e:
                        st.error(f"❌ Failed to initialize simulator: {e}")
                        simulator_active = False
                
                # Trading controls
                if simulator_active and hasattr(simulator, 'sim_data'):
                    st.markdown("---")
                    st.markdown("**Trading Controls**")
                    
                    # Current state display
                    state = simulator.get_current_state()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📅 Date", state['date'].strftime('%Y-%m-%d'))
                    with col2:
                        st.metric("💰 Cash", f"${state['cash']:.2f}")
                    with col3:
                        st.metric("📊 Positions", state['positions'])
                    with col4:
                        st.metric("💎 Equity", f"${state['total_equity']:.2f}")
                    
                    # Trading buttons
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        buy_qty = st.number_input(
                            "Buy Qty",
                            min_value=1,
                            max_value=10000,
                            value=100,
                            step=10
                        )
                        if st.button("🟢 BUY", use_container_width=True):
                            if simulator.execute_buy(buy_qty):
                                st.success(f"✅ Bought {buy_qty} shares at ${simulator.current_price:.2f}")
                                st.rerun()
                            else:
                                can_buy, reason = simulator.can_buy(buy_qty)
                                st.error(f"❌ {reason}")
                    
                    with col2:
                        sell_qty = st.number_input(
                            "Sell Qty",
                            min_value=1,
                            max_value=10000,
                            value=100,
                            step=10
                        )
                        if st.button("🔴 SELL", use_container_width=True):
                            if simulator.execute_sell(sell_qty):
                                st.success(f"✅ Sold {sell_qty} shares at ${simulator.current_price:.2f}")
                                st.rerun()
                            else:
                                can_sell, reason = simulator.can_sell(sell_qty)
                                st.error(f"❌ {reason}")
                    
                    with col3:
                        # Time controls
                        time_col1, time_col2, time_col3, time_col4 = st.columns(4)
                        with time_col1:
                            if st.button("⏮️", help="Go to start"):
                                simulator.go_to_date(simulator.sim_data.index[0])
                                st.rerun()
                        with time_col2:
                            if st.button("⏸️", help="Pause/Resume"):
                                pass  # Could implement auto-play
                        with time_col3:
                            if st.button("▶️", help="Advance 1 day"):
                                if simulator.advance_time(1):
                                    st.rerun()
                                else:
                                    st.info("End of simulation period reached")
                        with time_col4:
                            if st.button("⏭️", help="Advance 5 days"):
                                if simulator.advance_time(5):
                                    st.rerun()
                                else:
                                    st.info("End of simulation period reached")
                    
                    # Reset button
                    if st.button("🔄 Reset Simulation", type="secondary"):
                        reset_simulator()
                        st.rerun()
            else:
                st.session_state.simulator['active'] = False
                st.info("💡 Toggle 'Activate Simulator' to start manual trading practice")
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    st.title("📊 Quant Market Analytics")
    st.markdown("*Professional quantitative trading analysis and backtesting platform*")
    
    # Download data
    try:
        with st.spinner("📥 Downloading market data..."):
            data = download_data(tickers, start, end, interval)
    except Exception as e:
        st.error(f"❌ Failed to download data: {e}")
        st.info("💡 Check that ticker symbols are valid (e.g., AAPL not Apple)")
        st.stop()
    
    close = data["Close"]
    returns = compute_returns(close)
    dd = close / close.cummax() - 1
    
    # Ticker selector
    selected_ticker = st.selectbox("Select Ticker for Analysis", tickers, key="selected_ticker")
    
    # ========================================================================
    # ANALYSIS LOGIC (Backtesting or Simulator)
    # ========================================================================
    
    backtest_result = None
    backtest_metrics = None
    simulator_metrics = None
    simulator_trades_df = None
    
    if not is_simulator_mode:
        # BACKTESTING MODE
        if strategy_name and strategy_name != "None" and config is not None:
            try:
                with st.spinner("⚙️ Running backtest..."):
                    # Extract data
                    ticker_data = extract_ticker_data(data, selected_ticker, backtest_start, backtest_end)
                    
                    if len(ticker_data) == 0:
                        st.warning(f"⚠️ No data for {selected_ticker} in range {backtest_start} to {backtest_end}")
                    else:
                        # Prepare indicators
                        indicators = compute_all_indicators(ticker_data["Close"])
                        
                        # Check cache
                        cache_key = create_backtest_key(
                            strategy_name, backtest_start, backtest_end, 
                            selected_ticker, config['holding_period'], config['fee_pct']
                        )
                        
                        if cache_key in st.session_state.backtest_cache:
                            backtest_result = st.session_state.backtest_cache[cache_key]
                        else:
                            backtest_result = run_single_backtest(strategy_name, ticker_data["Close"], indicators, config)
                            st.session_state.backtest_cache[cache_key] = backtest_result
                        
                        # Extract metrics
                        backtest_metrics = {k: backtest_result[k] for k in 
                                          ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']}
            
            except ValueError as e:
                st.error(f"❌ Configuration error: {e}")
            except Exception as e:
                st.error(f"❌ Backtest failed: {e}")
                with st.expander("Debug Info"):
                    st.write(f"Error: {str(e)}")
    
    else:
        # SIMULATOR MODE
        if st.session_state.simulator.get('active', False) and hasattr(simulator, 'sim_data'):
            simulator_metrics = simulator.get_metrics()
            simulator_trades_df = simulator.get_trades_df()
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    
    # Mode-specific results display
    if not is_simulator_mode:
        # Backtesting results
        if backtest_metrics is not None:
            st.divider()
            col1, col2 = st.columns([3, 1])
            
            with col1:
                display_metrics_panel(backtest_metrics)
            
            with col2:
                st.metric("📅 Period", f"{(backtest_end - backtest_start).days}d")
            
            st.divider()
            
            # Trade log
            display_trade_log(backtest_result, strategy_name)
            st.divider()
    
    else:
        # Simulator results
        if simulator_metrics is not None:
            st.divider()
            st.subheader("📊 Simulator Performance")
            
            # Current state
            state = simulator.get_current_state()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💰 Cash", f"${state['cash']:.2f}")
            with col2:
                st.metric("📊 Positions", state['positions'])
            with col3:
                st.metric("💎 Total Equity", f"${state['total_equity']:.2f}")
            with col4:
                st.metric("📈 Unrealized P&L", f"${state['unrealized_pnl']:.2f}")
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                color = "🟢" if simulator_metrics['total_return'] > 0 else "🔴"
                st.metric("💰 Total Return", f"{color} {simulator_metrics['total_return']:.2f}%")
            with col2:
                st.metric("📊 Sharpe Ratio", f"{simulator_metrics['sharpe_ratio']:.2f}")
            with col3:
                st.metric("📉 Max Drawdown", f"{simulator_metrics['max_drawdown']:.2f}%")
            with col4:
                st.metric("🎯 Win Rate", f"{simulator_metrics['win_rate']:.1f}%")
            
            # Trade log
            if not simulator_trades_df.empty:
                st.divider()
                st.subheader("📋 Trade History")
                
                # Trade statistics
                total_trades = len(simulator_trades_df)
                winning_trades = len(simulator_trades_df[simulator_trades_df['realized_pnl'] > 0])
                losing_trades = total_trades - winning_trades
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Trades", total_trades)
                with col2:
                    st.metric("Wins", winning_trades)
                with col3:
                    st.metric("Losses", losing_trades)
                
                # Trade table
                st.dataframe(
                    simulator_trades_df[[
                        'action', 'quantity', 'price', 'proceeds', 
                        'cost_basis', 'realized_pnl', 'fee'
                    ]].round(2),
                    use_container_width=True
                )
                
                # CSV export
                csv = simulator_trades_df.to_csv(index=True)
                st.download_button(
                    "📥 Export Trades (CSV)",
                    csv,
                    "simulator_trades.csv",
                    "text/csv"
                )
            
            st.divider()
    
    # ========================================================================
    # CHARTS (Both modes)
    # ========================================================================
    
    # Advanced chart
    if show_price:
        if not is_simulator_mode:
            st.subheader("📈 Price Action & Technical Indicators")
            
            backtest_signals = None
            if backtest_result is not None:
                backtest_signals = {
                    'entries': backtest_result['entries'],
                    'exits': backtest_result['exits']
                }
            
            display_advanced_chart(data, selected_ticker, backtest_result, backtest_signals)
        else:
            # Simulator chart
            st.subheader("📈 Simulator Chart")
            
            if hasattr(simulator, 'sim_data') and simulator.sim_data is not None:
                # Create simulator-specific chart
                sim_data = simulator.sim_data.copy()
                
                # Add equity curve
                equity_curve = simulator.get_equity_curve()
                
                fig = sp.make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    row_heights=[0.7, 0.3],
                    vertical_spacing=0.05,
                    subplot_titles=("Price & Manual Trades", "Equity Curve")
                )
                
                # Price chart with manual trade markers
                fig.add_trace(
                    go.Candlestick(
                        x=sim_data.index,
                        open=sim_data["Open"],
                        high=sim_data["High"],
                        low=sim_data["Low"],
                        close=sim_data["Close"],
                        name="Price"
                    ),
                    row=1, col=1
                )
                
                # Add current position marker
                current_date = simulator.current_date
                current_price = simulator.current_price
                
                fig.add_trace(
                    go.Scatter(
                        x=[current_date],
                        y=[current_price],
                        mode="markers",
                        marker=dict(size=15, color="blue", symbol="diamond"),
                        name="Current Position"
                    ),
                    row=1, col=1
                )
                
                # Add buy/sell markers from trades
                if simulator_trades_df is not None and not simulator_trades_df.empty:
                    buy_trades = simulator_trades_df[simulator_trades_df['action'] == 'BUY']
                    sell_trades = simulator_trades_df[simulator_trades_df['action'] == 'SELL']
                    
                    if not buy_trades.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=buy_trades.index,
                                y=buy_trades['price'],
                                mode="markers",
                                marker=dict(size=10, color="green", symbol="triangle-up"),
                                name="Buy Orders"
                            ),
                            row=1, col=1
                        )
                    
                    if not sell_trades.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=sell_trades.index,
                                y=sell_trades['price'],
                                mode="markers",
                                marker=dict(size=10, color="red", symbol="triangle-down"),
                                name="Sell Orders"
                            ),
                            row=1, col=1
                        )
                
                # Equity curve
                fig.add_trace(
                    go.Scatter(
                        x=equity_curve.index,
                        y=equity_curve.values,
                        name="Equity",
                        line=dict(color="purple", width=2)
                    ),
                    row=2, col=1
                )
                
                # Add buy-and-hold comparison
                bh_equity = buy_hold_equity(sim_data["Close"], initial_equity=simulator.initial_equity)
                fig.add_trace(
                    go.Scatter(
                        x=bh_equity.index,
                        y=bh_equity.values,
                        name="Buy & Hold",
                        line=dict(color="gray", dash="dash")
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(height=600, showlegend=True)
                fig.update_xaxes(title_text="Date", row=2, col=1)
                fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="Equity ($)", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("💡 Activate the simulator to see the trading chart")
        
        st.divider()
        st.divider()
    
    # Drawdown chart
    if show_drawdown:
        st.subheader("📉 Drawdown Analysis")
        
        drawdown_fig = go.Figure()
        for ticker in tickers:
            drawdown_fig.add_trace(
                go.Scatter(x=dd.index, y=dd[ticker] * 100, name=ticker, mode='lines')
            )
        
        drawdown_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode='x unified',
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(drawdown_fig, use_container_width=True)
        st.divider()
    
    # Correlation heatmap
    if show_corr:
        st.subheader("🔗 Correlation Matrix")
        
        corr = correlation_matrix(returns)
        
        corr_fig = go.Figure(
            data=go.Heatmap(
                z=corr,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1
            )
        )
        
        corr_fig.update_layout(
            height=500,
            template='plotly_dark'
        )
        
        st.plotly_chart(corr_fig, use_container_width=True)
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 12px;'>
        📊 Quant Market Analytics v1.1 | Data from Yahoo Finance | Not financial advice
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================================
# WELCOME DASHBOARD
# ============================================================================

def show_welcome_dashboard():
    """Display welcome screen with guided onboarding."""
    
    # Hero section
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #1f77b4; font-size: 3rem; margin-bottom: 0.5rem;'>📊 Quant Market Analytics</h1>
        <p style='font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>Professional quantitative trading analysis and backtesting platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick start options
    st.markdown("### 🚀 Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📈 Backtesting Mode**
        - Test trading strategies automatically
        - Analyze historical performance
        - Compare against buy-and-hold
        """)
        
        if st.button("🎯 Start Backtesting", type="primary", use_container_width=True):
            st.session_state.show_welcome = False
            st.session_state.mode = "backtesting"
            st.rerun()
    
    with col2:
        st.markdown("""
        **🎮 Trading Simulator**
        - Practice manual trading
        - Real-time P&L tracking
        - Risk-free learning environment
        """)
        
        if st.button("🎲 Start Simulator", type="primary", use_container_width=True):
            st.session_state.show_welcome = False
            st.session_state.mode = "simulator"
            st.rerun()
    
    with col3:
        st.markdown("""
        **🔍 Stock Discovery**
        - Search and analyze stocks
        - Technical indicators
        - Market correlation analysis
        """)
        
        if st.button("🔎 Explore Stocks", type="primary", use_container_width=True):
            st.session_state.show_welcome = False
            st.session_state.mode = "analysis"
            st.rerun()
    
    st.divider()
    
    # Features overview
    st.markdown("### ✨ Key Features")
    
    features_col1, features_col2 = st.columns(2)
    
    with features_col1:
        st.markdown("""
        **🤖 Advanced Strategies**
        - Moving Average Crossover
        - RSI Mean-Reversion & Threshold
        - Bollinger Bands Breakout
        
        **📊 Technical Analysis**
        - Multiple timeframes (1m to 1d)
        - 50+ technical indicators
        - Interactive charts with Plotly
        
        **⚙️ Professional Tools**
        - Sharpe ratio, max drawdown
        - Win rate analysis
        - Trade logging & export
        """)
    
    with features_col2:
        st.markdown("""
        **🎮 Trading Simulator**
        - Manual buy/sell orders
        - Real-time portfolio tracking
        - Performance metrics
        
        **📈 Risk Management**
        - Position sizing options
        - Transaction cost modeling
        - Drawdown analysis
        
        **🔗 Market Analysis**
        - Multi-asset correlation
        - Sector analysis
        - Popular stocks discovery
        """)
    
    st.divider()
    
    # Settings
    st.markdown("### ⚙️ Preferences")
    
    settings_col1, settings_col2, settings_col3 = st.columns(3)
    
    with settings_col1:
        ui_mode = st.radio(
            "Interface Mode",
            ["Simple", "Expert"],
            index=0,
            help="Simple: Guided experience | Expert: Full controls"
        )
        st.session_state.ui_mode = ui_mode.lower()
    
    with settings_col2:
        theme = st.radio(
            "Theme",
            ["Dark", "Light"],
            index=0,
            help="Chart and interface theme"
        )
        st.session_state.theme = theme.lower()
    
    with settings_col3:
        if st.button("🔄 Reset All Settings", type="secondary"):
            # Reset to defaults
            st.session_state.clear()
            st.rerun()
    
    # Skip option
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("⏭️ Skip Welcome - Go to Dashboard", type="secondary", use_container_width=True):
            st.session_state.show_welcome = False
            st.rerun()

    # ========================================================================