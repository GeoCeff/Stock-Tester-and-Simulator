import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class Strategy(ABC):
    """
    Base strategy class for backtesting.
    Handles position logic, equity tracking, metrics computation, and signal visualization.
    Supports fixed (0/1) and dynamic (0-1 continuous) position sizing.
    Supports holding periods for day trading, swing trading, and position trading.
    """

    def __init__(self, holding_period=0, position_type="fixed", fee_pct=0.0):
        """
        Parameters:
        -----------
        holding_period : int
            Number of days to hold position. 0 = day trading (exit same day).
            1-5 = swing trading (1-5 days). 20+ = position trading.
        position_type : str
            "fixed" (all-in/out: 0 or 1) or "dynamic" (continuous 0-1).
        fee_pct : float
            Transaction cost as percentage (0.001 = 0.1%).
        """
        self.holding_period = holding_period
        self.position_type = position_type
        self.fee_pct = fee_pct

    @abstractmethod
    def generate_signals(self, price, indicators_dict):
        """
        Generate entry signals (0/1 or 0-1 for dynamic).

        Parameters:
        -----------
        price : pd.Series
            Close price series (single ticker).
        indicators_dict : dict
            Pre-computed indicators (ma50, ma200, rsi, etc.).

        Returns:
        --------
        pd.Series
            Signal values (0 or 1 for fixed, 0-1 for dynamic).
        """
        pass

    def compute_positions_and_equity(self, signals, close, initial_equity=100):
        """
        Convert signals to held positions accounting for holding period.
        Track equity curve, entry/exit dates, and trades.

        Parameters:
        -----------
        signals : pd.Series
            Entry signals from generate_signals().
        close : pd.Series
            Close price series.
        initial_equity : float
            Starting amount.

        Returns:
        --------
        dict with keys:
            - 'position': pd.Series (held position, 0-1)
            - 'daily_return': pd.Series (daily return %)
            - 'equity': pd.Series (equity curve starting at initial_equity)
            - 'entries': pd.Series (entry dates with signal strength, 0 if no entry)
            - 'exits': pd.Series (exit dates, 0 if no exit)
            - 'trades': list of (entry_date, entry_price, exit_date, exit_price, return%)
        """
        try:
            # Input validation
            if not isinstance(signals, pd.Series) or not isinstance(close, pd.Series):
                raise ValueError("Signals and close must be pandas Series")

            if len(signals) != len(close):
                raise ValueError("Signals and close series must have same length")

            if len(signals) == 0:
                raise ValueError("Empty data provided")

            if initial_equity <= 0:
                raise ValueError("Initial equity must be positive")

            df = pd.DataFrame({
                'close': close,
                'signal': signals
            })

            # Track position entry date (to enforce holding period)
            entry_dates = pd.Series(0, index=df.index, dtype=int)
            position = pd.Series(0.0, index=df.index)
            entries = pd.Series(0.0, index=df.index)
            exits = pd.Series(0.0, index=df.index)
            trades = []

            for i in range(len(df)):
                # Decrement days in position
                if i > 0 and entry_dates.iloc[i - 1] > 0:
                    entry_dates.iloc[i] = entry_dates.iloc[i - 1] + 1

                # Auto-exit if holding period exceeded
                if (self.holding_period > 0 and
                    entry_dates.iloc[i] > self.holding_period and
                    (i == 0 or position.iloc[i - 1] > 0)):
                    # Exit position due to holding period expiration
                    if i > 0 and position.iloc[i - 1] > 0:
                        entry_idx = i - entry_dates.iloc[i] + 1
                        if entry_idx >= 0 and entry_idx < len(df):
                            exit_price = df['close'].iloc[i]
                            entry_price = df['close'].iloc[entry_idx]
                            if entry_price > 0:
                                exit_return = (exit_price / entry_price - 1) * 100
                            else:
                                exit_return = 0
                            trades.append({
                                'entry_idx': entry_idx,
                                'entry_date': df.index[entry_idx],
                                'entry_price': entry_price,
                                'exit_idx': i,
                                'exit_date': df.index[i],
                                'exit_price': exit_price,
                                'return_pct': exit_return
                            })
                            exits.iloc[i] = 1.0
                            position.iloc[i] = 0.0
                            entry_dates.iloc[i] = 0
                            continue

                # New entry signal
                if df['signal'].iloc[i] > 0 and (i == 0 or position.iloc[i - 1] == 0):
                    position.iloc[i] = df['signal'].iloc[i]
                    entries.iloc[i] = df['signal'].iloc[i]
                    entry_dates.iloc[i] = 1
                elif i > 0 and position.iloc[i - 1] > 0:
                    # Hold position (if not expired)
                    if entry_dates.iloc[i] <= self.holding_period:
                        position.iloc[i] = position.iloc[i - 1]

                # Exit signal (strategy says exit)
                if df['signal'].iloc[i] == 0 and (i == 0 or position.iloc[i - 1] > 0):
                    if i > 0 and position.iloc[i - 1] > 0:
                        entry_idx = i - entry_dates.iloc[i] + 1
                        if entry_idx >= 0 and entry_idx < len(df):
                            exit_price = df['close'].iloc[i]
                            entry_price = df['close'].iloc[entry_idx]
                            if entry_price > 0:
                                exit_return = (exit_price / entry_price - 1) * 100
                            else:
                                exit_return = 0
                            trades.append({
                                'entry_idx': entry_idx,
                                'entry_date': df.index[entry_idx],
                                'entry_price': entry_price,
                                'exit_idx': i,
                                'exit_date': df.index[i],
                                'exit_price': exit_price,
                                'return_pct': exit_return
                            })
                            exits.iloc[i] = 1.0
                            position.iloc[i] = 0.0
                            entry_dates.iloc[i] = 0

            # Compute daily returns based on position
            price_returns = df['close'].pct_change()
            daily_returns = position.shift(1) * price_returns

            # Apply fees on trade days (entry/exit)
            trade_days = (entries > 0) | (exits > 0)
            daily_returns[trade_days] -= self.fee_pct

            # Fill NaN values (from pct_change and shift operations)
            daily_returns = daily_returns.fillna(0)

            # Compute equity curve
            equity = initial_equity * (1 + daily_returns).cumprod()

            return {
                'position': position,
                'daily_return': daily_returns,
                'equity': equity,
                'entries': entries,
                'exits': exits,
                'trades': trades
            }

        except Exception as e:
            print(f"Error in compute_positions_and_equity: {e}")
            # Return safe default values
            empty_series = pd.Series(dtype=float, index=close.index if isinstance(close, pd.Series) else pd.DatetimeIndex([]))
            return {
                'position': empty_series,
                'daily_return': empty_series,
                'equity': pd.Series([initial_equity], index=close.index[:1] if isinstance(close, pd.Series) else pd.DatetimeIndex([])),
                'entries': empty_series,
                'exits': empty_series,
                'trades': []
            }

    def compute_metrics(self, equity_series, daily_returns, interval="1d", risk_free_rate=0.02):
        """
        Compute backtest metrics.

        Parameters:
        -----------
        equity_series : pd.Series
            Equity curve.
        daily_returns : pd.Series
            Daily returns (as decimals, not %).
        interval : str
            "1d", "1h", "5m", "1m" for annualization.
        risk_free_rate : float
            Risk-free rate for Sharpe (annual, default 2%).

        Returns:
        --------
        dict with keys: total_return, sharpe_ratio, max_drawdown, win_rate
        """
        # Total return - handle division by zero
        initial_value = equity_series.iloc[0]
        final_value = equity_series.iloc[-1]
        
        if initial_value == 0 or pd.isna(initial_value):
            total_return = 0.0
        else:
            total_return = (final_value / initial_value - 1) * 100

        # Annualization factor
        periods_per_year = {
            "1d": 252,
            "1h": 252 * 6.5,
            "5m": 252 * 6.5 * 12,
            "1m": 252 * 6.5 * 60
        }
        periods = periods_per_year.get(interval, 252)

        # Sharpe ratio
        excess_return = daily_returns.mean() - (risk_free_rate / periods)
        sharpe = (excess_return / daily_returns.std()) * np.sqrt(periods) if daily_returns.std() > 0 else 0

        # Max drawdown
        running_max = equity_series.cummax()
        drawdown = (equity_series / running_max - 1) * 100
        max_dd = drawdown.min()

        # Win rate (% of profitable days)
        profitable_days = (daily_returns > 0).sum()
        total_days = (daily_returns != 0).sum()
        win_rate = (profitable_days / total_days * 100) if total_days > 0 else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate
        }


class MovingAverageCrossover(Strategy):
    """MA50/MA200 crossover strategy."""

    def generate_signals(self, price, indicators_dict):
        """
        Generate signals: 1 when MA50 > MA200, else 0.
        """
        ma50 = indicators_dict.get('ma50')
        ma200 = indicators_dict.get('ma200')

        if ma50 is None or ma200 is None:
            raise ValueError("MovingAverageCrossover requires 'ma50' and 'ma200' in indicators_dict")

        # Signal: 1 when MA50 > MA200
        signal = (ma50 > ma200).astype(float)
        # Shift to avoid lookahead bias
        signal = signal.shift(1).fillna(0)

        return signal


class RSIStrategy(Strategy):
    """RSI-based strategy with two modes: threshold and mean-reversion."""

    def __init__(self, mode="threshold", holding_period=0, position_type="fixed", fee_pct=0.0):
        """
        Parameters:
        -----------
        mode : str
            "threshold" (buy RSI<30, sell RSI>70) or
            "mean_reversion" (buy RSI<50 crossing up, sell RSI>50 crossing down)
        """
        super().__init__(holding_period, position_type, fee_pct)
        self.mode = mode

    def generate_signals(self, price, indicators_dict):
        """
        Generate RSI signals based on mode.
        """
        rsi_values = indicators_dict.get('rsi')

        if rsi_values is None:
            raise ValueError("RSIStrategy requires 'rsi' in indicators_dict")

        if self.mode == "threshold":
            signal = self._threshold_mode(rsi_values)
        elif self.mode == "mean_reversion":
            signal = self._mean_reversion_mode(rsi_values)
        else:
            raise ValueError(f"Unknown RSI mode: {self.mode}")

        # Shift to avoid lookahead bias
        signal = signal.shift(1).fillna(0)

        return signal

    def _threshold_mode(self, rsi):
        """RSI threshold: buy <30, sell >70."""
        signal = pd.Series(0.0, index=rsi.index)

        for i in range(1, len(rsi)):
            prev_signal = signal.iloc[i - 1]

            # Enter on RSI < 30
            if rsi.iloc[i] < 30 and prev_signal == 0:
                signal.iloc[i] = 1
            # Stay in until RSI > 70
            elif rsi.iloc[i] > 70 and prev_signal == 1:
                signal.iloc[i] = 0
            # Hold
            else:
                signal.iloc[i] = prev_signal

        return signal

    def _mean_reversion_mode(self, rsi):
        """RSI mean-reversion: buy RSI crosses above 50, sell crosses below 50."""
        signal = pd.Series(0.0, index=rsi.index)
        prev_pos = 0

        for i in range(1, len(rsi)):
            # Crossing above 50 -> buy
            if rsi.iloc[i] > 50 and rsi.iloc[i - 1] <= 50 and prev_pos == 0:
                signal.iloc[i] = 1
                prev_pos = 1
            # Crossing below 50 -> sell
            elif rsi.iloc[i] < 50 and rsi.iloc[i - 1] >= 50 and prev_pos == 1:
                signal.iloc[i] = 0
                prev_pos = 0
            else:
                signal.iloc[i] = prev_pos

        return signal


class BollingerBandsStrategy(Strategy):
    """Bollinger Bands mean-reversion strategy."""

    def generate_signals(self, price, indicators_dict):
        """
        Buy when price touches lower band, sell when touches upper band.
        """
        upper = indicators_dict.get('bb_upper')
        lower = indicators_dict.get('bb_lower')
        close = indicators_dict.get('close')

        if upper is None or lower is None or close is None:
            raise ValueError("BollingerBandsStrategy requires 'bb_upper', 'bb_lower', 'close' in indicators_dict")

        signal = pd.Series(0.0, index=close.index)

        for i in range(1, len(close)):
            prev_signal = signal.iloc[i - 1]

            # Buy when touching lower band
            if close.iloc[i] <= lower.iloc[i] and prev_signal == 0:
                signal.iloc[i] = 1
            # Sell when touching upper band
            elif close.iloc[i] >= upper.iloc[i] and prev_signal == 1:
                signal.iloc[i] = 0
            else:
                signal.iloc[i] = prev_signal

        signal = signal.shift(1).fillna(0)
        return signal


def buy_hold_equity(close, initial_equity=100):
    """
    Compute buy-and-hold equity curve for comparison.

    Parameters:
    -----------
    close : pd.Series
        Close price series.
    initial_equity : float
        Starting equity.

    Returns:
    --------
    pd.Series
        Equity curve (starting at initial_equity).
    """
    normalized = close / close.iloc[0]
    return initial_equity * normalized