import numpy as np
import pandas as pd

def portfolio_returns(returns, weights):

    weights = np.array(weights)

    port_returns = returns.dot(weights)

    return port_returns


def sharpe_ratio(returns, interval="1d", risk_free_rate=0.02):
    """
    Compute Sharpe ratio with interval-adjusted annualization.
    
    Parameters:
    -----------
    returns : pd.Series or pd.DataFrame
        Daily returns (as decimals, not %).
    interval : str
        "1d", "1h", "5m", "1m" for annualization.
    risk_free_rate : float
        Annual risk-free rate (default 2%).
    
    Returns:
    --------
    float
        Annualized Sharpe ratio.
    """
    periods_per_year = {
        "1d": 252,
        "1h": 252 * 6.5,
        "5m": 252 * 6.5 * 12,
        "1m": 252 * 6.5 * 60
    }
    periods = periods_per_year.get(interval, 252)
    
    excess_return = returns.mean() - (risk_free_rate / periods)
    std_dev = returns.std()
    
    if std_dev > 0:
        return (excess_return / std_dev) * np.sqrt(periods)
    else:
        return 0


def max_drawdown(price):

    rolling_max = price.cummax()

    drawdown = price / rolling_max - 1

    return drawdown.min()


def win_rate(daily_returns):
    """
    Compute win rate (percentage of profitable days).
    
    Parameters:
    -----------
    daily_returns : pd.Series
        Daily returns (as decimals, not %).
    
    Returns:
    --------
    float
        Win rate as percentage (0-100).
    """
    profitable_days = (daily_returns > 0).sum()
    total_days = (daily_returns != 0).sum()
    
    if total_days > 0:
        return (profitable_days / total_days) * 100
    else:
        return 0.0


def portfolio_backtest(prices, weights, rebalance='monthly'):
    """Simulate portfolio backtest with periodic rebalancing."""
    prices = prices.copy().dropna(how='all')
    returns = prices.pct_change().fillna(0)

    if isinstance(weights, dict):
        weights = np.array([weights.get(t, 0) for t in prices.columns])
    else:
        weights = np.array(weights)

    if not np.isclose(weights.sum(), 1.0):
        weights = weights / weights.sum()

    nav = pd.Series(1.0, index=prices.index)
    holdings = weights.copy()

    for i in range(1, len(returns)):
        nav.iloc[i] = nav.iloc[i - 1] * (1 + np.dot(returns.iloc[i], holdings))

        # rebalance at period starts
        if rebalance in ['monthly', 'weekly', 'daily']:
            date = prices.index[i]
            if rebalance == 'monthly' and date.day == 1:
                holdings = weights
            elif rebalance == 'weekly' and date.weekday() == 0:
                holdings = weights
            elif rebalance == 'daily':
                holdings = weights

    daily_returns = nav.pct_change().fillna(0)
    return {
        'nav': nav,
        'returns': daily_returns,
        'sharpe_ratio': sharpe_ratio(daily_returns, interval='1d'),
        'max_drawdown': max_drawdown(nav),
        'win_rate': win_rate(daily_returns)
    }


def value_at_risk(returns, confidence=0.95):
    """Compute historical Value at Risk (VaR)."""
    if isinstance(returns, pd.DataFrame):
        returns = returns.stack()
    return -np.percentile(returns.dropna(), 100 * (1 - confidence))


def conditional_value_at_risk(returns, confidence=0.95):
    """Compute Conditional Value at Risk (CVaR)."""
    if isinstance(returns, pd.DataFrame):
        returns = returns.stack()
    tail = returns[returns <= np.percentile(returns.dropna(), 100 * (1 - confidence))]
    if len(tail) > 0:
        return -tail.mean()
    else:
        return 0.0


def apply_stop_loss_take_profit(trade_df, stop_loss=0.05, take_profit=0.1):
    """Adjust trades by stop-loss and take-profit percentages."""
    adjusted_trades = []
    for trade in trade_df:
        r = trade.get('return_pct', 0) / 100.0
        if r <= -abs(stop_loss):
            trade['exit_reason'] = 'stop_loss'
        elif r >= abs(take_profit):
            trade['exit_reason'] = 'take_profit'
        else:
            trade['exit_reason'] = 'normal'
        adjusted_trades.append(trade)
    return adjusted_trades
