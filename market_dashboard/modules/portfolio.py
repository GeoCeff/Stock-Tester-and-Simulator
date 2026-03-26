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