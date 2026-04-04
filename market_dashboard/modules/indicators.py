import pandas as pd
import numpy as np

def moving_averages(price):
    """
    Calculate moving averages with proper error handling.

    Args:
        price: pandas Series of prices

    Returns:
        tuple: (ma50, ma200) or (None, None) if insufficient data
    """
    try:
        if not isinstance(price, pd.Series):
            raise ValueError("Price must be a pandas Series")

        if len(price) < 50:
            return None, None  # Insufficient data for MA50

        ma50 = price.rolling(50).mean()

        if len(price) < 200:
            ma200 = None  # Insufficient data for MA200
        else:
            ma200 = price.rolling(200).mean()

        return ma50, ma200

    except Exception as e:
        print(f"Error calculating moving averages: {e}")
        return None, None


def rsi(close, period=14):
    """
    Calculate RSI with proper error handling.

    Args:
        close: pandas Series of closing prices
        period: RSI period (default 14)

    Returns:
        pandas Series of RSI values or None if error
    """
    try:
        if not isinstance(close, pd.Series):
            raise ValueError("Close must be a pandas Series")

        if len(close) < period + 1:
            return pd.Series(dtype=float)  # Insufficient data

        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        rs = gain.rolling(period).mean() / loss.rolling(period).mean()

        # Handle division by zero
        rs = rs.replace([np.inf, -np.inf], np.nan)

        rsi_values = 100 - (100/(1+rs))

        # Fill NaN values with neutral RSI (50)
        rsi_values = rsi_values.fillna(50)

        return rsi_values

    except Exception as e:
        print(f"Error calculating RSI: {e}")
        return pd.Series(dtype=float)


def macd(close):
    """
    Calculate MACD with proper error handling.

    Args:
        close: pandas Series of closing prices

    Returns:
        tuple: (macd_line, signal) or (None, None) if insufficient data
    """
    try:
        if not isinstance(close, pd.Series):
            raise ValueError("Close must be a pandas Series")

        if len(close) < 26:
            return None, None  # Insufficient data

        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()

        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9).mean()

        return macd_line, signal

    except Exception as e:
        print(f"Error calculating MACD: {e}")
        return None, None


def bollinger(close, period=20, std_dev=2):
    """
    Calculate Bollinger Bands with proper error handling.

    Args:
        close: pandas Series of closing prices
        period: Moving average period (default 20)
        std_dev: Standard deviation multiplier (default 2)

    Returns:
        tuple: (upper, lower) or (None, None) if insufficient data
    """
    try:
        if not isinstance(close, pd.Series):
            raise ValueError("Close must be a pandas Series")

        if len(close) < period:
            return None, None  # Insufficient data

        ma20 = close.rolling(period).mean()
        std20 = close.rolling(period).std()

        upper = ma20 + std_dev * std20
        lower = ma20 - std_dev * std20

        return upper, lower

    except Exception as e:
        print(f"Error calculating Bollinger Bands: {e}")
        return None, None