import yfinance as yf
import pandas as pd
from typing import Optional

def download_data(tickers, start, end, interval) -> Optional[pd.DataFrame]:
    """
    Download stock data with proper error handling and validation.

    Args:
        tickers: Stock ticker(s) - string or list
        start: Start date
        end: End date
        interval: Data interval

    Returns:
        DataFrame with OHLCV data or None if failed
    """
    try:
        # Validate inputs
        if not tickers:
            raise ValueError("No tickers provided")

        if not isinstance(tickers, (str, list)):
            raise ValueError("Tickers must be string or list")

        if isinstance(tickers, str):
            tickers = [tickers]

        # Validate dates - ensure they can be converted to datetime
        try:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid date format: {e}")

        if start_dt >= end_dt:
            raise ValueError("Start date must be before end date")

        # Check for reasonable date ranges (not too far in the past/future)
        today = pd.Timestamp.now().normalize()
        min_date = pd.Timestamp('1900-01-01')
        max_date = today + pd.DateOffset(days=365)  # Allow up to 1 year in the future

        if start_dt < min_date or start_dt > max_date:
            raise ValueError(f"Start date must be between {min_date.date()} and {max_date.date()}")

        if end_dt < min_date or end_dt > max_date:
            raise ValueError(f"End date must be between {min_date.date()} and {max_date.date()}")

        # Validate interval
        valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
        if interval not in valid_intervals:
            raise ValueError(f"Invalid interval. Must be one of: {valid_intervals}")

        data = yf.download(
            tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False  # Disable progress bar for cleaner output
        )

        # Check if data was retrieved successfully
        if data is None or data.empty:
            raise ValueError("No data retrieved from Yahoo Finance")

        # Handle single ticker case (yfinance returns different structure)
        if len(tickers) == 1:
            ticker = tickers[0]
            # yfinance sometimes returns inconsistent MultiIndex structures
            # Normalize to (ticker, column) format
            if isinstance(data.columns, pd.MultiIndex):
                # Check if it's already in the right format
                if len(data.columns.levels) == 2 and data.columns.levels[0][0] == ticker:
                    # Already properly formatted
                    pass
                else:
                    # Reconstruct as proper MultiIndex
                    new_columns = pd.MultiIndex.from_product([tickers, ['Open', 'High', 'Low', 'Close', 'Volume']])
                    data.columns = new_columns
            else:
                # Single level columns - convert to MultiIndex
                data.columns = pd.MultiIndex.from_product([tickers, data.columns])

        # Validate data has required columns (be more lenient)
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if isinstance(data.columns, pd.MultiIndex):
            available_columns = data.columns.get_level_values(1).unique()
        else:
            available_columns = data.columns

        missing_columns = [col for col in required_columns if col not in available_columns]
        if missing_columns:
            # Instead of raising error, fill missing columns with NaN
            for col in missing_columns:
                if isinstance(data.columns, pd.MultiIndex):
                    for ticker in tickers:
                        if (ticker, col) not in data.columns:
                            data[(ticker, col)] = float('nan')
                else:
                    if col not in data.columns:
                        data[col] = float('nan')
        # Clean data: drop rows with all NaN in Close column, then forward fill
        if isinstance(data.columns, pd.MultiIndex):
            close_cols = [col for col in data.columns if col[0] == 'Close']
        else:
            close_cols = ['Close'] if 'Close' in data.columns else []
        
        if close_cols:
            data = data.dropna(subset=close_cols, how='all')
        data = data.ffill()

        # Check if we have minimum required data
        if len(data) < 10:  # Require at least 10 data points
            raise ValueError("Insufficient data after cleaning (need at least 10 data points)")

        return data

    except Exception as e:
        print(f"Error downloading data for {tickers}: {str(e)}")
        return None