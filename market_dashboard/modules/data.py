import yfinance as yf

def download_data(tickers, start, end, interval):

    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True
    )

    return data