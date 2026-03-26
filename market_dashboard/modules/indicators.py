import pandas as pd

def moving_averages(price):

    ma50 = price.rolling(50).mean()
    ma200 = price.rolling(200).mean()

    return ma50, ma200


def rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    rs = gain.rolling(period).mean() / loss.rolling(period).mean()

    return 100 - (100/(1+rs))


def macd(close):

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()

    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9).mean()

    return macd_line, signal


def bollinger(close):

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()

    upper = ma20 + 2*std20
    lower = ma20 - 2*std20

    return upper, lower