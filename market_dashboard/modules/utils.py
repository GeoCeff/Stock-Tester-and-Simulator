import pandas as pd

def compute_returns(close):

    return close.pct_change()


def correlation_matrix(returns):

    return returns.corr()