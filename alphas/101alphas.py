import pandas as pd
import numpy as np
from talib import WMA


# Helper Functions
def rank(df):
    return df.rank(axis=1, pct=True)

def scale(df):
    return df.div(df.abs().sum(axis=1), axis=0)

def log(df):
    return np.log1p(df)

def sign(df):
    return np.sign(df)

def power(df, exp):
    return df.pow(exp)

def ts_lag(df, t=1):
    return df.shift(t)

def ts_delta(df, period=1):
    return df.diff(period)

def ts_sum(df, window=10):
    return df.rolling(window).sum()

def ts_mean(df, window=10):
    return df.rolling(window).mean()

def ts_weighted_mean(df, period=10):
    return df.apply(lambda x: WMA(x, timeperiod=period))

def ts_std(df, window=10):
    return df.rolling(window).std()

def ts_rank(df, window=10):
    return df.rolling(window).apply(lambda x: x.rank().iloc[-1])

def ts_product(df, window=10):
    return df.rolling(window).apply(np.prod)

def ts_min(df, window=10):
    return df.rolling(window).min()

def ts_max(df, window=10):
    return df.rolling(window).max()

def ts_argmin(df, window=10):
    return df.rolling(window).apply(np.argmin).add(1)

def ts_argmax(df, window=10):
    return df.rolling(window).apply(np.argmin).add(1)

def ts_corr(x, y, window=10):
    return x.rolling(window).corr(y)

def ts_cov(x, y, window=10):
    return x.rolling(window).cov(y)




# ALPHAS

def alpha001(c, r):
    # rank(ts_argmax(power(((returns < 0) ? ts_std(returns, 20) : close), 2.), 5))
    c[r < 0] = ts_std(r, 20)
    return rank