"""Post 9: simple 200-day moving-average trend filter on BTC vs. buy-and-hold.

Educational only. Not investment advice.
Install: pip install pandas numpy yfinance
Run:     python trend_following_backtest.py
"""
import numpy as np
import pandas as pd
import yfinance as yf

MA_DAYS = 200
FEE = 0.001  # 0.1% cost per position change (fees + slippage estimate)
DAYS_PER_YEAR = 365  # crypto trades every day


def load_prices(ticker="BTC-USD", start="2015-01-01"):
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns one column per ticker
        close = close.iloc[:, 0]
    return close.dropna().rename("close")


def backtest(close, ma_days=MA_DAYS, fee=FEE):
    ma = close.rolling(ma_days).mean()
    # Signal is known at today's close; shift(1) means we hold it from the NEXT day.
    signal = (close > ma).astype(int).where(ma.notna()).shift(1)
    ret = close.pct_change()
    trades = signal.diff().abs().fillna(0)
    strat = (signal * ret - trades * fee).dropna()
    bench = ret.loc[strat.index]
    return strat, bench, int(trades.sum())


def stats(r, name):
    equity = (1 + r).cumprod()
    years = len(r) / DAYS_PER_YEAR
    cagr = equity.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(DAYS_PER_YEAR)
    sharpe = (r.mean() * DAYS_PER_YEAR) / vol if vol > 0 else np.nan
    max_dd = (equity / equity.cummax() - 1).min()
    return pd.Series(
        {"CAGR": cagr, "Volatility": vol, "Sharpe": sharpe, "Max drawdown": max_dd},
        name=name,
    )


if __name__ == "__main__":
    close = load_prices()
    strat, bench, n_trades = backtest(close)
    table = pd.concat([stats(strat, f"{MA_DAYS}d trend filter"), stats(bench, "Buy & hold")], axis=1)
    print(table.round(3))
    print(f"Position changes: {n_trades}")
