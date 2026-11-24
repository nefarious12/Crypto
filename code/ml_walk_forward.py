"""Post 11: walk-forward test of a simple classifier predicting next-day BTC direction.

Educational only. Not investment advice.
Install: pip install pandas numpy scikit-learn yfinance
Run:     python ml_walk_forward.py
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_prices(ticker="BTC-USD", start="2015-01-01"):
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna().rename("close")


def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / loss)


def make_features(close):
    ret = close.pct_change()
    X = pd.DataFrame(
        {
            "ret_1d": ret,
            "ret_5d": close.pct_change(5),
            "vol_20d": ret.rolling(20).std(),
            "rsi_14": rsi(close),
            "dist_ma50": close / close.rolling(50).mean() - 1,
        }
    )
    # Target: is TOMORROW's return positive? Features use only data up to today.
    y = (ret.shift(-1) > 0).astype(int)
    data = X.join(y.rename("target")).dropna()
    # The last row has no known next-day return, so drop it.
    data = data.iloc[:-1]
    return data.drop(columns="target"), data["target"]


def walk_forward(X, y, train_days=730, test_days=90):
    """Train on a rolling past window, predict the next block, then roll forward."""
    preds = pd.Series(index=y.index, dtype=float)
    start = train_days
    while start < len(X):
        end = min(start + test_days, len(X))
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(X.iloc[start - train_days:start], y.iloc[start - train_days:start])
        preds.iloc[start:end] = model.predict(X.iloc[start:end])
        start = end
    return preds.dropna()


if __name__ == "__main__":
    close = load_prices()
    X, y = make_features(close)
    preds = walk_forward(X, y)
    actual = y.loc[preds.index]
    accuracy = (preds == actual).mean()
    baseline = max(actual.mean(), 1 - actual.mean())  # always guess the majority class
    print(f"Out-of-sample predictions: {len(preds)}")
    print(f"Model accuracy:            {accuracy:.3f}")
    print(f"Always-majority baseline:  {baseline:.3f}")
