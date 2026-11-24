"""Generate LinkedIn chart images for the series into ../images/.

Educational only. Not investment advice.
Install: pip install pandas numpy yfinance matplotlib requests
Run:     python make_charts.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from ml_walk_forward import load_prices, rsi
from trend_following_backtest import backtest, stats

OUT = Path(__file__).resolve().parent.parent / "images"
OUT.mkdir(exist_ok=True)
SRC = "Source: Yahoo Finance (BTC-USD). Educational only, not investment advice."
plt.rcParams.update({"figure.figsize": (10, 5.6), "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)
    print("saved", name)


def chart_trend(close):  # Post 2
    d = close.loc["2020-01-01":]
    ma50 = close.rolling(50).mean().loc[d.index]
    ma200 = close.rolling(200).mean().loc[d.index]
    fig, ax = plt.subplots()
    ax.plot(d, label="BTC close", color="#222", lw=1)
    ax.plot(ma50, label="50-day MA", color="#1f77b4")
    ax.plot(ma200, label="200-day MA", color="#d62728")
    ax.set_yscale("log")
    ax.set_title("Bitcoin: price with 50- and 200-day moving averages (log scale)")
    ax.legend()
    fig.text(0.01, 0.01, SRC, fontsize=7, color="gray")
    save(fig, "post02_trend.png")


def chart_momentum(close):  # Post 3 (RSI; volume needs the raw frame)
    d = close.loc["2023-01-01":]
    r = rsi(close).loc[d.index]
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(d, color="#222", lw=1)
    a1.set_title("Bitcoin price and 14-day RSI: RSI stays 'overbought' in strong trends")
    a2.plot(r, color="#6a3d9a", lw=1)
    a2.axhline(70, color="r", ls="--", lw=0.8)
    a2.axhline(30, color="g", ls="--", lw=0.8)
    a2.set_ylim(0, 100)
    a2.set_ylabel("RSI")
    fig.text(0.01, 0.01, SRC, fontsize=7, color="gray")
    save(fig, "post03_rsi.png")


def chart_cycles(close):  # Post 4
    halvings = ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"]
    fig, ax = plt.subplots()
    ax.plot(close, color="#222", lw=1)
    ax.set_yscale("log")
    for h in halvings:
        h = pd.Timestamp(h)
        if h >= close.index[0]:
            ax.axvline(h, color="orange", ls="--")
            ax.text(h, ax.get_ylim()[1] * 0.6, " halving", rotation=90, va="top", color="orange")
    ax.set_title("Bitcoin price (log scale) with halving dates")
    fig.text(0.01, 0.01, SRC, fontsize=7, color="gray")
    save(fig, "post04_cycles.png")


def chart_active_addresses():  # Post 6
    url = "https://api.blockchain.info/charts/n-unique-addresses?timespan=3years&format=json&sampled=true"
    try:
        vals = requests.get(url, timeout=30).json()["values"]
    except Exception as e:  # network or API change
        print("skipped post06 chart:", e)
        return
    s = pd.Series({pd.to_datetime(v["x"], unit="s"): v["y"] for v in vals}).sort_index()
    fig, ax = plt.subplots()
    ax.plot(s, color="#2ca02c")
    ax.set_title("Bitcoin: unique active addresses per day")
    fig.text(0.01, 0.01, "Source: Blockchain.com charts API. Educational only, not investment advice.", fontsize=7, color="gray")
    save(fig, "post06_active_addresses.png")


def chart_funding():  # Post 7
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    try:
        r = requests.get(url, params={"symbol": "BTCUSDT", "limit": 1000}, timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
    except Exception as e:
        print("skipped post07 chart:", e)
        return
    df["t"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["rate"] = df["fundingRate"].astype(float) * 100
    fig, ax = plt.subplots()
    ax.bar(df["t"], df["rate"], width=0.3, color=np.where(df["rate"] >= 0, "#2ca02c", "#d62728"))
    ax.set_title("BTCUSDT perpetual funding rate (% per 8h, latest 1000 periods)")
    fig.text(0.01, 0.01, "Source: Binance Futures public API. Educational only, not investment advice.", fontsize=7, color="gray")
    save(fig, "post07_funding.png")


def chart_equity(close):  # Post 9
    strat, bench, _ = backtest(close)
    fig, ax = plt.subplots()
    ax.plot((1 + bench).cumprod(), label="Buy & hold", color="#888")
    ax.plot((1 + strat).cumprod(), label="200-day trend filter", color="#1f77b4")
    ax.set_yscale("log")
    s, b = stats(strat, "s"), stats(bench, "b")
    ax.set_title(f"Growth of $1: max drawdown {s['Max drawdown']:.0%} (filter) vs {b['Max drawdown']:.0%} (buy & hold)")
    ax.legend()
    fig.text(0.01, 0.01, SRC + " Includes 0.1% cost per position change.", fontsize=7, color="gray")
    save(fig, "post09_equity.png")


def chart_overfit(close, split="2021-01-01"):  # Post 10
    lengths = list(range(20, 301, 10))
    in_sample, out_sample = {}, {}
    for n in lengths:
        strat, _, _ = backtest(close, ma_days=n)
        in_sample[n] = stats(strat.loc[:split], "x")["Sharpe"]
        out_sample[n] = stats(strat.loc[split:], "x")["Sharpe"]
    best = max(in_sample, key=in_sample.get)
    fig, ax = plt.subplots()
    ax.plot(list(in_sample), list(in_sample.values()), label=f"In-sample Sharpe (to {split[:4]})", color="#1f77b4")
    ax.plot(list(out_sample), list(out_sample.values()), label=f"Out-of-sample Sharpe ({split[:4]} on)", color="#d62728")
    ax.axvline(best, color="k", ls=":")
    ax.text(best, ax.get_ylim()[0], f" best in-sample: {best}d", va="bottom")
    ax.set_xlabel("Moving average length (days)")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title(f"Picking the best parameter in-sample: {in_sample[best]:.2f} before, {out_sample[best]:.2f} after")
    ax.legend()
    fig.text(0.01, 0.01, SRC, fontsize=7, color="gray")
    save(fig, "post10_overfitting.png")
    print(f"best in-sample MA={best}: in {in_sample[best]:.2f}, out {out_sample[best]:.2f}")


if __name__ == "__main__":
    close = load_prices()
    chart_trend(close)
    chart_momentum(close)
    chart_cycles(close)
    chart_active_addresses()
    chart_funding()
    chart_equity(close)
    chart_overfit(close)
