# Crypto: technical analysis, on-chain data and AI

Python examples for a LinkedIn series on reading crypto markets. Educational only, not investment advice.

| Script | What it does |
|---|---|
| `code/sentiment_scoring.py` | Scores headlines for sentiment and averages by day |
| `code/trend_following_backtest.py` | Tests a 200-day moving-average trend filter on BTC against buy-and-hold |
| `code/ml_walk_forward.py` | Walk-forward test of a simple next-day direction classifier |
| `code/make_charts.py` | Builds the chart images in `images/` |

## Run
```
pip install -r requirements.txt
python code/trend_following_backtest.py
```

Price data comes from Yahoo Finance via `yfinance`, so results change as new data arrives.
Past performance does not guarantee future results.

## GenAI for finance
RAG over financial reports, evaluation and guardrails (runs offline): see [genai-finance/](genai-finance/).
