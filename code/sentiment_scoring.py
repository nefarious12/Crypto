"""Post 8: score headlines for sentiment and aggregate per day.

Uses NLTK's VADER (a simple lexicon model) so it runs offline with no API key.
You can swap score_headline() for an LLM call that returns a score from -1 to +1.

Educational only. Not investment advice.
Install: pip install pandas nltk
Run:     python sentiment_scoring.py
"""
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)
_analyzer = SentimentIntensityAnalyzer()


def score_headline(text: str) -> float:
    """Return a compound sentiment score between -1 (negative) and +1 (positive)."""
    return _analyzer.polarity_scores(text)["compound"]


def daily_sentiment(df: pd.DataFrame) -> pd.Series:
    """df needs columns: date, headline. Returns the mean score per day."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["score"] = df["headline"].map(score_headline)
    return df.groupby(df["date"].dt.date)["score"].mean().rename("sentiment")


if __name__ == "__main__":
    # Made-up sample headlines for demonstration. Replace with data you are allowed to use.
    sample = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "headline": [
                "Bitcoin rallies as institutional demand grows",
                "Exchange hack raises fears among investors",
                "Regulators approve new crypto product, markets cheer",
                "Major token plunges after security breach",
            ],
        }
    )
    print(daily_sentiment(sample).round(3))
    # Next step: join with daily price returns and test whether sentiment has any predictive value.
