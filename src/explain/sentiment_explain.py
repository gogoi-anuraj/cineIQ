import pandas as pd

def explain_sentiment_adjustment(movie_id: int, movie_sentiment: pd.DataFrame) -> str:
    """
    Rule-based explanation for the sentiment re-ranker's adjustment
    (spec 5.6): references the aggregate sentiment score of the linked
    critic reviews used. Returns None-equivalent language if the movie
    has no reliable sentiment data (consistent with rerank.py leaving
    such movies untouched).
    """
    row = movie_sentiment[movie_sentiment["movieId"] == movie_id]
    if row.empty or not row.iloc[0]["reliable"]:
        return "Ranking unaffected by critic sentiment (insufficient review data)."

    score = row.iloc[0]["sentiment_score"]
    review_count = int(row.iloc[0]["review_count"])

    if score >= 0.75:
        tone = "overwhelmingly positive"
    elif score >= 0.55:
        tone = "generally positive"
    elif score >= 0.45:
        tone = "mixed"
    elif score >= 0.25:
        tone = "generally negative"
    else:
        tone = "overwhelmingly negative"

    return f"Adjusted based on {tone} critic reception ({review_count} reviews, {score*100:.0f}% positive)."


if __name__ == "__main__":
    movie_sentiment = pd.read_parquet("data/interim/movie_sentiment.parquet")

    for movie_id in [5359, 5456, 1, 999999999]:  # high, low, mid, and a nonexistent id
        print(explain_sentiment_adjustment(movie_id, movie_sentiment))