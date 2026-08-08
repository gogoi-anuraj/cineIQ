import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

def load_rt_reviews(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def clean_rt_reviews(rt_reviews: pd.DataFrame, text_col="review_content") -> pd.DataFrame:
    """Drops reviews with no text — nothing to classify."""
    before = len(rt_reviews)
    cleaned = rt_reviews.dropna(subset=[text_col]).copy()
    cleaned = cleaned[cleaned[text_col].str.len() > 0]
    print(f"dropped {before - len(cleaned)} empty/null reviews ({100*(before-len(cleaned))/before:.1f}%)")
    return cleaned

def load_sentiment_classifier(model_path: str, device="cpu"):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer


def classify_batch(texts: list, model, tokenizer, device="cpu", max_length=256, batch_size=32):
    """Returns a list of predicted labels (1=positive, 0=negative) for a list of texts."""
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoding = tokenizer(
            batch_texts, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoding)
        preds = outputs.logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
    return all_preds

def rerank_by_sentiment(top_n_movie_ids: list, movie_sentiment: pd.DataFrame,
                         sentiment_weight: float = 0.3) -> list:
    """
    Re-ranks a Top-N list of movieIds using sentiment as a soft signal,
    per spec 5.5: sentiment adjusts ORDERING within the Top-N, it never
    adds or removes candidates, and it never re-ranks using unreliable
    sentiment (< 5 reviews) -- those movies keep their original
    meta-model position unchanged.

    Approach: blend original rank position with sentiment score into a
    combined score, so a movie's original rank still dominates (avoids
    letting sentiment alone override a strong collaborative/content/
    sequential signal -- consistent with 5.5's role as a lightweight
    nudge, not a primary ranking signal, especially given the 82.4%
    domain-shift accuracy found during validation).

    sentiment_weight: 0-1, how much influence sentiment has relative to
    original rank. 0.3 is a conservative starting point given the
    validated accuracy gap; can be tuned once end-to-end results are
    visible.
    """
    n = len(top_n_movie_ids)
    sentiment_lookup = movie_sentiment.set_index("movieId")[["sentiment_score", "reliable"]].to_dict("index")

    scored = []
    for rank, movie_id in enumerate(top_n_movie_ids):
        # Original rank position, normalized to [0, 1], higher = better (was higher-ranked)
        original_rank_score = 1.0 - (rank / max(n - 1, 1))

        info = sentiment_lookup.get(movie_id)
        if info is None or not info["reliable"]:
            # No reliable sentiment data -> keep original rank score unchanged
            combined_score = original_rank_score
            sentiment_used = None
        else:
            sentiment_score = info["sentiment_score"]
            combined_score = (1 - sentiment_weight) * original_rank_score + sentiment_weight * sentiment_score
            sentiment_used = sentiment_score

        scored.append((movie_id, combined_score, sentiment_used))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [movie_id for movie_id, _, _ in scored]


def rerank_with_explanation(top_n_movie_ids: list, movie_sentiment: pd.DataFrame,
                             movieid_to_title: dict = None, sentiment_weight: float = 0.3) -> pd.DataFrame:
    """Same as rerank_by_sentiment, but returns a full before/after
    comparison table -- useful for sanity-checking the re-ranker's
    behavior and for the explainability layer (Section 5.6) later."""
    reranked = rerank_by_sentiment(top_n_movie_ids, movie_sentiment, sentiment_weight)

    sentiment_lookup = movie_sentiment.set_index("movieId")[["sentiment_score", "review_count", "reliable"]].to_dict("index")

    rows = []
    for movie_id in top_n_movie_ids:
        info = sentiment_lookup.get(movie_id, {})
        rows.append({
            "movieId": movie_id,
            "title": movieid_to_title.get(movie_id, "?") if movieid_to_title else None,
            "original_rank": top_n_movie_ids.index(movie_id) + 1,
            "new_rank": reranked.index(movie_id) + 1,
            "sentiment_score": info.get("sentiment_score"),
            "review_count": info.get("review_count"),
            "reliable": info.get("reliable", False),
        })

    df = pd.DataFrame(rows).sort_values("new_rank")
    return df





if __name__ == "__main__":
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, tokenizer = load_sentiment_classifier("data/interim/distilbert_sentiment", device=device)

    # rt_reviews = load_rt_reviews("data/interim/rt_reviews_matched.parquet")
    # rt_clean = clean_rt_reviews(rt_reviews)
    # print("rt_clean shape:", rt_clean.shape)

    # print("shape:", rt_reviews.shape)
    # print("columns:", rt_reviews.columns.tolist())
    # print("\nsample rows:")
    # print(rt_reviews[["movieId", "review_content"]].head(5) if "review_content" in rt_reviews.columns
    #       else rt_reviews.head(5))

    # # Check for nulls in the review text column specifically
    # text_col = "review_content" if "review_content" in rt_reviews.columns else None
    # if text_col:
    #     print(f"\nnull {text_col}:", rt_reviews[text_col].isna().sum())
    #     print("empty string reviews:", (rt_reviews[text_col].fillna("").str.len() == 0).sum())

    #     lengths = rt_reviews[text_col].fillna("").str.split().str.len()
    #     print("\nreview word count stats:")
    #     print(lengths.describe())

    # spot_check_sample = rt_clean.sample(n=20, random_state=42)[["movieId", "review_content", "review_type"]]
    # print("\n--- spot-check sample (20 reviews) ---")
    # print("Note: review_type is RT's own Fresh/Rotten label -- we can use this")
    # print("as an independent ground truth to check our classifier against,")
    # print("rather than manually labeling from scratch.")
    # for _, row in spot_check_sample.iterrows():
    #     print(f"\n[{row['review_type']}] {row['review_content']}")

    # spot_check = rt_clean[rt_clean["review_type"].isin(["Fresh", "Rotten"])].sample(
    #     n=500, random_state=42
    # ).copy()
    # spot_check["true_label"] = (spot_check["review_type"] == "Fresh").astype(int)

    # preds = classify_batch(spot_check["review_content"].tolist(), model, tokenizer, device=device)
    # spot_check["predicted_label"] = preds

    # accuracy = (spot_check["true_label"] == spot_check["predicted_label"]).mean()
    # print(f"spot-check accuracy on RT reviews (n=500): {accuracy:.4f}")
    # print(f"(compare to IMDB val accuracy: 0.9247)")

    # print("\nconfusion breakdown:")
    # print(pd.crosstab(spot_check["true_label"], spot_check["predicted_label"],
    #                    rownames=["true (Fresh=1)"], colnames=["predicted (pos=1)"]))

    # # Look at a few disagreements to understand WHY
    # disagreements = spot_check[spot_check["true_label"] != spot_check["predicted_label"]]
    # print(f"\n{len(disagreements)} disagreements out of 500 -- sample of 5:")
    # for _, row in disagreements.head(5).iterrows():
    #     print(f"\n[{row['review_type']}] predicted={'pos' if row['predicted_label']==1 else 'neg'}")
    #     print(f"  {row['review_content']}")

    movie_sentiment = pd.read_parquet("data/interim/movie_sentiment.parquet")
    movies_master = pd.read_parquet("data/interim/movies_master.parquet")
    movieid_to_title = dict(zip(movies_master["movieId"], movies_master["title"]))

    # Placeholder test: simulate a Top-10 list (in practice this comes from the meta-model)
    # Using Toy Story's content-similar movies from Week 2 as a stand-in example
    fake_top_10 = [1, 3114, 78499, 106022, 120474, 68919, 153194, 5584, 115875, 119782]

    result = rerank_with_explanation(fake_top_10, movie_sentiment, movieid_to_title, sentiment_weight=0.3)
    print(result.to_string(index=False))

    # Test with movies that have more spread-out sentiment scores, to visibly confirm reordering works
    test_ids = movie_sentiment[movie_sentiment["reliable"]].sort_values("review_count", ascending=False)["movieId"].head(10).tolist()

    # Artificially treat them as a Top-10 in review_count order (NOT sentiment order) to see if
    # the re-ranker pulls high-sentiment movies upward
    result2 = rerank_with_explanation(test_ids, movie_sentiment, movieid_to_title, sentiment_weight=0.3)
    print(result2.to_string(index=False))

    print("\n--- same test, sentiment_weight=0.6 (stronger sentiment influence) ---")
    result3 = rerank_with_explanation(test_ids, movie_sentiment, movieid_to_title, sentiment_weight=0.6)
    print(result3.to_string(index=False))