import pandas as pd

def explain_content_recommendation(recommended_movie_id: int, user_liked_movie_ids: list,
                                    movies_master: pd.DataFrame, top_n_shared: int = 3) -> str:
    """
    Explains a content-based recommendation by finding which specific
    genres/keywords/cast the recommended movie shares with the user's
    liked movies, per spec 5.6's "rule-based template referencing top
    contributing features."
    """
    movie_row = movies_master[movies_master["movieId"] == recommended_movie_id]
    if movie_row.empty:
        return "Recommended based on content similarity to movies you've enjoyed."

    rec_terms = set(movie_row.iloc[0]["genre_list"]) | set(movie_row.iloc[0]["keyword_list"]) | set(movie_row.iloc[0]["cast_list"])

    liked_rows = movies_master[movies_master["movieId"].isin(user_liked_movie_ids)]
    shared_term_counts = {}
    shared_term_sources = {}  # which liked movie(s) share each term

    for _, liked_row in liked_rows.iterrows():
        liked_terms = set(liked_row["genre_list"]) | set(liked_row["keyword_list"]) | set(liked_row["cast_list"])
        shared = rec_terms & liked_terms
        for term in shared:
            shared_term_counts[term] = shared_term_counts.get(term, 0) + 1
            shared_term_sources.setdefault(term, []).append(liked_row["title"])

    if not shared_term_counts:
        return "Recommended based on general similarity to movies you've enjoyed."

    top_terms = sorted(shared_term_counts.items(), key=lambda x: x[1], reverse=True)[:top_n_shared]
    term_list = ", ".join(term for term, _ in top_terms)

    example_movie = shared_term_sources[top_terms[0][0]][0]
    return f"Recommended because it shares {term_list} with {example_movie}, which you liked."

def explain_svd_recommendation(recommended_movie_id: int, svd_score: float,
                                score_percentile: float = None) -> str:
    """
    SVD's latent factors have no direct semantic meaning, so rather than
    fabricating a false "because of feature X" explanation, this uses
    honest, calibrated language about the collaborative signal strength
    itself -- grounded in how confident the prediction is, not in
    factors that don't actually correspond to anything interpretable.
    """
    if score_percentile is not None:
        if score_percentile >= 0.9:
            confidence = "very strong"
        elif score_percentile >= 0.7:
            confidence = "strong"
        elif score_percentile >= 0.5:
            confidence = "moderate"
        else:
            confidence = "some"
        return (f"Recommended because users with similar taste to yours rated this "
                f"highly ({confidence} match, predicted rating {svd_score:.1f}/5.0).")

    return f"Recommended because users with similar taste to yours rated this highly (predicted rating {svd_score:.1f}/5.0)."


def compute_svd_score_percentile(svd_score: float, reference_scores: pd.Series) -> float:
    """Where does this score fall relative to a reference distribution
    (e.g. all SVD scores computed during meta-model training) -- gives
    the confidence language above real calibration instead of arbitrary
    thresholds."""
    return (reference_scores < svd_score).mean()



if __name__ == "__main__":
    movies_master = pd.read_parquet("data/interim/movies_master.parquet")

    # Toy Story 3 recommended based on liking Toy Story + Toy Story 2
    explanation = explain_content_recommendation(78499, [1, 3114], movies_master)
    print(explanation)

    # A case with likely less overlap
    explanation2 = explain_content_recommendation(2797, [1, 3114], movies_master)  # Big (1988)
    print(explanation2)

    print(explain_svd_recommendation(78499, 4.65, score_percentile=0.92))
    print(explain_svd_recommendation(2797, 3.9, score_percentile=0.55))
    print(explain_svd_recommendation(214, 3.3, score_percentile=None))