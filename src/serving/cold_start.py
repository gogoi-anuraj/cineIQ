"""
cold_start.py

Cold-start routing (spec Section 5.4, "required, not optional"). Users
with fewer than N=25 train-period ratings get routed AROUND the
meta-model entirely -- SVD and GRU both need substantial watch history
to produce meaningful (non-degenerate) scores, and the meta-model itself
was trained only on users with sufficient history, so it has no learned
way to handle sparse-history users gracefully.

Fallback: a blend of content-based similarity (from whatever few movies
the user HAS liked) and popularity.
"""

import sys
import pandas as pd
import numpy as np

sys.path.append("src")  # adjust to your actual import setup
from models.content_model import score_candidates_for_user as content_score_candidates
from models.baselines import build_popularity_ranking


# ---------------------------------------------------------------------------
# Routing decision
# ---------------------------------------------------------------------------

def is_cold_start_user(user_id, ratings_train: pd.DataFrame, threshold: int = 25,
                        user_col: str = "userId") -> bool:
    """True if the user has fewer than `threshold` train-period ratings."""
    user_rating_count = (ratings_train[user_col] == user_id).sum()
    return user_rating_count < threshold


def get_cold_start_users(ratings_train: pd.DataFrame, threshold: int = 25,
                          user_col: str = "userId") -> set:
    """Batch version -- the full set of cold-start userIds. More
    efficient than repeated is_cold_start_user() calls."""
    counts = ratings_train.groupby(user_col).size()
    return set(counts[counts < threshold].index)


# ---------------------------------------------------------------------------
# Fallback recommendation (content-based + popularity blend)
# ---------------------------------------------------------------------------

def cold_start_recommend(user_liked_movie_ids: list, ratings_train: pd.DataFrame,
                          movies_master: pd.DataFrame, tfidf_matrix,
                          candidate_movie_ids: list = None, top_n: int = 10,
                          content_weight: float = 0.5) -> pd.DataFrame:
    """
    Serves cold-start users a blend of content-based score and
    popularity. Never touches SVD or GRU.

    ratings_train: only needs a `movieId` column -- pass
    pd.read_parquet(path, columns=["movieId"]) to avoid loading the full
    20.8M-row table into memory unnecessarily.

    Already-liked movies are excluded from candidates (a user shouldn't
    be re-recommended something they told us they already liked -- same
    principle as the GRU's seen-item masking).

    If candidate_movie_ids is None, scores the full catalog. For a truly
    new user with almost no signal, consider restricting to a smaller
    candidate pool (e.g. top 2,000 most-popular movies) for speed, since
    obscure long-tail titles are rarely the right call here anyway.
    """
    if candidate_movie_ids is None:
        candidate_movie_ids = movies_master["movieId"].tolist()

    liked_set = set(user_liked_movie_ids)
    candidate_movie_ids = [m for m in candidate_movie_ids if m not in liked_set]

    popularity_ranking = build_popularity_ranking(ratings_train)
    pop_rank_lookup = {m: i for i, m in enumerate(popularity_ranking)}
    max_rank = len(popularity_ranking)

    def popularity_score(movie_id):
        rank = pop_rank_lookup.get(movie_id)
        if rank is None:
            return 0.0
        return 1.0 - (rank / max_rank)

    content_scores = content_score_candidates(
        user_liked_movie_ids, candidate_movie_ids, movies_master, tfidf_matrix
    )
    content_scores["popularity_score"] = content_scores["movieId"].apply(popularity_score)
    content_scores["blended_score"] = (
        content_weight * content_scores["content_score"]
        + (1 - content_weight) * content_scores["popularity_score"]
    )

    result = content_scores.sort_values("blended_score", ascending=False).head(top_n)
    return result.merge(movies_master[["movieId", "title"]], on="movieId")


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pickle
    from scipy import sparse

    ratings_train = pd.read_parquet("data/interim/ratings_train.parquet", columns=["userId", "movieId"])
    warm_users = set(pd.read_parquet("data/interim/warm_users.parquet")["userId"])
    movies_master = pd.read_parquet("data/interim/movies_master.parquet")

    with open("data/interim/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    tfidf_matrix = sparse.load_npz("data/interim/tfidf_matrix.npz")

    # --- Sanity check: cold-start / warm split agrees with split.py's definition ---
    cold_start_users = get_cold_start_users(ratings_train, threshold=25)
    print("cold-start users:", len(cold_start_users))
    print("warm users:", len(warm_users))
    print("overlap (should be 0):", len(cold_start_users & warm_users))
    print("union == all train users:", (cold_start_users | warm_users) == set(ratings_train["userId"].unique()))

    # --- Fallback recommendation test ---
    ratings_movieid_only = ratings_train[["movieId"]]
    new_user_liked = [1, 3114]  # Toy Story, Toy Story 2 -- e.g. their only 2 ratings so far

    recs = cold_start_recommend(
        new_user_liked, ratings_movieid_only, movies_master, tfidf_matrix,
        candidate_movie_ids=None, top_n=10, content_weight=0.5
    )
    print("\ncold-start recommendations:")
    print(recs.to_string())