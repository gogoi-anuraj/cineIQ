import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import json

def build_markov_transitions(ratings_train: pd.DataFrame,
                              user_col="userId", item_col="movieId",
                              timestamp_col="timestamp") -> dict:
    """
    Builds a first-order Markov transition table: for each movie, a
    Counter of how often each other movie immediately follows it across
    all users' chronological train-split sequences.

    Returns: dict[movieId] -> Counter[next_movieId] -> count
    """
    sorted_ratings = ratings_train.sort_values([user_col, timestamp_col])

    transitions = defaultdict(Counter)

    # Group by user, then walk each user's sequence pairwise
    for user_id, group in sorted_ratings.groupby(user_col):
        sequence = group[item_col].tolist()
        for i in range(len(sequence) - 1):
            current_movie = sequence[i]
            next_movie = sequence[i + 1]
            transitions[current_movie][next_movie] += 1

    return transitions

def build_popularity_ranking(ratings_train: pd.DataFrame, item_col="movieId") -> list:
    """
    Simple popularity baseline: movies ranked by train-split rating count,
    most-rated first. Used standalone as its own baseline (Section 7),
    and as a fallback here when Markov has no transitions for a movie.
    """
    counts = ratings_train[item_col].value_counts()
    return counts.index.tolist()


def predict_next_markov(movie_id: int, transitions: dict, popularity_ranking: list,
                         top_n: int = 10) -> list:
    """
    Given a movie, returns the top_n most likely next movies by Markov
    transition frequency. Falls back to the popularity ranking (minus
    the query movie itself) if this movie has no recorded transitions.
    """
    if movie_id in transitions and len(transitions[movie_id]) > 0:
        return [m for m, _ in transitions[movie_id].most_common(top_n)]

    # Fallback: most popular movies, excluding the query movie itself
    fallback = [m for m in popularity_ranking if m != movie_id][:top_n]
    return fallback


def build_eval_pairs(ratings_train: pd.DataFrame, ratings_val: pd.DataFrame,
                      user_col="userId", item_col="movieId", timestamp_col="timestamp") -> pd.DataFrame:
    """
    For each user present in both train and val, builds an eval pair:
    (last movie watched in train, first movie watched in val) — i.e.
    "given what they watched last, what did they actually watch next?"
    This is the ground truth the Markov/GRU models are scored against.
    """
    last_train = (
        ratings_train.sort_values(timestamp_col)
        .groupby(user_col)
        .tail(1)[[user_col, item_col]]
        .rename(columns={item_col: "last_train_movie"})
    )
    first_val = (
        ratings_val.sort_values(timestamp_col)
        .groupby(user_col)
        .head(1)[[user_col, item_col]]
        .rename(columns={item_col: "true_next_movie"})
    )
    eval_pairs = last_train.merge(first_val, on=user_col, how="inner")
    return eval_pairs


def hit_rate_at_k(eval_pairs: pd.DataFrame, transitions: dict, popularity_ranking: list, k=10) -> float:
    hits = 0
    for _, row in eval_pairs.iterrows():
        preds = predict_next_markov(row["last_train_movie"], transitions, popularity_ranking, top_n=k)
        if row["true_next_movie"] in preds:
            hits += 1
    return hits / len(eval_pairs)


def ndcg_at_k(eval_pairs: pd.DataFrame, transitions: dict, popularity_ranking: list, k=10) -> float:
    ndcgs = []
    for _, row in eval_pairs.iterrows():
        preds = predict_next_markov(row["last_train_movie"], transitions, popularity_ranking, top_n=k)
        true_movie = row["true_next_movie"]
        if true_movie in preds:
            rank = preds.index(true_movie)  # 0-indexed
            ndcgs.append(1.0 / np.log2(rank + 2))  # +2 because rank is 0-indexed and log2(1)=0
        else:
            ndcgs.append(0.0)
    return np.mean(ndcgs)

def hit_rate_at_k_popularity_only(eval_pairs: pd.DataFrame, popularity_ranking: list, k=10) -> float:
    """Pure popularity baseline: same top-k for every user, no personalization."""
    top_k_popular = set(popularity_ranking[:k])
    hits = eval_pairs["true_next_movie"].isin(top_k_popular).sum()
    return hits / len(eval_pairs)


def ndcg_at_k_popularity_only(eval_pairs: pd.DataFrame, popularity_ranking: list, k=10) -> float:
    top_k_popular = popularity_ranking[:k]
    ndcgs = []
    for true_movie in eval_pairs["true_next_movie"]:
        if true_movie in top_k_popular:
            rank = top_k_popular.index(true_movie)
            ndcgs.append(1.0 / np.log2(rank + 2))
        else:
            ndcgs.append(0.0)
    return np.mean(ndcgs)

def markov_fallback_rate(eval_pairs: pd.DataFrame, transitions: dict) -> float:
    """What fraction of eval users hit the popularity fallback because
    their last-watched movie had no recorded Markov transitions?"""
    fallback_count = sum(
        1 for m in eval_pairs["last_train_movie"]
        if m not in transitions or len(transitions[m]) == 0
    )
    return fallback_count / len(eval_pairs)

def save_baseline_results(results: dict, path: str):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved baseline results to {path}")


# if __name__ == "__main__":
#     ratings_train = pd.read_parquet("data/interim/ratings_train.parquet")
#     print("ratings_train shape:", ratings_train.shape)

#     sample_users = ratings_train["userId"].drop_duplicates().sample(n=2000, random_state=42)
#     ratings_sample = ratings_train[ratings_train["userId"].isin(sample_users)]
#     print("sample shape:", ratings_sample.shape)

#     import time
#     start = time.time()
#     transitions = build_markov_transitions(ratings_sample)
#     elapsed = time.time() - start
#     print(f"\nbuilt transitions in {elapsed:.1f}s")
#     print("movies with at least one outgoing transition:", len(transitions))

#     # Inspect: most common transitions FROM Toy Story (movieId=1), if present
#     if 1 in transitions:
#         print("\ntop transitions from Toy Story (movieId=1):")
#         for movie_id, count in transitions[1].most_common(10):
#             print(f"  -> {movie_id}: {count}")
#     else:
#         print("\nmovieId 1 has no outgoing transitions in this sample")

#     popularity_ranking = build_popularity_ranking(ratings_sample)
#     print("top 10 most popular movies (sample):", popularity_ranking[:10])

#     # Test: a movie WITH transitions (Toy Story)
#     preds = predict_next_markov(1, transitions, popularity_ranking, top_n=10)
#     print("\npredicted next movies after Toy Story (movieId=1):", preds)

#     # Test: a movie with NO transitions (pick a rare movie not in the transitions dict)
#     all_movies = set(ratings_sample["movieId"].unique())
#     movies_without_transitions = all_movies - set(transitions.keys())
#     if movies_without_transitions:
#         test_movie = list(movies_without_transitions)[0]
#         preds_fallback = predict_next_markov(test_movie, transitions, popularity_ranking, top_n=10)
#         print(f"\npredicted next movies after movieId={test_movie} (no transitions -> popularity fallback):")
#         print(preds_fallback)

#     ratings_val_warm = pd.read_parquet("data/interim/ratings_val_warm.parquet")

#     eval_pairs = build_eval_pairs(ratings_sample, ratings_val_warm)
#     print("eval pairs built:", len(eval_pairs))
#     print(eval_pairs.head())

#     hr10 = hit_rate_at_k(eval_pairs, transitions, popularity_ranking, k=10)
#     ndcg10 = ndcg_at_k(eval_pairs, transitions, popularity_ranking, k=10)
#     print(f"\nHit Rate@10: {hr10:.4f}")
#     print(f"NDCG@10: {ndcg10:.4f}")


if __name__ == "__main__":
    import time

    ratings_train = pd.read_parquet("data/interim/ratings_train.parquet")
    ratings_val_warm = pd.read_parquet("data/interim/ratings_val_warm.parquet")
    print("ratings_train shape:", ratings_train.shape)

    start = time.time()
    transitions = build_markov_transitions(ratings_train)
    elapsed = time.time() - start
    print(f"built transitions in {elapsed:.1f}s")
    print("movies with at least one outgoing transition:", len(transitions))

    popularity_ranking = build_popularity_ranking(ratings_train)

    eval_pairs = build_eval_pairs(ratings_train, ratings_val_warm)
    print("\neval pairs built:", len(eval_pairs))

    start = time.time()
    hr10 = hit_rate_at_k(eval_pairs, transitions, popularity_ranking, k=10)
    ndcg10 = ndcg_at_k(eval_pairs, transitions, popularity_ranking, k=10)
    elapsed = time.time() - start
    print(f"evaluation completed in {elapsed:.1f}s")

    print(f"\nHit Rate@10: {hr10:.4f}")
    print(f"NDCG@10: {ndcg10:.4f}")


    hr10_pop = hit_rate_at_k_popularity_only(eval_pairs, popularity_ranking, k=10)
    ndcg10_pop = ndcg_at_k_popularity_only(eval_pairs, popularity_ranking, k=10)
    fallback_rate = markov_fallback_rate(eval_pairs, transitions)

    print(f"\n--- Comparison ---")
    print(f"Markov  Hit Rate@10: {hr10:.4f}   NDCG@10: {ndcg10:.4f}")
    print(f"Popularity Hit Rate@10: {hr10_pop:.4f}   NDCG@10: {ndcg10_pop:.4f}")
    print(f"\nfraction of eval users who hit the popularity fallback: {fallback_rate:.1%}")

    results = {
        "markov": {"hit_rate_at_10": round(hr10, 4), "ndcg_at_10": round(ndcg10, 4)},
        "popularity": {"hit_rate_at_10": round(hr10_pop, 4), "ndcg_at_10": round(ndcg10_pop, 4)},
        "markov_fallback_rate": round(fallback_rate, 4),
        "eval_users": len(eval_pairs),
        "movies_with_transitions": len(transitions),
        "transition_build_time_seconds": round(elapsed, 1),
    }

    save_baseline_results(results, "data/interim/baseline_results.json")
    print(json.dumps(results, indent=2))