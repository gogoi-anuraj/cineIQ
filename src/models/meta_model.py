"""
meta_model.py

Ensemble meta-model (spec Section 5.4) — blends SVD, content-based, and
GRU scores via a learned logistic regression, trained on the validation
split. Coefficients double as the explainability signal for "how much
did each component contribute" (Section 5.6/7).

Result from Week 3 training: test AUC 0.7877 (vs GBM's 0.7929 -- kept
logistic regression for its clearer, more balanced signal-contribution
story; the GBM concentrated ~95% of importance on SVD alone). SVD
dominates the blend not because content/GRU are uninformative, but
because SVD's score distribution has much lower variance -- see
meta_model_results.json's "finding" field for the full explanation.

This module assumes the three base signals are already trained and
their artifacts are loadable via svd_model.py, content_model.py, and
gru_model.py's respective load functions.
"""

import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.append("src")
from models.svd_model import load_svd_model, score_batch as svd_score_batch
from models.content_model import score_candidates_for_user as content_score_candidates
from models.gru_model import load_gru_artifacts, score_candidates_for_user as gru_score_candidates

FEATURE_COLS = ["svd_score", "content_score", "gru_score"]


# ---------------------------------------------------------------------------
# Build labeled training data
# ---------------------------------------------------------------------------

def build_labeled_examples(ratings_val_warm: pd.DataFrame, all_movie_ids: list,
                            positive_threshold: float = 4.0, n_random_negatives_per_user: int = 5,
                            random_state: int = 42) -> pd.DataFrame:
    """
    (userId, movieId, label) examples: real val-split ratings
    (>=threshold -> 1, else 0) plus sampled random negatives (movies
    never rated at all) -- both are useful "not relevant" signal but
    aren't the same thing, so both are included.
    """
    rng = np.random.RandomState(random_state)

    real_examples = ratings_val_warm[["userId", "movieId", "rating"]].copy()
    real_examples["label"] = (real_examples["rating"] >= positive_threshold).astype(int)
    real_examples = real_examples[["userId", "movieId", "label"]]

    user_seen = ratings_val_warm.groupby("userId")["movieId"].apply(set).to_dict()
    all_movie_ids_arr = np.array(all_movie_ids)

    negative_rows = []
    for user_id, seen_movies in user_seen.items():
        candidates = rng.choice(all_movie_ids_arr, size=n_random_negatives_per_user * 3, replace=False)
        chosen = [m for m in candidates if m not in seen_movies][:n_random_negatives_per_user]
        for movie_id in chosen:
            negative_rows.append({"userId": user_id, "movieId": movie_id, "label": 0})

    return pd.concat([real_examples, pd.DataFrame(negative_rows)], ignore_index=True)


def build_user_train_sequences(ratings_train: pd.DataFrame, movie_to_idx: dict,
                                user_col="userId", item_col="movieId", timestamp_col="timestamp",
                                max_seq_len: int = 100) -> dict:
    sorted_ratings = ratings_train.sort_values([user_col, timestamp_col])
    sequences = {}
    for user_id, group in sorted_ratings.groupby(user_col):
        encoded = [movie_to_idx[m] for m in group[item_col].tolist() if m in movie_to_idx]
        if len(encoded) > max_seq_len:
            encoded = encoded[-max_seq_len:]
        sequences[user_id] = encoded
    return sequences


def score_all_signals(examples: pd.DataFrame, svd_model, tfidf_matrix, movies_master,
                       gru_model, gru_movie_to_idx, user_train_sequences: dict,
                       ratings_train_by_user: dict, gru_sentinel_value: float = None,
                       device: str = "cpu") -> pd.DataFrame:
    """
    Scores every (userId, movieId) example with all 3 signals, grouped
    by user. gru_sentinel_value replaces -inf scores (movies outside the
    GRU's training vocabulary) -- if None, computed as the min real
    score in this batch.
    """
    scored_chunks = []
    for user_id, group in examples.groupby("userId"):
        candidate_movie_ids = group["movieId"].tolist()

        svd_pairs = pd.DataFrame({"userId": user_id, "movieId": candidate_movie_ids})
        svd_scored = svd_score_batch(svd_model, svd_pairs)

        liked_movies = ratings_train_by_user.get(user_id, [])
        content_scored = content_score_candidates(liked_movies, candidate_movie_ids, movies_master, tfidf_matrix)

        user_seq = user_train_sequences.get(user_id, [])
        if len(user_seq) >= 1:
            gru_scored = gru_score_candidates(gru_model, user_seq, candidate_movie_ids, gru_movie_to_idx,
                                               device=device, exclude_seen=False)
        else:
            gru_scored = pd.DataFrame({"movieId": candidate_movie_ids, "gru_score": 0.0})

        merged = group.merge(svd_scored[["movieId", "svd_score"]], on="movieId", how="left")
        merged = merged.merge(content_scored[["movieId", "content_score"]], on="movieId", how="left")
        merged = merged.merge(gru_scored[["movieId", "gru_score"]], on="movieId", how="left")
        scored_chunks.append(merged)

    result = pd.concat(scored_chunks, ignore_index=True)

    if gru_sentinel_value is None:
        finite = result.loc[~np.isinf(result["gru_score"]), "gru_score"]
        gru_sentinel_value = finite.min() if len(finite) else 0.0
    result["gru_score"] = result["gru_score"].replace(-np.inf, gru_sentinel_value)

    return result


# ---------------------------------------------------------------------------
# Train the meta-model
# ---------------------------------------------------------------------------

def train_meta_model(scored_examples: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Trains logistic regression on the 3 signal scores. Returns
    (model, scaler, metrics_dict)."""
    X = scored_examples[FEATURE_COLS]
    y = scored_examples["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train_scaled)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

    metrics = {
        "train_auc": round(train_auc, 4),
        "test_auc": round(test_auc, 4),
        "coefficients": {name: round(float(c), 4) for name, c in zip(FEATURE_COLS, model.coef_[0])},
        "intercept": round(float(model.intercept_[0]), 4),
    }
    return model, scaler, metrics


# ---------------------------------------------------------------------------
# Inference — score a candidate pool for one user, blend, rank
# ---------------------------------------------------------------------------

def generate_top_n(user_id, ratings_train: pd.DataFrame, user_train_sequences: dict,
                    ratings_train_by_user: dict, svd_model, tfidf_matrix, movies_master,
                    gru_model, gru_movie_to_idx, meta_model, scaler, gru_sentinel_value: float,
                    candidate_movie_ids: list = None, top_n: int = 10, device: str = "cpu") -> pd.DataFrame:
    """
    Full meta-model pipeline for one WARM user (caller is responsible for
    cold-start routing -- see src/serving/pipeline.py for the unified
    function that decides which path to use).
    """
    if candidate_movie_ids is None:
        already_seen = set(ratings_train[ratings_train["userId"] == user_id]["movieId"])
        candidate_movie_ids = [m for m in gru_movie_to_idx.keys() if m not in already_seen]

    svd_pairs = pd.DataFrame({"userId": user_id, "movieId": candidate_movie_ids})
    svd_scored = svd_score_batch(svd_model, svd_pairs)

    liked_movies = ratings_train_by_user.get(user_id, [])
    content_scored = content_score_candidates(liked_movies, candidate_movie_ids, movies_master, tfidf_matrix)

    user_seq = user_train_sequences.get(user_id, [])
    gru_scored = gru_score_candidates(gru_model, user_seq, candidate_movie_ids, gru_movie_to_idx,
                                       device=device, exclude_seen=False)

    merged = svd_scored[["movieId", "svd_score"]].merge(
        content_scored[["movieId", "content_score"]], on="movieId", how="inner"
    ).merge(gru_scored[["movieId", "gru_score"]], on="movieId", how="inner")

    merged["gru_score"] = merged["gru_score"].replace(-np.inf, gru_sentinel_value)

    X = scaler.transform(merged[FEATURE_COLS])
    merged["meta_score"] = meta_model.predict_proba(X)[:, 1]

    top_n_result = merged.sort_values("meta_score", ascending=False).head(top_n)
    return top_n_result.merge(movies_master[["movieId", "title"]], on="movieId")


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_meta_model(model, scaler, gru_sentinel_value, model_path, scaler_path, sentinel_path):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(sentinel_path, "wb") as f:
        pickle.dump(gru_sentinel_value, f)


def load_meta_model(model_path, scaler_path, sentinel_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(sentinel_path, "rb") as f:
        gru_sentinel_value = pickle.load(f)
    return model, scaler, gru_sentinel_value


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from scipy import sparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings_train", default="data/interim/ratings_train.parquet")
    parser.add_argument("--ratings_val_warm", default="data/interim/ratings_val_warm.parquet")
    parser.add_argument("--movies_master", default="data/interim/movies_master.parquet")
    parser.add_argument("--svd_model", default="data/interim/svd_model.pkl")
    parser.add_argument("--tfidf_vectorizer", default="data/interim/tfidf_vectorizer.pkl")
    parser.add_argument("--tfidf_matrix", default="data/interim/tfidf_matrix.npz")
    parser.add_argument("--gru_model", default="data/interim/gru_model_final.pt")
    parser.add_argument("--gru_mappings", default="data/interim/gru_mappings.pkl")
    parser.add_argument("--out_model", default="data/interim/meta_model.pkl")
    parser.add_argument("--out_scaler", default="data/interim/meta_model_scaler.pkl")
    parser.add_argument("--out_sentinel", default="data/interim/gru_sentinel.pkl")
    parser.add_argument("--out_results", default="data/interim/meta_model_results.json")
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    svd_model = load_svd_model(args.svd_model)
    with open(args.tfidf_vectorizer, "rb") as f:
        vectorizer = pickle.load(f)
    tfidf_matrix = sparse.load_npz(args.tfidf_matrix)
    movies_master = pd.read_parquet(args.movies_master)
    gru_model, gru_movie_to_idx, gru_idx_to_movie = load_gru_artifacts(args.gru_model, args.gru_mappings, device=device)

    ratings_val_warm = pd.read_parquet(args.ratings_val_warm)
    ratings_train = pd.read_parquet(args.ratings_train, columns=["userId", "movieId", "rating", "timestamp"])

    relevant_users = set(ratings_val_warm["userId"].unique())
    ratings_train_relevant = ratings_train[ratings_train["userId"].isin(relevant_users)]

    print("building labeled examples...")
    all_movie_ids = movies_master["movieId"].tolist()
    examples = build_labeled_examples(ratings_val_warm, all_movie_ids)
    print("examples:", len(examples))

    user_train_sequences = build_user_train_sequences(ratings_train_relevant, gru_movie_to_idx)
    ratings_train_by_user = (
        ratings_train_relevant[ratings_train_relevant["rating"] >= 4.0]
        .groupby("userId")["movieId"].apply(list).to_dict()
    )

    print("scoring examples with all 3 signals...")
    scored_examples = score_all_signals(
        examples, svd_model, tfidf_matrix, movies_master, gru_model, gru_movie_to_idx,
        user_train_sequences, ratings_train_by_user, device=device,
    )

    finite = scored_examples.loc[~np.isinf(scored_examples["gru_score"]), "gru_score"]
    gru_sentinel_value = finite.min()

    print("training meta-model...")
    meta_model, scaler, metrics = train_meta_model(scored_examples)
    print(json.dumps(metrics, indent=2))

    save_meta_model(meta_model, scaler, gru_sentinel_value, args.out_model, args.out_scaler, args.out_sentinel)

    results = {**metrics, "gru_sentinel_value": round(float(gru_sentinel_value), 4),
               "training_data_size": len(scored_examples), "training_data_users": int(scored_examples["userId"].nunique())}
    with open(args.out_results, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved: {args.out_model}, {args.out_scaler}, {args.out_sentinel}, {args.out_results}")
