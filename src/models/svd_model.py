# src/models/svd_model.py — Part 1 (local dev subsample)

import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD
from collections import defaultdict
from surprise import accuracy
import pickle
import pandas as pd

def load_ratings_for_surprise(ratings_train_path: str, warm_users: set = None,
                               subsample_n_users: int = None, random_state: int = 42,
                               rating_scale=(0.5, 5.0)):
    """
    subsample_n_users: if set, randomly samples this many users (from
    warm_users if provided) for fast local iteration. Leave None for
    the real full training run (intended to run on Kaggle given local
    RAM constraints).
    """
    ratings_train = pd.read_parquet(ratings_train_path)

    if warm_users is not None:
        ratings_train = ratings_train[ratings_train["userId"].isin(warm_users)]

    if subsample_n_users is not None:
        all_users = ratings_train["userId"].unique()
        rng = pd.Series(all_users).sample(n=subsample_n_users, random_state=random_state)
        ratings_train = ratings_train[ratings_train["userId"].isin(set(rng))]
        print(f"dev subsample: {subsample_n_users} users -> {len(ratings_train):,} rows")

    df_for_surprise = ratings_train[["userId", "movieId", "rating"]]
    reader = Reader(rating_scale=rating_scale)
    dataset = Dataset.load_from_df(df_for_surprise, reader)

    return dataset, ratings_train

def train_svd(trainset, n_factors=100, n_epochs=20, random_state=42):
    """
    Trains Surprise's SVD on a trainset. Defaults are Surprise's
    standard starting point (100 latent factors, 20 epochs) — we'll
    tune these later once we have a validation metric to tune against.
    """
    model = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=random_state)
    model.fit(trainset)
    return model

def evaluate_rmse(model, testset):
    """
    testset: list of (uid, iid, true_rating) tuples — Surprise's format.
    Returns RMSE over the testset.
    """
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=True)
    return rmse, predictions


def precision_at_k(predictions, k=10, threshold=3.5):
    """
    Precision@k: of the top-k items we'd recommend to each user (ranked
    by predicted rating), what fraction are actually 'relevant' (true
    rating >= threshold)?

    Standard Surprise FAQ implementation pattern — groups predictions
    by user, sorts by estimated rating, checks the top k.
    """
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = {}
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        n_relevant_and_recommended = sum(
            (true_r >= threshold) for (_, true_r) in top_k
        )
        precisions[uid] = n_relevant_and_recommended / len(top_k) if top_k else 0

    return sum(precisions.values()) / len(precisions)

def save_svd_model(model, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"saved SVD model to {path}")


def load_svd_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def score_batch(model, user_item_pairs: pd.DataFrame,
                 user_col: str = "userId", item_col: str = "movieId") -> pd.DataFrame:
    """
    Scores many (user, item) pairs at once — this is what the meta-model
    (Week 3) will call to get the SVD feature column for its training
    data. Returns the input df with an added 'svd_score' column.

    Note: Surprise has no true vectorized batch-predict — under the hood
    this still loops .predict() per row, but wrapping it here keeps that
    detail out of meta-model code and gives us one place to optimize
    later if it becomes a bottleneck at full scale.
    """
    out = user_item_pairs.copy()
    out["svd_score"] = out.apply(
        lambda row: model.predict(row[user_col], row[item_col]).est, axis=1
    )
    return out


if __name__ == "__main__":
    warm_users_df = pd.read_parquet("data/interim/warm_users.parquet")
    warm_users = set(warm_users_df["userId"])

    # LOCAL DEV MODE: small subsample so this runs fast on 8GB RAM
    dataset, ratings_train = load_ratings_for_surprise(
        "data/interim/ratings_train.parquet",
        warm_users=warm_users,
        subsample_n_users=5000,
    )

    print("unique users:", ratings_train["userId"].nunique())
    print("unique movies:", ratings_train["movieId"].nunique())

    trainset = dataset.build_full_trainset()
    print("\nSurprise trainset n_users:", trainset.n_users)
    print("Surprise trainset n_items:", trainset.n_items)
    print("Surprise trainset n_ratings:", trainset.n_ratings)

    print("training SVD...")
    model = train_svd(trainset)
    print("done.")

    # Sanity check: predict a rating for a real user-item pair that exists in training data
    sample_row = ratings_train.iloc[0]
    uid, iid, true_r = sample_row["userId"], sample_row["movieId"], sample_row["rating"]

    pred = model.predict(uid, iid, r_ui=true_r)
    print(f"\nsample prediction for userId={uid}, movieId={iid}:")
    print(f"  true rating: {true_r}")
    print(f"  predicted:   {pred.est:.3f}")
    print(f"  full pred object: {pred}")

    # Predict for a user-item pair that was NOT in training (movie the user never rated)
    unseen_movie = ratings_train[ratings_train["movieId"] != iid]["movieId"].iloc[0]
    pred_unseen = model.predict(uid, unseen_movie)
    print(f"\nprediction for an unrated movie (movieId={unseen_movie}):")
    print(f"  predicted: {pred_unseen.est:.3f}")
    print(f"  details:   {pred_unseen}")

    val_warm = pd.read_parquet("data/interim/ratings_val_warm.parquet")

    # For this dev run, restrict val to just the 5000 subsampled users so
    # the model has actually seen these users during training
    dev_user_ids = set(ratings_train["userId"].unique())
    val_dev = val_warm[val_warm["userId"].isin(dev_user_ids)]
    print("val_dev rows (subsample users only):", len(val_dev))

    testset = list(zip(val_dev["userId"], val_dev["movieId"], val_dev["rating"]))

    rmse, predictions = evaluate_rmse(model, testset)

    prec_at_10 = precision_at_k(predictions, k=10, threshold=3.5)
    print(f"\nPrecision@10 (threshold=3.5): {prec_at_10:.4f}")


    save_svd_model(model, "data/interim/svd_model_dev.pkl")

    reloaded = load_svd_model("data/interim/svd_model_dev.pkl")

    # Quick check: batch-score a handful of pairs from val_dev, confirm
    # reloaded model gives identical results to the in-memory one
    sample_pairs = val_dev[["userId", "movieId"]].head(5)
    scored = score_batch(reloaded, sample_pairs)
    print(scored)