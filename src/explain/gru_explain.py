import pandas as pd

def explain_gru_recommendation(user_sequence: list, idx_to_movie: dict, movieid_to_title: dict,
                                n_recent: int = 3) -> str:
    """
    Rule-based explanation for a GRU-driven recommendation (spec 5.6):
    "Recommended because you recently watched X, Y, Z" -- built from the
    N most recent items in the user's sequence, decoded back to titles.

    user_sequence: the user's ENCODED sequence (list of movie indices),
    same format used for GRU inference throughout the project.
    """
    if not user_sequence:
        return "Recommended based on general viewing patterns."

    recent_idxs = user_sequence[-n_recent:]
    recent_titles = []
    for idx in reversed(recent_idxs):  # most recent first
        movie_id = idx_to_movie.get(idx)
        title = movieid_to_title.get(movie_id, None) if movie_id else None
        if title:
            recent_titles.append(title)

    if not recent_titles:
        return "Recommended based on your recent viewing sequence."

    if len(recent_titles) == 1:
        return f"Recommended because you recently watched {recent_titles[0]}."
    elif len(recent_titles) == 2:
        return f"Recommended because you recently watched {recent_titles[0]} and {recent_titles[1]}."
    else:
        listed = ", ".join(recent_titles[:-1]) + f", and {recent_titles[-1]}"
        return f"Recommended because you recently watched {listed}."


if __name__ == "__main__":
    import pickle

    movies_master = pd.read_parquet("data/interim/movies_master.parquet")
    movieid_to_title = dict(zip(movies_master["movieId"], movies_master["title"]))

    with open("data/interim/gru_mappings.pkl", "rb") as f:
        gru_meta = pickle.load(f)
    idx_to_movie = gru_meta["idx_to_movie"]

    # Test with a few sequence lengths, using real encoded sequences if
    # available, otherwise a small hand-built example for a shape check
    test_sequences = {
        "empty": [],
        "one_movie": [1],  # whatever index 1 maps to
        "three_movies": [1, 2, 3],
    }

    for label, seq in test_sequences.items():
        print(f"--- {label} ---")
        print(explain_gru_recommendation(seq, idx_to_movie, movieid_to_title, n_recent=3))
        print()

    # Test against a real user's actual encoded sequence

    ratings_train = pd.read_parquet("data/interim/ratings_train.parquet",
                                    columns=["userId", "movieId", "timestamp"])
    warm_users = pd.read_parquet("data/interim/warm_users.parquet")["userId"]
    movie_to_idx = gru_meta["movie_to_idx"]

    test_user = warm_users.iloc[0]
    user_ratings = ratings_train[ratings_train["userId"] == test_user].sort_values("timestamp")
    real_sequence = [movie_to_idx[m] for m in user_ratings["movieId"] if m in movie_to_idx]

    print(f"userId={test_user}, sequence length={len(real_sequence)}")
    print(explain_gru_recommendation(real_sequence, idx_to_movie, movieid_to_title, n_recent=3))

    # Also check the tail of their ACTUAL watch history directly, to confirm
    # the explanation matches their true most-recent movies
    print("\nlast 5 movies in their real chronological history:")
    print(user_ratings["movieId"].tail(5).map(movieid_to_title).tolist())