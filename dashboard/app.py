import streamlit as st
import pandas as pd
import numpy as np
# import plotly.graph_objects as go
import pickle
import torch
import torch.nn as nn
from scipy import sparse
import sys
import json
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append("src")

from data_prep import (
    load_dashboard_data, get_user_genre_profile, build_genre_radar_chart,
    get_user_decade_profile, build_decade_bar_chart,
    get_user_cast_affinity, get_user_director_affinity, build_affinity_bar_chart,
    get_user_taste_trajectory, build_trajectory_chart,
)

st.set_page_config(page_title="CINEIQ", layout="wide")
st.title("🎬 CINEIQ")
st.caption("Built from validation-split ratings only, per project evaluation discipline.")

@st.cache_data
def load_data():
    movies_master, ratings_val = load_dashboard_data()
    movies_master_with_director = pd.read_parquet("data/interim/movies_master_with_director.parquet")
    return movies_master, movies_master_with_director, ratings_val


movies_master, movies_master_with_director, ratings_val = load_data()

class GRU4RecTied(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.output_dropout = nn.Dropout(dropout)
        self.projection = nn.Identity() if hidden_dim == embedding_dim else nn.Linear(hidden_dim, embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

    def embed_sequence(self, input_seqs, lengths):
        embedded = self.embedding_dropout(self.embedding(input_seqs))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.projection(self.output_dropout(hidden[-1]))

    def score_full_vocab(self, projected):
        return projected @ self.embedding.weight.T + self.output_bias

    def forward(self, input_seqs, lengths):
        return self.score_full_vocab(self.embed_sequence(input_seqs, lengths))


@st.cache_resource
def load_models():
    with open("data/interim/svd/svd_model.pkl", "rb") as f:
        svd_model = pickle.load(f)

    with open("data/interim/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    tfidf_matrix = sparse.load_npz("data/interim/tfidf_matrix.npz")

    with open("data/interim/gru_mappings.pkl", "rb") as f:
        gru_meta = pickle.load(f)
    gru_movie_to_idx = gru_meta["movie_to_idx"]
    gru_idx_to_movie = gru_meta["idx_to_movie"]
    gru_model = GRU4RecTied(vocab_size=len(gru_movie_to_idx) + 1)
    gru_model.load_state_dict(torch.load("data/interim/gru_model_final.pt", map_location="cpu"))
    gru_model.eval()

    with open("data/interim/meta_model.pkl", "rb") as f:
        meta_model = pickle.load(f)
    with open("data/interim/meta_model_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("data/interim/meta_model_results.json", "r") as f:
        meta_results = json.load(f)
    gru_sentinel_value = meta_results.get("gru_sentinel_value", -9.0396)

    movie_sentiment = pd.read_parquet("data/interim/movie_sentiment.parquet")

    ratings_train = pd.read_parquet("data/interim/ratings_train.parquet",
                                     columns=["userId", "movieId", "rating", "timestamp"])
    popularity_ranking = ratings_train["movieId"].value_counts().index.tolist()

    return {
        "svd_model": svd_model, "vectorizer": vectorizer, "tfidf_matrix": tfidf_matrix,
        "gru_model": gru_model, "gru_movie_to_idx": gru_movie_to_idx, "gru_idx_to_movie": gru_idx_to_movie,
        "meta_model": meta_model, "scaler": scaler, "gru_sentinel_value": gru_sentinel_value,
        "movie_sentiment": movie_sentiment, "ratings_train": ratings_train,
        "popularity_ranking": popularity_ranking,
    }


def score_candidates_content_fast(user_liked_movie_ids, candidate_movie_ids, movies_master, tfidf_matrix, min_content_tokens=6):
    id_to_idx = {mid: i for i, mid in enumerate(movies_master["movieId"])}
    token_counts = movies_master["content_text"].str.split().str.len().fillna(0).values
    liked_idxs = [id_to_idx[m] for m in user_liked_movie_ids if m in id_to_idx]
    liked_idxs = [i for i in liked_idxs if token_counts[i] >= min_content_tokens]
    if not liked_idxs:
        return pd.DataFrame({"movieId": candidate_movie_ids, "content_score": 0.0})
    valid_candidates = [m for m in candidate_movie_ids if m in id_to_idx]
    candidate_idxs = [id_to_idx[m] for m in valid_candidates]
    sim_matrix = cosine_similarity(tfidf_matrix[candidate_idxs], tfidf_matrix[liked_idxs])
    avg_sims = sim_matrix.mean(axis=1)
    candidate_token_counts = token_counts[candidate_idxs]
    avg_sims = np.where(candidate_token_counts < min_content_tokens, 0.0, avg_sims)
    return pd.DataFrame({"movieId": valid_candidates, "content_score": avg_sims})


def score_candidates_gru_fast(model, user_sequence, candidate_movie_ids, movie_to_idx):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([user_sequence], dtype=torch.long)
        length_tensor = torch.tensor([len(user_sequence)])
        logits = model(input_tensor, length_tensor)[0]
    scores = [logits[movie_to_idx[m]].item() if m in movie_to_idx else float("-inf") for m in candidate_movie_ids]
    return pd.DataFrame({"movieId": candidate_movie_ids, "gru_score": scores})


def generate_dashboard_recommendations(user_id, models, movies_master, top_n=5, candidate_pool_size=500):
    """
    Dashboard-speed recommendation generation: restricts candidates to
    the top `candidate_pool_size` most popular movies rather than the
    full ~37K catalog, matching the fast-diagnostic pattern from Week 4
    (full-catalog scoring took ~2s/user, far too slow for an interactive
    UI; the top-500-popular restriction ran in a fraction of that).
    """
    ratings_train = models["ratings_train"]
    user_history = ratings_train[ratings_train["userId"] == user_id]
    already_seen = set(user_history["movieId"])

    candidate_pool = [m for m in models["popularity_ranking"][:candidate_pool_size] if m not in already_seen]

    svd_pairs = pd.DataFrame({"userId": user_id, "movieId": candidate_pool})
    svd_pairs["svd_score"] = svd_pairs.apply(lambda row: models["svd_model"].predict(row["userId"], row["movieId"]).est, axis=1)

    liked_movies = user_history[user_history["rating"] >= 4.0]["movieId"].tolist()
    content_scored = score_candidates_content_fast(liked_movies, candidate_pool, movies_master, models["tfidf_matrix"])

    user_seq_movies = user_history.sort_values("timestamp")["movieId"].tolist()
    user_seq = [models["gru_movie_to_idx"][m] for m in user_seq_movies if m in models["gru_movie_to_idx"]]
    gru_scored = score_candidates_gru_fast(models["gru_model"], user_seq, candidate_pool, models["gru_movie_to_idx"])

    merged = svd_pairs[["movieId", "svd_score"]].merge(
        content_scored[["movieId", "content_score"]], on="movieId", how="inner"
    ).merge(gru_scored[["movieId", "gru_score"]], on="movieId", how="inner")
    merged["gru_score"] = merged["gru_score"].replace(-np.inf, models["gru_sentinel_value"])

    X = models["scaler"].transform(merged[["svd_score", "content_score", "gru_score"]])
    merged["meta_score"] = models["meta_model"].predict_proba(X)[:, 1]

    top_n_result = merged.sort_values("meta_score", ascending=False).head(top_n)
    return top_n_result.merge(movies_master[["movieId", "title"]], on="movieId"), user_seq
# User selector -- restrict to users with a reasonable amount of val-split activity,
# so the dashboard doesn't show an empty/sparse profile
user_counts = ratings_val.groupby("userId").size()
eligible_users = user_counts[user_counts >= 50].index.tolist()

user_id = st.selectbox("Select a user", eligible_users, index=0)
st.write(f"**{user_counts[user_id]}** validation-split ratings for this user")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Genre Preferences")
    genre_profile = get_user_genre_profile(user_id, ratings_val, movies_master)
    if len(genre_profile) > 0:
        fig = build_genre_radar_chart(genre_profile, top_n=8)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough highly-rated movies to build a genre profile.")

with col2:
    st.subheader("Decade Preferences")
    decade_profile = get_user_decade_profile(user_id, ratings_val, movies_master)
    if len(decade_profile) > 0:
        fig = build_decade_bar_chart(decade_profile)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough highly-rated movies to build a decade profile.")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Director & Actor Affinity")
    tab1, tab2 = st.tabs(["Directors", "Actors"])

    with tab1:
        director_affinity = get_user_director_affinity(user_id, ratings_val, movies_master_with_director)
        if len(director_affinity) > 0:
            if director_affinity.max() <= 3:
                st.caption("⚠️ Limited rating history — affinity counts are low and many directors are tied.")
            fig = build_affinity_bar_chart(director_affinity, title="Top Directors")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No director data available for this user's rated movies.")

    with tab2:
        cast_affinity = get_user_cast_affinity(user_id, ratings_val, movies_master)
        if len(cast_affinity) > 0:
            if cast_affinity.max() <= 3:
                st.caption("⚠️ Limited rating history — affinity counts are low and many actors are tied.")
            fig = build_affinity_bar_chart(cast_affinity, title="Top Actors")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cast data available for this user's rated movies.")

with col4:
    st.subheader("Taste Trajectory")
    st.caption("How genre preferences have shifted across this user's rating history")
    trajectory_df = get_user_taste_trajectory(user_id, ratings_val, movies_master, n_periods=5)
    if len(trajectory_df) > 0:
        fig = build_trajectory_chart(trajectory_df, top_n_genres=5)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough rating history to build a taste trajectory.")

st.caption(
    "⚠️ Director data available for ~67.4% of movies (limited by TMDB metadata join coverage). Some users may show incomplete director profiles as a result."
)



st.divider()
st.header("🎯 Recommendations & Why")

models = load_models()

if st.button("Generate Recommendations for This User"):
    with st.spinner("Scoring candidates..."):
        import time
        start = time.time()
        recs, user_seq = generate_dashboard_recommendations(user_id, models, movies_master, top_n=5)
        st.write(f"⏱️ generation took {time.time()-start:.1f}s")

    if len(recs) == 0:
        st.warning("Could not generate recommendations for this user.")
    else:
        # Pull explanation dependencies once
        movieid_to_title = dict(zip(movies_master["movieId"], movies_master["title"]))
        idx_to_movie = models["gru_idx_to_movie"]
        coefficients = json.load(open("data/interim/meta_model_results.json"))["coefficients"]

        for _, row in recs.iterrows():
            with st.container(border=True):
                st.subheader(row["title"])

                col_a, col_b = st.columns([1, 2])

                with col_a:
                    st.metric("Match Score", f"{row['meta_score']*100:.0f}%")

                with col_b:
                    # Meta-model explanation (standardized contribution)
                    X_single = pd.DataFrame(
                        [[row["svd_score"], row["content_score"], row["gru_score"]]],
                        columns=["svd_score", "content_score", "gru_score"]
                    )
                    standardized = models["scaler"].transform(X_single)[0]
                    contributions = {
                        "svd_score": standardized[0] * coefficients["svd_score"],
                        "content_score": standardized[1] * coefficients["content_score"],
                        "gru_score": standardized[2] * coefficients["gru_score"],
                    }
                    total = sum(abs(v) for v in contributions.values())
                    top_signal = max(contributions, key=lambda k: abs(contributions[k]))

                    signal_labels = {
                        "svd_score": "👥 Similar users liked this",
                        "content_score": "🎭 Matches your taste profile",
                        "gru_score": "📺 Fits your recent viewing",
                    }
                    st.write(f"**Primary reason:** {signal_labels[top_signal]}")

                    # GRU-style recent-history explanation, shown as supporting context
                    if user_seq:
                        recent_idxs = user_seq[-3:]
                        recent_titles = [movieid_to_title.get(idx_to_movie.get(i), None) for i in reversed(recent_idxs)]
                        recent_titles = [t for t in recent_titles if t]
                        if recent_titles:
                            st.caption(f"Recently watched: {', '.join(recent_titles)}")

                    # Sentiment context, if available
                    sentiment_row = models["movie_sentiment"][models["movie_sentiment"]["movieId"] == row["movieId"]]
                    if len(sentiment_row) > 0 and sentiment_row.iloc[0]["reliable"]:
                        score = sentiment_row.iloc[0]["sentiment_score"]
                        st.caption(f"Critic sentiment: {score*100:.0f}% positive ({int(sentiment_row.iloc[0]['review_count'])} reviews)")