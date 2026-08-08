import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_movies_master(path: str) -> pd.DataFrame:
    movies_master = pd.read_parquet(path)
    return movies_master

def build_tfidf_matrix(movies_master: pd.DataFrame, min_df=2, max_features=None):
    """
    Vectorizes content_text with TF-IDF. min_df=2 drops terms appearing
    in only 1 movie (mostly noise — misspelled names, one-off keywords)
    since our documents are already short (median 7 words).
    No max_df cap for now — with short documents, even "common" terms
    like top genre words (Comedy, Drama) still carry real signal.
    """
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_features=max_features,
        token_pattern=r"(?u)\b\w[\w'-]*\b",  # keep underscored multi-word tokens intact
    )
    tfidf_matrix = vectorizer.fit_transform(movies_master["content_text"].fillna(""))
    return vectorizer, tfidf_matrix


# def get_similar_movies(movie_id: int, movies_master: pd.DataFrame, tfidf_matrix,
#                         top_n: int = 10):
#     """
#     Returns the top_n most content-similar movies to the given movie_id,
#     by cosine similarity over TF-IDF vectors. Computed on demand (one row
#     vs. the full matrix) rather than precomputing a full pairwise matrix,
#     which would be prohibitively large at ~62K movies.
#     """
#     idx_lookup = movies_master.index[movies_master["movieId"] == movie_id]
#     if len(idx_lookup) == 0:
#         raise ValueError(f"movieId {movie_id} not found in movies_master")
#     idx = idx_lookup[0]

#     sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

#     # Exclude the movie itself, take top_n by similarity score
#     sims[idx] = -1
#     top_idxs = sims.argsort()[::-1][:top_n]

#     result = movies_master.iloc[top_idxs][["movieId", "title"]].copy()
#     result["similarity"] = sims[top_idxs]
#     return result.reset_index(drop=True)

# def get_similar_movies(movie_id: int, movies_master: pd.DataFrame, tfidf_matrix,
#                         top_n: int = 10, min_content_tokens: int = 5):
#     """
#     Returns the top_n most content-similar movies to the given movie_id.
#     Candidates with fewer than min_content_tokens words in content_text
#     are excluded — short/sparse metadata produces unreliable similarity
#     scores that can be dominated by a single coincidentally-shared term
#     (e.g. a minor shared cast member), rather than genuine topical overlap.
#     """
#     idx_lookup = movies_master.index[movies_master["movieId"] == movie_id]
#     if len(idx_lookup) == 0:
#         raise ValueError(f"movieId {movie_id} not found in movies_master")
#     idx = idx_lookup[0]

#     sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
#     sims[idx] = -1  # exclude the movie itself

#     # Mask out low-signal candidates
#     token_counts = movies_master["content_text"].str.split().str.len().fillna(0).values
#     sims[token_counts < min_content_tokens] = -1

#     top_idxs = sims.argsort()[::-1][:top_n]
#     result = movies_master.iloc[top_idxs][["movieId", "title"]].copy()
#     result["similarity"] = sims[top_idxs]
#     return result.reset_index(drop=True)




def get_similar_movies(movie_id: int, movies_master: pd.DataFrame, tfidf_matrix,
                        top_n: int = 10, min_content_tokens: int = 6):
    """ 
    Returns top_n content-similar movies to movie_id, by cosine similarity
    over TF-IDF vectors. Candidates with fewer than min_content_tokens
    words in content_text are excluded.
    
    """
    idx_lookup = movies_master.index[movies_master["movieId"] == movie_id]
    if len(idx_lookup) == 0:
        raise ValueError(f"movieId {movie_id} not found in movies_master")
    idx = idx_lookup[0]

    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sims[idx] = -1

    token_counts = movies_master["content_text"].str.split().str.len().fillna(0).values
    sims[token_counts < min_content_tokens] = -1

    top_idxs = sims.argsort()[::-1][:top_n]
    result = movies_master.iloc[top_idxs][["movieId", "title"]].copy()
    result["similarity"] = sims[top_idxs]
    return result.reset_index(drop=True)


def score_candidates_for_user(user_liked_movie_ids: list, candidate_movie_ids: list,
                               movies_master: pd.DataFrame, tfidf_matrix,
                               min_content_tokens: int = 6) -> pd.DataFrame:
    """
    For a user's liked movies (e.g. rating >= 4.0 in train split), scores
    each candidate movie by average cosine similarity to the user's
    liked-movie set. This is the content-based signal the meta-model
    consumes as one of its 3 input features (spec 5.2/5.4).
 
    Both the user's liked-set and the candidates are filtered by the same
    min_content_tokens safeguard as get_similar_movies — a liked movie
    with too little content signal doesn't meaningfully inform the
    user's taste profile, and a candidate with too little content signal
    can't be scored reliably.
    """
    id_to_idx = {mid: i for i, mid in enumerate(movies_master["movieId"])}
    token_counts = movies_master["content_text"].str.split().str.len().fillna(0).values
 
    liked_idxs = [id_to_idx[m] for m in user_liked_movie_ids if m in id_to_idx]
    liked_idxs = [i for i in liked_idxs if token_counts[i] >= min_content_tokens]
 
    if not liked_idxs:
        # User has no likes with sufficient content signal -> no content score available.
        # Downstream (meta-model / cold-start routing) should treat this as "missing",
        # not as a genuine low-similarity signal.
        return pd.DataFrame({"movieId": candidate_movie_ids, "content_score": 0.0})
 
    valid_candidates = [m for m in candidate_movie_ids if m in id_to_idx]
    candidate_idxs = [id_to_idx[m] for m in valid_candidates]
 
    sim_matrix = cosine_similarity(tfidf_matrix[candidate_idxs], tfidf_matrix[liked_idxs])
    avg_sims = sim_matrix.mean(axis=1)
 
    # Candidates below the content-token floor get a 0 score (unreliable, not "dissimilar")
    candidate_token_counts = token_counts[candidate_idxs]
    avg_sims = np.where(candidate_token_counts < min_content_tokens, 0.0, avg_sims)
 
    return pd.DataFrame({"movieId": valid_candidates, "content_score": avg_sims})


# Add to content_model.py — saving

import pickle
from scipy import sparse

def save_content_artifacts(vectorizer, tfidf_matrix, vectorizer_path: str, matrix_path: str):
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    sparse.save_npz(matrix_path, tfidf_matrix)  # scipy sparse matrices need their own save format, not pickle
    print(f"saved vectorizer to {vectorizer_path}")
    print(f"saved tfidf_matrix to {matrix_path}")


def load_content_artifacts(vectorizer_path: str, matrix_path: str):
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    tfidf_matrix = sparse.load_npz(matrix_path)
    return vectorizer, tfidf_matrix


if __name__ == "__main__":
    movies_master = load_movies_master("data/interim/movies_master.parquet")
    vectorizer, tfidf_matrix = build_tfidf_matrix(movies_master)
 
    print("movies_master shape:", movies_master.shape)
    print("tfidf_matrix shape:", tfidf_matrix.shape)
    print("vocabulary size:", len(vectorizer.vocabulary_))
 
    print("\n--- similar movies to Toy Story (movieId=1) ---")
    similar = get_similar_movies(1, movies_master, tfidf_matrix, top_n=10)
    print(similar)
 
    print("\n--- per-user candidate scoring ---")
    liked = [1, 3114, 6377]  # Toy Story, Toy Story 2, Finding Nemo
    candidates = [78499, 2797, 1198, 5349]  # Toy Story 3, Big, Raiders of Lost Ark, Spider-Man
 
    scores = score_candidates_for_user(liked, candidates, movies_master, tfidf_matrix)
    scores = scores.merge(movies_master[["movieId", "title"]], on="movieId")
    print(scores.sort_values("content_score", ascending=False))
 
    token_counts = movies_master["content_text"].str.split().str.len().fillna(0)
    excluded = (token_counts < 6).sum()
    print(f"\n[known limitation] movies excluded as similarity sources "
          f"(< {6} tokens): {excluded} "
          f"({100*excluded/len(movies_master):.1f}%)")
    
    save_content_artifacts(
        vectorizer, tfidf_matrix,
        "data/interim/tfidf_vectorizer.pkl",
        "data/interim/tfidf_matrix.npz",
    )