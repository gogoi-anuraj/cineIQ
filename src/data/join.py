"""
join.py

All cross-dataset joins for CINEIQ+:
  1. MovieLens movies <-> TMDB metadata, via links.csv (clean ID join)
  2. content_text construction for TF-IDF, with MovieLens-genre fallback
  3. MovieLens movies <-> RT critic reviews, via normalized title + year
     (fuzzy join — RT has no shared ID system with MovieLens/TMDB)
"""

import re

import pandas as pd


# ---------------------------------------------------------------------------
# 1. MovieLens <-> TMDB
# ---------------------------------------------------------------------------

def build_movies_master(movies: pd.DataFrame, links_clean: pd.DataFrame,
                         tmdb_full: pd.DataFrame) -> pd.DataFrame:
    """
    Joins MovieLens movies -> links.csv -> TMDB metadata.
    Expected coverage (from Week 1 run): ~68.6% of movies matched,
    ~98.9% of RATINGS matched (unmatched movies skew toward obscure,
    low-rating-volume titles).
    """
    movies_with_tmdb_id = movies.merge(
        links_clean[["movieId", "tmdbId", "imdbId"]], on="movieId", how="left"
    )

    movies_joined = movies_with_tmdb_id.merge(
        tmdb_full[["id", "genre_list", "keyword_list", "cast_list",
                   "overview", "vote_average", "vote_count"]],
        left_on="tmdbId", right_on="id", how="left",
    )

    movies_master = movies_joined.drop(columns=["id"]).rename(
        columns={"tmdbId": "tmdb_id", "imdbId": "imdb_id"}
    )

    for col in ["genre_list", "keyword_list", "cast_list"]:
        movies_master[col] = movies_master[col].apply(
            lambda x: x if isinstance(x, list) else []
        )

    return movies_master


def add_content_text(movies_master: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the TF-IDF-ready `content_text` field per spec 5.2: TMDB
    genre + keyword + top-billed cast, falling back to MovieLens' own
    pipe-separated `genres` column when TMDB gave nothing usable.

    Expected result (Week 1 run): empty rate drops from 32.3% (TMDB-only)
    to 4.5% after the fallback. That remaining 4.5% has no genre/keyword/
    cast info anywhere and is a documented limitation, not a bug.
    """
    out = movies_master.copy()

    out["ml_genre_list"] = out["genres"].apply(
        lambda x: [] if x == "(no genres listed)" else x.split("|")
    )

    def _build(row):
        parts = row["genre_list"] + row["keyword_list"] + row["cast_list"]
        if not parts:
            parts = row["ml_genre_list"]
        return " ".join(str(p).replace(" ", "_") for p in parts)

    out["content_text"] = out.apply(_build, axis=1)
    return out


# ---------------------------------------------------------------------------
# 2. MovieLens <-> RT critic reviews (fuzzy join)
# ---------------------------------------------------------------------------

def normalize_title(title) -> str:
    """Lowercase, strip trailing '(YYYY)', strip punctuation, strip a
    leading article ('the'/'a'/'an'), collapse whitespace."""
    if pd.isna(title):
        return ""
    t = title.lower()
    t = re.sub(r"\(\d{4}\)\s*$", "", t)
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"^(the|a|an)\s+", "", t.strip())
    t = re.sub(r"\s+", " ", t).strip()
    return t


def join_rt_reviews(movies_master: pd.DataFrame, rt_movies: pd.DataFrame,
                     year_tolerance: int = 1) -> pd.DataFrame:
    """
    Fuzzy-joins movies_master to RT movies on (normalized_title, year),
    allowing +/- `year_tolerance` years. Where multiple RT candidates tie
    on year_diff for the same movieId, breaks the tie deterministically
    by preferring the RT entry with the higher `tomatometer_count`
    (more reviews = more reliable match), rather than arbitrary row
    order — per the review-flagged nondeterminism in the original
    notebook version.

    Returns a movieId -> rotten_tomatoes_link mapping (one row per
    matched movieId). Expected coverage (Week 1 run): ~17.6% of movies,
    ~66.5% of ratings.
    """
    m = movies_master[["movieId", "title", "year"]].copy()
    m["title_norm"] = m["title"].apply(normalize_title)

    rt = rt_movies.copy()
    rt["title_norm"] = rt["movie_title"].apply(normalize_title)

    candidates = m.merge(
        rt[["rotten_tomatoes_link", "title_norm", "year", "tomatometer_count"]],
        on="title_norm", how="inner", suffixes=("_ml", "_rt"),
    )
    candidates["year_diff"] = (candidates["year_ml"] - candidates["year_rt"]).abs()
    candidates = candidates[candidates["year_diff"] <= year_tolerance]

    # Deterministic tie-break: smallest year_diff first, then highest
    # tomatometer_count (more reviews = more trustworthy match)
    candidates = candidates.sort_values(
        ["year_diff", "tomatometer_count"], ascending=[True, False]
    )
    candidates = candidates.drop_duplicates(subset="movieId", keep="first")

    return candidates[["movieId", "rotten_tomatoes_link"]].drop_duplicates(
        subset="movieId"
    )


def attach_rt_flag(movies_master: pd.DataFrame,
                    movie_to_rt: pd.DataFrame) -> pd.DataFrame:
    """Adds a boolean `has_rt` column to movies_master — convenience for
    the decade/genre skew check and for filtering to RT-covered movies
    when building the sentiment re-ranker's candidate set."""
    out = movies_master.copy()
    out["has_rt"] = out["movieId"].isin(movie_to_rt["movieId"])
    return out
