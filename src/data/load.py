"""
load.py

Raw dataset loaders for CINEIQ+'s four data sources. Each function returns
a cleaned-but-unjoined dataframe (dedup'd, list columns parsed where
applicable). Joining across datasets lives in join.py, not here.
"""

import ast
import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# MovieLens 25M
# ---------------------------------------------------------------------------

def load_movielens(path: str):
    """Loads ratings.csv, movies.csv, links.csv from a MovieLens 25M dir.
    Adds `year` (parsed from title) to movies, and drops the ~0.17% of
    links rows with a null tmdbId (cast to int on the rest)."""
    path = Path(path)
    ratings = pd.read_csv(path / "ratings.csv")
    movies = pd.read_csv(path / "movies.csv")
    links = pd.read_csv(path / "links.csv")

    movies = movies.copy()
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$").astype("Int64")

    links_clean = links.dropna(subset=["tmdbId"]).copy()
    links_clean["tmdbId"] = links_clean["tmdbId"].astype(int)

    return ratings, movies, links_clean


# ---------------------------------------------------------------------------
# TMDB metadata (+ keywords, credits)
# ---------------------------------------------------------------------------

def _parse_names(x, key: str = "name", top_n: int | None = None):
    """Safely parses a stringified list-of-dicts column (TMDB's genres/
    keywords/cast format) into a plain list of names. Returns [] on any
    parse failure or missing value."""
    if pd.isna(x):
        return []
    try:
        items = ast.literal_eval(x)
        names = [d[key] for d in items]
        return names[:top_n] if top_n else names
    except (ValueError, SyntaxError):
        return []


def load_tmdb(path: str, top_n_cast: int = 5) -> pd.DataFrame:
    """
    Loads movies_metadata.csv, keywords.csv, credits.csv from "The Movies
    Dataset" dir. Drops the handful of corrupted `id` rows (non-numeric),
    dedupes all three tables on id (this dataset ships ~1-2% true
    duplicate rows), merges, and parses genres/keywords/cast into clean
    list columns plus a combined `content_text` field ready for TF-IDF.
    """
    path = Path(path)
    tmdb = pd.read_csv(path / "movies_metadata.csv", low_memory=False)
    keywords = pd.read_csv(path / "keywords.csv")
    credits = pd.read_csv(path / "credits.csv")

    non_numeric_id = ~tmdb["id"].astype(str).str.match(r"^\d+$")
    tmdb_clean = tmdb[~non_numeric_id].copy()
    tmdb_clean["id"] = tmdb_clean["id"].astype(int)

    tmdb_clean = tmdb_clean.drop_duplicates(subset="id", keep="first")
    keywords = keywords.drop_duplicates(subset="id", keep="first")
    credits = credits.drop_duplicates(subset="id", keep="first")

    tmdb_full = tmdb_clean.merge(keywords, on="id", how="left")
    tmdb_full = tmdb_full.merge(credits[["id", "cast"]], on="id", how="left")

    tmdb_full["genre_list"] = tmdb_full["genres"].apply(_parse_names)
    tmdb_full["keyword_list"] = tmdb_full["keywords"].apply(_parse_names)
    tmdb_full["cast_list"] = tmdb_full["cast"].apply(
        lambda x: _parse_names(x, top_n=top_n_cast)
    )

    return tmdb_full


# ---------------------------------------------------------------------------
# IMDB 50K Reviews (sentiment classifier training data)
# ---------------------------------------------------------------------------

def _clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_imdb_reviews(path: str) -> pd.DataFrame:
    """
    Loads IMDB Dataset.csv, dedupes exact-duplicate reviews (~0.8% of this
    dataset), strips HTML (mainly <br /> tags), and returns
    [review_clean, sentiment]. Used ONLY to train the sentiment classifier
    — not linked to specific movies.
    """
    path = Path(path)
    imdb = pd.read_csv(path / "IMDB Dataset.csv")
    imdb_clean = imdb.drop_duplicates(subset="review", keep="first").copy()
    imdb_clean["review_clean"] = imdb_clean["review"].apply(_clean_html)
    return imdb_clean[["review_clean", "sentiment"]]


# ---------------------------------------------------------------------------
# RT critic reviews (movie-linked review dataset)
# ---------------------------------------------------------------------------

def load_rt(path: str):
    """Loads rotten_tomatoes_movies.csv and rotten_tomatoes_movie_reviews.csv.
    Adds a parsed `year` column to rt_movies from original_release_date."""
    path = Path(path)
    rt_movies = pd.read_csv(path / "rotten_tomatoes_movies.csv")
    rt_reviews = pd.read_csv(path / "rotten_tomatoes_movie_reviews.csv")

    rt_movies = rt_movies.copy()
    rt_movies["year"] = pd.to_datetime(
        rt_movies["original_release_date"], errors="coerce"
    ).dt.year

    return rt_movies, rt_reviews
