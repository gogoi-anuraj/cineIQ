"""
test_join.py

Tests join.py's content_text construction and RT fuzzy-join tie-breaking
using small synthetic data. Does not require real TMDB/RT files.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.join import normalize_title, add_content_text, join_rt_reviews


def test_normalize_title_strips_year_punctuation_article():
    assert normalize_title("Toy Story (1995)") == "toy story"
    assert normalize_title("The Matrix") == "matrix"
    assert normalize_title("A Beautiful Mind") == "beautiful mind"
    assert normalize_title("12 Angry Men (Twelve Angry Men)") == "12 angry men twelve angry men"


def test_normalize_title_handles_null():
    assert normalize_title(None) == ""
    assert normalize_title(float("nan")) == ""


def test_content_text_uses_tmdb_when_available():
    movies_master = pd.DataFrame({
        "movieId": [1],
        "genres": ["Comedy|Romance"],
        "genre_list": [["Comedy", "Romance"]],
        "keyword_list": [["wedding"]],
        "cast_list": [["Tom Hanks"]],
    })
    out = add_content_text(movies_master)
    assert out["content_text"].iloc[0] == "Comedy Romance wedding Tom_Hanks"


def test_content_text_falls_back_to_ml_genres_when_tmdb_empty():
    movies_master = pd.DataFrame({
        "movieId": [1],
        "genres": ["Comedy|Drama"],
        "genre_list": [[]],
        "keyword_list": [[]],
        "cast_list": [[]],
    })
    out = add_content_text(movies_master)
    assert out["content_text"].iloc[0] == "Comedy Drama"


def test_content_text_empty_when_no_signal_anywhere():
    movies_master = pd.DataFrame({
        "movieId": [1],
        "genres": ["(no genres listed)"],
        "genre_list": [[]],
        "keyword_list": [[]],
        "cast_list": [[]],
    })
    out = add_content_text(movies_master)
    assert out["content_text"].iloc[0] == ""


def test_join_rt_reviews_prefers_higher_review_count_on_tie():
    """Two RT candidates match the same movieId with identical year_diff —
    the join should deterministically pick the one with more reviews,
    not whichever pandas happens to return first."""
    movies_master = pd.DataFrame({
        "movieId": [1],
        "title": ["Hamlet (2000)"],
        "year": [2000],
    })
    rt_movies = pd.DataFrame({
        "rotten_tomatoes_link": ["m/hamlet_a", "m/hamlet_b"],
        "movie_title": ["Hamlet", "Hamlet"],
        "year": [2000, 2000],
        "tomatometer_count": [12, 87],
    })
    result = join_rt_reviews(movies_master, rt_movies)
    assert len(result) == 1
    assert result["rotten_tomatoes_link"].iloc[0] == "m/hamlet_b"


def test_join_rt_reviews_respects_year_tolerance():
    movies_master = pd.DataFrame({
        "movieId": [1, 2],
        "title": ["Movie A (2000)", "Movie B (2000)"],
        "year": [2000, 2000],
    })
    rt_movies = pd.DataFrame({
        "rotten_tomatoes_link": ["m/a_close", "m/b_far"],
        "movie_title": ["Movie A", "Movie B"],
        "year": [2001, 2005],  # A within +/-1, B is not
        "tomatometer_count": [10, 10],
    })
    result = join_rt_reviews(movies_master, rt_movies, year_tolerance=1)
    assert set(result["movieId"]) == {1}


def test_join_rt_reviews_no_match_returns_empty():
    movies_master = pd.DataFrame({
        "movieId": [1],
        "title": ["Totally Obscure Film (1932)"],
        "year": [1932],
    })
    rt_movies = pd.DataFrame({
        "rotten_tomatoes_link": ["m/unrelated"],
        "movie_title": ["Something Else"],
        "year": [2010],
        "tomatometer_count": [5],
    })
    result = join_rt_reviews(movies_master, rt_movies)
    assert result.empty
