"""
split.py

Single source of truth for CINEIQ+'s global temporal split and the
warm-user / cold-start definition. Every component (SVD, content-based,
GRU, meta-model, cold-start router) should import from here rather than
re-deriving cutoff dates or thresholds independently.

Values (train_cutoff, val_cutoff, cold_start_threshold_N, etc.) are read
from week1_config.json, produced during Week 1 data prep, so this module
has no hardcoded dates/thresholds of its own.
"""

import json
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np


DEFAULT_CONFIG_PATH = Path("data/interim/week1_config.json")


@dataclass
class SplitConfig:
    train_cutoff: pd.Timestamp
    val_cutoff: pd.Timestamp
    cold_start_threshold_n: int

    @classmethod
    def from_json(cls, path: Path = DEFAULT_CONFIG_PATH) -> "SplitConfig":
        with open(path, "r") as f:
            cfg = json.load(f)
        return cls(
            train_cutoff=pd.Timestamp(cfg["train_cutoff"]),
            val_cutoff=pd.Timestamp(cfg["val_cutoff"]),
            cold_start_threshold_n=int(cfg["cold_start_threshold_N"]),
        )


def assign_split(ratings: pd.DataFrame, config: SplitConfig,
                  timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Adds a 'split' column ('train' / 'val' / 'test') to a ratings dataframe
    using the single global temporal cutoff. `timestamp_col` is assumed to
    be unix epoch seconds (MovieLens format); converted internally.

    This is the ONE place split boundaries are applied — never mix random
    splits for some components with temporal splits for others.
    """
    out = ratings.copy()
    dt = pd.to_datetime(out[timestamp_col], unit="s")
    out["split"] = np.select(
        [dt < config.train_cutoff, dt < config.val_cutoff],
        ["train", "val"],
        default="test",
    )
    return out


def get_warm_users(ratings_train: pd.DataFrame, config: SplitConfig,
                    user_col: str = "userId") -> set:
    """
    Returns the set of userIds with >= cold_start_threshold_n ratings in
    the train split. These are the users SVD/GRU/meta-model can be fairly
    evaluated and served on; everyone else routes through cold-start
    fallback logic (see serving/cold_start.py, once written).
    """
    counts = ratings_train.groupby(user_col).size()
    return set(counts[counts >= config.cold_start_threshold_n].index)


def is_cold_start_user(user_train_rating_count: int, config: SplitConfig) -> bool:
    """
    Single-user check for use at serving time: given how many train-period
    ratings a user has, should they be routed around the meta-model?
    """
    return user_train_rating_count < config.cold_start_threshold_n


def filter_to_warm_users(ratings: pd.DataFrame, warm_users: set,
                          user_col: str = "userId") -> pd.DataFrame:
    """Filters any split (val/test) down to ratings from warm users only —
    the fair evaluation subset for SVD/GRU/meta-model metrics."""
    return ratings[ratings[user_col].isin(warm_users)]


def build_user_sequences(ratings_train: pd.DataFrame, user_col: str = "userId",
                          item_col: str = "movieId",
                          timestamp_col: str = "timestamp") -> pd.Series:
    """
    Groups train-split ratings into chronological per-user sequences of
    movieIds — the input format the GRU (Section 5.3) needs. Returns a
    Series indexed by userId, values are ordered lists of movieId.
    """
    sorted_ratings = ratings_train.sort_values([user_col, timestamp_col])
    return sorted_ratings.groupby(user_col)[item_col].apply(list)


def split_summary(ratings_with_split: pd.DataFrame) -> pd.DataFrame:
    """Quick sanity-check table: row counts and % share per split."""
    counts = ratings_with_split["split"].value_counts()
    pct = (counts / len(ratings_with_split) * 100).round(1)
    return pd.DataFrame({"count": counts, "pct": pct})
