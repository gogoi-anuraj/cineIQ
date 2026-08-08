"""
test_split.py

Tests split.py's temporal split assignment, warm-user derivation, and
cold-start check using small synthetic data — doesn't require the real
MovieLens files, so these can run in CI / before Week 2 data is even
loaded.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.split import (
    SplitConfig,
    assign_split,
    get_warm_users,
    is_cold_start_user,
    filter_to_warm_users,
    build_user_sequences,
)


@pytest.fixture
def config():
    return SplitConfig(
        train_cutoff=pd.Timestamp("2017-01-01"),
        val_cutoff=pd.Timestamp("2018-07-01"),
        cold_start_threshold_n=3,
    )


def _ts(date_str: str) -> int:
    return int(pd.Timestamp(date_str).timestamp())


@pytest.fixture
def sample_ratings():
    # userId 1: 4 ratings, all pre-2017 -> warm, train user
    # userId 2: 1 rating in train, 1 in val -> thin train history (< N)
    # userId 3: only ratings in val/test, zero train history -> cold start
    return pd.DataFrame({
        "userId":    [1, 1, 1, 1, 2, 2, 3, 3],
        "movieId":   [10, 20, 30, 40, 10, 20, 10, 20],
        "rating":    [4.0] * 8,
        "timestamp": [
            _ts("2016-01-01"), _ts("2016-02-01"),
            _ts("2016-03-01"), _ts("2016-04-01"),
            _ts("2016-05-01"), _ts("2017-06-01"),
            _ts("2017-08-01"), _ts("2018-08-01"),
        ],
    })


def test_assign_split_boundaries(sample_ratings, config):
    out = assign_split(sample_ratings, config)
    # userId 1's 4 ratings are all before 2017-01-01 -> train
    assert (out[out["userId"] == 1]["split"] == "train").all()
    # userId 2: first rating 2016 (train), second 2017-06 (val, since < 2018-07-01)
    u2 = out[out["userId"] == 2].sort_values("timestamp")
    assert u2["split"].tolist() == ["train", "val"]
    # userId 3: 2017-08 -> val, 2018-08 -> test
    u3 = out[out["userId"] == 3].sort_values("timestamp")
    assert u3["split"].tolist() == ["val", "test"]


def test_no_split_leakage(sample_ratings, config):
    """No train-split row should have a timestamp >= train_cutoff, and
    no val-split row should have a timestamp >= val_cutoff."""
    out = assign_split(sample_ratings, config)
    dt = pd.to_datetime(out["timestamp"], unit="s")
    assert (dt[out["split"] == "train"] < config.train_cutoff).all()
    assert (dt[out["split"] == "val"] < config.val_cutoff).all()
    assert (dt[out["split"] == "val"] >= config.train_cutoff).all()
    assert (dt[out["split"] == "test"] >= config.val_cutoff).all()


def test_warm_users(sample_ratings, config):
    out = assign_split(sample_ratings, config)
    train = out[out["split"] == "train"]
    warm = get_warm_users(train, config)
    # userId 1 has 4 train ratings >= N=3 -> warm
    assert 1 in warm
    # userId 2 has 1 train rating < N=3 -> not warm
    assert 2 not in warm
    # userId 3 has 0 train ratings -> not warm (not even in train grouping)
    assert 3 not in warm


def test_is_cold_start_user(config):
    assert is_cold_start_user(0, config) is True
    assert is_cold_start_user(2, config) is True
    assert is_cold_start_user(3, config) is False
    assert is_cold_start_user(10, config) is False


def test_filter_to_warm_users(sample_ratings, config):
    out = assign_split(sample_ratings, config)
    train = out[out["split"] == "train"]
    val = out[out["split"] == "val"]
    warm = get_warm_users(train, config)

    val_warm = filter_to_warm_users(val, warm)
    # Only userId 1 is warm; userId 1 has no val ratings in this fixture,
    # so val_warm should be empty
    assert val_warm.empty

    # userId 2's val rating should be excluded (not warm)
    assert 2 not in val_warm["userId"].values


def test_build_user_sequences_is_chronological(sample_ratings):
    seqs = build_user_sequences(sample_ratings)
    # userId 1's ratings were inserted in timestamp order already: 10,20,30,40
    assert seqs.loc[1] == [10, 20, 30, 40]


def test_split_config_from_json(tmp_path):
    import json
    cfg_path = tmp_path / "week1_config.json"
    cfg_path.write_text(json.dumps({
        "train_cutoff": "2017-01-01",
        "val_cutoff": "2018-07-01",
        "cold_start_threshold_N": 25,
    }))
    cfg = SplitConfig.from_json(cfg_path)
    assert cfg.train_cutoff == pd.Timestamp("2017-01-01")
    assert cfg.cold_start_threshold_n == 25
