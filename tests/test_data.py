from pathlib import Path

import pandas as pd


# PROCESSED_DIR = Path(
#     "data/processed"
# )

from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def test_files_exist():
    required = [
        "users.parquet",
        "items.parquet",
        "interactions.parquet",
        "train.parquet",
        "validation.parquet",
        "test.parquet",
    ]

    for file in required:
        assert (
            PROCESSED_DIR / file
        ).exists()


def test_no_missing_user_ids():
    users = pd.read_parquet(
        PROCESSED_DIR /
        "users.parquet"
    )

    assert (
        users["user_id"]
        .isna()
        .sum()
        == 0
    )


def test_no_missing_item_ids():
    items = pd.read_parquet(
        PROCESSED_DIR /
        "items.parquet"
    )

    assert (
        items["item_id"]
        .isna()
        .sum()
        == 0
    )


def test_reward_binary():
    interactions = (
        pd.read_parquet(
            PROCESSED_DIR /
            "interactions.parquet"
        )
    )

    assert set(
        interactions["reward"]
        .unique()
    ).issubset(
        {0, 1}
    )


def test_split_sizes():
    interactions = (
        pd.read_parquet(
            PROCESSED_DIR /
            "interactions.parquet"
        )
    )

    train = pd.read_parquet(
        PROCESSED_DIR /
        "train.parquet"
    )

    validation = (
        pd.read_parquet(
            PROCESSED_DIR /
            "validation.parquet"
        )
    )

    test = pd.read_parquet(
        PROCESSED_DIR /
        "test.parquet"
    )

    total = (
        len(train)
        + len(validation)
        + len(test)
    )

    assert total == len(
        interactions
    )


def test_user_features():
    users = pd.read_parquet(
        PROCESSED_DIR /
        "users.parquet"
    )

    required = [
        "avg_rating",
        "interaction_count",
        "positive_rate",
    ]

    for col in required:
        assert col in users.columns