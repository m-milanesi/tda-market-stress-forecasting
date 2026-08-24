import math
from pathlib import Path

import numpy as np
import pandas as pd

from tda_market_crash_predictor import (
    build_models,
    make_target,
    persistence_distance,
    train_test_split,
)


def test_persistence_distance_returns_two_nonnegative_values():
    time = np.linspace(0, 2 * np.pi, 20)
    first = np.column_stack((np.sin(time), np.cos(time), time / 10))
    second = np.column_stack((1.2 * np.sin(time), np.cos(time), time / 8))

    h0, h1 = persistence_distance(first, second)

    assert np.isfinite([h0, h1]).all()
    assert h0 >= 0 and h1 >= 0


def test_target_uses_exactly_the_next_three_returns():
    index = pd.bdate_range("2024-01-01", periods=6)
    returns = pd.Series([0.01, 0.02, -0.03, 0.04, -0.05, 0.06], index=index)

    target = make_target(returns)
    expected = math.sqrt(252 * np.mean(np.square([0.02, -0.03, 0.04])))

    assert math.isclose(target.iloc[0]["future_volatility_3d"], expected)
    assert target.iloc[-3:]["target"].isna().all()


def test_split_is_chronological_and_purged():
    index = pd.bdate_range("2018-12-01", "2020-07-10")
    data = pd.DataFrame({"target": 0}, index=index)
    observations_before_2019 = (index < pd.Timestamp("2019-01-01")).sum()

    train, test = train_test_split(data)

    assert len(train) == observations_before_2019 - 3
    assert train.index.max() < pd.Timestamp("2019-01-01")
    assert test.index.min() >= pd.Timestamp("2019-01-01")
    assert test.index.max() < pd.Timestamp("2020-07-01")


def test_project_compares_three_models():
    assert list(build_models()) == [
        "Logistic",
        "Random Forest",
        "Gradient Boosting",
    ]
