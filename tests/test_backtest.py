import numpy as np
import pandas as pd

from backtest import calculate_performance, run_backtest


def test_signal_is_traded_one_day_later():
    index = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"^GSPC": [100, 101, 102, 103]}, index=index)
    predictions = pd.DataFrame(
        {
            "Gradient Boosting - Classical": [0.2, 0.4, 0.6, 0.8],
            "Gradient Boosting - Classical + TDA": [0.1, 0.3, 0.5, 0.7],
        },
        index=index,
    )

    daily = run_backtest(prices, predictions)

    assert daily.iloc[0]["Classical signal exposure"] == 0.8
    assert daily.iloc[0]["Classical + TDA signal exposure"] == 0.9


def test_performance_metrics_are_finite():
    returns = pd.Series([0.01, -0.02, 0.015, 0.005])
    metrics = calculate_performance(returns)

    assert set(metrics) == {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
    }
    assert np.isfinite(list(metrics.values())).all()
