"""Small risk-overlay backtest using the out-of-sample stress probabilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_FILE = Path("data/market_stress_panel.csv")
PREDICTION_FILE = Path("results/test_predictions.csv")
RESULTS_DIR = Path("results")
MARKET = "^GSPC"
TRANSACTION_COST = 0.0005

SIGNALS = {
    "Classical signal": "Gradient Boosting - Classical",
    "Classical + TDA signal": "Gradient Boosting - Classical + TDA",
}


def calculate_performance(returns):
    """Calculate standard performance statistics from daily strategy returns."""
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    annualized_volatility = returns.std() * np.sqrt(252)

    return {
        "total_return": equity.iloc[-1] - 1,
        "annualized_return": equity.iloc[-1] ** (252 / len(returns)) - 1,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": returns.mean() / returns.std() * np.sqrt(252),
        "maximum_drawdown": drawdown.min(),
    }


def run_backtest(prices, predictions):
    """Convert yesterday's stress score into today's S&P 500 exposure."""
    market_returns = prices[MARKET].pct_change().reindex(predictions.index)
    daily = pd.DataFrame({"Buy and hold": market_returns})

    for strategy_name, probability_column in SIGNALS.items():
        probability = predictions[probability_column]

        # shift(1) prevents using today's closing information for today's trade
        exposure = 1 - probability.shift(1)
        turnover = exposure.diff().abs().fillna(0)
        strategy_returns = exposure * market_returns - TRANSACTION_COST * turnover

        daily[f"{strategy_name} exposure"] = exposure
        daily[strategy_name] = strategy_returns

    return daily.dropna()


def performance_table(daily):
    rows = []

    for strategy_name in ["Buy and hold"] + list(SIGNALS):
        scores = calculate_performance(daily[strategy_name])
        scores["strategy"] = strategy_name
        rows.append(scores)

    columns = [
        "strategy",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
    ]
    return pd.DataFrame(rows)[columns]


def save_equity_plot(daily):
    strategy_columns = ["Buy and hold"] + list(SIGNALS)
    equity = (1 + daily[strategy_columns]).cumprod()

    ax = equity.plot(figsize=(10, 5), linewidth=2)
    ax.set_title("Out-of-sample risk-overlay backtest")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "backtest_equity.png", dpi=180)
    plt.close()


def main():
    prices = pd.read_csv(DATA_FILE, index_col="date", parse_dates=True)
    predictions = pd.read_csv(PREDICTION_FILE, index_col="date", parse_dates=True)

    daily = run_backtest(prices, predictions)
    metrics = performance_table(daily)

    daily.to_csv(RESULTS_DIR / "backtest_daily.csv")
    metrics.to_csv(RESULTS_DIR / "backtest_metrics.csv", index=False)
    save_equity_plot(daily)

    percentages = [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "maximum_drawdown",
    ]
    printable = metrics.copy()
    printable[percentages] = 100 * printable[percentages]

    print("Backtest: January 2019 to June 2020")
    print("Returns, volatility and drawdown are shown as percentages.\n")
    print(printable.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
