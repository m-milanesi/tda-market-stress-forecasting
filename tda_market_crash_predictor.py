"""Test whether persistent-homology features improve market-stress forecasts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from persim import wasserstein
from ripser import ripser
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 43
HORIZON = 3
STRESS_LEVEL = 0.25
TDA_WINDOW = 20

START = "1999-01-01"
END = "2020-07-10"
TEST_START = "2019-01-01"
TEST_END = "2020-07-01"

TICKERS = ["^GSPC", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
MARKET = "^GSPC"
SECTORS = TICKERS[1:]

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
PRICE_FILE = DATA_DIR / "market_stress_panel.csv"
TDA_FILE = DATA_DIR / "covid_tda.csv"

CLASSICAL_FEATURES = [
    "market_return_1d",
    "market_momentum_5d",
    "market_volatility_5d",
    "market_momentum_20d",
    "market_volatility_20d",
    "market_momentum_60d",
    "market_volatility_60d",
    "sector_mean_return_1d",
    "sector_dispersion_1d",
    "sector_negative_share_1d",
]

TDA_FEATURES = ["tda_h0_wasserstein", "tda_h1_wasserstein"]
MODEL_NAMES = ["Logistic", "Random Forest", "Gradient Boosting"]


def load_prices():
    """Download adjusted prices once, then read the saved CSV on later runs."""
    DATA_DIR.mkdir(exist_ok=True)

    if PRICE_FILE.exists():
        prices = pd.read_csv(PRICE_FILE, index_col="date", parse_dates=True)
    else:
        prices = yf.download(
            TICKERS,
            start=START,
            end=END,
            auto_adjust=True,
            progress=False,
        )["Close"]
        prices = prices[TICKERS].dropna()
        prices.index.name = "date"
        prices.to_csv(PRICE_FILE)

    return prices[TICKERS].sort_index()


def persistence_distance(earlier, later):
    """Calculate H0 and H1 Wasserstein distances between two point clouds."""
    earlier_diagrams = ripser(earlier, maxdim=1)["dgms"]
    later_diagrams = ripser(later, maxdim=1)["dgms"]
    distances = []

    for dimension in [0, 1]:
        first = earlier_diagrams[dimension]
        second = later_diagrams[dimension]

        first = first[np.isfinite(first[:, 1])]
        second = second[np.isfinite(second[:, 1])]

        if len(first) == 0 and len(second) == 0:
            distances.append(0.0)
        else:
            distances.append(wasserstein(first, second))

    return distances[0], distances[1]


def make_tda_features(sector_returns):
    """Compare two recent 20-day persistence diagrams on every date."""
    if TDA_FILE.exists():
        return pd.read_csv(TDA_FILE, index_col="date", parse_dates=True)

    h0_values = []
    h1_values = []
    values = sector_returns.to_numpy()

    print("Computing persistent homology...")

    for i in range(2 * TDA_WINDOW, len(values)):
        earlier = values[i - 2 * TDA_WINDOW : i - TDA_WINDOW]
        later = values[i - TDA_WINDOW + 1 : i + 1]
        h0, h1 = persistence_distance(earlier, later)
        h0_values.append(h0)
        h1_values.append(h1)

    tda = pd.DataFrame(
        {
            "tda_h0_wasserstein": h0_values,
            "tda_h1_wasserstein": h1_values,
        },
        index=sector_returns.index[2 * TDA_WINDOW :],
    )
    tda.index.name = "date"
    tda.to_csv(TDA_FILE)
    return tda


def make_classical_features(returns):
    """Create standard momentum, volatility and cross-sectional features."""
    market_returns = returns[MARKET]
    sector_returns = returns[SECTORS]
    features = pd.DataFrame(index=returns.index)

    features["market_return_1d"] = market_returns

    for window in [5, 20, 60]:
        features[f"market_momentum_{window}d"] = market_returns.rolling(window).sum()
        features[f"market_volatility_{window}d"] = (
            market_returns.rolling(window).std() * np.sqrt(252)
        )

    features["sector_mean_return_1d"] = sector_returns.mean(axis=1)
    features["sector_dispersion_1d"] = sector_returns.std(axis=1)
    features["sector_negative_share_1d"] = (sector_returns < 0).mean(axis=1)
    return features


def make_target(market_returns):
    """Label dates followed by three days of annualized volatility above 25%."""
    future_returns = pd.concat(
        [market_returns.shift(-day) for day in range(1, HORIZON + 1)],
        axis=1,
    )
    future_volatility = np.sqrt(252 * future_returns.pow(2).mean(axis=1))
    target = (future_volatility > STRESS_LEVEL).astype(float)
    target[future_returns.isna().any(axis=1)] = np.nan

    return pd.DataFrame(
        {"future_volatility_3d": future_volatility, "target": target}
    )


def make_dataset():
    prices = load_prices()
    returns = np.log(prices / prices.shift(1)).dropna()

    classical = make_classical_features(returns)
    topology = make_tda_features(returns[SECTORS])
    target = make_target(returns[MARKET])

    data = classical.join(topology).join(target).dropna()
    data["target"] = data["target"].astype(int)
    return data


def train_test_split(data):
    """Use pre-2019 data for training and 2019-June 2020 as the case study."""
    train = data[data.index < TEST_START].iloc[:-HORIZON].copy()
    test = data[(data.index >= TEST_START) & (data.index < TEST_END)].copy()
    return train, test


def build_models():
    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        max_features=0.7,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    gradient_boosting = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
    )

    return {
        "Logistic": logistic,
        "Random Forest": random_forest,
        "Gradient Boosting": gradient_boosting,
    }


def feature_sets():
    return {
        "Classical": CLASSICAL_FEATURES,
        "TDA only": TDA_FEATURES,
        "Classical + TDA": CLASSICAL_FEATURES + TDA_FEATURES,
    }


def cross_validate(train):
    """Run four expanding-window validation folds without shuffling time."""
    rows = []
    splitter = TimeSeriesSplit(n_splits=4, gap=HORIZON)

    for feature_name, columns in feature_sets().items():
        for model_name, model in build_models().items():
            for fold, (train_index, valid_index) in enumerate(splitter.split(train), 1):
                fold_train = train.iloc[train_index]
                fold_valid = train.iloc[valid_index]

                model.fit(fold_train[columns], fold_train["target"])
                probability = model.predict_proba(fold_valid[columns])[:, 1]

                rows.append(
                    {
                        "model": model_name,
                        "feature_set": feature_name,
                        "fold": fold,
                        "average_precision": average_precision_score(
                            fold_valid["target"], probability
                        ),
                        "roc_auc": roc_auc_score(fold_valid["target"], probability),
                    }
                )

    return pd.DataFrame(rows)


def evaluate_test(train, test):
    """Fit on all training data and evaluate each model-feature combination."""
    rows = []
    predictions = pd.DataFrame({"target": test["target"]}, index=test.index)

    for feature_name, columns in feature_sets().items():
        for model_name, model in build_models().items():
            model.fit(train[columns], train["target"])
            probability = model.predict_proba(test[columns])[:, 1]

            predictions[f"{model_name} - {feature_name}"] = probability
            rows.append(
                {
                    "model": model_name,
                    "feature_set": feature_name,
                    "average_precision": average_precision_score(
                        test["target"], probability
                    ),
                    "roc_auc": roc_auc_score(test["target"], probability),
                }
            )

    return pd.DataFrame(rows), predictions


def save_plot(test_results):
    table = test_results.pivot(
        index="model", columns="feature_set", values="average_precision"
    )
    table = table.reindex(MODEL_NAMES)
    table = table[["Classical", "Classical + TDA"]]

    ax = table.plot(kind="bar", figsize=(9, 5), color=["#8393AA", "#258477"])
    ax.set_title("Does TDA improve market-stress forecasting?")
    ax.set_ylabel("Average precision")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Feature set")
    ax.grid(axis="y", alpha=0.2)

    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", padding=3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tda_value_added.png", dpi=180)
    plt.close()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    data = make_dataset()
    train, test = train_test_split(data)

    cv_results = cross_validate(train)
    test_results, predictions = evaluate_test(train, test)

    cv_results.to_csv(RESULTS_DIR / "cross_validation_results.csv", index=False)
    test_results.to_csv(RESULTS_DIR / "test_results.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "test_predictions.csv")
    save_plot(test_results)

    table = test_results.pivot(
        index="model", columns="feature_set", values="average_precision"
    ).reindex(MODEL_NAMES)
    table["TDA gain"] = table["Classical + TDA"] - table["Classical"]

    print(f"Train: {train.index.min().date()} to {train.index.max().date()}")
    print(f"Test:  {test.index.min().date()} to {test.index.max().date()}")
    print("\nHeld-out average precision")
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
