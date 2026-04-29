"""
scripts/helpers.py
══════════════════════════════════════════════════════════════════════
Shared utilities for economic-metric forecasting notebooks.

Sections
────────
1. Error Metrics    — MAPE, SMAPE, MAE, RMSE
2. DataFrame Utils  — build_pred_df, get_benchmark, evaluate_pred_df
3. Data Loading     — load_config, load_actual_data, load_benchmark_data
4. Visualisation    — plot_results, plot_all_horizons
══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import io
import datetime
import warnings

import boto3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════
# 1.  Error Metrics
# ══════════════════════════════════════════════════════════════════

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (%).
    Formula: mean( |y - ŷ| / |y| ) × 100
    """
    return float(mean_absolute_percentage_error(y_true, y_pred) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (%).
    Formula: mean( 2|y - ŷ| / (|y| + |ŷ|) ) × 100

    Unlike MAPE, SMAPE is bounded and treats over/under-forecasts symmetrically.
    """
    denom = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(2 * np.abs(y_true - y_pred) / denom) * 100)


def print_metrics(label: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Pretty-print all four error metrics for a single model."""
    print(f"  {label}")
    print(f"    MAPE  : {mape(y_true, y_pred):.2f}%")
    print(f"    SMAPE : {smape(y_true, y_pred):.2f}%")
    print(f"    MAE   : {mean_absolute_error(y_true, y_pred):>16,.0f}")
    print(f"    RMSE  : {np.sqrt(mean_squared_error(y_true, y_pred)):>16,.0f}")


# ══════════════════════════════════════════════════════════════════
# 2.  DataFrame Utilities
# ══════════════════════════════════════════════════════════════════

def build_pred_df(
    forecast_values: np.ndarray,
    test_series: pd.Series,
    model_label: str = "SARIMAX",
    benchmark_dfs: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Combine Actual + Model Prediction (+ optional benchmark series)
    into a single long-format DataFrame.

    Parameters
    ----------
    forecast_values : array of predicted values aligned with test_series.index
    test_series     : pd.Series of actual values (index = dates)
    model_label     : label for the prediction rows (e.g. 'SARIMAX')
    benchmark_dfs   : list of DataFrames returned by get_benchmark()

    Returns
    -------
    DataFrame with columns: DATE, FORECAST_TYPE, METRIC_VALUE
    """
    actual_rows = pd.DataFrame({
        "DATE":          test_series.index,
        "FORECAST_TYPE": "Actual",
        "METRIC_VALUE":  test_series.values,
    })
    pred_rows = pd.DataFrame({
        "DATE":          test_series.index,
        "FORECAST_TYPE": model_label,
        "METRIC_VALUE":  forecast_values,
    })

    parts = [actual_rows, pred_rows]
    if benchmark_dfs:
        parts.extend(benchmark_dfs)

    pred_df = pd.concat(parts, ignore_index=True)
    # Normalise DATE type to plain date (not Timestamp)
    pred_df["DATE"] = pred_df["DATE"].apply(
        lambda x: x.date() if isinstance(x, pd.Timestamp) else x
    )
    return pred_df


def get_benchmark(
    metric_df: pd.DataFrame,
    portfolio: str,
    sub_portfolio: str | None,
    target_metric: str,
    test_dates: pd.Index,
    forecast_type_label: str,
) -> pd.DataFrame:
    """
    Extract a named benchmark series (e.g. '2025 0+12', 'original base case')
    from the full long-format metrics DataFrame loaded from S3.

    Returns a DataFrame with columns: DATE, FORECAST_TYPE, METRIC_VALUE
    filtered to dates that appear in test_dates.
    """
    mask = (
        (metric_df["PORTFOLIO"]     == portfolio)
        & (metric_df["METRIC"]      == target_metric)
        & (metric_df["FORECAST_TYPE"] == forecast_type_label)
    )
    if sub_portfolio:
        mask &= metric_df["SUB_PORTFOLIO"] == sub_portfolio

    bdf = (
        metric_df[mask][["DATE", "METRIC_VALUE"]]
        .copy()
        .assign(
            DATE=lambda x: pd.to_datetime(x["DATE"]).dt.date,
            FORECAST_TYPE=forecast_type_label,
        )
    )
    return bdf[bdf["DATE"].isin(test_dates)]


def evaluate_pred_df(
    pred_df: pd.DataFrame,
    model_labels: str | list[str],          # now accepts one or many
    target_metric: str,
) -> None:
    if isinstance(model_labels, str):
        model_labels = [model_labels]

    y_actual = (
        pred_df[pred_df["FORECAST_TYPE"] == "Actual"]
        .sort_values("DATE")
        .set_index("DATE")["METRIC_VALUE"]
        .dropna()
    )
    print(f"\n{'═'*55}")
    print(f"  Error Metrics — {target_metric}")
    print(f"{'═'*55}")

    for label in model_labels:
        subset = pred_df[pred_df["FORECAST_TYPE"] == label]
        if subset.empty:
            print(f"  {label} — no data")
            continue
        y_pred = (
            subset.sort_values("DATE")
            .set_index("DATE")["METRIC_VALUE"]
            .dropna()
        )
        common = y_actual.index.intersection(y_pred.index)
        if not len(common):
            print(f"  {label} — no overlapping dates")
            continue
        print_metrics(
            label,
            y_actual[common].values.astype(float),
            y_pred[common].values.astype(float),
        )
    print(f"{'═'*55}")


# ══════════════════════════════════════════════════════════════════
# 3.  Data Loading
# ══════════════════════════════════════════════════════════════════

def load_config(path: str = "config.yaml") -> dict:
    """Load and return the YAML configuration as a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_actual_data(
    data_dir: str,
    portfolio: str,
    sub_portfolio: str | None,
    forecast_type: str,
) -> pd.DataFrame:
    """
    Load the actual-values CSV from the data directory.

    File naming convention:
        {portfolio}_{sub_portfolio}_{forecast_type}.csv
        e.g.  Buckeye_LP_Actual.csv

    The DATE column is parsed and set as a date (not datetime) index.
    """
    suffix = f"_{sub_portfolio}" if sub_portfolio else ""
    path   = f"{data_dir}/{portfolio}{suffix}_{forecast_type}.csv"
    df     = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
    df = df.set_index("DATE").sort_index()
    return df


def load_benchmark_data(
    bucket: str,
    folder: str,
    metrics_file: str,
) -> pd.DataFrame:
    """
    Load the combined metrics parquet (original base case, 0+12 forecasts, etc.)
    from S3.  Returns the raw long-format DataFrame; DATE is converted to date.
    """
    s3  = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=f"{folder}/{metrics_file}")
    df  = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
    return df


def slice_horizon(
    data_df: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Slice data_df into train and test DataFrames based on ISO date strings.

    Returns
    -------
    train_df, test_df
    """
    ts = datetime.date.fromisoformat(train_start)
    te = datetime.date.fromisoformat(train_end)
    vs = datetime.date.fromisoformat(test_start)
    ve = datetime.date.fromisoformat(test_end)

    train_df = data_df[(data_df.index >= ts) & (data_df.index <= te)]
    test_df  = data_df[(data_df.index >= vs) & (data_df.index <= ve)]
    return train_df, test_df


# ══════════════════════════════════════════════════════════════════
# 4.  Visualisation
# ══════════════════════════════════════════════════════════════════

_PALETTE: dict[str, str] = {
    "Actual":                        "#A2CEED",
    "SARIMAX":                       "orange",
    "Statistical Model prediction":  "orange",
    "original base case":            "green",
    "2024 0+12":                     "gold",
    "2025 0+12":                     "purple",
    "SARIMA":                        "orange",
    "Holt-Winters":                  "red",
    "ETS":                           "red",
}

_PLOT_ORDER: list[str] = [
    "Actual", "SARIMAX", "Statistical Model prediction",
    "original base case", "SARIMA", "Holt-Winters", "ETS",
    "2024 0+12", "2025 0+12",
]


def plot_results(pred_df: pd.DataFrame, title: str) -> None:
    """
    Line chart of all FORECAST_TYPE series in a long-format prediction DataFrame.
    Unknown labels get a default colour; order follows _PLOT_ORDER.
    """
    existing = [k for k in _PLOT_ORDER if k in pred_df["FORECAST_TYPE"].unique()]
    # Include any extra labels not in the default order
    for k in pred_df["FORECAST_TYPE"].unique():
        if k not in existing:
            existing.append(k)

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.set_title(title, fontsize=14, fontweight="bold")

    for label in existing:
        subset = pred_df[pred_df["FORECAST_TYPE"] == label].sort_values("DATE")
        ax.plot(
            range(len(subset)),
            subset["METRIC_VALUE"].values,
            marker=None,
            label=label,
            color=_PALETTE.get(label),
            linewidth=2,
        )

    anchor = pred_df[pred_df["FORECAST_TYPE"] == existing[0]].sort_values("DATE")
    x_labels = anchor["DATE"].astype(str).tolist()
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_xlabel("DATE")
    ax.set_ylabel("METRIC_VALUE")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    plt.show()


def plot_all_horizons(
    pred_2024: pd.DataFrame,
    pred_2025: pd.DataFrame,
    target_metric: str,
    model_label: str = "SARIMAX",
    pred_2023: pd.DataFrame | None = None,   # optional
) -> None:
    """
    Render three side-by-side panels — one per test horizon — with MAPE
    annotations in each subplot title.

    Parameters
    ----------
    pred_2023/2024/2025 : long-format DataFrames from build_pred_df()
    target_metric       : string shown in the overall figure title
    model_label         : FORECAST_TYPE value to compare against 'Actual'
    """
    horizon_specs = []
    if pred_2023 is not None:
        horizon_specs.append(("2023 Test Horizon (tuning)", pred_2023))
    horizon_specs += [
        ("2024 Test Horizon (tuning)", pred_2024),
        ("2025 Production Forecast",   pred_2025),
    ]
    n_panels = len(horizon_specs)
    fig, axes = plt.subplots(1, n_panels, figsize=(10 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]   # ensure iterable when only one panel
    fig.suptitle(target_metric, fontsize=15, fontweight="bold", y=1.01)

    for ax, (title, pred_df) in zip(axes, horizon_specs):
        existing = [k for k in _PLOT_ORDER if k in pred_df["FORECAST_TYPE"].unique()]
        for k in pred_df["FORECAST_TYPE"].unique():
            if k not in existing:
                existing.append(k)

        for label in existing:
            subset = pred_df[pred_df["FORECAST_TYPE"] == label].sort_values("DATE")
            ax.plot(
                range(len(subset)),
                subset["METRIC_VALUE"].values,
                label=label,
                color=_PALETTE.get(label),
                linewidth=2,
            )

        # ── MAPE annotation (drop missing actuals) ───────────────
        actual_series = (
            pred_df[pred_df["FORECAST_TYPE"] == "Actual"]
            .set_index("DATE")["METRIC_VALUE"]
            .dropna()
        )
        model_series = (
            pred_df[pred_df["FORECAST_TYPE"] == model_label]
            .set_index("DATE")["METRIC_VALUE"]
            .dropna()
        )
        common = actual_series.index.intersection(model_series.index)
        if len(common):
            actual_vals = actual_series.loc[common].values.astype(float)
            model_vals = model_series.loc[common].values.astype(float)
            m = float(np.mean(np.abs((actual_vals - model_vals) / actual_vals)) * 100)
            ax.set_title(f"{title}\nMAPE: {m:.2f}%", fontsize=11)
        else:
            ax.set_title(title, fontsize=11)

        anchor   = pred_df[pred_df["FORECAST_TYPE"] == existing[0]].sort_values("DATE")
        x_labels = anchor["DATE"].astype(str).tolist()
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=90, fontsize=7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
