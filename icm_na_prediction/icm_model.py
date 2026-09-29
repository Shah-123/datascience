"""Predict "Not Available" (NA) children per monitor per polio campaign.

The raw data is one row per cluster checked by an Independent Campaign
Monitoring (ICM) monitor. We aggregate it to one row per monitor per campaign
and predict how many children the monitor should find "not available".

Leakage note
------------
For 99.99% of rows the survey satisfies an accounting identity::

    (RECALL_*_CHK - RECALL_*_VAC) == NT + TVBMC + NA + ASLEEP + REFUSAL + OTHER_REASON

so NA can be reconstructed exactly from the other reason columns and the
unvaccinated counts. Those columns (and every ``*_VAC`` column) are excluded
from the features. The model only sees workload information from the current
campaign plus the monitor's NA history from *previous* campaigns.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "icm_data.csv"
TARGET = "NA"
KEYS = ["MONITORID", "CAMP_ID"]

# Columns that encode the target through the accounting identity above.
LEAKY_COLUMNS = [
    "NT", "TVBMC", "ASLEEP", "REFUSAL", "OTHER_REASON", "VBNFM",
    "RECALL_011_VAC", "RECALL_1259_VAC", "GUEST_VAC", "FM_059_VAC",
    "NT_VAC", "TVBM_VAC", "NA_VAC", "ASLEEP_VAC", "REFUSAL_VAC", "OTHER_VAC", "VBNFM_VAC",
]

WORKLOAD_FEATURES = [
    "n_clusters", "n_ucs", "TOTAL_HH", "RECALL_011_CHK", "RECALL_1259_CHK",
    "FM_059_CHK", "GUEST_CHK", "ZERO_DOSE_023", "door_mark_rate", "hrmp_rate",
]
HISTORY_FEATURES = ["prev_na_rate", "hist_na_rate", "n_prev_campaigns"]
FEATURES = WORKLOAD_FEATURES + HISTORY_FEATURES

REQUIRED_COLUMNS = [
    "MONITORID", "UCID", "CAMP_ID", "CLUSTER_DATE", "TOTAL_HH", "HRMP",
    "RECALL_011_CHK", "RECALL_1259_CHK", "FM_059_CHK", "GUEST_CHK",
    "ZERO_DOSE_023", "CORRECT_DOOR_MARK", TARGET,
]


def load_raw(source=DATA_PATH) -> pd.DataFrame:
    """Read the cluster-level ICM export (a path or an uploaded file)."""
    raw = pd.read_csv(source)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    raw["CLUSTER_DATE"] = pd.to_datetime(raw["CLUSTER_DATE"], format="mixed")
    return raw


def build_monitor_campaign_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate clusters to monitor x campaign and add history features."""
    table = (
        raw.assign(n_clusters=1)
        .groupby(KEYS)
        .agg(
            campaign_start=("CLUSTER_DATE", "min"),
            n_clusters=("n_clusters", "sum"),
            n_ucs=("UCID", "nunique"),
            TOTAL_HH=("TOTAL_HH", "sum"),
            RECALL_011_CHK=("RECALL_011_CHK", "sum"),
            RECALL_1259_CHK=("RECALL_1259_CHK", "sum"),
            FM_059_CHK=("FM_059_CHK", "sum"),
            GUEST_CHK=("GUEST_CHK", "sum"),
            ZERO_DOSE_023=("ZERO_DOSE_023", "sum"),
            door_mark_rate=("CORRECT_DOOR_MARK", "mean"),
            hrmp_rate=("HRMP", "mean"),
            NA=(TARGET, "sum"),
        )
        .reset_index()
    )
    # Campaign IDs are not in date order (e.g. campaign 0 ran after 1 and 2),
    # so order campaigns by their first survey date.
    campaign_start = table.groupby("CAMP_ID")["campaign_start"].min()
    table["campaign_order"] = table["CAMP_ID"].map(campaign_start.rank(method="dense").astype(int))
    table = table.sort_values(["campaign_order", "MONITORID"]).reset_index(drop=True)

    children_checked = (table["RECALL_011_CHK"] + table["RECALL_1259_CHK"]).clip(lower=1)
    na_rate = table[TARGET] / children_checked
    by_monitor = na_rate.groupby(table["MONITORID"])
    # shift(1) so a campaign only sees the monitor's *earlier* campaigns.
    table["prev_na_rate"] = by_monitor.shift(1)
    table["hist_na_rate"] = by_monitor.transform(lambda s: s.shift(1).expanding().mean())
    table["n_prev_campaigns"] = table.groupby("MONITORID").cumcount()
    return table


def temporal_split(table: pd.DataFrame, n_test_campaigns: int = 5):
    """Hold out the most recent campaigns as the test set."""
    cutoff = table["campaign_order"].max() - n_test_campaigns
    is_test = table["campaign_order"] > cutoff
    return table[~is_test], table[is_test]


def make_models() -> dict:
    return {
        "Poisson GLM": make_pipeline(
            # GLMs can't take NaNs; a monitor's first campaign has no history.
            SimpleImputer(strategy="median"), StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=1000)
        ),
        "Gradient boosting (Poisson)": HistGradientBoostingRegressor(
            loss="poisson", learning_rate=0.05, max_iter=400, random_state=42
        ),
    }


def baseline_predict(train: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    """Overall NA rate from training data times children checked."""
    checked_train = train["RECALL_011_CHK"] + train["RECALL_1259_CHK"]
    rate = train[TARGET].sum() / checked_train.sum()
    return rate * (frame["RECALL_011_CHK"] + frame["RECALL_1259_CHK"]).to_numpy()


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


@dataclass
class TrainingResult:
    model: object
    model_name: str
    metrics: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame


def fit_and_evaluate(table: pd.DataFrame, n_test_campaigns: int = 5) -> TrainingResult:
    """Compare a baseline and two models on a time-based hold-out.

    The best model is then refit on all campaigns so it can score new data.
    """
    train, test = temporal_split(table, n_test_campaigns)
    rows = [{"Model": "Baseline (avg NA rate x children checked)",
             **regression_metrics(test[TARGET], baseline_predict(train, test))}]
    fitted = {}
    for name, model in make_models().items():
        model.fit(train[FEATURES], train[TARGET])
        fitted[name] = model
        rows.append({"Model": name, **regression_metrics(test[TARGET], model.predict(test[FEATURES]))})
    metrics = pd.DataFrame(rows).set_index("Model")

    best_name = metrics.drop(index=metrics.index[0])["MAE"].idxmin()
    test = test.assign(Predicted_NA=fitted[best_name].predict(test[FEATURES]))
    final_model = make_models()[best_name].fit(table[FEATURES], table[TARGET])
    return TrainingResult(final_model, best_name, metrics, train, test)


def flag_unusual_reports(frame: pd.DataFrame, predicted: np.ndarray, threshold: float = 3.0) -> pd.DataFrame:
    """Score how far each reported NA is from the model's expectation.

    Uses a Pearson residual, (actual - expected) / sqrt(expected), rescaled
    by a robust spread estimate because the counts are over-dispersed.
    """
    expected = np.clip(predicted, 0.5, None)
    pearson = (frame[TARGET].to_numpy() - expected) / np.sqrt(expected)
    mad = np.median(np.abs(pearson - np.median(pearson))) * 1.4826
    score = pearson / (mad if mad > 0 else 1.0)
    out = frame[KEYS + ["campaign_start", TARGET]].copy()
    out["Predicted_NA"] = np.round(predicted, 1)
    out["Difference"] = np.round(out[TARGET] - predicted, 1)
    out["Anomaly_score"] = np.round(score, 2)
    out["Flag"] = np.select(
        [score > threshold, score < -threshold],
        ["Higher than expected", "Lower than expected"],
        default="",
    )
    return out
