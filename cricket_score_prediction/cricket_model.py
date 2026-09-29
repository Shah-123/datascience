"""Predict the final first-innings score of an ODI match from the current match state.

The training table has one row per ball (from ball 30 onwards, when the
"runs in the last five overs" feature is available). Rows from the same
innings are highly correlated, so evaluation always splits by innings:
a random row split lets the model see other balls of the very same innings
during training and overstates accuracy.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "data" / "odi_first_innings.csv"
MODEL_PATH = PROJECT_DIR / "models" / "cricket_model.joblib"

TARGET = "final_score"
CATEGORICAL = ["venue", "batting_team", "bowling_team"]
NUMERIC = ["balls_left", "wickets_left", "current_score", "current_run_rate", "last_five"]
FEATURES = CATEGORICAL + NUMERIC
BALLS_PER_INNINGS = 300

XGB_PARAMS = dict(n_estimators=500, learning_rate=0.03, max_depth=3, min_child_weight=50, n_jobs=-1, random_state=42)


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load the ball-by-ball table, give columns clear names and add an innings id."""
    df = pd.read_csv(path).rename(
        columns={
            "runs_off_bat_x": TARGET,
            "wicket_left": "wickets_left",
            "Current_Score": "current_score",
            "Crr": "current_run_rate",
        }
    )
    df["innings_id"] = infer_innings_id(df)
    return df


def infer_innings_id(df: pd.DataFrame) -> pd.Series:
    """Recover innings boundaries (the exported file dropped match_id).

    Rows are stored innings by innings. A new innings starts when the teams,
    venue or final score change, when the score goes down, or when balls_left
    jumps up by more than an over's worth of extras.
    """
    key = df[[TARGET, "venue", "batting_team", "bowling_team"]]
    new_innings = (
        (key != key.shift()).any(axis=1)
        | (df["current_score"] < df["current_score"].shift())
        | (df["balls_left"] - df["balls_left"].shift() > 30)
    )
    return new_innings.cumsum().rename("innings_id")


def make_input(venue, batting_team, bowling_team, balls_left, wickets_left, current_score, last_five) -> pd.DataFrame:
    """Build a single-row feature frame; the run rate is derived, not typed in."""
    balls_bowled = max(BALLS_PER_INNINGS - balls_left, 1)
    return pd.DataFrame(
        [
            {
                "venue": venue,
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "balls_left": balls_left,
                "wickets_left": wickets_left,
                "current_score": current_score,
                "current_run_rate": current_score * 6 / balls_bowled,
                "last_five": last_five,
            }
        ]
    )


def make_xgb_pipeline(**params) -> Pipeline:
    encoder = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)], remainder="passthrough")
    return Pipeline([("encode", encoder), ("model", XGBRegressor(**{**XGB_PARAMS, **params}))])


def make_ridge_pipeline() -> Pipeline:
    encoder = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
    return Pipeline([("encode", encoder), ("model", Ridge())])


def run_rate_projection(frame: pd.DataFrame) -> np.ndarray:
    """Commentator's baseline: current score + current run rate x overs left."""
    return (frame["current_score"] + frame["current_run_rate"] * frame["balls_left"] / 6).to_numpy()


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def innings_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split so that every ball of an innings lands on the same side."""
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(df, groups=df["innings_id"])
    )
    return df.iloc[train_idx], df.iloc[test_idx]


def compare_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return hold-out metrics per model and the test rows with predictions."""
    train, test = innings_split(df)
    preds = {"Run-rate projection (baseline)": run_rate_projection(test)}
    for name, model in {"Ridge regression": make_ridge_pipeline(), "XGBoost": make_xgb_pipeline()}.items():
        preds[name] = model.fit(train[FEATURES], train[TARGET]).predict(test[FEATURES])
    metrics = pd.DataFrame({name: regression_metrics(test[TARGET], p) for name, p in preds.items()}).T
    return metrics, test.assign(predicted=preds["XGBoost"])


def train_and_save(df: pd.DataFrame | None = None, path: Path = MODEL_PATH) -> dict:
    """Fit the final model on every innings and save it with its input options."""
    df = load_data() if df is None else df
    bundle = {
        "model": make_xgb_pipeline().fit(df[FEATURES], df[TARGET]),
        "venues": sorted(df["venue"].unique()),
        "teams": sorted(set(df["batting_team"]) | set(df["bowling_team"])),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle


def load_or_train(path: Path = MODEL_PATH) -> dict:
    if path.exists():
        return joblib.load(path)
    return train_and_save(path=path)
