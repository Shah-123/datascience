"""Clean Pakistani property listings and model sale prices and monthly rents.

Two separate models are trained because a sale price (millions of PKR) and a
monthly rent (thousands of PKR) are different targets. Mixing them in one
regression, as the first version of this project did, mostly teaches the
model to tell sales from rentals.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "data" / "house_prices_raw.csv"
MODEL_PATH = PROJECT_DIR / "models" / "house_models.joblib"

TARGET = "price"
CATEGORICAL = ["property_type", "city", "area_location"]
NUMERIC = ["Area_in_Marla", "bedrooms", "baths"]
FEATURES = CATEGORICAL + NUMERIC
PURPOSES = ["For Sale", "For Rent"]


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    # The export includes a pandas index column. Keeping it made every row
    # unique, so the original drop_duplicates() never removed anything.
    raw = pd.read_csv(path)
    return raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")])


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return the cleaned listings and a count of rows removed at each step."""
    report = {"raw rows": len(raw)}
    df = raw.assign(location=raw["location"].str.strip(), city=raw["city"].str.strip())
    df = df.drop_duplicates()
    report["after removing exact duplicates"] = len(df)
    df = df[(df["Area_in_Marla"] > 0) & (df["price"] > 0)]
    report["after removing zero area/price"] = len(df)
    # Trim the extreme 0.5% on each side within each purpose. These are mostly
    # data-entry errors, e.g. a sale price typed into a rental listing.
    bounds = df.groupby("purpose")["price"].quantile([0.005, 0.995]).unstack()
    lo = df["purpose"].map(bounds[0.005])
    hi = df["purpose"].map(bounds[0.995])
    df = df[df["price"].between(lo, hi)]
    report["after trimming extreme prices"] = len(df)
    # The same neighbourhood name can exist in several cities (e.g. DHA Defence).
    df = df.assign(area_location=df["city"] + " / " + df["location"]).reset_index(drop=True)
    return df, report


def _encoder(scale_numeric: bool) -> ColumnTransformer:
    onehot = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)
    if scale_numeric:
        # log1p keeps a linear model on log(price) from exploding for very large plots.
        numeric = make_pipeline(FunctionTransformer(np.log1p), StandardScaler())
        return ColumnTransformer([("cat", onehot, CATEGORICAL), ("num", numeric, NUMERIC)])
    return ColumnTransformer([("cat", onehot, CATEGORICAL)], remainder="passthrough")


def _log_target(pipeline: Pipeline) -> TransformedTargetRegressor:
    # Prices are right-skewed, so fit on log(price) and report errors in PKR.
    return TransformedTargetRegressor(regressor=pipeline, func=np.log, inverse_func=np.exp)


def make_xgb_model() -> TransformedTargetRegressor:
    xgb = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=42)
    return _log_target(Pipeline([("encode", _encoder(False)), ("model", xgb)]))


def make_ridge_model() -> TransformedTargetRegressor:
    return _log_target(Pipeline([("encode", _encoder(True)), ("model", Ridge(alpha=1.0))]))


class PricePerMarlaBaseline:
    """Median price per marla in the neighbourhood (falling back to the city) x area."""

    def fit(self, X: pd.DataFrame, y: pd.Series):
        per_marla = y / X["Area_in_Marla"]
        self.by_location_ = per_marla.groupby(X["area_location"]).median()
        self.by_city_ = per_marla.groupby(X["city"]).median()
        self.overall_ = per_marla.median()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        rate = X["area_location"].map(self.by_location_).fillna(X["city"].map(self.by_city_)).fillna(self.overall_)
        return (rate * X["Area_in_Marla"]).to_numpy()


def price_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "MAE (PKR)": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "Median APE": float(np.median(np.abs(y_pred - y_true) / y_true)),
        "R2": r2_score(y_true, y_pred),
    }


def split(df: pd.DataFrame, purpose: str, seed: int = 42):
    subset = df[df["purpose"] == purpose]
    return train_test_split(subset, test_size=0.2, random_state=seed)


def compare_models(df: pd.DataFrame, purpose: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold-out metrics for each model, plus the test rows with XGBoost predictions."""
    train, test = split(df, purpose)
    models = {
        "Baseline (median price per marla)": PricePerMarlaBaseline(),
        "Ridge regression (log price)": make_ridge_model(),
        "XGBoost (log price)": make_xgb_model(),
    }
    preds = {name: m.fit(train[FEATURES], train[TARGET]).predict(test[FEATURES]) for name, m in models.items()}
    metrics = pd.DataFrame({name: price_metrics(test[TARGET], p) for name, p in preds.items()}).T
    return metrics, test.assign(predicted=preds["XGBoost (log price)"])


def train_and_save(df: pd.DataFrame | None = None, path: Path = MODEL_PATH) -> dict:
    """Fit one XGBoost model per purpose on all listings and save them with the input options."""
    if df is None:
        df, _ = clean(load_raw())
    bundle = {"models": {}, "typical_error": {}}
    for purpose in PURPOSES:
        # Median APE from the hold-out split, shown in the app as the expected error.
        _, test = compare_models(df, purpose)
        bundle["typical_error"][purpose] = float(np.median(np.abs(test["predicted"] - test[TARGET]) / test[TARGET]))
        subset = df[df["purpose"] == purpose]
        bundle["models"][purpose] = make_xgb_model().fit(subset[FEATURES], subset[TARGET])
    bundle["property_types"] = sorted(df["property_type"].unique())
    # Options per purpose: e.g. the data has no rental listings in Lahore.
    bundle["locations"] = {
        purpose: {city: sorted(g["location"].unique()) for city, g in df[df["purpose"] == purpose].groupby("city")}
        for purpose in PURPOSES
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle


def load_or_train(path: Path = MODEL_PATH) -> dict:
    if path.exists():
        return joblib.load(path)
    return train_and_save(path=path)


def make_input(property_type, city, location, area_marla, bedrooms, baths) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "property_type": property_type,
                "city": city,
                "area_location": f"{city} / {location}",
                "Area_in_Marla": area_marla,
                "bedrooms": bedrooms,
                "baths": baths,
            }
        ]
    )


def format_pkr(amount: float) -> str:
    """Format rupees the way Pakistani listings do (crore / lakh)."""
    if amount >= 1e7:
        return f"PKR {amount / 1e7:.2f} crore"
    if amount >= 1e5:
        return f"PKR {amount / 1e5:.1f} lakh"
    return f"PKR {amount:,.0f}"
