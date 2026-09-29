"""Models for the exam-scores dataset.

Two questions, two models:
- Early warning: how well can background factors alone (known before any
  exam) predict a math score?
- Consistency: given reading and writing scores, what math score is expected?
  This is easy because the three scores are strongly correlated.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "exams.csv"
TARGET = "math score"
BACKGROUND = ["gender", "ethnicity", "parental level of education", "lunch", "test preparation course"]
OTHER_SCORES = ["reading score", "writing score"]
FEATURE_SETS = {"Background only": BACKGROUND, "Background + reading/writing": BACKGROUND + OTHER_SCORES}
EDUCATION_ORDER = [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree",
]


def load_data(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path).rename(columns={"race/ethnicity": "ethnicity"})


def make_model(features: list[str]) -> Pipeline:
    numeric = [f for f in features if f in OTHER_SCORES]
    # drop="first" gives each coefficient a clear reference category.
    steps = [("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), [f for f in features if f in BACKGROUND])]
    if numeric:
        steps.append(("num", StandardScaler(), numeric))
    return Pipeline([("encode", ColumnTransformer(steps)), ("model", LinearRegression())])


def cross_validate_models(df: pd.DataFrame) -> pd.DataFrame:
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {"Baseline (predict the mean)": (BACKGROUND, Pipeline([("model", DummyRegressor())]))}
    models.update({f"Linear regression: {name}": (feats, make_model(feats)) for name, feats in FEATURE_SETS.items()})
    rows = {}
    for name, (features, model) in models.items():
        scores = cross_validate(
            model,
            df[features],
            df[TARGET],
            cv=cv,
            scoring=("neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"),
        )
        rows[name] = {
            "MAE": -scores["test_neg_mean_absolute_error"].mean(),
            "RMSE": -scores["test_neg_root_mean_squared_error"].mean(),
            "R2": scores["test_r2"].mean(),
        }
    return pd.DataFrame(rows).T


def fit_models(df: pd.DataFrame) -> dict:
    return {name: make_model(feats).fit(df[feats], df[TARGET]) for name, feats in FEATURE_SETS.items()}


def background_effects(model: Pipeline) -> pd.Series:
    """Coefficients of the background-only model, in math-score points vs the reference group."""
    names = model.named_steps["encode"].get_feature_names_out()
    coefs = pd.Series(model.named_steps["model"].coef_, index=[n.split("__", 1)[1] for n in names])
    return coefs.sort_values()


def reference_groups(model: Pipeline) -> dict:
    """The category each background coefficient is compared against."""
    encoder = model.named_steps["encode"].named_transformers_["cat"]
    return {feature: cats[0] for feature, cats in zip(BACKGROUND, encoder.categories_, strict=True)}
