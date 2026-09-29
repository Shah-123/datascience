"""Heart-failure mortality risk from clinical records.

Data caveats handled here:
- 3,680 of the 5,000 rows are exact duplicates; they are removed.
- ``time`` (the follow-up period) is excluded. Patients who died have short
  follow-up *because* they died, so it leaks the outcome and is not known
  when a prediction would be made.
- This version of the dataset was synthetically expanded from ~300 real
  patients, so near-copies of one patient can land in both train and test.
  Cross-validation keeps rows with the same age and sex in the same fold,
  which is stricter than a random split, but the scores remain optimistic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "heart_failure_clinical_records.csv"
TARGET = "DEATH_EVENT"
LEAKY = ["time"]
FEATURES = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
    "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking",
]
BASELINE_FEATURES = ["ejection_fraction", "serum_creatinine"]


def load_data(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path).drop_duplicates().reset_index(drop=True)


def patient_groups(df: pd.DataFrame) -> pd.Series:
    return df.groupby(["age", "sex"]).ngroup()


def make_models() -> dict:
    return {
        "Baseline: logistic regression on ejection fraction + creatinine": (
            BASELINE_FEATURES,
            make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
        ),
        "Logistic regression (all features)": (
            FEATURES,
            make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
        ),
        "Random forest": (
            FEATURES,
            RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=42),
        ),
    }


def cross_validate_models(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Grouped, stratified CV. Returns the mean and std of each metric per model."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = patient_groups(df)
    rows = []
    for name, (features, model) in make_models().items():
        for train_idx, test_idx in cv.split(df, df[TARGET], groups):
            train, test = df.iloc[train_idx], df.iloc[test_idx]
            model.fit(train[features], train[TARGET])
            proba = model.predict_proba(test[features])[:, 1]
            pred = (proba >= 0.5).astype(int)
            rows.append({
                "Model": name,
                "ROC-AUC": roc_auc_score(test[TARGET], proba),
                "PR-AUC": average_precision_score(test[TARGET], proba),
                "Recall": recall_score(test[TARGET], pred),
                "Precision": precision_score(test[TARGET], pred),
            })
    scores = pd.DataFrame(rows).groupby("Model", sort=False)
    return scores.mean().join(scores.std(), rsuffix=" std")


def train_final(df: pd.DataFrame):
    _, model = make_models()["Random forest"]
    return model.fit(df[FEATURES], df[TARGET])


def risk_band(probability: float) -> str:
    if probability >= 0.6:
        return "High"
    if probability >= 0.35:
        return "Moderate"
    return "Low"


def feature_importance(model) -> pd.Series:
    return pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

