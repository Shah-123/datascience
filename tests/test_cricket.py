import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api
import cricket_model as cm


def make_balls():
    rows = []
    for final, venue in [(250, "A"), (250, "A"), (180, "B")]:
        for score, balls_left in [(20, 270), (21, 269), (22, 268)]:
            rows.append(
                {
                    cm.TARGET: final,
                    "venue": venue,
                    "batting_team": "X",
                    "bowling_team": "Y",
                    "balls_left": balls_left,
                    "current_score": score,
                }
            )
    return pd.DataFrame(rows)


def test_infer_innings_id_splits_on_score_reset_and_team_change():
    ids = cm.infer_innings_id(make_balls())
    # Innings 1 and 2 share teams, venue and final score but the score resets.
    assert ids.tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_make_input_derives_run_rate():
    row = cm.make_input("V", "X", "Y", balls_left=150, wickets_left=7, current_score=150, last_five=30)
    assert row["current_run_rate"].iloc[0] == pytest.approx(6.0)
    assert list(row.columns) == cm.FEATURES


def test_innings_split_keeps_innings_together():
    df = make_balls().assign(innings_id=lambda d: cm.infer_innings_id(d))
    train, test = cm.innings_split(df, test_size=0.34)
    assert not set(train["innings_id"]) & set(test["innings_id"])


@pytest.fixture(scope="module")
def client():
    df = cm.load_data().sample(20_000, random_state=0)
    bundle = {
        "model": cm.make_xgb_pipeline(n_estimators=20).fit(df[cm.FEATURES], df[cm.TARGET]),
        "venues": sorted(df["venue"].unique()),
        "teams": sorted(set(df["batting_team"]) | set(df["bowling_team"])),
    }
    original = cm.load_or_train
    cm.load_or_train = lambda *args, **kwargs: bundle
    try:
        with TestClient(api.app) as c:
            yield c, bundle
    finally:
        cm.load_or_train = original


def valid_request(bundle):
    return {
        "venue": bundle["venues"][0],
        "batting_team": "India",
        "bowling_team": "Pakistan",
        "balls_left": 150,
        "wickets_left": 7,
        "current_score": 130,
        "last_five": 35,
    }


def test_api_predicts(client):
    c, bundle = client
    response = c.post("/predict", json=valid_request(bundle))
    assert response.status_code == 200
    assert response.json()["predicted_score"] >= 130


@pytest.mark.parametrize(
    "change",
    [
        {"bowling_team": "India"},
        {"venue": "Nowhere"},
        {"last_five": 200},
        {"wickets_left": 0},
    ],
)
def test_api_rejects_invalid_input(client, change):
    c, bundle = client
    assert c.post("/predict", json={**valid_request(bundle), **change}).status_code == 422
