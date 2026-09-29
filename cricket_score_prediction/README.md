# 🏏 ODI First-Innings Score Predictor

Predicts the final first-innings score of a one-day international from the live match situation. It includes
a Streamlit app, a FastAPI REST service and a Dockerfile.

## Problem

Broadcasters show a "projected score" based only on the current run rate. This project learns from 1,234 past
innings (323,930 balls, top 10 teams, 51 venues) to also account for wickets in hand, venue and teams.

## Approach

- **Features:** venue, batting team, bowling team, balls left, wickets left, current score, current run rate, runs in the last 5 overs.
- **Innings-aware validation:** consecutive balls of one innings are almost identical, so train/test splits and
  cross-validation are grouped by innings (`GroupShuffleSplit`, `GroupKFold`). The export had lost `match_id`,
  so innings boundaries are recovered from row order. The check passes for 100% of innings.
- **Models:** run-rate projection (baseline), Ridge regression, XGBoost tuned with 5-fold grouped CV.

## Results (20% of innings held out)

| Model | MAE (runs) | RMSE | R² |
|---|---|---|---|
| Run-rate projection (baseline) | 41.6 | 54.8 | 0.16 |
| Ridge regression | 30.7 | 41.0 | 0.53 |
| **XGBoost (regularised)** | **29.5** | **40.5** | **0.54** |

The error drops from about 45 runs in the first 10 overs to about 15 runs after 40 overs.

The first version of this project reported R² = 0.96 from a random row split, because balls from the same
innings were in both train and test. With the same settings on unseen innings, R² is 0.44. The notebook walks through this.

## Project structure

| File | Purpose |
|---|---|
| `cricket_model.py` | Data loading, innings recovery, pipelines, evaluation, model save/load |
| `train.py` | Evaluates the models and saves `models/cricket_model.joblib` |
| `analysis.ipynb` | EDA, split comparison, grouped CV tuning, error by innings stage, feature importance |
| `app.py` | Streamlit app (the run rate is calculated automatically, so inputs stay consistent) |
| `api.py` | FastAPI service: `GET /options`, `POST /predict` |
| `Dockerfile` | Container for the API |

## Run it

From the repository root:

```bash
pip install -r requirements.txt
python cricket_score_prediction/train.py          # optional; the app trains on first run if needed
streamlit run cricket_score_prediction/app.py
```

API:

```bash
uvicorn api:app --app-dir cricket_score_prediction
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "venue": "Gaddafi Stadium", "batting_team": "Pakistan", "bowling_team": "India",
  "balls_left": 150, "wickets_left": 7, "current_score": 130, "last_five": 35}'
```

Docker:

```bash
docker build -f cricket_score_prediction/Dockerfile -t odi-score-api .
docker run -p 8000:8000 odi-score-api       # interactive docs at http://localhost:8000/docs
```

## Limitations

- Only runs off the bat (extras aren't included), first innings only, top 10 teams.
- A time-based split (older seasons → recent seasons) would be even more realistic, but it needs the match dates from the raw data.
