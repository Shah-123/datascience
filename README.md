# Data Science Portfolio

[![CI](https://github.com/Shah-123/datascience/actions/workflows/ci.yml/badge.svg)](https://github.com/Shah-123/datascience/actions/workflows/ci.yml)

End-to-end data science projects: data cleaning, honest evaluation, interpretable results and deployable apps.
Each project has a documented notebook, a reusable Python module, a Streamlit app and its own README.

## Projects

| Project | Problem | Result (held-out data) | Stack |
|---|---|---|---|
| [💉 Polio campaign monitoring](icm_na_prediction) | Predict "Not Available" children per monitor per campaign and flag unusual reports | R² **0.62** / MAE 4.9 children on 5 unseen campaigns (baseline 0.42 / 6.2) | Poisson gradient boosting, time-based split, anomaly scoring |
| [🏏 ODI score predictor](cricket_score_prediction) | Predict the final first-innings score from the live match state | MAE **29.5 runs** on unseen innings (run-rate projection: 41.6) | XGBoost, grouped CV, **FastAPI + Docker** |
| [🏠 Property price estimator](house_price_prediction) | Estimate sale prices and rents in 5 Pakistani cities | Median error **~18%**, R² 0.80 (baseline 21–27%) | XGBoost on log price, 61k de-duplicated listings |
| [❤️ Heart failure risk](heart_failure_prediction) | Estimate mortality risk from clinical records | ROC-AUC **0.85** with grouped CV (2-feature baseline: 0.76) | Random forest, leakage and duplicate analysis |
| [📚 Exam score drivers](student_performance) | How much of a math score can background explain? | R² 0.29 from background; lunch type worth about 12 points | Linear models, effect sizes |
| [🎬 Netflix catalogue EDA](netflix_eda) | How has the catalogue grown, who is it for, and where does it come from? | 8,807 titles, interactive dashboard | pandas, Plotly |
| [📊 EDA Master](eda_app) | No-code exploratory analysis for any CSV/Excel file | Hypothesis tests, clustering, PCA/t-SNE, time series | Streamlit, SciPy, statsmodels |

## Evaluation lessons from these projects

The first versions of several projects reported excellent scores (R² 0.94–0.99, 99% accuracy) that didn't hold up.
Finding out why was the most valuable part of the work:

- **Duplicates across train and test.** 38% of house listings and 74% of heart-failure records were exact copies.
- **Target leakage.** An accounting identity rebuilt the ICM target exactly, and a follow-up-time column encoded the heart-failure outcome.
- **Correlated rows.** Balls from the same cricket innings were in both train and test. Splits are now by innings, campaign or patient group.
- **No baseline.** Every model is now compared with a simple, domain-sensible baseline.

## Repository layout

```
<project>/
├── README.md          problem, approach, results, how to run
├── analysis.ipynb     documented, executed analysis
├── <name>_model.py    reusable data and model code shared by the notebook, app and tests
├── app.py             Streamlit app
└── data/              dataset
tests/                 unit tests and smoke tests that run every app headlessly
```

## Run locally

```bash
git clone https://github.com/Shah-123/datascience.git
cd datascience
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

streamlit run icm_na_prediction/app.py        # or any other <project>/app.py
jupyter notebook                               # to open the analyses
```

Development checks (the same ones CI runs):

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
pytest
```

## Deploy the apps (free)

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **Create app**, pick this repository and branch, and set the main file path to e.g. `cricket_score_prediction/app.py`.
3. Repeat for each app, then add the live links to the table above and to your CV.

Apps that need a model train it on first start (10–15 seconds) and cache it.

## Contact

- GitHub: [Shah-123](https://github.com/Shah-123)
- Email: [shahkarahmad342@gmail.com](mailto:shahkarahmad342@gmail.com)
