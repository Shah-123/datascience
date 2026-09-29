# 💉 Polio Campaign Monitoring: Predicting "Not Available" Children

Predicts how many children each Independent Campaign Monitoring (ICM) monitor should record as **Not Available (NA)**
in a polio vaccination campaign, and flags reports that are far from that expectation for review.

## Problem

After every campaign, monitors check clusters of households and record each missed child and the reason.
*Not Available* (the child was away from home when the team visited) is one of the largest reasons.
Knowing where to expect NA children helps plan revisits, and unusual reports point supervisors to data worth checking.

**Data:** 120,378 cluster visits by 788 monitors across 34 campaigns (Mar 2020 – May 2024), aggregated to
10,908 monitor × campaign rows.

## Approach

1. **Leakage check.** In 99.999% of rows, `checked − vaccinated = NT + TVBMC + NA + ASLEEP + REFUSAL + OTHER_REASON`.
   Given those columns, a linear regression reaches R² = 1.0, so every `*_VAC` column and every other "reason"
   column is excluded.
2. **Features.** Workload in the current campaign (clusters, households, children checked, finger-marked and guest
   children, door-marking rate, share of high-risk/mobile clusters) plus the monitor's NA rate in *earlier* campaigns.
3. **Validation.** Time-based split: trained on campaigns up to Oct 2023 and tested on the 5 most recent campaigns.
4. **Models.** A baseline (average NA rate × children checked), a Poisson GLM and gradient boosting with Poisson loss.
5. **Anomaly score.** Pearson residual, rescaled by a robust spread estimate; |score| > 3 is flagged.

## Results (5 unseen campaigns, 1,420 monitor-campaigns)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (avg NA rate × children checked) | 6.18 | 8.06 | 0.42 |
| Poisson GLM | 5.39 | 6.96 | 0.57 |
| **Gradient boosting (Poisson)** | **4.89** | **6.57** | **0.62** |

The first version of this project reported R² = 0.94. That score came from the accounting identity and a
random split, and doesn't hold up. The notebook shows the leak step by step.

## Project structure

| File | Purpose |
|---|---|
| `icm_model.py` | Data loading, feature engineering, models, evaluation and anomaly scoring |
| `analysis.ipynb` | Full analysis: leakage check, EDA, model comparison, feature importance |
| `app.py` | Streamlit dashboard: performance on unseen campaigns, per-campaign review, CSV export |
| `data/icm_data.csv` | Cluster-level ICM export |

## Run it

From the repository root:

```bash
pip install -r requirements.txt
streamlit run icm_na_prediction/app.py
```

You can upload your own ICM export in the sidebar, as long as it uses the same column names.

## Limitations

- The monitor is used as a proxy for the area they cover. Area-level (`UCID`) history and seasonality are natural next steps.
- The anomaly threshold should be agreed with the programme team, based on how many reports they can review.
