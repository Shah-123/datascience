# ❤️ Heart Failure Mortality Risk

Estimates the risk that a heart-failure patient dies during follow-up, from 11 clinical measurements.
The project's main lesson is how easily a medical dataset produces a misleading "99% accuracy".

## What was wrong, and what changed

| Issue | Effect | Fix |
|---|---|---|
| 3,680 of 5,000 rows (73.6%) are exact duplicates | The same patients in train and test gave 99.2% accuracy | Drop duplicates (1,320 rows remain) |
| `time` (follow-up days) is in the features | Patients who die have short follow-up *because* they died; the value is unknown at prediction time | Excluded |
| The file is a synthetic expansion of ~300 real patients | Near-copies of a patient can still be in train and test | Stratified **group** CV (rows with the same age and sex stay in one fold) |
| Accuracy on 30% positives | Hides missed deaths | Report ROC-AUC, PR-AUC, recall and precision |

## Results (5-fold stratified group CV)

| Model | ROC-AUC | PR-AUC | Recall | Precision |
|---|---|---|---|---|
| Baseline: logistic regression on ejection fraction + serum creatinine | 0.76 | 0.64 | 0.70 | 0.57 |
| Logistic regression (all features) | 0.78 | 0.62 | 0.69 | 0.55 |
| **Random forest** | **0.85** | **0.75** | 0.69 | **0.63** |

The most important features are serum creatinine, ejection fraction, age and serum sodium. Because the data is partly
synthetic, treat these scores as optimistic.

## Project structure

| File | Purpose |
|---|---|
| `heart_model.py` | Loading and deduplication, models, grouped cross-validation, risk bands |
| `analysis.ipynb` | Data-quality checks, leakage demo, EDA, model comparison |
| `app.py` | Streamlit app: patient risk score, model performance, data exploration |

## Run it

From the repository root:

```bash
pip install -r requirements.txt
streamlit run heart_failure_prediction/app.py
```

> Educational project, not a medical device.
