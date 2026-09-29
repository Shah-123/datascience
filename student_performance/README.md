# 📚 What Drives Math Exam Scores?

Analyses 1,000 students' exam scores to answer two questions:

1. **Early warning:** how much of a math score can be predicted from background alone (gender, ethnic group, parental education, lunch type, test preparation)?
2. **Drivers:** which factors matter, and by how many points?

> The dataset is a fictional teaching dataset, so the findings illustrate the method rather than real schools.

## Results (5-fold cross-validation)

| Model | MAE (points) | R² |
|---|---|---|
| Baseline (predict the mean) | 12.3 | 0.00 |
| Linear regression: background only | 10.3 | 0.29 |
| Linear regression: background + reading/writing | 4.4 | 0.87 |

The high R² with reading and writing scores comes from the three scores being strongly correlated (r ≈ 0.8–0.95), so
that model mainly checks consistency. Background alone explains less than a third of the variation.

**Effect sizes** (background-only model, compared with a reference group):
- Standard lunch (a proxy for family income): about **+12 points**
- No test preparation: about **−5 points**
- Parent with *some high school*: about **−7 points** compared with *associate's degree*

## Project structure

| File | Purpose |
|---|---|
| `student_model.py` | Loading, both models, cross-validation, effect sizes |
| `analysis.ipynb` | EDA, model comparison, interpretation |
| `app.py` | Streamlit app: key drivers, model performance, exploration, score predictor |

## Run it

```bash
pip install -r requirements.txt
streamlit run student_performance/app.py
```

## What changed from the first version

- The app trained and evaluated the model twice on every run. It now trains once and caches the result.
- The app description said the data came from three US high schools. It's fictional, and the text now says so.
- Added a background-only model and effect sizes, so the project answers a real question instead of only predicting math from reading and writing.
