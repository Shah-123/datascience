# 🏠 Pakistan Property Price Estimator

Estimates the **sale price** or **monthly rent** of a property in Karachi, Lahore, Islamabad, Rawalpindi or
Faisalabad from its neighbourhood, type, size (marla), bedrooms and bathrooms.

## Data and cleaning

99,499 listings scraped from a Pakistani property portal.

| Step | Rows |
|---|---|
| Raw listings | 99,499 |
| Remove exact duplicates | 61,641 |
| Remove zero area/price | 61,631 |
| Trim extreme 0.5% of prices per purpose | 61,141 |

**38% of the raw rows were exact duplicates.** The original cleaning missed them because the CSV's index column
made every row unique. Duplicates that appear in both train and test inflated the first version's R² to 0.985.

Other fixes:
- Sales and rentals are modelled separately, since their prices differ by about 200x.
- Neighbourhoods are keyed by city, because *DHA Defence* in Lahore is not *DHA Defence* in Karachi.
- The old app relabelled "Room" listings as "House"; that relabelling has been removed.

## Approach

- Target: log(price), with errors reported in rupees.
- Features: property type, city, city+neighbourhood (rare ones grouped), area, bedrooms, bathrooms.
- Models: baseline (median price per marla in the neighbourhood × area), Ridge regression, XGBoost.
- Evaluation: random 80/20 split within each purpose, after deduplication.

## Results (20% hold-out)

| Model | Sale: median error | Sale: R² | Rent: median error | Rent: R² |
|---|---|---|---|---|
| Baseline (price per marla) | 21.2% | −0.37 | 27.3% | 0.43 |
| Ridge regression | 18.2% | 0.60 | 18.9% | 0.76 |
| **XGBoost** | **17.9%** | **0.80** | **16.9%** | **0.80** |

Area and neighbourhood are the most important features. The app shows each estimate with this typical error as a range.

## Project structure

| File | Purpose |
|---|---|
| `house_model.py` | Cleaning, baseline, models, evaluation, save/load, PKR formatting |
| `train.py` | Prints the cleaning report and metrics, then saves `models/house_models.joblib` |
| `analysis.ipynb` | Data-quality checks, EDA, model comparison, error analysis, feature importance |
| `app.py` | Streamlit app: price estimator, charts, model performance, cleaning report |
| `data/house_prices_raw.csv` | Raw listings |

## Run it

From the repository root:

```bash
pip install -r requirements.txt
python house_price_prediction/train.py      # optional; the app trains on first run if needed
streamlit run house_price_prediction/app.py
```

## Limitations

- These are asking prices, not sale prices. The data is a single undated snapshot, so inflation isn't captured.
- There are no rental listings for Lahore.
- Re-posted listings with slightly different details may still make the scores a little optimistic.
