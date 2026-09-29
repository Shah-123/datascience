# 📊 EDA Master: No-Code Exploratory Data Analysis

A Streamlit app that lets you upload any CSV or Excel file and explore, clean and analyse it without writing code.
Sample data is included, so it works as soon as it opens.

## Features

| Area | What you can do |
|---|---|
| **Overview** | Size, missing values, duplicates, data types, descriptive statistics |
| **Preprocessing** | Parse dates, handle missing values per column, drop duplicates/columns, IQR outlier removal, log/sqrt transforms, scaling, one-hot/label encoding |
| **Feature engineering** | Binning, date parts (year/month/weekday) |
| **Statistics** | Pearson/Spearman correlation; Welch t-test and Mann-Whitney U, ANOVA and Kruskal-Wallis, chi-square test of independence, each with a plain-language verdict |
| **Visualisation** | Histogram, box, violin, scatter, line, bar, pie, pair plot, optionally coloured by a category |
| **Text** | Word cloud, TextBlob sentiment |
| **Clustering** | K-Means / Agglomerative on standardised features, cluster profiles, PCA view |
| **Dimensionality reduction** | PCA with explained variance, t-SNE |
| **Time series** | Resampling (day/week/month/quarter) and seasonal decomposition |
| **Export** | Download the processed dataset |

## Run it

```bash
pip install -r requirements.txt
streamlit run eda_app/app.py
```

## What changed from the first version

- **Missing-value handling did nothing** under current pandas: `data[col].fillna(..., inplace=True)` modifies a copy. Values are now assigned back.
- Hypothesis tests used to compare two arbitrary columns and crashed on text. They now compare a numeric column across groups, and ANOVA and chi-square, previously "not implemented", work.
- Removed the "Automated EDA Report" menu option, which did nothing.
- Fixed crashes: correlation on text columns, pair plot rendering, t-SNE with more than 3 components, time series without a date column, clustering with missing values.
- Missing-value options only appear for columns that have missing values. Log/sqrt are only offered for non-negative columns.
- Removed CSS aimed at old Streamlit class names, and added sample data and export.
