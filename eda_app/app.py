"""EDA Master: upload a CSV/Excel file and explore, clean and analyse it without code."""

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "student_performance" / "data" / "exams.csv"
MAX_ROWS_TSNE = 2000

st.set_page_config(page_title="EDA Master", page_icon="📊", layout="wide", initial_sidebar_state="expanded")


def numeric_cols(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def datetime_cols(df):
    return df.select_dtypes(include="datetime").columns.tolist()


def categorical_cols(df):
    return [c for c in df.columns if c not in numeric_cols(df) and c not in datetime_cols(df)]


@st.cache_data
def read_file(name: str, content: bytes) -> pd.DataFrame:
    if name.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(content))
    return pd.read_csv(io.BytesIO(content))


def overview(data):
    st.subheader("🔎 Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(data):,}")
    c2.metric("Columns", data.shape[1])
    c3.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")
    with st.expander("Data sample"):
        st.dataframe(data.head(100), width="stretch")
    with st.expander("Column summary"):
        summary = pd.DataFrame(
            {
                "dtype": data.dtypes.astype(str),
                "missing": data.isna().sum(),
                "missing %": (data.isna().mean() * 100).round(1),
                "unique": data.nunique(),
            }
        )
        st.dataframe(summary, width="stretch")
    with st.expander("Descriptive statistics"):
        st.dataframe(data.describe(include="all").T, width="stretch")


def preprocess(data):
    st.subheader("🛠️ Preprocessing")

    with st.expander("Convert columns to dates"):
        candidates = categorical_cols(data)
        for col in st.multiselect("Columns to parse as dates", candidates, key="to_date"):
            data[col] = pd.to_datetime(data[col], errors="coerce", format="mixed")
            st.caption(f"{col}: {data[col].isna().sum()} values could not be parsed.")

    with st.expander("Handle missing values"):
        missing = [c for c in data.columns if data[c].isna().any()]
        if not missing:
            st.success("No missing values.")
        if st.checkbox("Drop duplicate rows", key="drop_dups"):
            data = data.drop_duplicates()
        for col in missing:
            is_numeric = col in numeric_cols(data)
            options = (
                ["Keep", "Drop rows", "Mean", "Median", "Mode", "Interpolate"]
                if is_numeric
                else ["Keep", "Drop rows", "Mode", "Fill with 'Unknown'"]
            )
            method = st.selectbox(f"{col} ({data[col].isna().sum()} missing)", options, key=f"missing_{col}")
            # Assign back instead of inplace=True, which does nothing on a
            # column under pandas copy-on-write.
            if method == "Drop rows":
                data = data.dropna(subset=[col])
            elif method == "Mean":
                data[col] = data[col].fillna(data[col].mean())
            elif method == "Median":
                data[col] = data[col].fillna(data[col].median())
            elif method == "Mode":
                data[col] = data[col].fillna(data[col].mode().iloc[0])
            elif method == "Interpolate":
                data[col] = data[col].interpolate()
            elif method == "Fill with 'Unknown'":
                data[col] = data[col].fillna("Unknown")

    with st.expander("Drop columns"):
        to_drop = st.multiselect("Columns to drop", data.columns, key="drop_cols")
        data = data.drop(columns=to_drop)

    with st.expander("Remove outliers (1.5 × IQR rule)"):
        for col in st.multiselect("Columns", numeric_cols(data), key="outliers"):
            q1, q3 = data[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            before = len(data)
            data = data[data[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr) | data[col].isna()]
            st.caption(f"{col}: removed {before - len(data)} rows.")

    with st.expander("Transform skewed columns"):
        for col in numeric_cols(data):
            if data[col].min() < 0:
                continue  # log/sqrt are undefined for negative values
            choice = st.selectbox(f"Transform {col}", ["None", "Log (log1p)", "Square root"], key=f"tf_{col}")
            if choice == "Log (log1p)":
                data[col] = np.log1p(data[col])
            elif choice == "Square root":
                data[col] = np.sqrt(data[col])
        st.caption("Columns with negative values are not listed.")

    with st.expander("Scale and encode"):
        if st.checkbox("Standard-scale numeric columns", key="scale"):
            cols = numeric_cols(data)
            data[cols] = StandardScaler().fit_transform(data[cols])
        method = st.selectbox("Encode categorical columns", ["None", "One-hot", "Label"], key="encode")
        cats = categorical_cols(data)
        if method == "One-hot":
            data = pd.get_dummies(data, columns=cats, dtype=int)
        elif method == "Label":
            for col in cats:
                data[col] = LabelEncoder().fit_transform(data[col].astype(str))
    return data


def feature_engineering(data):
    st.subheader("🔧 Feature engineering")
    with st.expander("Bin numeric columns"):
        for col in st.multiselect("Columns", numeric_cols(data), key="bin_cols"):
            bins = st.slider(f"Bins for {col}", 2, 10, 4, key=f"bins_{col}")
            data[f"{col}_binned"] = pd.cut(data[col], bins=bins).astype(str)
    with st.expander("Extract date parts"):
        dates = datetime_cols(data)
        if not dates:
            st.caption("No date columns. Convert one under Preprocessing → Convert columns to dates.")
        for col in st.multiselect("Columns", dates, key="date_parts"):
            data[f"{col}_year"] = data[col].dt.year
            data[f"{col}_month"] = data[col].dt.month
            data[f"{col}_dayofweek"] = data[col].dt.dayofweek
    return data


def statistical_analysis(data):
    st.subheader("📈 Statistical analysis")
    nums, cats = numeric_cols(data), categorical_cols(data)
    with st.expander("Correlation matrix", expanded=True):
        if len(nums) < 2:
            st.info("Needs at least two numeric columns.")
        else:
            method = st.radio("Method", ["pearson", "spearman"], horizontal=True, key="corr_method")
            fig, ax = plt.subplots(figsize=(min(2 + len(nums), 14), min(1.5 + 0.8 * len(nums), 12)))
            sns.heatmap(
                data[nums].corr(method=method),
                annot=len(nums) <= 15,
                fmt=".2f",
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                ax=ax,
            )
            st.pyplot(fig)

    with st.expander("Hypothesis tests", expanded=True):
        test = st.selectbox(
            "Test",
            [
                "Compare two groups (t-test / Mann-Whitney U)",
                "Compare several groups (ANOVA / Kruskal-Wallis)",
                "Association between two categorical columns (chi-square)",
            ],
            key="test",
        )
        if test.startswith("Association"):
            if len(cats) < 2:
                st.info("Needs at least two categorical columns.")
                return
            a = st.selectbox("First column", cats, key="chi_a")
            b = st.selectbox("Second column", [c for c in cats if c != a], key="chi_b")
            table = pd.crosstab(data[a], data[b])
            chi2, p, dof, _ = stats.chi2_contingency(table)
            st.dataframe(table, width="stretch")
            report(p, f"χ² = {chi2:.2f}, degrees of freedom = {dof}", f"{a} and {b} are associated")
            return

        if not nums or not cats:
            st.info("Needs a numeric column and a categorical grouping column.")
            return
        value = st.selectbox("Numeric column", nums, key="test_value")
        group = st.selectbox("Group by", cats, key="test_group")
        groups = {k: g[value].dropna() for k, g in data.groupby(group) if g[value].notna().sum() >= 2}
        if len(groups) < 2:
            st.info("Needs at least two groups with two or more values.")
            return
        if test.startswith("Compare two"):
            pair = st.multiselect(
                "Pick two groups", list(groups), default=list(groups)[:2], max_selections=2, key="test_pair"
            )
            if len(pair) != 2:
                st.info("Pick exactly two groups.")
                return
            g1, g2 = pair
            t = stats.ttest_ind(groups[g1], groups[g2], equal_var=False)
            u = stats.mannwhitneyu(groups[g1], groups[g2])
            report(t.pvalue, f"Welch t = {t.statistic:.3f}", f"mean {value} differs between {g1} and {g2}")
            report(
                u.pvalue, f"Mann-Whitney U = {u.statistic:.0f} (no normality assumption)", "the distributions differ"
            )
        else:
            f = stats.f_oneway(*groups.values())
            k = stats.kruskal(*groups.values())
            report(f.pvalue, f"ANOVA F = {f.statistic:.3f}", f"mean {value} differs across {group} groups")
            report(
                k.pvalue, f"Kruskal-Wallis H = {k.statistic:.3f} (no normality assumption)", "the distributions differ"
            )
        fig = px.box(data, x=group, y=value, points=False, title=f"{value} by {group}")
        st.plotly_chart(fig, width="stretch")


def report(p, detail, finding, alpha=0.05):
    verdict = f"✅ Significant at α = {alpha}: {finding}." if p < alpha else f"❌ Not significant at α = {alpha}."
    st.write(f"{detail}, p-value = {p:.4g}. {verdict}")


def visualization(data):
    st.subheader("🎨 Visualisation")
    nums, cols = numeric_cols(data), data.columns.tolist()
    kind = st.selectbox(
        "Plot type",
        [
            "Histogram",
            "Box plot",
            "Violin plot",
            "Scatter plot",
            "Line plot",
            "Bar chart (counts)",
            "Pie chart",
            "Pair plot",
        ],
        key="plot_kind",
    )
    color = st.selectbox("Colour by (optional)", ["None"] + categorical_cols(data), key="plot_color")
    color = None if color == "None" else color
    if kind in ("Histogram", "Box plot", "Violin plot"):
        col = st.selectbox("Column", nums or cols, key="plot_col")
        fn = {"Histogram": px.histogram, "Box plot": px.box, "Violin plot": px.violin}[kind]
        fig = fn(data, x=col, color=color) if kind == "Histogram" else fn(data, y=col, x=color, color=color)
    elif kind in ("Scatter plot", "Line plot"):
        x = st.selectbox("X axis", cols, key="plot_x")
        y = st.selectbox("Y axis", nums or cols, key="plot_y")
        fn = px.scatter if kind == "Scatter plot" else px.line
        fig = fn(data.sort_values(x) if kind == "Line plot" else data, x=x, y=y, color=color)
    elif kind in ("Bar chart (counts)", "Pie chart"):
        col = st.selectbox("Column", cols, key="plot_col_cat")
        counts = data[col].astype(str).value_counts()
        if len(counts) > 15:
            counts = pd.concat([counts.head(14), pd.Series({"Other": counts.iloc[14:].sum()})])
        counts = counts.rename_axis(col).reset_index(name="count")
        fig = px.bar(counts, x=col, y="count") if kind.startswith("Bar") else px.pie(counts, names=col, values="count")
    else:
        chosen = st.multiselect("Columns (up to 6)", nums, default=nums[:4], max_selections=6, key="pair_cols")
        if len(chosen) < 2:
            st.info("Pick at least two numeric columns.")
            return
        sample = data.sample(min(len(data), 2000), random_state=0)
        grid = sns.pairplot(sample, vars=chosen, hue=color, corner=True)
        st.pyplot(grid.figure)
        return
    st.plotly_chart(fig, width="stretch")


def text_analysis(data):
    st.subheader("🗣️ Text analysis")
    texts = categorical_cols(data)
    if not texts:
        st.info("No text columns in this dataset.")
        return
    col = st.selectbox("Text column", texts, key="text_col")
    text = " ".join(data[col].dropna().astype(str))
    if not text.strip():
        st.info("The column is empty.")
        return
    from wordcloud import WordCloud

    cloud = WordCloud(width=900, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    if st.checkbox("Run sentiment analysis (TextBlob polarity)", key="sentiment"):
        from textblob import TextBlob

        polarity = data[col].dropna().astype(str).map(lambda t: TextBlob(t).sentiment.polarity)
        st.plotly_chart(
            px.histogram(
                polarity,
                nbins=40,
                labels={"value": "polarity (-1 negative to +1 positive)"},
                title=f"Sentiment of {col}",
            ),
            width="stretch",
        )


def clustering(data):
    st.subheader("🔍 Clustering")
    nums = numeric_cols(data)
    features = st.multiselect("Features", nums, default=nums[:5], key="cluster_features")
    if len(features) < 2:
        st.info("Pick at least two numeric columns.")
        return
    X = data[features].dropna()
    if len(X) < 10:
        st.info("Not enough complete rows.")
        return
    n = st.slider("Number of clusters", 2, 10, 3, key="n_clusters")
    method = st.selectbox("Method", ["K-Means", "Agglomerative"], key="cluster_method")
    scaled = StandardScaler().fit_transform(X)
    model = (
        KMeans(n_clusters=n, n_init=10, random_state=42)
        if method == "K-Means"
        else AgglomerativeClustering(n_clusters=n)
    )
    labels = pd.Series(model.fit_predict(scaled), index=X.index).astype(str)
    st.caption(f"Features are standardised first. {len(data) - len(X)} rows with missing values were skipped.")
    st.dataframe(X.groupby(labels).mean().round(2).assign(size=labels.value_counts()), width="stretch")
    coords = PCA(n_components=2).fit_transform(scaled)
    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        color=labels,
        labels={"x": "PC 1", "y": "PC 2", "color": "cluster"},
        title="Clusters projected onto the first two principal components",
    )
    st.plotly_chart(fig, width="stretch")


def dimensionality_reduction(data):
    st.subheader("🔻 Dimensionality reduction")
    nums = numeric_cols(data)
    if len(nums) < 3:
        st.info("Needs at least three numeric columns.")
        return
    X = data[nums].dropna()
    method = st.selectbox("Method", ["PCA", "t-SNE"], key="dr_method")
    color = st.selectbox("Colour by (optional)", ["None"] + categorical_cols(data), key="dr_color")
    scaled = StandardScaler().fit_transform(X)
    if method == "PCA":
        pca = PCA().fit(scaled)
        explained = pd.Series(pca.explained_variance_ratio_.cumsum(), index=range(1, len(nums) + 1))
        st.plotly_chart(
            px.line(
                explained,
                markers=True,
                labels={"index": "components", "value": "cumulative variance explained"},
                title="Explained variance",
            ),
            width="stretch",
        )
        coords = pca.transform(scaled)[:, :2]
        index = X.index
    else:
        index = X.sample(min(len(X), MAX_ROWS_TSNE), random_state=0).index
        coords = TSNE(n_components=2, random_state=42, init="pca").fit_transform(scaled[X.index.get_indexer(index)])
        st.caption(f"t-SNE uses up to {MAX_ROWS_TSNE:,} sampled rows.")
    hue = None if color == "None" else data.loc[index, color].astype(str)
    fig = px.scatter(
        x=coords[:, 0], y=coords[:, 1], color=hue, labels={"x": "component 1", "y": "component 2"}, opacity=0.6
    )
    st.plotly_chart(fig, width="stretch")


def time_series(data):
    st.subheader("⏰ Time series")
    dates, nums = datetime_cols(data), numeric_cols(data)
    if not dates or not nums:
        st.info("Needs a date column (see Preprocessing → Convert columns to dates) and a numeric column.")
        return
    date_col = st.selectbox("Date column", dates, key="ts_date")
    value_col = st.selectbox("Value column", nums, key="ts_value")
    freq = st.selectbox(
        "Aggregate by",
        {"D": "Day", "W": "Week", "MS": "Month", "QS": "Quarter"}.items(),
        index=2,
        format_func=lambda kv: kv[1],
        key="ts_freq",
    )[0]
    how = st.radio("Aggregation", ["mean", "sum"], horizontal=True, key="ts_how")
    series = data.set_index(date_col)[value_col].resample(freq).agg(how).dropna()
    st.plotly_chart(px.line(series, title=f"{how} of {value_col} per period"), width="stretch")
    period = st.number_input("Seasonal period (in periods)", 2, 365, 12 if freq == "MS" else 7, key="ts_period")
    if len(series) < 2 * period:
        st.info(f"Seasonal decomposition needs at least {2 * period} periods; there are {len(series)}.")
        return
    fig = sm.tsa.seasonal_decompose(series, model="additive", period=int(period)).plot()
    fig.set_size_inches(10, 7)
    st.pyplot(fig)


ANALYSES = {
    "Statistical analysis": statistical_analysis,
    "Visualisation": visualization,
    "Text analysis": text_analysis,
    "Clustering": clustering,
    "Dimensionality reduction": dimensionality_reduction,
    "Time series": time_series,
}


def main():
    st.title("📊 EDA Master")
    st.markdown("Upload a dataset to explore, clean and analyse it without writing code.")

    st.sidebar.header("Data")
    uploaded = st.sidebar.file_uploader("CSV or Excel file", type=["csv", "xlsx"])
    use_sample = st.sidebar.checkbox("Use sample data (student exam scores)", value=uploaded is None)
    if uploaded is not None:
        data = read_file(uploaded.name, uploaded.getvalue())
    elif use_sample:
        data = pd.read_csv(SAMPLE_PATH)
    else:
        st.info("👈 Upload a file or tick 'Use sample data' to begin.")
        return

    data = data.copy()
    overview(data)
    data = preprocess(data)
    data = feature_engineering(data)
    if data.empty:
        st.warning("No rows left after preprocessing.")
        return

    analysis = st.sidebar.radio("Analysis", list(ANALYSES))
    ANALYSES[analysis](data)
    st.sidebar.download_button("📥 Download processed data", data.to_csv(index=False), "processed_data.csv", "text/csv")


if __name__ == "__main__":
    main()
