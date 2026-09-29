import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

import student_model as sm

st.set_page_config(page_title="Exam Scores Analysis", page_icon="📚", layout="wide")


@st.cache_data
def get_data():
    return sm.load_data()


@st.cache_resource(show_spinner="Training models...")
def get_models():
    return sm.fit_models(get_data())


@st.cache_data(show_spinner="Cross-validating...")
def get_cv():
    return sm.cross_validate_models(get_data())


def predictor(df, models):
    sb = st.sidebar
    sb.header("🔮 Predict a math score")
    student = {f: sb.selectbox(f.capitalize(), sorted(df[f].unique())) for f in sm.BACKGROUND}
    know_scores = sb.checkbox("I know the reading and writing scores")
    if know_scores:
        student["reading score"] = sb.slider("Reading score", 0, 100, 70)
        student["writing score"] = sb.slider("Writing score", 0, 100, 70)
    name = "Background + reading/writing" if know_scores else "Background only"
    cv = get_cv()
    mae = cv.loc[f"Linear regression: {name}", "MAE"]
    prediction = models[name].predict(pd.DataFrame([student])[sm.FEATURE_SETS[name]])[0]
    sb.metric("Predicted math score", f"{min(max(prediction, 0), 100):.0f}", help=f"Average error ±{mae:.0f} points")
    sb.caption(f"Model: {name}. Average error ±{mae:.1f} points.")


def main():
    st.title("📚 What Drives Math Exam Scores?")
    st.markdown(
        "1,000 students' math, reading and writing scores with background information. "
        "The dataset is fictional (a widely used teaching dataset), so the findings illustrate the method rather than real schools."
    )
    df = get_data()
    models = get_models()
    predictor(df, models)

    tab1, tab2, tab3 = st.tabs(["Key drivers", "Model performance", "Explore"])
    with tab1:
        effects = sm.background_effects(models["Background only"])
        refs = sm.reference_groups(models["Background only"])
        fig, ax = plt.subplots(figsize=(8, 5))
        effects.plot.barh(ax=ax, color=["#d62728" if v < 0 else "#2ca02c" for v in effects])
        ax.axvline(0, color="black", lw=0.8)
        ax.set(xlabel="Math-score points compared with the reference group", title="Effect of each factor (other factors held fixed)")
        st.pyplot(fig)
        st.caption("Reference groups: " + "; ".join(f"{k} = {v}" for k, v in refs.items()))
        st.markdown(
            "- **Standard lunch** (a proxy for family income) is worth about **12 points** compared with free/reduced lunch.\n"
            "- **Not completing test preparation** costs about **5 points**.\n"
            "- Children of parents with **some high school** score about **7 points** below the associate's-degree group."
        )
    with tab2:
        st.markdown("5-fold cross-validation. Each model is trained once and reused.")
        st.dataframe(get_cv().style.format("{:.3f}"), width="stretch")
        st.markdown(
            "Background factors explain about **29%** of the variation in math scores. "
            "Adding reading and writing scores raises that to about **87%**, because the three scores are strongly correlated."
        )
    with tab3:
        st.dataframe(df.head(), width="stretch")
        c1, c2 = st.columns(2)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="lunch", y="math score", hue="test preparation course", ax=ax)
        ax.set_title("Math score by lunch type and test preparation")
        c1.pyplot(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        means = df.groupby("parental level of education")[["math score", "reading score", "writing score"]].mean()
        means.reindex(sm.EDUCATION_ORDER).plot.barh(ax=ax)
        ax.set(title="Average scores by parental education", xlabel="score")
        c2.pyplot(fig)
        st.markdown("**Correlation between the three scores**")
        st.dataframe(df[["math score", "reading score", "writing score"]].corr().style.format("{:.2f}"))


if __name__ == "__main__":
    main()
