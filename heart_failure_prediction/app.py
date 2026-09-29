import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

import heart_model as hm

st.set_page_config(page_title="Heart Failure Risk", page_icon="❤️", layout="wide")


@st.cache_data
def get_data():
    return hm.load_data()


@st.cache_resource(show_spinner="Training model...")
def get_model():
    return hm.train_final(get_data())


@st.cache_data(show_spinner="Cross-validating models...")
def get_cv_metrics():
    return hm.cross_validate_models(get_data())


def patient_inputs():
    sb = st.sidebar
    sb.header("Patient data")
    yes_no = lambda label, help=None: 1 if sb.selectbox(label, ("No", "Yes"), help=help) == "Yes" else 0  # noqa: E731
    return {
        "age": sb.slider("Age (years)", 40, 95, 60),
        "sex": 1 if sb.selectbox("Sex", ("Female", "Male")) == "Male" else 0,
        "anaemia": yes_no("Anaemia", "Decrease of red blood cells or haemoglobin"),
        "diabetes": yes_no("Diabetes"),
        "high_blood_pressure": yes_no("High blood pressure"),
        "smoking": yes_no("Smoking"),
        "ejection_fraction": sb.slider(
            "Ejection fraction (%)", 14, 80, 38, help="Percentage of blood leaving the heart at each contraction"
        ),
        "serum_creatinine": sb.slider("Serum creatinine (mg/dL)", 0.5, 9.5, 1.1, 0.1),
        "serum_sodium": sb.slider("Serum sodium (mEq/L)", 113, 148, 137),
        "creatinine_phosphokinase": sb.number_input("CPK enzyme (mcg/L)", 20, 8000, 250),
        "platelets": sb.number_input("Platelets (per mL)", 25_000, 850_000, 263_000, step=1_000),
    }


def main():
    st.title("❤️ Heart Failure Mortality Risk")
    st.warning("Educational project, not a medical device. Don't use it for clinical decisions.")
    df = get_data()
    model = get_model()
    patient = pd.DataFrame([patient_inputs()])[hm.FEATURES]

    risk = float(model.predict_proba(patient)[0, 1])
    c1, c2 = st.columns(2)
    c1.metric("Model risk score", f"{risk:.0%}")
    c2.metric("Risk band", hm.risk_band(risk))
    st.caption(
        "The score comes from a class-balanced random forest. Treat it as a relative ranking, not a calibrated probability."
    )

    tab1, tab2 = st.tabs(["Model performance", "Data exploration"])
    with tab1:
        st.markdown(
            "5-fold **stratified group** cross-validation on the de-duplicated data (1,320 rows). Rows with the same "
            "age and sex stay in the same fold, and the follow-up `time` column is excluded because it leaks the outcome."
        )
        cv = get_cv_metrics()
        st.dataframe(cv[["ROC-AUC", "PR-AUC", "Recall", "Precision"]].style.format("{:.3f}"), width="stretch")
        st.info(
            "This dataset was synthetically expanded from about 300 real patients, so even these scores are likely "
            "optimistic. The original project reported 99% accuracy because duplicate rows were in both train and test."
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        hm.feature_importance(model).sort_values().plot.barh(ax=ax, title="Random forest feature importance")
        st.pyplot(fig)

    with tab2:
        c1, c2 = st.columns(2)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x=hm.TARGET, y="ejection_fraction", ax=ax)
        ax.set(xlabel="Death event (0 = survived, 1 = died)", title="Ejection fraction by outcome")
        c1.pyplot(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x=hm.TARGET, y="serum_creatinine", ax=ax)
        ax.set(xlabel="Death event (0 = survived, 1 = died)", title="Serum creatinine by outcome", yscale="log")
        c2.pyplot(fig)

        feature = st.selectbox("Death rate by", ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"])
        rates = df.groupby(feature)[hm.TARGET].mean().rename(index={0: "No", 1: "Yes"})
        if feature == "sex":
            rates = rates.rename(index={"No": "Female", "Yes": "Male"})
        st.bar_chart(rates, y_label="Death rate")

        if st.checkbox("Show correlation heatmap"):
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df.drop(columns=hm.LEAKY).corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)


if __name__ == "__main__":
    main()
