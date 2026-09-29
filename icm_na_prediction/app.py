import io

import plotly.express as px
import streamlit as st

import icm_model as icm

st.set_page_config(page_title="ICM Not-Available Children", page_icon="💉", layout="wide")


@st.cache_data(show_spinner="Preparing data...")
def prepare(file_bytes: bytes | None):
    source = io.BytesIO(file_bytes) if file_bytes else icm.DATA_PATH
    return icm.build_monitor_campaign_table(icm.load_raw(source))


@st.cache_resource(show_spinner="Training model...")
def train(table, n_test_campaigns: int):
    return icm.fit_and_evaluate(table, n_test_campaigns)


def main():
    st.title("💉 Polio Campaign Monitoring: Not-Available Children")
    st.markdown(
        "Predicts how many children an ICM monitor should record as **Not Available (NA)** in a "
        "campaign, using only the monitor's workload and their NA history from earlier campaigns. "
        "Reports far from the expected value are flagged for review."
    )

    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload an ICM export (CSV)", type=["csv"])
        st.caption("Leave empty to use the bundled dataset.")
        n_test = st.slider("Campaigns held out for testing", 2, 8, 5)
        threshold = st.slider("Anomaly threshold (robust z-score)", 2.0, 6.0, 3.0, 0.5)

    try:
        table = prepare(uploaded.getvalue() if uploaded else None)
    except ValueError as err:
        st.error(str(err))
        st.stop()

    result = train(table, n_test)

    st.subheader("Model performance on unseen campaigns")
    st.caption(
        f"Trained on the earliest campaigns and tested on the {n_test} most recent ones "
        f"({len(result.test):,} monitor-campaigns). Selected model: **{result.model_name}**."
    )
    best = result.metrics.loc[result.model_name]
    baseline = result.metrics.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (children)", f"{best.MAE:.2f}", f"{best.MAE - baseline.MAE:+.2f} vs baseline", delta_color="inverse")
    c2.metric("RMSE", f"{best.RMSE:.2f}", f"{best.RMSE - baseline.RMSE:+.2f} vs baseline", delta_color="inverse")
    c3.metric("R²", f"{best.R2:.3f}", f"{best.R2 - baseline.R2:+.3f} vs baseline")
    st.dataframe(result.metrics.style.format("{:.3f}"), width="stretch")

    fig = px.scatter(
        result.test, x="NA", y="Predicted_NA", opacity=0.4,
        labels={"NA": "Reported NA", "Predicted_NA": "Predicted NA"},
        title="Predicted vs reported NA (test campaigns)",
    )
    top = max(result.test["NA"].max(), result.test["Predicted_NA"].max())
    fig.add_shape(type="line", x0=0, y0=0, x1=top, y1=top, line=dict(dash="dash", color="red"))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Review a campaign")
    campaigns = (
        table.drop_duplicates("CAMP_ID").sort_values("campaign_order")[["CAMP_ID", "campaign_start"]]
    )
    labels = {
        row.CAMP_ID: f"Campaign {row.CAMP_ID} ({row.campaign_start:%b %Y})" for row in campaigns.itertuples()
    }
    camp_id = st.selectbox("Campaign", list(labels), index=len(labels) - 1, format_func=labels.get)
    frame = table[table["CAMP_ID"] == camp_id]
    # The final model has seen every campaign, so in-sample campaigns will look
    # better than the hold-out metrics above.
    flags = icm.flag_unusual_reports(frame, result.model.predict(frame[icm.FEATURES]), threshold)

    c1, c2, c3 = st.columns(3)
    c1.metric("Reported NA", f"{flags['NA'].sum():,}")
    c2.metric("Expected NA", f"{flags['Predicted_NA'].sum():,.0f}")
    c3.metric("Flagged monitors", int((flags["Flag"] != "").sum()))

    flagged = flags[flags["Flag"] != ""].sort_values("Anomaly_score", key=abs, ascending=False)
    if flagged.empty:
        st.success("No monitor is outside the anomaly threshold for this campaign.")
    else:
        st.dataframe(flagged, width="stretch", hide_index=True)

    with st.expander("All monitors in this campaign"):
        st.dataframe(flags, width="stretch", hide_index=True)
    st.download_button(
        "📥 Download campaign results (CSV)",
        flags.to_csv(index=False),
        file_name=f"icm_campaign_{camp_id}_predictions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
