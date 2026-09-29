import streamlit as st

import cricket_model as cm

st.set_page_config(page_title="ODI Score Predictor", page_icon="🏏", layout="centered")


@st.cache_resource(show_spinner="Loading model (the first run trains it, about 10 seconds)...")
def get_bundle():
    return cm.load_or_train()


def main():
    st.title("🏏 ODI First-Innings Score Predictor")
    st.markdown(
        "Predicts the final first-innings score (runs off the bat) from the current match situation. "
        "Trained on ball-by-ball ODI data for the top 10 teams. On innings it never saw in training, "
        "the model's average error is about **30 runs**, compared with 42 runs for the usual run-rate projection."
    )
    bundle = get_bundle()

    c1, c2, c3 = st.columns(3)
    venue = c1.selectbox("Venue", bundle["venues"])
    batting_team = c2.selectbox("Batting team", bundle["teams"])
    bowling_options = [t for t in bundle["teams"] if t != batting_team]
    bowling_team = c3.selectbox("Bowling team", bowling_options)

    c1, c2, c3 = st.columns(3)
    overs = c1.number_input("Overs completed", min_value=5, max_value=49, value=25, step=1)
    balls = c2.number_input("Balls into current over", min_value=0, max_value=5, value=0, step=1)
    wickets_fallen = c3.number_input("Wickets fallen", min_value=0, max_value=9, value=3, step=1)

    c1, c2 = st.columns(2)
    current_score = c1.number_input("Current score", min_value=0, max_value=450, value=130, step=1)
    last_five = c2.number_input("Runs in the last 5 overs", min_value=0, max_value=150, value=30, step=1)

    balls_bowled = overs * 6 + balls
    if last_five > current_score:
        st.warning("Runs in the last 5 overs can't be more than the current score.")
        st.stop()

    features = cm.make_input(
        venue, batting_team, bowling_team,
        balls_left=cm.BALLS_PER_INNINGS - balls_bowled,
        wickets_left=10 - wickets_fallen,
        current_score=current_score,
        last_five=last_five,
    )
    if st.button("Predict final score", type="primary"):
        predicted = max(float(bundle["model"].predict(features[cm.FEATURES])[0]), current_score)
        projection = float(cm.run_rate_projection(features)[0])
        c1, c2 = st.columns(2)
        c1.metric("Predicted final score", f"{predicted:.0f}", help="Typical error is about ±30 runs")
        c2.metric("Run-rate projection", f"{projection:.0f}", help="Current score + current run rate × overs left")
        st.caption(f"Current run rate: {features['current_run_rate'].iloc[0]:.2f}")

    st.markdown("---")
    st.caption("Scores are runs off the bat and exclude extras. Treat the prediction as a guide, not a certainty.")


if __name__ == "__main__":
    main()
