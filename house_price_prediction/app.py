import plotly.express as px
import streamlit as st

import house_model as hm

st.set_page_config(page_title="Pakistan Property Prices", page_icon="🏠", layout="wide")


@st.cache_data(show_spinner="Loading listings...")
def get_listings():
    return hm.clean(hm.load_raw())


@st.cache_resource(show_spinner="Loading models (the first run trains them, about 15 seconds)...")
def get_bundle():
    return hm.load_or_train()


@st.cache_data(show_spinner="Evaluating models...")
def get_metrics(purpose: str):
    df, _ = get_listings()
    return hm.compare_models(df, purpose)


def predictor(bundle):
    st.sidebar.header("🔮 Estimate a price")
    purpose = st.sidebar.radio("Purpose", hm.PURPOSES, horizontal=True)
    city = st.sidebar.selectbox("City", list(bundle["locations"][purpose]))
    location = st.sidebar.selectbox("Location", bundle["locations"][purpose][city])
    property_type = st.sidebar.selectbox("Property type", bundle["property_types"])
    area = st.sidebar.number_input("Area (marla)", min_value=1.0, max_value=200.0, value=10.0, step=0.5)
    bedrooms = st.sidebar.slider("Bedrooms", 0, 10, 3)
    baths = st.sidebar.slider("Bathrooms", 0, 10, 3)

    if st.sidebar.button("Estimate", type="primary"):
        features = hm.make_input(property_type, city, location, area, bedrooms, baths)
        price = float(bundle["models"][purpose].predict(features[hm.FEATURES])[0])
        error = bundle["typical_error"][purpose]
        label = "Estimated monthly rent" if purpose == "For Rent" else "Estimated sale price"
        st.sidebar.metric(label, hm.format_pkr(price))
        st.sidebar.caption(
            f"Typical error is ±{error:.0%}, so a likely range is "
            f"{hm.format_pkr(price * (1 - error))} – {hm.format_pkr(price * (1 + error))}."
        )


def explore(df):
    purpose = st.radio("Listings", hm.PURPOSES, horizontal=True, key="explore_purpose")
    subset = df[df["purpose"] == purpose].assign(price_per_marla=lambda d: d["price"] / d["Area_in_Marla"])
    unit = "monthly rent" if purpose == "For Rent" else "sale price"

    c1, c2 = st.columns(2)
    fig = px.histogram(subset, x="price", nbins=60, title=f"Distribution of {unit} (PKR)")
    c1.plotly_chart(fig, width="stretch")
    by_city = subset.groupby("city")["price_per_marla"].median().sort_values().reset_index()
    fig = px.bar(by_city, x="price_per_marla", y="city", orientation="h",
                 title=f"Median {unit} per marla by city (PKR)")
    c2.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    fig = px.box(subset, x="property_type", y="price", points=False, log_y=True,
                 title=f"{unit.capitalize()} by property type (log scale)")
    c1.plotly_chart(fig, width="stretch")
    counts = subset.groupby(["city", "property_type"]).size().reset_index(name="listings")
    fig = px.bar(counts, x="city", y="listings", color="property_type", title="Listings by city and property type")
    c2.plotly_chart(fig, width="stretch")


def performance():
    purpose = st.radio("Model", hm.PURPOSES, horizontal=True, key="perf_purpose")
    metrics, test = get_metrics(purpose)
    st.markdown(
        "Evaluated on a random 20% of listings that the models never saw, after exact duplicates were removed. "
        "Models are trained on log(price), and errors are reported in rupees."
    )
    st.dataframe(
        metrics.style.format({"MAE (PKR)": "{:,.0f}", "MAPE": "{:.1%}", "Median APE": "{:.1%}", "R2": "{:.3f}"}),
        width="stretch",
    )
    fig = px.scatter(test, x="price", y="predicted", opacity=0.25, log_x=True, log_y=True,
                     labels={"price": "Actual (PKR)", "predicted": "Predicted (PKR)"},
                     title="XGBoost: actual vs predicted (log scales)")
    lo, hi = test["price"].min(), test["price"].max()
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="red"))
    st.plotly_chart(fig, width="stretch")


def main():
    st.title("🏠 Pakistan Property Prices")
    st.caption("Sale prices and monthly rents from online listings in Karachi, Lahore, Islamabad, Rawalpindi and Faisalabad.")
    df, report = get_listings()
    predictor(get_bundle())

    tab1, tab2, tab3 = st.tabs(["Explore", "Model performance", "Data cleaning"])
    with tab1:
        explore(df)
    with tab2:
        performance()
    with tab3:
        st.markdown("Rows remaining after each cleaning step:")
        st.table({"Step": list(report), "Rows": [f"{n:,}" for n in report.values()]})


if __name__ == "__main__":
    main()
