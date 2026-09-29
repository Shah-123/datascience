import plotly.express as px
import streamlit as st

import netflix_data as nd

st.set_page_config(page_title="Netflix Catalogue Analysis", page_icon="🎬", layout="wide")


@st.cache_data
def get_titles():
    return nd.load_titles()


def main():
    st.title("🎬 Netflix Catalogue Analysis")
    st.caption("8,807 movies and TV shows on Netflix, snapshot up to September 2021.")
    df = get_titles()

    st.sidebar.header("Filters")
    types = st.sidebar.multiselect("Type", ["Movie", "TV Show"], default=["Movie", "TV Show"])
    first, last = int(df["year_added"].min()), int(df["year_added"].max())
    years = st.sidebar.slider("Year added to Netflix", first, last, (2015, last))
    view = df[df["type"].isin(types) & df["year_added"].between(*years)]
    if view.empty:
        st.warning("No titles match the filters.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Titles", f"{len(view):,}")
    c2.metric("Movies", f"{(view['type'] == 'Movie').mean():.0%}")
    c3.metric("Countries", view["countries"].explode().nunique())
    c4.metric("Median years from release to Netflix", f"{view['years_to_netflix'].median():.0f}")

    tab1, tab2, tab3, tab4 = st.tabs(["Growth", "Audience & genres", "Countries", "Duration & people"])

    with tab1:
        added = view.groupby(["year_added", "type"]).size().reset_index(name="titles")
        fig = px.bar(added, x="year_added", y="titles", color="type", title="Titles added per year")
        st.plotly_chart(fig, width="stretch")
        st.caption("2021 only covers January to September.")
        fig = px.histogram(
            view,
            x="years_to_netflix",
            color="type",
            nbins=60,
            barmode="overlay",
            title="Years between release and arrival on Netflix",
            range_x=[-1, 40],
        )
        st.plotly_chart(fig, width="stretch")

    with tab2:
        c1, c2 = st.columns(2)
        share = view.groupby("type")["audience"].value_counts(normalize=True).mul(100)
        audience = share.reset_index(name="percent")
        fig = px.bar(
            audience,
            x="percent",
            y="type",
            color="audience",
            orientation="h",
            category_orders={"audience": nd.AUDIENCE_ORDER},
            title="Intended audience (share of titles, %)",
        )
        c1.plotly_chart(fig, width="stretch")
        genres = nd.explode_counts(view, "genres", 12).sort_values()
        fig = px.bar(
            x=genres.values,
            y=genres.index,
            orientation="h",
            labels={"x": "titles", "y": ""},
            title="Most common genres",
        )
        c2.plotly_chart(fig, width="stretch")

    with tab3:
        countries = nd.explode_counts(view, "countries", 15).sort_values()
        fig = px.bar(
            x=countries.values,
            y=countries.index,
            orientation="h",
            labels={"x": "titles", "y": ""},
            title="Countries producing the most titles (co-productions count for each country)",
        )
        st.plotly_chart(fig, width="stretch")
        top = nd.explode_counts(view, "countries", 5).index
        exploded = view.explode("countries")
        trend = exploded[exploded["countries"].isin(top)].groupby(["year_added", "countries"]).size()
        fig = px.line(
            trend.reset_index(name="titles"),
            x="year_added",
            y="titles",
            color="countries",
            markers=True,
            title="Titles added per year from the top 5 countries",
        )
        st.plotly_chart(fig, width="stretch")

    with tab4:
        c1, c2 = st.columns(2)
        fig = px.histogram(view.dropna(subset=["minutes"]), x="minutes", nbins=50, title="Movie length (minutes)")
        c1.plotly_chart(fig, width="stretch")
        seasons = view["seasons"].dropna().astype(int).value_counts().sort_index()
        fig = px.bar(
            x=seasons.index,
            y=seasons.values,
            labels={"x": "seasons", "y": "TV shows"},
            title="Number of seasons per TV show",
        )
        c2.plotly_chart(fig, width="stretch")
        c1, c2 = st.columns(2)
        directors = nd.explode_counts(view, "directors", 10).sort_values()
        c1.plotly_chart(
            px.bar(
                x=directors.values,
                y=directors.index,
                orientation="h",
                labels={"x": "titles", "y": ""},
                title="Directors with the most titles",
            ),
            width="stretch",
        )
        actors = nd.explode_counts(view, "cast_list", 10).sort_values()
        c2.plotly_chart(
            px.bar(
                x=actors.values,
                y=actors.index,
                orientation="h",
                labels={"x": "titles", "y": ""},
                title="Most frequent cast members",
            ),
            width="stretch",
        )


if __name__ == "__main__":
    main()
