"""Load and clean the Netflix titles catalogue (snapshot up to September 2021)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "netflix_titles.csv"
SNAPSHOT_END = pd.Timestamp("2021-09-30")

# Content ratings grouped by intended audience. Ratings describe *who a title
# is for*, not how good it is, so they are never averaged as a quality score.
AUDIENCE = {
    "TV-Y": "Kids", "TV-Y7": "Kids", "TV-Y7-FV": "Kids", "TV-G": "Kids", "G": "Kids",
    "PG": "Older kids", "TV-PG": "Older kids",
    "PG-13": "Teens", "TV-14": "Teens",
    "R": "Adults", "TV-MA": "Adults", "NC-17": "Adults", "A": "Adults",
    "NR": "Unrated", "UR": "Unrated",
}
AUDIENCE_ORDER = ["Kids", "Older kids", "Teens", "Adults", "Unrated"]


def split_list(value) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def load_titles(path=DATA_PATH) -> pd.DataFrame:
    # The file is UTF-8; reading it as ISO-8859-1 garbled accented names.
    df = pd.read_csv(path, encoding="utf-8")
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

    # Three rows have the duration typed into the rating column.
    shifted = df["rating"].str.contains("min", na=False)
    df.loc[shifted, "duration"] = df.loc[shifted, "rating"]
    df.loc[shifted, "rating"] = pd.NA

    df["date_added"] = pd.to_datetime(df["date_added"].str.strip(), format="%B %d, %Y", errors="coerce")
    # The snapshot runs to September 2021; two rows were appended in 2024 and
    # would show up as a misleading spike after a two-year gap.
    df = df[~(df["date_added"] > SNAPSHOT_END)].reset_index(drop=True)
    df["year_added"] = df["date_added"].dt.year.astype("Int64")
    df["years_to_netflix"] = df["year_added"] - df["release_year"]
    df["audience"] = df["rating"].map(AUDIENCE).fillna("Unrated")

    number = pd.to_numeric(df["duration"].str.extract(r"(\d+)")[0], errors="coerce")
    df["minutes"] = number.where(df["type"] == "Movie")
    df["seasons"] = number.where(df["type"] == "TV Show")

    for col, new in [("listed_in", "genres"), ("country", "countries"), ("cast", "cast_list"), ("director", "directors")]:
        df[new] = df[col].map(split_list)
    df["main_country"] = df["countries"].str[0]
    return df


def explode_counts(df: pd.DataFrame, list_column: str, top: int = 10) -> pd.Series:
    """Most common values in a list column (genres, countries, cast...)."""
    return df[list_column].explode().dropna().value_counts().head(top)
