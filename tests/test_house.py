import pandas as pd

import house_model as hm


def make_raw():
    base = {
        "property_type": "House",
        "location": " DHA Defence ",
        "city": "Lahore",
        "baths": 3,
        "purpose": "For Sale",
        "bedrooms": 3,
        "Area_in_Marla": 10.0,
        "price": 20_000_000,
    }
    rows = [base, base, {**base, "city": "Karachi"}, {**base, "Area_in_Marla": 0}]
    rows += [{**base, "price": 10_000_001 + i * 100_000} for i in range(300)]
    return pd.DataFrame(rows)


def test_clean_removes_duplicates_and_invalid_rows():
    df, report = hm.clean(make_raw())
    assert not df.duplicated().any()
    assert (df["Area_in_Marla"] > 0).all()
    assert report["after removing exact duplicates"] == report["raw rows"] - 1


def test_neighbourhoods_are_keyed_by_city():
    df, _ = hm.clean(make_raw())
    assert {"Lahore / DHA Defence", "Karachi / DHA Defence"} <= set(df["area_location"])


def test_load_raw_drops_index_column(tmp_path):
    path = tmp_path / "raw.csv"
    make_raw().to_csv(path)  # writes the index, like the original export
    assert "Unnamed: 0" not in hm.load_raw(path).columns


def test_baseline_uses_price_per_marla():
    X = pd.DataFrame({"area_location": ["a", "a", "b"], "city": ["c"] * 3, "Area_in_Marla": [5, 10, 10]})
    y = pd.Series([500, 1000, 3000])
    model = hm.PricePerMarlaBaseline().fit(X, y)
    new = pd.DataFrame({"area_location": ["a", "unseen"], "city": ["c", "c"], "Area_in_Marla": [20, 1]})
    assert model.predict(new).tolist() == [2000, 100]


def test_format_pkr():
    assert hm.format_pkr(25_000_000) == "PKR 2.50 crore"
    assert hm.format_pkr(8_500_000) == "PKR 85.0 lakh"
    assert hm.format_pkr(45_000) == "PKR 45,000"
