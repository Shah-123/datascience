import heart_model as hf
import netflix_data as nd
import student_model as sm


def test_heart_data_has_no_duplicates_or_leaky_feature():
    df = hf.load_data()
    assert not df.duplicated().any()
    assert "time" not in hf.FEATURES


def test_heart_groups_never_cross_folds():
    from sklearn.model_selection import StratifiedGroupKFold

    df = hf.load_data()
    groups = hf.patient_groups(df)
    for train, test in StratifiedGroupKFold(n_splits=5).split(df, df[hf.TARGET], groups):
        assert not set(groups.iloc[train]) & set(groups.iloc[test])


def test_student_models_and_effects():
    df = sm.load_data()
    models = sm.fit_models(df)
    effects = sm.background_effects(models["Background only"])
    assert "lunch_standard" in effects.index
    assert sm.reference_groups(models["Background only"])["lunch"] == "free/reduced"


def test_netflix_cleaning():
    df = nd.load_titles()
    assert not df["title"].str.contains("Ã", regex=False).any()  # no mojibake
    assert not df["rating"].str.contains("min", na=False).any()
    assert df["year_added"].max() <= 2021
    assert set(df["audience"]) <= set(nd.AUDIENCE_ORDER)
    assert df["minutes"].notna().sum() > 6000
