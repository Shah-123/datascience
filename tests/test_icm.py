import numpy as np
import pandas as pd

import icm_model as icm


def make_raw():
    """Two monitors over three campaigns, one cluster row each."""
    rows = []
    for camp, date in [(1, "2020-03-01"), (2, "2020-06-01"), (0, "2020-09-01")]:
        for monitor, na in [(10, 2 * (camp + 1)), (20, 1)]:
            rows.append(
                {
                    "MONITORID": monitor,
                    "UCID": 1,
                    "CAMP_ID": camp,
                    "CLUSTER_DATE": date,
                    "TOTAL_HH": 7,
                    "HRMP": 0,
                    "RECALL_011_CHK": 4,
                    "RECALL_1259_CHK": 16,
                    "FM_059_CHK": 10,
                    "GUEST_CHK": 1,
                    "ZERO_DOSE_023": 0,
                    "CORRECT_DOOR_MARK": 7,
                    "NA": na,
                }
            )
    raw = pd.DataFrame(rows)
    raw["CLUSTER_DATE"] = pd.to_datetime(raw["CLUSTER_DATE"])
    return raw


def test_leaky_columns_are_not_features():
    assert not set(icm.LEAKY_COLUMNS) & set(icm.FEATURES)
    assert icm.TARGET not in icm.FEATURES


def test_campaigns_are_ordered_by_date_not_id():
    table = icm.build_monitor_campaign_table(make_raw())
    order = table.drop_duplicates("CAMP_ID").set_index("CAMP_ID")["campaign_order"]
    assert order[1] < order[2] < order[0]


def test_history_features_only_use_earlier_campaigns():
    table = icm.build_monitor_campaign_table(make_raw())
    monitor = table[table["MONITORID"] == 10].sort_values("campaign_order")
    assert np.isnan(monitor["prev_na_rate"].iloc[0])
    # NA in campaigns (by date) is 4, 6, 2 over 20 children checked.
    assert monitor["prev_na_rate"].tolist()[1:] == [4 / 20, 6 / 20]
    assert monitor["hist_na_rate"].iloc[2] == (4 / 20 + 6 / 20) / 2


def test_temporal_split_tests_on_latest_campaigns():
    table = icm.build_monitor_campaign_table(make_raw())
    train, test = icm.temporal_split(table, n_test_campaigns=1)
    assert train["campaign_order"].max() < test["campaign_order"].min()
    assert set(test["CAMP_ID"]) == {0}


def test_flags_obvious_outlier():
    frame = pd.DataFrame(
        {
            "MONITORID": range(20),
            "CAMP_ID": 1,
            "campaign_start": pd.Timestamp("2020-01-01"),
            "NA": [10] * 19 + [60],
        }
    )
    flags = icm.flag_unusual_reports(frame, np.full(20, 10.0))
    assert flags["Flag"].iloc[-1] == "Higher than expected"
    assert (flags["Flag"].iloc[:-1] == "").all()
