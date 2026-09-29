"""Smoke tests: every Streamlit app runs end to end without raising."""

import pytest
from streamlit.testing.v1 import AppTest

from conftest import ROOT

APPS = [
    "icm_na_prediction/app.py",
    "cricket_score_prediction/app.py",
    "house_price_prediction/app.py",
    "heart_failure_prediction/app.py",
    "student_performance/app.py",
    "netflix_eda/app.py",
    "eda_app/app.py",
]


@pytest.mark.parametrize("path", APPS)
def test_app_runs(path):
    app = AppTest.from_file(str(ROOT / path), default_timeout=300).run()
    assert not app.exception, [e.value for e in app.exception]
    for button in app.button:
        button.click().run()
        assert not app.exception, [e.value for e in app.exception]
