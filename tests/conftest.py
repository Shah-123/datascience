import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = [
    "icm_na_prediction",
    "cricket_score_prediction",
    "house_price_prediction",
    "heart_failure_prediction",
    "student_performance",
    "netflix_eda",
    "eda_app",
]
for project in PROJECTS:
    sys.path.insert(0, str(ROOT / project))
