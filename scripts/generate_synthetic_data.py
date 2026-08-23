"""Generate a synthetic time-use dataset in the format the pipeline expects.

Honesty note: the real ATUS extract this project's notebooks were built
against (`Data/raw/atus_full_selected.csv`) is not included in this
repository - it's 3.3M rows, well beyond what's practical to commit to
git, and the exact selection/cleaning steps that produced it aren't
fully specified in the notebooks either. Rather than leave the pipeline
undemonstrable, this script generates a SYNTHETIC dataset that mimics
real time-use patterns (activities weighted by realistic time-of-day
probabilities: sleep at night, work on weekdays during business hours,
meals at typical times, etc.), in the exact column format
(`hour`, `ACTIVITY_NAME`, `ACTIVITY_NAME_ENC`) that
`notebooks/environment_setup.py`'s `ScheduleEnv` and `ActivityPatterns`
already expect. This lets the real training pipeline run end-to-end and
produce a genuinely trained, working agent - clearly labeled as trained
on synthetic data, not the original ATUS survey.

Run: python scripts/generate_synthetic_data.py
Produces: data/synthetic_time_use.csv
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
OUT_PATH = ROOT / "data" / "synthetic_time_use.csv"

random.seed(42)
np.random.seed(42)

# Activities and, for each, an hourly weight profile (24 values, need not
# sum to 1 - normalized per activity below). Profiles are hand-authored
# from common-sense time-use patterns (sleep at night, work 9-5 on
# weekdays, etc.), not derived from any real survey.
ACTIVITIES = {
    "Sleeping": [8, 8, 8, 8, 7, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 7],
    "Personal Care": [0, 0, 0, 0, 1, 3, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 2, 1, 1, 0],
    "Eating and Drinking": [0, 0, 0, 0, 0, 1, 3, 4, 2, 1, 1, 2, 5, 3, 1, 1, 1, 2, 5, 3, 1, 1, 0, 0],
    "Working": [0, 0, 0, 0, 0, 0, 1, 3, 6, 8, 8, 8, 4, 7, 8, 8, 7, 3, 1, 0, 0, 0, 0, 0],
    "Household Activities": [0, 0, 0, 0, 0, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 2, 1, 0],
    "Caring for Household Members": [0, 0, 0, 0, 0, 0, 2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0],
    "Shopping": [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 2, 2, 2, 2, 3, 3, 2, 1, 0, 0, 0, 0],
    "Socializing and Leisure": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4, 6, 7, 6, 3, 1],
    "Sports and Exercise": [0, 0, 0, 0, 0, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 2, 1, 1, 0, 0],
    "Education": [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 2, 4, 4, 3, 2, 1, 1, 1, 1, 0, 0, 0],
    "Traveling / Commuting": [0, 0, 0, 0, 0, 1, 3, 5, 3, 1, 1, 1, 2, 1, 1, 1, 3, 5, 3, 1, 1, 0, 0, 0],
    "Religious and Volunteer Activities": [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 0, 0],
}

N_SYNTHETIC_DAYS = 2000


def sample_activity_for_hour(hour: int) -> str:
    names = list(ACTIVITIES.keys())
    weights = np.array([ACTIVITIES[name][hour] + 0.1 for name in names])  # +0.1 floor so nothing is impossible
    weights = weights / weights.sum()
    return np.random.choice(names, p=weights)


def main():
    rows = []
    activity_names_sorted = sorted(ACTIVITIES.keys())
    name_to_code = {name: i for i, name in enumerate(activity_names_sorted)}

    for day in range(N_SYNTHETIC_DAYS):
        day_of_week = day % 7
        for hour in range(24):
            activity = sample_activity_for_hour(hour)
            rows.append({
                "day_id": day,
                "day_of_week": day_of_week,
                "hour": hour,
                "ACTIVITY_NAME": activity,
                "ACTIVITY_NAME_ENC": name_to_code[activity],
            })

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Generated {len(df)} rows ({N_SYNTHETIC_DAYS} synthetic days) -> {OUT_PATH}")
    print(f"{len(activity_names_sorted)} activity types: {activity_names_sorted}")


if __name__ == "__main__":
    main()
