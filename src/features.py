"""
Feature engineering: combine the simulated queue CSVs, per-student
event CSVs, and real class counts into one feature table.
"""

import os
import pandas as pd
import numpy as np

from config import (
    FEATURES_FILE,
    QUEUE_FILE_TEMPLATE, EVENTS_FILE_TEMPLATE,
    CLASS_COUNTS_CSV, WEEKDAYS,
    SIM_START_LUNCH, SIM_START_DINNER,
)


def load_arrival_events(events_path):
    df = pd.read_csv(events_path)
    df["students"] = 1
    return df.sort_values("arrival_minute", kind="stable").reset_index(drop=True)


def build_feature_table_for_file(queue_path, events_path,
                                  meal_period, day_index, weekday,
                                  real_counts_df):
    queue_df = pd.read_csv(queue_path)
    queue_df["timestamp"] = pd.to_datetime(queue_df["timestamp"])

    # Map the simulation's "total_queue" to "queue_length" used everywhere else
    if "total_queue" in queue_df.columns:
        queue_df["queue_length"] = queue_df["total_queue"]

    sim_start = SIM_START_LUNCH if meal_period == "lunch" else SIM_START_DINNER
    day_start = sim_start + pd.Timedelta(days=day_index)
    queue_df["minute"] = (
        (queue_df["timestamp"] - day_start).dt.total_seconds() / 60
    ).astype(int)

    events = load_arrival_events(events_path)

    # ── Arrival-window features ─────────────────────────────────
    for w in [5, 10, 15, 30, 60]:
        queue_df[f"students_arrived_last_{w}min"] = 0
    queue_df["time_since_last_arrival"] = 999.0
    queue_df["last_building_walking_time"] = 0
    queue_df["last_class_size"] = 0

    if not events.empty:
        arrivals = events["arrival_minute"].values
        for idx, row in queue_df.iterrows():
            m = row["minute"]
            for w in [5, 10, 15, 30, 60]:
                count = int(((arrivals > m - w) & (arrivals <= m)).sum())
                queue_df.at[idx, f"students_arrived_last_{w}min"] = count

            past = events[events["arrival_minute"] <= m]
            if not past.empty:
                last = past.iloc[-1]
                queue_df.at[idx, "time_since_last_arrival"] = m - last["arrival_minute"]
                queue_df.at[idx, "last_building_walking_time"] = last["walking_time"]
                queue_df.at[idx, "last_class_size"] = last["class_size"]

    # ── Time features ───────────────────────────────────────────
    queue_df["hour"] = queue_df["timestamp"].dt.hour
    queue_df["minute_of_day"] = (queue_df["timestamp"].dt.hour * 60
                                 + queue_df["timestamp"].dt.minute)

    if meal_period == "lunch":
        queue_df["is_peak"] = (
            (queue_df["hour"] == 12) |
            ((queue_df["hour"] == 11) & (queue_df["timestamp"].dt.minute >= 30))
        ).astype(int)
        queue_df["is_dinner"] = 0
    else:
        queue_df["is_peak"] = (
            (queue_df["hour"] == 18) |
            ((queue_df["hour"] == 17) & (queue_df["timestamp"].dt.minute >= 30))
        ).astype(int)
        queue_df["is_dinner"] = 1

    # ── Per-building class-count features (weekday-specific) ────
    day_counts = real_counts_df[real_counts_df["weekday"] == weekday]
    building_cols = []
    for b in real_counts_df["building"].unique():
        col = f"classes_{b.replace(' ', '_')}"
        building_cols.append(col)
        hourly = day_counts[day_counts["building"] == b].set_index("hour")["num_classes"]
        queue_df[col] = queue_df["hour"].map(hourly).fillna(0)

    queue_df["total_classes"] = queue_df[building_cols].sum(axis=1)

    # ── Lag features ────────────────────────────────────────────
    queue_df = queue_df.sort_values("minute").reset_index(drop=True)
    queue_df["queue_length_lag1"] = queue_df["queue_length"].shift(1).fillna(0)
    for lag in [5, 10, 15, 30, 60]:
        queue_df[f"queue_lag_{lag}"] = queue_df["queue_length"].shift(lag).fillna(0)

    queue_df["day_index"] = day_index
    queue_df["weekday"]   = weekday
    queue_df["meal"]      = meal_period
    return queue_df


def build_multi_day_feature_table(num_days=30, weekdays=None,
                                   output=FEATURES_FILE):
    if weekdays is None:
        weekdays = WEEKDAYS

    real_counts = pd.read_csv(CLASS_COUNTS_CSV)
    frames = []

    for day in range(num_days):
        weekday = weekdays[day % len(weekdays)]
        for meal in ["lunch", "dinner"]:
            queue_path  = QUEUE_FILE_TEMPLATE.format(meal=meal, day=day)
            events_path = EVENTS_FILE_TEMPLATE.format(meal=meal, day=day)
            if not os.path.exists(queue_path):
                print(f"Skipping missing {queue_path}")
                continue
            frames.append(build_feature_table_for_file(
                queue_path, events_path, meal, day, weekday, real_counts))

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} rows to {output}")
    return df


if __name__ == "__main__":
    build_multi_day_feature_table(num_days=30)