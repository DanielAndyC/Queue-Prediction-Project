import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from config import (
    FEATURES_FILE, TRAIN_DAYS, TEST_DAYS,
    SCHEDULE_FEATURES, ALL_FEATURES,
    EVENTS_FILE_TEMPLATE, COUNTER_NAMES, BASE_SEED,
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["figure.dpi"] = 100


def load_features(filepath=FEATURES_FILE):
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ──────────────────────────────────────────────────────────────────
# Queue over time (single meal or both meals)
# ──────────────────────────────────────────────────────────────────
def _plot_one_meal(ax, df, meal, days_to_plot=5):
    subset = df[df["meal"] == meal]
    for day in sorted(subset["day_index"].unique())[:days_to_plot]:
        d = subset[subset["day_index"] == day]
        weekday = d["weekday"].iloc[0]
        hours = d["timestamp"].dt.hour + d["timestamp"].dt.minute / 60
        ax.plot(hours, d["queue_length"],
                label=f"Day {day} ({weekday})", alpha=0.7)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Queue length (total)")
    ax.set_title(f"Canteen Queue Length by Day ({meal.capitalize()})")
    ax.legend()
    ax.grid(alpha=0.3)


def plot_queue_over_time(df, meal="both", days_to_plot=5, save_path=None):
    if meal == "both":
        fig, axes = plt.subplots(2, 1, figsize=(14, 9))
        _plot_one_meal(axes[0], df, "lunch",  days_to_plot)
        _plot_one_meal(axes[1], df, "dinner", days_to_plot)
        fig.tight_layout()
        save_path = save_path or "data/plot_queue_over_time.png"
        fig.savefig(save_path); plt.close()
        print(f"Saved: {save_path}")
    else:
        fig, ax = plt.subplots(figsize=(14, 5))
        _plot_one_meal(ax, df, meal, days_to_plot)
        fig.tight_layout()
        save_path = save_path or f"data/plot_queue_over_time_{meal}.png"
        fig.savefig(save_path); plt.close()
        print(f"Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────
# NEW: Per-counter queue length over time
# ──────────────────────────────────────────────────────────────────
def plot_counter_queues(df, day=0, meal="lunch", save_path=None):
    """
    Show the queue at each counter and at the kiosks over one day.
    Requires per-counter columns in the feature CSV.
    """
    d = df[(df["day_index"] == day) & (df["meal"] == meal)].copy()
    counter_cols = [f"counter_{n}_queue" for n in COUNTER_NAMES
                    if f"counter_{n}_queue" in d.columns]
    if not counter_cols:
        print(f"Skipping plot_counter_queues: no per-counter columns found.")
        return

    fig, ax = plt.subplots(figsize=(13, 5))
    hours = d["timestamp"].dt.hour + d["timestamp"].dt.minute / 60

    if "kiosk_queue" in d.columns:
        ax.plot(hours, d["kiosk_queue"], label="Kiosk (Stage 1)",
                linewidth=2.5, color="black")

    for col in counter_cols:
        label = col.replace("counter_", "").replace("_queue", "").replace("_", " ")
        ax.plot(hours, d[col], label=label, alpha=0.8)

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Queue length")
    ax.set_title(f"Per-counter queue length (Day {day}, {meal})")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_path = save_path or f"data/plot_counter_queues_{meal}_day{day}.png"
    fig.savefig(save_path); plt.close()
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────
# Arrivals vs queue (single meal)
# ──────────────────────────────────────────────────────────────────
def plot_arrivals_vs_queue(df, day=0, meal="lunch", save_path=None):
    d = df[(df["day_index"] == day) & (df["meal"] == meal)]
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 7))

    ax1.bar(d["timestamp"], d["students_arrived_last_15min"],
            color="coral", width=0.02)
    ax1.set_ylabel("Students arrived\n(last 15 min)")
    ax1.set_title(f"Arrivals vs Queue Length (Day {day}, {meal})")

    ax2.plot(d["timestamp"], d["queue_length"], color="steelblue", linewidth=1.5)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Total queue length")

    fig.autofmt_xdate()
    fig.tight_layout()
    save_path = save_path or f"data/plot_arrivals_vs_queue_{meal}.png"
    fig.savefig(save_path); plt.close()
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────
# Feature importance
# ──────────────────────────────────────────────────────────────────
def plot_feature_importance(df, save_path="data/plot_feature_importance.png"):
    train_df = df[df["day_index"].isin(TRAIN_DAYS)]
    X, y = train_df[SCHEDULE_FEATURES], train_df["queue_length"]
    dt = DecisionTreeRegressor(max_depth=7, random_state=BASE_SEED).fit(X, y)

    imp = pd.Series(dt.feature_importances_, index=SCHEDULE_FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(imp.index, imp.values, color="mediumseagreen")
    ax.set_xlabel("Importance")
    ax.set_title("Decision Tree Feature Importance (train days)")
    fig.tight_layout()
    fig.savefig(save_path); plt.close()
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────
# Predictions scatter
# ──────────────────────────────────────────────────────────────────
def plot_predictions(df, save_path="data/plot_predictions.png"):
    train_df = df[df["day_index"].isin(TRAIN_DAYS)]
    test_df  = df[df["day_index"].isin(TEST_DAYS)]

    X_train, y_train = train_df[ALL_FEATURES], train_df["queue_length"]
    X_test,  y_test  = test_df[ALL_FEATURES],  test_df["queue_length"]

    lr = LinearRegression().fit(X_train, y_train)
    dt = DecisionTreeRegressor(max_depth=7, random_state=BASE_SEED).fit(X_train, y_train)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, y_pred, title in zip(
        axes,
        [lr.predict(X_test), dt.predict(X_test)],
        ["Linear Regression", "Decision Tree (depth=7)"],
    ):
        ax.scatter(y_test, y_pred, alpha=0.5, color="steelblue", edgecolor="k", s=25)
        lims = [0, max(y_test.max(), y_pred.max()) + 5]
        ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
        ax.set_xlabel("Actual queue length")
        ax.set_ylabel("Predicted queue length")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path); plt.close()
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────
# Residuals
# ──────────────────────────────────────────────────────────────────
def plot_residuals(df, save_path="data/plot_residuals.png"):
    train_df = df[df["day_index"].isin(TRAIN_DAYS)]
    test_df  = df[df["day_index"].isin(TEST_DAYS)]
    X_train, y_train = train_df[ALL_FEATURES], train_df["queue_length"]
    X_test,  y_test  = test_df[ALL_FEATURES],  test_df["queue_length"]

    dt = DecisionTreeRegressor(max_depth=7, random_state=BASE_SEED).fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    residuals = y_test - y_pred

    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, alpha=0.5, color="purple", edgecolor="k", s=25)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted queue length")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residuals - Decision Tree")
    fig.tight_layout()
    fig.savefig(save_path); plt.close()
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────
# Stage 1 vs Stage 2 wait times
# ──────────────────────────────────────────────────────────────────
def _plot_stage_waits_one(ax, meal):
    rows = []
    for day in range(30):
        path = EVENTS_FILE_TEMPLATE.format(meal=meal, day=day)
        if not os.path.exists(path):
            continue
        ev = pd.read_csv(path)
        if ev.empty:
            continue
        rows.append({
            "day": day,
            "kiosk_wait":   ev["kiosk_wait"].mean(),
            "counter_wait": ev["counter_wait"].mean(),
        })
    if not rows:
        ax.set_title(f"No events found for {meal}")
        return
    df = pd.DataFrame(rows)
    ax.plot(df["day"], df["kiosk_wait"],   label="Kiosk wait (Stage 1)",   marker="o")
    ax.plot(df["day"], df["counter_wait"], label="Counter wait (Stage 2)", marker="s")
    ax.set_xlabel("Day")
    ax.set_ylabel("Mean wait (min)")
    ax.set_title(f"Stage 1 vs Stage 2 wait times ({meal})")
    ax.legend()
    ax.grid(alpha=0.3)


def plot_stage_waits(meal="both", save_path="data/plot_stage_waits.png"):
    if meal == "both":
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        for ax, m in zip(axes, ["lunch", "dinner"]):
            _plot_stage_waits_one(ax, m)
        fig.tight_layout()
        fig.savefig(save_path); plt.close()
        print(f"Saved: {save_path}")
    else:
        fig, ax = plt.subplots(figsize=(12, 5))
        _plot_stage_waits_one(ax, meal)
        fig.tight_layout()
        save_path = f"data/plot_stage_waits_{meal}.png"
        fig.savefig(save_path); plt.close()
        print(f"Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_features()
    print("Generating plots...")

    plot_queue_over_time(df, meal="both")
    plot_queue_over_time(df, meal="lunch",
                         save_path="data/plot_queue_over_time_lunch.png")
    plot_queue_over_time(df, meal="dinner",
                         save_path="data/plot_queue_over_time_dinner.png")

    # NEW: per-counter queue plots for day 0
    plot_counter_queues(df, day=0, meal="lunch")
    plot_counter_queues(df, day=0, meal="dinner")

    plot_arrivals_vs_queue(df, day=0, meal="lunch")
    plot_arrivals_vs_queue(df, day=0, meal="dinner")

    plot_feature_importance(df)
    plot_predictions(df)
    plot_residuals(df)

    plot_stage_waits(meal="both")
    plot_stage_waits(meal="lunch")
    plot_stage_waits(meal="dinner")