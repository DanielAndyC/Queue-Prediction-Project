import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import (
    FEATURES_FILE, TRAIN_DAYS, TEST_DAYS,
    SCHEDULE_FEATURES, ALL_FEATURES, BASE_SEED,
)


def load_features(filepath=FEATURES_FILE):
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def day_split(df):
    train_df = df[df["day_index"].isin(TRAIN_DAYS)].copy()
    test_df  = df[df["day_index"].isin(TEST_DAYS)].copy()
    return train_df, test_df


def evaluate(name, y_true, y_pred, baseline_mae=None):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    if baseline_mae:
        red = (baseline_mae - mae) / baseline_mae * 100
        print(f"  {name:35s}  MAE = {mae:6.2f}   RMSE = {rmse:6.2f}   "
              f"Reduction = {red:5.1f}%")
    else:
        print(f"  {name:35s}  MAE = {mae:6.2f}   RMSE = {rmse:6.2f}")
    return mae


def baseline(y_train, y_test):
    return np.full(len(y_test), y_train.mean())


def run_experiment(title, train_df, test_df, feature_cols, target_col):
    print(f"\n=== {title} ===")
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test,  y_test  = test_df[feature_cols],  test_df[target_col]

    baseline_mae = evaluate("Baseline (mean)", y_test, baseline(y_train, y_test))

    lr = LinearRegression().fit(X_train, y_train)
    evaluate("Linear Regression", y_test, lr.predict(X_test), baseline_mae)

    dt = DecisionTreeRegressor(max_depth=7, random_state=BASE_SEED).fit(X_train, y_train)
    evaluate("Decision Tree (depth=7)", y_test, dt.predict(X_test), baseline_mae)

    return dt


def print_importance(dt, feature_cols):
    print("\n  Feature importance (Decision Tree):")
    imp = pd.Series(dt.feature_importances_, index=feature_cols)
    for feat, val in imp.sort_values(ascending=False).items():
        print(f"    {feat:35s}  {val:.3f}")


def experiment_a(df):
    train_df, test_df = day_split(df)
    dt = run_experiment("Experiment A: Schedule-only, predict current queue",
                        train_df, test_df, SCHEDULE_FEATURES, "queue_length")
    print_importance(dt, SCHEDULE_FEATURES)


def experiment_b(df):
    df = df.copy()
    df["target_future"] = df.groupby(["day_index", "meal"])["queue_length"].shift(-10)
    df = df.dropna(subset=["target_future"])
    train_df, test_df = day_split(df)
    run_experiment("Experiment B: Predict queue 10 minutes ahead",
                   train_df, test_df, ALL_FEATURES, "target_future")


def experiment_c(df):
    df = df.copy()
    df["queue_change"] = df["queue_length"] - df["queue_length_lag1"]
    train_df, test_df = day_split(df)
    dt = run_experiment("Experiment C: Predict change in queue length",
                        train_df, test_df, SCHEDULE_FEATURES, "queue_change")
    print_importance(dt, SCHEDULE_FEATURES)


def experiment_d(df):
    train_df, test_df = day_split(df)
    run_experiment("Experiment D: All features, predict current queue",
                   train_df, test_df, ALL_FEATURES, "queue_length")


if __name__ == "__main__":
    df = load_features()
    experiment_a(df)
    experiment_b(df)
    experiment_c(df)
    experiment_d(df)