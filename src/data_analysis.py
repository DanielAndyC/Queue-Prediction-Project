import pandas as pd
from config import FEATURES_FILE


def load_data(filepath=FEATURES_FILE):
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def summarise_queue(df):
    print("=== Queue Data Summary ===")
    print(f"Number of records: {len(df)}")
    print(f"Number of days:    {df['day_index'].nunique()}")
    print(f"Meals:             {sorted(df['meal'].unique())}")
    print(f"Mean queue length: {df['queue_length'].mean():.2f}")
    print(f"Max queue length:  {df['queue_length'].max()}")
    print(f"Std deviation:     {df['queue_length'].std():.2f}")
    print()

    print("=== Per-Meal Summary ===")
    for meal, group in df.groupby("meal"):
        print(f"{meal:<7}  mean = {group['queue_length'].mean():6.2f}   "
              f"max = {group['queue_length'].max():5.0f}   "
              f"std = {group['queue_length'].std():6.2f}")


def find_peak(df):
    peak_row = df.loc[df["queue_length"].idxmax()]
    print("\n=== Peak Queue ===")
    print(f"Peak {int(peak_row['queue_length'])} on day {int(peak_row['day_index'])} "
          f"({peak_row['weekday']}, {peak_row['meal']}) "
          f"at {peak_row['timestamp'].strftime('%H:%M')}")


if __name__ == "__main__":
    df = load_data()
    summarise_queue(df)
    find_peak(df)