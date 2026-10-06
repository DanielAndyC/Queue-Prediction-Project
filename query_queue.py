"""
Query the simulated queue at a specific (day, time).
"""
import pandas as pd
import matplotlib.pyplot as plt

from src.config import FEATURES_FILE


class QuitQuery(Exception):
    """Raised when the user types 'q' at any prompt."""
    pass


# ──────────────────────────────────────────────────────────────────
# Query the loaded dataframe
# ──────────────────────────────────────────────────────────────────
def query(df, day, time_str):
    # Infer meal from hour
    hour = int(time_str[:2])
    if 10 <= hour <= 15:
        meal = "lunch"
    elif 16 <= hour <= 21:
        meal = "dinner"
    else:
        print(f"\nTime {time_str} is outside meal windows "
              f"(lunch 10:30-14:59, dinner 16:30-20:59).")
        return

    mask = (
        (df["day_index"] == day) &
        (df["meal"] == meal) &
        (df["timestamp"].dt.strftime("%H:%M") == time_str)
    )
    row = df[mask]
    if row.empty:
        print(f"\nNo data for day={day}, {meal} at {time_str}")
        return

    r = row.iloc[0]
    print(f"\nDay {day} ({r['weekday']}, {meal}) at {time_str}")
    print("-" * 45)
    print(f"  {'Kiosk (Stage 1)':<25}{r['kiosk_queue']:>5}")

    counter_cols = [c for c in df.columns
                    if c.startswith("counter_") and c.endswith("_queue")]
    for col in counter_cols:
        name = col.replace("counter_", "").replace("_queue", "").replace("_", " ")
        print(f"  {name:<25}{r[col]:>5}")

    print("-" * 45)
    print(f"  {'TOTAL':<25}{r['total_queue']:>5}")


# ──────────────────────────────────────────────────────────────────
# Show per-counter graph for a single day (lunch + dinner)
# ──────────────────────────────────────────────────────────────────
def show_day_graph(df, day):
    """Display per-counter queue length over time for a single day."""
    day_df = df[df["day_index"] == day]
    if day_df.empty:
        print(f"\nNo data for day={day}")
        return

    weekday = day_df["weekday"].iloc[0]
    counter_cols = [c for c in df.columns
                    if c.startswith("counter_") and c.endswith("_queue")]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    for ax, meal in zip(axes, ["lunch", "dinner"]):
        d = day_df[day_df["meal"] == meal].copy()
        if d.empty:
            ax.set_title(f"{meal.capitalize()} — no data")
            continue

        hours = d["timestamp"].dt.hour + d["timestamp"].dt.minute / 60

        if "kiosk_queue" in d.columns:
            ax.plot(hours, d["kiosk_queue"], label="Kiosk (Stage 1)",
                    linewidth=2.5, color="black")

        for col in counter_cols:
            name = col.replace("counter_", "").replace("_queue", "").replace("_", " ")
            ax.plot(hours, d[col], label=name, alpha=0.8)

        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Queue length")
        ax.set_title(f"Per-counter queue length (Day {day} — {weekday}, {meal})")
        ax.legend(fontsize=9, ncol=2)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────
# Input prompts
# ──────────────────────────────────────────────────────────────────
def prompt_int(prompt, lo, hi, allow_graph=False):
    while True:
        raw = input(prompt).strip()
        if raw.lower() == "q":
            raise QuitQuery
        if allow_graph and raw.lower() in ("g", "graph"):
            return "graph"
        try:
            val = int(raw)
        except ValueError:
            print(f"  Please enter a whole number between {lo} and {hi}.")
            continue
        if lo <= val <= hi:
            return val
        print(f"  Out of range. Enter a number between {lo} and {hi}.")


def prompt_time():
    """Accepts HH:MM, HH.MM, or HHMM. Returns canonical HH:MM."""
    while True:
        raw = input("Time (HH.MM): ").strip()
        if raw.lower() == "q":
            raise QuitQuery

        # Normalise separators
        cleaned = raw.replace(".", ":").replace(" ", "")
        if ":" in cleaned:
            parts = cleaned.split(":")
            if len(parts) == 2 and len(parts[0]) <= 2 and len(parts[1]) == 2:
                hh, mm = parts[0].zfill(2), parts[1]
            else:
                print("  Invalid time. Try 13:08, 13.08, or 1308.")
                continue
        elif cleaned.isdigit() and len(cleaned) == 4:
            hh, mm = cleaned[:2], cleaned[2:]
        else:
            print("  Invalid time. Try 13:08, 13.08, or 1308.")
            continue

        try:
            h, m = int(hh), int(mm)
        except ValueError:
            print("  Invalid time. Try 13:08, 13.08, or 1308.")
            continue

        if 0 <= h <= 23 and 0 <= m <= 59:
            return f"{h:02d}:{m:02d}"
        print("  Invalid time — hours must be 00-23 and minutes 00-59.")


# ──────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────
def main():
    print("=== CYM Queue Query ===")
    print("Type 'q' at any prompt to quit.\n")
    print("Type 'g' at the Day prompt to view a per-counter graph.\n")

    df = pd.read_csv(FEATURES_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    query_count = 0
    action_count = 0

    while True:
        try:
            if action_count > 0:
                print()  # blank line between queries

            day = prompt_int("Day (0-29): ", 0, 29, allow_graph=True)

            if day == "graph":
                graph_day = prompt_int("Graph day (0-29): ", 0, 29)
                show_day_graph(df, graph_day)
                action_count += 1
                continue

            time_str = prompt_time()

            query(df, day, time_str)
            query_count += 1
            action_count += 1

        except QuitQuery:
            print(f"\nBye. You ran {query_count} "
                  f"{'query' if query_count == 1 else 'queries'}.")
            break
        except (KeyboardInterrupt, EOFError):
            print(f"\nInterrupted. You ran {query_count} "
                  f"{'query' if query_count == 1 else 'queries'}.")
            break


if __name__ == "__main__":
    main()