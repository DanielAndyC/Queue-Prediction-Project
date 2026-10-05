"""
Query the simulated queue at a specific (day, time).
"""
import pandas as pd

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
              f"(lunch 10:30–14:59, dinner 16:30–20:59).")
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
# Input prompts
# ──────────────────────────────────────────────────────────────────
def prompt_int(prompt, lo, hi):
    while True:
        raw = input(prompt).strip()
        if raw.lower() == "q":
            raise QuitQuery
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
        print("  Invalid time — hours must be 00–23 and minutes 00–59.")


# ──────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────
def main():
    print("=== CYM Queue Query ===")
    print("Type 'q' at any prompt to quit.\n")

    df = pd.read_csv(FEATURES_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    query_count = 0

    while True:
        try:
            if query_count > 0:
                print()  # blank line between queries

            day      = prompt_int("Day (0-29): ", 0, 29)
            time_str = prompt_time()

            query(df, day, time_str)
            query_count += 1

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