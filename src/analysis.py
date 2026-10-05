import pandas as pd
import numpy as np
from scipy.signal import correlate

from config import FEATURES_FILE, BASE_SEED
from simulation import run_simulation


def cross_correlate_events_and_queue(features_file=FEATURES_FILE,
                                     day_index=3, meal="lunch", max_lag=90):
    df = pd.read_csv(features_file)
    day = df[(df["day_index"] == day_index) & (df["meal"] == meal)].copy()

    event_series = (day["students_arrived_last_5min"] > 0).astype(int).values
    queue_series = day["queue_length"].values

    e = event_series - event_series.mean()
    q = queue_series - queue_series.mean()

    # Normalised cross-correlation → values in [-1, 1]
    corr = correlate(q, e, mode="full")
    norm = np.sqrt(np.sum(q ** 2) * np.sum(e ** 2))
    corr = corr / norm

    lags = np.arange(-len(e) + 1, len(e))
    mask = (lags >= -max_lag) & (lags <= max_lag)
    lags, corr = lags[mask], corr[mask]

    best_idx = int(np.argmax(corr))
    best_lag = int(lags[best_idx])
    best_r   = float(corr[best_idx])
    print(f"=== Cross-Correlation ({meal}, day {day_index}) ===")
    print(f"Strongest positive correlation at lag = {best_lag} min (r = {best_r:.3f})")
    return lags, corr


def sensitivity_analysis(n_seeds=5, meal_period="lunch"):
    print(f"\n=== Sensitivity Analysis ({meal_period}, avg over {n_seeds} seeds) ===")
    print(f"{'Scenario':40s}  {'Mean':>8s}  {'Peak':>8s}")

    scenarios = [
        ("Baseline",                        dict()),
        ("7 kiosks",                        dict(num_kiosks=7)),
        ("5 kiosks",                        dict(num_kiosks=5)),
        ("Faster kiosk service (0.67x)",    dict(kiosk_service_factor=0.67)),
        ("Slower kiosk service (1.33x)",    dict(kiosk_service_factor=1.33)),
        ("Faster counters (0.7x)",          dict(counter_service_factor=0.7)),
        ("Slower counters (1.3x)",          dict(counter_service_factor=1.3)),
        ("Students walk 2x faster",         dict(walking_speed_factor=2.0)),
        ("Students walk 2x slower",         dict(walking_speed_factor=0.5)),
    ]

    for name, kwargs in scenarios:
        means, peaks = [], []
        for s in range(n_seeds):
            seed = BASE_SEED + s
            qlog = run_simulation(
                meal_period, day=0, weekday="Monday",
                seed=seed, save_files=False, **kwargs,
            )
            means.append(qlog.mean())
            peaks.append(qlog.max())
        print(f"{name:40s}  {np.mean(means):8.2f}  {np.mean(peaks):8.2f}")


if __name__ == "__main__":
    cross_correlate_events_and_queue()
    sensitivity_analysis(n_seeds=5, meal_period="lunch")