"""
Two-stage SimPy simulation of the CYM canteen.

  Stage 1: 6 ordering/payment kiosks (order + pay + drinks).
  Stage 2: 4 food counters, each staffed by 3 parallel servers:
           Supreme BBQ, China Kitchen, Noodle Spot, Gourmet Kitchen.

Writes per (meal, day):
  simulated_queues_{meal}_day{N}.csv   per-minute queue-length samples,
                                       per kiosk and per counter
  events_{meal}_day{N}.csv             per-student journey record
"""

import os
import csv
import math
import random
import numpy as np
import simpy
import pandas as pd
from datetime import timedelta

from config import (
    SIM_START_LUNCH, SIM_END_LUNCH,
    SIM_START_DINNER, SIM_END_DINNER,
    QUEUE_RECORD_INTERVAL,
    NUM_KIOSKS, KIOSK_SERVICE_MEAN, KIOSK_SERVICE_STD,
    COUNTER_NAMES, COUNTER_CAPACITY,
    COUNTER_WEIGHTS_LUNCH, COUNTER_WEIGHTS_DINNER,
    COUNTER_SERVICE_MEAN, COUNTER_SERVICE_STD,
    CYM_ATTENDANCE_FRACTION, AVG_CLASS_SIZE, CLASS_SIZE_LOG_STD,
    BALK_THRESHOLD, BALK_PROBABILITY,
    BUILDINGS, CLASS_COUNTS_CSV,
    QUEUE_FILE_TEMPLATE, EVENTS_FILE_TEMPLATE,
    WEEKDAYS, BASE_SEED,
)


# ──────────────────────────────────────────────────────────────────
# Real class schedule → arrival events
# ──────────────────────────────────────────────────────────────────
def load_real_schedule(weekday, meal_period, csv_path=CLASS_COUNTS_CSV):
    """
    Load real class counts and turn them into per-class arrival events.

    CSV columns: weekday, hour, building, num_classes.
    A class labelled hour=H ends at (H-1):50 in real clock time.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["weekday"] == weekday]

    if meal_period == "lunch":
        sim_start = SIM_START_LUNCH
        hours = [11, 12, 13]
    else:
        sim_start = SIM_START_DINNER
        hours = [17, 18, 19]

    df = df[df["hour"].isin(hours)]
    sim_start_minutes = sim_start.hour * 60 + sim_start.minute

    log_mean = math.log(AVG_CLASS_SIZE) - CLASS_SIZE_LOG_STD ** 2 / 2
    events = []

    for _, row in df.iterrows():
        hour = int(row["hour"])
        building = row["building"]
        if building not in BUILDINGS:
            raise ValueError(f"Unknown building in CSV: {building!r}")
        num_classes = int(row["num_classes"])
        if num_classes == 0:
            continue

        class_end_real = (hour - 1) * 60 + 50
        class_end_minute = class_end_real - sim_start_minutes

        for _ in range(num_classes):
            class_size = max(1, round(random.lognormvariate(
                log_mean, CLASS_SIZE_LOG_STD)))
            attendees = int(np.random.binomial(class_size, CYM_ATTENDANCE_FRACTION))
            if attendees > 0:
                events.append((class_end_minute, building, class_size, attendees))

    return events


# ──────────────────────────────────────────────────────────────────
# Student process
# ──────────────────────────────────────────────────────────────────
def student_journey(env, arrival_offset, building, class_size, walking_time,
                    kiosks, food_counters, counter_weights,
                    kiosk_service_factor, counter_service_factor,
                    events_log, student_id):
    """One student: arrive → queue at kiosk → queue at food counter → leave."""
    delay = arrival_offset - env.now
    if delay > 0:
        yield env.timeout(delay)

    arrival_time = env.now

    # Balking at the kiosks
    if len(kiosks.queue) > BALK_THRESHOLD and random.random() < BALK_PROBABILITY:
        return

    # ── Stage 1: kiosk (order + pay + drinks) ────────────────────
    with kiosks.request() as req:
        yield req
        kiosk_start = env.now
        kiosk_wait = kiosk_start - arrival_time
        service_time = max(0.2, random.gauss(
            KIOSK_SERVICE_MEAN, KIOSK_SERVICE_STD) * kiosk_service_factor)
        yield env.timeout(service_time)
    kiosk_end = env.now

    # ── Stage 2: food counter (3 parallel servers) ───────────────
    counter_name = random.choices(
        list(counter_weights.keys()),
        weights=list(counter_weights.values()),
        k=1,
    )[0]
    counter = food_counters[counter_name]

    with counter.request() as req:
        yield req
        counter_start = env.now
        counter_wait = counter_start - kiosk_end
        serve_time = max(0.3, random.gauss(
            COUNTER_SERVICE_MEAN[counter_name], COUNTER_SERVICE_STD)
            * counter_service_factor)
        yield env.timeout(serve_time)

    total_time = env.now - arrival_time

    events_log.append({
        "student_id":    student_id,
        "arrival_minute": round(arrival_time, 3),
        "building":      building,
        "class_size":    class_size,
        "walking_time":  walking_time,
        "kiosk_wait":    round(kiosk_wait, 3),
        "counter_name":  counter_name,
        "counter_wait":  round(counter_wait, 3),
        "total_time":    round(total_time, 3),
    })


def record_queue(env, kiosks, food_counters, queue_log, interval):
    """Sample queue length every interval, per resource."""
    while True:
        snapshot = {"kiosk_queue": len(kiosks.queue)}
        for name, res in food_counters.items():
            snapshot[f"counter_{name}_queue"] = len(res.queue)
        snapshot["total_queue"] = (
            snapshot["kiosk_queue"]
            + sum(v for k, v in snapshot.items()
                  if k.startswith("counter_") and k.endswith("_queue"))
        )
        queue_log.append(snapshot)
        yield env.timeout(interval)


# ──────────────────────────────────────────────────────────────────
# Simulate one (meal, day) → returns (queue_log, events_log)
# ──────────────────────────────────────────────────────────────────
def simulate(meal_period, day, weekday, seed=BASE_SEED,
             num_kiosks=NUM_KIOSKS,
             kiosk_service_factor=1.0,
             counter_service_factor=1.0,
             counter_capacity=COUNTER_CAPACITY,
             walking_speed_factor=1.0):
    random.seed(seed)
    np.random.seed(seed)

    env = simpy.Environment()
    kiosks = simpy.Resource(env, capacity=num_kiosks)
    food_counters = {name: simpy.Resource(env, capacity=counter_capacity)
                     for name in COUNTER_NAMES}
    queue_log = []
    events_log = []

    if meal_period == "lunch":
        counter_weights = COUNTER_WEIGHTS_LUNCH
        sim_start = SIM_START_LUNCH
        sim_end   = SIM_END_LUNCH
    else:
        counter_weights = COUNTER_WEIGHTS_DINNER
        sim_start = SIM_START_DINNER
        sim_end   = SIM_END_DINNER

    env.process(record_queue(
        env, kiosks, food_counters, queue_log, QUEUE_RECORD_INTERVAL))

    schedule = load_real_schedule(weekday, meal_period)
    student_id = 0
    for class_end_minute, building, class_size, attendees in schedule:
        walk = BUILDINGS[building]["walk"] / walking_speed_factor
        for _ in range(attendees):
            spread = random.gauss(0, 5.0)
            arrival_offset = max(0, class_end_minute + walk + spread)
            env.process(student_journey(
                env, arrival_offset, building, class_size, walk,
                kiosks, food_counters, counter_weights,
                kiosk_service_factor, counter_service_factor,
                events_log, student_id,
            ))
            student_id += 1

    sim_duration = (sim_end - sim_start).total_seconds() / 60.0
    env.run(until=sim_duration)

    return queue_log, events_log, sim_start


def run_simulation(meal_period, day, weekday, seed=BASE_SEED, save_files=True,
                   **kwargs):
    queue_log, events_log, sim_start = simulate(
        meal_period, day, weekday, seed=seed, **kwargs)

    if not save_files:
        # Return only the total queue as a numpy array for sensitivity analysis
        return np.array([snap["total_queue"] for snap in queue_log])

    queue_path  = QUEUE_FILE_TEMPLATE.format(meal=meal_period, day=day)
    events_path = EVENTS_FILE_TEMPLATE.format(meal=meal_period, day=day)
    os.makedirs(os.path.dirname(queue_path), exist_ok=True)

    with open(queue_path, "w", newline="") as f:
        w = csv.writer(f)
        header = (["timestamp", "kiosk_queue"]
                  + [f"counter_{n}_queue" for n in COUNTER_NAMES]
                  + ["total_queue"])
        w.writerow(header)
        for i, snap in enumerate(queue_log):
            ts = sim_start + timedelta(days=day, minutes=i)
            row = ([ts.isoformat(), snap["kiosk_queue"]]
                   + [snap[f"counter_{n}_queue"] for n in COUNTER_NAMES]
                   + [snap["total_queue"]])
            w.writerow(row)

    with open(events_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["student_id", "arrival_minute", "building", "class_size",
                    "walking_time", "kiosk_wait", "counter_name",
                    "counter_wait", "total_time"])
        for e in events_log:
            w.writerow([e["student_id"], e["arrival_minute"], e["building"],
                        e["class_size"], e["walking_time"], e["kiosk_wait"],
                        e["counter_name"], e["counter_wait"], e["total_time"]])

    print(f"Day {day:>2} ({weekday:<9}, {meal_period:<6}): "
          f"{len(queue_log):>3} queue samples, {len(events_log):>4} students served.")


def run_all_days(num_days=30, weekdays=None):
    if weekdays is None:
        weekdays = WEEKDAYS
    for day in range(num_days):
        weekday = weekdays[day % len(weekdays)]
        for meal in ["lunch", "dinner"]:
            seed = BASE_SEED + day * 2 + (0 if meal == "lunch" else 1)
            run_simulation(meal, day, weekday, seed=seed)


if __name__ == "__main__":
    run_all_days(num_days=30)