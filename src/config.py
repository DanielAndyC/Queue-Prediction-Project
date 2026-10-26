"""
Shared constants for the Canteen Queue Prediction project.

Two-stage CYM queue model:
  Stage 1: 6 ordering/payment kiosks (single pooled FCFS queue).
  Stage 2: 4 food counters, each with 3 parallel servers:
           Supreme BBQ, China Kitchen, Noodle Spot, Gourmet Kitchen.

Building walking times and distances are all in one place.
"""
from datetime import datetime

# ── Random seed ─────────────────────────────────────────────────
BASE_SEED = 42

# ── Simulation windows ──────────────────────────────────────────
SIM_START_LUNCH  = datetime(2026, 10, 6, 10, 30)
SIM_END_LUNCH    = datetime(2026, 10, 6, 15, 0)
SIM_START_DINNER = datetime(2026, 10, 6, 16, 30)
SIM_END_DINNER   = datetime(2026, 10, 6, 21, 0)
QUEUE_RECORD_INTERVAL = 1     # minutes

# ── Stage 1: kiosks (order + pay) ---------──────────────────────
NUM_KIOSKS = 6
KIOSK_SERVICE_MEAN = 0.7      # minutes
KIOSK_SERVICE_STD  = 0.5

# ── Stage 2: food counters ──────────────────────────────────────
COUNTER_NAMES = [
    "Supreme_BBQ",
    "China_Kitchen",
    "Noodle_Spot",
    "Gourmet_Kitchen",
]
COUNTER_CAPACITY = 3          # parallel servers per counter

# Popularity weights (must sum to 1.0).
COUNTER_WEIGHTS_LUNCH = {
    "Supreme_BBQ":     0.25,
    "China_Kitchen":   0.25,
    "Noodle_Spot":     0.15,
    "Gourmet_Kitchen": 0.35,
}
COUNTER_WEIGHTS_DINNER = {
    "Supreme_BBQ":     0.15,
    "China_Kitchen":   0.30,
    "Noodle_Spot":     0.20,
    "Gourmet_Kitchen": 0.35,
}

# Food-serving time per counter (minutes).
COUNTER_SERVICE_MEAN = {
    "Supreme_BBQ":     3.0,
    "China_Kitchen":   1.5,
    "Noodle_Spot":     2.0,
    "Gourmet_Kitchen": 1.5,
}
COUNTER_SERVICE_STD = 0.8

# ── Class-size assumptions ──────────────────────────────────────
CYM_ATTENDANCE_FRACTION = 0.10
AVG_CLASS_SIZE = 60
CLASS_SIZE_LOG_STD = 0.8

# ── Balking ─────────────────────────────────────────────────────
BALK_THRESHOLD = 25
BALK_PROBABILITY = 0.5

# ── Buildings (all in one place for easy editing) ───────────────
BUILDINGS = {
    "CYM Phys":              {"walk": 1,  "dist": 30},
    "Meng Wah Complex":      {"walk": 1,  "dist": 50},
    "Runme Shaw":            {"walk": 2,  "dist": 130},
    "Run Run Shaw":          {"walk": 3,  "dist": 180},
    "KK Leung":              {"walk": 4,  "dist": 150},
    "Knowles":               {"walk": 4,  "dist": 170},
    "James Hsioung Lee":     {"walk": 4,  "dist": 240},
    "CYM Chem":              {"walk": 9,  "dist": 500},
    "Rayson Huang Theatre":  {"walk": 12, "dist": 750},
}

# ── File paths ──────────────────────────────────────────────────
CLASS_COUNTS_CSV    = "data/real_class_counts.csv"
QUEUE_FILE_TEMPLATE = "data/simulated_queues_{meal}_day{day}.csv"
EVENTS_FILE_TEMPLATE = "data/events_{meal}_day{day}.csv"
FEATURES_FILE       = "data/features_multi.csv"

# ── Weekdays & train/test split ─────────────────────────────────
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TRAIN_DAYS = tuple(range(0, 20))
TEST_DAYS  = tuple(range(20, 30))

# ── Feature lists ───────────────────────────────────────────────
SCHEDULE_FEATURES = [
    "students_arrived_last_5min",
    "students_arrived_last_10min",
    "students_arrived_last_15min",
    "students_arrived_last_30min",
    "students_arrived_last_60min",
    "time_since_last_arrival",
    "last_building_walking_time",
    "last_class_size",
    "hour",
    "minute_of_day",
    "is_peak",
    "is_dinner",
    "total_classes",
]
ALL_FEATURES = SCHEDULE_FEATURES + ["queue_length_lag1"]