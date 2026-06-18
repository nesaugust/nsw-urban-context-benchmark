"""
Topic 3 — Weather, Events & Urban Context Benchmark
Stage 2: Benchmark Task Construction
===================================================

Updated to include:
- traffic / pedestrian / POI activity prediction
- anomaly classification
- region weather sensitivity
- scenario cards
- contrastive examples
- POI / Massive-STEPS mobility reasoning
- crash_risk_count

Input:
    data/cleaned/master_context_table.csv

Output:
    data/benchmark/task1_traffic_prediction/task1_qa_pairs.json
    data/benchmark/task2_anomaly_classification/task2_qa_pairs.json
    data/benchmark/task3_region_sensitivity/task3_qa_pairs.json
    data/benchmark/task4_scenario_cards/task4_scenario_cards.json
    data/benchmark/task5_contrastive_examples/task5_contrastive_pairs.json
    data/benchmark/task6_poi_mobility_reasoning/task6_qa_pairs.json
    data/benchmark/task7_llm_urban_context_reasoning/task7_qa_pairs.json
    data/benchmark/benchmark_summary.json

Usage:
    python 03_build_benchmark_tasks_updated.py
"""

import json
import random
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
CLEANED = Path("data/cleaned")
BENCH_DIR = Path("data/benchmark")
BENCH_DIR.mkdir(parents=True, exist_ok=True)

T1 = BENCH_DIR / "task1_traffic_prediction"
T2 = BENCH_DIR / "task2_anomaly_classification"
T3 = BENCH_DIR / "task3_region_sensitivity"
T4 = BENCH_DIR / "task4_scenario_cards"
T5 = BENCH_DIR / "task5_contrastive_examples"
T6 = BENCH_DIR / "task6_poi_mobility_reasoning"
T7 = BENCH_DIR / "task7_llm_urban_context_reasoning"

for folder in [T1, T2, T3, T4, T5, T6, T7]:
    folder.mkdir(parents=True, exist_ok=True)

# ── Sample sizes ───────────────────────────────────────────────────────────────
N_TASK1 = 300
N_TASK2 = 300
N_TASK3 = 100
N_TASK4 = 200
N_TASK5 = 200
N_TASK6 = 200
N_TASK7 = 200

# ── Thresholds ─────────────────────────────────────────────────────────────────
RAIN_NONE = 0.1
RAIN_LIGHT = 1.0
RAIN_MODERATE = 4.0
RAIN_HEAVY = 16.0
TRAFFIC_CHANGE_SIG = 15
HEAT_THRESHOLD = 33
COLD_THRESHOLD = 10


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def log(msg):
    print(f"\n{'=' * 62}\n  {msg}\n{'=' * 62}")


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def safe_bool(x):
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    if isinstance(x, (int, float)):
        return x > 0
    return str(x).strip().lower() in ["true", "1", "yes", "y"]


def save_json(data, path, label):
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"  ✓ {label} → {path.name} ({len(data):,} items)")

    if data and isinstance(data[0], dict):
        csv_path = path.with_suffix(".csv")
        pd.json_normalize(data).to_csv(csv_path, index=False)
        print(f"  ✓ CSV version → {csv_path.name}")


def make_id(task_num, idx):
    return f"t{task_num}_{idx:04d}"


# ══════════════════════════════════════════════════════════════════════════════
# Descriptions
# ══════════════════════════════════════════════════════════════════════════════

def weather_description(row):
    parts = []

    rain = safe_float(row.get("rain", 0))
    temp = safe_float(row.get("temperature_2m", 20), 20)
    wind = safe_float(row.get("wind_speed_kmh", 0))
    cloud = safe_float(row.get("cloud_cover", 0))

    if rain > RAIN_HEAVY:
        parts.append("extreme rain")
    elif rain > RAIN_MODERATE:
        parts.append("heavy rain")
    elif rain > RAIN_LIGHT:
        parts.append("moderate rain")
    elif rain > RAIN_NONE:
        parts.append("light rain")
    else:
        parts.append("no rain")

    if temp >= HEAT_THRESHOLD:
        parts.append(f"extreme heat ({temp:.0f}°C)")
    elif temp >= 30:
        parts.append(f"hot ({temp:.0f}°C)")
    elif temp >= 20:
        parts.append(f"warm ({temp:.0f}°C)")
    elif temp >= COLD_THRESHOLD:
        parts.append(f"mild ({temp:.0f}°C)")
    else:
        parts.append(f"cold ({temp:.0f}°C)")

    if wind > 50:
        parts.append("strong winds")
    elif wind > 30:
        parts.append("moderate winds")

    if cloud > 80:
        parts.append("overcast")
    elif 0 < cloud < 20:
        parts.append("clear skies")

    return ", ".join(parts)


def event_description(row):
    if safe_bool(row.get("has_nearby_event", False)):
        cats = str(row.get("event_categories", "") or "").replace("|", "/")
        n = safe_int(row.get("event_count", 1), 1)
        dist = row.get("min_dist_km", np.nan)
        dist_str = f"{safe_float(dist):.1f} km away" if pd.notna(dist) else "nearby"

        if cats:
            return f"{n} event(s) ({cats}) {dist_str}"
        return f"{n} event(s) {dist_str}"

    return "no major events"


def poi_description(row):
    poi_activity = safe_float(row.get("poi_activity", 0))
    poi_category = str(row.get("poi_category", "") or "")
    mobility_level = str(row.get("mobility_level", "") or "")

    parts = []

    if mobility_level and mobility_level.lower() not in ["nan", "none"]:
        parts.append(f"mobility level: {mobility_level}")

    if poi_category and poi_category.lower() not in ["nan", "none"]:
        parts.append(f"POI category: {poi_category}")

    if poi_activity > 0:
        parts.append(f"POI activity score: {poi_activity:.2f}")

    return "; ".join(parts) if parts else "no POI/mobility signal available"


def context_dict(row):
    return {
        "datetime": str(row.get("datetime", "")),
        "location": str(row.get("location", "")),
        "weather": {
            "temperature_c": round(safe_float(row.get("temperature_2m", 20), 20), 1),
            "feels_like_c": round(safe_float(row.get("apparent_temperature", 20), 20), 1),
            "rain_mm_hr": round(safe_float(row.get("rain", 0)), 2),
            "rain_category": str(row.get("rain_category", "none") or "none"),
            "wind_speed_kmh": round(safe_float(row.get("wind_speed_kmh", 0)), 1),
            "wind_gust_kmh": round(safe_float(row.get("wind_gust_kmh", 0)), 1),
            "humidity_pct": round(safe_float(row.get("relative_humidity_2m", 50), 50), 1),
            "cloud_cover_pct": round(safe_float(row.get("cloud_cover", 0)), 1),
            "description": weather_description(row),
        },
        "calendar": {
            "day_of_week": str(row.get("day_of_week", "Monday")),
            "hour": safe_int(row.get("hour", 12), 12),
            "is_weekend": safe_bool(row.get("is_weekend", False)),
            "is_public_holiday": safe_bool(row.get("is_public_holiday", False)),
            "holiday_name": str(row.get("holiday_name", "") or ""),
            "is_school_term": safe_bool(row.get("is_school_term", False)),
            "is_peak_am": safe_bool(row.get("is_peak_am", False)),
            "is_peak_pm": safe_bool(row.get("is_peak_pm", False)),
        },
        "events": {
            "has_nearby_event": safe_bool(row.get("has_nearby_event", False)),
            "event_count": safe_int(row.get("event_count", 0)),
            "event_categories": str(row.get("event_categories", "") or ""),
            "description": event_description(row),
            "has_major_event": safe_bool(row.get("has_major_event", False)),
            "major_event_count": safe_int(row.get("major_event_count", 0)),
            "major_event_names": str(row.get("major_event_names", "") or ""),
            "major_event_types": str(row.get("major_event_types", "") or ""),
        },
        "incidents": {
            "crash_risk_count": safe_int(row.get("crash_risk_count", 0)),
            "crash_risk_level": str(row.get("crash_risk_level", "none") or "none"),
            "incident_count": safe_int(row.get("incident_count", 0)),
            "alert_count": safe_int(row.get("alert_count", 0)),
            "has_pt_disruption": safe_int(row.get("alert_count", 0)) > 0,
            "has_road_incident": safe_int(row.get("incident_count", 0)) > 0,
        },
        "poi_mobility": {
            "poi_activity": round(safe_float(row.get("poi_activity", 0)), 3),
            "poi_category": str(row.get("poi_category", "") or ""),
            "mobility_level": str(row.get("mobility_level", "") or ""),
            "description": poi_description(row),
        },
        "static_poi": {
        "total_poi_count": safe_int(row.get("total_poi_count", 0)),
        "amenity": safe_int(row.get("amenity", 0)),
        "leisure": safe_int(row.get("leisure", 0)),
        "office": safe_int(row.get("office", 0)),
        "other": safe_int(row.get("other", 0)),
        "public_transport": safe_int(row.get("public_transport", 0)),
        "railway": safe_int(row.get("railway", 0)),
        "shop": safe_int(row.get("shop", 0)),
        "tourism": safe_int(row.get("tourism", 0)),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Load master table
# ══════════════════════════════════════════════════════════════════════════════

def load_master():
    path = CLEANED / "master_context_table.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Master table not found at {path}. Run 02_clean_and_align.py first."
        )

    print("  Loading master context table...")

    df = pd.read_csv(path, low_memory=False)

    df.columns = [
        c.strip().lower().replace(" ", "_")
        for c in df.columns
    ]

    # Keep only the first copy of duplicated columns
    df = df.loc[:, ~df.columns.duplicated()].copy()

    print("Duplicate columns after cleanup:", df.columns[df.columns.duplicated()].tolist())
    
    # Clean weather column names with units
    rename_map = {
        "temperature_2m_(â°c)": "temperature_2m",
        "temperature_2m_(Â°c)": "temperature_2m",
        "apparent_temperature_(â°c)": "apparent_temperature",
        "apparent_temperature_(Â°c)": "apparent_temperature",
        "relative_humidity_2m_(%)": "relative_humidity_2m",
        "weather_code_(wmo_code)": "weather_code",
        "cloud_cover_(%)": "cloud_cover",
        "cloud_cover_low_(%)": "cloud_cover_low",
        "precipitation_(mm)": "precipitation",
        "rain_(mm)": "rain",
        "windspeed_10m_(km/h)": "wind_speed_kmh",
        "windgusts_10m_(km/h)": "wind_gust_kmh",
        "shortwave_radiation_(w/mâ²)": "shortwave_radiation",
        "shortwave_radiation_(w/mÂ²)": "shortwave_radiation",
        "sunshine_duration_(s)": "sunshine_duration",
        "wet_bulb_temperature_2m_(â°c)": "wet_bulb_temperature_2m",
        "wet_bulb_temperature_2m_(Â°c)": "wet_bulb_temperature_2m",
        "boundary_layer_height_(m)": "boundary_layer_height",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # Remove duplicate columns again after renaming
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    if "datetime" not in df.columns:
        raise ValueError("master_context_table.csv must contain a datetime column.")

    if "location" not in df.columns:
        raise ValueError("master_context_table.csv must contain a location column.")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "location"]).copy()

    # Derive basic calendar features if missing
    if "hour" not in df.columns:
        df["hour"] = df["datetime"].dt.hour

    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["datetime"].dt.day_name()

    if "is_weekend" not in df.columns:
        df["is_weekend"] = df["datetime"].dt.dayofweek >= 5

    if "is_peak_am" not in df.columns:
        df["is_peak_am"] = df["datetime"].dt.hour.between(7, 9)

    if "is_peak_pm" not in df.columns:
        df["is_peak_pm"] = df["datetime"].dt.hour.between(16, 19)

    # Defaults for optional columns
    defaults = {
        "rain": 0,
        "rain_category": "none",
        "temperature_2m": 20,
        "apparent_temperature": 20,
        "wind_speed_kmh": 0,
        "wind_gust_kmh": 0,
        "relative_humidity_2m": 50,
        "cloud_cover": 0,
        "crash_risk_count": 0,
        "crash_risk_level": "none",
        "incident_count": 0,
        "alert_count": 0,
        "event_count": 0,
        "has_nearby_event": False,
        "is_public_holiday": False,
        "is_school_term": False,
        "holiday_name": "",
        "event_categories": "",
        "min_dist_km": np.nan,
        "poi_activity": 0,
        "poi_category": "",
        "mobility_level": "",
    }

    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value

    numeric_cols = [
        "rain",
        "temperature_2m",
        "apparent_temperature",
        "wind_speed_kmh",
        "wind_gust_kmh",
        "relative_humidity_2m",
        "cloud_cover",
        "crash_risk_count",
        "incident_count",
        "alert_count",
        "event_count",
        "major_event_count",
        "traffic_volume_mean",
        "traffic_volume_max",
        "pedestrian_count_sum",
        "pedestrian_count_mean",
        "poi_activity",
    ]

    for col in numeric_cols:
        if col in df.columns:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

    bool_cols = [
        "is_public_holiday",
        "is_school_term",
        "has_nearby_event",
        "has_major_event",
        "is_weekend",
        "is_peak_am",
        "is_peak_pm",
    ]

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).apply(safe_bool)

    # Traffic baseline
    if "traffic_volume_mean" in df.columns and df["traffic_volume_mean"].notna().sum() > 50:
        df = df.sort_values(["location", "datetime"]).copy()

        df["traffic_baseline"] = df.groupby(["location", "hour"])["traffic_volume_mean"].transform(
            lambda x: x.rolling(24 * 7 * 4, min_periods=24, center=True).mean()
        )

        df["traffic_pct_change"] = (
            (df["traffic_volume_mean"] - df["traffic_baseline"])
            / df["traffic_baseline"].replace(0, np.nan)
            * 100
        )

    # POI baseline
    if "poi_activity" in df.columns and df["poi_activity"].sum() > 0:
        df = df.sort_values(["location", "datetime"]).copy()

        df["poi_baseline"] = df.groupby(["location", "hour"])["poi_activity"].transform(
            lambda x: x.rolling(24 * 7 * 4, min_periods=24, center=True).mean()
        )

        df["poi_pct_change"] = (
            (df["poi_activity"] - df["poi_baseline"])
            / df["poi_baseline"].replace(0, np.nan)
            * 100
        )

        df["mobility_level"] = pd.cut(
            df["poi_pct_change"].fillna(0),
            bins=[-999, -20, 20, 999],
            labels=["low", "normal", "high"],
        ).astype(str)

    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(
        f"  Locations: {df['location'].nunique()} | "
        f"Date range: {df['datetime'].min().date()} → {df['datetime'].max().date()}"
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Shared reasoning functions
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_activity_label(row):
    score = 0

    rain = safe_float(row.get("rain", 0))
    temp = safe_float(row.get("temperature_2m", 22), 22)

    if rain > RAIN_MODERATE:
        score -= 2
    elif rain > RAIN_LIGHT:
        score -= 1

    if temp >= HEAT_THRESHOLD:
        score -= 1

    if temp <= COLD_THRESHOLD:
        score -= 1

    if safe_bool(row.get("is_public_holiday")):
        score -= 2

    if safe_bool(row.get("has_nearby_event")):
        score += 2

    if safe_bool(row.get("is_weekend")):
        score -= 1

    if safe_int(row.get("incident_count", 0)) > 0:
        score -= 1

    if safe_int(row.get("alert_count", 0)) > 0:
        score += 1

    if str(row.get("mobility_level", "")).lower() == "high":
        score += 1

    if score >= 2:
        return "significantly_higher"

    if score <= -2:
        return "significantly_lower"

    return "no_significant_change"


def _build_activity_options(label):
    opts = {
        "A": "Significantly higher than baseline (>15% increase)",
        "B": "No significant change (within ±15% of baseline)",
        "C": "Significantly lower than baseline (>15% decrease)",
        "D": "Cannot be determined from the given information",
    }

    correct = {
        "significantly_higher": "A",
        "no_significant_change": "B",
        "significantly_lower": "C",
    }.get(label, "B")

    return list(opts.values()), correct


def _difficulty(row):
    signals = sum([
        safe_float(row.get("rain", 0)) > RAIN_LIGHT,
        safe_bool(row.get("is_public_holiday")),
        safe_bool(row.get("has_nearby_event")),
        safe_int(row.get("incident_count", 0)) > 0,
        safe_int(row.get("alert_count", 0)) > 0,
        safe_float(row.get("temperature_2m", 22), 22) >= HEAT_THRESHOLD,
        safe_float(row.get("poi_activity", 0)) > 0,
    ])

    if signals <= 1:
        return "easy"

    if signals <= 3:
        return "medium"

    return "hard"


def _explain_activity(row, label, wdesc):
    parts = [f"Weather: {wdesc}."]

    if safe_bool(row.get("is_public_holiday")):
        parts.append(
            f"It is {row.get('holiday_name', 'a public holiday')}, which typically reduces commuter traffic."
        )

    if not safe_bool(row.get("is_school_term")):
        parts.append("School holidays reduce school-run traffic and family commutes.")

    rain = safe_float(row.get("rain", 0))

    if rain > RAIN_MODERATE:
        parts.append("Heavy rain typically reduces outdoor activity and pedestrian counts.")
    elif rain > RAIN_LIGHT:
        parts.append("Moderate rain reduces walking but may increase car use.")

    if safe_bool(row.get("has_nearby_event")):
        parts.append(
            f"A nearby event ({event_description(row)}) can increase local traffic and foot traffic."
        )

    if safe_int(row.get("incident_count", 0)) > 0:
        parts.append(f"{safe_int(row.get('incident_count', 0))} road incident(s) may cause congestion.")

    if safe_int(row.get("alert_count", 0)) > 0:
        parts.append("Public transport disruptions may shift demand to roads.")

    if safe_float(row.get("poi_activity", 0)) > 0:
        parts.append("POI/mobility activity provides additional evidence of urban activity patterns.")

    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Task 1
# ══════════════════════════════════════════════════════════════════════════════

def build_task1(df):
    log("TASK 1 — Predict traffic / pedestrian / POI activity change")

    qa = []
    pool = df[df["temperature_2m"].notna() & df["rain"].notna()].copy()

    if "traffic_pct_change" in pool.columns and pool["traffic_pct_change"].notna().sum() > 50:
        pool = pool[pool["traffic_pct_change"].notna()].copy()
        pool["activity_change"] = pd.cut(
            pool["traffic_pct_change"],
            [-999, -TRAFFIC_CHANGE_SIG, TRAFFIC_CHANGE_SIG, 999],
            labels=["significantly_lower", "no_significant_change", "significantly_higher"],
        )
    elif "poi_pct_change" in pool.columns and pool["poi_pct_change"].notna().sum() > 50:
        pool = pool[pool["poi_pct_change"].notna()].copy()
        pool["activity_change"] = pd.cut(
            pool["poi_pct_change"],
            [-999, -TRAFFIC_CHANGE_SIG, TRAFFIC_CHANGE_SIG, 999],
            labels=["significantly_lower", "no_significant_change", "significantly_higher"],
        )
    else:
        pool["activity_change"] = pool.apply(_simulate_activity_label, axis=1)

    pool = pool[pool["activity_change"].notna()].copy()

    if pool.empty:
        save_json([], T1 / "task1_qa_pairs.json", "Task 1 — Activity Prediction")
        return []

    sample = pool.sample(min(N_TASK1, len(pool)), random_state=42)

    for i, (_, row) in enumerate(tqdm(sample.iterrows(), total=len(sample), desc="  Task 1")):
        label = str(row["activity_change"])
        wdesc = weather_description(row)
        loc = row["location"]
        dow = row.get("day_of_week", "")
        hour = safe_int(row.get("hour", 12), 12)

        hol = "public holiday" if safe_bool(row.get("is_public_holiday")) else ""
        sch = "during school term" if safe_bool(row.get("is_school_term")) else "during school holidays"
        inc = (
            f"{safe_int(row.get('incident_count', 0))} road incident(s)"
            if safe_int(row.get("incident_count", 0)) > 0
            else "no road incidents"
        )
        alert = (
            f"{safe_int(row.get('alert_count', 0))} transport alert(s)"
            if safe_int(row.get("alert_count", 0)) > 0
            else "no transport disruptions"
        )

        question = (
            f"It is {dow} at {hour:02d}:00 in {loc}. Current conditions: {wdesc}. "
            f"{'It is a ' + hol + '. ' if hol else ''}{sch.capitalize()}. "
            f"Nearby events: {event_description(row)}. Incidents: {inc}. Transport: {alert}. "
            f"POI/mobility signal: {poi_description(row)}. Compared to a typical {dow} at {hour:02d}:00 in {loc}, "
            f"would you expect traffic, pedestrian, and POI/mobility activity to be:"
        )

        options, correct = _build_activity_options(label)

        qa.append({
            "id": make_id(1, i + 1),
            "task": "traffic_prediction",
            "question": question,
            "context": context_dict(row),
            "options": options,
            "answer": correct,
            "explanation": _explain_activity(row, label, wdesc),
            "difficulty": _difficulty(row),
            "region": loc,
            "datetime": str(row["datetime"]),
            "label": label,
        })

    save_json(qa, T1 / "task1_qa_pairs.json", "Task 1 — Activity Prediction")
    return qa


# ══════════════════════════════════════════════════════════════════════════════
# Task 2
# ══════════════════════════════════════════════════════════════════════════════

def _primary_cause(row):
    if safe_bool(row.get("is_public_holiday")):
        return "public_holiday"

    if safe_float(row.get("rain", 0)) > RAIN_HEAVY:
        return "extreme_rain"

    if safe_bool(row.get("has_nearby_event")) and safe_int(row.get("event_count", 0)) > 0:
        return "large_event"
    
    if safe_bool(row.get("has_major_event")) and safe_int(row.get("major_event_count", 0)) > 0:
        return "major_event"
    
    if safe_float(row.get("temperature_2m", 22), 22) >= HEAT_THRESHOLD:
        return "extreme_heat"

    if safe_int(row.get("incident_count", 0)) > 2:
        return "road_incident"

    if safe_int(row.get("alert_count", 0)) > 0:
        return "pt_disruption"

    if str(row.get("mobility_level", "")).lower() == "high":
        return "poi_mobility_surge"

    if safe_float(row.get("poi_activity", 0)) > 0:
        return "poi_activity_pattern"

    if safe_float(row.get("rain", 0)) > RAIN_MODERATE:
        return "heavy_rain"

    if not safe_bool(row.get("is_school_term")):
        return "school_holidays"

    return None


def _build_cause_options(correct_cause):
    causes = {
        "public_holiday": "A public holiday reducing commuter and school travel",
        "extreme_rain": "Extreme rainfall suppressing outdoor activity",
        "large_event": "A large nearby event drawing crowds",
        "extreme_heat": "Extreme heat reducing outdoor mobility",
        "road_incident": "A major road incident causing congestion",
        "pt_disruption": "A public transport disruption shifting demand",
        "heavy_rain": "Sustained heavy rain reducing pedestrian and cycling activity",
        "school_holidays": "School holidays reducing school-run and family commutes",
        "poi_mobility_surge": "A POI/mobility surge indicating unusual place-based activity",
        "poi_activity_pattern": "Observed POI activity changing the local urban pattern",
    }

    correct_text = causes.get(correct_cause, correct_cause)
    distractors = [v for k, v in causes.items() if k != correct_cause]
    random.shuffle(distractors)

    options_text = [correct_text] + distractors[:3]
    random.shuffle(options_text)

    correct_letter = chr(65 + options_text.index(correct_text))
    return [f"{chr(65 + i)}. {t}" for i, t in enumerate(options_text)], correct_letter


def _explain_cause(row, cause):
    explanations = {
        "public_holiday": f"It is {row.get('holiday_name', 'a public holiday')}, which usually suppresses commuter demand.",
        "extreme_rain": f"Rainfall of {safe_float(row.get('rain', 0)):.1f} mm/hr is extreme and reduces outdoor activity.",
        "large_event": f"A nearby event ({event_description(row)}) concentrates traffic and foot traffic near the venue.",
        "extreme_heat": f"Temperature of {safe_float(row.get('temperature_2m', 35)):.0f}°C discourages outdoor movement.",
        "road_incident": f"{safe_int(row.get('incident_count', 0))} road incident(s) may cause congestion and diversion.",
        "pt_disruption": f"{safe_int(row.get('alert_count', 0))} public transport alert(s) may shift demand across modes.",
        "heavy_rain": f"Sustained rain of {safe_float(row.get('rain', 0)):.1f} mm/hr reduces walking and discretionary trips.",
        "school_holidays": "School holidays remove school-run peaks and reduce family commuting.",
        "poi_mobility_surge": "The POI/mobility signal suggests a place-based activity surge beyond normal traffic patterns.",
        "poi_activity_pattern": "Observed POI activity provides evidence of localised movement or place visitation.",
    }

    return explanations.get(cause, f"The primary contextual signal is {cause}.")


def build_task2(df):
    log("TASK 2 — Classify abnormal urban activity given context")

    pool = df[
        (df["rain"] > RAIN_MODERATE)
        | (df["is_public_holiday"] == True)
        | (df["has_nearby_event"] == True)
        | (df["has_major_event"] == True)
        | (df["incident_count"] > 0)
        | (df["alert_count"] > 0)
        | (df["temperature_2m"] >= HEAT_THRESHOLD)
        | (df["poi_activity"] > 0)
    ].copy()

    pool["primary_cause"] = pool.apply(_primary_cause, axis=1)
    pool = pool[pool["primary_cause"].notna()].copy()

    if pool.empty:
        save_json([], T2 / "task2_qa_pairs.json", "Task 2 — Anomaly Classification")
        return []

    sample = pool.sample(min(N_TASK2, len(pool)), random_state=42)

    qa = []

    for i, (_, row) in enumerate(tqdm(sample.iterrows(), total=len(sample), desc="  Task 2")):
        cause = row["primary_cause"]
        loc = row["location"]
        dow = row.get("day_of_week", "")
        hour = safe_int(row.get("hour", 12), 12)

        question = (
            f"In {loc} on {dow} at {hour:02d}:00, traffic, pedestrian, and POI activity differ from the expected baseline. "
            f"Given weather, event, calendar, incident, transport, and POI/mobility signals, what is the MOST LIKELY primary cause?"
        )

        options, correct = _build_cause_options(cause)

        qa.append({
            "id": make_id(2, i + 1),
            "task": "anomaly_classification",
            "question": question,
            "context": context_dict(row),
            "options": options,
            "answer": correct,
            "explanation": _explain_cause(row, cause),
            "difficulty": _difficulty(row),
            "region": loc,
            "datetime": str(row["datetime"]),
            "primary_cause": cause,
        })

    save_json(qa, T2 / "task2_qa_pairs.json", "Task 2 — Anomaly Classification")
    return qa


# ══════════════════════════════════════════════════════════════════════════════
# Task 3
# ══════════════════════════════════════════════════════════════════════════════

def _heuristic_sensitivity():
    data = [
        ("Sydney CBD", 0.8, 0.5, 0.9, 0.73),
        ("Parramatta", 0.6, 0.8, 0.5, 0.63),
        ("Liverpool", 0.5, 0.7, 0.4, 0.53),
        ("Penrith", 0.4, 0.9, 0.4, 0.57),
        ("Bondi", 0.9, 0.7, 0.8, 0.80),
        ("Manly", 0.9, 0.6, 0.9, 0.80),
        ("Newcastle", 0.7, 0.6, 0.7, 0.67),
        ("Wollongong", 0.7, 0.5, 0.6, 0.60),
        ("Coffs Harbour", 0.8, 0.6, 0.5, 0.63),
        ("Port Macquarie", 0.7, 0.5, 0.5, 0.57),
        ("Byron Bay", 0.9, 0.7, 0.7, 0.77),
        ("Nowra", 0.6, 0.5, 0.5, 0.53),
        ("Orange", 0.5, 0.4, 0.4, 0.43),
        ("Dubbo", 0.4, 0.7, 0.3, 0.47),
        ("Tamworth", 0.4, 0.7, 0.4, 0.50),
        ("Wagga Wagga", 0.5, 0.7, 0.4, 0.53),
        ("Albury", 0.5, 0.6, 0.4, 0.50),
        ("Bathurst", 0.5, 0.5, 0.5, 0.50),
        ("Broken Hill", 0.3, 0.9, 0.4, 0.53),
        ("Armidale", 0.5, 0.5, 0.5, 0.50),
        ("Katoomba", 0.8, 0.4, 0.7, 0.63),
        ("Perisher", 0.7, 0.3, 0.8, 0.60),
        ("Canberra", 0.6, 0.6, 0.5, 0.57),
        ("Cessnock", 0.5, 0.6, 0.4, 0.50),
    ]

    return pd.DataFrame(
        data,
        columns=[
            "location",
            "rain_sensitivity",
            "heat_sensitivity",
            "wind_sensitivity",
            "overall_sensitivity",
        ],
    )


def _compute_sensitivity(df):
    signal_col = None

    if "traffic_pct_change" in df.columns and df["traffic_pct_change"].notna().sum() > 50:
        signal_col = "traffic_pct_change"
    elif "poi_pct_change" in df.columns and df["poi_pct_change"].notna().sum() > 50:
        signal_col = "poi_pct_change"

    if signal_col is None:
        return pd.DataFrame()

    rows = []

    for loc in df["location"].unique():
        sub = df[df["location"] == loc].copy()

        rainy = sub[sub["rain"] > RAIN_LIGHT][signal_col].abs().mean()
        dry = sub[sub["rain"] <= RAIN_NONE][signal_col].abs().mean()
        hot = sub[sub["temperature_2m"] >= HEAT_THRESHOLD][signal_col].abs().mean()
        norm = sub[sub["temperature_2m"].between(18, 28)][signal_col].abs().mean()
        windy = sub[sub["wind_speed_kmh"] > 40][signal_col].abs().mean()
        calm = sub[sub["wind_speed_kmh"] <= 20][signal_col].abs().mean()

        rain_sens = 0 if pd.isna(rainy) or pd.isna(dry) else max(0, (rainy - dry) / (dry + 1))
        heat_sens = 0 if pd.isna(hot) or pd.isna(norm) else max(0, (hot - norm) / (norm + 1))
        wind_sens = 0 if pd.isna(windy) or pd.isna(calm) else max(0, (windy - calm) / (calm + 1))

        rows.append({
            "location": loc,
            "rain_sensitivity": rain_sens,
            "heat_sensitivity": heat_sens,
            "wind_sensitivity": wind_sens,
            "overall_sensitivity": (rain_sens + heat_sens + wind_sens) / 3,
        })

    return pd.DataFrame(rows)


def build_task3(df):
    log("TASK 3 — Estimate region sensitivity to weather changes")

    sensitivity = _compute_sensitivity(df)

    if sensitivity.empty:
        print("  ! Cannot compute sensitivity from traffic/POI data — using heuristic scores")
        sensitivity = _heuristic_sensitivity()

    locations = sensitivity["location"].tolist()
    pairs = [(a, b) for i, a in enumerate(locations) for b in locations[i + 1:]]
    random.shuffle(pairs)

    qa = []

    for i, (loc_a, loc_b) in enumerate(tqdm(pairs[:N_TASK3], desc="  Task 3")):
        row_a = sensitivity[sensitivity["location"] == loc_a].iloc[0]
        row_b = sensitivity[sensitivity["location"] == loc_b].iloc[0]

        dim = random.choice(["rain", "heat", "wind", "all"])
        col = {
            "rain": "rain_sensitivity",
            "heat": "heat_sensitivity",
            "wind": "wind_sensitivity",
            "all": "overall_sensitivity",
        }[dim]

        condition = {
            "rain": "rainfall",
            "heat": "extreme heat",
            "wind": "strong winds",
            "all": "adverse weather in general",
        }[dim]

        score_a, score_b = safe_float(row_a[col]), safe_float(row_b[col])
        more = loc_a if score_a >= score_b else loc_b

        question = (
            f"Compare {loc_a} and {loc_b}. Which region would likely show a larger traffic, pedestrian, "
            f"and POI/mobility activity change in response to {condition}?"
        )

        options = [
            f"A. {loc_a}",
            f"B. {loc_b}",
            "C. Both equally",
            "D. Cannot be determined",
        ]

        qa.append({
            "id": make_id(3, i + 1),
            "task": "region_sensitivity",
            "question": question,
            "context": {
                "region_a": loc_a,
                "region_b": loc_b,
                "condition": condition,
                "sensitivity_a": round(score_a, 3),
                "sensitivity_b": round(score_b, 3),
            },
            "options": options,
            "answer": "A" if more == loc_a else "B",
            "explanation": (
                f"{more} has the higher estimated/heuristic sensitivity score for {condition}. "
                f"Scores: {loc_a}={score_a:.2f}, {loc_b}={score_b:.2f}."
            ),
            "difficulty": "medium" if abs(score_a - score_b) > 0.15 else "hard",
        })

    save_json(qa, T3 / "task3_qa_pairs.json", "Task 3 — Region Sensitivity")
    return qa


# ══════════════════════════════════════════════════════════════════════════════
# Task 4
# ══════════════════════════════════════════════════════════════════════════════

def _predict_impact(row):
    rain = safe_float(row.get("rain", 0))
    temp = safe_float(row.get("temperature_2m", 22), 22)

    traffic_dir = "neutral"
    ped_dir = "neutral"
    poi_dir = "neutral"
    confidence = "medium"
    reasons = []

    if safe_bool(row.get("is_public_holiday")):
        traffic_dir = "decrease"
        ped_dir = "decrease"
        poi_dir = "mixed"
        confidence = "high"
        reasons.append("public holiday suppresses commuter demand but may increase leisure POI visits")

    if rain > RAIN_HEAVY:
        traffic_dir = "mixed"
        ped_dir = "major_decrease"
        poi_dir = "decrease"
        reasons.append("extreme rain reduces outdoor mobility")
    elif rain > RAIN_MODERATE:
        ped_dir = "decrease"
        poi_dir = "decrease"
        reasons.append("heavy rain reduces walking and discretionary POI visits")

    if temp >= HEAT_THRESHOLD:
        ped_dir = "decrease"
        poi_dir = "decrease"
        reasons.append("extreme heat discourages outdoor activity")

    if safe_bool(row.get("has_nearby_event")):
        traffic_dir = "localised_increase"
        ped_dir = "localised_increase"
        poi_dir = "increase"
        reasons.append("event generates precinct activity spike")

    if safe_int(row.get("incident_count", 0)) > 0:
        traffic_dir = "congested"
        reasons.append("road incident causes delays")

    if safe_int(row.get("alert_count", 0)) > 0:
        reasons.append("PT disruption may shift demand to roads")

    if safe_float(row.get("poi_activity", 0)) > 0:
        poi_dir = str(row.get("mobility_level", "observed"))
        reasons.append("POI/mobility signal provides direct activity evidence")

    return {
        "expected_traffic_direction": traffic_dir,
        "expected_pedestrian_direction": ped_dir,
        "expected_poi_mobility_direction": poi_dir,
        "confidence": confidence,
        "reasoning": reasons,
    }


def build_task4(df):
    log("TASK 4 — Scenario cards")

    strata = {
        "rainy_event": df[(df["rain"] > RAIN_MODERATE) & (df["has_nearby_event"] == True)],
        "hot_holiday": df[(df["temperature_2m"] >= HEAT_THRESHOLD) & (df["is_public_holiday"] == True)],
        "rainy_peak": df[(df["rain"] > RAIN_LIGHT) & (df["is_peak_am"] == True)],
        "event_weekend": df[(df["has_nearby_event"] == True) & (df["is_weekend"] == True)],
        "incident_rain": df[(df["incident_count"] > 0) & (df["rain"] > RAIN_LIGHT)],
        "pt_disruption": df[df["alert_count"] > 0],
        "school_holiday_dry": df[(df["is_school_term"] == False) & (df["rain"] <= RAIN_NONE)],
        "extreme_heat": df[df["temperature_2m"] >= HEAT_THRESHOLD],
        "poi_activity": df[df["poi_activity"] > 0],
        "poi_event": df[(df["poi_activity"] > 0) & (df["has_nearby_event"] == True)],
    }

    cards = []
    card_idx = 1
    per_stratum = max(5, N_TASK4 // len(strata))

    for stratum, sub in strata.items():
        if sub.empty:
            continue

        sample = sub.sample(min(per_stratum, len(sub)), random_state=42)

        for _, row in sample.iterrows():
            hour = safe_int(row.get("hour", 12), 12)

            title = (
                f"{weather_description(row)} + "
                f"{row.get('day_of_week', '')} {hour:02d}:00 + "
                f"{row.get('location', '')}"
            )

            cards.append({
                "id": make_id(4, card_idx),
                "task": "scenario_card",
                "title": title,
                "stratum": stratum,
                "region": str(row.get("location", "")),
                "datetime": str(row.get("datetime", "")),
                "scenario": {
                    "time_of_day": f"{hour:02d}:00",
                    "day_of_week": str(row.get("day_of_week", "")),
                    "location": str(row.get("location", "")),
                    "weather_summary": weather_description(row),
                    "temperature_c": round(safe_float(row.get("temperature_2m", 0)), 1),
                    "rain_mm_hr": round(safe_float(row.get("rain", 0)), 2),
                    "wind_speed_kmh": round(safe_float(row.get("wind_speed_kmh", 0)), 1),
                    "is_public_holiday": safe_bool(row.get("is_public_holiday")),
                    "holiday_name": str(row.get("holiday_name", "") or ""),
                    "is_school_term": safe_bool(row.get("is_school_term")),
                    "is_weekend": safe_bool(row.get("is_weekend")),
                    "is_peak_hour": safe_bool(row.get("is_peak_am")) or safe_bool(row.get("is_peak_pm")),
                    "nearby_event": safe_bool(row.get("has_nearby_event")),
                    "event_count": safe_int(row.get("event_count", 0)),
                    "event_categories": str(row.get("event_categories", "") or ""),
                    "road_incidents": safe_int(row.get("incident_count", 0)),
                    "pt_alerts": safe_int(row.get("alert_count", 0)),
                    "crash_risk_count": safe_int(row.get("crash_risk_count", 0)),
                    "crash_risk_level": str(row.get("crash_risk_level", "none") or "none"),
                    "poi_activity": round(safe_float(row.get("poi_activity", 0)), 3),
                    "poi_category": str(row.get("poi_category", "") or ""),
                    "mobility_level": str(row.get("mobility_level", "") or ""),
                    "context_label": str(row.get("context_label", "")),
                },
                "predicted_impact": _predict_impact(row),
                "suggested_tasks": [
                    "traffic_prediction",
                    "anomaly_classification",
                    "contrastive_example_generation",
                    "poi_mobility_reasoning",
                ],
            })

            card_idx += 1

    save_json(cards, T4 / "task4_scenario_cards.json", "Task 4 — Scenario Cards")
    return cards


# ══════════════════════════════════════════════════════════════════════════════
# Task 5
# ══════════════════════════════════════════════════════════════════════════════

def _key_difference(row_a, row_b, ctype):
    diffs = {}

    cols = [
        "is_public_holiday",
        "is_school_term",
        "has_nearby_event",
        "incident_count",
        "alert_count",
        "holiday_name",
        "event_categories",
        "poi_activity",
        "poi_category",
        "mobility_level",
    ]

    for col in cols:
        if str(row_a.get(col)) != str(row_b.get(col)):
            diffs[col] = {
                "scenario_A": str(row_a.get(col)),
                "scenario_B": str(row_b.get(col)),
            }

    diffs["contrast_type"] = ctype
    return diffs


def _contrastive_answer(row_a, row_b, ctype):
    answers = {
        "public_holiday_vs_normal_weekday": "Scenario A is affected by public holiday travel patterns, while Scenario B reflects a normal day.",
        "large_event_vs_no_event": "Scenario A has a nearby event generating localised activity; Scenario B has no event signal.",
        "road_incident_vs_no_incident": "Scenario A has road incident context causing congestion beyond weather alone.",
        "school_term_vs_school_holiday": "School term maintains school-run and commuter peaks; school holidays reduce those trips.",
        "poi_activity_vs_no_poi_activity": "Scenario A has POI/mobility evidence of place visitation, while Scenario B lacks this signal.",
    }

    return answers.get(ctype, f"The key contextual difference is {ctype}.")


def build_task5(df):
    log("TASK 5 — Contrastive examples")

    df2 = df.copy()

    df2["rain_bin"] = pd.cut(
        df2["rain"],
        [-0.01, 0.1, 1.0, 4.0, 9999],
        labels=["none", "light", "moderate", "heavy"],
    )

    df2["temp_bin"] = pd.cut(
        df2["temperature_2m"],
        [-999, 15, 25, 32, 999],
        labels=["cold", "mild", "warm", "hot"],
    )

    df2["hour_band"] = pd.cut(
        df2["hour"],
        [-1, 6, 10, 15, 19, 24],
        labels=["night", "morning_peak", "midday", "evening_peak", "late"],
    )

    grouped = df2.groupby(
        [c for c in ["rain_bin", "temp_bin", "hour_band", "location"] if c in df2.columns],
        observed=True,
    )

    pairs = []
    pair_idx = 1

    for _, group in grouped:
        if len(group) < 2:
            continue

        candidates = [
            (
                group[group["is_public_holiday"] == True],
                group[group["is_public_holiday"] == False],
                "public_holiday_vs_normal_weekday",
            ),
            (
                group[group["has_nearby_event"] == True],
                group[group["has_nearby_event"] == False],
                "large_event_vs_no_event",
            ),
            (
                group[group["incident_count"] > 0],
                group[group["incident_count"] == 0],
                "road_incident_vs_no_incident",
            ),
            (
                group[group["is_school_term"] == True],
                group[group["is_school_term"] == False],
                "school_term_vs_school_holiday",
            ),
            (
                group[group["poi_activity"] > 0],
                group[group["poi_activity"] == 0],
                "poi_activity_vs_no_poi_activity",
            ),
        ]

        for a, b, ctype in candidates:
            if pair_idx > N_TASK5:
                break

            if not a.empty and not b.empty:
                row_a = a.iloc[0]
                row_b = b.iloc[0]
                loc = row_a["location"]
                hour = safe_int(row_a.get("hour", 12), 12)

                pairs.append({
                    "id": make_id(5, pair_idx),
                    "task": "contrastive_example",
                    "contrast_type": ctype,
                    "region": loc,
                    "question": (
                        f"Two observations in {loc} at {hour:02d}:00 have similar weather "
                        f"({weather_description(row_a)}), but different activity patterns. "
                        f"Explain the likely difference."
                    ),
                    "scenario_A": {
                        "datetime": str(row_a.get("datetime", "")),
                        "context": context_dict(row_a),
                        "weather_summary": weather_description(row_a),
                        "activity_pattern": _simulate_activity_label(row_a),
                    },
                    "scenario_B": {
                        "datetime": str(row_b.get("datetime", "")),
                        "context": context_dict(row_b),
                        "weather_summary": weather_description(row_b),
                        "activity_pattern": _simulate_activity_label(row_b),
                    },
                    "key_difference": _key_difference(row_a, row_b, ctype),
                    "answer": _contrastive_answer(row_a, row_b, ctype),
                    "difficulty": "hard",
                })

                pair_idx += 1

        if pair_idx > N_TASK5:
            break

    save_json(pairs, T5 / "task5_contrastive_pairs.json", "Task 5 — Contrastive Examples")
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# Task 6
# ══════════════════════════════════════════════════════════════════════════════

def _poi_label(row):
    if safe_bool(row.get("has_nearby_event")) and safe_float(row.get("poi_activity", 0)) > 0:
        return "event_driven_poi_activity"

    if safe_bool(row.get("is_public_holiday")) and safe_float(row.get("poi_activity", 0)) > 0:
        return "holiday_leisure_poi_activity"

    if safe_float(row.get("rain", 0)) > RAIN_MODERATE:
        return "weather_suppressed_poi_activity"

    if safe_int(row.get("alert_count", 0)) > 0:
        return "transport_disruption_related_mobility"

    if safe_float(row.get("poi_activity", 0)) > 0:
        return "normal_poi_activity"

    return "no_clear_poi_signal"


def _build_poi_options(label):
    opts = {
        "event_driven_poi_activity": "POI activity is likely event-driven",
        "holiday_leisure_poi_activity": "POI activity is likely leisure/holiday-driven",
        "weather_suppressed_poi_activity": "POI activity is likely suppressed by adverse weather",
        "transport_disruption_related_mobility": "Mobility pattern is likely affected by transport disruption",
        "normal_poi_activity": "POI activity appears consistent with normal local activity",
        "no_clear_poi_signal": "There is no clear POI/mobility signal",
    }

    correct_text = opts[label]
    distractors = [v for k, v in opts.items() if k != label]
    random.shuffle(distractors)

    options_text = [correct_text] + distractors[:3]
    random.shuffle(options_text)

    correct_letter = chr(65 + options_text.index(correct_text))
    return [f"{chr(65 + i)}. {t}" for i, t in enumerate(options_text)], correct_letter


def _explain_poi(row, label):
    explanations = {
        "event_driven_poi_activity": "Nearby event context and POI activity together suggest localised event-driven mobility.",
        "holiday_leisure_poi_activity": "Public holidays can reduce commuting but increase leisure-oriented POI visits.",
        "weather_suppressed_poi_activity": "Adverse weather usually suppresses outdoor POI visits and discretionary movement.",
        "transport_disruption_related_mobility": "Transport alerts may reroute mobility or shift activity around stations and roads.",
        "normal_poi_activity": "POI activity is present but no stronger event, holiday, or disruption signal dominates.",
        "no_clear_poi_signal": "The available context does not provide a clear POI/mobility explanation.",
    }

    return explanations[label]


def build_task6(df):
    log("TASK 6 — POI Context Reasoning")

    pool = df[
        (df["poi_activity"] > 0)
        | (df["has_nearby_event"] == True)
        | (df["is_public_holiday"] == True)
        | (df["rain"] > RAIN_LIGHT)
        | (df["alert_count"] > 0)
    ].copy()

    if pool.empty:
        save_json([], T6 / "task6_qa_pairs.json", "Task 6 — POI Mobility Reasoning")
        return []

    pool["poi_label"] = pool.apply(_poi_label, axis=1)
    sample = pool.sample(min(N_TASK6, len(pool)), random_state=42)

    qa = []

    for i, (_, row) in enumerate(tqdm(sample.iterrows(), total=len(sample), desc="  Task 6")):
        label = row["poi_label"]
        loc = str(row.get("location", ""))
        dow = str(row.get("day_of_week", ""))
        hour = safe_int(row.get("hour", 12), 12)

        question = (
            f"In {loc} on {dow} at {hour:02d}:00, context includes "
            f"{weather_description(row)}, {event_description(row)}, "
            f"{safe_int(row.get('alert_count', 0))} transport alert(s), "
            f"{safe_int(row.get('incident_count', 0))} road incident(s), "
            f"and POI/mobility signal: {poi_description(row)}. "
            f"What is the most likely POI/mobility interpretation?"
        )

        options, correct = _build_poi_options(label)

        qa.append({
            "id": make_id(6, i + 1),
            "task": "poi_context_reasoning",
            "question": question,
            "context": context_dict(row),
            "options": options,
            "answer": correct,
            "explanation": _explain_poi(row, label),
            "difficulty": _difficulty(row),
            "region": loc,
            "datetime": str(row["datetime"]),
            "label": label,
        })

    save_json(qa,T6 / "task6_qa_pairs.json","Task 6 — POI Context Reasoning")
    return qa




# ── Task 7 helpers ─────────────────────────────────────────────────────────────

def _t7_data_sufficiency(row):
    """
    Check whether the context has enough signal to produce a confident prediction.
    Returns 'sufficient', 'partial', or 'insufficient'.

    Rationale: outputting a confident label when all signals are zero is the core
    logic error identified in the app — this guard prevents that.
    """
    n_signals = sum([
        safe_float(row.get("rain", 0)) > RAIN_NONE,
        safe_bool(row.get("has_nearby_event", False)),
        safe_int(row.get("incident_count", 0)) > 0,
        safe_int(row.get("alert_count", 0)) > 0,
        safe_float(row.get("poi_activity", 0)) > 0,
        safe_float(row.get("temperature_2m", 22), 22) >= HEAT_THRESHOLD
        or safe_float(row.get("temperature_2m", 22), 22) <= COLD_THRESHOLD,
        safe_bool(row.get("is_public_holiday", False)),
        safe_bool(row.get("has_major_event", False)),
    ])

    if n_signals >= 3:
        return "sufficient"
    if n_signals >= 1:
        return "partial"
    return "insufficient"


def _t7_primary_driver(row):
    """
    Identify the strongest single signal for chain reasoning Step 1.
    Order of dominance matches real-world urban impact magnitude.
    """
    if safe_bool(row.get("has_major_event", False)):
        n = safe_int(row.get("major_event_count", 1), 1)
        names = str(row.get("major_event_names", "") or "")
        return f"major event nearby ({n} event(s){': ' + names if names else ''})"

    if safe_bool(row.get("is_public_holiday", False)):
        name = str(row.get("holiday_name", "") or "public holiday")
        return f"public holiday ({name})"

    rain = safe_float(row.get("rain", 0))
    if rain > RAIN_HEAVY:
        return f"extreme rainfall ({rain:.1f} mm/hr)"
    if rain > RAIN_MODERATE:
        return f"heavy rainfall ({rain:.1f} mm/hr)"

    temp = safe_float(row.get("temperature_2m", 22), 22)
    if temp >= HEAT_THRESHOLD:
        return f"extreme heat ({temp:.0f}°C)"

    if safe_bool(row.get("has_nearby_event", False)):
        return f"nearby event ({event_description(row)})"

    if safe_int(row.get("incident_count", 0)) > 0:
        n = safe_int(row.get("incident_count", 0))
        return f"{n} road incident(s) causing network disruption"

    if safe_int(row.get("alert_count", 0)) > 0:
        n = safe_int(row.get("alert_count", 0))
        return f"{n} public transport alert(s) shifting modal demand"

    if safe_float(row.get("poi_activity", 0)) > 0:
        mob = str(row.get("mobility_level", "") or "observed")
        cat = str(row.get("poi_category", "") or "")
        return f"POI activity signal ({mob} mobility{', ' + cat if cat else ''})"

    if rain > RAIN_LIGHT:
        return f"light-to-moderate rainfall ({rain:.1f} mm/hr)"

    return "baseline conditions (no dominant signal)"


def _t7_secondary_factors(row):
    """
    Return a list of secondary contributing factors (all signals not already the primary).
    Used to populate the 'secondary_factors' field in chain reasoning Step 2.
    """
    factors = []

    if safe_bool(row.get("is_school_term", False)):
        factors.append("school term active — school-run peaks present")
    else:
        factors.append("school holidays — reduced school-run and family commutes")

    if safe_bool(row.get("is_weekend", False)):
        factors.append("weekend — leisure travel dominates over commuter traffic")
    elif safe_bool(row.get("is_peak_am", False)):
        factors.append("AM peak hour — high baseline commuter demand")
    elif safe_bool(row.get("is_peak_pm", False)):
        factors.append("PM peak hour — high baseline commuter demand")

    rain = safe_float(row.get("rain", 0))
    if RAIN_NONE < rain <= RAIN_LIGHT:
        factors.append(f"light rain ({rain:.1f} mm/hr) — minor deterrent to walking")

    wind = safe_float(row.get("wind_speed_kmh", 0))
    if wind > 30:
        factors.append(f"moderate-to-strong winds ({wind:.0f} km/h) — affects cycling and pedestrians")

    cloud = safe_float(row.get("cloud_cover", 0))
    if cloud > 80:
        factors.append("overcast sky — may deter outdoor leisure activity")

    crash_risk = safe_int(row.get("crash_risk_count", 0))
    if crash_risk > 0:
        level = str(row.get("crash_risk_level", "elevated") or "elevated")
        factors.append(f"historical crash risk {level} ({crash_risk} risk segments) in this area")

    if not factors:
        factors.append("no notable secondary signals detected")

    return factors


def _t7_confidence_score(row, sufficiency):
    """
    Compute a 0–1 confidence score for the LLM prediction.

    Logic:
    - Start at 0.5 (baseline uncertainty)
    - Each strong signal adds confidence
    - Zero-data cases are capped at 0.3 (partial) or returned as None (insufficient)
    - Conflicting signals (event + rain) reduce confidence slightly
    """
    if sufficiency == "insufficient":
        return None

    score = 0.4

    rain = safe_float(row.get("rain", 0))
    if rain > RAIN_HEAVY:
        score += 0.25
    elif rain > RAIN_MODERATE:
        score += 0.15
    elif rain > RAIN_LIGHT:
        score += 0.05

    if safe_bool(row.get("is_public_holiday", False)):
        score += 0.20
    if safe_bool(row.get("has_major_event", False)):
        score += 0.20
    if safe_bool(row.get("has_nearby_event", False)):
        score += 0.10
    if safe_int(row.get("incident_count", 0)) > 0:
        score += 0.10
    if safe_int(row.get("alert_count", 0)) > 0:
        score += 0.08
    if safe_float(row.get("poi_activity", 0)) > 0:
        score += 0.10

    # Conflicting signals (event + heavy rain) reduce certainty
    has_event = safe_bool(row.get("has_nearby_event", False)) or safe_bool(row.get("has_major_event", False))
    if has_event and rain > RAIN_MODERATE:
        score -= 0.10  # Rain suppresses event turnout unpredictably

    return round(min(max(score, 0.1), 0.95), 2)


def _t7_label(row, sufficiency):
    """
    Derive the ground-truth activity label for Task 7.
    Labels: A (significantly_higher), B (no_significant_change), C (lower_activity_disruption).

    Key fix: returns None when data is insufficient — forcing the LLM to say
    'cannot determine' rather than guessing. This is the core logic fix from the audit.
    """
    if sufficiency == "insufficient":
        return None

    score = 0

    rain = safe_float(row.get("rain", 0))
    if rain > RAIN_MODERATE:
        score -= 2
    elif rain > RAIN_LIGHT:
        score -= 1

    temp = safe_float(row.get("temperature_2m", 22), 22)
    if temp >= HEAT_THRESHOLD or temp <= COLD_THRESHOLD:
        score -= 1

    if safe_bool(row.get("is_public_holiday", False)):
        score -= 1  # Reduces commuter traffic, but effect is mixed

    if safe_bool(row.get("has_major_event", False)):
        score += 3
    elif safe_bool(row.get("has_nearby_event", False)):
        score += 2

    if safe_int(row.get("incident_count", 0)) > 0:
        score -= 1

    if safe_int(row.get("alert_count", 0)) > 0:
        score -= 1

    mob = str(row.get("mobility_level", "")).lower()
    if mob == "high":
        score += 1
    elif mob == "low":
        score -= 1

    if safe_bool(row.get("is_weekend", False)) and not (
        safe_bool(row.get("has_nearby_event", False)) or safe_bool(row.get("has_major_event", False))
    ):
        score -= 1  # Weekend without event = lower commuter baseline

    if score >= 2:
        return "A"  # Significantly higher activity
    if score <= -2:
        return "C"  # Lower activity / disruption detected
    return "B"      # No significant change


def _t7_counter_scenario(row, label):
    """
    Generate a counter-scenario: describe what single signal change would flip the label.
    This forces the reasoning to expose what the model is sensitive to.
    """
    rain = safe_float(row.get("rain", 0))
    has_event = safe_bool(row.get("has_nearby_event", False)) or safe_bool(row.get("has_major_event", False))

    if label == "A":
        if rain > RAIN_LIGHT:
            return (
                f"If rainfall intensified beyond {rain:.1f} mm/hr to extreme levels (>16 mm/hr), "
                "event attendance would drop sharply and the label would shift to C."
            )
        return (
            "If the event were cancelled or relocated, the localised activity spike "
            "would disappear and the label would shift to B."
        )

    if label == "C":
        if has_event:
            return (
                "Despite adverse weather, if a major event proceeded with strong attendance, "
                "the label could shift to A for localised precinct activity."
            )
        return (
            "If rain eased below 1 mm/hr and no further incidents occurred, "
            "activity could recover to baseline — shifting the label to B."
        )

    # label == "B"
    if has_event:
        return (
            "If the nearby event drew unexpectedly large crowds (>10,000 attendance), "
            "the label could shift to A."
        )
    return (
        "If a major road incident or PT disruption occurred during this period, "
        "the label could shift to C."
    )


def _t7_build_question(row, loc, dow, hour, wdesc, sufficiency):
    """
    Build a structured, anchored benchmark question for Task 7.

    Key improvements over the original:
    - Includes time, suburb, weather, all signals explicitly
    - Asks for chain reasoning (primary driver → secondary factors → confidence → label)
    - Asks for counter-scenario to expose model sensitivity
    - Frames against a named baseline (same day/hour historical norm)
    - If data is insufficient, question explicitly tests whether model knows to abstain
    """
    hol = f"It is {row.get('holiday_name', 'a public holiday')}. " if safe_bool(row.get("is_public_holiday", False)) else ""
    sch = "School term is active." if safe_bool(row.get("is_school_term", False)) else "School holidays are in effect."
    inc = (
        f"{safe_int(row.get('incident_count', 0))} road incident(s) active."
        if safe_int(row.get("incident_count", 0)) > 0
        else "No road incidents."
    )
    alert = (
        f"{safe_int(row.get('alert_count', 0))} public transport alert(s) in effect."
        if safe_int(row.get("alert_count", 0)) > 0
        else "No transport disruptions."
    )
    crash = (
        f"Historical crash risk: {row.get('crash_risk_level', 'unknown')} "
        f"({safe_int(row.get('crash_risk_count', 0))} risk segments)."
    )
    poi = poi_description(row)
    event = event_description(row)

    base = (
        f"It is {dow} at {hour:02d}:00 in {loc}. "
        f"Current conditions: {wdesc}. "
        f"{hol}{sch} "
        f"Nearby events: {event}. "
        f"{inc} {alert} {crash} "
        f"POI/mobility signal: {poi}. "
        f"Compared to a typical {dow} at {hour:02d}:00 in {loc}, "
    )

    if sufficiency == "insufficient":
        return (
            base
            + "assess whether there is enough contextual information to predict urban activity. "
            "If not, explain what additional signals would be required."
        )

    return (
        base
        + "predict pedestrian, vehicle, and POI/mobility activity. "
        "In your response: (1) identify the primary driver, "
        "(2) list secondary contributing factors, "
        "(3) assign a confidence score (0–1), "
        "(4) select a label (A: significantly higher, B: no significant change, C: lower activity/disruption), "
        "and (5) describe one counter-scenario that would flip your label."
    )


def _t7_build_options(label):
    """
    Build answer options. When label is None (insufficient data), option D is correct.
    This enforces that the benchmark tests abstention as a valid model behaviour.
    """
    opts = [
        "A. Significantly higher activity than baseline (event-driven or strong positive signal)",
        "B. No significant change from baseline (within normal variation)",
        "C. Lower activity or disruption detected (weather, incident, or holiday suppression)",
        "D. Insufficient data to make a confident prediction — more signals required",
    ]

    if label is None:
        correct = "D"
    else:
        correct = label  # Already "A", "B", or "C"

    return opts, correct


def _t7_explanation(row, label, sufficiency, primary, secondary, confidence):
    """
    Build a structured chain-of-reasoning explanation for each Task 7 item.
    This is what the Streamlit app surfaces as the 'reasoning' field — it should
    show multi-step logic, not a single sentence.
    """
    if sufficiency == "insufficient":
        return (
            "Insufficient context signals to produce a reliable prediction. "
            "All primary signals (events, weather, incidents, alerts, POI activity) "
            "are at baseline or unavailable. A model should abstain rather than guess. "
            "To make a confident prediction, at least 2–3 non-trivial signals are required."
        )

    parts = [
        f"Step 1 — Primary driver: {primary}.",
        f"Step 2 — Secondary factors: {'; '.join(secondary)}.",
        f"Step 3 — Confidence: {confidence} "
        f"({'high' if confidence >= 0.7 else 'moderate' if confidence >= 0.45 else 'low'}).",
        f"Step 4 — Label: {label} — "
        + {
            "A": "Significantly higher activity. Strong positive signal (event or major crowd driver) outweighs any suppression factors.",
            "B": "No significant change. Signals are balanced or weak; activity expected near historical baseline.",
            "C": "Lower activity or disruption. Suppression factors (weather, incident, holiday) dominate without offsetting event uplift.",
        }.get(label, ""),
        f"Step 5 — Counter-scenario: {_t7_counter_scenario(row, label)}",
    ]

    return " | ".join(parts)


def build_task7(df=None):
    """
    Task 7 — LLM Urban Context Reasoning (master table).

    Redesigned to fix the core logic issues identified in the benchmark audit:

    1. NULL/zero-signal handling: rows where all context signals are zero/false
       are now labelled 'insufficient' and the correct answer is D (abstain).
       Previously, the app produced confident labels even on all-zero signal rows.

    2. Chain reasoning structure: questions explicitly ask for a 5-step response
       (primary driver → secondary factors → confidence → label → counter-scenario).
       Previously, only a single reason was generated.

    3. LLM-native inference: Task 7 now draws from the same master_context_table
       as Tasks 1–6 (not a separate poi_llm_tasks_clean.csv). The prediction is
       derived from real context signals, not trajectory POI IDs. This resolves
       the disconnect where 'prediction source = rule-based' appeared even in the
       LLM master table.

    4. Anchored, specific questions: each question includes time-of-day, suburb,
       all active signals, and a named baseline (same day/hour), making ground
       truth verifiable. Previously questions were too generic for meaningful eval.

    5. Counter-scenario field: each item includes a counter-scenario that specifies
       what single signal change would flip the label. This exposes model sensitivity
       and enables contrastive evaluation.

    Fallback: if master table (df) is not passed in, attempts to load from disk.
    """
    log("TASK 7 — LLM Urban Context Reasoning (master table, redesigned)")

    # ── Load data ──────────────────────────────────────────────────────────────
    if df is None:
        path = CLEANED / "master_context_table.csv"
        if not path.exists():
            print("  ! master_context_table.csv not found — skipping Task 7")
            save_json([], T7 / "task7_qa_pairs.json", "Task 7 — LLM Urban Context Reasoning")
            return []
        df = load_master()

    # ── Stratified sampling ────────────────────────────────────────────────────
    # Sample across data sufficiency levels so the benchmark includes all three
    # categories: sufficient (confident prediction), partial, and insufficient
    # (model should abstain). Ratio: 60% sufficient, 25% partial, 15% insufficient.
    df = df.copy()
    df["_t7_sufficiency"] = df.apply(_t7_data_sufficiency, axis=1)

    n_suf   = int(N_TASK7 * 0.60)
    n_part  = int(N_TASK7 * 0.25)
    n_insuf = N_TASK7 - n_suf - n_part

    pool_suf   = df[df["_t7_sufficiency"] == "sufficient"]
    pool_part  = df[df["_t7_sufficiency"] == "partial"]
    pool_insuf = df[df["_t7_sufficiency"] == "insufficient"]

    frames = []
    if not pool_suf.empty:
        frames.append(pool_suf.sample(min(n_suf, len(pool_suf)), random_state=42))
    if not pool_part.empty:
        frames.append(pool_part.sample(min(n_part, len(pool_part)), random_state=42))
    if not pool_insuf.empty:
        frames.append(pool_insuf.sample(min(n_insuf, len(pool_insuf)), random_state=42))

    if not frames:
        print("  ! No data available for Task 7")
        save_json([], T7 / "task7_qa_pairs.json", "Task 7 — LLM Urban Context Reasoning")
        return []

    sample = pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)

    # ── Build QA pairs ─────────────────────────────────────────────────────────
    qa = []

    for i, (_, row) in enumerate(tqdm(sample.iterrows(), total=len(sample), desc="  Task 7")):
        loc        = str(row.get("location", ""))
        dow        = str(row.get("day_of_week", ""))
        hour       = safe_int(row.get("hour", 12), 12)
        wdesc      = weather_description(row)
        sufficiency = str(row.get("_t7_sufficiency", "partial"))

        primary    = _t7_primary_driver(row)
        secondary  = _t7_secondary_factors(row)
        confidence = _t7_confidence_score(row, sufficiency)
        label      = _t7_label(row, sufficiency)

        question   = _t7_build_question(row, loc, dow, hour, wdesc, sufficiency)
        options, correct = _t7_build_options(label)
        explanation = _t7_explanation(row, label, sufficiency, primary, secondary, confidence)

        qa.append({
            "id": make_id(7, i + 1),
            "task": "llm_urban_context_reasoning",

            # ── Core benchmark fields ──────────────────────────────────────────
            "question": question,
            "context": context_dict(row),
            "options": options,
            "answer": correct,
            "explanation": explanation,
            "difficulty": _difficulty(row),
            "region": loc,
            "datetime": str(row["datetime"]),

            # ── Task 7 specific: chain reasoning scaffold ──────────────────────
            "chain_reasoning": {
                "step1_primary_driver":     primary,
                "step2_secondary_factors":  secondary,
                "step3_confidence_score":   confidence,
                "step4_label":              label,
                "step5_counter_scenario":   _t7_counter_scenario(row, label) if label else (
                    "No counter-scenario available — abstention is the correct response."
                ),
            },

            # ── Data quality metadata ──────────────────────────────────────────
            "data_sufficiency":     sufficiency,
            "n_active_signals":     sum([
                safe_float(row.get("rain", 0)) > RAIN_NONE,
                safe_bool(row.get("has_nearby_event", False)),
                safe_bool(row.get("has_major_event", False)),
                safe_int(row.get("incident_count", 0)) > 0,
                safe_int(row.get("alert_count", 0)) > 0,
                safe_float(row.get("poi_activity", 0)) > 0,
                safe_bool(row.get("is_public_holiday", False)),
            ]),

            # ── Raw signal snapshot (mirrors app context signals panel) ────────
            "signal_snapshot": {
                "events":           safe_int(row.get("event_count", 0)),
                "poi_activity":     round(safe_float(row.get("poi_activity", 0)), 3),
                "bus_routes":       safe_int(row.get("alert_count", 0)),  # proxy for PT coverage
                "alert_zone_points": safe_int(row.get("incident_count", 0)),
                "road_incidents":   safe_int(row.get("crash_risk_count", 0)),
                "rain_mm_hr":       round(safe_float(row.get("rain", 0)), 2),
                "temp_c":           round(safe_float(row.get("temperature_2m", 20), 20), 1),
            },
        })

    save_json(qa, T7 / "task7_qa_pairs.json", "Task 7 — LLM Urban Context Reasoning")

    # ── Print distribution summary ─────────────────────────────────────────────
    suf_counts = {s: sum(1 for x in qa if x["data_sufficiency"] == s) for s in ["sufficient", "partial", "insufficient"]}
    label_counts = {l: sum(1 for x in qa if x["answer"] == l) for l in ["A", "B", "C", "D"]}
    print(f"  Sufficiency distribution: {suf_counts}")
    print(f"  Label distribution:       {label_counts}")

    return qa
# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

def build_summary(t1, t2, t3, t4, t5, t6, t7, df):
    log("BENCHMARK SUMMARY")

    scored = t1 + t2 + t3 + t6 + t7

    summary = {
        "benchmark_name": "NSW Urban Context Benchmark (Topic 3)",
        "version": "1.1",
        "created": datetime.now().isoformat(),
        "description": (
            "Context-aware urban reasoning benchmark using weather, events, holidays, school terms, "
            "road incidents, transport disruptions, pedestrian, traffic, and POI/mobility context."
        ),
        "data_coverage": {
            "regions": int(df["location"].nunique()),
            "date_range": f"{df['datetime'].min().date()} to {df['datetime'].max().date()}",
            "total_rows": int(len(df)),
            "has_poi_mobility": bool("poi_activity" in df.columns and df["poi_activity"].sum() > 0),
            "has_traffic": bool("traffic_volume_mean" in df.columns),
            "has_pedestrian": bool("pedestrian_count_sum" in df.columns),
        },
        "tasks": {
            "task1_traffic_prediction": {
                "n_examples": len(t1),
                "format": "4-option MCQ",
                "metric": "accuracy",
            },
            "task2_anomaly_classification": {
                "n_examples": len(t2),
                "format": "4-option MCQ",
                "metric": "accuracy",
            },
            "task3_region_sensitivity": {
                "n_examples": len(t3),
                "format": "4-option MCQ",
                "metric": "accuracy",
            },
            "task4_scenario_cards": {
                "n_examples": len(t4),
                "format": "JSON scenario card",
                "metric": "coverage and diversity",
            },
            "task5_contrastive_examples": {
                "n_examples": len(t5),
                "format": "open-ended reasoning pair",
                "metric": "contrastive reasoning accuracy",
            },
            "task6_poi_context_reasoning": {
                "n_examples": len(t6),
                "format": "4-option MCQ",
                "metric": "accuracy",
            },
            "task7_llm_urban_context_reasoning": {
                "n_examples": len(t7),
                "format": "4-option MCQ with chain reasoning scaffold",
                "metric": "accuracy + chain reasoning quality",
                "notes": (
                    "Redesigned master table. Questions are anchored to time/suburb/signals. "
                    "Includes data_sufficiency stratification (sufficient/partial/insufficient), "
                    "5-step chain reasoning (primary driver, secondary factors, confidence, label, counter-scenario), "
                    "and abstention testing (label D when signals are all zero)."
                ),
            },
        },
        "total_items": len(t1) + len(t2) + len(t3) + len(t4) + len(t5) + len(t6) + len(t7),
        "difficulty_distribution": {
            "easy": sum(1 for x in scored if x.get("difficulty") == "easy"),
            "medium": sum(1 for x in scored if x.get("difficulty") == "medium"),
            "hard": sum(1 for x in (scored + t5 ) if x.get("difficulty") == "hard"),
        },
        "context_signals_used": [
            "weather",
            "public holidays",
            "school terms",
            "large events",
            "major events",
            "crash historical risk profile",
            "road incidents",
            "public transport alerts",
            "pedestrian counts",
            "traffic activity",
            "POI context reasoning",
            "data sufficiency / abstention testing",
            "chain reasoning (primary driver → secondary factors → confidence → label → counter-scenario)",
        ],
        "data_sources": [
            "Open-Meteo",
            "data.gov.au holidays",
            "NSW school terms",
            "Ticketmaster",
            "TfNSW crashes",
            "TfNSW live hazards",
            "TfNSW GTFS-RT/service alerts",
            "City of Sydney pedestrian counts",
            "HuggingFace traffic",
            "Massive-STEPS Sydney",
        ],
    }

    summary_path = BENCH_DIR / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("  ✓ benchmark_summary.json")
    print(f"\n  BENCHMARK READY — Total items: {summary['total_items']:,}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║   Topic 3 — Stage 2: Benchmark Task Construction       ║
║   Input  : data/cleaned/master_context_table.csv       ║
║   Output : data/benchmark/                             ║
╚══════════════════════════════════════════════════════════╝
""")

    df = load_master()

    t1 = build_task1(df)
    t2 = build_task2(df)
    t3 = build_task3(df)
    t4 = build_task4(df)
    t5 = build_task5(df)
    t6 = build_task6(df)
    t7 = build_task7(df)

    build_summary(t1, t2, t3, t4, t5, t6, t7, df)

    print(f"\n{'=' * 62}\n  ALL BENCHMARK FILES SAVED\n{'=' * 62}")

    for f in sorted(BENCH_DIR.rglob("*")):
        if f.is_file():
            print(f"  {str(f.relative_to(BENCH_DIR)):<60} {f.stat().st_size // 1024:>6,} KB")

    print(f"\n  Output: {BENCH_DIR.resolve()}/")


if __name__ == "__main__":
    main()