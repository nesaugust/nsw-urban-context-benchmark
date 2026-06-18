"""
Topic 3 — Weather, Events & Urban Context Benchmark
Stage 1: Data Cleaning & Alignment
====================================

Updated version:
- fixes Open-Meteo CSV ParserError by safely detecting the real header row
- includes POI / Massive-STEPS mobility data
- outputs data/cleaned/poi_mobility_clean.csv
- joins POI mobility into data/cleaned/master_context_table.csv

Reads all downloaded raw files and standardises them into a common schema:

    datetime (hourly) | location | latitude | longitude | [domain columns]

Then produces:

    data/cleaned/master_context_table.csv

Usage:
    python 02_clean_and_align.py

Requirements:
    pip install pandas numpy openpyxl tqdm
"""

from importlib.resources import files
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW = Path("data")
OUT = Path("data/cleaned")
OUT.mkdir(parents=True, exist_ok=True)

# ── NSW regions reference ──────────────────────────────────────────────────────
REGIONS = [
    ("Sydney CBD", -33.8688, 151.2093),
    ("Parramatta", -33.8136, 151.0034),
    ("Liverpool", -33.9200, 150.9200),
    ("Penrith", -33.7508, 150.6942),
    ("Bondi", -33.8915, 151.2767),
    ("Manly", -33.7969, 151.2868),
    ("Newcastle", -32.9283, 151.7817),
    ("Wollongong", -34.4278, 150.8931),
    ("Coffs Harbour", -30.2963, 153.1135),
    ("Port Macquarie", -31.4333, 152.9000),
    ("Byron Bay", -28.6474, 153.6020),
    ("Nowra", -34.8793, 150.6008),
    ("Orange", -33.2833, 149.1000),
    ("Dubbo", -32.2569, 148.6011),
    ("Tamworth", -31.0833, 150.9167),
    ("Wagga Wagga", -35.1082, 147.3598),
    ("Albury", -36.0737, 146.9135),
    ("Bathurst", -33.4167, 149.5833),
    ("Broken Hill", -31.9505, 141.4534),
    ("Armidale", -30.5128, 151.6686),
    ("Katoomba", -33.7153, 150.3115),
    ("Perisher", -36.4073, 148.4100),
    ("Canberra", -35.2809, 149.1300),
    ("Cessnock", -32.8337, 151.3554),
]
REGIONS_DF = pd.DataFrame(REGIONS, columns=["location", "latitude", "longitude"])


# ── Helpers ────────────────────────────────────────────────────────────────────
def log(msg):
    print(f"\n{'=' * 62}\n  {msg}\n{'=' * 62}")


def save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    path = OUT / name
    df.to_csv(path, index=False)
    print(f"  ✓ Saved → {name} ({len(df):,} rows × {len(df.columns)} cols)")
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def nearest_region(lat: float, lon: float) -> str:
    dists = REGIONS_DF.apply(
        lambda r: haversine_km(lat, lon, r["latitude"], r["longitude"]),
        axis=1,
    )
    return REGIONS_DF.loc[dists.idxmin(), "location"]


def find_files(folder: Path, *patterns) -> list:
    found = []
    if not folder.exists():
        return found
    for p in patterns:
        found.extend(sorted(folder.glob(p)))
    return found


def normalise_columns(df):
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def read_openmeteo_csv(file):

    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_row = None

    for i, line in enumerate(lines):

        line_lower = line.lower()

        if (
            "temperature_2m" in line_lower
            or line_lower.startswith("time,")
            or line_lower.startswith("datetime,")
        ):
            header_row = i
            break

    if header_row is None:
        raise ValueError(
            f"Could not locate header row in {file.name}"
        )

    return pd.read_csv(
        file,
        skiprows=header_row
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. WEATHER — Open-Meteo
# ══════════════════════════════════════════════════════════════════════════════
def clean_weather():
    log("1/10 WEATHER — Open-Meteo")

    src = RAW / "01_weather" / "open_meteo" / "nsw_all_regions_weather.csv"

    if not src.exists():

        # ONLY load regional weather files
        files = sorted(
            (RAW / "01_weather" / "open_meteo").glob("*_weather_*.csv")
        )

        if not files:
            print("  ! No Open-Meteo files found — skipping")
            return pd.DataFrame()

        print(f"  Loading {len(files)} individual region files...")

        frames = []

        for f in files:

            try:
                temp = read_openmeteo_csv(f)
                temp = normalise_columns(temp)

                # infer location from filename
                loc = (
                    f.stem
                    .replace("_weather_2022_2026", "")
                    .replace("_weather", "")
                    .replace("_", " ")
                    .title()
                )

                temp["location"] = loc

                frames.append(temp)

            except Exception as e:
                print(f"  ! Failed to read {f.name}: {e}")

        if not frames:
            print("  ! No readable weather files")
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)

    else:
        print(f"  Loading merged file: {src.name}")

        df = read_openmeteo_csv(src)
        df = normalise_columns(df)

    df = normalise_columns(df)

    # find datetime column
    time_col = None

    for c in df.columns:
        if c in ["time", "datetime"]:
            time_col = c
            break

    if time_col is None:
        print(f"  ! No datetime column found")
        print(df.columns.tolist())
        return pd.DataFrame()

    df.rename(columns={time_col: "datetime"}, inplace=True)

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    df = df.dropna(subset=["datetime"])

    if "location" not in df.columns:
        print("  ! No location column")
        return pd.DataFrame()

    # rename weather columns
    df.rename(
        columns={
            "windspeed_10m": "wind_speed_kmh",
            "windgusts_10m": "wind_gust_kmh",
        },
        inplace=True,
    )

    # numeric conversions
    numeric_cols = [
        "temperature_2m",
        "relative_humidity_2m",
        "apparent_temperature",
        "rain",
        "precipitation",
        "wind_speed_kmh",
        "wind_gust_kmh",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # rain features
    if "rain" not in df.columns:
        df["rain"] = 0

    df["rain"] = df["rain"].fillna(0)

    df["rain_category"] = pd.cut(
        df["rain"],
        bins=[-0.01, 0.1, 1.0, 4.0, 16.0, 9999],
        labels=["none", "light", "moderate", "heavy", "extreme"]
    )

    # temperature features
    if "temperature_2m" in df.columns:

        df["is_hot_day"] = df["temperature_2m"] >= 35
        df["is_cold_day"] = df["temperature_2m"] <= 10

    else:

        df["temperature_2m"] = np.nan
        df["is_hot_day"] = False
        df["is_cold_day"] = False

    # datetime features
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.day_name()
    df["date"] = df["datetime"].dt.date
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    df["doy"] = df["datetime"].dt.dayofyear

    # temperature anomaly
    if "temperature_2m" in df.columns:

        df["temp_mean_doy"] = (
            df.groupby(["location", "doy"])["temperature_2m"]
            .transform("mean")
        )

        df["temp_std_doy"] = (
            df.groupby(["location", "doy"])["temperature_2m"]
            .transform("std")
        )

        df["temp_anomaly_z"] = (
            (df["temperature_2m"] - df["temp_mean_doy"])
            / df["temp_std_doy"].replace(0, np.nan)
        )

    # rain anomaly
    df["rain_mean_doy"] = (
        df.groupby(["location", "doy"])["rain"]
        .transform("mean")
    )

    df["rain_std_doy"] = (
        df.groupby(["location", "doy"])["rain"]
        .transform("std")
    )

    df["rain_anomaly_z"] = (
        (df["rain"] - df["rain_mean_doy"])
        / df["rain_std_doy"].replace(0, np.nan)
    )

    # cleanup
    drop_cols = [
        "doy",
        "temp_mean_doy",
        "temp_std_doy",
        "rain_mean_doy",
        "rain_std_doy",
    ]

    df.drop(
        columns=[c for c in drop_cols if c in df.columns],
        inplace=True
    )

    df = df.sort_values(
        ["location", "datetime"]
    ).reset_index(drop=True)

    return save(df, "weather_clean.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PUBLIC HOLIDAYS
# ══════════════════════════════════════════════════════════════════════════════
def clean_holidays():
    log("2/10 EVENTS — NSW Public Holidays")

    src = RAW / "02_events" / "public_holidays" / "nsw_public_holidays_2021_2025.csv"
    if not src.exists():
        src = RAW / "02_events" / "public_holidays" / "australian_public_holidays_combined_2021_2025.csv"

    if not src.exists():
        print("  ! No holiday file found — skipping")
        return pd.DataFrame()

    df = pd.read_csv(src)
    df = normalise_columns(df)

    date_col = next((c for c in df.columns if "date" in c), None)
    name_col = next((c for c in df.columns if "name" in c or "holiday" in c), None)
    juris_col = next((c for c in df.columns if "jurisd" in c), None)

    if date_col is None:
        print("  ! No date column found")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    if juris_col and "nsw" not in src.name.lower():
        df = df[df[juris_col].astype(str).str.upper() == "NSW"].copy()

    out = pd.DataFrame({"date": df["date"]})
    out["holiday_name"] = df[name_col].values if name_col else ""
    out["is_public_holiday"] = True

    rows = []
    for _, r in out.iterrows():
        for h in range(24):
            rows.append(
                {
                    "datetime": pd.Timestamp(str(r["date"]) + f" {h:02d}:00:00"),
                    "holiday_name": r.get("holiday_name", ""),
                    "is_public_holiday": True,
                }
            )

    hourly = pd.DataFrame(rows)
    return save(hourly, "holidays_clean.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 3. SCHOOL TERMS
# ══════════════════════════════════════════════════════════════════════════════
def clean_school_terms():
    log("3/10 EVENTS — NSW School Terms")

    src = RAW / "02_events" / "school_terms" / "nsw_school_terms_daily_2022_2026.csv"

    if not src.exists():
        print("  ! School terms file not found — skipping")
        return pd.DataFrame()

    df = pd.read_csv(src)
    df = normalise_columns(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    rows = []
    for _, r in df.iterrows():
        for h in range(24):
            rows.append(
                {
                    "datetime": pd.Timestamp(str(r["date"]) + f" {h:02d}:00:00"),
                    "school_term": int(r.get("term", 0)) if pd.notna(r.get("term", 0)) else 0,
                    "school_year": int(r.get("year", 0)) if pd.notna(r.get("year", 0)) else 0,
                    "is_school_term": True,
                }
            )

    hourly = pd.DataFrame(rows)
    return save(hourly, "school_terms_clean.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVENTS — Ticketmaster
# ══════════════════════════════════════════════════════════════════════════════
def clean_events():
    log("4/10 EVENTS — Ticketmaster concerts/sports/festivals")

    folder = RAW / "02_events" / "ticketmaster"

    files = list(folder.glob("*.csv"))

    if not files:
        print("  ! No Ticketmaster files found")
        return pd.DataFrame()

    frames = []

    for f in files:
        try:
            temp = pd.read_csv(f)
            temp["source_file"] = f.name
            frames.append(temp)
        except Exception as e:
            print(f"  ! {f.name}: {e}")

    df = pd.concat(frames, ignore_index=True)
    df = normalise_columns(df)

    if "date" not in df.columns:
        print("  ! No date column in Ticketmaster file")
        return pd.DataFrame()

    if "time" not in df.columns:
        df["time"] = "00:00:00"

    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].fillna("00:00:00").astype(str),
        errors="coerce",
    )

    df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    df = df.dropna(subset=["datetime", "latitude", "longitude"])

    print("  Snapping events to nearest benchmark region...")
    df["nearest_region"] = df.apply(lambda r: nearest_region(r["latitude"], r["longitude"]), axis=1)
    df["dist_to_region_km"] = df.apply(
        lambda r: haversine_km(
            r["latitude"],
            r["longitude"],
            REGIONS_DF.loc[REGIONS_DF["location"] == r["nearest_region"], "latitude"].iloc[0],
            REGIONS_DF.loc[REGIONS_DF["location"] == r["nearest_region"], "longitude"].iloc[0],
        ),
        axis=1,
    )

    keep = [
        "datetime",
        "event_id",
        "name",
        "category",
        "genre",
        "venue",
        "city",
        "latitude",
        "longitude",
        "nearest_region",
        "dist_to_region_km",
        "url",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    print("  Building hourly event presence flags per region...")
    event_rows = []
    for _, ev in df.iterrows():
        for offset in range(-1, 4):
            event_rows.append(
                {
                    "datetime": ev["datetime"] + pd.Timedelta(hours=offset),
                    "location": ev["nearest_region"],
                    "event_name": ev.get("name", ""),
                    "event_category": ev.get("category", ""),
                    "event_venue": ev.get("venue", ""),
                    "event_active": True,
                    "dist_km": ev["dist_to_region_km"],
                }
            )

    if not event_rows:
        return pd.DataFrame()

    events_hourly = pd.DataFrame(event_rows)

    agg = (
        events_hourly.groupby(["datetime", "location"])
        .agg(
            event_count=("event_active", "sum"),
            event_categories=("event_category", lambda x: "|".join(x.dropna().astype(str).unique())),
            min_dist_km=("dist_km", "min"),
        )
        .reset_index()
    )
    agg["has_nearby_event"] = True

    return save(agg, "events_clean.csv")


def clean_major_events():
    log("4b/10 EVENTS — Curated Major NSW Events")

    src = RAW / "02_events" / "major_events" / "nsw_major_events_2018_2026.csv"

    if not src.exists():
        print("  ! No major events file found — skipping")
        return pd.DataFrame()

    df = pd.read_csv(src)
    df = normalise_columns(df)

    required = {"event_name", "start_date", "end_date", "location", "event_type"}
    missing = required - set(df.columns)

    if missing:
        print(f"  ! Missing columns in major events file: {missing}")
        return pd.DataFrame()

    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df = df.dropna(subset=["start_date", "end_date"])

    rows = []

    for _, r in df.iterrows():
        location = str(r["location"])

        if location == "NSW":
            affected_locations = REGIONS_DF["location"].tolist()
        elif location == "Sydney":
            affected_locations = ["Sydney CBD"]
        else:
            affected_locations = [location]

        for loc in affected_locations:
            for d in pd.date_range(r["start_date"].date(), r["end_date"].date(), freq="D"):
                for h in range(24):
                    rows.append(
                        {
                            "datetime": pd.Timestamp(d) + pd.Timedelta(hours=h),
                            "location": loc,
                            "major_event_count": 1,
                            "major_event_name": r["event_name"],
                            "major_event_type": r["event_type"],
                            "major_event_source": r.get("source", "manual_seed"),
                            "has_major_event": True,
                        }
                    )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    agg = (
        out.groupby(["datetime", "location"])
        .agg(
            major_event_count=("major_event_count", "sum"),
            major_event_names=("major_event_name", lambda x: "|".join(sorted(set(x.dropna().astype(str))))),
            major_event_types=("major_event_type", lambda x: "|".join(sorted(set(x.dropna().astype(str))))),
            has_major_event=("has_major_event", "max"),
        )
        .reset_index()
    )

    return save(agg, "major_events_clean.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 5. ROAD CRASHES
# ══════════════════════════════════════════════════════════════════════════════
def clean_road_crashes():
    log("5/10 INCIDENTS — NSW Road Crashes")

    folder = RAW / "03_incidents" / "road_crashes"

    files = list(folder.glob("*"))
    for f in files:
        print(f"{f.name} | {f.suffix}")

    if not files:
        print("  ! No crash files found — skipping")
        return pd.DataFrame()

    crash_file = None

    for f in files:
        if "crash" in f.name.lower():
            crash_file = f
            break

    if crash_file is None:
        print("  ! Could not find crash dataset")
        return pd.DataFrame()

    print(f"  Loading {crash_file.name}")

    try:

        # CSV
        if crash_file.suffix.lower() == ".csv":
            df = pd.read_csv(
                crash_file,
                low_memory=False
            )

        # XLSX
        elif crash_file.suffix.lower() == ".xlsx":
            df = pd.read_excel(
                crash_file,
                engine="openpyxl"
            )

        # XLS
        elif crash_file.suffix.lower() == ".xls":
            df = pd.read_excel(
                crash_file,
                engine="xlrd"
            )

        else:

            # Try CSV first
            try:
                df = pd.read_csv(
                    crash_file,
                    low_memory=False
                )

            except Exception:

                try:
                    df = pd.read_excel(
                        crash_file,
                        engine="openpyxl"
                    )

                except Exception:
                    df = pd.read_excel(
                        crash_file,
                        engine="xlrd"
                    )

    except Exception as e:
        print(f"  ! Failed to load crash file: {e}")
        return pd.DataFrame()

    df = normalise_columns(df)

    print(f"  Loaded {len(df):,} crash records")

    print("\nColumns found:")
    print(df.columns.tolist())

    required = [
        "month_of_crash",
        "day_of_week_of_crash",
        "two-hour_intervals",
        "latitude",
        "longitude",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"  ! Missing columns: {missing}")
        return pd.DataFrame()

    month_map = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    df["month"] = (
        df["month_of_crash"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(month_map)
    )

    df["month"] = df["month"].fillna(
        pd.to_numeric(
            df["month_of_crash"],
            errors="coerce"
        )
    )

    df["day_of_week"] = (
        df["day_of_week_of_crash"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    df["hour"] = (
        df["two-hour_intervals"]
        .astype(str)
        .str.extract(r"(\d{1,2})")[0]
    )

    df["hour"] = pd.to_numeric(
        df["hour"],
        errors="coerce"
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "month",
            "day_of_week",
            "hour",
            "latitude",
            "longitude",
        ]
    )

    if df.empty:
        print("  ! No usable crash records")
        return pd.DataFrame()

    df["month"] = df["month"].astype(int)
    df["hour"] = df["hour"].astype(int)

    print("  Assigning nearest benchmark region...")

    df["location"] = [
        nearest_region(lat, lon)
        for lat, lon in zip(
            df["latitude"],
            df["longitude"]
        )
    ]

    if "no._killed" in df.columns:
        df["no._killed"] = pd.to_numeric(
            df["no._killed"],
            errors="coerce"
        ).fillna(0)
    else:
        df["no._killed"] = 0

    if "no._seriously_injured" in df.columns:
        df["no._seriously_injured"] = pd.to_numeric(
            df["no._seriously_injured"],
            errors="coerce"
        ).fillna(0)
    else:
        df["no._seriously_injured"] = 0

    count_col = (
        "crash_id"
        if "crash_id" in df.columns
        else "location"
    )

    agg = (
        df.groupby(
            [
                "location",
                "month",
                "day_of_week",
                "hour",
            ]
        )
        .agg(
            crash_risk_count=(count_col, "count"),
            fatal_crashes=(
                "no._killed",
                lambda x: (x > 0).sum(),
            ),
            serious_crashes=(
                "no._seriously_injured",
                lambda x: (x > 0).sum(),
            ),
        )
        .reset_index()
    )

    agg["crash_risk_level"] = pd.cut(
        agg["crash_risk_count"],
        bins=[-1, 0, 5, 20, 999999],
        labels=[
            "none",
            "low",
            "medium",
            "high",
        ],
    )

    print(
        f"  Generated {len(agg):,} crash-risk profiles"
    )

    return save(
        agg,
        "road_crashes_clean.csv"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. LIVE TRAFFIC HAZARDS
# ══════════════════════════════════════════════════════════════════════════════
def clean_traffic_hazards():
    log("6/10 INCIDENTS — Live Traffic Hazards")

    folder = RAW / "03_incidents" / "live_traffic_hazards"
    files = find_files(folder, "*.csv", "*.geojson")

    if not files:
        print("  ! No hazard files found — skipping")
        return pd.DataFrame()

    all_frames = []

    for f in files:
        try:
            if f.suffix.lower() == ".geojson":
                data = json.loads(f.read_text(encoding="utf-8", errors="ignore"))
                rows = []
                for feat in data.get("features", []):
                    p = feat.get("properties", {})
                    c = feat.get("geometry", {}).get("coordinates", [None, None])
                    rows.append(
                        {
                            "incident_id": p.get("id"),
                            "type": p.get("mainCategory", ""),
                            "sub_type": p.get("subCategoryA", ""),
                            "description": p.get("headline", ""),
                            "start_time": p.get("created"),
                            "last_updated": p.get("lastUpdated"),
                            "latitude": c[1] if len(c) > 1 else None,
                            "longitude": c[0] if len(c) > 0 else None,
                        }
                    )
                df = pd.DataFrame(rows)
            else:
                df = pd.read_csv(f)

            all_frames.append(df)
        except Exception as e:
            print(f"  ! Failed to load {f.name}: {e}")

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df = normalise_columns(df)

    time_col = next((c for c in df.columns if "time" in c or "start" in c or "created" in c), None)
    if time_col is None:
        print("  ! No time column found in hazard data")
        return pd.DataFrame()

    df["datetime"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"])
    df["datetime"] = df["datetime"].dt.tz_convert("Australia/Sydney").dt.tz_localize(None).dt.floor("h")

    lat_col = next((c for c in df.columns if "lat" in c), None)
    lon_col = next((c for c in df.columns if "lon" in c or "lng" in c), None)

    if lat_col and lon_col:
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        valid = df.dropna(subset=[lat_col, lon_col])
        df.loc[valid.index, "location"] = valid.apply(lambda r: nearest_region(r[lat_col], r[lon_col]), axis=1)

    if "location" not in df.columns:
        print("  ! No location could be assigned for hazards")
        return pd.DataFrame()

    type_col = "type" if "type" in df.columns else "datetime"
    count_col = "incident_id" if "incident_id" in df.columns else "datetime"

    agg = (
        df.groupby(["datetime", "location"])
        .agg(
            incident_count=(count_col, "count"),
            incident_types=(type_col, lambda x: "|".join(x.dropna().astype(str).unique())),
        )
        .reset_index()
    )

    return save(agg, "traffic_hazards_clean.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 7. PUBLIC TRANSPORT ALERTS
# ══════════════════════════════════════════════════════════════════════════════
def clean_transport_alerts():
    log("7/10 PUBLIC TRANSPORT — Service Alerts & Disruptions")

    folder = RAW / "04_public_transport" / "service_alerts"
    files = find_files(folder, "*.csv")

    if not files:
        print("  ! No transport alert CSV files found — skipping")
        return pd.DataFrame()

    all_frames = []

    for f in files:
        try:
            df = pd.read_csv(f)
            all_frames.append(df)
            print(f"  Loaded {f.name} ({len(df):,} rows)")
        except Exception as e:
            print(f"  ! {f.name}: {e}")

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df = normalise_columns(df)

    time_col = next((c for c in df.columns if "start" in c or "time" in c or "date" in c or "created" in c), None)

    if time_col is None:
        print("  ! Cannot find timestamp column")
        return pd.DataFrame()

    # Try Unix seconds first, then ISO/date strings
    numeric_time = pd.to_numeric(df[time_col], errors="coerce")
    if numeric_time.notna().sum() > len(df) * 0.5:
        df["datetime"] = pd.to_datetime(numeric_time, errors="coerce", unit="s")
    else:
        df["datetime"] = pd.to_datetime(df[time_col], errors="coerce")

    df = df.dropna(subset=["datetime"])
    df["datetime"] = df["datetime"].dt.floor("h")

    effect_col = next((c for c in df.columns if "effect" in c), None)
    route_col = next((c for c in df.columns if "route" in c), None)

    agg = (
        df.groupby("datetime")
        .agg(
            alert_count=("datetime", "count"),
            alert_effects=(effect_col, lambda x: "|".join(x.dropna().astype(str).unique())) if effect_col else ("datetime", lambda x: ""),
            affected_routes=(route_col, lambda x: "|".join(x.dropna().astype(str).unique())) if route_col else ("datetime", lambda x: ""),
        )
        .reset_index()
    )

    # Broadcast system-wide alerts to all benchmark regions
    region_rows = []
    for _, r in agg.iterrows():
        for loc, *_ in REGIONS:
            row = r.to_dict()
            row["location"] = loc
            region_rows.append(row)

    result = pd.DataFrame(region_rows)
    return save(result, "transport_alerts_clean.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 8. TRAFFIC COUNTS
# ══════════════════════════════════════════════════════════════════════════════
def clean_traffic():
    log("8/10 TRAFFIC — NSW Hourly Traffic Counts")

    folder = RAW / "05_traffic" / "nsw_traffic_hf"
    files = [folder / "Traffic_X.csv"]
    files = [f for f in files if f.exists()]

    if not files:
        print("  ! No traffic count files found — skipping")
        return pd.DataFrame()

    all_frames = []

    for f in files:
        try:
            df = pd.read_csv(f)
            df["split"] = f.stem.replace("nsw_traffic_", "")
            all_frames.append(df)
            print(f"  Loaded {f.name} ({len(df):,} rows)")
        except Exception as e:
            print(f"  ! {f.name}: {e}")

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df = normalise_columns(df)

    id_cols = [c for c in df.columns if not c.startswith("value_") and c != "split"]
    val_cols = [c for c in df.columns if c.startswith("value_")]

    if val_cols:
        print(f"  Melting {len(val_cols)} hourly value columns to long format...")
        df_long = df.melt(
            id_vars=id_cols,
            value_vars=val_cols,
            var_name="hour_offset",
            value_name="traffic_volume",
        )
        df_long["hour"] = df_long["hour_offset"].str.extract(r"(\d+)").astype(int)
        df_long.drop(columns=["hour_offset"], inplace=True)
        df_long["traffic_volume"] = pd.to_numeric(df_long["traffic_volume"], errors="coerce")
        df = df_long

    sensor_col = next((c for c in df.columns if c in ("sensor_id", "series_name", "station", "location", "site")), None)

    if sensor_col and sensor_col != "location":
        df.rename(columns={sensor_col: "sensor_id"}, inplace=True)
        if "location" not in df.columns:
            df["location"] = "Sydney CBD"

    if "location" not in df.columns:
        df["location"] = "Sydney CBD"

    group_cols = [c for c in ["location", "hour"] if c in df.columns]

    if group_cols and "traffic_volume" in df.columns:
        agg = (
            df.groupby(group_cols)
            .agg(
                traffic_volume_mean=("traffic_volume", "mean"),
                traffic_volume_max=("traffic_volume", "max"),
                traffic_sensor_count=("traffic_volume", "count"),
            )
            .reset_index()
        )
        return save(agg, "traffic_counts_clean.csv")

    return save(df, "traffic_counts_clean.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 9. PEDESTRIAN COUNTS
# ══════════════════════════════════════════════════════════════════════════════
def clean_pedestrian():
    log("9/10 PEDESTRIAN — City of Sydney CBD")

    folder = RAW / "06_pedestrian" / "sydney_pedestrian"
    files = find_files(folder, "*.csv")

    if not files:
        print("  ! No pedestrian files found — skipping")
        return pd.DataFrame()

    all_frames = []

    for f in files:
        try:
            df = pd.read_csv(f)
            all_frames.append(df)
            print(f"  Loaded {f.name} ({len(df):,} rows)")
        except Exception as e:
            print(f"  ! {f.name}: {e}")

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df = normalise_columns(df)

    time_col = next((c for c in df.columns if "date" in c or "time" in c or "timestamp" in c), None)
    count_col = next((c for c in df.columns if "count" in c or "volume" in c or "total" in c or "pedestrian" in c), None)
    sensor_col = next((c for c in df.columns if "sensor" in c or "location" in c or "site" in c or "name" in c), None)

    if time_col is None:
        print(f"  ! Cannot find time column. Available: {list(df.columns)}")
        return pd.DataFrame()

    df["datetime"] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True, utc=True)
    df = df.dropna(subset=["datetime"])
    df["datetime"] = df["datetime"].dt.tz_convert("Australia/Sydney").dt.tz_localize(None).dt.floor("h")
    df["location"] = "Sydney CBD"

    if sensor_col and sensor_col != "location":
        df.rename(columns={sensor_col: "sensor_name"}, inplace=True)

    if count_col:
        df[count_col] = pd.to_numeric(df[count_col], errors="coerce")
        agg = (
            df.groupby(["datetime", "location"])
            .agg(
                pedestrian_count_sum=(count_col, "sum"),
                pedestrian_count_mean=(count_col, "mean"),
                pedestrian_sensor_count=(count_col, "count"),
            )
            .reset_index()
        )
    else:
        agg = df.groupby(["datetime", "location"]).size().reset_index(name="pedestrian_count_sum")
        agg["pedestrian_count_mean"] = agg["pedestrian_count_sum"]

    return save(agg, "pedestrian_clean.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 10. POI / MASSIVE-STEPS MOBILITY
# ══════════════════════════════════════════════════════════════════════════════
def clean_poi_mobility():
    log("10/10 POI / MOBILITY — Massive-STEPS Sydney")

    folder = RAW / "07_poi_mobility" / "massive_steps"
    files = find_files(folder, "*.csv", "*.xlsx", "*.xls")

    if not files:
        print("  ! No POI / Massive-STEPS files found — skipping")
        return pd.DataFrame()

    all_frames = []

    for f in files:
        try:
            if f.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(f)
            else:
                df = pd.read_csv(f, low_memory=False)

            df["source_file"] = f.name
            all_frames.append(df)
            print(f"  Loaded {f.name} ({len(df):,} rows)")
        except Exception as e:
            print(f"  ! Failed to load {f.name}: {e}")

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df = normalise_columns(df)

    # Detect datetime/time columns
    datetime_col = next(
        (
            c
            for c in df.columns
            if c in ["datetime", "timestamp", "time", "date_time", "start_time", "created_at"]
            or "datetime" in c
            or "timestamp" in c
        ),
        None,
    )

    date_col = next((c for c in df.columns if c in ["date", "day"]), None)
    hour_col = next((c for c in df.columns if c in ["hour", "hr", "time_hour"]), None)

    if datetime_col:
        df["datetime"] = pd.to_datetime(df[datetime_col], errors="coerce")
    elif date_col and hour_col:
        df["datetime"] = pd.to_datetime(
            df[date_col].astype(str)
            + " "
            + pd.to_numeric(df[hour_col], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(2)
            + ":00:00",
            errors="coerce",
        )
    elif date_col:
        df["datetime"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        print(f"  ! Cannot find datetime/date columns in POI data. Available: {list(df.columns)}")
        return pd.DataFrame()

    df = df.dropna(subset=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").dt.floor("h")

    # Detect lat/lon if available
    lat_col = next((c for c in df.columns if c in ["lat", "latitude", "poi_lat", "venue_lat"]), None)
    lon_col = next((c for c in df.columns if c in ["lon", "lng", "longitude", "poi_lon", "poi_lng", "venue_lon", "venue_lng"]), None)

    # Detect location/region column
    loc_col = next((c for c in df.columns if c in ["location", "region", "suburb", "area", "city"]), None)

    if lat_col and lon_col:
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        valid = df.dropna(subset=[lat_col, lon_col])

        if not valid.empty:
            print("  Snapping POI records to nearest benchmark regions...")
            df.loc[valid.index, "location"] = valid.apply(lambda r: nearest_region(r[lat_col], r[lon_col]), axis=1)

    elif loc_col:
        df["location"] = df[loc_col].astype(str)

        # Map common Sydney labels to Sydney CBD if region names do not match
        df["location"] = df["location"].replace(
            {
                "Sydney": "Sydney CBD",
                "Sydney City": "Sydney CBD",
                "CBD": "Sydney CBD",
                "City of Sydney": "Sydney CBD",
            }
        )

        valid_regions = set(REGIONS_DF["location"])
        unmatched = ~df["location"].isin(valid_regions)

        # Massive-STEPS Sydney is usually CBD/metropolitan. Use Sydney CBD fallback.
        df.loc[unmatched, "location"] = "Sydney CBD"

    else:
        # Fallback because Massive-STEPS folder in your screenshot is Sydney data.
        df["location"] = "Sydney CBD"

    # Detect POI category
    poi_category_col = next(
        (
            c
            for c in df.columns
            if "category" in c
            or "poi_type" in c
            or "venue_type" in c
            or "place_type" in c
        ),
        None,
    )

    if poi_category_col:
        df["poi_category"] = df[poi_category_col].astype(str)
    else:
        df["poi_category"] = "unknown"

    # Detect activity/count columns
    candidate_activity_cols = [
        "poi_activity",
        "activity",
        "count",
        "visits",
        "visit_count",
        "checkins",
        "checkin_count",
        "mobility",
        "flow",
        "demand",
        "volume",
        "num_trips",
        "trip_count",
        "value",
    ]

    activity_col = next((c for c in candidate_activity_cols if c in df.columns), None)

    if activity_col is None:
        value_cols = [c for c in df.columns if c.startswith("value_")]

        if value_cols:
            print(f"  Melting {len(value_cols)} POI value columns to long format...")
            id_cols = [c for c in df.columns if c not in value_cols]
            df = df.melt(
                id_vars=id_cols,
                value_vars=value_cols,
                var_name="value_hour",
                value_name="poi_activity",
            )

            df["value_hour_num"] = pd.to_numeric(
                df["value_hour"].str.extract(r"(\d+)")[0],
                errors="coerce",
            ).fillna(0).astype(int)

            # If original datetime was daily, add hour offset
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.floor("D") + pd.to_timedelta(df["value_hour_num"], unit="h")
            activity_col = "poi_activity"
        else:
            # If there is no explicit activity column, count rows per hour as visits/activity.
            df["poi_activity"] = 1
            activity_col = "poi_activity"

    df[activity_col] = pd.to_numeric(df[activity_col], errors="coerce").fillna(0)

    # Aggregate to hourly region level
    agg = (
        df.groupby(["datetime", "location"])
        .agg(
            poi_activity=(activity_col, "sum"),
            poi_record_count=(activity_col, "count"),
            poi_category=("poi_category", lambda x: "|".join(x.dropna().astype(str).unique()[:10])),
        )
        .reset_index()
    )

    # Baseline and mobility level
    agg["hour"] = agg["datetime"].dt.hour
    agg = agg.sort_values(["location", "datetime"]).reset_index(drop=True)

    agg["poi_baseline"] = agg.groupby(["location", "hour"])["poi_activity"].transform(
        lambda x: x.rolling(24 * 7 * 4, min_periods=6, center=True).mean()
    )

    agg["poi_pct_change"] = (
        (agg["poi_activity"] - agg["poi_baseline"])
        / agg["poi_baseline"].replace(0, np.nan)
        * 100
    )

    agg["mobility_level"] = pd.cut(
        agg["poi_pct_change"].fillna(0),
        bins=[-999999, -20, 20, 999999],
        labels=["low", "normal", "high"],
    ).astype(str)

    return save(agg, "poi_mobility_clean.csv")


def clean_osm_poi_static():
    log("10b/10 POI — OSM Static Regional Features")

    raw_path = RAW / "07_poi_mobility" / "osm_poi" / "nsw_osm_poi_raw.csv"
    feature_path = RAW / "07_poi_mobility" / "osm_poi" / "nsw_osm_poi_region_features.csv"

    frames = []

    if raw_path.exists():
        raw = pd.read_csv(raw_path, low_memory=False)
        raw = normalise_columns(raw)

        if "location" not in raw.columns:
            print("  ! No location column in raw OSM POI file")
        else:
            category_cols = [
                c for c in [
                    "amenity", "shop", "tourism", "leisure",
                    "office", "public_transport", "railway"
                ]
                if c in raw.columns
            ]

            raw["poi_main_category"] = "other"

            for c in category_cols:
                raw.loc[raw[c].notna(), "poi_main_category"] = c

            raw_agg = (
                raw.groupby("location")
                .agg(
                    total_poi_count=("location", "count"),
                    amenity=("poi_main_category", lambda x: (x == "amenity").sum()),
                    shop=("poi_main_category", lambda x: (x == "shop").sum()),
                    tourism=("poi_main_category", lambda x: (x == "tourism").sum()),
                    leisure=("poi_main_category", lambda x: (x == "leisure").sum()),
                    office=("poi_main_category", lambda x: (x == "office").sum()),
                    public_transport=("poi_main_category", lambda x: (x == "public_transport").sum()),
                    railway=("poi_main_category", lambda x: (x == "railway").sum()),
                    other=("poi_main_category", lambda x: (x == "other").sum()),
                )
                .reset_index()
            )

            frames.append(raw_agg)

    if feature_path.exists():
        feat = pd.read_csv(feature_path, low_memory=False)
        feat = normalise_columns(feat)

        if "location" in feat.columns:
            for col in feat.columns:
                if col != "location":
                    feat[col] = pd.to_numeric(feat[col], errors="coerce").fillna(0)

            frames.append(feat)
        else:
            print("  ! No location column in OSM region features file")

    if not frames:
        print("  ! No OSM POI files found — skipping")
        return pd.DataFrame()

    out = frames[0]

    for f in frames[1:]:
        out = out.merge(f, on="location", how="outer", suffixes=("", "_region"))

    for col in list(out.columns):
        if col.endswith("_region"):
            base = col.replace("_region", "")
            if base in out.columns:
                out[base] = out[base].fillna(0) + out[col].fillna(0)
                out.drop(columns=[col], inplace=True)
            else:
                out.rename(columns={col: base}, inplace=True)

    out["location"] = (
        out["location"]
        .astype(str)
        .str.strip()
        .str.replace("_", " ", regex=False)
        .str.title()
    )

    poi_count_cols = [
        "amenity",
        "shop",
        "tourism",
        "leisure",
        "office",
        "public_transport",
        "railway",
        "other",
    ]

    for col in poi_count_cols:
        if col not in out.columns:
            out[col] = 0

    for col in poi_count_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    if "total_poi_count" not in out.columns:
        out["total_poi_count"] = out[poi_count_cols].sum(axis=1)
    else:
        out["total_poi_count"] = pd.to_numeric(out["total_poi_count"], errors="coerce").fillna(0)

    max_poi = out["total_poi_count"].max()

    if max_poi > 0:
        out["poi_activity"] = out["total_poi_count"] / max_poi
    else:
        out["poi_activity"] = 0

    out["mobility_level"] = pd.cut(
        out["poi_activity"],
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["low", "normal", "high"],
    ).astype(str)

    def dominant_poi_category(row):
        counts = row[poi_count_cols]
        if counts.sum() == 0:
            return ""
        return counts.idxmax()

    out["poi_category"] = out.apply(dominant_poi_category, axis=1)

    keep_cols = [
        "location",
        "total_poi_count",
        "amenity",
        "shop",
        "tourism",
        "leisure",
        "office",
        "public_transport",
        "railway",
        "other",
        "poi_activity",
        "mobility_level",
        "poi_category",
    ]

    out = out[[c for c in keep_cols if c in out.columns]].copy()

    return save(out, "osm_poi_static_clean.csv")

def clean_massive_steps_llm_tasks():
    log("10c/10 POI — Massive-STEPS LLM Trajectory Tasks")

    folder = RAW / "07_poi_mobility" / "massive_steps"
    files = find_files(folder, "*.csv")

    if not files:
        print("  ! No Massive-STEPS LLM files found — skipping")
        return pd.DataFrame()

    import re

    frames = []

    for f in files:
        try:
            df = pd.read_csv(f)
            df = normalise_columns(df)
            df["source_file"] = f.name
            frames.append(df)
            print(f"  Loaded {f.name} ({len(df):,} rows)")
        except Exception as e:
            print(f"  ! Failed to load {f.name}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Expected columns: user_id, trail_id, inputs, targets
    required = {"user_id", "trail_id", "inputs", "targets"}
    missing = required - set(df.columns)

    if missing:
        print(f"  ! Missing columns: {missing}")
        return pd.DataFrame()

    input_pattern = re.compile(
        r"At\s+([\d\-]+\s+[\d:]+),\s+user\s+(\d+)\s+visited\s+POI id\s+(\d+)\s+which is a\s+([^,]+),\s+at\s+([^,]+),\s+AU",
        re.IGNORECASE,
    )

    query_pattern = re.compile(
        r"Given the data,\s+At\s+([\d\-]+\s+[\d:]+),\s+Which POI id will user\s+(\d+)\s+visit",
        re.IGNORECASE,
    )

    target_pattern = re.compile(
        r"At\s+([\d\-]+\s+[\d:]+),\s+user\s+(\d+)\s+will visit POI id\s+(\d+)",
        re.IGNORECASE,
    )

    rows = []

    for _, r in df.iterrows():
        input_text = str(r["inputs"])
        target_text = str(r["targets"])

        input_match = input_pattern.search(input_text)
        query_match = query_pattern.search(input_text)
        target_match = target_pattern.search(target_text)

        previous_time = None
        previous_user_id = None
        previous_poi_id = None
        previous_poi_category = None
        previous_suburb = None

        if input_match:
            previous_time = input_match.group(1)
            previous_user_id = input_match.group(2)
            previous_poi_id = input_match.group(3)
            previous_poi_category = input_match.group(4)
            previous_suburb = input_match.group(5)

        query_time = None
        query_user_id = None

        if query_match:
            query_time = query_match.group(1)
            query_user_id = query_match.group(2)

        target_time = None
        target_user_id = None
        target_poi_id = None

        if target_match:
            target_time = target_match.group(1)
            target_user_id = target_match.group(2)
            target_poi_id = target_match.group(3)

        rows.append(
            {
                "user_id": r.get("user_id"),
                "trail_id": r.get("trail_id"),
                "source_file": r.get("source_file"),
                "input_text": input_text,
                "target_text": target_text,
                "previous_time": previous_time,
                "query_time": query_time,
                "target_time": target_time,
                "previous_poi_id": previous_poi_id,
                "previous_poi_category": previous_poi_category,
                "previous_suburb": previous_suburb,
                "target_poi_id": target_poi_id,
                "task_type": "poi_next_location_prediction",
            }
        )

    out = pd.DataFrame(rows)

    out["previous_time"] = pd.to_datetime(out["previous_time"], errors="coerce")
    out["query_time"] = pd.to_datetime(out["query_time"], errors="coerce")
    out["target_time"] = pd.to_datetime(out["target_time"], errors="coerce")

    out["previous_hour"] = out["previous_time"].dt.hour
    out["query_hour"] = out["query_time"].dt.hour
    out["previous_day_of_week"] = out["previous_time"].dt.day_name()
    out["query_day_of_week"] = out["query_time"].dt.day_name()

    out["previous_poi_id"] = pd.to_numeric(out["previous_poi_id"], errors="coerce")
    out["target_poi_id"] = pd.to_numeric(out["target_poi_id"], errors="coerce")

    return save(out, "poi_llm_tasks_clean.csv")

# Join all tables to master grid
def build_base_region_time_grid(start="2022-01-01", end="2026-12-31", freq="h"):
    dates = pd.date_range(start=start, end=end + " 23:00:00", freq=freq)

    grid = (
        REGIONS_DF.assign(key=1)
        .merge(pd.DataFrame({"datetime": dates, "key": 1}), on="key")
        .drop(columns="key")
    )

    grid["datetime"] = pd.to_datetime(grid["datetime"]).dt.floor("h")
    return grid

# ══════════════════════════════════════════════════════════════════════════════
# MASTER JOIN TABLE
# ══════════════════════════════════════════════════════════════════════════════
def build_master_table():
    log("MASTER JOIN — Building context table")

    def load(name):
        path = OUT / name
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            df = normalise_columns(df)

            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                df["datetime"] = df["datetime"].dt.floor("h")

            print(f"  Loaded {name} ({len(df):,} rows)")
            return df

        print(f"  ! {name} not found — skipping layer")
        return None

    weather = load("weather_clean.csv")
    holidays = load("holidays_clean.csv")
    school = load("school_terms_clean.csv")
    events = load("events_clean.csv")
    major_events = load("major_events_clean.csv")
    crashes = load("road_crashes_clean.csv")
    hazards = load("traffic_hazards_clean.csv")
    alerts = load("transport_alerts_clean.csv")
    traffic = load("traffic_counts_clean.csv")
    pedestrian = load("pedestrian_clean.csv")
    poi = load("poi_mobility_clean.csv")
    osm_poi = load("osm_poi_static_clean.csv")
    master = build_base_region_time_grid(
        start="2022-01-01",
        end="2026-12-31",
        freq="h"
    )

    print(f"  Created full region-hour grid: {len(master):,} rows")

    if weather is not None and not weather.empty:
        weather_cols = [c for c in weather.columns if c not in ["latitude", "longitude"]]
        weather_slim = weather[weather_cols].copy()

        master = master.merge(
            weather_slim,
            on=["datetime", "location"],
            how="left"
        )
    else:
        print("  ! No weather data found — weather columns will be blank")

    # Holidays
    if holidays is not None and not holidays.empty:
        holidays_slim = holidays[["datetime", "holiday_name", "is_public_holiday"]].drop_duplicates("datetime")
        master = master.merge(holidays_slim, on="datetime", how="left")
        master["is_public_holiday"] = master["is_public_holiday"].fillna(False)
        master["holiday_name"] = master["holiday_name"].fillna("")
    else:
        master["is_public_holiday"] = False
        master["holiday_name"] = ""

    # School terms
    if school is not None and not school.empty:
        cols = [c for c in ["datetime", "is_school_term", "school_term", "school_year"] if c in school.columns]
        school_slim = school[cols].drop_duplicates("datetime")
        master = master.merge(school_slim, on="datetime", how="left")
        master["is_school_term"] = master["is_school_term"].fillna(False)
    else:
        master["is_school_term"] = False

    # Events
    if events is not None and not events.empty and "location" in events.columns:
        ev_cols = [c for c in ["datetime", "location", "event_count", "event_categories", "has_nearby_event", "min_dist_km"] if c in events.columns]
        ev_slim = events[ev_cols].copy()
        master = master.merge(ev_slim, on=["datetime", "location"], how="left")
        master["has_nearby_event"] = master["has_nearby_event"].fillna(False)
        master["event_count"] = master["event_count"].fillna(0).astype(int)
        if "event_categories" in master.columns:
            master["event_categories"] = master["event_categories"].fillna("")
    else:
        master["has_nearby_event"] = False
        master["event_count"] = 0
        master["event_categories"] = ""

    # Curated major events
    if major_events is not None and not major_events.empty and "location" in major_events.columns:
        me_cols = [
            c for c in [
                "datetime",
                "location",
                "major_event_count",
                "major_event_names",
                "major_event_types",
                "has_major_event",
            ]
            if c in major_events.columns
        ]

        me_slim = major_events[me_cols].copy()

        master = master.merge(
            me_slim,
            on=["datetime", "location"],
            how="left"
        )

        master["major_event_count"] = master["major_event_count"].fillna(0).astype(int)
        master["has_major_event"] = master["has_major_event"].fillna(False)

        if "major_event_names" in master.columns:
            master["major_event_names"] = master["major_event_names"].fillna("")

        if "major_event_types" in master.columns:
            master["major_event_types"] = master["major_event_types"].fillna("")
    else:
        master["major_event_count"] = 0
        master["has_major_event"] = False
        master["major_event_names"] = ""
        master["major_event_types"] = ""

    # Calendar keys for crash profile and traffic slot join
    master["month"] = pd.to_datetime(master["datetime"]).dt.month
    master["day_of_week"] = pd.to_datetime(master["datetime"]).dt.day_name()
    master["hour"] = pd.to_datetime(master["datetime"]).dt.hour

    # Road crashes historical profile
    if (
        crashes is not None
        and not crashes.empty
        and {"location", "month", "day_of_week", "hour", "crash_risk_count"}.issubset(crashes.columns)
    ):
        cr_slim = crashes[
            [
                "location",
                "month",
                "day_of_week",
                "hour",
                "crash_risk_count",
                "fatal_crashes",
                "serious_crashes",
                "crash_risk_level",
            ]
        ].copy()

        master = master.merge(
            cr_slim,
            on=["location", "month", "day_of_week", "hour"],
            how="left",
        )

        master["crash_risk_count"] = master["crash_risk_count"].fillna(0).astype(int)
        master["fatal_crashes"] = master["fatal_crashes"].fillna(0).astype(int)
        master["serious_crashes"] = master["serious_crashes"].fillna(0).astype(int)
        master["crash_risk_level"] = master["crash_risk_level"].fillna("none")
    else:
        master["crash_risk_count"] = 0
        master["fatal_crashes"] = 0
        master["serious_crashes"] = 0
        master["crash_risk_level"] = "none"

    # Traffic hazards
    if hazards is not None and not hazards.empty and "location" in hazards.columns:
        hz_cols = [c for c in ["datetime", "location", "incident_count", "incident_types"] if c in hazards.columns]
        hz_slim = hazards[hz_cols].copy()
        master = master.merge(hz_slim, on=["datetime", "location"], how="left")
        master["incident_count"] = master["incident_count"].fillna(0).astype(int)
        if "incident_types" in master.columns:
            master["incident_types"] = master["incident_types"].fillna("")
    else:
        master["incident_count"] = 0
        master["incident_types"] = ""

    # Transport alerts
    if alerts is not None and not alerts.empty and "location" in alerts.columns:
        al_cols = [c for c in ["datetime", "location", "alert_count", "alert_effects", "affected_routes"] if c in alerts.columns]
        al_slim = alerts[al_cols].copy()
        master = master.merge(al_slim, on=["datetime", "location"], how="left")
        master["alert_count"] = master["alert_count"].fillna(0).astype(int)
        if "alert_effects" in master.columns:
            master["alert_effects"] = master["alert_effects"].fillna("")
    else:
        master["alert_count"] = 0
        master["alert_effects"] = ""

    # Traffic counts: join by location + hour because current cleaned traffic is hourly pattern
    if traffic is not None and not traffic.empty:
        if {"location", "hour"}.issubset(traffic.columns):
            tr_cols = [c for c in ["location", "hour", "traffic_volume_mean", "traffic_volume_max", "traffic_sensor_count"] if c in traffic.columns]
            tr_slim = traffic[tr_cols].drop_duplicates(["location", "hour"])
            master = master.merge(tr_slim, on=["location", "hour"], how="left")
        elif "hour" in traffic.columns:
            tr_cols = [c for c in ["hour", "traffic_volume_mean", "traffic_volume_max", "traffic_sensor_count"] if c in traffic.columns]
            tr_slim = traffic[tr_cols].drop_duplicates("hour")
            master = master.merge(tr_slim, on="hour", how="left")

    # Pedestrian counts: Sydney CBD only by datetime
    if pedestrian is not None and not pedestrian.empty:
        ped_cols = [c for c in ["datetime", "pedestrian_count_sum", "pedestrian_count_mean", "pedestrian_sensor_count"] if c in pedestrian.columns]
        ped_slim = pedestrian[ped_cols].drop_duplicates("datetime")

        sydney_mask = master["location"] == "Sydney CBD"
        master_sydney = master[sydney_mask].merge(ped_slim, on="datetime", how="left")
        master_other = master[~sydney_mask].copy()

        for c in ped_cols:
            if c != "datetime" and c not in master_other.columns:
                master_other[c] = np.nan

        master = pd.concat([master_sydney, master_other], ignore_index=True)

    # POI / mobility join
    if poi is not None and not poi.empty and {"datetime", "location"}.issubset(poi.columns):
        poi_cols = [
            c
            for c in [
                "datetime",
                "location",
                "poi_activity",
                "poi_record_count",
                "poi_category",
                "poi_baseline",
                "poi_pct_change",
                "mobility_level",
            ]
            if c in poi.columns
        ]
        poi_slim = poi[poi_cols].copy()
        master = master.merge(poi_slim, on=["datetime", "location"], how="left")

        master["poi_activity"] = master["poi_activity"].fillna(0)
        master["poi_record_count"] = master.get("poi_record_count", pd.Series(index=master.index, dtype=float)).fillna(0)
        master["poi_category"] = master.get("poi_category", pd.Series(index=master.index, dtype=str)).fillna("")
        master["mobility_level"] = master.get("mobility_level", pd.Series(index=master.index, dtype=str)).fillna("normal")
    else:
        master["poi_activity"] = 0
        master["poi_record_count"] = 0
        master["poi_category"] = ""
        master["mobility_level"] = "normal"

        # OSM static POI join
    if (
        osm_poi is not None
        and not osm_poi.empty
        and "location" in osm_poi.columns
    ):
        osm_poi["location"] = (
            osm_poi["location"]
            .astype(str)
            .str.strip()
            .str.replace("_", " ", regex=False)
            .str.title()
        )

        master["location"] = (
            master["location"]
            .astype(str)
            .str.strip()
            .str.replace("_", " ", regex=False)
            .str.title()
        )

        osm_poi_slim = osm_poi.drop_duplicates("location")

        master = master.merge(
            osm_poi_slim,
            on="location",
            how="left",
            suffixes=("", "_osm")
        )

        poi_static_cols = [
            "total_poi_count",
            "amenity",
            "shop",
            "tourism",
            "leisure",
            "office",
            "public_transport",
            "railway",
            "other",
        ]

        for col in poi_static_cols:
            if col in master.columns:
                master[col] = pd.to_numeric(master[col], errors="coerce").fillna(0)
            else:
                master[col] = 0

        if "poi_activity_osm" in master.columns:
            master["poi_activity"] = np.where(
                master["poi_activity"].fillna(0) > 0,
                master["poi_activity"],
                master["poi_activity_osm"].fillna(0)
            )

        if "mobility_level_osm" in master.columns:
            master["mobility_level"] = np.where(
                master["mobility_level"].fillna("").astype(str).str.lower().isin(["", "normal", "nan"]),
                master["mobility_level_osm"].fillna("normal"),
                master["mobility_level"]
            )

        if "poi_category_osm" in master.columns:
            master["poi_category"] = np.where(
                master["poi_category"].fillna("").astype(str).str.strip().eq(""),
                master["poi_category_osm"].fillna(""),
                master["poi_category"]
            )

        drop_osm_cols = [c for c in master.columns if c.endswith("_osm")]
        master = master.drop(columns=drop_osm_cols)

    else:
        master["total_poi_count"] = 0
        master["amenity"] = 0
        master["shop"] = 0
        master["tourism"] = 0
        master["leisure"] = 0
        master["office"] = 0
        master["public_transport"] = 0
        master["railway"] = 0
        master["other"] = 0

    def context_label(row):
        parts = []

        rain = row.get("rain", 0) or 0
        if rain > 4:
            parts.append("heavy_rain")
        elif rain > 1:
            parts.append("moderate_rain")
        elif rain > 0.1:
            parts.append("light_rain")
        else:
            parts.append("dry")

        temp = row.get("temperature_2m", 20) or 20
        if temp >= 35:
            parts.append("extreme_heat")
        elif temp >= 30:
            parts.append("hot")
        elif temp <= 10:
            parts.append("cold")

        if row.get("is_public_holiday"):
            parts.append("public_holiday")
        if row.get("is_school_term"):
            parts.append("school_term")
        if row.get("has_nearby_event"):
            parts.append("event_nearby")
        if row.get("has_major_event"):
            parts.append("major_event")
        if row.get("incident_count", 0) > 0:
            parts.append("road_incident")
        if row.get("alert_count", 0) > 0:
            parts.append("pt_disruption")
        if row.get("poi_activity", 0) > 0:
            parts.append("poi_activity")
        if str(row.get("mobility_level", "")).lower() == "high":
            parts.append("high_mobility")

        return "+".join(parts)

    print("  Building context labels...")
    master["context_label"] = master.apply(context_label, axis=1)

    master = master.sort_values(["location", "datetime"]).reset_index(drop=True)
    
    master = master.drop_duplicates(
        subset=["location", "datetime"],
        keep="first"
    )
    save(master, "master_context_table.csv")

    print("\n  Context label distribution (top 20):")
    print(master["context_label"].value_counts().head(20).to_string())

    return master


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def print_summary():
    print(f"\n{'=' * 62}")
    print("  CLEANING COMPLETE — output files")
    print(f"{'=' * 62}")

    for f in sorted(OUT.glob("*.csv")):
        kb = f.stat().st_size // 1024
        try:
            rows = sum(1 for _ in open(f, encoding="utf-8", errors="ignore")) - 1
        except Exception:
            rows = 0
        print(f"  {f.name:<45} {rows:>10,} rows {kb:>8,} KB")

    print(f"\n  Output folder: {OUT.resolve()}/")
    print("\nNext step → Stage 2: run  python 03_build_benchmark_tasks_updated.py")


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║   Topic 3 — Stage 1: Data Cleaning & Alignment         ║
║   Input  : data/  (raw downloaded files)               ║
║   Output : data/cleaned/                               ║
╚══════════════════════════════════════════════════════════╝
""")

    clean_weather()
    clean_holidays()
    clean_school_terms()
    clean_events()
    clean_major_events()
    clean_road_crashes()
    clean_traffic_hazards()
    clean_transport_alerts()
    clean_traffic()
    clean_pedestrian()
    clean_poi_mobility()
    clean_osm_poi_static()
    clean_massive_steps_llm_tasks()

    build_master_table() 
    print_summary()
    


if __name__ == "__main__":
    main()
