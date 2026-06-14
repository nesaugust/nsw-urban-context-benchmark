"""
Topic 3 — Weather, Events & Urban Context Benchmark
Master Dataset Downloader — NSW ONLY (Complete)
================================================
Covers ALL layers required by the benchmark spec:
  Weather · Events (concerts/sports/festivals) · Holidays · School Terms
  Road Incidents · Public Transport Disruptions · Traffic Crashes
  Pedestrian Counts · POI/Mobility · Urban Context · Benchmark QA sets

Folder structure:
    data/
    ├── 01_weather/
    │   ├── open_meteo/              ← hourly ERA5 weather, 24 NSW regions
    │   ├── nasa_power/              ← daily solar+temp, 24 NSW regions
    │   └── bom_meteostat/           ← BoM station obs, 24 NSW regions
    ├── 02_events/
    │   ├── public_holidays/         ← NSW public holidays 2022–2025
    │   ├── school_terms/            ← NSW school term dates (scraped)
    │   └── ticketmaster/            ← concerts/sports/festivals via API
    ├── 03_incidents/
    │   ├── road_crashes/            ← NSW crash data XLSX 2020–2024
    │   └── live_traffic_hazards/    ← road incidents GeoJSON (TfNSW)
    ├── 04_public_transport/
    │   └── service_alerts/          ← GTFS-RT alerts (TfNSW API)
    ├── 05_traffic/
    │   └── nsw_traffic_hf/          ← NSW hourly traffic counts (HuggingFace)
    ├── 06_pedestrian/
    │   └── sydney_pedestrian/       ← City of Sydney CBD pedestrian counts
    ├── 07_poi_mobility/
    │   └── massive_steps/           ← Massive-STEPS Sydney (HuggingFace)
    └── 08_benchmarks/
        ├── stark_1k/
        ├── stark_10k/
        ├── stbench/
        └── tempreason/

Install:
    pip install requests pandas datasets meteostat tqdm openpyxl

Usage:
    python download_all_datasets.py              # run everything
    python download_all_datasets.py --only weather
    python download_all_datasets.py --only events
    python download_all_datasets.py --only incidents
    python download_all_datasets.py --only transport
    python download_all_datasets.py --only traffic
    python download_all_datasets.py --only pedestrian
    python download_all_datasets.py --only poi
    python download_all_datasets.py --only benchmarks

Note on Ticketmaster:
    Get a free API key at https://developer.ticketmaster.com/
    Set it as an environment variable before running:
        Windows:  set TICKETMASTER_API_KEY=your_key_here
        Mac/Linux: export TICKETMASTER_API_KEY=your_key_here
    Without a key the events section is skipped gracefully.

Note on TfNSW API:
    Get a free API key at https://opendata.transport.nsw.gov.au/
    Set it as:
        Windows:  set TFNSW_API_KEY=your_key_here
"""

import os
import sys
import time
import json
import argparse
import requests
import pandas as pd
from io import StringIO
from pathlib import Path
from datetime import datetime, timedelta

# ── Dependency check ───────────────────────────────────────────────────────────
def check_imports():
    missing = []
    for pkg in ["requests", "pandas", "datasets", "meteostat", "tqdm", "openpyxl"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Run:  pip install {' '.join(missing)}")
        sys.exit(1)

check_imports()

from datasets import load_dataset
from meteostat import Point, Hourly

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_DIR   = Path("data")
START_DATE = "2022-01-01"
END_DATE   = (datetime.today() - timedelta(days=5)).strftime("%Y-%m-%d")
TIMEZONE   = "Australia%2FSydney"

# API keys from environment variables (optional but recommended)
TICKETMASTER_KEY = os.environ.get("TICKETMASTER_API_KEY", "")
TFNSW_KEY        = os.environ.get("TFNSW_API_KEY", "")

OPEN_METEO_VARS = ",".join([
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "weather_code", "cloud_cover", "cloud_cover_low",
    "precipitation", "rain",
    "windspeed_10m", "windgusts_10m",
    "shortwave_radiation", "sunshine_duration",
    "wet_bulb_temperature_2m", "boundary_layer_height",
])

NASA_VARS = "T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,WS10M,ALLSKY_SFC_SW_DWN"

# 24 NSW regions
REGIONS = [
    ("Sydney_CBD",    -33.8688, 151.2093, False),
    ("Parramatta",    -33.8136, 151.0034, False),
    ("Liverpool",     -33.9200, 150.9200, False),
    ("Penrith",       -33.7508, 150.6942, False),
    ("Bondi",         -33.8915, 151.2767, False),
    ("Manly",         -33.7969, 151.2868, False),
    ("Newcastle",     -32.9283, 151.7817, False),
    ("Wollongong",    -34.4278, 150.8931, False),
    ("Coffs_Harbour", -30.2963, 153.1135, False),
    ("Port_Macquarie",-31.4333, 152.9000, False),
    ("Byron_Bay",     -28.6474, 153.6020, False),
    ("Nowra",         -34.8793, 150.6008, False),
    ("Orange",        -33.2833, 149.1000, False),
    ("Dubbo",         -32.2569, 148.6011, False),
    ("Tamworth",      -31.0833, 150.9167, False),
    ("Wagga_Wagga",   -35.1082, 147.3598, False),
    ("Albury",        -36.0737, 146.9135, False),
    ("Bathurst",      -33.4167, 149.5833, False),
    ("Broken_Hill",   -31.9505, 141.4534, False),
    ("Armidale",      -30.5128, 151.6686, False),
    ("Katoomba",      -33.7153, 150.3115, False),
    ("Perisher",      -36.4073, 148.4100, True),
    ("Canberra",      -35.2809, 149.1300, False),
    ("Cessnock",      -32.8337, 151.3554, False),
]

# Public holidays direct CSV links (data.gov.au — verified working)
HOLIDAY_URLS = {
    "australian_public_holidays_combined_2021_2025.csv":
        "https://data.gov.au/data/dataset/b1bc6077-dadd-4f61-9f8c-002ab2cdff10/resource/33673aca-0857-42e5-b8f0-9981b4755686/download/australian-public-holidays-combined-2021-2025.csv",
    "australian_public_holidays_2025.csv":
        "https://data.gov.au/data/dataset/b1bc6077-dadd-4f61-9f8c-002ab2cdff10/resource/4d4d744b-50ed-45b9-ae77-760bc478ad75/download/australian_public_holidays_2025.csv",
    "australian_public_holidays_2024.csv":
        "https://data.gov.au/data/dataset/b1bc6077-dadd-4f61-9f8c-002ab2cdff10/resource/9e920340-0744-4031-a497-98ab796633e8/download/australian_public_holidays_2024.csv",
    "australian_public_holidays_2023.csv":
        "https://data.gov.au/data/dataset/b1bc6077-dadd-4f61-9f8c-002ab2cdff10/resource/d256f989-8f49-46eb-9770-1c6ee9bd2661/download/australian_public_holidays_2023.csv",
    "australian_public_holidays_2022.csv":
        "https://data.gov.au/data/dataset/b1bc6077-dadd-4f61-9f8c-002ab2cdff10/resource/a24ecaf2-044a-4e66-989e-a3b0d25bfb50/download/australian_public_holidays_2022.csv",
}

# NSW Road Crash Data direct XLSX links (TfNSW — verified working)
CRASH_URLS = {
    "nsw_crash_2020_2024.xlsx":
        "https://opendata.transport.nsw.gov.au/data/dataset/06f9cf3d-0a9d-4098-b0f0-fa9efbdd3921/resource/6d4d0513-9d31-47e9-a6df-a3d2494cb4d9/download/nsw_road_crash_data_2020-2024_crash.xlsx",
    "nsw_traffic_unit_2020_2024.xlsx":
        "https://opendata.transport.nsw.gov.au/data/dataset/06f9cf3d-0a9d-4098-b0f0-fa9efbdd3921/resource/0dddf23e-e95e-4e80-a0d4-d8bbf8f47a00/download/nsw_road_crash_data_2020-2024_traffic_unit.xlsx",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def log(msg):
    print(f"\n{'='*62}\n  {msg}\n{'='*62}")

def save_hf_dataset(ds, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ds.keys():
        df = ds[split].to_pandas()
        path = out_dir / f"{name}_{split}.csv"
        df.to_csv(path, index=False)
        print(f"  ✓  {path.name}  ({len(df):,} rows)")

def download_file(url: str, dest: Path, label: str = "", timeout: int = 120) -> bool:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        kb = len(resp.content) // 1024
        print(f"  ✓  {label or dest.name}  ({kb:,} KB)")
        return True
    except Exception as e:
        print(f"  ✗  {label or dest.name}  FAILED: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 1. WEATHER
# ══════════════════════════════════════════════════════════════════════════════

def download_open_meteo():
    log("1a/8  WEATHER — Open-Meteo ERA5  (24 NSW regions)")
    out = mkdir(BASE_DIR / "01_weather" / "open_meteo")
    merged, failed = [], []

    for i, (name, lat, lon, snow) in enumerate(REGIONS, 1):
        vars_ = OPEN_METEO_VARS + (",snowfall" if snow else "")
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={START_DATE}&end_date={END_DATE}"
            f"&timezone={TIMEZONE}&hourly={vars_}&format=csv"
        )
        dest = out / f"{name}_weather_{START_DATE[:4]}_{END_DATE[:4]}.csv"
        print(f"  [{i:02d}/24] {name.replace('_',' '):<22}", end="  ")
        try:
            resp = requests.get(url, timeout=90)
            resp.raise_for_status()
            lines = resp.text.splitlines()
            hidx  = next((i for i, l in enumerate(lines) if l.startswith("time")), 0)
            rows  = [l for l in lines[hidx+1:] if l.strip()]
            hdr   = "location,latitude,longitude," + lines[hidx]
            data  = [f"{name.replace('_',' ')},{lat},{lon},{r}" for r in rows]
            dest.write_text("\n".join(lines[:hidx] + [hdr] + data), encoding="utf-8")
            print(f"✓  {len(data):,} rows  →  {dest.name}")
            merged.append(pd.read_csv(dest, comment="#"))
            time.sleep(0.6)
        except Exception as e:
            print(f"✗  FAILED: {e}")
            failed.append(name)

    if merged:
        pd.concat(merged, ignore_index=True).to_csv(
            out / "nsw_all_regions_weather.csv", index=False)
        print(f"\n  ✓  MERGED  →  nsw_all_regions_weather.csv")
    if failed:
        print(f"\n  ✗  Failed: {', '.join(failed)}")


def download_nasa_power():
    log("1b/8  WEATHER — NASA POWER daily  (24 NSW regions)")
    out = mkdir(BASE_DIR / "01_weather" / "nasa_power")
    start, end = START_DATE.replace("-",""), END_DATE.replace("-","")
    merged = []

    for i, (name, lat, lon, _) in enumerate(REGIONS, 1):
        url = (
            f"https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters={NASA_VARS}&community=AG"
            f"&longitude={lon}&latitude={lat}&start={start}&end={end}&format=CSV"
        )
        dest = out / f"{name}_nasa_{START_DATE[:4]}_{END_DATE[:4]}.csv"
        print(f"  [{i:02d}/24] {name.replace('_',' '):<22}", end="  ")
        try:
            resp = requests.get(url, timeout=90)
            resp.raise_for_status()
            lines = resp.text.splitlines()
            didx  = next((i for i, l in enumerate(lines) if l.startswith("YEAR")), None)
            if didx is None:
                raise ValueError("No YEAR header found")
            rows = [l for l in lines[didx+1:] if l.strip()]
            hdr  = "location,latitude,longitude," + lines[didx]
            data = [f"{name.replace('_',' ')},{lat},{lon},{r}" for r in rows]
            dest.write_text("\n".join(lines[:didx] + [hdr] + data), encoding="utf-8")
            print(f"✓  {len(data):,} rows  →  {dest.name}")
            merged.append(pd.read_csv(StringIO("\n".join([hdr]+data))))
            time.sleep(0.5)
        except Exception as e:
            print(f"✗  FAILED: {e}")

    if merged:
        pd.concat(merged, ignore_index=True).to_csv(
            out / "nsw_all_regions_nasa_power.csv", index=False)
        print(f"\n  ✓  MERGED  →  nsw_all_regions_nasa_power.csv")


def download_bom_meteostat():
    log("1c/8  WEATHER — BoM via Meteostat  (24 NSW regions)")
    out = mkdir(BASE_DIR / "01_weather" / "bom_meteostat")
    RENAME = {
        "temp":"temperature_c","dwpt":"dewpoint_c","rhum":"humidity_pct",
        "prcp":"precipitation_mm","snow":"snow_depth_cm","wdir":"wind_direction_deg",
        "wspd":"wind_speed_kmh","wpgt":"wind_gust_kmh","pres":"pressure_hpa",
        "tsun":"sunshine_min","coco":"condition_code",
    }
    COND = {
        1:"Clear sky",2:"Few clouds",3:"Scattered clouds",4:"Broken clouds",
        5:"Overcast",6:"Fog",7:"Light rain",8:"Rain",9:"Heavy rain",
        14:"Light snowfall",15:"Snowfall",16:"Heavy snowfall",
        28:"Lightning",29:"Hail",30:"Thunderstorm",32:"Storm",
    }
    s = datetime.strptime(START_DATE, "%Y-%m-%d")
    e = datetime.today() - timedelta(days=1)
    merged = []

    for i, (name, lat, lon, _) in enumerate(REGIONS, 1):
        print(f"  [{i:02d}/24] {name.replace('_',' '):<22}", end="  ")
        try:
            df = Hourly(Point(lat, lon), s, e, timezone="Australia/Sydney").fetch()
            if df.empty:
                raise ValueError("No data")
            df = df.reset_index().rename(columns={"time":"datetime"})
            df.rename(columns=RENAME, inplace=True)
            df.insert(0,"location", name.replace("_"," "))
            df.insert(1,"latitude", lat)
            df.insert(2,"longitude", lon)
            if "condition_code" in df.columns:
                df["condition_label"] = df["condition_code"].map(COND)
            path = out / f"{name}_bom_{START_DATE[:4]}_{END_DATE[:4]}.csv"
            df.to_csv(path, index=False)
            miss = df["temperature_c"].isna().mean() * 100
            print(f"✓  {len(df):,} rows  ({miss:.0f}% missing)  →  {path.name}")
            merged.append(df)
            time.sleep(0.5)
        except Exception as e:
            print(f"✗  FAILED: {e}")

    if merged:
        pd.concat(merged, ignore_index=True).to_csv(
            out / "nsw_all_regions_bom.csv", index=False)
        print(f"\n  ✓  MERGED  →  nsw_all_regions_bom.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 2. EVENTS — Holidays, School Terms, Concerts/Sports
# ══════════════════════════════════════════════════════════════════════════════

def download_holidays():
    log("2a/8  EVENTS — NSW Public Holidays  (data.gov.au)")
    out = mkdir(BASE_DIR / "02_events" / "public_holidays")

    for fname, url in HOLIDAY_URLS.items():
        dest = out / fname
        print(f"  {fname}  ", end="")
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            print(f"✓  ({len(resp.text.splitlines())-1:,} rows)")
        except Exception as e:
            print(f"✗  {e}")
        time.sleep(0.3)

    # NSW-only filtered file
    try:
        df   = pd.read_csv(out / "australian_public_holidays_combined_2021_2025.csv")
        jcol = next((c for c in df.columns if c.lower()=="jurisdiction"), None)
        if jcol:
            nsw = df[df[jcol].str.upper()=="NSW"].copy()
            nsw.to_csv(out / "nsw_public_holidays_2021_2025.csv", index=False)
            print(f"\n  ✓  NSW-only  →  nsw_public_holidays_2021_2025.csv  ({len(nsw)} rows)")
    except Exception as e:
        print(f"\n  ! Could not create NSW-only file: {e}")


def download_school_terms():
    log("2b/8  EVENTS — NSW School Term Dates  (scraped from education.nsw.gov.au)")
    out = mkdir(BASE_DIR / "02_events" / "school_terms")

    # Hardcoded term dates 2022–2026 from education.nsw.gov.au
    # These are stable published dates — no scraping needed
    terms = [
        # year, term, start, end
        (2022,1,"2022-01-28","2022-04-09"),
        (2022,2,"2022-04-26","2022-07-01"),
        (2022,3,"2022-07-19","2022-09-23"),
        (2022,4,"2022-10-10","2022-12-20"),
        (2023,1,"2023-01-27","2023-04-06"),
        (2023,2,"2023-04-24","2023-06-30"),
        (2023,3,"2023-07-17","2023-09-22"),
        (2023,4,"2023-10-09","2023-12-19"),
        (2024,1,"2024-01-29","2024-04-12"),
        (2024,2,"2024-04-29","2024-07-05"),
        (2024,3,"2024-07-22","2024-09-27"),
        (2024,4,"2024-10-14","2024-12-20"),
        (2025,1,"2025-01-28","2025-04-11"),
        (2025,2,"2025-04-28","2025-07-04"),
        (2025,3,"2025-07-21","2025-09-26"),
        (2025,4,"2025-10-13","2025-12-19"),
        (2026,1,"2026-01-27","2026-04-10"),
        (2026,2,"2026-04-27","2026-07-03"),
    ]
    df = pd.DataFrame(terms, columns=["year","term","term_start","term_end"])
    df["term_start"] = pd.to_datetime(df["term_start"])
    df["term_end"]   = pd.to_datetime(df["term_end"])
    df["is_school_term"] = True

    # Expand to daily rows for easy joining with other datasets
    rows = []
    for _, r in df.iterrows():
        d = r["term_start"]
        while d <= r["term_end"]:
            rows.append({"date": d.date(), "year": r["year"],
                         "term": r["term"], "is_school_term": True,
                         "term_start": r["term_start"].date(),
                         "term_end": r["term_end"].date()})
            d += timedelta(days=1)
    daily = pd.DataFrame(rows)
    daily.to_csv(out / "nsw_school_terms_daily_2022_2026.csv", index=False)
    df.to_csv(out / "nsw_school_terms_summary_2022_2026.csv", index=False)
    print(f"  ✓  {len(daily):,} school days  →  nsw_school_terms_daily_2022_2026.csv")
    print(f"  ✓  Summary  →  nsw_school_terms_summary_2022_2026.csv")


def download_ticketmaster_events():
    log("2c/8  EVENTS — Concerts, Sports & Festivals  (Ticketmaster Discovery API)")
    out = mkdir(BASE_DIR / "02_events" / "ticketmaster")

    if not TICKETMASTER_KEY:
        print("  ! TICKETMASTER_API_KEY not set — skipping.")
        print("  → Get a free key at: https://developer.ticketmaster.com/")
        print("  → Then run:  set TICKETMASTER_API_KEY=your_key  (Windows)")
        print("               export TICKETMASTER_API_KEY=your_key  (Mac/Linux)")
        print("  → Re-run:    python download_all_datasets.py --only events")
        return

    BASE_URL = "https://app.ticketmaster.com/discovery/v2/events.json"
    all_events = []

    # Pull events year by year to stay within pagination limits
    for year in range(2022, 2027):
        start_dt = f"{year}-01-01T00:00:00Z"
        end_dt   = f"{year}-12-31T23:59:59Z"
        if year == 2026:
            end_dt = f"{END_DATE}T23:59:59Z"

        page = 0
        while True:
            params = {
                "apikey":         TICKETMASTER_KEY,
                "countryCode":    "AU",
                "stateCode":      "NSW",
                "startDateTime":  start_dt,
                "endDateTime":    end_dt,
                "size":           200,
                "page":           page,
            }
            try:
                resp = requests.get(BASE_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                events = data.get("_embedded", {}).get("events", [])
                if not events:
                    break

                for ev in events:
                    venue = (ev.get("_embedded", {}).get("venues") or [{}])[0]
                    loc   = venue.get("location", {})
                    all_events.append({
                        "event_id":    ev.get("id"),
                        "name":        ev.get("name"),
                        "date":        ev.get("dates", {}).get("start", {}).get("localDate"),
                        "time":        ev.get("dates", {}).get("start", {}).get("localTime"),
                        "venue":       venue.get("name"),
                        "city":        venue.get("city", {}).get("name"),
                        "state":       venue.get("state", {}).get("stateCode"),
                        "latitude":    loc.get("latitude"),
                        "longitude":   loc.get("longitude"),
                        "category":    (ev.get("classifications") or [{}])[0]
                                        .get("segment", {}).get("name"),
                        "genre":       (ev.get("classifications") or [{}])[0]
                                        .get("genre", {}).get("name"),
                        "url":         ev.get("url"),
                    })

                total_pages = data.get("page", {}).get("totalPages", 1)
                print(f"  {year}  page {page+1}/{total_pages}  "
                      f"({len(all_events)} events so far)", end="\r")
                page += 1
                if page >= total_pages:
                    break
                time.sleep(0.25)  # rate limit

            except Exception as e:
                print(f"\n  ✗  Year {year} page {page}: {e}")
                break

    if all_events:
        df = pd.DataFrame(all_events)
        df.to_csv(out / "nsw_events_ticketmaster_2022_2026.csv", index=False)
        print(f"\n  ✓  {len(df):,} events  →  nsw_events_ticketmaster_2022_2026.csv")

        # Save per-category breakdowns
        for cat in df["category"].dropna().unique():
            safe = cat.replace("/","_").replace(" ","_")
            sub  = df[df["category"]==cat]
            sub.to_csv(out / f"nsw_events_{safe.lower()}.csv", index=False)
            print(f"  ✓  {cat}  ({len(sub)} events)  →  nsw_events_{safe.lower()}.csv")
    else:
        print("\n  ! No events returned.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. ROAD INCIDENTS & CRASHES
# ══════════════════════════════════════════════════════════════════════════════

def download_road_incidents():
    log("3a/8  INCIDENTS — NSW Road Crashes 2020–2024  (TfNSW)")
    out = mkdir(BASE_DIR / "03_incidents" / "road_crashes")

    for fname, url in CRASH_URLS.items():
        dest = out / fname
        print(f"  {fname}  ", end="")
        ok = download_file(url, dest, fname)
        if ok and fname.endswith(".xlsx"):
            # Also convert to CSV for easier loading
            try:
                df = pd.read_excel(dest)
                csv_path = dest.with_suffix(".csv")
                df.to_csv(csv_path, index=False)
                print(f"  ✓  Converted  →  {csv_path.name}  ({len(df):,} rows)")
            except Exception as e:
                print(f"  ! Could not convert to CSV: {e}")
        time.sleep(0.5)

    print("\n  Portal: https://opendata.transport.nsw.gov.au/dataset/nsw-crash-data")


def download_live_traffic_hazards():
    log("3b/8  INCIDENTS — Live Traffic Hazards GeoJSON  (TfNSW)")
    out = mkdir(BASE_DIR / "03_incidents" / "live_traffic_hazards")

    # TfNSW Live Traffic Hazards — GeoJSON endpoint (public, no auth)
    # Covers: incidents, fires, floods, roadworks, major events, alpine conditions
    url  = "https://api.transport.nsw.gov.au/v1/live/hazards/incident/open"
    dest = out / f"nsw_live_hazards_{datetime.today().strftime('%Y%m%d')}.geojson"

    headers = {}
    if TFNSW_KEY:
        headers["Authorization"] = f"apikey {TFNSW_KEY}"

    print(f"  Fetching current hazards snapshot...", end="  ")
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
        features = len(data.get("features", []))
        print(f"✓  {features} active incidents  →  {dest.name}")

        # Also flatten to CSV for easier analysis
        if features > 0:
            rows = []
            for f in data["features"]:
                props = f.get("properties", {})
                geom  = f.get("geometry", {})
                coords = geom.get("coordinates", [None, None])
                rows.append({
                    "incident_id":    props.get("id"),
                    "type":           props.get("mainCategory"),
                    "sub_type":       props.get("subCategoryA"),
                    "description":    props.get("headline"),
                    "start_time":     props.get("created"),
                    "last_updated":   props.get("lastUpdated"),
                    "road":           props.get("roads", [{}])[0].get("fullDescription") if props.get("roads") else None,
                    "suburb":         props.get("roads", [{}])[0].get("suburb") if props.get("roads") else None,
                    "latitude":       coords[1] if len(coords) > 1 else None,
                    "longitude":      coords[0] if len(coords) > 0 else None,
                })
            pd.DataFrame(rows).to_csv(
                out / f"nsw_live_hazards_{datetime.today().strftime('%Y%m%d')}.csv",
                index=False
            )
            print(f"  ✓  Flattened CSV saved")
    except requests.HTTPError as e:
        if resp.status_code == 401:
            print(f"✗  401 Unauthorized")
            print("  → Get a free TfNSW API key at: https://opendata.transport.nsw.gov.au/")
            print("  → Set it: set TFNSW_API_KEY=your_key  then re-run")
        else:
            print(f"✗  {e}")
    except Exception as e:
        print(f"✗  {e}")

    print("\n  Historical incidents (last 3 months) via Live Traffic search:")
    print("  https://www.livetraffic.com/")
    print("  TfNSW Historical Traffic API:")
    print("  https://opendata.transport.nsw.gov.au/dataset/historical-traffic-api")


# ══════════════════════════════════════════════════════════════════════════════
# 4. PUBLIC TRANSPORT DISRUPTIONS
# ══════════════════════════════════════════════════════════════════════════════

def download_transport_alerts():
    log("4/8  PUBLIC TRANSPORT — Service Alerts & Disruptions  (TfNSW GTFS-RT)")
    out = mkdir(BASE_DIR / "04_public_transport" / "service_alerts")

    # ── Historic Alerts spreadsheet (June–Dec 2017, free download) ──
    hist_url  = "https://opendata.transport.nsw.gov.au/data/dataset/449cc355-cbab-4262-a377-73b29492221d/resource/b8b41bcd-3a69-4f32-b9de-d8f04a4b0b9c/download/historicalservicealerts.xlsx"
    hist_dest = out / "historic_service_alerts_2017.xlsx"
    print("  Downloading historic alerts (2017 sample)...", end="  ")
    ok = download_file(hist_url, hist_dest, "historic_service_alerts_2017.xlsx")
    if ok:
        try:
            df = pd.read_excel(hist_dest)
            df.to_csv(out / "historic_service_alerts_2017.csv", index=False)
            print(f"  ✓  Converted  →  historic_service_alerts_2017.csv  ({len(df):,} rows)")
        except Exception as e:
            print(f"  ! Convert failed: {e}")

    # ── Realtime GTFS-RT alerts snapshot (requires TfNSW API key) ──
    print("\n  Fetching current GTFS-RT service alerts snapshot...")
    if not TFNSW_KEY:
        print("  ! TFNSW_API_KEY not set — skipping realtime alerts.")
        print("  → Register free at: https://opendata.transport.nsw.gov.au/")
        print("  → Set key: set TFNSW_API_KEY=your_key  then re-run with --only transport")
    else:
        # GTFS-RT alerts come as protobuf — we request JSON wrapper via TfNSW API
        url = "https://api.transport.nsw.gov.au/v2/gtfs/alerts"
        headers = {"Authorization": f"apikey {TFNSW_KEY}",
                   "Accept": "application/json"}
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            snapshot_path = out / f"gtfs_alerts_{datetime.today().strftime('%Y%m%d_%H%M')}.json"
            snapshot_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            n = len(data.get("entity", []))
            print(f"  ✓  {n} active alerts  →  {snapshot_path.name}")

            # Flatten to CSV
            rows = []
            for entity in data.get("entity", []):
                alert = entity.get("alert", {})
                for period in alert.get("active_period", [{}]):
                    for informed in alert.get("informed_entity", [{}]):
                        rows.append({
                            "alert_id":     entity.get("id"),
                            "effect":       alert.get("effect"),
                            "cause":        alert.get("cause"),
                            "header":       alert.get("header_text", {}).get("translation", [{}])[0].get("text"),
                            "description":  alert.get("description_text", {}).get("translation", [{}])[0].get("text"),
                            "start":        period.get("start"),
                            "end":          period.get("end"),
                            "route_id":     informed.get("route_id"),
                            "stop_id":      informed.get("stop_id"),
                            "agency_id":    informed.get("agency_id"),
                        })
            if rows:
                pd.DataFrame(rows).to_csv(
                    out / f"gtfs_alerts_{datetime.today().strftime('%Y%m%d')}.csv",
                    index=False)
                print(f"  ✓  Flattened to CSV")
        except Exception as e:
            print(f"  ✗  Realtime alerts failed: {e}")

    print("\n  GTFS-RT alerts portal:")
    print("  https://opendata.transport.nsw.gov.au/dataset/public-transport-realtime-alerts-v2")
    print("\n  Opal patronage (tap-on/tap-off per stop):")
    print("  https://opendata.transport.nsw.gov.au/dataset/opal-tap-on-and-tap-off")


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAFFIC COUNTS
# ══════════════════════════════════════════════════════════════════════════════

def download_traffic():
    log("5/8  TRAFFIC — NSW Hourly Traffic Counts  (HuggingFace: monster-monash/Traffic)")
    out = mkdir(BASE_DIR / "05_traffic" / "nsw_traffic_hf")
    print("  Loading from HuggingFace (may take a few minutes)...")
    try:
        ds = load_dataset("monster-monash/Traffic", trust_remote_code=True)
        save_hf_dataset(ds, out, "nsw_traffic")
    except Exception as e:
        print(f"  ✗  FAILED: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PEDESTRIAN — City of Sydney (NSW only)
# ══════════════════════════════════════════════════════════════════════════════

def download_pedestrian():
    log("6/8  PEDESTRIAN — City of Sydney CBD Counts  (NSW only)")
    out = mkdir(BASE_DIR / "06_pedestrian" / "sydney_pedestrian")

    url  = "https://opendata.arcgis.com/datasets/1707db9c1c434fd09c68e00e0e4d7ab6_0.csv"
    dest = out / "sydney_cbd_pedestrian_counts.csv"
    print("  Trying City of Sydney ArcGIS endpoint...", end="  ")
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        rows = len(resp.text.splitlines()) - 1
        print(f"✓  {rows:,} rows  →  {dest.name}")
    except Exception as e:
        print(f"✗  {e}")
        print("  ─── MANUAL DOWNLOAD ──────────────────────────────────────")
        print("  1. Go to: https://opendata.cityofsydney.nsw.gov.au/datasets/pedestrian-counts")
        print("  2. Click Download → CSV")
        print(f"  3. Save to: {out.resolve()}/")
        print("  ──────────────────────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════════
# 7. POI & MOBILITY — Massive-STEPS Sydney
# ══════════════════════════════════════════════════════════════════════════════

def download_poi():
    log("7/8  POI & MOBILITY — Massive-STEPS Sydney  (CRUISE/UNSW via HuggingFace)")
    out = mkdir(BASE_DIR / "07_poi_mobility" / "massive_steps")
    print("  Loading from HuggingFace...")
    try:
        ds = load_dataset("CRUISEResearchGroup/Massive-STEPS-Sydney",
                          trust_remote_code=True)
        save_hf_dataset(ds, out, "massive_steps_sydney")
    except Exception as e:
        print(f"  ✗  Sydney failed ({e}), trying parent repo...")
        try:
            ds = load_dataset("CRUISEResearchGroup/Massive-STEPS",
                              trust_remote_code=True)
            save_hf_dataset(ds, out, "massive_steps")
        except Exception as e2:
            print(f"  ✗  {e2}")
            print("  → git clone https://github.com/cruiseresearchgroup/Massive-STEPS")


# ══════════════════════════════════════════════════════════════════════════════
# 8. BENCHMARK QA DATASETS
# ══════════════════════════════════════════════════════════════════════════════

def download_benchmarks():
    log("8/8  BENCHMARKS — STARK / STBench / TempReason  (HuggingFace)")
    datasets = [
        ("prquan/STARK_1k",      BASE_DIR / "08_benchmarks" / "stark_1k",   "stark_1k"),
        ("prquan/STARK_10k",     BASE_DIR / "08_benchmarks" / "stark_10k",  "stark_10k"),
        ("LwbXc/STBench",        BASE_DIR / "08_benchmarks" / "stbench",    "stbench"),
        ("tonytan48/TempReason", BASE_DIR / "08_benchmarks" / "tempreason", "tempreason"),
    ]
    for hf_id, out_dir, name in datasets:
        print(f"\n  Loading {hf_id} ...")
        try:
            ds = load_dataset(hf_id, trust_remote_code=True)
            save_hf_dataset(ds, out_dir, name)
        except Exception as e:
            print(f"  ✗  {hf_id}  FAILED: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY + MAIN
# ══════════════════════════════════════════════════════════════════════════════

SECTION_MAP = {
    "weather":    [download_open_meteo, download_nasa_power, download_bom_meteostat],
    "events":     [download_holidays, download_school_terms, download_ticketmaster_events],
    "incidents":  [download_road_incidents, download_live_traffic_hazards],
    "transport":  [download_transport_alerts],
    "traffic":    [download_traffic],
    "pedestrian": [download_pedestrian],
    "poi":        [download_poi],
    "benchmarks": [download_benchmarks],
}

def print_summary():
    print(f"\n{'='*62}")
    print("  DOWNLOAD COMPLETE — file summary")
    print(f"{'='*62}")
    all_files = sorted(
        [f for f in BASE_DIR.rglob("*") if f.is_file()
         and f.suffix in (".csv",".json",".geojson",".xlsx",".xls")]
    )
    if not all_files:
        print("  No files found.")
        return
    cur = None
    for f in all_files:
        folder = f.parent.relative_to(BASE_DIR)
        if folder != cur:
            print(f"\n  📁 data/{folder}/")
            cur = folder
        kb = f.stat().st_size // 1024
        print(f"       {f.name:<50}  {kb:>8,} KB")
    total_kb = sum(f.stat().st_size for f in all_files) // 1024
    print(f"\n  {len(all_files)} files  |  {total_kb:,} KB total")
    print(f"  Root: {BASE_DIR.resolve()}/")


def main():
    parser = argparse.ArgumentParser(description="Topic 3 NSW full dataset downloader")
    parser.add_argument("--only", choices=list(SECTION_MAP.keys()),
                        help="Download only one section")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║   Topic 3 — NSW Full Dataset Downloader                 ║
║   Period  : {START_DATE}  →  {END_DATE}           ║
║   Sections: weather · events · incidents · transport    ║
║             traffic · pedestrian · poi · benchmarks     ║
╚══════════════════════════════════════════════════════════╝

API keys (optional):
  Ticketmaster : {'SET ✓' if TICKETMASTER_KEY else 'NOT SET — events section will be skipped'}
  TfNSW        : {'SET ✓' if TFNSW_KEY else 'NOT SET — realtime alerts will be skipped'}
""")

    sections = (SECTION_MAP[args.only] if args.only
                else [fn for fns in SECTION_MAP.values() for fn in fns])

    for fn in sections:
        try:
            fn()
        except KeyboardInterrupt:
            print("\n\nInterrupted. Partial downloads saved.")
            break
        except Exception as e:
            print(f"\n  ✗  Unexpected error: {e}\n  Continuing...\n")

    print_summary()


if __name__ == "__main__":
    main()