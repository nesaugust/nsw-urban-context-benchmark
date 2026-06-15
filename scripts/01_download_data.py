"""
NSW Urban Context Benchmark — Clean Stage 0 Downloader
======================================================
Downloads the raw data folders for Topic 3 in a consistent structure.

What this script captures
-------------------------
1) Weather
   - Open-Meteo ERA5 hourly      : 1940-01-01 -> yesterday, 24 regions
   - NASA POWER daily            : 1981-01-01 -> yesterday, 24 regions
   - BoM/Meteostat hourly        : 2000-01-01 -> yesterday, 24 regions

2) Calendar / events
   - NSW public holidays         : 2021 -> 2025 from data.gov.au
   - NSW school terms            : 2022 -> 2026 static curated table
   - Ticketmaster NSW events     : available upcoming/recent events only

3) Incidents / transport
   - NSW road crash bundles      : TfNSW XLSX bundles, converted to CSV
   - TfNSW live hazards          : today snapshot only
   - TfNSW GTFS-RT alerts        : today snapshot only

4) Traffic / pedestrian
   - HuggingFace monster-monash/Traffic dataset
   - TfNSW AADT counts           : tries listed URLs and skips failed ones
   - City of Sydney pedestrian   : available public CSV endpoints

Important limitations
---------------------
- Ticketmaster Discovery API free tier generally does NOT provide full historical events.
- TfNSW live hazards and GTFS-RT alerts are live feeds, not historical archives.
- Open-Meteo full 1940 -> yesterday is chunked by year range to avoid partial/failed large downloads.
- Do NOT hardcode API keys in GitHub. Set them as environment variables instead.

PowerShell setup
----------------
$env:TICKETMASTER_API_KEY="KoO0tugWwgV1LcSVe0ykl6Lsb58IRXZ2"
$env:TFNSW_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIwSkVsRERJWTZyYU83YjgxSFU1S2NGMURjd1NZS2ZFYW9qZ3V6amlGZTRnIiwiaWF0IjoxNzgxNDI5NTA1fQ.vsQbFNTnyYDmtFBqdVf2GFUOiE5OTKLkYBzhFj3YTUY"
python 01_download_all_data_clean.py

Run one section only
--------------------
python 01_download_all_data_clean.py --only weather
python 01_download_all_data_clean.py --only holidays
python 01_download_all_data_clean.py --only events
python 01_download_all_data_clean.py --only crashes
python 01_download_all_data_clean.py --only transport
python 01_download_all_data_clean.py --only traffic
python 01_download_all_data_clean.py --only pedestrian
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable


def ensure_packages() -> None:
    packages = {
        "requests": "requests",
        "pandas": "pandas",
        "meteostat": "meteostat",
        "openpyxl": "openpyxl",
        "tqdm": "tqdm",
        "huggingface_hub": "huggingface_hub",
        "gtfs_realtime_bindings": "gtfs-realtime-bindings",
        "google.protobuf": "protobuf",
        "pyarrow": "pyarrow",
    }
    missing = []
    import importlib.util
    for import_name, pip_name in packages.items():
        root = import_name.split(".")[0]
        if importlib.util.find_spec(root) is None:
            missing.append(pip_name)
    if missing:
        print(f"Installing missing packages: {' '.join(sorted(set(missing)))}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *sorted(set(missing)), "-q"])


ensure_packages()

import pandas as pd
import requests
from meteostat import Hourly, Point


# =============================================================================
# Config
# =============================================================================

BASE_DIR = Path(__file__).parent.resolve() / "data"
TODAY = date.today()
YESTERDAY = TODAY - timedelta(days=1)
YESTERDAY_STR = YESTERDAY.strftime("%Y-%m-%d")

OPENMETEO_START = "1940-01-01"
NASA_START = "1981-01-01"
BOM_START = "2000-01-01"

TICKETMASTER_KEY = os.getenv("TICKETMASTER_API_KEY", "").strip()
TFNSW_KEY = os.getenv("TFNSW_API_KEY", "").strip()

OPEN_METEO_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "weather_code",
    "cloud_cover",
    "cloud_cover_low",
    "precipitation",
    "rain",
    "windspeed_10m",
    "windgusts_10m",
    "shortwave_radiation",
    "sunshine_duration",
    "wet_bulb_temperature_2m",
    "boundary_layer_height",
]
NASA_VARS = "T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,WS10M,ALLSKY_SFC_SW_DWN"

REGIONS = [
    ("Sydney_CBD", -33.8688, 151.2093, False),
    ("Parramatta", -33.8136, 151.0034, False),
    ("Liverpool", -33.9200, 150.9200, False),
    ("Penrith", -33.7508, 150.6942, False),
    ("Bondi", -33.8915, 151.2767, False),
    ("Manly", -33.7969, 151.2868, False),
    ("Newcastle", -32.9283, 151.7817, False),
    ("Wollongong", -34.4278, 150.8931, False),
    ("Coffs_Harbour", -30.2963, 153.1135, False),
    ("Port_Macquarie", -31.4333, 152.9000, False),
    ("Byron_Bay", -28.6474, 153.6020, False),
    ("Nowra", -34.8793, 150.6008, False),
    ("Orange", -33.2833, 149.1000, False),
    ("Dubbo", -32.2569, 148.6011, False),
    ("Tamworth", -31.0833, 150.9167, False),
    ("Wagga_Wagga", -35.1082, 147.3598, False),
    ("Albury", -36.0737, 146.9135, False),
    ("Bathurst", -33.4167, 149.5833, False),
    ("Broken_Hill", -31.9505, 141.4534, False),
    ("Armidale", -30.5128, 151.6686, False),
    ("Katoomba", -33.7153, 150.3115, False),
    ("Perisher", -36.4073, 148.4100, True),
    ("Canberra", -35.2809, 149.1300, False),
    ("Cessnock", -32.8337, 151.3554, False),
]

HOLIDAY_URLS = {
    "australian_public_holidays_combined_2021_2025.csv": "https://data.gov.au/data/dataset/b1bc6077-dadd-4f61-9f8c-002ab2cdff10/resource/33673aca-0857-42e5-b8f0-9981b4755686/download/australian-public-holidays-combined-2021-2025.csv",
    "australian_public_holidays_2025.csv": "https://data.gov.au/data/dataset/b1bc6077-dadd-4f61-9f8c-002ab2cdff10/resource/4d4d744b-50ed-45b9-ae77-760bc478ad75/download/australian_public_holidays_2025.csv",
    "australian_public_holidays_2024.csv": "https://data.gov.au/data/dataset/b1bc6077-dadd-4f61-9f8c-002ab2cdff10/resource/9e920340-0744-4031-a497-98ab796633e8/download/australian_public_holidays_2024.csv",
    "australian_public_holidays_2023.csv": "https://data.gov.au/data/dataset/b1bc6077-dadd-4f61-9f8c-002ab2cdff10/resource/d256f989-8f49-46eb-9770-1c6ee9bd2661/download/australian_public_holidays_2023.csv",
    "australian_public_holidays_2022.csv": "https://data.gov.au/data/dataset/b1bc6077-dadd-4f61-9f8c-002ab2cdff10/resource/a24ecaf2-044a-4e66-989e-a3b0d25bfb50/download/australian_public_holidays_2022.csv",
}

# Multiple fallback URLs are deliberate because TfNSW resource IDs change.
CRASH_BUNDLES = [
    {
        "filename": "nsw_crash_2020_2024.xlsx",
        "urls": [
            "https://opendata.transport.nsw.gov.au/data/dataset/06f9cf3d-0a9d-4098-b0f0-fa9efbdd3921/resource/c6351d27-b1b0-48e9-93a6-a612cba88f99/download/nsw_road_crash_data_2020-2024_crash.xlsx",
            "https://opendata.transport.nsw.gov.au/data/dataset/06f9cf3d-0a9d-4098-b0f0-fa9efbdd3921/resource/6d4d0513-9d31-47e9-a6df-a3d2494cb4d9/download/nsw_road_crash_data_2020-2024_crash.xlsx",
        ],
    },
    {
        "filename": "nsw_traffic_unit_2020_2024.xlsx",
        "urls": [
            "https://opendata.transport.nsw.gov.au/data/dataset/06f9cf3d-0a9d-4098-b0f0-fa9efbdd3921/resource/0dddf23e-e95e-4e80-a0d4-d8bbf8f47a00/download/nsw_road_crash_data_2020-2024_traffic_unit.xlsx",
        ],
    },
]

AADT_URLS = {
    "nsw_aadt_2022.csv": "https://opendata.transport.nsw.gov.au/data/dataset/aadt-traffic-volume-data/resource/6a8f9c37-ea58-4b1b-b55a-2b01f2c5c04d/download/2022_aadt.csv",
    "nsw_aadt_2023.csv": "https://opendata.transport.nsw.gov.au/data/dataset/aadt-traffic-volume-data/resource/8a2b4c61-fa71-4e9a-b662-3d12f6d5c07e/download/2023_aadt.csv",
}


# =============================================================================
# Helpers
# =============================================================================


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def log(message: str) -> None:
    print(f"\n{'=' * 72}\n{message}\n{'=' * 72}")


def valid_file(path: Path, min_kb: int = 5) -> bool:
    return path.exists() and path.stat().st_size >= min_kb * 1024


def download_file(url: str, dest: Path, headers: dict | None = None, timeout: int = 240) -> bool:
    if valid_file(dest):
        print(f"  ↷ exists: {dest.name} ({dest.stat().st_size // 1024:,} KB)")
        return True
    hdrs = {"User-Agent": "Mozilla/5.0 (compatible; NSW-Urban-Benchmark/1.0)"}
    if headers:
        hdrs.update(headers)
    try:
        print(f"  ↓ {dest.name}", end="  ", flush=True)
        r = requests.get(url, headers=hdrs, timeout=timeout, stream=True)
        if r.status_code in (401, 403, 404):
            print(f"✗ HTTP {r.status_code}")
            return False
        r.raise_for_status()
        content_type = r.headers.get("content-type", "").lower()
        first = b""
        with open(dest, "wb") as f:
            for chunk in r.iter_content(64 * 1024):
                if not chunk:
                    continue
                if not first:
                    first = chunk[:500]
                    if "text/html" in content_type and (b"<html" in first.lower() or b"login" in first.lower()):
                        print("✗ HTML/login page")
                        dest.unlink(missing_ok=True)
                        return False
                f.write(chunk)
        if dest.stat().st_size < 1024:
            print("✗ too small")
            dest.unlink(missing_ok=True)
            return False
        print(f"✓ {dest.stat().st_size // 1024:,} KB")
        return True
    except Exception as exc:
        print(f"✗ {exc}")
        dest.unlink(missing_ok=True)
        return False


def csv_with_location_from_openmeteo(text: str, location: str, lat: float, lon: float) -> pd.DataFrame:
    lines = text.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith("time")), None)
    if header_idx is None:
        raise ValueError("Open-Meteo response did not contain a CSV header starting with 'time'")
    csv_text = "\n".join(lines[header_idx:])
    from io import StringIO
    df = pd.read_csv(StringIO(csv_text))
    df.insert(0, "longitude", lon)
    df.insert(0, "latitude", lat)
    df.insert(0, "location", location.replace("_", " "))
    return df


def year_chunks(start_year: int, end_year: int, chunk_years: int = 5) -> Iterable[tuple[str, str]]:
    y = start_year
    while y <= end_year:
        end = min(y + chunk_years - 1, end_year)
        start_date = f"{y}-01-01"
        end_date = f"{end}-12-31"
        if end == YESTERDAY.year:
            end_date = YESTERDAY_STR
        yield start_date, end_date
        y = end + 1


def convert_xlsx_to_csv(path: Path) -> bool:
    csv_path = path.with_suffix(".csv")
    if valid_file(csv_path, min_kb=1):
        print(f"  ↷ exists: {csv_path.name}")
        return True
    try:
        df = pd.read_excel(path, engine="openpyxl")
        df.to_csv(csv_path, index=False)
        print(f"  ✓ converted: {csv_path.name} ({len(df):,} rows)")
        return True
    except Exception as exc:
        print(f"  ✗ conversion failed for {path.name}: {exc}")
        return False


# =============================================================================
# 1. Weather
# =============================================================================


def download_weather_openmeteo() -> None:
    log(f"1a. Open-Meteo ERA5 hourly | {OPENMETEO_START} -> {YESTERDAY_STR}")
    out = mkdir(BASE_DIR / "01_weather" / "open_meteo")
    merged_files = []

    for idx, (name, lat, lon, snow) in enumerate(REGIONS, 1):
        final_path = out / f"{name}_weather_{OPENMETEO_START[:4]}_{YESTERDAY.year}.csv"
        if valid_file(final_path, min_kb=200):
            print(f"  [{idx:02d}/24] {name:<18} ↷ full file exists")
            merged_files.append(final_path)
            continue

        frames = []
        vars_list = OPEN_METEO_VARS + (["snowfall"] if snow else [])
        hourly = ",".join(vars_list)
        print(f"  [{idx:02d}/24] {name:<18}")

        for start_date, end_date in year_chunks(1940, YESTERDAY.year, chunk_years=5):
            part_path = out / "parts" / f"{name}_{start_date[:4]}_{end_date[:4]}.csv"
            mkdir(part_path.parent)
            if valid_file(part_path, min_kb=20):
                try:
                    frames.append(pd.read_csv(part_path))
                    print(f"      ↷ {start_date[:4]}-{end_date[:4]}")
                    continue
                except Exception:
                    part_path.unlink(missing_ok=True)

            url = (
                "https://archive-api.open-meteo.com/v1/archive"
                f"?latitude={lat}&longitude={lon}"
                f"&start_date={start_date}&end_date={end_date}"
                f"&timezone=Australia%2FSydney&hourly={hourly}&format=csv"
            )
            try:
                r = requests.get(url, timeout=180)
                r.raise_for_status()
                df = csv_with_location_from_openmeteo(r.text, name, lat, lon)
                if len(df) < 100:
                    raise ValueError("too few rows returned")
                df.to_csv(part_path, index=False)
                frames.append(df)
                print(f"      ✓ {start_date} -> {end_date}: {len(df):,} rows")
                time.sleep(0.6)
            except Exception as exc:
                print(f"      ✗ {start_date} -> {end_date}: {exc}")

        if frames:
            full = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["location", "time"])
            full.to_csv(final_path, index=False)
            merged_files.append(final_path)
            print(f"      ✓ saved full: {final_path.name} ({len(full):,} rows)")
        else:
            print(f"      ✗ no data saved for {name}")

    if merged_files:
        frames = [pd.read_csv(p) for p in merged_files]
        merged = pd.concat(frames, ignore_index=True)
        merged.to_csv(out / "nsw_all_regions_weather.csv", index=False)
        print(f"  ✓ merged Open-Meteo: {len(merged):,} rows")


def download_weather_nasa() -> None:
    log(f"1b. NASA POWER daily | {NASA_START} -> {YESTERDAY_STR}")
    out = mkdir(BASE_DIR / "01_weather" / "nasa_power")
    frames = []
    start = NASA_START.replace("-", "")
    end = YESTERDAY_STR.replace("-", "")

    for idx, (name, lat, lon, _) in enumerate(REGIONS, 1):
        dest = out / f"{name}_nasa_{NASA_START[:4]}_{YESTERDAY.year}.csv"
        if valid_file(dest, min_kb=20):
            print(f"  [{idx:02d}/24] {name:<18} ↷ exists")
            try: frames.append(pd.read_csv(dest))
            except Exception: pass
            continue
        url = (
            "https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters={NASA_VARS}&community=AG&longitude={lon}&latitude={lat}"
            f"&start={start}&end={end}&format=CSV"
        )
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            lines = r.text.splitlines()
            hidx = next(i for i, line in enumerate(lines) if line.startswith("YEAR"))
            from io import StringIO
            df = pd.read_csv(StringIO("\n".join(lines[hidx:])))
            df.insert(0, "longitude", lon)
            df.insert(0, "latitude", lat)
            df.insert(0, "location", name.replace("_", " "))
            df.to_csv(dest, index=False)
            frames.append(df)
            print(f"  [{idx:02d}/24] {name:<18} ✓ {len(df):,} rows")
            time.sleep(0.5)
        except Exception as exc:
            print(f"  [{idx:02d}/24] {name:<18} ✗ {exc}")

    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out / "nsw_all_regions_nasa_power.csv", index=False)


def download_weather_bom() -> None:
    log(f"1c. BoM via Meteostat hourly | {BOM_START} -> {YESTERDAY_STR}")
    out = mkdir(BASE_DIR / "01_weather" / "bom_meteostat")
    start_dt = datetime.strptime(BOM_START, "%Y-%m-%d")
    end_dt = datetime(YESTERDAY.year, YESTERDAY.month, YESTERDAY.day, 23)
    frames = []
    rename = {
        "time": "datetime", "temp": "temperature_c", "dwpt": "dewpoint_c",
        "rhum": "humidity_pct", "prcp": "precipitation_mm", "snow": "snow_depth_cm",
        "wdir": "wind_direction_deg", "wspd": "wind_speed_kmh", "wpgt": "wind_gust_kmh",
        "pres": "pressure_hpa", "tsun": "sunshine_min", "coco": "condition_code",
    }

    for idx, (name, lat, lon, _) in enumerate(REGIONS, 1):
        dest = out / f"{name}_bom_{BOM_START[:4]}_{YESTERDAY.year}.csv"
        if valid_file(dest, min_kb=20):
            print(f"  [{idx:02d}/24] {name:<18} ↷ exists")
            try: frames.append(pd.read_csv(dest))
            except Exception: pass
            continue
        try:
            df = Hourly(Point(lat, lon), start_dt, end_dt, timezone="Australia/Sydney").fetch()
            if df.empty:
                raise ValueError("Meteostat returned no rows")
            df = df.reset_index().rename(columns=rename)
            df.insert(0, "longitude", lon)
            df.insert(0, "latitude", lat)
            df.insert(0, "location", name.replace("_", " "))
            df.to_csv(dest, index=False)
            frames.append(df)
            print(f"  [{idx:02d}/24] {name:<18} ✓ {len(df):,} rows")
            time.sleep(0.4)
        except Exception as exc:
            print(f"  [{idx:02d}/24] {name:<18} ✗ {exc}")

    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out / "nsw_all_regions_bom.csv", index=False)


def download_weather() -> None:
    download_weather_openmeteo()
    download_weather_nasa()
    download_weather_bom()


# =============================================================================
# 2. Holidays and school terms
# =============================================================================


def download_holidays() -> None:
    log("2a. NSW public holidays")
    out = mkdir(BASE_DIR / "02_events" / "public_holidays")
    for fname, url in HOLIDAY_URLS.items():
        download_file(url, out / fname, timeout=120)

    combined = out / "australian_public_holidays_combined_2021_2025.csv"
    nsw_out = out / "nsw_public_holidays_2021_2025.csv"
    if combined.exists():
        try:
            df = pd.read_csv(combined)
            jurisdiction_col = next((c for c in df.columns if c.lower() == "jurisdiction"), None)
            if jurisdiction_col:
                nsw = df[df[jurisdiction_col].astype(str).str.upper().eq("NSW")].copy()
                nsw.to_csv(nsw_out, index=False)
                print(f"  ✓ NSW-only holidays: {len(nsw):,} rows")
        except Exception as exc:
            print(f"  ✗ NSW holiday filter failed: {exc}")


def download_school_terms() -> None:
    log("2b. NSW school terms daily flags")
    out = mkdir(BASE_DIR / "02_events" / "school_terms")
    dest = out / "nsw_school_terms_daily_2022_2026.csv"
    if valid_file(dest, min_kb=1):
        print(f"  ↷ exists: {dest.name}")
        return
    terms = [
        (2022, 1, "2022-01-28", "2022-04-09"), (2022, 2, "2022-04-26", "2022-07-01"),
        (2022, 3, "2022-07-19", "2022-09-23"), (2022, 4, "2022-10-10", "2022-12-20"),
        (2023, 1, "2023-01-27", "2023-04-06"), (2023, 2, "2023-04-24", "2023-06-30"),
        (2023, 3, "2023-07-17", "2023-09-22"), (2023, 4, "2023-10-09", "2023-12-19"),
        (2024, 1, "2024-01-29", "2024-04-12"), (2024, 2, "2024-04-29", "2024-07-05"),
        (2024, 3, "2024-07-22", "2024-09-27"), (2024, 4, "2024-10-14", "2024-12-20"),
        (2025, 1, "2025-01-28", "2025-04-11"), (2025, 2, "2025-04-28", "2025-07-04"),
        (2025, 3, "2025-07-21", "2025-09-26"), (2025, 4, "2025-10-13", "2025-12-19"),
        (2026, 1, "2026-01-27", "2026-04-10"), (2026, 2, "2026-04-27", "2026-07-03"),
        (2026, 3, "2026-07-20", "2026-09-25"), (2026, 4, "2026-10-12", "2026-12-18"),
    ]
    rows = []
    for year, term, start, end in terms:
        for d in pd.date_range(start, end, freq="D"):
            rows.append({"date": d.date().isoformat(), "year": year, "term": term, "is_school_term": True, "term_start": start, "term_end": end})
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"  ✓ saved {len(rows):,} school-term days")


def download_holiday_layers() -> None:
    download_holidays()
    download_school_terms()


# =============================================================================
# 3. Ticketmaster events
# =============================================================================


def parse_ticketmaster_event(event: dict) -> dict:
    venue = (event.get("_embedded", {}).get("venues") or [{}])[0]
    loc = venue.get("location", {})
    classification = (event.get("classifications") or [{}])[0]
    price = (event.get("priceRanges") or [{}])[0]
    return {
        "event_id": event.get("id", ""),
        "name": event.get("name", ""),
        "date": event.get("dates", {}).get("start", {}).get("localDate", ""),
        "time": event.get("dates", {}).get("start", {}).get("localTime", ""),
        "status": event.get("dates", {}).get("status", {}).get("code", ""),
        "venue": venue.get("name", ""),
        "city": venue.get("city", {}).get("name", ""),
        "state": venue.get("state", {}).get("stateCode", ""),
        "address": venue.get("address", {}).get("line1", ""),
        "latitude": loc.get("latitude", ""),
        "longitude": loc.get("longitude", ""),
        "category": classification.get("segment", {}).get("name", ""),
        "genre": classification.get("genre", {}).get("name", ""),
        "sub_genre": classification.get("subGenre", {}).get("name", ""),
        "min_price": price.get("min", ""),
        "max_price": price.get("max", ""),
        "url": event.get("url", ""),
    }


def download_events_ticketmaster() -> None:
    log("3. Ticketmaster NSW events | API available upcoming/recent only")
    out = mkdir(BASE_DIR / "02_events" / "ticketmaster")
    dest = out / "nsw_events_ticketmaster_available.csv"
    if not TICKETMASTER_KEY:
        print("  ✗ TICKETMASTER_API_KEY not set; skipping Ticketmaster")
        return
    if valid_file(dest, min_kb=5):
        print(f"  ↷ exists: {dest.name}")
        return

    base_url = "https://app.ticketmaster.com/discovery/v2/events.json"
    rows = []
    page = 0
    total_pages = 1
    while page < total_pages and page < 50:
        params = {
            "apikey": TICKETMASTER_KEY,
            "countryCode": "AU",
            "stateCode": "NSW",
            "size": 200,
            "page": page,
            "sort": "date,asc",
        }
        try:
            r = requests.get(base_url, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", "3")))
                continue
            if r.status_code == 401:
                print("  ✗ Ticketmaster key invalid")
                return
            r.raise_for_status()
            data = r.json()
            events = data.get("_embedded", {}).get("events", [])
            if not events:
                break
            rows.extend(parse_ticketmaster_event(e) for e in events)
            page_info = data.get("page", {})
            total_pages = min(page_info.get("totalPages", 1), 50)
            print(f"  page {page + 1}/{total_pages}: total {len(rows):,}")
            page += 1
            time.sleep(0.25)
        except Exception as exc:
            print(f"  ✗ Ticketmaster failed on page {page}: {exc}")
            break

    if rows:
        df = pd.DataFrame(rows).drop_duplicates(subset=["event_id"])
        df.to_csv(dest, index=False)
        print(f"  ✓ saved {len(df):,} events")


# =============================================================================
# 4. Road crashes
# =============================================================================


def download_crashes() -> None:
    log("4. NSW road crash bundles")
    out = mkdir(BASE_DIR / "03_incidents" / "road_crashes")
    for bundle in CRASH_BUNDLES:
        dest = out / bundle["filename"]
        ok = False
        for url in bundle["urls"]:
            ok = download_file(url, dest, timeout=300)
            if ok:
                break
        if ok:
            convert_xlsx_to_csv(dest)
        else:
            print(f"  ⚠ Could not download {bundle['filename']}; check TfNSW portal manually")


# =============================================================================
# 5. Live hazards and GTFS alerts
# =============================================================================


def download_live_hazards() -> None:
    log("5a. TfNSW live traffic hazards | today snapshot")
    out = mkdir(BASE_DIR / "03_incidents" / "live_traffic_hazards")
    if not TFNSW_KEY:
        print("  ✗ TFNSW_API_KEY not set; skipping live hazards")
        return
    today = TODAY.strftime("%Y%m%d")
    headers = {"Authorization": f"apikey {TFNSW_KEY}", "Accept": "application/json"}
    endpoints = {
        "incident": "https://api.transport.nsw.gov.au/v1/live/hazards/incident/open",
        "roadwork": "https://api.transport.nsw.gov.au/v1/live/hazards/roadwork/open",
        "majorevent": "https://api.transport.nsw.gov.au/v1/live/hazards/majorevent/open",
        "alpine": "https://api.transport.nsw.gov.au/v1/live/hazards/alpine/open",
    }
    combined = []
    for name, url in endpoints.items():
        dest = out / f"nsw_{name}_{today}.geojson"
        if not valid_file(dest, min_kb=1):
            download_file(url, dest, headers=headers, timeout=60)
        if dest.exists():
            try:
                data = json.loads(dest.read_text(encoding="utf-8"))
                for feature in data.get("features", []):
                    props = feature.get("properties", {})
                    geom = feature.get("geometry", {})
                    coords = geom.get("coordinates", [None, None])
                    road = (props.get("roads") or [{}])[0]
                    combined.append({
                        "snapshot_date": TODAY.isoformat(), "category": name,
                        "incident_id": props.get("id", ""), "type": props.get("mainCategory", ""),
                        "sub_type": props.get("subCategoryA", ""), "headline": props.get("headline", ""),
                        "created": props.get("created", ""), "last_updated": props.get("lastUpdated", ""),
                        "is_major": props.get("isMajor", False), "road": road.get("fullDescription", ""),
                        "suburb": road.get("suburb", ""),
                        "longitude": coords[0] if len(coords) > 0 else None,
                        "latitude": coords[1] if len(coords) > 1 else None,
                    })
            except Exception as exc:
                print(f"  ✗ parse failed for {dest.name}: {exc}")
    if combined:
        pd.DataFrame(combined).to_csv(out / f"nsw_all_hazards_{today}.csv", index=False)
        print(f"  ✓ combined hazards: {len(combined):,} rows")


def download_transport_alerts() -> None:
    log("5b. TfNSW GTFS-RT alerts | today snapshot")
    out = mkdir(BASE_DIR / "04_public_transport" / "service_alerts")
    if not TFNSW_KEY:
        print("  ✗ TFNSW_API_KEY not set; skipping GTFS alerts")
        return
    try:
        from google.transit import gtfs_realtime_pb2
    except Exception as exc:
        print(f"  ✗ gtfs-realtime-bindings import failed: {exc}")
        return

    today = TODAY.strftime("%Y%m%d")
    headers = {"Authorization": f"apikey {TFNSW_KEY}", "Accept": "application/x-google-protobuf"}
    modes = {
        "sydneytrains": "https://api.transport.nsw.gov.au/v2/gtfs/alerts/sydneytrains",
        "buses": "https://api.transport.nsw.gov.au/v2/gtfs/alerts/buses",
        "ferries": "https://api.transport.nsw.gov.au/v2/gtfs/alerts/ferries",
        "lightrail": "https://api.transport.nsw.gov.au/v2/gtfs/alerts/lightrail",
        "metro": "https://api.transport.nsw.gov.au/v2/gtfs/alerts/metro",
        "nswtrains": "https://api.transport.nsw.gov.au/v2/gtfs/alerts/nswtrains",
    }
    rows = []
    for mode, url in modes.items():
        try:
            r = requests.get(url, headers=headers, timeout=60)
            if r.status_code in (401, 403, 404):
                print(f"  {mode:<14} ✗ HTTP {r.status_code}")
                continue
            r.raise_for_status()
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(r.content)
            for entity in feed.entity:
                if not entity.HasField("alert"):
                    continue
                alert = entity.alert
                header = alert.header_text.translation[0].text if alert.header_text.translation else ""
                desc = alert.description_text.translation[0].text if alert.description_text.translation else ""
                periods = alert.active_period or [None]
                informed = alert.informed_entity or [None]
                for period in periods:
                    for inf in informed:
                        rows.append({
                            "snapshot_date": TODAY.isoformat(), "mode": mode, "alert_id": entity.id,
                            "effect": str(alert.effect), "cause": str(alert.cause),
                            "header": header, "description": desc,
                            "start_unix": period.start if period else "", "end_unix": period.end if period else "",
                            "route_id": inf.route_id if inf else "", "stop_id": inf.stop_id if inf else "",
                        })
            print(f"  {mode:<14} ✓ {len(feed.entity):,} entities")
        except Exception as exc:
            print(f"  {mode:<14} ✗ {exc}")
    if rows:
        df = pd.DataFrame(rows)
        for col in ["start_unix", "end_unix"]:
            df[col.replace("unix", "dt")] = pd.to_datetime(pd.to_numeric(df[col], errors="coerce"), unit="s", errors="coerce")
        df.to_csv(out / f"gtfs_alerts_all_{today}.csv", index=False)
        print(f"  ✓ combined alerts: {len(df):,} rows")


def download_transport() -> None:
    download_live_hazards()
    download_transport_alerts()


# =============================================================================
# 6. Traffic
# =============================================================================


def download_traffic_hf() -> None:
    log("6a. HuggingFace traffic dataset")
    out = mkdir(BASE_DIR / "05_traffic" / "nsw_traffic_hf")
    if any(p.stat().st_size > 50_000 for p in out.rglob("*.csv")):
        print("  ↷ existing traffic CSV found; skipping")
        return
    try:
        from datasets import load_dataset
        ds = load_dataset("monster-monash/Traffic")
        for split in ds.keys():
            df = ds[split].to_pandas()
            df.to_csv(out / f"nsw_traffic_{split}.csv", index=False)
            print(f"  ✓ {split}: {len(df):,} rows")
        return
    except Exception as exc:
        print(f"  datasets method failed: {exc}")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="monster-monash/Traffic", repo_type="dataset", local_dir=str(out), ignore_patterns=["*.md", "*.txt"])
        print("  ✓ snapshot downloaded")
    except Exception as exc:
        print(f"  ✗ HuggingFace snapshot failed: {exc}")


def download_aadt() -> None:
    log("6b. TfNSW AADT counts")
    out = mkdir(BASE_DIR / "05_traffic" / "tfnsw_aadt")
    for fname, url in AADT_URLS.items():
        download_file(url, out / fname, timeout=180)


def download_traffic() -> None:
    download_traffic_hf()
    download_aadt()


# =============================================================================
# 7. Pedestrian
# =============================================================================


def download_pedestrian() -> None:
    log("7. City of Sydney pedestrian counts")
    out = mkdir(BASE_DIR / "06_pedestrian" / "sydney_pedestrian")
    dest = out / "sydney_cbd_pedestrian_counts.csv"
    if valid_file(dest, min_kb=100):
        print(f"  ↷ exists: {dest.name}")
        return
    endpoints = [
        "https://opendata.cityofsydney.nsw.gov.au/api/explore/v2.1/catalog/datasets/pedestrian-counts/exports/csv?lang=en&timezone=Australia%2FSydney&use_labels=true&delimiter=%2C",
        "https://data-cityofsydney.opendata.arcgis.com/datasets/66421e1dfe264bb19c76179ae92281cf_0.csv",
        "https://opendata.arcgis.com/datasets/1707db9c1c434fd09c68e00e0e4d7ab6_0.csv",
    ]
    for url in endpoints:
        if download_file(url, dest, timeout=300):
            return
    print("  ⚠ Could not auto-download pedestrian counts; download manually from City of Sydney Open Data")


# =============================================================================
# Summary / main
# =============================================================================


def print_summary() -> None:
    log("Download status")
    checks = {
        "01_weather/open_meteo": BASE_DIR / "01_weather" / "open_meteo",
        "01_weather/nasa_power": BASE_DIR / "01_weather" / "nasa_power",
        "01_weather/bom_meteostat": BASE_DIR / "01_weather" / "bom_meteostat",
        "02_events/public_holidays": BASE_DIR / "02_events" / "public_holidays",
        "02_events/school_terms": BASE_DIR / "02_events" / "school_terms",
        "02_events/ticketmaster": BASE_DIR / "02_events" / "ticketmaster",
        "03_incidents/road_crashes": BASE_DIR / "03_incidents" / "road_crashes",
        "03_incidents/live_traffic_hazards": BASE_DIR / "03_incidents" / "live_traffic_hazards",
        "04_public_transport/service_alerts": BASE_DIR / "04_public_transport" / "service_alerts",
        "05_traffic/nsw_traffic_hf": BASE_DIR / "05_traffic" / "nsw_traffic_hf",
        "05_traffic/tfnsw_aadt": BASE_DIR / "05_traffic" / "tfnsw_aadt",
        "06_pedestrian/sydney_pedestrian": BASE_DIR / "06_pedestrian" / "sydney_pedestrian",
    }
    for label, folder in checks.items():
        files = [p for p in folder.rglob("*.*") if p.is_file() and p.stat().st_size > 1024] if folder.exists() else []
        size_kb = sum(p.stat().st_size for p in files) // 1024
        marker = "✓" if files else "✗"
        print(f"  {marker} {label:<45} {len(files):>4} files {size_kb:>10,} KB")


SECTION_MAP = {
    "weather": [download_weather],
    "holidays": [download_holiday_layers],
    "events": [download_events_ticketmaster],
    "crashes": [download_crashes],
    "transport": [download_transport],
    "traffic": [download_traffic],
    "pedestrian": [download_pedestrian],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NSW Urban Context Benchmark raw data")
    parser.add_argument("--only", choices=SECTION_MAP.keys(), help="Run one section only")
    args = parser.parse_args()

    print(f"""
NSW Urban Context Benchmark — Stage 0 Downloader
Target end date: {YESTERDAY_STR}
Ticketmaster key: {'SET' if TICKETMASTER_KEY else 'NOT SET'}
TfNSW key       : {'SET' if TFNSW_KEY else 'NOT SET'}
""")

    funcs = SECTION_MAP[args.only] if args.only else [fn for section in SECTION_MAP.values() for fn in section]
    for fn in funcs:
        try:
            fn()
        except KeyboardInterrupt:
            print("\nInterrupted. Partial downloads are kept.")
            break
        except Exception as exc:
            import traceback
            print(f"\nERROR in {fn.__name__}: {exc}")
            traceback.print_exc()
            print("Continuing to next section...\n")
    print_summary()


if __name__ == "__main__":
    main()
