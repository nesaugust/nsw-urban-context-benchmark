import time
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import requests

OUT = Path("data/01_weather/open_meteo")
OUT.mkdir(parents=True, exist_ok=True)

START_DATE = "2022-01-01"
END_DATE = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

VARS = ",".join([
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
])

MISSING_AREAS = [
    ("Wollongong", -34.4278, 150.8931),
    ("Coffs_Harbour", -30.2963, 153.1135),
    ("Port_Macquarie", -31.4333, 152.9000),
    ("Byron_Bay", -28.6474, 153.6020),
    ("Nowra", -34.8806, 150.6000),
    ("Orange", -33.2833, 149.1000),
    ("Dubbo", -32.2569, 148.6011),
]

def download_location(name, lat, lon):
    out_file = OUT / f"{name}_weather_2022_2026.csv"

    if out_file.exists() and out_file.stat().st_size > 1000:
        print(f"✓ already exists: {out_file.name}")
        return

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": "Australia/Sydney",
        "hourly": VARS,
        "format": "csv",
    }

    for attempt in range(1, 6):
        try:
            print(f"Downloading {name} attempt {attempt}...")
            r = requests.get(url, params=params, timeout=120)

            if r.status_code == 429:
                wait = 120 * attempt
                print(f"Rate limited. Waiting {wait} seconds...")
                time.sleep(wait)
                continue

            r.raise_for_status()

            lines = r.text.splitlines()
            header_idx = next(
                i for i, line in enumerate(lines)
                if line.startswith("time,")
            )

            df = pd.read_csv(
                pd.io.common.StringIO("\n".join(lines[header_idx:]))
            )

            df.insert(0, "location", name.replace("_", " "))
            df.insert(1, "latitude", lat)
            df.insert(2, "longitude", lon)

            df.to_csv(out_file, index=False)
            print(f"✓ saved {out_file.name}: {len(df):,} rows")
            time.sleep(20)
            return

        except Exception as e:
            print(f"✗ failed {name}: {e}")
            time.sleep(60)

    print(f"✗ could not download {name}")

def main():
    print(f"Downloading missing Open-Meteo areas: {START_DATE} to {END_DATE}")

    for name, lat, lon in MISSING_AREAS:
        download_location(name, lat, lon)

    files = sorted(OUT.glob("*_weather_2022_2026.csv"))
    if files:
        merged = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        merged.to_csv(OUT / "nsw_all_regions_weather_2022_2026.csv", index=False)
        print(f"✓ merged file saved: {len(merged):,} rows")

if __name__ == "__main__":
    main()