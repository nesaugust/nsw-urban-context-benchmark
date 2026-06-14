# Topic 3 — Data Dictionary
## Weather, Events & Urban Context Benchmark — NSW

All data lives under: `C:\Users\agnes\Downloads\Capstone\weather_env\data\`

---

## 01_weather/open_meteo/
**Source:** Open-Meteo ERA5 reanalysis (archive-api.open-meteo.com)
**Date range:** 1940-01-01 → yesterday
**Format:** CSV, hourly rows
**Size:** ~81 MB (1 merged file + 24 per-region files)

### Files
| File | What's inside |
|---|---|
| `nsw_all_regions_weather.csv` | **Master file** — all 24 regions merged into one CSV |
| `Sydney_CBD_weather_1940_2026.csv` | Hourly weather for Sydney CBD only |
| `Parramatta_weather_1940_2026.csv` | Hourly weather for Parramatta only |
| *(one file per region × 24 regions)* | Same structure for each NSW location |

### Columns
| Column | Type | Description |
|---|---|---|
| `location` | string | Region name (e.g. "Sydney CBD") |
| `latitude` | float | Latitude coordinate |
| `longitude` | float | Longitude coordinate |
| `time` | datetime | Hourly timestamp (YYYY-MM-DD HH:MM) |
| `temperature_2m` | float | Air temperature at 2m height (°C) |
| `apparent_temperature` | float | Feels-like temperature (°C) |
| `relative_humidity_2m` | float | Relative humidity at 2m (%) |
| `weather_code` | int | WMO weather condition code (1=clear, 61=rain, 95=thunderstorm) |
| `cloud_cover` | float | Total cloud cover (%) |
| `cloud_cover_low` | float | Low-level cloud cover — fog/drizzle indicator (%) |
| `precipitation` | float | Total precipitation incl. snow (mm/hr) |
| `rain` | float | Liquid rainfall only (mm/hr) |
| `windspeed_10m` | float | Wind speed at 10m (km/h) |
| `windgusts_10m` | float | Wind gust speed at 10m (km/h) |
| `shortwave_radiation` | float | Solar radiation / sunshine intensity (W/m²) |
| `sunshine_duration` | float | Sunshine minutes per hour (min) |
| `wet_bulb_temperature_2m` | float | Wet bulb temp — heat+humidity stress measure (°C) |
| `boundary_layer_height` | float | Atmospheric boundary layer height — heat trapping (m) |
| `snowfall` | float | Snowfall (cm) — Perisher only, NaN elsewhere |

---

## 01_weather/nasa_power/
**Source:** NASA POWER daily meteorological API (power.larc.nasa.gov)
**Date range:** 1981-01-01 → yesterday
**Format:** CSV, daily rows
**Size:** ~27 MB (24 per-region files)

### Files
| File | What's inside |
|---|---|
| `Sydney_CBD_nasa_1981_2026.csv` | Daily weather for Sydney CBD |
| *(one file per region × 24 regions)* | Same structure per location |

### Columns
| Column | Type | Description |
|---|---|---|
| `location` | string | Region name |
| `latitude` | float | Latitude |
| `longitude` | float | Longitude |
| `YEAR` | int | Year |
| `MO` | int | Month |
| `DY` | int | Day |
| `T2M_MAX` | float | Maximum daily temperature at 2m (°C) |
| `T2M_MIN` | float | Minimum daily temperature at 2m (°C) |
| `PRECTOTCORR` | float | Corrected total precipitation (mm/day) |
| `RH2M` | float | Relative humidity at 2m (%) |
| `WS10M` | float | Wind speed at 10m (m/s) |
| `ALLSKY_SFC_SW_DWN` | float | All-sky surface solar radiation (kW·h/m²/day) |

---

## 01_weather/bom_meteostat/
**Source:** Bureau of Meteorology station observations via Meteostat library
**Date range:** 2000-01-01 → yesterday
**Format:** CSV, hourly rows
**Size:** ~602 MB (25 files including merged)

### Files
| File | What's inside |
|---|---|
| `nsw_all_regions_bom.csv` | **Master file** — all 24 regions merged (3.36M rows) |
| `Sydney_CBD_bom_2000_2026.csv` | BoM station obs for Sydney CBD |
| *(one file per region × 24 regions)* | Same structure per location |

### Columns
| Column | Type | Description |
|---|---|---|
| `location` | string | Region name |
| `latitude` | float | Latitude |
| `longitude` | float | Longitude |
| `datetime` | datetime | Hourly timestamp |
| `temperature_c` | float | Air temperature (°C) — actual BoM station reading |
| `dewpoint_c` | float | Dew point temperature (°C) |
| `humidity_pct` | float | Relative humidity (%) |
| `precipitation_mm` | float | Precipitation (mm/hr) |
| `snow_depth_cm` | float | Snow depth (cm) — alpine regions only |
| `wind_direction_deg` | float | Wind direction (degrees, 0=North) |
| `wind_speed_kmh` | float | Wind speed (km/h) |
| `wind_gust_kmh` | float | Wind gust speed (km/h) |
| `pressure_hpa` | float | Sea-level atmospheric pressure (hPa) |
| `sunshine_min` | float | Sunshine duration (minutes per hour) |
| `condition_code` | int | Meteostat weather condition code |
| `condition_label` | string | Human-readable condition (e.g. "Light rain", "Clear sky") |

> **Note:** "Cannot load hourly/YYYY/XXXXX.csv.gz" warnings are normal — Meteostat tries multiple nearby stations and uses whichever has data. All 24 locations returned data successfully.

---

## 02_events/public_holidays/
**Source:** data.gov.au — Australian Public Holidays Machine Readable Dataset
**Date range:** 2021-01-01 → 2025-12-31
**Format:** CSV, one row per holiday per jurisdiction

### Files
| File | What's inside |
|---|---|
| `australian_public_holidays_combined_2021_2025.csv` | All states + territories combined |
| `australian_public_holidays_2022.csv` | 2022 only, all states |
| `australian_public_holidays_2023.csv` | 2023 only, all states |
| `australian_public_holidays_2024.csv` | 2024 only, all states |
| `australian_public_holidays_2025.csv` | 2025 only, all states |
| `nsw_public_holidays_2021_2025.csv` | **NSW only** — pre-filtered (65 rows) |

### Columns
| Column | Type | Description |
|---|---|---|
| `Date` | date | Holiday date (YYYYMMDD or YYYY-MM-DD) |
| `Holiday Name` | string | Name of the holiday (e.g. "Christmas Day") |
| `Information` | string | Additional notes |
| `More Information` | string | URL reference |
| `Jurisdiction` | string | State code (NSW, VIC, QLD etc.) |

---

## 02_events/school_terms/
**Source:** education.nsw.gov.au — hardcoded from official published dates
**Date range:** 2022-01-01 → 2026-07-03
**Format:** CSV, one row per school day

### Files
| File | What's inside |
|---|---|
| `nsw_school_terms_daily_2022_2026.csv` | Daily flag — is it a school term day? (1,255 rows) |

### Columns
| Column | Type | Description |
|---|---|---|
| `date` | date | Calendar date |
| `year` | int | School year |
| `term` | int | Term number (1, 2, 3, or 4) |
| `is_school_term` | bool | True = school is in session |
| `term_start` | date | First day of this term |
| `term_end` | date | Last day of this term |

---

## 02_events/ticketmaster/
**Source:** Ticketmaster Discovery API v2
**Date range:** Upcoming and recent NSW events only (free tier limitation)
**Format:** CSV, one row per event

### Files
| File | What's inside |
|---|---|
| `nsw_events_ticketmaster_2022_2026.csv` | **Master file** — all events (1,200 events) |
| `nsw_events_arts_&_theatre.csv` | Arts & Theatre category only (689 events) |
| `nsw_events_music.csv` | Music/concerts category (235 events) |
| `nsw_events_sports.csv` | Sports events (97 events) |
| `nsw_events_miscellaneous.csv` | Miscellaneous category (60 events) |
| `nsw_events_undefined.csv` | Uncategorised events (116 events) |
| `nsw_events_film.csv` | Film screenings (3 events) |
| `nsw_events_2026.csv` | 2026 events only |

### Columns
| Column | Type | Description |
|---|---|---|
| `event_id` | string | Unique Ticketmaster event ID |
| `name` | string | Event name |
| `date` | date | Event date (local) |
| `time` | time | Event start time (local) |
| `status` | string | Ticket status (onsale, offsale, cancelled) |
| `venue` | string | Venue name (e.g. "Allianz Stadium") |
| `city` | string | City |
| `state` | string | State code (NSW) |
| `address` | string | Street address |
| `latitude` | float | Venue latitude |
| `longitude` | float | Venue longitude |
| `category` | string | Event category (Music, Sports, Arts & Theatre etc.) |
| `genre` | string | Genre within category |
| `sub_genre` | string | Sub-genre |
| `min_price` | float | Minimum ticket price (AUD) |
| `max_price` | float | Maximum ticket price (AUD) |
| `url` | string | Ticketmaster event URL |

> **Important:** Free tier only returns upcoming/recent events. Historical events (2022–2025) are not accessible without a paid enterprise plan.

---

## 03_incidents/road_crashes/
**Source:** Transport for NSW — NSW Road Crash Data
**Date range:** 2020-01-01 → 2024-12-31
**Format:** XLSX + CSV (XLSX files are currently broken — need manual download)
**Status:** ⚠ XLSX files are HTML pages (49KB each). Manual download required.

### Manual download steps:
1. Go to: `https://opendata.transport.nsw.gov.au/dataset/nsw-crash-data`
2. Click "Download" (CSV) for each year
3. Save to: `data\03_incidents\road_crashes\`

### Files (after manual download)
| File | What's inside |
|---|---|
| `nsw_crash_2020_2024.csv` | Crash records with location, severity, time |
| `nsw_traffic_unit_2020_2024.csv` | Vehicle/traffic unit details per crash |

### Key columns (crash file)
| Column | Type | Description |
|---|---|---|
| `Crash_ID` | string | Unique crash identifier |
| `Crash_Date` | date | Date of crash |
| `Crash_Time` | time | Time of crash |
| `Crash_Severity` | string | Fatal / Serious injury / Minor injury / Non-casualty |
| `LGA_Name` | string | Local Government Area |
| `Suburb` | string | Suburb name |
| `Latitude` | float | Crash location latitude |
| `Longitude` | float | Crash location longitude |
| `Road_Name` | string | Road where crash occurred |
| `Weather_Conditions` | string | Weather at time of crash |
| `Light_Conditions` | string | Lighting conditions |

---

## 03_incidents/live_traffic_hazards/
**Source:** TfNSW Live Traffic API (api.transport.nsw.gov.au)
**Date range:** Today's snapshot only (live feed, no historical archive)
**Format:** GeoJSON + CSV

### Files
| File | What's inside |
|---|---|
| `nsw_incident_YYYYMMDD.geojson` | Active road incidents (60 features) |
| `nsw_roadwork_YYYYMMDD.geojson` | Active roadworks (381 features) |
| `nsw_majorevent_YYYYMMDD.geojson` | Major events affecting roads (12 features) |
| `nsw_alpine_YYYYMMDD.geojson` | Alpine road conditions (6 features) |
| `nsw_all_hazards_YYYYMMDD.csv` | **All categories merged** (459 rows) |

### Columns (combined CSV)
| Column | Type | Description |
|---|---|---|
| `category` | string | incident / roadwork / majorevent / alpine |
| `incident_id` | string | Unique hazard ID |
| `type` | string | Main category (e.g. "Crash", "Road Maintenance") |
| `sub_type` | string | Sub-category detail |
| `headline` | string | Human-readable description |
| `created` | datetime | When the incident was first reported |
| `last_updated` | datetime | Last update time |
| `is_major` | bool | Whether classified as a major incident |
| `road` | string | Affected road name and direction |
| `suburb` | string | Suburb where incident is located |
| `delay_mins` | int | Estimated delay in minutes |
| `latitude` | float | Incident location latitude |
| `longitude` | float | Incident location longitude |

---

## 04_public_transport/service_alerts/
**Source:** TfNSW GTFS-RT API (protobuf format)
**Date range:** Today's snapshot (live feed, not archival)
**Format:** CSV (parsed from protobuf binary)

### Files
| File | What's inside |
|---|---|
| `gtfs_alerts_sydneytrains_YYYYMMDD.csv` | Sydney Trains service alerts (68 entities) |
| `gtfs_alerts_buses_YYYYMMDD.csv` | Bus network alerts (129 entities) |
| `gtfs_alerts_ferries_YYYYMMDD.csv` | Ferry service alerts (6 entities) |
| `gtfs_alerts_lightrail_YYYYMMDD.csv` | Light rail alerts (6 entities) |
| `gtfs_alerts_metro_YYYYMMDD.csv` | Metro alerts (4 entities) |
| `gtfs_alerts_nswtrains_YYYYMMDD.csv` | NSW Trains (intercity) alerts (27 entities) |
| `gtfs_alerts_all_YYYYMMDD.csv` | **All modes combined** (3,368 rows) |

### Columns
| Column | Type | Description |
|---|---|---|
| `mode` | string | Transport mode (sydneytrains, buses, ferries etc.) |
| `alert_id` | string | Unique alert identifier |
| `effect` | string | Effect type (DETOUR, NO_SERVICE, REDUCED_SERVICE etc.) |
| `cause` | string | Cause (MAINTENANCE, ACCIDENT, WEATHER etc.) |
| `header` | string | Short alert headline |
| `description` | string | Full alert description |
| `start_unix` | int | Alert start time (Unix timestamp) |
| `end_unix` | int | Alert end time (Unix timestamp) |
| `start_dt` | datetime | Alert start (human-readable) |
| `end_dt` | datetime | Alert end (human-readable) |
| `route_id` | string | Affected route ID |
| `stop_id` | string | Affected stop ID |
| `agency_id` | string | Agency responsible |

---

## 05_traffic/nsw_traffic_hf/
**Source:** HuggingFace — monster-monash/Traffic dataset
**Date range:** Full dataset (multi-year time series windows)
**Format:** CSV / Parquet
**Size:** ~533 MB

### Files
| File | What's inside |
|---|---|
| `nsw_traffic_train.csv` | Training split — time series of hourly traffic counts |
| `nsw_traffic_test.csv` | Test split |
| *(other HF dataset files)* | Parquet shards, metadata |

### Columns
| Column | Type | Description |
|---|---|---|
| `series_name` | string | Sensor/station identifier |
| `value_0` to `value_23` | float | Hourly traffic count for each hour in a 24-hour window |
| *(wide format)* | | Each row = one 24-hour window for one sensor |

> **Note:** This dataset is in wide format — each row represents a 24-hour window. Use `pd.melt()` to convert to long format with one row per hour.

---

## 05_traffic/tfnsw_aadt/
**Source:** TfNSW — Annual Average Daily Traffic (AADT)
**Date range:** 2022 → 2024
**Format:** CSV, one row per count station per year
**Size:** ~148 KB (3 files)

### Files
| File | What's inside |
|---|---|
| `nsw_aadt_2022.csv` | AADT for all permanent count stations, 2022 |
| `nsw_aadt_2023.csv` | AADT for all permanent count stations, 2023 |
| `nsw_aadt_2024.csv` | AADT for all permanent count stations, 2024 |

### Key columns
| Column | Type | Description |
|---|---|---|
| `Station_ID` | string | Permanent count station ID |
| `Road_Name` | string | Road name |
| `LGA` | string | Local Government Area |
| `Direction` | string | Traffic direction (North/South/East/West) |
| `AADT` | int | Annual Average Daily Traffic count |
| `Latitude` | float | Station latitude |
| `Longitude` | float | Station longitude |

---

## 06_pedestrian/sydney_pedestrian/
**Source:** City of Sydney — Pedestrian Counting System (ArcGIS open data)
**Date range:** 2020-02-14 → 2025-07-04 *(partial — full history starts 2009-02-01)*
**Format:** CSV, one row per sensor per hour
**Size:** ~16 MB (188,720 rows)

### Files
| File | What's inside |
|---|---|
| `sydney_cbd_pedestrian_counts.csv` | Hourly pedestrian counts at ~60 CBD sensor locations |

### Columns
| Column | Type | Description |
|---|---|---|
| `sensor_id` | string | Sensor identifier |
| `sensor_name` | string | Location name (e.g. "Town Hall", "Pitt St Mall") |
| `datetime` | datetime | Hourly timestamp |
| `hourly_counts` | int | Number of pedestrians counted in that hour |
| `latitude` | float | Sensor latitude |
| `longitude` | float | Sensor longitude |

> **Note:** Full history from 2009 requires manual download from:
> https://opendata.cityofsydney.nsw.gov.au/datasets/pedestrian-counts/explore
> → Download → CSV → save to `data\06_pedestrian\sydney_pedestrian\`

---

## How datasets connect

```
WEATHER (hourly, by location)
    + EVENTS (daily flags: holiday, school term, event nearby)
    + INCIDENTS (hourly counts: crashes, hazards per location)
    + TRANSPORT (hourly alerts: disruptions per mode)
    + TRAFFIC (hourly counts per sensor/road)
    + PEDESTRIAN (hourly counts per CBD sensor)
    ↓
master_context_table.csv
(one row per location × hour, all signals joined)
    ↓
benchmark QA tasks (Task 1–5)
```

---

## Key join keys

| Join | Key columns |
|---|---|
| Weather ↔ Events | `datetime` (hour), optionally `location` |
| Weather ↔ Traffic | `datetime` (hour) + `location` |
| Weather ↔ Pedestrian | `datetime` (hour) — Sydney CBD only |
| Events ↔ Traffic | `datetime` (hour) + spatial join on lat/lon |
| Incidents ↔ Weather | `datetime` (hour) + `location` |
| Transport alerts ↔ all | `datetime` (hour) — broadcast to all locations |