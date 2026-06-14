# NSW Urban Context Benchmark — Topic 3
### Weather, Events & Urban Context Benchmark

A benchmark suite for testing whether LLMs can use contextual signals
(weather, events, holidays, road incidents, public transport disruptions)
to improve urban activity prediction and reasoning — grounded in NSW open data.

---

## Project Overview

Many urban patterns cannot be explained by traffic or mobility data alone.
This project builds the **contextual layer** of a benchmark suite that aligns
external activities (weather, events, incidents) within NSW regions to test
whether models can use contextual signals to improve prediction and reasoning.

### Benchmark Tasks
| Task | Description |
|---|---|
| Task 1 — Traffic Prediction | Predict whether traffic/activity changes under conditions |
| Task 2 — Anomaly Classification | Classify abnormal urban activity given context |
| Task 3 — Region Sensitivity | Estimate which regions are most sensitive to weather |
| Task 4 — Scenario Cards | Generate reusable "rainy Friday near a stadium" cards |
| Task 5 — Contrastive Examples | Same traffic pattern, different causes |

---

## Repository Structure

```
nsw-urban-context-benchmark/
├── download_everything.py      # Download all raw data
├── download_crashes.py         # NSW road crash data specifically
├── fix_missing2.py             # Fix/retry failed downloads
├── clean_and_align.py          # Stage 1: clean + join all data
├── build_benchmark_tasks.py    # Stage 2: generate QA pairs & scenario cards
├── data_dictionary.md          # Column-level documentation for all datasets
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md

data/                           # NOT in repo — download via scripts
├── 01_weather/
│   ├── open_meteo/             # ERA5 hourly 1940→present, 24 NSW regions
│   ├── nasa_power/             # Daily solar/met 1981→present
│   └── bom_meteostat/         # BoM station obs 2000→present
├── 02_events/
│   ├── public_holidays/        # NSW public holidays 2021–2025
│   ├── school_terms/           # NSW school term dates 2022–2026
│   └── ticketmaster/           # Concerts, sports, festivals (upcoming)
├── 03_incidents/
│   ├── road_crashes/           # TfNSW crash data 2016–2024
│   └── live_traffic_hazards/   # TfNSW live hazard snapshots
├── 04_public_transport/
│   └── service_alerts/         # GTFS-RT alerts (trains, buses, ferries)
├── 05_traffic/
│   ├── nsw_traffic_hf/         # HuggingFace monster-monash/Traffic
│   └── tfnsw_aadt/             # Annual Average Daily Traffic 2022–2024
├── 06_pedestrian/
│   ├── sydney_pedestrian/      # City of Sydney CBD counts 2020–2025
│   └── melbourne_pedestrian/   # Melbourne counts 2009–present
└── cleaned/                    # Output of clean_and_align.py
    └── master_context_table.csv
```

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/nsw-urban-context-benchmark.git
cd nsw-urban-context-benchmark
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set API keys
```bash
# Windows PowerShell
$env:TICKETMASTER_API_KEY="your_key"
$env:TFNSW_API_KEY="your_key"

# Or edit the keys directly at the top of download_everything.py
```

Get free keys at:
- Ticketmaster: https://developer.ticketmaster.com/
- TfNSW Open Data: https://opendata.transport.nsw.gov.au/

### 5. Download all data
```bash
python download_everything.py
```

### 6. Clean and align
```bash
python clean_and_align.py
```

### 7. Build benchmark tasks
```bash
python build_benchmark_tasks.py
```

---

## Data Sources

| Layer | Source | Date Range | Key |
|---|---|---|---|
| Weather (hourly) | Open-Meteo ERA5 | 1940 → present | None |
| Weather (daily) | NASA POWER | 1981 → present | None |
| Weather (station) | BoM via Meteostat | 2000 → present | None |
| Public holidays | data.gov.au | 2021 → 2025 | None |
| School terms | education.nsw.gov.au | 2022 → 2026 | None |
| Events | Ticketmaster Discovery API | Upcoming | Free key |
| Road crashes | TfNSW Open Data | 2016 → 2024 | Free account |
| Live hazards | TfNSW API | Live snapshot | Free key |
| Transport alerts | TfNSW GTFS-RT | Live snapshot | Free key |
| Traffic counts | HuggingFace (monster-monash) | Multi-year | None |
| AADT counts | TfNSW Open Data | 2022 → 2024 | None |
| Pedestrian (Sydney) | City of Sydney Open Data | 2020 → 2025 | None |
| Pedestrian (Melbourne) | City of Melbourne / Zenodo | 2009 → present | None |

---

## NSW Regions Covered (24 locations)

**Greater Sydney metro:** Sydney CBD, Parramatta, Liverpool, Penrith, Bondi, Manly

**NSW Coastal:** Newcastle, Wollongong, Coffs Harbour, Port Macquarie, Byron Bay, Nowra

**Inland Regional:** Orange, Dubbo, Tamworth, Wagga Wagga, Albury, Bathurst, Broken Hill, Armidale

**Alpine & Special:** Katoomba, Perisher, Canberra, Cessnock

---

## Related Papers

| Paper | Relevance |
|---|---|
| STBench (arXiv:2406.19065) | ST reasoning benchmark template |
| STARK (OpenReview:zRhO4hizR8) | Hierarchical ST benchmark |
| TempReason (arXiv:2306.08952) | Temporal QA difficulty levels |
| STReasoner (arXiv:2601.03248) | Etiological ST reasoning |
| Massive-STEPS (arXiv:2505.11239) | POI check-in trajectories (UNSW) |
| UrbanPlanBench (arXiv:2504.21027) | Urban context knowledge benchmark |

---

## Team

Topic 3 — Weather, Events & Urban Context Benchmark
Capstone Project — University of Technology Sydney

---

## License

Data used in this project is sourced from open data portals under
Creative Commons Attribution licences. Scripts are MIT licensed.
