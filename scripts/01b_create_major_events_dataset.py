from pathlib import Path
import pandas as pd

OUT = Path("data/02_events/major_events")
OUT.mkdir(parents=True, exist_ok=True)

LOCATION_COORDS = {
    "NSW": (-33.8688, 151.2093),
    "Sydney": (-33.8688, 151.2093),
    "Sydney CBD": (-33.8688, 151.2093),
    "Sydney Olympic Park": (-33.8474, 151.0673),
    "Bondi": (-33.8915, 151.2767),
}

events = []

def add(event_name, start_date, end_date, location, event_type, source="manual_seed"):
    lat, lon = LOCATION_COORDS.get(location, (None, None))
    events.append({
        "event_name": event_name,
        "start_date": start_date,
        "end_date": end_date,
        "location": location,
        "latitude": lat,
        "longitude": lon,
        "event_type": event_type,
        "source": source,
    })

# Recurring events
for y in range(2018, 2027):
    add("Australia Day", f"{y}-01-26", f"{y}-01-26", "NSW", "holiday_event")
    add("ANZAC Day", f"{y}-04-25", f"{y}-04-25", "NSW", "holiday_event")
    add("Sydney New Year's Eve", f"{y}-12-31", f"{y+1}-01-01", "Sydney CBD", "festival")

vivid = {
    2018: ("2018-05-25", "2018-06-16"),
    2019: ("2019-05-24", "2019-06-15"),
    2022: ("2022-05-27", "2022-06-18"),
    2023: ("2023-05-26", "2023-06-17"),
    2024: ("2024-05-24", "2024-06-15"),
    2025: ("2025-05-23", "2025-06-14"),
    2026: ("2026-05-22", "2026-06-13"),
}
for _, (s, e) in vivid.items():
    add("Vivid Sydney", s, e, "Sydney CBD", "festival", "vivid_archive_seed")

royal_show = {
    2018: ("2018-03-23", "2018-04-03"),
    2019: ("2019-04-12", "2019-04-23"),
    2021: ("2021-04-01", "2021-04-12"),
    2022: ("2022-04-08", "2022-04-19"),
    2023: ("2023-04-06", "2023-04-17"),
    2024: ("2024-03-22", "2024-04-02"),
    2025: ("2025-04-11", "2025-04-22"),
    2026: ("2026-04-02", "2026-04-14"),
}
for _, (s, e) in royal_show.items():
    add("Sydney Royal Easter Show", s, e, "Sydney Olympic Park", "festival")

city2surf = {
    2018: "2018-08-12",
    2019: "2019-08-11",
    2022: "2022-08-14",
    2023: "2023-08-13",
    2024: "2024-08-11",
    2025: "2025-08-10",
    2026: "2026-08-09",
}
for _, d in city2surf.items():
    add("City2Surf", d, d, "Bondi", "sport")

sydney_marathon = {
    2018: "2018-09-16",
    2019: "2019-09-15",
    2022: "2022-09-18",
    2023: "2023-09-17",
    2024: "2024-09-15",
    2025: "2025-08-31",
    2026: "2026-08-30",
}
for _, d in sydney_marathon.items():
    add("Sydney Marathon", d, d, "Sydney CBD", "sport")

nrl_grand_final = {
    2018: "2018-09-30",
    2019: "2019-10-06",
    2020: "2020-10-25",
    2021: "2021-10-03",
    2022: "2022-10-02",
    2023: "2023-10-01",
    2024: "2024-10-06",
    2025: "2025-10-05",
    2026: "2026-10-04",
}
for _, d in nrl_grand_final.items():
    add("NRL Grand Final", d, d, "Sydney Olympic Park", "sport")

state_of_origin_sydney = {
    2018: "2018-06-24",
    2019: "2019-07-10",
    2020: "2020-11-11",
    2021: "2021-06-09",
    2022: "2022-06-08",
    2023: "2023-07-12",
    2024: "2024-06-05",
    2025: "2025-05-28",
    2026: "2026-05-27",
}
for _, d in state_of_origin_sydney.items():
    add("State of Origin Sydney Game", d, d, "Sydney Olympic Park", "sport")

# Major concerts
concerts = [
    ("Taylor Swift Reputation Stadium Tour", "2018-11-02", "2018-11-02"),
    ("Eminem Rapture Tour", "2019-02-22", "2019-02-22"),
    ("Queen + Adam Lambert", "2020-02-15", "2020-02-15"),
    ("Fire Fight Australia", "2020-02-16", "2020-02-16"),
    ("Guns N' Roses", "2022-11-27", "2022-11-27"),
    ("Red Hot Chili Peppers", "2023-02-02", "2023-02-02"),
    ("Red Hot Chili Peppers", "2023-02-04", "2023-02-04"),
    ("Ed Sheeran Mathematics Tour", "2023-02-24", "2023-02-24"),
    ("Ed Sheeran Mathematics Tour", "2023-02-25", "2023-02-25"),
    ("Harry Styles Love On Tour", "2023-03-03", "2023-03-03"),
    ("Harry Styles Love On Tour", "2023-03-04", "2023-03-04"),
    ("P!NK Summer Carnival", "2024-02-09", "2024-02-09"),
    ("P!NK Summer Carnival", "2024-02-10", "2024-02-10"),
    ("Taylor Swift The Eras Tour", "2024-02-23", "2024-02-23"),
    ("Taylor Swift The Eras Tour", "2024-02-24", "2024-02-24"),
    ("Taylor Swift The Eras Tour", "2024-02-25", "2024-02-25"),
    ("Taylor Swift The Eras Tour", "2024-02-26", "2024-02-26"),
    ("P!NK Summer Carnival", "2024-03-16", "2024-03-16"),
    ("The Weeknd After Hours Til Dawn", "2024-10-22", "2024-10-22"),
    ("The Weeknd After Hours Til Dawn", "2024-10-23", "2024-10-23"),
    ("Coldplay Music of the Spheres", "2024-11-06", "2024-11-06"),
    ("Coldplay Music of the Spheres", "2024-11-07", "2024-11-07"),
    ("Coldplay Music of the Spheres", "2024-11-09", "2024-11-09"),
    ("Coldplay Music of the Spheres", "2024-11-10", "2024-11-10"),
    ("Ed Sheeran Loop Tour", "2026-02-13", "2026-02-13"),
    ("Ed Sheeran Loop Tour", "2026-02-14", "2026-02-14"),
    ("Ed Sheeran Loop Tour", "2026-02-15", "2026-02-15"),
]

for name, s, e in concerts:
    add(name, s, e, "Sydney Olympic Park", "concert", "major_concert_seed")

df = pd.DataFrame(events)

df["start_date"] = pd.to_datetime(df["start_date"])
df["end_date"] = pd.to_datetime(df["end_date"])

df = df[
    [
        "event_name",
        "start_date",
        "end_date",
        "location",
        "latitude",
        "longitude",
        "event_type",
        "source",
    ]
]

df = df.sort_values(["start_date", "event_name"]).reset_index(drop=True)

outfile = OUT / "nsw_major_events_2018_2026.csv"
df.to_csv(outfile, index=False)

print(f"Saved: {outfile}")
print(f"Total events: {len(df):,}")
print(df.head(20))