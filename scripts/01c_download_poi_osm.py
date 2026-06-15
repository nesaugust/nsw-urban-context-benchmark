import osmnx as ox
import pandas as pd
from pathlib import Path

OUT = Path("data/07_poi_mobility/osm_poi")
OUT.mkdir(parents=True, exist_ok=True)

REGIONS = [
    "Sydney CBD, New South Wales, Australia",
    "Parramatta, New South Wales, Australia",
    "Liverpool, New South Wales, Australia",
    "Penrith, New South Wales, Australia",
    "Bondi, New South Wales, Australia",
    "Manly, New South Wales, Australia",
    "Newcastle, New South Wales, Australia",
    "Wollongong, New South Wales, Australia",
    "Coffs Harbour, New South Wales, Australia",
    "Port Macquarie, New South Wales, Australia",
    "Byron Bay, New South Wales, Australia",
    "Nowra, New South Wales, Australia",
    "Orange, New South Wales, Australia",
    "Dubbo, New South Wales, Australia",
    "Tamworth, New South Wales, Australia",
    "Wagga Wagga, New South Wales, Australia",
    "Albury, New South Wales, Australia",
    "Bathurst, New South Wales, Australia",
    "Broken Hill, New South Wales, Australia",
    "Armidale, New South Wales, Australia",
    "Katoomba, New South Wales, Australia",
    "Perisher, New South Wales, Australia",
    "Canberra, Australia",
    "Cessnock, New South Wales, Australia",
]

TAGS = {
    "amenity": True,
    "shop": True,
    "tourism": True,
    "leisure": True,
    "public_transport": True,
    "railway": True,
    "office": True,
    "building": True,
}

rows = []

for region in REGIONS:
    print(f"Downloading POIs for {region}...")

    try:
        gdf = ox.features_from_place(region, TAGS)

        if gdf.empty:
            print(f"  No POIs found for {region}")
            continue

        gdf = gdf.reset_index()

        gdf["location"] = region.split(",")[0]

        keep_cols = [
            "location",
            "osmid",
            "name",
            "amenity",
            "shop",
            "tourism",
            "leisure",
            "public_transport",
            "railway",
            "office",
            "building",
            "geometry",
        ]

        keep_cols = [c for c in keep_cols if c in gdf.columns]

        gdf = gdf[keep_cols]

        gdf["latitude"] = gdf.geometry.centroid.y
        gdf["longitude"] = gdf.geometry.centroid.x

        gdf = gdf.drop(columns=["geometry"])

        rows.append(gdf)

        print(f"  Saved {len(gdf):,} POIs")

    except Exception as e:
        print(f"  Failed for {region}: {e}")

if rows:
    poi = pd.concat(rows, ignore_index=True)

    poi.to_csv(
        OUT / "nsw_osm_poi_raw.csv",
        index=False
    )

    print(f"\nSaved raw POI data: {OUT / 'nsw_osm_poi_raw.csv'}")
    print(f"Total POIs: {len(poi):,}")

    # Aggregate POI counts by region
    poi["poi_category"] = "other"

    for col in ["amenity", "shop", "tourism", "leisure", "public_transport", "railway", "office"]:
        if col in poi.columns:
            poi.loc[poi[col].notna(), "poi_category"] = col

    agg = (
        poi.groupby(["location", "poi_category"])
        .size()
        .reset_index(name="poi_count")
    )

    agg.to_csv(
        OUT / "nsw_osm_poi_counts_by_region.csv",
        index=False
    )

    wide = (
        agg.pivot_table(
            index="location",
            columns="poi_category",
            values="poi_count",
            fill_value=0
        )
        .reset_index()
    )

    wide["total_poi_count"] = wide.drop(columns=["location"]).sum(axis=1)

    wide.to_csv(
        OUT / "nsw_osm_poi_region_features.csv",
        index=False
    )

    print(f"Saved POI region features: {OUT / 'nsw_osm_poi_region_features.csv'}")

else:
    print("No POI data downloaded.")