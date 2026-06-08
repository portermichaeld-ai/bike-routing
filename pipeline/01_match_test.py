#!/usr/bin/env python3
"""
pipeline/01_match_test.py

OSM-as-base conflation test: sample 1000 OSM highway ways, find the best-matching
NYSDOT segment per way via 15m spatial buffer, 70% overlap ratio, and ≤30° bearing
difference. NYSDOT attributes are joined onto OSM ways as nysri_* columns.

Uses a two-pass OSM read (no location index) to avoid memory issues on the full
NYS PBF: pass 1 collects way tags + node refs, pass 2 collects only the node
locations needed for the sampled ways.

Output (data/ny/match-test/):
  osm_sample.geojson    -- 1000 OSM ways with nysri_* attributes joined (null if unmatched)
  nysri_matched.geojson -- matched NYSDOT segments, linked back by osm_way_id
"""

import json
import math
import random
from pathlib import Path

import geopandas as gpd
import osmium
import pandas as pd
from shapely.geometry import LineString

# ── configuration ──────────────────────────────────────────────────────────────
SAMPLE_SIZE = 1000
BUFFER_M = 15.0
OVERLAP_THRESHOLD = 0.70
BEARING_THRESHOLD = 30.0   # degrees, direction-agnostic (0–90 scale)
RANDOM_SEED = 42

OSM_PBF = Path("data/ny/new-york-260531.osm.pbf")
NYSRI_PATH = Path("data/ny/ny_roadway_inventory.geojson")
OUTPUT_DIR = Path("data/ny/match-test")

CRS_GEO = "EPSG:4326"
CRS_METER = "EPSG:32618"  # UTM Zone 18N — covers all of NYS

ROUTABLE_HIGHWAY_TYPES = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "unclassified", "residential", "living_street",
    "service", "road",
    "cycleway", "path", "track",
}

OSM_EXPORT_TAGS = [
    "highway", "name", "ref", "maxspeed", "oneway",
    "bicycle", "access", "cycleway", "lanes", "surface", "shoulder",
]

# NYSDOT fields relevant to bikeability scoring (joined as nysri_<fieldname_lower>)
NYSRI_JOIN_FIELDS = [
    "OBJECTID",
    "Roadway_Name",
    "Functional_Class",
    "Posted_Speed_Limit_MPH",
    "AADT_Current_Estimate",
    "AADT_Last_Actual",
    "AADT_Last_Actual_Count_Year",
    "Truck_Percent_Average",
    "Truck_Percent_Actual",
    "Shoulder_Width_Rght_Pr_Dir",
    "Shoulder_Width_Left_Pr_Dir",
    "Travel_Lanes_Pr_Dir",
    "Travel_Lane_Width_Pr_Dir",
    "Pavement_Type",
    "One_Way",
    "Access_Control",
    "National_Hghway_System",
    "Segment_Length",
    "County_Name",
    "DOT_Region_Number",
]


# ── OSM two-pass readers ───────────────────────────────────────────────────────

class WayPassOne(osmium.SimpleHandler):
    """Pass 1 (no locations): collect highway way records and node ref lists."""

    def __init__(self):
        super().__init__()
        self.ways: list[dict] = []

    def way(self, w):
        tags = dict(w.tags)
        if tags.get("highway") not in ROUTABLE_HIGHWAY_TYPES:
            return
        row: dict = {
            "osm_way_id": w.id,
            "_node_refs": [n.ref for n in w.nodes],
        }
        for tag in OSM_EXPORT_TAGS:
            row[f"osm_{tag}"] = tags.get(tag)
        self.ways.append(row)


class NodePassTwo(osmium.SimpleHandler):
    """Pass 2 (no locations): read node entities and collect coords for needed IDs.

    Node lat/lon is stored directly in the PBF node entities and is accessible
    in the node handler even without building a location index (locations=False).
    """

    def __init__(self, needed: set[int]):
        super().__init__()
        self._needed = needed
        self.locs: dict[int, tuple[float, float]] = {}

    def node(self, n):
        if n.id in self._needed and n.location.valid():
            self.locs[n.id] = (n.location.lon, n.location.lat)


# ── geometry helpers ───────────────────────────────────────────────────────────

def _bearing(geom) -> float:
    """Direction of a LineString from first to last coord, mapped to 0–180°."""
    c = list(geom.coords)
    dx, dy = c[-1][0] - c[0][0], c[-1][1] - c[0][1]
    return math.degrees(math.atan2(dy, dx)) % 180


def bearing_diff(a, b) -> float:
    """Unsigned angular difference between two LineStrings, 0–90°."""
    diff = abs(_bearing(a) - _bearing(b)) % 180
    return min(diff, 180 - diff)


def _write_geojson(gdf: "gpd.GeoDataFrame", path: "Path") -> None:
    """Write RFC 7946-compliant GeoJSON (no 'crs' member). ArcGIS Online and
    most web tools expect the crs-less form and assume WGS84 lon/lat."""
    fc = json.loads(gdf.to_json())
    fc.pop("crs", None)
    with open(path, "w") as fh:
        json.dump(fc, fh)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. OSM pass 1: collect all highway ways (no location index) ────────────
    print("OSM pass 1: collecting highway way IDs and tags...")
    p1 = WayPassOne()
    p1.apply_file(str(OSM_PBF), locations=False)
    print(f"  {len(p1.ways):,} routable highway ways found")

    # ── 2. Sample 1000 ways; collect only the node IDs they reference ──────────
    sampled_ways = random.sample(p1.ways, SAMPLE_SIZE)
    needed_nodes: set[int] = set()
    for w in sampled_ways:
        needed_nodes.update(w["_node_refs"])
    print(f"  Sampled {SAMPLE_SIZE} ways, need {len(needed_nodes):,} node locations")

    # ── 3. OSM pass 2: collect node locations for sampled ways only ────────────
    print("OSM pass 2: collecting node locations...")
    p2 = NodePassTwo(needed_nodes)
    p2.apply_file(str(OSM_PBF), locations=False)
    print(f"  {len(p2.locs):,} node locations resolved")

    # ── 4. Build OSM GeoDataFrame ──────────────────────────────────────────────
    records = []
    for w in sampled_ways:
        coords = [p2.locs[nid] for nid in w["_node_refs"] if nid in p2.locs]
        if len(coords) < 2:
            continue
        rec = {k: v for k, v in w.items() if k != "_node_refs"}
        rec["geometry"] = LineString(coords)
        records.append(rec)

    osm_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=CRS_GEO)
    print(f"  {len(osm_gdf):,} OSM ways with valid geometries built")

    # ── 5. Load NYSDOT inventory ───────────────────────────────────────────────
    print("Loading NYSDOT roadway inventory...")
    nysri = gpd.read_file(NYSRI_PATH)
    print(f"  {len(nysri):,} NYSDOT segments loaded")

    # ── 6. Reproject both datasets to UTM 18N ─────────────────────────────────
    print("Reprojecting to UTM 18N...")
    osm_m = osm_gdf.to_crs(CRS_METER)
    nysri_m = nysri.to_crs(CRS_METER)
    nysri_sindex = nysri_m.sindex

    # ── 7. Match: for each OSM way find the best NYSDOT segment ───────────────
    print(f"Matching OSM ways to NYSDOT "
          f"(buffer={BUFFER_M}m, overlap≥{OVERLAP_THRESHOLD:.0%}, bearing≤{BEARING_THRESHOLD}°)...")

    match_results: list[dict] = []
    matched_nysri_rows: list[dict] = []

    for i, (_, osm_row) in enumerate(osm_m.iterrows(), 1):
        if i % 100 == 0:
            n_hit = sum(1 for r in match_results if r["nysri_id"] is not None)
            print(f"  {i}/{len(osm_m)}  matched so far: {n_hit}")

        osm_geom = osm_row.geometry
        osm_len = osm_geom.length
        if osm_len == 0:
            continue

        buffered = osm_geom.buffer(BUFFER_M)
        candidates = nysri_sindex.query(buffered)

        best_ratio = 0.0
        best_nysri_pos = None

        for pos in candidates:
            nysri_geom = nysri_m.iloc[pos].geometry

            # Overlap: what fraction of the OSM way length is covered by this NYSDOT segment
            clipped_len = nysri_geom.intersection(buffered).length
            ratio = min(1.0, clipped_len / osm_len)
            if ratio < OVERLAP_THRESHOLD:
                continue

            # Bearing: same-road alignment check
            if bearing_diff(osm_geom, nysri_geom) > BEARING_THRESHOLD:
                continue

            if ratio > best_ratio:
                best_ratio = ratio
                best_nysri_pos = pos

        result: dict = {"osm_way_id": osm_row["osm_way_id"]}

        if best_nysri_pos is not None:
            nr = nysri_m.iloc[best_nysri_pos]
            result["nysri_id"] = nr["OBJECTID"]
            result["match_overlap"] = round(best_ratio, 4)
            result["match_bearing_diff"] = round(
                bearing_diff(osm_geom, nr.geometry), 2
            )
            for field in NYSRI_JOIN_FIELDS:
                result[f"nysri_{field.lower()}"] = nr[field] if field in nr.index else None

            nysri_rec: dict = {"osm_way_id": osm_row["osm_way_id"], "geometry": nr.geometry}
            for field in NYSRI_JOIN_FIELDS:
                nysri_rec[field] = nr[field] if field in nr.index else None
            matched_nysri_rows.append(nysri_rec)
        else:
            result["nysri_id"] = None
            result["match_overlap"] = None
            result["match_bearing_diff"] = None
            for field in NYSRI_JOIN_FIELDS:
                result[f"nysri_{field.lower()}"] = None

        match_results.append(result)

    n_matched = sum(1 for r in match_results if r["nysri_id"] is not None)
    print(f"  Done — {n_matched}/{len(match_results)} OSM ways matched")

    # ── 8. Join match results onto OSM GeoDataFrame ───────────────────────────
    match_df = pd.DataFrame(match_results)
    osm_enriched = osm_gdf.merge(match_df, on="osm_way_id", how="left")

    osm_tag_cols = [f"osm_{t}" for t in OSM_EXPORT_TAGS]
    match_cols = ["nysri_id", "match_overlap", "match_bearing_diff"]
    nysri_attr_cols = [f"nysri_{f.lower()}" for f in NYSRI_JOIN_FIELDS]
    col_order = ["osm_way_id"] + osm_tag_cols + match_cols + nysri_attr_cols + ["geometry"]
    osm_enriched = osm_enriched[[c for c in col_order if c in osm_enriched.columns]]

    # ── 9. Export ──────────────────────────────────────────────────────────────
    osm_out = OUTPUT_DIR / "osm_sample.geojson"
    _write_geojson(osm_enriched, osm_out)
    print(f"Wrote {osm_out} ({len(osm_enriched)} features)")

    if matched_nysri_rows:
        nysri_out_gdf = gpd.GeoDataFrame(matched_nysri_rows, geometry="geometry", crs=CRS_METER)
        nysri_out_gdf = nysri_out_gdf.to_crs(CRS_GEO)
        nysri_out = OUTPUT_DIR / "nysri_matched.geojson"
        _write_geojson(nysri_out_gdf, nysri_out)
        print(f"Wrote {nysri_out} ({len(nysri_out_gdf)} features)")

    # ── 10. Summary ───────────────────────────────────────────────────────────
    print(f"\nSummary:")
    print(f"  OSM ways sampled:             {len(osm_enriched)}")
    print(f"  Matched to NYSDOT segment:    {n_matched}  ({n_matched/len(osm_enriched):.0%})")
    print(f"  Unmatched (use FC defaults):  {len(osm_enriched) - n_matched}")
    if n_matched:
        hit = osm_enriched[osm_enriched["nysri_id"].notna()]
        print(f"  Avg overlap ratio:            {hit['match_overlap'].mean():.3f}")
        print(f"  Avg bearing diff:             {hit['match_bearing_diff'].mean():.1f}°")


if __name__ == "__main__":
    main()
