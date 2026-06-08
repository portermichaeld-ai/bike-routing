#!/usr/bin/env python3
"""
pipeline/01_conflate.py

Production conflation: join NYSDOT road inventory attributes onto all
OSM highway ways in NYS. Uses shapely 2.x STRtree bulk query for
vectorized spatial matching — much faster than per-way iteration.

Matching criteria: 15m buffer, 70% overlap ratio, ≤30° bearing difference.

Output: data/ny/osm_nysri_join.parquet
  One row per OSM way. NYSRI columns are null for unmatched ways.
  No geometry — OSM geometries remain in the PBF for routing/tile steps.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import osmium
import pandas as pd
import shapely
from shapely import STRtree
from shapely.geometry import LineString

# ── configuration ──────────────────────────────────────────────────────────────
BUFFER_M = 15.0
OVERLAP_THRESHOLD = 0.70
BEARING_THRESHOLD = 30.0

OSM_PBF   = Path("data/ny/new-york-260531.osm.pbf")
NYSRI_PATH = Path("data/ny/ny_roadway_inventory.geojson")
OUTPUT    = Path("data/ny/osm_nysri_join.parquet")

CRS_GEO   = "EPSG:4326"
CRS_METER = "EPSG:32618"   # UTM Zone 18N — covers all of NYS

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

# NYSRI fields required by the bikeability scoring model
NYSRI_JOIN_FIELDS = [
    "OBJECTID",
    "Roadway_Name",
    "Functional_Class",           # hard excludes + FC defaults
    "Posted_Speed_Limit_MPH",     # step 6 speed penalty
    "AADT_Current_Estimate",      # step 6 sigmoid AADT
    "AADT_Last_Actual",
    "AADT_Last_Actual_Count_Year",
    "Truck_Percent_Average",      # step 6 truck penalty + floor (all null in current NYSRI release)
    "Truck_Percent_Actual",       # use this instead — ~9% coverage on arterials
    "Shoulder_Width_Rght_Pr_Dir", # step 6 shoulder component
    "Shoulder_Width_Left_Pr_Dir",
    "Travel_Lanes_Pr_Dir",        # lane modifier
    "Travel_Lane_Width_Pr_Dir",
    "Pavement_Type",              # surface modifier
    "One_Way",
    "Access_Control",             # expressway hard exclude
    "National_Hghway_System",
    "Segment_Length",             # country-lane override (>0.5 mi)
    "County_Name",
    "DOT_Region_Number",
]


# ── OSM two-pass readers ───────────────────────────────────────────────────────

class WayPassOne(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.ways: list[dict] = []

    def way(self, w):
        tags = dict(w.tags)
        if tags.get("highway") not in ROUTABLE_HIGHWAY_TYPES:
            return
        row: dict = {"osm_way_id": w.id, "_node_refs": [n.ref for n in w.nodes]}
        for tag in OSM_EXPORT_TAGS:
            row[f"osm_{tag}"] = tags.get(tag)
        self.ways.append(row)


class NodePassTwo(osmium.SimpleHandler):
    def __init__(self, needed: set[int]):
        super().__init__()
        self._needed = needed
        self.locs: dict[int, tuple[float, float]] = {}

    def node(self, n):
        if n.id in self._needed and n.location.valid():
            self.locs[n.id] = (n.location.lon, n.location.lat)


# ── geometry helpers ───────────────────────────────────────────────────────────

def bulk_bearing(geoms: np.ndarray) -> np.ndarray:
    """First-to-last bearing for an array of LineStrings, mapped to 0–180°."""
    p0  = shapely.get_coordinates(shapely.get_point(geoms, 0))
    pm1 = shapely.get_coordinates(shapely.get_point(geoms, -1))
    dx = pm1[:, 0] - p0[:, 0]
    dy = pm1[:, 1] - p0[:, 1]
    return np.degrees(np.arctan2(dy, dx)) % 180


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. OSM pass 1 ─────────────────────────────────────────────────────────
    print("OSM pass 1: collecting highway ways...")
    p1 = WayPassOne()
    p1.apply_file(str(OSM_PBF), locations=False)
    all_ways = p1.ways
    print(f"  {len(all_ways):,} routable ways found")

    # ── 2. Collect needed node IDs ────────────────────────────────────────────
    needed_nodes: set[int] = set()
    for w in all_ways:
        needed_nodes.update(w["_node_refs"])
    print(f"  {len(needed_nodes):,} unique nodes needed")

    # ── 3. OSM pass 2 ─────────────────────────────────────────────────────────
    print("OSM pass 2: collecting node locations...")
    p2 = NodePassTwo(needed_nodes)
    p2.apply_file(str(OSM_PBF), locations=False)
    del needed_nodes
    print(f"  {len(p2.locs):,} locations resolved")

    # ── 4. Build geometries ───────────────────────────────────────────────────
    print("Building OSM geometries...")
    records = []
    for w in all_ways:
        coords = [p2.locs[nid] for nid in w["_node_refs"] if nid in p2.locs]
        if len(coords) < 2:
            continue
        rec = {k: v for k, v in w.items() if k != "_node_refs"}
        rec["geometry"] = LineString(coords)
        records.append(rec)
    del all_ways, p2

    osm_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=CRS_GEO)
    del records
    print(f"  {len(osm_gdf):,} ways with valid geometries")

    # ── 5. Load NYSRI ─────────────────────────────────────────────────────────
    print("Loading NYSDOT road inventory...")
    nysri = gpd.read_file(NYSRI_PATH)
    print(f"  {len(nysri):,} segments loaded")

    # ── 6. Reproject to UTM 18N ───────────────────────────────────────────────
    print("Reprojecting to UTM 18N...")
    osm_m   = osm_gdf.to_crs(CRS_METER).reset_index(drop=True)
    nysri_m = nysri.to_crs(CRS_METER).reset_index(drop=True)
    del osm_gdf, nysri

    # ── 7. Vectorized STRtree matching ────────────────────────────────────────
    print("Matching (STRtree bulk query)...")

    osm_geoms   = np.array(osm_m.geometry)
    nysri_geoms = np.array(nysri_m.geometry)
    osm_lens    = shapely.length(osm_geoms)

    print("  Buffering OSM ways...")
    osm_bufs = shapely.buffer(osm_geoms, BUFFER_M)

    print("  Building NYSRI spatial index + querying candidates...")
    tree = STRtree(nysri_geoms)
    osm_cand, nysri_cand = tree.query(osm_bufs, predicate="intersects")
    print(f"  {len(osm_cand):,} candidate pairs")

    print("  Computing overlap ratios...")
    clipped_lens = shapely.length(
        shapely.intersection(nysri_geoms[nysri_cand], osm_bufs[osm_cand])
    )
    overlap = np.minimum(1.0, clipped_lens / np.maximum(osm_lens[osm_cand], 1e-10))
    del osm_bufs, clipped_lens

    print("  Computing bearing differences...")
    bear_diff = np.abs(bulk_bearing(osm_geoms[osm_cand]) - bulk_bearing(nysri_geoms[nysri_cand])) % 180
    bear_diff = np.minimum(bear_diff, 180 - bear_diff)

    print("  Filtering and selecting best match per OSM way...")
    mask = (overlap >= OVERLAP_THRESHOLD) & (bear_diff <= BEARING_THRESHOLD)
    pairs = (
        pd.DataFrame({
            "osm_pos":            osm_cand[mask],
            "nysri_pos":          nysri_cand[mask],
            "match_overlap":      np.round(overlap[mask], 4),
            "match_bearing_diff": np.round(bear_diff[mask], 2),
        })
        .sort_values("match_overlap", ascending=False)
        .groupby("osm_pos", sort=False)
        .first()
    )
    del osm_cand, nysri_cand, overlap, bear_diff, mask

    n_matched = len(pairs)
    print(f"  {n_matched:,} / {len(osm_m):,} OSM ways matched  "
          f"({n_matched / len(osm_m):.1%})")

    # ── 8. Build join table ───────────────────────────────────────────────────
    print("Building output table...")

    # OSM base (non-spatial)
    osm_tag_cols = [f"osm_{t}" for t in OSM_EXPORT_TAGS]
    osm_base = osm_m[["osm_way_id"] + osm_tag_cols].copy()

    # NYSRI attributes for matched positions
    nysri_cols   = [f"nysri_{f.lower()}" for f in NYSRI_JOIN_FIELDS]
    nysri_subset = nysri_m.iloc[pairs["nysri_pos"].values][NYSRI_JOIN_FIELDS].copy()
    nysri_subset.columns = nysri_cols
    nysri_subset = nysri_subset.reset_index(drop=True)
    nysri_subset["osm_way_id"]         = osm_m["osm_way_id"].iloc[pairs.index].values
    nysri_subset["match_overlap"]      = pairs["match_overlap"].values
    nysri_subset["match_bearing_diff"] = pairs["match_bearing_diff"].values

    out_df = osm_base.merge(nysri_subset, on="osm_way_id", how="left")

    # ── 9. Write Parquet ──────────────────────────────────────────────────────
    out_df.to_parquet(OUTPUT, index=False)
    mb = OUTPUT.stat().st_size / 1e6
    print(f"Wrote {OUTPUT}  ({len(out_df):,} rows, {mb:.1f} MB)")

    matched = out_df["nysri_objectid"].notna()
    print(f"\nSummary:")
    print(f"  OSM ways total:               {len(out_df):,}")
    print(f"  Matched to NYSRI segment:     {matched.sum():,}  ({matched.mean():.1%})")
    print(f"  Unmatched (use FC defaults):  {(~matched).sum():,}")
    if matched.any():
        print(f"  Avg match overlap ratio:      {out_df.loc[matched, 'match_overlap'].mean():.3f}")
        print(f"  Avg match bearing diff:       {out_df.loc[matched, 'match_bearing_diff'].mean():.1f}°")


if __name__ == "__main__":
    main()
