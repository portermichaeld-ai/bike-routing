# Handoff Note — Bike Routing Pipeline

## What we're building

`pipeline/01_match_test.py` — a conflation test script that joins NYSDOT road inventory attributes onto OSM ways, producing two GeoJSON files in `data/ny/match-test/`:
- `osm_sample.geojson` — 1000 sampled OSM highway ways with `nysri_*` attributes joined
- `nysri_matched.geojson` — the matched NYSDOT segments, linked back by `osm_way_id`

## Architecture decision

**OSM is the base network** (not NYSDOT). Reason: OSM is contiguous across NY/NJ/CT state lines, which is required for the tri-state routing graph. NYSDOT data is joined onto OSM as enrichment. Where no NYSDOT match exists, the scoring model falls back to functional class defaults (already designed in `doc/04_bikeability_score.docx`).

## Matching logic

For each OSM way:
1. **15m spatial buffer** — find NYSDOT candidate segments whose geometry intersects
2. **70% overlap ratio** — `min(1.0, nysri_clipped_length / osm_way_length) >= 0.70` — ensures NYSDOT segment covers most of the OSM way's length
3. **≤30° bearing difference** — compares first-to-last-node direction of each geometry — filters out intersecting roads that enter the buffer perpendicularly but are not the same road
4. **Best match** — highest overlap ratio wins when multiple candidates pass

Name matching was considered and rejected: too many nulls, format inconsistencies between datasets (e.g. "NY-9W" vs "9W"), and intersecting roads can share name fragments.

## OSM read strategy

Two-pass read (avoids segfault/OOM on the 490MB NYS PBF with `locations=True`):
- **Pass 1** (`locations=False`): collect all highway way tags + node ref ID lists
- Sample 1000 ways, collect the ~10k node IDs they reference
- **Pass 2** (`locations=False`): read node entities, collect lat/lon for only those node IDs
- Build `LineString` geometries from collected node locations

## Where it was left off

The script runs without crashing but produces **0 matches**. The two-pass OSM read and geometry build are working correctly (1000 ways with valid geometries confirmed). Zero matches suggests a CRS or spatial alignment problem — the NYSDOT and OSM layers may not be overlapping after reprojection, or the sindex query is not finding candidates.

**Next debugging steps:**
1. Print bounding boxes of both layers after `to_crs(CRS_METER)` and confirm they overlap
2. Spot-check a single OSM way — print its geometry coords, buffer bounds, and how many NYSDOT candidates the sindex returns before the overlap/bearing filters
3. Verify the NYSDOT GeoJSON CRS is actually EPSG:4326 (not a state plane that got mislabeled)
4. Check if `nysri_sindex.query(buffered)` returns anything at all for a known-good segment

## Key files

| File | Purpose |
|---|---|
| `pipeline/01_match_test.py` | Script described above |
| `data/ny/ny_roadway_inventory.geojson` | NYSDOT road inventory (200,241 segments) |
| `data/ny/new-york-260531.osm.pbf` | NYS OSM extract from Geofabrik (~490MB, Git LFS) |
| `data/ny/match-test/` | Output directory |
| `doc/02_stack_and_architecture.docx` | Stack decisions |
| `doc/04_bikeability_score.docx` | Scoring model — defines which NYSDOT fields matter |
| `config/bikeability_config.yaml` | Scoring parameters (not yet created) |

## Installed packages

`geopandas==1.1.3`, `osmium==4.3.1`, `shapely==2.1.2`, `pyproj==3.7.2`, `pyogrio==0.12.1`
